# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed MoE layer on Gaudi2 HPU (FP8 E4M3 per-channel + Softcapped SiTU).

Uses static-shape, zero-host-sync 3D batched `torch.ops.hpu.fp8_gemm_v2` so the
entire MoE block is HPU-Graph and `torch.compile` compatible.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from .ops import GAUDI2_FP8_MAX, fp8_bmm, fp8_linear, quantize_fp8_hpu, softcapped_situ
from .tpc_ops import has_tpc_gate, has_tpc_gate_splitk, has_tpc_moe, has_tpc_moe_grouped

# Crossover measured on Gaudi2 HPU (128 experts, top-8):
# - With custom TPC MoE (zero-copy indirect FP8 GEMV + fused Softcapped SiTU):
#   M <= 16 uses TPC kernels (3.1x faster at M=1, 2.05x faster at M=8),
#   M > 16 uses zero-copy 4D fp8_gemm_v2 directly over the packed weights.
# - Without custom TPC MoE (fallback):
#   M < 8 gathers active experts via index_select + 3D fp8_bmm;
#   M >= 8 reads all 128 experts once via 2D w13 GEMM + scatter 3D w2 BMM.
_MOE_TPC_MAX_TOKENS = 16
_MOE_DENSE_MIN_TOKENS = 8
_MOE_DENSE_CHUNK = max(128, int(os.environ.get("INKLING2_MOE_DENSE_CHUNK", "512")))


class Inkling2HPUGate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.use_gate_bias
        num_experts, hidden = config.n_routed_experts, config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = float(config.route_scale)
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        for param in (self.weight, self.bias):
            set_weight_attrs(param, {"weight_loader": default_weight_loader})
        self._weight_f32_ready = False
        self._weight_packed_ready = False

    def process_weights_after_loading(self) -> None:
        """Cache fp32 gate weight (prefill) and packed bf16 weight (fused TPC decode)."""
        if self._weight_f32_ready or self.weight.numel() == 0:
            return
        num_e, h = self.weight.shape
        if num_e == 128 and (h % 128) == 0:
            # Pack [128, H] -> [H, 128] with row h = [e0, e64, e1, e65, ..., e63, e127]
            # for 1-load v_convert_bf16_to_f32_all_b inside inkling2_moe_gate_gemv_splitk.
            w_pack = (
                self.weight.data.view(2, 64, h)
                .permute(2, 1, 0)
                .contiguous()
                .view(h, 128)
            )
            self.register_buffer("weight_packed_bf16", w_pack, persistent=False)
            self._weight_packed_ready = True
        self.register_buffer("weight_f32", self.weight.data.float().contiguous(), persistent=False)
        self.weight.data = self.weight.data.new_empty((0,))
        self._weight_f32_ready = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(topk_ids [T, topk], topk_weights [T, topk] fp32)`.

        Matches upstream `ops/moe.py`:
          - selection score = sigmoid(logits) + bias
          - routing weight = softmax(log_sigmoid(selected_logits)) * route_scale
        """
        w_f32 = self.weight_f32 if getattr(self, "_weight_f32_ready", False) else self.weight.float()
        logits = F.linear(x.float(), w_f32)
        num_e = logits.shape[-1]
        if (
            logits.device.type == "hpu"
            and logits.ndim == 2
            and (num_e % 64) == 0
            and num_e <= 256
            and 1 <= self.topk <= 16
            and has_tpc_gate()
        ):
            return torch.ops.custom_op.inkling2_moe_gate_topk(
                logits.unsqueeze(1).contiguous(), self.bias.contiguous(), self.topk, self.route_scale
            )
        selection = torch.sigmoid(logits) + self.bias.unsqueeze(0)
        _, topk_ids = torch.topk(selection, self.topk, dim=-1, largest=True, sorted=True)
        selected_logits = logits.gather(-1, topk_ids)
        log_probs = F.logsigmoid(selected_logits)
        weights = torch.softmax(log_probs, dim=-1) * self.route_scale
        return topk_ids, weights


class Inkling2HPUFP8Experts(nn.Module):
    """Static-shape, zero-sync per-channel FP8 E4M3 routed experts with Softcapped SiTU."""

    def __init__(self, config, fp8_max: float = GAUDI2_FP8_MAX) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.expert_start = self.ep_rank * self.num_experts
        self.expert_end = self.expert_start + self.num_experts
        self.topk = config.num_experts_per_tok
        self.gate_cap = float(config.mlp_gate_softcap)
        self.up_cap = float(config.mlp_up_softcap)
        self.fp8_max = float(fp8_max)
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter

        self.w13_weight = nn.Parameter(
            torch.empty(self.num_experts, 2 * inter, hidden, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.empty(self.num_experts, 2 * inter, dtype=torch.float32),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(self.num_experts, hidden, inter, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.empty(self.num_experts, hidden, dtype=torch.float32),
            requires_grad=False,
        )
        for p in (
            self.w13_weight,
            self.w13_weight_scale,
            self.w2_weight,
            self.w2_weight_scale,
        ):
            set_weight_attrs(p, {"weight_loader": self._load_expert_shard})
        self._use_tpc_packed = False

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        shard = loaded[self.expert_start : self.expert_end]
        assert shard.shape == param.shape and shard.dtype == param.dtype, (
            f"{shard.shape} {shard.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(shard)

    def process_weights_after_loading(self) -> None:
        """Repack FP8 MoE weights in-place for zero-copy Gaudi2 TPC MoE kernels."""
        if self._use_tpc_packed or self.w13_weight.numel() == 0:
            return
        if self.w13_weight.device.type != "hpu" or not has_tpc_moe():
            return
        h, inter, e = self.hidden_size, self.intermediate_size, self.num_experts
        if (
            (h % 256) != 0
            or (inter % 128) != 0
            or (h // 4) > 1024
            or (inter // 4) > 512
            or (inter // 128) > 32
            or self.gate_cap <= 0.0
            or self.up_cap <= 0.0
            or self.fp8_max <= 0.0
        ):
            return
        n_w13_blks = inter // 128
        n_w2_blks = h // 256

        # Pack w13 [E, 2*I, H] -> [E, n_w13_blks, H, 256] (64 quads of g_e, g_o, u_e, u_o)
        # and s13 [E, 2*I] -> [E, n_w13_blks, 4, 64] flattened to [E, 2*I].
        w13_p = (
            self.w13_weight.data.view(e, n_w13_blks, 64, 2, 2, h)
            .permute(0, 1, 5, 2, 4, 3)
            .contiguous()
            .view(e, n_w13_blks, h, 256)
        )
        s13_p = (
            self.w13_weight_scale.data.view(e, n_w13_blks, 64, 2, 2)
            .permute(0, 1, 4, 3, 2)
            .contiguous()
            .view(e, 2 * inter)
        )
        self.register_buffer("w13_packed", w13_p, persistent=False)
        self.register_buffer("s13_packed", s13_p, persistent=False)
        self.w13_weight.data = self.w13_weight.data.new_empty((0,))
        self.w13_weight_scale.data = self.w13_weight_scale.data.new_empty((0,))

        # Pack w2 [E, H, I] -> [E, n_w2_blks, I_compact, 256] where each 128-group of I is
        # reordered [even0..63, odd0..63] to match compact 128B act_f8 from w13_situ_f8.
        w2_p = (
            self.w2_weight.data.view(e, n_w2_blks, 2, 64, 2, n_w13_blks, 64, 2)
            .permute(0, 1, 5, 7, 6, 3, 2, 4)
            .contiguous()
            .view(e, n_w2_blks, inter, 256)
        )
        s2_p = (
            self.w2_weight_scale.data.view(e, n_w2_blks, 2, 64, 2)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
            .view(e, h)
        )
        self.register_buffer("w2_packed", w2_p, persistent=False)
        self.register_buffer("s2_packed", s2_p, persistent=False)
        self.w2_weight.data = self.w2_weight.data.new_empty((0,))
        self.w2_weight_scale.data = self.w2_weight_scale.data.new_empty((0,))
        self._use_tpc_packed = True

    def _run_tpc_chunk(
        self,
        x_chunk: torch.Tensor,
        ids_chunk: torch.Tensor,
        weights_chunk: torch.Tensor,
        *,
        return_partial: bool = False,
        x_q: torch.Tensor | None = None,
        x_sc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Zero-copy indirect FP8 MoE via custom Gaudi2 TPC kernels (`M <= 16`)."""
        m, hidden = x_chunk.shape
        inter = self.intermediate_size
        n_groups = inter // 128

        if self.ep_size > 1:
            local_mask = (ids_chunk >= self.expert_start) & (ids_chunk < self.expert_end)
            local_ids = (
                (ids_chunk - self.expert_start).clamp(0, self.num_experts - 1).to(torch.int32).contiguous()
            )
            local_w = (weights_chunk * local_mask.float()).float().contiguous()
        else:
            local_ids = ids_chunk.to(torch.int32).contiguous()
            local_w = weights_chunk.float().contiguous()

        if x_q is None or x_sc is None:
            x_q, x_sc = quantize_fp8_hpu(
                x_chunk, fp8_max=self.fp8_max, per_row=True, use_cguid=True
            )
        x_q_i32 = x_q.contiguous().view(torch.int32)
        x_sc_1d = x_sc.reshape(-1).contiguous()
        if (
            m >= 2
            and m * self.topk <= 128
            and (hidden // 4) <= 640
            and (inter // 4) <= 384
            and self.num_experts <= 256
            and has_tpc_moe_grouped()
        ):
            group_meta, group_count = torch.ops.custom_op.inkling2_moe_group_experts(
                local_ids, self.num_experts, n_groups, hidden // 256
            )
            act_f8, act_sc = torch.ops.custom_op.inkling2_moe_w13_grouped_f8(
                x_q_i32,
                x_sc_1d,
                self.w13_packed,
                self.s13_packed,
                local_w,
                group_meta,
                group_count,
                self.gate_cap,
                self.up_cap,
                self.fp8_max,
            )
            partial = torch.ops.custom_op.inkling2_moe_w2_grouped_f8(
                act_f8.view(m, self.topk, inter).view(torch.int32),
                act_sc.view(m, self.topk, n_groups),
                self.w2_packed,
                self.s2_packed,
                group_meta,
                group_count,
            )
        else:
            act_f8, act_sc = torch.ops.custom_op.inkling2_moe_w13_situ_f8(
                x_q_i32,
                x_sc_1d,
                self.w13_packed,
                self.s13_packed,
                local_ids,
                local_w,
                self.gate_cap,
                self.up_cap,
                self.fp8_max,
            )
            partial = torch.ops.custom_op.inkling2_moe_w2_f8(
                act_f8.view(m, self.topk, inter).view(torch.int32),
                act_sc.view(m, self.topk, n_groups),
                self.w2_packed,
                self.s2_packed,
                local_ids,
            )
        partial_3d = partial.view(m, self.topk, hidden)
        if return_partial:
            return partial_3d
        return partial_3d.float().sum(dim=1).to(x_chunk.dtype)

    def _run_gather_chunk(
        self,
        x_chunk: torch.Tensor,
        ids_chunk: torch.Tensor,
        weights_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-sync gather + 3D batched FP8 MoE for small decode batches (`M < 8`)."""
        m, hidden = x_chunk.shape
        flat_ids = ids_chunk.reshape(-1).long()  # [M * topk]
        flat_w = weights_chunk.reshape(-1, 1)  # [M * topk, 1]

        if self.ep_size > 1:
            local_mask = (flat_ids >= self.expert_start) & (flat_ids < self.expert_end)
            local_ids = (flat_ids - self.expert_start).clamp(0, self.num_experts - 1)
            flat_w = flat_w * local_mask.unsqueeze(-1).float()
        else:
            local_ids = flat_ids

        # Quantize x per-token once before repeating top-k times (8x less quant work).
        if x_chunk.device.type == "hpu":
            x_q, x_sc = quantize_fp8_hpu(
                x_chunk, fp8_max=self.fp8_max, per_row=True, use_cguid=True
            )
            x_rep = x_q.repeat_interleave(self.topk, dim=0).unsqueeze(1)  # [R, 1, H]
            x_scale = x_sc.repeat_interleave(self.topk, dim=0).unsqueeze(1)  # [R, 1, 1]
        else:
            x_scale = None
            x_rep = x_chunk.repeat_interleave(self.topk, dim=0).unsqueeze(1)

        # Gather active experts and run one 3D batched FP8 GEMM for w13 and one for w2.
        w13 = self.w13_weight.index_select(0, local_ids)  # [R, 2*I, H]
        s13 = self.w13_weight_scale.index_select(0, local_ids)  # [R, 2*I]

        gate_up = fp8_bmm(x_rep, w13, s13, fp8_max=self.fp8_max, x_scale=x_scale).squeeze(1)  # [R, 2*I]
        act = softcapped_situ(gate_up, flat_w, self.gate_cap, self.up_cap).unsqueeze(1)  # [R, 1, I]

        w2 = self.w2_weight.index_select(0, local_ids)  # [R, H, I]
        s2 = self.w2_weight_scale.index_select(0, local_ids)  # [R, H]
        down = fp8_bmm(act, w2, s2, fp8_max=self.fp8_max, use_cguid=False).squeeze(1)  # [R, H]

        return down.view(m, self.topk, hidden).float().sum(dim=1).to(x_chunk.dtype)

    def _run_dense_chunk(
        self,
        x_chunk: torch.Tensor,
        ids_chunk: torch.Tensor,
        weights_chunk: torch.Tensor,
        *,
        x_q: torch.Tensor | None = None,
        x_sc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Zero-copy expert-once FP8 MoE for `M >= 8` (c8+ decode and prefill).

        Reads all `E` experts exactly once with no `index_select` weight copy:
          1. 2D FP8 GEMM `x @ w13_flat.T` -> gather top-k -> Softcapped SiTU.
          2. Scatter activations to `[E, M, I]` -> 3D batched FP8 GEMM with `w2` -> sum over `E`.
        """
        m, hidden = x_chunk.shape
        inter = self.intermediate_size
        e = self.num_experts

        if self.ep_size > 1:
            local_mask = (ids_chunk >= self.expert_start) & (ids_chunk < self.expert_end)
            local_ids = (ids_chunk - self.expert_start).clamp(0, e - 1).long()
            local_w = weights_chunk * local_mask.float()
        else:
            local_ids = ids_chunk.long()
            local_w = weights_chunk

        if self._use_tpc_packed:
            n_w13_blks = inter // 128
            n_w2_blks = hidden // 256
            if x_q is None or x_sc is None:
                x_q, x_sc = quantize_fp8_hpu(
                    x_chunk, fp8_max=self.fp8_max, per_row=True, use_cguid=True
                )
            s13_g = (
                self.s13_packed.view(e, n_w13_blks, 4, 64)
                .transpose(-2, -1)
                .contiguous()
                .view(e, n_w13_blks, 1, 256)
            )
            gu4 = torch.ops.hpu.fp8_gemm_v2(
                A=x_q.view(1, 1, m, hidden),
                trans_A=False,
                B=self.w13_packed,
                trans_B=False,
                D=None,
                out_dtype=torch.bfloat16,
                A_scale_inv=x_sc.view(1, 1, m, 1),
                B_scale_inv=s13_g,
                bias=None,
                accumulate=False,
            )  # [E, n_w13_blks, M, 256]
            gu_me = gu4.permute(2, 0, 1, 3).reshape(m, e, 2 * inter)
            gu_topk = gu_me.gather(1, local_ids.unsqueeze(-1).expand(-1, -1, 2 * inter)).view(
                m, self.topk, n_w13_blks, 64, 2, 2
            )
            gate = gu_topk[..., 0, :].transpose(-1, -2).reshape(m, self.topk, inter).float()
            up = gu_topk[..., 1, :].transpose(-1, -2).reshape(m, self.topk, inter).float()
            sig = torch.sigmoid(gate)
            if self.gate_cap > 0.0:
                gate = self.gate_cap * torch.tanh(gate / self.gate_cap)
            if self.up_cap > 0.0:
                up = self.up_cap * torch.tanh(up / self.up_cap)
            act = (((gate * sig) * up) * local_w.unsqueeze(-1).float()).to(torch.bfloat16)

            act_e = torch.zeros((m, e, inter), dtype=act.dtype, device=act.device)
            if self.ep_size == 1:
                act_e.scatter_(1, local_ids.unsqueeze(-1).expand(-1, -1, inter), act)
            else:
                act_e.scatter_add_(1, local_ids.unsqueeze(-1).expand(-1, -1, inter), act)

            a3 = act_e.transpose(0, 1).contiguous()
            a_fp8_2d, a_sc_2d = quantize_fp8_hpu(
                a3.reshape(e * m, inter), fp8_max=self.fp8_max, per_row=True, use_cguid=False
            )
            s2_g = (
                self.s2_packed.view(e, n_w2_blks, 4, 64)
                .transpose(-2, -1)
                .contiguous()
                .view(e, n_w2_blks, 1, 256)
            )
            down4 = torch.ops.hpu.fp8_gemm_v2(
                A=a_fp8_2d.view(e, 1, m, inter),
                trans_A=False,
                B=self.w2_packed,
                trans_B=False,
                D=None,
                out_dtype=torch.bfloat16,
                A_scale_inv=a_sc_2d.view(e, 1, m, 1),
                B_scale_inv=s2_g,
                bias=None,
                accumulate=False,
            )  # [E, n_w2_blks, M, 256]
            summed = (
                down4.sum(dim=0, dtype=torch.float32)
                .view(n_w2_blks, m, 64, 2, 2)
                .permute(1, 0, 3, 2, 4)
                .reshape(m, hidden)
            )
            return summed.to(x_chunk.dtype)

        gu_all = fp8_linear(
            x_chunk,
            self.w13_weight.view(e * 2 * inter, hidden),
            self.w13_weight_scale.view(e * 2 * inter),
            fp8_max=self.fp8_max,
            use_cguid=True,
        ).view(m, e, 2 * inter)
        gu_topk = gu_all.gather(1, local_ids.unsqueeze(-1).expand(-1, -1, 2 * inter))
        act = softcapped_situ(gu_topk, local_w.unsqueeze(-1), self.gate_cap, self.up_cap)

        act_e = torch.zeros((m, e, inter), dtype=act.dtype, device=act.device)
        if self.ep_size == 1:
            act_e.scatter_(1, local_ids.unsqueeze(-1).expand(-1, -1, inter), act)
        else:
            act_e.scatter_add_(1, local_ids.unsqueeze(-1).expand(-1, -1, inter), act)

        down_e = fp8_bmm(
            act_e.transpose(0, 1).contiguous(),
            self.w2_weight,
            self.w2_weight_scale,
            fp8_max=self.fp8_max,
            use_cguid=False,
        )  # [E, M, H]
        return down_e.float().sum(dim=0).to(x_chunk.dtype)

    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        return_partial: bool = False,
        x_q: torch.Tensor | None = None,
        x_sc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """bf16 `[T, H]` -> bf16 `[T, H]` (or `[T, topK, H]` when `return_partial=True` on TPC TP=1)."""
        num_tokens = x.shape[0]
        if return_partial:
            if not (self._use_tpc_packed and self.ep_size == 1 and num_tokens <= _MOE_TPC_MAX_TOKENS):
                raise ValueError(
                    "return_partial=True requires the TP=1 TPC MoE decode path "
                    f"(num_tokens={num_tokens}, ep_size={self.ep_size}, packed={self._use_tpc_packed})"
                )
            return self._run_tpc_chunk(
                x, topk_ids, topk_weights, return_partial=True, x_q=x_q, x_sc=x_sc
            )
        if self._use_tpc_packed and num_tokens <= _MOE_TPC_MAX_TOKENS:
            res = self._run_tpc_chunk(x, topk_ids, topk_weights, x_q=x_q, x_sc=x_sc)
        elif not self._use_tpc_packed and num_tokens < _MOE_DENSE_MIN_TOKENS:
            res = self._run_gather_chunk(x, topk_ids, topk_weights)
        elif num_tokens <= _MOE_DENSE_CHUNK:
            res = self._run_dense_chunk(x, topk_ids, topk_weights, x_q=x_q, x_sc=x_sc)
        else:
            chunks = [
                self._run_dense_chunk(
                    x[start : start + _MOE_DENSE_CHUNK],
                    topk_ids[start : start + _MOE_DENSE_CHUNK],
                    topk_weights[start : start + _MOE_DENSE_CHUNK],
                    x_q=None if x_q is None else x_q[start : start + _MOE_DENSE_CHUNK],
                    x_sc=None if x_sc is None else x_sc[start : start + _MOE_DENSE_CHUNK],
                )
                for start in range(0, num_tokens, _MOE_DENSE_CHUNK)
            ]
            res = torch.cat(chunks, dim=0)
        return res if self.ep_size == 1 else tensor_model_parallel_all_reduce(res)


class Inkling2HPUMoE(nn.Module):
    def __init__(self, config, fp8_max: float = GAUDI2_FP8_MAX) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2HPUGate(config)
        self.experts = Inkling2HPUFP8Experts(config, fp8_max=fp8_max)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_partial: bool = False,
        x_q: torch.Tensor | None = None,
        x_sc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            x.device.type == "hpu"
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and 1 <= x.shape[0] <= _MOE_TPC_MAX_TOKENS
            and (x.shape[1] % 128) == 0
            and 1 <= self.gate.topk <= 16
            and getattr(self.gate, "_weight_packed_ready", False)
            and has_tpc_gate()
            and has_tpc_gate_splitk()
        ):
            partial_logits = torch.ops.custom_op.inkling2_moe_gate_gemv_splitk(
                x.contiguous(), self.gate.weight_packed_bf16
            )
            topk_ids, topk_weights = torch.ops.custom_op.inkling2_moe_gate_topk(
                partial_logits,
                self.gate.bias.contiguous(),
                self.gate.topk,
                self.gate.route_scale,
            )
            return self.experts(
                x, topk_ids, topk_weights, return_partial=return_partial, x_q=x_q, x_sc=x_sc
            )
        topk_ids, topk_weights = self.gate(x)
        return self.experts(
            x, topk_ids, topk_weights, return_partial=return_partial, x_q=x_q, x_sc=x_sc
        )
