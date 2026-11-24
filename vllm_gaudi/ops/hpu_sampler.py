# SPDX-License-Identifier: Apache-2.0
"""Top-k/top-p sampling without sorting the vocabulary.

The native vLLM path (TopKTopPSampler.forward_native -> apply_top_k_top_p_pytorch) sorts the full
vocabulary for every row whenever top-p is set. Qwen-style generation configs default every request to
top_k=20, top_p=0.95, so each decode step bitonic-sorts [batch, 248k] logits on the TPCs: ~5.6 ms of the
step at batch 32. When every row's k is at most TOPK_CAP (known on the host from the input batch, so no
device sync), top-k is a fixed-size torch.topk and top-p only needs the k candidates, which are already
sorted. The candidate distribution is identical to the native path's (top-k mask, then top-p over the
renormalized survivors); only the random noise layout differs.
"""

from __future__ import annotations

import torch

from vllm_gaudi.v1._vendored.sample.ops import topk_topp_sampler as _native

from vllm_gaudi import envs

TOPK_CAP = envs.VLLM_HPU_SAMPLER_TOPK_CAP  # static topk width: one compiled sampler graph for every k <= cap
# topk over 248k columns is a TPC bitonic sort (bs 32: 2.1 ms); top-cap of 256 groups, then of the 256 * cap
# survivors, is exact and takes 1.2 ms.
TOPK_GROUPS = 256


def topk_cap(input_batch) -> int | None:
    """TOPK_CAP when every request in the batch has 0 < top_k <= TOPK_CAP, else None (native path)."""
    n = input_batch.num_reqs
    if TOPK_CAP <= 0 or n == 0 or len(input_batch.top_k_reqs) != n:
        return None
    return TOPK_CAP if int(input_batch.top_k_cpu[:n].max()) <= TOPK_CAP else None


def _topk(logits: torch.Tensor, cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact descending top-cap values and int32 vocab indices."""
    rows, vocab = logits.shape
    width = vocab // TOPK_GROUPS
    if vocab % TOPK_GROUPS or width < cap:
        vals, idx = logits.topk(cap, dim=-1)
        return vals, idx.int()
    vals, idx = logits.view(rows, TOPK_GROUPS, width).topk(cap, dim=-1)
    idx = idx.int() + torch.arange(0, vocab, width, dtype=torch.int32, device=logits.device).view(1, -1, 1)
    vals, pick = vals.view(rows, -1).topk(cap, dim=-1)
    # int64 gather fails to compile on HPU (synStatus 26): gather int32 data.
    return vals, idx.view(rows, -1).gather(1, pick)


def capped_candidates(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None,
                      cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-cap logits (descending) with the top-k / top-p rejects set to -inf, and their int32 vocab indices."""
    vals, idx = _topk(logits, cap)
    pos = torch.arange(cap, device=logits.device)
    vals = vals.masked_fill(pos >= k.unsqueeze(1), -float("inf"))
    if p is not None:
        probs = vals.softmax(dim=-1, dtype=torch.float32)
        # Native rule in descending order: drop a token once the mass before it reaches p (the top token
        # always survives).
        vals = vals.masked_fill(probs.cumsum(dim=-1) - probs >= p.unsqueeze(1), -float("inf"))
    return vals, idx


def sample_capped(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None, cap: int,
                  generators: dict[int, torch.Generator], use_fp64_gumbel: bool = False) -> torch.Tensor:
    vals, idx = capped_candidates(logits, k, p, cap)
    choice = _native.random_sample(vals.softmax(dim=-1, dtype=torch.float32), generators, use_fp64_gumbel)
    return idx.gather(1, choice.unsqueeze(1)).squeeze(1).long()


_forward_native = _native.TopKTopPSampler.forward_native


def _forward_native_capped(self, logits, generators, k, p):
    cap = getattr(self, "hpu_topk_cap", None)
    if cap is None or k is None or self.logprobs_mode in _native.PROCESSED_LOGPROBS_MODES:
        return _forward_native(self, logits, generators, k, p)
    return sample_capped(logits, k, p, cap, generators, self.use_fp64_gumbel), None


# TopKTopPSampler.__init__ binds self.forward = self.forward_native: import this module before any Sampler
# is constructed (hpu_model_runner does at import).
_native.TopKTopPSampler.forward_native = _forward_native_capped
