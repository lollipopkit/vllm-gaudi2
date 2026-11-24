# SPDX-License-Identifier: Apache-2.0
"""Torch-only pieces of DFlash2 drafting on HPU (no vLLM imports, so they test on CPU)."""
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim with the vLLM rounding order: normalize in fp32, cast, scale."""
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x32.to(x.dtype) * weight


def compute_context_kv(
    states: torch.Tensor,
    positions: torch.Tensor,
    *,
    norm_weight: torch.Tensor,
    eps: float,
    kv_weight: torch.Tensor,
    kv_bias: Optional[torch.Tensor],
    k_norm_weight: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    rope: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    v_scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Context K/V of every draft layer from the combined target hidden states.

    states: [N, hidden] (after the fc combine); kv_weight: the K/V rows of all layers' qkv_proj stacked
    ([L * 2 * kv_size, hidden]); k_norm_weight: [L, head_dim]. Returns K (normed, RoPE'd) and V, each
    [L, N, num_kv_heads, head_dim].
    """
    num_layers = k_norm_weight.shape[0]
    num_ctx = states.shape[0]
    kv = F.linear(rms_norm(states, norm_weight, eps), kv_weight, kv_bias)
    kv = kv.view(num_ctx, num_layers, 2, num_kv_heads, head_dim).permute(2, 1, 0, 3, 4)
    k = rms_norm(kv[0], k_norm_weight.view(num_layers, 1, 1, head_dim), eps)
    k = rope(positions.repeat(num_layers), k.reshape(num_layers * num_ctx, num_kv_heads * head_dim))
    v = kv[1]
    if v_scale is not None:
        v = v * v_scale
    return k.reshape(num_layers, num_ctx, num_kv_heads, head_dim), v.contiguous()


def chunked_topk(scores: torch.Tensor, k: int, chunk: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.topk over the last dim via chunk maxima: the k largest values lie in the k chunks with the largest
    maxima, so only k * chunk candidates are sorted (a full-vocab topk is a radix sort on HPU).
    Requires the last dim to be a multiple of chunk. Returns (values, ids) sorted descending."""
    rows, n = scores.shape
    blocks = scores.view(rows, n // chunk, chunk)
    _, top_blocks = blocks.amax(dim=-1).topk(k, dim=-1)  # [rows, k]
    cand = blocks.gather(1, top_blocks.unsqueeze(-1).expand(-1, -1, chunk)).view(rows, k * chunk)
    values, pos = cand.topk(k, dim=-1)
    # int32 gather: Gaudi2 cannot gather int64 data.
    ids = top_blocks.int().gather(1, pos // chunk).long() * chunk + pos % chunk
    return values, ids


def score_edges(predecessor_table: torch.Tensor, successor_table: torch.Tensor, candidate_ids: torch.Tensor,
                unary_logits: torch.Tensor, hidden: torch.Tensor, anchor_token_ids: torch.Tensor) -> torch.Tensor:
    """Upstream _score_edges as a batched matmul: the HPU einsum lowering reshapes [B, S, K, 1, R] into the
    [B, S, K, K] output for B > 1 and fails. candidate_ids: [B, S, K]; hidden: [B, S, R]; returns [B, S, K, K]."""
    top_k = candidate_ids.shape[-1]
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat((anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]), dim=1)
    predecessors = predecessor_table[predecessor_ids] * hidden[:, :, None]
    return unary_logits[:, :, None] + torch.matmul(predecessors, successors.transpose(-1, -2))


def greedy_candidate_walk(scores: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    """Temperature-0 selector walk (upstream _selector_walk_kernel).

    scores: [B, S, K, K], scores[b, s, i, j] = score of candidate j at step s after candidate i at step s - 1
    (all rows of step 0 follow the anchor); candidate_ids: [B, S, K]. Step s takes the first argmax of the
    row selected at step s - 1 (row 0 at step 0). Returns the draft tokens [B, S].
    """
    batch_size, num_steps, top_k, _ = scores.shape
    previous = torch.zeros(batch_size, 1, 1, dtype=torch.long, device=scores.device)
    indices = []
    for step in range(num_steps):
        row = scores[:, step].gather(1, previous.expand(batch_size, 1, top_k)).squeeze(1)
        index = row.argmax(dim=-1, keepdim=True)
        indices.append(index)
        previous = index.view(batch_size, 1, 1)
    # Gaudi2 cannot gather int64 data (graph compile fails): gather the token ids as int32.
    tokens = candidate_ids.int().gather(2, torch.stack(indices, dim=1)).squeeze(-1)
    return tokens.to(candidate_ids.dtype)


@dataclass
class DFlashQueryLayout:
    """One DFlash query block per request as num_query_per_req virtual decode rows."""
    input_ids: torch.Tensor  # [padded_batch_size * T, 1]: bonus token, then mask tokens
    positions: torch.Tensor  # [padded_batch_size * T, 1]
    slot_mapping: torch.Tensor  # [padded_batch_size * T, 1]
    row_block_tables: list[list[int]]  # per real row: the blocks it attends to
    row_last_slots: list[list[int]]  # per real row: slot of its last visible key (sets the block usage)
    row_first_offsets: list[int]  # per real row: first visible offset in its first block (sliding window)


def build_query_layout(
    positions: list[int],
    bonus_token_ids: list[int],
    block_tables: list[list[int]],
    *,
    num_query_per_req: int,
    mask_token_id: int,
    block_size: int,
    padded_batch_size: int,
    pad_slot_id: int,
    sliding_window: Optional[int] = None,
    causal: bool = False,
) -> DFlashQueryLayout:
    """positions[i] is the position p of request i's bonus token; its query rows sit at p .. p + T - 1.

    Non-causal rows see keys up to p + T - 1 (the whole block), causal rows up to their own position; a
    sliding window of W keeps keys at q - W + 1 .. (symmetric FlashAttention window of the query at q).
    block_tables[i] holds request i's KV blocks from position 0; positions past them write to pad slots.
    """
    t = num_query_per_req
    num_rows = padded_batch_size * t
    input_ids = torch.zeros(num_rows, 1, dtype=torch.int32)
    row_positions = torch.zeros(num_rows, 1, dtype=torch.int64)
    slots = (pad_slot_id + torch.arange(num_rows, dtype=torch.int64) % block_size).view(-1, 1)
    row_block_tables: list[list[int]] = []
    row_last_slots: list[list[int]] = []
    row_first_offsets: list[int] = []
    for i, (p, bonus, blocks) in enumerate(zip(positions, bonus_token_ids, block_tables)):
        capacity = len(blocks) * block_size
        for r in range(t):
            row = i * t + r
            q = p + r
            input_ids[row, 0] = bonus if r == 0 else mask_token_id
            row_positions[row, 0] = q
            if q < capacity:
                slots[row, 0] = blocks[q // block_size] * block_size + q % block_size
            hi = min(q if causal else p + t - 1, capacity - 1)
            lo = 0 if sliding_window is None else max(0, q - sliding_window + 1)
            if hi < 0:
                row_block_tables.append([])
                row_last_slots.append([pad_slot_id])
                row_first_offsets.append(0)
                continue
            lo = min(lo, hi)
            row_block_tables.append(blocks[lo // block_size:hi // block_size + 1])
            row_last_slots.append([blocks[hi // block_size] * block_size + hi % block_size])
            row_first_offsets.append(lo % block_size)
    return DFlashQueryLayout(input_ids, row_positions, slots, row_block_tables, row_last_slots, row_first_offsets)


def window_block_bias(row_block_tables: list[list[int]], row_first_offsets: list[int], block_usage: torch.Tensor,
                      block_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Decode block bias [num_blocks, block_size] masking keys past each block's usage and, in each row's first
    block, keys before the row's first visible offset. Rows' blocks are laid out in order (non-contiguous PA)."""
    first = torch.zeros(block_usage.shape[0], dtype=torch.int32)
    index = 0
    for blocks, offset in zip(row_block_tables, row_first_offsets):
        if blocks:
            first[index] = offset
            index += len(blocks)
    offsets = torch.arange(block_size, dtype=torch.int32).view(1, -1)
    mask = (offsets < first.view(-1, 1)) | (offsets >= block_usage.view(-1, 1).to(torch.int32))
    return torch.zeros(mask.shape, dtype=dtype).masked_fill_(mask, -math.inf)


def block_mapping_from_groups(block_groups: torch.Tensor, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
    """One-hot [num_blocks, batch_size] block -> row mapping; padding blocks (group -1) map nowhere."""
    valid = block_groups >= 0
    mapping = F.one_hot(block_groups.clamp(min=0).long(), num_classes=batch_size)
    return (mapping * valid.unsqueeze(-1)).to(dtype)
