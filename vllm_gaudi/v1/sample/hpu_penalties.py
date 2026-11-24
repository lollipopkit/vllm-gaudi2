# SPDX-License-Identifier: Apache-2.0
"""Presence / frequency / repetition penalties from per-request token statistics kept on the device.

vLLM's v1 path (vllm/v1/sample/ops/penalties.py) rebuilds a [batch, output_len] tensor from Python lists and bins
the prompt and output tokens over the vocabulary at every step: host work and device work grow with the output
length, and every new length is a new shape. Here each request holds a prompt-token mask (set once when the
request is added) and output-token counts (incremented by the tokens appended since the last step), as in vLLM's
model runner v2 (vllm/v1/worker/gpu/sample/penalties.py). The counts follow the same output_token_ids lists the
v1 path reads, so async placeholders, discarded samples and preemption keep their meaning.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_gaudi.utils import async_h2d_copy

# Index tensors are padded to powers of two (at least this) so the scatter shapes stay few.
_MIN_PAIRS = 64


def _bucket(n: int) -> int:
    return max(_MIN_PAIRS, 1 << (n - 1).bit_length())


@dataclass
class HpuSamplingMetadata(SamplingMetadata):
    # [num_rows, vocab] statistics in sampling order (HpuPenaltyStats.gather); None selects vLLM's v1 path
    penalty_prompt_mask: Optional[torch.Tensor] = None
    penalty_output_counts: Optional[torch.Tensor] = None


class HpuPenaltyStats:
    """Token statistics per input-batch row.

    The device buffers are indexed by a physical slot; `slot[row]` maps an input-batch row to it, so swapping and
    condensing rows only permute host arrays. Slot `max_num_reqs` and column `vocab_size` are scratch targets for
    the padding of the index tensors. The buffers are allocated with the input batch, before KV-cache sizing, so
    the memory profile includes them: (max_num_reqs + 1) x (vocab + 1) x 5 bytes.
    """

    def __init__(self, max_num_reqs: int, vocab_size: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size
        self.device = device
        shape = (max_num_reqs + 1, vocab_size + 1)
        # int8, not bool: HPU has no bool index_fill_ / index_put_
        self.prompt_mask = torch.zeros(shape, dtype=torch.int8, device=device)
        self.output_counts = torch.zeros(shape, dtype=torch.int32, device=device)
        self.slot = np.arange(max_num_reqs, dtype=np.int64)
        self.num_seen = np.zeros(max_num_reqs, dtype=np.int64)  # output_token_ids entries already looked at
        self.pending: list[list[int]] = [[] for _ in range(max_num_reqs)]  # positions holding a -1 placeholder
        self.stale = np.ones(max_num_reqs, dtype=bool)  # row must be rebuilt (new or resumed request)
        self._gather_key: Optional[tuple] = None
        self._gather_index: Optional[torch.Tensor] = None
        self._ones_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        self._zero_i8 = torch.zeros((), dtype=torch.int8, device=device)
        self._zero_i32 = torch.zeros((), dtype=torch.int32, device=device)

    def reset(self, row: int) -> None:
        self.stale[row] = True
        self.num_seen[row] = 0
        self.pending[row] = []

    def swap(self, i1: int, i2: int) -> None:
        for a in (self.slot, self.num_seen, self.stale):
            a[[i1, i2]] = a[[i2, i1]]
        self.pending[i1], self.pending[i2] = self.pending[i2], self.pending[i1]

    def move(self, src: int, dst: int) -> None:
        # dst takes src's slot; src takes dst's old slot and is reset when a request is added there
        self.swap(src, dst)

    def _h2d(self, values: np.ndarray) -> torch.Tensor:
        # non-blocking like the rest of the runner (utils.async_h2d_copy): HPU has no pinned memory, and a blocking
        # copy waits for the queued device work, a host stall of ~0.4 ms per decode step (docs/lfm3.md)
        return async_h2d_copy(torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64)), device=self.device)

    def _scatter(self, target: torch.Tensor, flat: np.ndarray, accumulate: bool) -> None:
        n = _bucket(len(flat))
        idx = np.full(n, self.max_num_reqs * (self.vocab_size + 1) + self.vocab_size, dtype=np.int64)
        idx[:len(flat)] = flat
        idx_t = self._h2d(idx)
        key = (n, target.dtype)
        ones = self._ones_cache.get(key)
        if ones is None:
            ones = torch.ones(n, dtype=target.dtype, device=self.device)
            self._ones_cache[key] = ones
        if accumulate:
            # index_add_ on the flat view: 0.1 ms on HPU against 2.3 ms for index_put_(accumulate=True)
            target.view(-1).index_add_(0, idx_t, ones)
        else:
            target.view(-1).index_put_((idx_t, ), ones)

    def sync(self, rows: list[int], output_token_ids: list[list[int]], token_ids_cpu: np.ndarray,
             num_prompt_tokens: np.ndarray) -> None:
        """Bring the given rows up to date with their output_token_ids (duplicates allowed). Every entry that is
        not a -1 placeholder is counted, as in vLLM's v1 path; a placeholder is counted once it is repaired."""
        width = self.vocab_size + 1
        rebuild: list[int] = []
        prompt_flat: list[np.ndarray] = []
        out_flat: list[int] = []
        seen = set()
        for row, out in zip(rows, output_token_ids):
            if row in seen:
                continue
            seen.add(row)
            base = int(self.slot[row]) * width
            if self.stale[row] or len(out) < self.num_seen[row]:
                rebuild.append(int(self.slot[row]))
                self.num_seen[row] = 0
                self.pending[row] = []
                self.stale[row] = False
                prompt_flat.append(base + token_ids_cpu[row, :num_prompt_tokens[row]].astype(np.int64))
            still_pending = []
            for pos in self.pending[row] + list(range(int(self.num_seen[row]), len(out))):
                if out[pos] == -1:
                    still_pending.append(pos)
                else:
                    out_flat.append(base + out[pos])
            self.pending[row] = still_pending
            self.num_seen[row] = len(out)
        if rebuild:
            idx = self._h2d(np.array(rebuild))
            # tensor fill values: the int_Tensor overload is the one HPU implements for every dtype
            self.prompt_mask.index_fill_(0, idx, self._zero_i8)
            self.output_counts.index_fill_(0, idx, self._zero_i32)
            if prompt_flat:
                self._scatter(self.prompt_mask, np.concatenate(prompt_flat), accumulate=False)
        if out_flat:
            self._scatter(self.output_counts, np.array(out_flat, dtype=np.int64), accumulate=True)

    def gather(self, rows: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """[len(rows), vocab] statistics in the given row order, always laid out the same way (index_select)."""
        key = tuple(self.slot[rows].tolist())
        if key != self._gather_key:
            self._gather_key, self._gather_index = key, self._h2d(np.array(key))
        return (self.prompt_mask.index_select(0, self._gather_index)[:, :self.vocab_size],
                self.output_counts.index_select(0, self._gather_index)[:, :self.vocab_size])


def apply_penalties_from_stats(logits: torch.Tensor, prompt_mask: torch.Tensor, output_counts: torch.Tensor,
                               presence_penalties: torch.Tensor, frequency_penalties: torch.Tensor,
                               repetition_penalties: torch.Tensor) -> torch.Tensor:
    """The arithmetic of vllm.model_executor.layers.utils.apply_penalties on precomputed statistics."""
    output_mask = output_counts > 0
    penalties = torch.where((prompt_mask > 0) | output_mask, repetition_penalties.unsqueeze(1), 1.0)
    logits = logits * torch.where(logits > 0, 1.0 / penalties, penalties)
    logits = logits - frequency_penalties.unsqueeze(1) * output_counts
    return logits - presence_penalties.unsqueeze(1) * output_mask


class HpuSampler(Sampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._penalty_fn: Callable[..., torch.Tensor] = apply_penalties_from_stats
        self._penalty_fn_compiled = False

    def compile_penalties(self, compile_fn: Callable) -> None:
        """One fused kernel: 0.26 ms against 1.6 ms eager on HPU for 8 x 128K. The logits-processor stage that
        calls it stays eager (it reads Python lists)."""
        self._penalty_fn = compile_fn(apply_penalties_from_stats)
        self._penalty_fn_compiled = True

    def warmup_penalties(self, stats: HpuPenaltyStats, num_rows: int) -> None:
        """Compile for one batch size with the tensor layout of a real step (HpuPenaltyStats.gather)."""
        if not self._penalty_fn_compiled:
            return
        mask, counts = stats.gather([0] * num_rows)
        logits = torch.zeros(num_rows, stats.vocab_size, dtype=torch.float32, device=stats.device)
        p = torch.zeros(num_rows, dtype=torch.float32, device=stats.device)
        self._penalty_fn(logits, mask, counts, p, p, p + 1)

    def apply_penalties(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata,
                        output_token_ids: list[list[int]]) -> torch.Tensor:
        prompt_mask = getattr(sampling_metadata, "penalty_prompt_mask", None)
        if sampling_metadata.no_penalties or prompt_mask is None:
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)
        # Spec decode combines the draft tokens into output_token_ids; its input batch (is_spec_decode=True)
        # keeps vLLM's path, so a combined list here means the statistics would miss the drafts.
        if output_token_ids is not sampling_metadata.output_token_ids:
            raise RuntimeError("penalty statistics do not cover speculative tokens: build the input batch with "
                               "is_spec_decode=True for speculative decoding")
        return self._penalty_fn(logits, prompt_mask, sampling_metadata.penalty_output_counts,
                                sampling_metadata.presence_penalties, sampling_metadata.frequency_penalties,
                                sampling_metadata.repetition_penalties)
