# SPDX-License-Identifier: Apache-2.0
"""HpuPenaltyStats + apply_penalties_from_stats against vLLM's v1 penalty path, over a simulated request (200 steps)
lifecycle (appends, async placeholders, swaps, condense moves, preemption resets, padded duplicate rows)."""
import os
import random

import numpy as np
import torch

from vllm.model_executor.layers.utils import apply_penalties
from vllm_gaudi.v1.sample.hpu_penalties import HpuPenaltyStats, apply_penalties_from_stats

VOCAB, ROWS, MAX_LEN = 97, 6, 64
# hpu on Gaudi (the default there), cpu elsewhere
DEVICE = torch.device(os.environ.get("PENALTY_TEST_DEVICE", "hpu" if hasattr(torch, "hpu") and
                                     torch.hpu.is_available() else "cpu"))


def _padded(lists, pad):
    width = max(1, max(len(x) for x in lists))
    return torch.tensor([x + [pad] * (width - len(x)) for x in lists], dtype=torch.int64)


def _reference(logits, prompts, outputs, pres, freq, rep):
    outs = [[t for t in o if t != -1] for o in outputs]
    return apply_penalties(logits.clone(), _padded(prompts, VOCAB), _padded(outs, VOCAB), pres, freq, rep)


def test_stats_match_v1_penalties():
    rng = random.Random(0)
    stats = HpuPenaltyStats(ROWS, VOCAB, DEVICE)
    token_ids = np.zeros((ROWS, MAX_LEN), dtype=np.int32)
    num_prompt = np.zeros(ROWS, dtype=np.int32)
    outputs: list[list[int]] = [[] for _ in range(ROWS)]

    def add(row):
        n = rng.randint(1, 20)
        token_ids[row, :n] = [rng.randrange(VOCAB) for _ in range(n)]
        num_prompt[row] = n
        outputs[row] = [rng.randrange(VOCAB) for _ in range(rng.randint(0, 3))]  # resumed after preemption
        stats.reset(row)

    for row in range(ROWS):
        add(row)
    for step in range(200):
        event = rng.random()
        if event < 0.1:
            i1, i2 = rng.sample(range(ROWS), 2)
            stats.swap(i1, i2)
            token_ids[[i1, i2]] = token_ids[[i2, i1]]
            num_prompt[[i1, i2]] = num_prompt[[i2, i1]]
            outputs[i1], outputs[i2] = outputs[i2], outputs[i1]
        elif event < 0.15:
            src, dst = rng.sample(range(ROWS), 2)
            stats.move(src, dst)
            token_ids[dst] = token_ids[src]
            num_prompt[dst] = num_prompt[src]
            outputs[dst] = list(outputs[src])
            add(src)
        elif event < 0.2:
            add(rng.randrange(ROWS))
        for row in range(ROWS):
            # repair placeholders late or never, then append this step's token (sometimes as a placeholder)
            for pos, tok in enumerate(outputs[row]):
                if tok == -1 and rng.random() < 0.6:
                    outputs[row][pos] = rng.randrange(VOCAB)
            outputs[row].append(-1 if rng.random() < 0.3 else rng.randrange(VOCAB))
        rows = list(range(ROWS)) + [0, 0]  # padded batch: duplicate first row
        stats.sync(rows, [outputs[r] for r in rows], token_ids, num_prompt)
        mask, counts = stats.gather(rows)
        logits = torch.randn(len(rows), VOCAB)
        pres, freq = torch.rand(len(rows)), torch.rand(len(rows))
        rep = 1 + torch.rand(len(rows))
        prompts = [token_ids[r, :num_prompt[r]].tolist() for r in rows]
        ref = _reference(logits, prompts, [outputs[r] for r in rows], pres, freq, rep)
        got = apply_penalties_from_stats(*(t.to(DEVICE) for t in (logits, mask, counts, pres, freq, rep)))
        torch.testing.assert_close(got.cpu(), ref, msg=f"step {step}")


def test_nonblocking_copy_snapshots_the_host_buffer():
    """HpuPenaltyStats (and the packed vision tower) copy temporary host arrays without blocking
    (utils.async_h2d_copy): the device copy must hold the values at the time of the call even when the host buffer
    is overwritten right after it."""
    from vllm_gaudi.utils import async_h2d_copy
    rng = np.random.default_rng(0)
    for _ in range(200):
        host = rng.integers(0, 1 << 30, size=4096, dtype=np.int64)
        expected = host.copy()
        dev = async_h2d_copy(torch.from_numpy(host), device=DEVICE)
        host[:] = -1  # overwrite before anything synchronizes
        assert torch.equal(dev.cpu(), torch.from_numpy(expected))
