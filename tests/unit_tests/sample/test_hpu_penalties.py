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


def test_two_stage_topk_exact_for_lfm3_inkling2_qwen_vocabs():
    """64-group two-stage _topk must match torch.topk identically across LFM3 (131072), Inkling2 (201024),
    and Qwen3.5 (248320) vocab sizes."""
    from vllm_gaudi.ops import hpu_sampler

    torch.manual_seed(42)
    for vocab in (65536, 131072, 201024, 248320):
        logits = torch.randn(4, vocab, device=DEVICE, dtype=torch.float32)
        got_vals, got_idx = hpu_sampler._topk(logits, 64)
        ref_vals, _ = logits.topk(64, dim=-1)
        assert torch.equal(got_vals.cpu(), ref_vals.cpu())
        assert torch.equal(logits.gather(1, got_idx.long()).cpu(), ref_vals.cpu())
        for r in range(4):
            assert torch.unique(got_idx[r].cpu()).numel() == 64


def test_hpu_sampler_fast_paths_and_fallback():
    """HpuSampler.compile_sampler fast paths cover greedy, random, top-k, and top-k+top-p with and without
    device penalty statistics, and fall back cleanly when logprobs or generators are active."""
    from vllm_gaudi import patches
    from vllm_gaudi.ops import hpu_sampler
    from vllm_gaudi.v1._vendored.sample.logits_processor import LogitsProcessors
    from vllm_gaudi.v1.sample.hpu_penalties import HpuSampler, HpuSamplingMetadata

    patches._patch_gather_logprobs()

    vocab = 4096  # 64 * 64
    rows = 4
    stats = HpuPenaltyStats(rows, vocab, DEVICE)
    token_ids = np.zeros((rows, 16), dtype=np.int32)
    token_ids[:, 0] = 7
    num_prompt = np.ones(rows, dtype=np.int32)
    stats.sync(list(range(rows)), [[11] for _ in range(rows)], token_ids, num_prompt)
    mask, counts = stats.gather(list(range(rows)))

    sampler = HpuSampler()
    sampler.compile_sampler(lambda fn, **kw: fn)
    sampler.warmup_penalties(stats, rows, torch.bfloat16)

    logits = torch.randn(rows, vocab, device=DEVICE, dtype=torch.bfloat16)
    logits[:, 7] = 20.0
    logits[:, 11] = 19.0
    logits[:, 19] = 18.0
    lp = LogitsProcessors()
    zero = torch.zeros(rows, device=DEVICE, dtype=torch.float32)
    rep = torch.full((rows, ), 100.0, device=DEVICE, dtype=torch.float32)
    temp = torch.full((rows, ), 0.1, device=DEVICE, dtype=torch.float32)
    topk = torch.full((rows, ), 8, device=DEVICE, dtype=torch.int32)
    topp = torch.full((rows, ), 0.9, device=DEVICE, dtype=torch.float32)

    # Greedy without penalties picks 7; with heavy repetition_penalty=100 on {7, 11}, picks 19.
    sm_g = HpuSamplingMetadata(
        temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
        generators={}, max_num_logprobs=None, no_penalties=True, prompt_token_ids=None,
        frequency_penalties=zero, presence_penalties=zero, repetition_penalties=rep,
        output_token_ids=[[] for _ in range(rows)], allowed_token_ids_mask=None,
        bad_words_token_ids={}, logitsprocs=lp,
    )
    assert sampler(logits, sm_g).sampled_token_ids.cpu().squeeze(-1).tolist() == [7] * rows

    sm_gp = HpuSamplingMetadata(
        temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
        generators={}, max_num_logprobs=None, no_penalties=False, prompt_token_ids=None,
        frequency_penalties=zero, presence_penalties=zero, repetition_penalties=rep,
        output_token_ids=[[11] for _ in range(rows)], allowed_token_ids_mask=None,
        bad_words_token_ids={}, logitsprocs=lp, penalty_prompt_mask=mask, penalty_output_counts=counts,
    )
    assert sampler(logits, sm_gp).sampled_token_ids.cpu().squeeze(-1).tolist() == [19] * rows

    # Random, top-k, and top-k + top-p with penalties all stay inside valid vocab range and pick peaking survivor 19.
    logits_sharp = torch.full((rows, vocab), -50.0, device=DEVICE, dtype=torch.bfloat16)
    logits_sharp[:, 7] = 50.0
    logits_sharp[:, 19] = 40.0
    sampler.topk_topp_sampler.hpu_topk_cap = hpu_sampler.TOPK_CAP
    for k_arg, p_arg in ((None, None), (topk, None), (topk, topp)):
        sm_r = HpuSamplingMetadata(
            temperature=temp, all_greedy=False, all_random=True, top_p=p_arg, top_k=k_arg,
            generators={}, max_num_logprobs=None, no_penalties=False, prompt_token_ids=None,
            frequency_penalties=zero, presence_penalties=zero, repetition_penalties=rep,
            output_token_ids=[[11] for _ in range(rows)], allowed_token_ids_mask=None,
            bad_words_token_ids={}, logitsprocs=lp, penalty_prompt_mask=mask, penalty_output_counts=counts,
        )
        out = sampler(logits_sharp, sm_r)
        assert out.sampled_token_ids.shape == (rows, 1)
        assert out.sampled_token_ids.dtype == torch.int32
        assert out.sampled_token_ids.cpu().squeeze(-1).tolist() == [19] * rows

    # Fallback path when max_num_logprobs is requested returns logprobs_tensors cleanly.
    sm_lp = HpuSamplingMetadata(
        temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
        generators={}, max_num_logprobs=2, no_penalties=False, prompt_token_ids=None,
        frequency_penalties=zero, presence_penalties=zero, repetition_penalties=rep,
        output_token_ids=[[11] for _ in range(rows)], allowed_token_ids_mask=None,
        bad_words_token_ids={}, logitsprocs=lp, penalty_prompt_mask=mask, penalty_output_counts=counts,
    )
    out_lp = sampler(logits, sm_lp)
    assert out_lp.logprobs_tensors is not None
    assert out_lp.sampled_token_ids.cpu().squeeze(-1).tolist() == [19] * rows

    # Edge cases: non-contiguous sliced logits, tiny temperature (1e-5), p=1.0 / p=0.0, and small vocab < TOPK_CAP.
    padded_logits = torch.randn(rows, vocab + 64, device=DEVICE, dtype=torch.bfloat16)
    sliced_logits = padded_logits[:, :vocab]
    assert not sliced_logits.is_contiguous()
    vals_nc, idx_nc = hpu_sampler._topk(sliced_logits, hpu_sampler.TOPK_CAP)
    ref_vals_nc, _ = sliced_logits.topk(hpu_sampler.TOPK_CAP, dim=-1)
    assert torch.equal(vals_nc, ref_vals_nc)
    assert torch.equal(sliced_logits.gather(1, idx_nc.long()), ref_vals_nc)

    tiny_temp = torch.full((rows, ), 1e-5, device=DEVICE, dtype=torch.float32)
    huge_logits = torch.full((rows, vocab), 1000.0, device=DEVICE, dtype=torch.bfloat16)
    huge_logits[:, 23] = 1024.0
    sm_tiny = HpuSamplingMetadata(
        temperature=tiny_temp, all_greedy=False, all_random=True,
        top_p=torch.tensor([1.0, 0.0, 0.9, 1.0], device=DEVICE, dtype=torch.float32),
        top_k=topk, generators={}, max_num_logprobs=None, no_penalties=True, prompt_token_ids=None,
        frequency_penalties=zero, presence_penalties=zero, repetition_penalties=rep,
        output_token_ids=[[] for _ in range(rows)], allowed_token_ids_mask=None,
        bad_words_token_ids={}, logitsprocs=lp,
    )
    assert sampler(huge_logits, sm_tiny).sampled_token_ids.cpu().squeeze(-1).tolist() == [23] * rows

    class _DummyBatch:
        num_reqs = 2
        vocab_size = 32
        top_k_reqs = {"r0", "r1"}
        top_k_cpu = np.array([8, 8], dtype=np.int32)

    assert hpu_sampler.topk_cap(_DummyBatch) is None
    tiny_vocab_logits = torch.zeros(rows, 32, device=DEVICE, dtype=torch.float32)
    tiny_vocab_logits[:, 5] = 10.0
    sm_small = HpuSamplingMetadata(
        temperature=temp, all_greedy=False, all_random=True, top_p=topp,
        top_k=torch.full((rows, ), 4, device=DEVICE, dtype=torch.int32),
        generators={}, max_num_logprobs=None, no_penalties=True, prompt_token_ids=None,
        frequency_penalties=zero, presence_penalties=zero, repetition_penalties=rep,
        output_token_ids=[[] for _ in range(rows)], allowed_token_ids_mask=None,
        bad_words_token_ids={}, logitsprocs=lp,
    )
    assert sampler(tiny_vocab_logits, sm_small).sampled_token_ids.cpu().squeeze(-1).tolist() == [5] * rows
