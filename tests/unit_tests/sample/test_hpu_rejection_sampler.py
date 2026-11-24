# SPDX-License-Identifier: Apache-2.0
"""CPU test of the HPU greedy rejection sampler against a per-request reference."""
import torch

from vllm_gaudi.v1.sample.hpu_rejection_sampler import (PLACEHOLDER_TOKEN_ID, rejection_random_sample_pytorch,
                                                        rejection_sample_pytorch)


def _reference(drafts, targets, bonus, num_drafts, k):
    out = []
    for d, t, b, n in zip(drafts, targets, bonus, num_drafts):
        row = []
        for j in range(n):
            row.append(t[j])
            if d[j] != t[j]:
                break
        else:
            row.append(b)
        out.append(row + [PLACEHOLDER_TOKEN_ID] * (k + 1 - len(row)))
    return out


def test_rejection_sample_matches_reference():
    torch.manual_seed(0)
    k = 7
    num_drafts = [7, 7, 0, 3, 7, 5]
    n = len(num_drafts)
    targets = torch.randint(0, 5, (n, k))
    drafts = targets.clone()
    drafts[0, 2] += 1  # mismatch at 2
    drafts[3, 0] += 1  # mismatch at 0
    drafts[4, 6] += 1  # mismatch at the last draft
    drafts[5, 5:] += 1  # mismatch only in padding: all 5 accepted
    bonus = torch.randint(0, 5, (n, 1))
    cu = torch.tensor(num_drafts).cumsum(0)
    out = rejection_sample_pytorch(drafts.reshape(-1, 1), targets.reshape(-1), bonus, num_drafts, cu)
    assert out.dtype == torch.int32
    assert out.tolist() == _reference(drafts.tolist(), targets.tolist(), bonus.flatten().tolist(), num_drafts, k)


def test_random_rejection_preserves_target_distribution():
    """Speculative sampling with a deterministic draft emits the first token with the target distribution."""
    torch.manual_seed(0)
    n, vocab = 40000, 6
    probs = torch.tensor([0.05, 0.4, 0.1, 0.25, 0.15, 0.05])
    logits = probs.log().expand(n, vocab).contiguous()
    drafts = torch.full((n, 1), 3)
    bonus = torch.full((n, 1), 5)
    out = rejection_random_sample_pytorch(drafts, logits, bonus, [1] * n, torch.arange(1, n + 1),
                                          torch.zeros(n, dtype=torch.bool), torch.rand(n),
                                          torch.empty(n, vocab).exponential_().reciprocal())
    first = torch.bincount(out[:, 0].long(), minlength=vocab).float() / n
    torch.testing.assert_close(first, probs, atol=0.01, rtol=0)
    # Accepted rows (draft 3, p = 0.25) also emit the bonus token.
    assert abs((out[:, 1] == 5).float().mean().item() - 0.25) < 0.01


def test_random_rejection_greedy_rows_match_greedy_path():
    torch.manual_seed(0)
    k, num_drafts = 4, [4, 4, 2]
    n = len(num_drafts)
    logits = torch.randn(n * k, 9)
    targets = logits.argmax(-1).view(n, k)
    drafts = targets.clone()
    drafts[0, 1] = (drafts[0, 1] + 1) % 9
    bonus = torch.randint(0, 9, (n, 1))
    cu = torch.tensor(num_drafts).cumsum(0)
    out = rejection_random_sample_pytorch(drafts.reshape(-1, 1), logits, bonus, num_drafts, cu,
                                          torch.ones(n, dtype=torch.bool), torch.rand(n * k),
                                          torch.empty(n, 9).exponential_().reciprocal())
    ref = rejection_sample_pytorch(drafts.reshape(-1, 1), targets.reshape(-1), bonus, num_drafts, cu)
    assert out.tolist() == ref.tolist()
