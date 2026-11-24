# SPDX-License-Identifier: Apache-2.0

from vllm.v1.sample import rejection_sampler
import torch
from typing import Optional
from vllm.v1.sample.metadata import SamplingMetadata

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE


def rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Greedy rejection sampling on device, without host syncs (async scheduling keeps the result on device).

    HPU spec decode pads every request to max_draft_tokens rows, so the drafts and targets are
    [num_seqs * max_draft_tokens] and bonus_token_ids is [num_seqs, 1]. A request accepts its drafts up to the
    first mismatch and emits the target token there; if all its drafts match it also emits the bonus token.
    Returns [num_seqs, max_draft_tokens + 1] int32 with PLACEHOLDER_TOKEN_ID in the unused slots.
    """
    num_seqs = len(num_draft_tokens)
    drafts = padded_draft_token_ids.view(num_seqs, -1).to(torch.int32)
    targets = padded_target_token_ids.view(num_seqs, -1).to(torch.int32)
    bonus = bonus_token_ids.view(num_seqs, 1).to(torch.int32)
    max_draft_tokens = drafts.shape[-1]
    device = drafts.device
    cu = cu_num_draft_tokens.to(device=device, dtype=torch.int32)
    num_drafts = torch.diff(cu, prepend=cu.new_zeros(1)).view(num_seqs, 1)

    pos = torch.arange(max_draft_tokens + 1, device=device, dtype=torch.int32).view(1, -1)
    mismatch = (drafts != targets) & (pos[:, :max_draft_tokens] < num_drafts)
    any_mismatch = mismatch.any(dim=1, keepdim=True)
    first_mismatch = torch.argmax(mismatch.to(torch.int32), dim=1, keepdim=True).to(torch.int32)
    # Tokens emitted: up to and including the first mismatch, else all drafts' targets plus the bonus.
    num_emitted = torch.where(any_mismatch, first_mismatch + 1, num_drafts + 1)
    placeholder = torch.full((num_seqs, 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device)
    candidates = torch.where(pos == num_drafts, bonus, torch.cat((targets, placeholder), dim=1))
    return torch.where(pos < num_emitted, candidates, placeholder)


def rejection_random_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    is_greedy: torch.Tensor,
    uniform_probs: torch.Tensor,
    inv_q: torch.Tensor,
) -> torch.Tensor:
    """Rejection sampling for deterministic drafts (no draft probs, e.g. DFlash), on device.

    target_logits [num_seqs * max_draft_tokens, vocab] after temperature / top-k / top-p; is_greedy [num_seqs];
    uniform_probs [num_seqs * max_draft_tokens]; inv_q [num_seqs, vocab] (1 / Exp(1) noise). A random request
    accepts draft x with probability p(x) and at the first rejection emits a sample of p with x removed
    (argmax of p * inv_q, as upstream's recovered tokens); a greedy request accepts x iff it is the argmax and
    emits the argmax. Returns [num_seqs, max_draft_tokens + 1] int32 like rejection_sample_pytorch.
    """
    num_seqs = len(num_draft_tokens)
    drafts = padded_draft_token_ids.view(num_seqs, -1).long()
    max_draft_tokens = drafts.shape[-1]
    device = drafts.device
    probs = target_logits.float().softmax(dim=-1).view(num_seqs, max_draft_tokens, -1)
    argmax = probs.argmax(dim=-1)
    p_draft = probs.gather(-1, drafts.unsqueeze(-1)).squeeze(-1)
    greedy = is_greedy.view(num_seqs, 1)
    accept = torch.where(greedy, drafts == argmax, uniform_probs.view(num_seqs, -1).to(p_draft.dtype) <= p_draft)
    vocab = torch.arange(probs.shape[-1], device=device).view(1, 1, -1)
    residual = torch.where(vocab == drafts.unsqueeze(-1), 0.0, probs) * inv_q.view(num_seqs, 1, -1)
    recovered = torch.where(greedy, argmax, residual.argmax(dim=-1)).to(torch.int32)

    cu = cu_num_draft_tokens.to(device=device, dtype=torch.int32)
    num_drafts = torch.diff(cu, prepend=cu.new_zeros(1)).view(num_seqs, 1)
    pos = torch.arange(max_draft_tokens + 1, device=device, dtype=torch.int32).view(1, -1)
    reject = ~accept & (pos[:, :max_draft_tokens] < num_drafts)
    any_reject = reject.any(dim=1, keepdim=True)
    first_reject = torch.argmax(reject.to(torch.int32), dim=1, keepdim=True).to(torch.int32)
    num_emitted = torch.where(any_reject, first_reject + 1, num_drafts + 1)
    placeholder = torch.full((num_seqs, 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device)
    tokens = torch.cat((drafts.to(torch.int32), placeholder), dim=1)
    tokens = torch.where(any_reject & (pos == first_reject), torch.cat((recovered, placeholder), dim=1), tokens)
    tokens = torch.where(~any_reject & (pos == num_drafts), bonus_token_ids.view(num_seqs, 1).to(torch.int32),
                         tokens)
    return torch.where(pos < num_emitted, tokens, placeholder)


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
    use_fp64_gumbel: bool = False,
    fly_window_size: Optional[int] = None,
    fly_entropy_threshold: Optional[float] = None,
    fly_entropy_top_k: int = 3,
) -> torch.Tensor:
    # The FLy deferred-verification knobs are accepted for parity; FLy itself is not supported on HPU.
    assert fly_window_size is None, "FLy speculative decoding is not supported on HPU."
    if not sampling_metadata.all_greedy:
        # target_probs holds the processed target logits (upstream renamed it target_logits).
        assert draft_probs is None, "HPU random rejection sampling supports deterministic drafts only"
        assert not sampling_metadata.generators, "Seeded requests are not supported with HPU spec decode"
        num_seqs = len(num_draft_tokens)
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
        uniform_probs = torch.rand(draft_token_ids.shape[0], device=target_probs.device)
        inv_q = torch.empty(num_seqs, target_probs.shape[-1], device=target_probs.device).exponential_().reciprocal()
        return rejection_random_sample_pytorch(draft_token_ids, target_probs, bonus_token_ids, num_draft_tokens,
                                               cu_num_draft_tokens, is_greedy[:num_seqs], uniform_probs, inv_q)

    target_argmax = target_probs.argmax(dim=-1)
    output_token_ids = rejection_sample_pytorch(draft_token_ids, target_argmax, bonus_token_ids, num_draft_tokens,
                                                cu_num_draft_tokens)
    return output_token_ids


def expand_batch_to_tokens(
    x: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """PyTorch replacement for the Triton ``expand_batch_to_tokens``: repeat x[i] for each token of
    request i, mapping ``replace_from`` to ``replace_to``."""
    counts = torch.diff(cu_num_tokens, prepend=cu_num_tokens.new_zeros(1))
    x = torch.where(x == replace_from, torch.full_like(x, replace_to), x)
    return torch.repeat_interleave(x, counts.long(), output_size=num_tokens)


rejection_sampler.rejection_sample = rejection_sample
rejection_sampler.expand_batch_to_tokens = expand_batch_to_tokens
