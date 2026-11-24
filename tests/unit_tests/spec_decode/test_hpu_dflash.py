# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the torch-only DFlash2 pieces (vllm_gaudi.v1.spec_decode.dflash_ops)."""
from types import SimpleNamespace

import torch

from vllm_gaudi.v1.spec_decode.dflash_ops import (block_mapping_from_groups, build_query_layout, chunked_topk,
                                                  compute_context_kv,
                                                  greedy_candidate_walk, rms_norm, score_edges,
                                                  window_block_bias)


def _walk_reference(scores: torch.Tensor, candidate_ids: torch.Tensor) -> list[list[int]]:
    """Python port of upstream _selector_walk_kernel at temperature 0 on flat buffers."""
    batch_size, num_steps, top_k, _ = scores.shape
    flat_scores = scores.flatten().tolist()
    flat_candidates = candidate_ids.flatten().tolist()
    out = []
    for row in range(batch_size):
        previous = 0
        tokens = []
        for step in range(num_steps):
            flat = row * num_steps + step
            base = (flat * top_k + previous) * top_k
            values = flat_scores[base:base + top_k]
            index = values.index(max(values))  # tl.max(return_indices) picks the first maximum
            tokens.append(flat_candidates[flat * top_k + index])
            previous = index
        out.append(tokens)
    return out


def test_greedy_candidate_walk_matches_kernel():
    torch.manual_seed(0)
    batch_size, num_steps, top_k = 3, 7, 16
    scores = torch.randn(batch_size, num_steps, top_k, top_k)
    # Ties resolve to the first maximum; integer scores make ties likely.
    scores[1] = torch.randint(0, 3, (num_steps, top_k, top_k)).float()
    candidate_ids = torch.randint(0, 248320, (batch_size, num_steps, top_k))
    tokens = greedy_candidate_walk(scores, candidate_ids)
    assert tokens.shape == (batch_size, num_steps)
    assert tokens.tolist() == _walk_reference(scores, candidate_ids)


def _neox_rope(positions: torch.Tensor, x: torch.Tensor, cos_sin_cache: torch.Tensor, head_dim: int) -> torch.Tensor:
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    x = x.view(x.shape[0], -1, head_dim)
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(1)


def test_compute_context_kv_matches_per_layer_projection():
    torch.manual_seed(0)
    num_layers, hidden, num_heads, num_kv_heads, head_dim, num_ctx, eps = 3, 64, 4, 2, 16, 5, 1e-6
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    states = torch.randn(num_ctx, hidden)
    positions = torch.tensor([0, 3, 7, 8, 30])
    norm_weight = torch.rand(hidden) + 0.5
    qkv_weights = [torch.randn(q_size + 2 * kv_size, hidden) / 8 for _ in range(num_layers)]
    k_norm_weight = torch.rand(num_layers, head_dim) + 0.5
    inv_freq = 1.0 / (10000**(torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.arange(64).float()[:, None] * inv_freq[None]
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    keys, values = compute_context_kv(
        states,
        positions,
        norm_weight=norm_weight,
        eps=eps,
        kv_weight=torch.cat([w[q_size:] for w in qkv_weights]),
        kv_bias=None,
        k_norm_weight=k_norm_weight,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope=lambda pos, k: _neox_rope(pos, k, cos_sin_cache, head_dim),
        v_scale=0.5,
    )
    assert keys.shape == values.shape == (num_layers, num_ctx, num_kv_heads, head_dim)

    var = states.pow(2).mean(-1, keepdim=True)
    normed = states * torch.rsqrt(var + eps) * norm_weight
    for layer, weight in enumerate(qkv_weights):
        _, k, v = (normed @ weight.T).split([q_size, kv_size, kv_size], dim=-1)
        k = k.view(num_ctx, num_kv_heads, head_dim)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * k_norm_weight[layer]
        k = _neox_rope(positions, k.flatten(1), cos_sin_cache, head_dim)
        torch.testing.assert_close(keys[layer], k.view(num_ctx, num_kv_heads, head_dim))
        torch.testing.assert_close(values[layer], 0.5 * v.view(num_ctx, num_kv_heads, head_dim))


def test_rms_norm_rounds_like_vllm():
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    weight = torch.rand(32, dtype=torch.bfloat16)
    x32 = x.float()
    expected = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * weight
    torch.testing.assert_close(rms_norm(x, weight, 1e-6), expected, rtol=0, atol=0)


def _layout(sliding_window=None, causal=False, padded_batch_size=3):
    # Block ids are unique so a block id identifies (request, logical block).
    block_tables = [[10, 11, 12, 13], [20, 21]]
    return block_tables, build_query_layout([9, 2], [111, 222],
                                            block_tables,
                                            num_query_per_req=4,
                                            mask_token_id=7,
                                            block_size=4,
                                            padded_batch_size=padded_batch_size,
                                            pad_slot_id=1000,
                                            sliding_window=sliding_window,
                                            causal=causal)


def _slot(block_tables, req, position, block_size=4):
    return block_tables[req][position // block_size] * block_size + position % block_size


def test_query_layout_rows():
    block_tables, layout = _layout()
    assert layout.input_ids.view(-1).tolist() == [111, 7, 7, 7, 222, 7, 7, 7] + [0] * 4
    assert layout.positions.view(-1).tolist()[:8] == [9, 10, 11, 12, 2, 3, 4, 5]
    slots = layout.slot_mapping.view(-1).tolist()
    assert slots[:8] == [_slot(block_tables, 0, q) for q in (9, 10, 11, 12)] + \
        [_slot(block_tables, 1, q) for q in (2, 3, 4, 5)]
    assert slots[8:] == [1000, 1001, 1002, 1003]  # padding rows write to pad slots
    # Non-causal: every row of a request sees its whole block.
    assert layout.row_block_tables == [[10, 11, 12, 13]] * 4 + [[20, 21]] * 4
    assert layout.row_last_slots == [[_slot(block_tables, 0, 12)]] * 4 + [[_slot(block_tables, 1, 5)]] * 4
    assert layout.row_first_offsets == [0] * 8


def test_query_layout_causal_rows_end_at_their_position():
    block_tables, layout = _layout(causal=True)
    assert layout.row_last_slots[:4] == [[_slot(block_tables, 0, q)] for q in (9, 10, 11, 12)]
    assert layout.row_block_tables[:4] == [[10, 11, 12]] * 3 + [[10, 11, 12, 13]]


def _paged_buffers(row_block_tables, row_last_slots, block_size, padded_len):
    """Non-contiguous HPUModelRunner.get_habana_paged_attn_buffers without bucketing."""
    block_list = [b for bt in row_block_tables for b in bt]
    groups = [i for i, bt in enumerate(row_block_tables) for _ in bt]
    usage = [
        u for bt, s in zip(row_block_tables, row_last_slots) if bt
        for u in [block_size] * (len(bt) - 1) + [s[0] % block_size + 1]
    ]
    pad = padded_len - len(block_list)
    return (torch.tensor(block_list + [0] * pad), torch.tensor(groups + [-1] * pad),
            torch.tensor(usage + [1] * pad, dtype=torch.bfloat16))


def test_sliding_window_bias_matches_symmetric_window():
    window, block_size = 6, 4
    block_tables, layout = _layout(sliding_window=window)
    block_list, groups, usage = _paged_buffers(layout.row_block_tables, layout.row_last_slots, block_size, 32)
    bias = window_block_bias(layout.row_block_tables, layout.row_first_offsets, usage, block_size, torch.bfloat16)
    logical = {b: (req, i) for req, bt in enumerate(block_tables) for i, b in enumerate(bt)}
    for row, (p, req) in enumerate([(9, 0)] * 4 + [(2, 1)] * 4):
        q = p + row % 4
        visible = {
            logical[int(block_list[e])][1] * block_size + j
            for e in range(len(block_list)) if int(groups[e]) == row for j in range(block_size) if bias[e, j] == 0
        }
        assert visible == set(range(max(0, q - window + 1), p + 4)), (row, sorted(visible))
    assert torch.isinf(bias[len(groups[groups >= 0]):]).sum() == (32 - len(groups[groups >= 0])) * (block_size - 1)


def test_block_mapping_from_groups_skips_padding():
    mapping = block_mapping_from_groups(torch.tensor([0, 0, 2, -1]), 3, torch.bfloat16)
    assert mapping.tolist() == [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]]


def test_dflash_aux_layers_from_config():
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import get_eagle3_aux_layers_from_config
    hf_config = SimpleNamespace(dflash_config={"target_layer_ids": [5, 19, 33, 47, 61]})
    spec_config = SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf_config))
    assert get_eagle3_aux_layers_from_config(spec_config) == (6, 20, 34, 48, 62)


def test_draft_kv_cache_spec_joins_target_attention_group():
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

    def spec(num_kv_heads, head_size):
        return FullAttentionSpec(block_size=128, num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.bfloat16)

    runner = SimpleNamespace(drafter=SimpleNamespace(_draft_attn_layer_names={"d.0", "d.1", "d.2"}),
                             _draft_kv_cache_views={})
    specs = {"t.0": spec(4, 256), "t.1": spec(4, 256), "d.0": spec(8, 128), "d.1": spec(4, 256), "d.2": spec(8, 64)}
    HPUModelRunner._alias_draft_kv_cache_specs(runner, specs)
    assert specs["d.0"] == specs["t.0"] and specs["d.1"] == specs["t.0"]
    assert specs["d.2"] == spec(8, 64)  # different per-token size: own group
    assert runner._draft_kv_cache_views == {"d.0": (8, 128)}


def test_score_edges_matches_einsum():
    torch.manual_seed(0)
    vocab, rank, batch_size, num_steps, top_k = 50, 8, 3, 4, 5
    pred, succ = torch.randn(vocab, rank), torch.randn(vocab, rank)
    ids = torch.randint(0, vocab, (batch_size, num_steps, top_k))
    unary, hidden = torch.randn(batch_size, num_steps, top_k), torch.randn(batch_size, num_steps, rank)
    anchors = torch.randint(0, vocab, (batch_size, ))
    # Upstream vllm.model_executor.models.qwen3_dflash2._score_edges
    pred_ids = torch.cat((anchors[:, None, None].expand(-1, 1, top_k), ids[:, :-1]), dim=1)
    ref = unary[:, :, None] + torch.einsum("blpr,blcr->blpc", pred[pred_ids] * hidden[:, :, None], succ[ids])
    torch.testing.assert_close(score_edges(pred, succ, ids, unary, hidden, anchors), ref)


def test_chunked_topk_matches_topk():
    torch.manual_seed(0)
    scores = torch.randn(5, 128 * 40)
    values, ids = chunked_topk(scores, 16)
    ref_values, ref_ids = torch.topk(scores, 16, dim=-1)
    torch.testing.assert_close(values, ref_values)
    assert torch.equal(ids, ref_ids)


def test_store_context_kv_accepts_int32_slot_mapping_eager():
    from vllm_gaudi.extension.utils import VLLMKVCache
    from vllm_gaudi.models.qwen3_dflash2 import HpuDFlash2Qwen3Model

    k_cache = torch.zeros(16, 2, 4, dtype=torch.bfloat16)
    v_cache = torch.zeros(16, 2, 4, dtype=torch.bfloat16)
    attn = SimpleNamespace(
        kv_cache=(k_cache, v_cache),
        impl=SimpleNamespace(k_cache=VLLMKVCache(), v_cache=VLLMKVCache(is_v_cache=True)),
    )
    model = SimpleNamespace(_attn_layers=[attn])
    keys = torch.arange(1, 17, dtype=torch.bfloat16).view(1, 2, 2, 4)
    values = -keys
    slots = torch.tensor([3, 11], dtype=torch.int32)
    HpuDFlash2Qwen3Model.store_context_kv(model, keys, values, slots)
    torch.testing.assert_close(k_cache[slots.long()], keys[0])
    torch.testing.assert_close(v_cache[slots.long()], values[0])


def test_hpu_dflash_proposer_init_parallel_drafting_defaults(monkeypatch):
    from vllm_gaudi.v1._vendored.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    from vllm_gaudi.v1.spec_decode.hpu_dflash import HpuDFlashProposer

    hf_config = SimpleNamespace(
        dflash_config={"mask_token_id": 248070},
        layer_types=["sliding_attention"],
        num_hidden_layers=1,
        is_causal=False,
    )
    for parallel_drafting_cfg in (None, True):
        spec_config = SimpleNamespace(
            method="dflash",
            num_speculative_tokens=7,
            parallel_drafting=parallel_drafting_cfg,
            draft_model_config=SimpleNamespace(hf_config=hf_config),
        )
        vllm_config = SimpleNamespace(speculative_config=spec_config)

        def fake_super_init(self, *, vllm_config, device, pass_hidden_states_to_model, runner):
            self.vllm_config = vllm_config
            self.speculative_config = vllm_config.speculative_config
            self.draft_model_config = vllm_config.speculative_config.draft_model_config
            self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
            self.parallel_drafting = vllm_config.speculative_config.parallel_drafting
            self.parallel_drafting_token_id = 0
            self.parallel_drafting_hidden_state_tensor = (
                torch.empty(4) if parallel_drafting_cfg else None
            )

        monkeypatch.setattr(SpecDecodeBaseProposer, "__init__", fake_super_init)
        proposer = HpuDFlashProposer(vllm_config=vllm_config, device=torch.device("cpu"))
        assert proposer.parallel_drafting is True
        assert proposer.parallel_drafting_token_id == 248070
        assert proposer.parallel_drafting_hidden_state_tensor is None
        assert proposer.extra_slots_per_request == 7
        assert proposer.net_num_new_slots_per_request == 7
        assert proposer.needs_extra_input_slots is True
        assert proposer.dflash_causal is False
