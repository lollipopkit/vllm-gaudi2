# SPDX-License-Identifier: Apache-2.0
"""The numpy paged-attention buffers (flat_paged_blocks / pad_paged_blocks) against the list implementation they
replaced in HPUModelRunner.get_habana_paged_attn_buffers."""
import itertools
import random

import numpy as np
import pytest

from vllm_gaudi.v1.worker.hpu_model_runner import flat_paged_blocks, pad_paged_blocks

PAD_BLOCK_ID = 999


def _reference(block_tables, last_block_usage, block_size, bucket, contiguous):
    block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
    block_usage = [[block_size] * (len(bt) - 1) + [lbu] for bt, lbu in zip(block_tables, last_block_usage) if bt]
    block_list = list(itertools.chain(*block_tables))
    block_groups = list(itertools.chain(*block_groups))
    block_usage = list(itertools.chain(*block_usage))
    if contiguous:
        indices = [None] * bucket
        for i, bid in enumerate(block_list):
            indices[bid] = i

        def pad(values, v):
            return [values[i] if i is not None else v for i in indices]
    else:

        def pad(values, v):
            return values + [v] * (bucket - len(values))

    return pad(block_list, PAD_BLOCK_ID), pad(block_groups, -1), pad(block_usage, 1)


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_matches_list_implementation(contiguous: bool, seed: int) -> None:
    rng = random.Random(seed)
    block_size, num_ids = 128, 200
    ids = rng.sample(range(num_ids - 1), num_ids - 1) + [num_ids - 1]  # distinct, ending with the bucket's last slot
    tables = []
    for _ in range(rng.randint(1, 20)):
        n = rng.choice([0, 1, 2, 5, 9])  # empty rows (padding) included
        tables.append(ids[-n:] if n else [])
        ids = ids[:-n] if n else ids
    if contiguous and len(tables) > 1 and tables[0]:
        tables[1] = tables[0][:1] + tables[1]  # a shared block (warmup tables, prefix caching): last occurrence kept
    # the runner passes one usage per slot-mapping row, padding rows included
    last_usage = [rng.randint(1, block_size) for _ in range(len(tables) + rng.randint(0, 8))]
    flat = flat_paged_blocks(tables, last_usage, block_size)
    total = sum(map(len, tables))
    bucket = num_ids if contiguous else total + rng.randint(0, 16)
    got = pad_paged_blocks(flat, (PAD_BLOCK_ID, -1, 1), bucket, contiguous)
    want = _reference(tables, last_usage, block_size, bucket, contiguous)
    for g, w in zip(got, want):
        assert g.dtype == np.int32 and g.tolist() == w


def test_empty_batch() -> None:
    flat = flat_paged_blocks([[], []], [1, 1], 128)
    got = pad_paged_blocks(flat, (PAD_BLOCK_ID, -1, 1), 4, False)
    assert [g.tolist() for g in got] == [[PAD_BLOCK_ID] * 4, [-1] * 4, [1] * 4]
