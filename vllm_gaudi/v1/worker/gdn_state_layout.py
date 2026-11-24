# SPDX-License-Identifier: Apache-2.0
# Derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0.
"""CPU ownership proofs for the group-major compact GDN state pool."""

from collections.abc import Collection

import torch


def compact_gdn_state_index(group_offset: int,
                            base_slot: int,
                            max_num_reqs: int,
                            rows_per_req: int = 1,
                            t: int = 0) -> int:
    """Row ``t`` of ``base_slot`` in GDN group ``group_offset`` (row 0 is a sentinel).

    With speculative decoding a request owns ``rows_per_req = 1 + k`` consecutive rows per group: the SSM
    state after verify token t lives in row t, the committed state of a prefill or non-spec step in row k.
    """
    return (group_offset * max_num_reqs + base_slot) * rows_per_req + t + 1


def has_direct_gdn_state(
    state_indices: torch.Tensor,
    group_offsets: dict[int, int],
    max_num_reqs: int,
    num_reqs: int,
    padded_reqs: int,
    free_slots: Collection[int] = (),
) -> bool:
    """Prove that request ``i`` owns row ``i`` of every compact GDN group.

    ``state_indices`` is the ``[num_groups, padded_reqs]`` CPU tensor built for
    the decode batch. A contiguous view also updates the padding rows, so a
    padded batch qualifies only if the base slots of the padding rows are free
    (owned by no request; prefill zeroes the state of a request without initial
    state, so the garbage written there is never read). Must run on CPU metadata
    before the H2D copy so it never synchronizes the device.
    """
    if (not group_offsets or not 0 < num_reqs <= padded_reqs <= max_num_reqs or state_indices.device.type != "cpu"
            or state_indices.ndim != 2 or state_indices.shape[1] != padded_reqs):
        return False
    if any(slot not in free_slots for slot in range(num_reqs, padded_reqs)):
        return False
    base = torch.arange(num_reqs, dtype=state_indices.dtype)
    for group_idx, group_offset in group_offsets.items():
        if not 0 <= group_idx < state_indices.shape[0]:
            return False
        if not torch.equal(state_indices[group_idx, :num_reqs],
                           base + compact_gdn_state_index(group_offset, 0, max_num_reqs)):
            return False
    return True
