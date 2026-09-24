"""The per-cache registry and the read-only view it hands to adapters."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render.scheduling import (
    DEAD,
    CacheRegistry,
    ChunkState,
    RegistryView,
    Tier,
)


def _registry(keys: list[int]) -> CacheRegistry:
    reg = CacheRegistry()
    reg.insert(np.array(keys, dtype=np.int64), {"rank": np.arange(len(keys))})
    return reg


def test_insert_keeps_keys_sorted_with_their_values() -> None:
    reg = _registry([30, 10, 20])
    assert reg.key.tolist() == [10, 20, 30]
    # Values follow their key through the sort.
    assert reg.rank.tolist() == [1, 2, 0]
    assert (reg.state == ChunkState.QUEUED).all()
    assert (reg.tier == Tier.VISIBLE).all()
    assert (reg.slot == -1).all()


def test_find_and_find_many() -> None:
    reg = _registry([10, 20, 30])
    assert reg.find(20) == 1
    assert reg.find(25) == -1
    assert reg.find(99) == -1
    rows, found = reg.find_many(np.array([30, 5, 10, 40], dtype=np.int64))
    assert found.tolist() == [True, False, True, False]
    assert rows[found].tolist() == [2, 0]


def test_find_many_on_an_empty_registry() -> None:
    rows, found = CacheRegistry().find_many(np.array([1, 2], dtype=np.int64))
    assert not found.any()
    assert len(rows) == 2


def test_kill_is_a_tombstone_until_compaction() -> None:
    reg = _registry([10, 20, 30])
    reg.slot[1] = 4
    reg.kill(1)
    assert len(reg) == 2
    assert len(reg.key) == 3  # still there, dead
    assert reg.state[1] == DEAD
    assert reg.slot[1] == -1
    assert reg.find(20) == -1
    # Killing twice does not double count.
    reg.kill(np.array([1]))
    assert len(reg) == 2

    reg.compact()
    assert reg.key.tolist() == [10, 30]
    assert len(reg) == 2


def test_insert_after_kill_does_not_duplicate() -> None:
    reg = _registry([10, 20])
    reg.kill(reg.find(20))
    reg.insert(np.array([20], dtype=np.int64), {})
    assert reg.key.tolist() == [10, 20]
    assert reg.find(20) == 1


def test_view_is_read_only_and_compacted() -> None:
    reg = _registry([10, 20, 30])
    reg.kill(0)
    view = reg.view(generation=3, complete=False)
    assert isinstance(view, RegistryView)
    assert view.key.tolist() == [20, 30]
    assert view.generation == 3
    assert view.complete is False
    with pytest.raises(ValueError, match="read-only"):
        view.state[0] = 1


def _view(rows: list[tuple[int, int, int, int, int]], complete: bool) -> RegistryView:
    """``(key, state, tier, slice_id, wanted_gen)`` rows, sorted by key."""
    arr = np.array(rows, dtype=np.int64)
    arrays = {
        "key": arr[:, 0],
        "state": arr[:, 1].astype(np.uint8),
        "tier": arr[:, 2].astype(np.uint8),
        "cls": np.zeros(len(rows), np.uint8),
        "rank": np.zeros(len(rows), np.int32),
        "slot": np.arange(len(rows), dtype=np.int32),
        "slice_id": arr[:, 3].astype(np.int32),
        "wanted_gen": arr[:, 4].astype(np.int32),
    }
    return RegistryView.from_arrays(arrays, generation=9, complete=complete)


R, Q = int(ChunkState.RESIDENT), int(ChunkState.QUEUED)
V, RC = int(Tier.VISIBLE), int(Tier.RECENT)


def test_paint_groups_background_oldest_slice_first() -> None:
    view = _view(
        [
            (1, R, RC, 5, 2),  # slice 5, wanted at gen 2
            (2, R, RC, 7, 4),  # slice 7, gen 4 (newest background)
            (3, R, RC, 5, 3),  # slice 5 again: its newest is gen 3
            (4, R, V, 8, 9),  # foreground
            (5, Q, V, 8, 9),  # not resident: not drawn
            (6, R, RC, 6, 1),  # slice 6, oldest
        ],
        complete=False,
    )
    groups = [view.key[g].tolist() for g in view.paint_groups()]
    assert groups == [[6], [1, 3], [2], [4]]


def test_paint_groups_drop_the_background_once_complete() -> None:
    view = _view([(1, R, RC, 5, 2), (4, R, V, 8, 9)], complete=True)
    groups = [view.key[g].tolist() for g in view.paint_groups()]
    assert groups == [[4]]


def test_paint_groups_with_nothing_resident() -> None:
    view = _view([(1, Q, V, 5, 2)], complete=False)
    groups = view.paint_groups()
    assert len(groups) == 1
    assert len(groups[0]) == 0
