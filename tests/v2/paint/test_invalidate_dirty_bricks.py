"""The multiscale paint controller invalidates GPU data over dirty bricks."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from cellier.paint._multiscale import MultiscalePaintController
from cellier.paint._write_layer import BrickKey, WriteLayer


def _paint_controller(dirty: set[BrickKey], block_size: int = 16):
    calls: list = []
    store = SimpleNamespace(id=uuid4())
    layer = WriteLayer(data_store_id=store.id, block_size=block_size)
    for key in dirty:
        layer.mark_dirty(key)
    paint = object.__new__(MultiscalePaintController)
    paint._write_layer = layer
    paint._data_store = store
    paint._controller = SimpleNamespace(
        _invalidate_painted_regions=lambda sid, regions: (
            calls.append((sid, regions)) or [7]
        )
    )
    return paint, store, calls


def test_each_dirty_brick_becomes_a_level_0_region() -> None:
    paint, store, calls = _paint_controller(
        {BrickKey(0, (2, 0, 1)), BrickKey(0, (2, 3, 3))}
    )
    assert paint._invalidate_dirty_bricks() == [7]
    ((store_id, regions),) = calls
    assert store_id == store.id
    # Every data axis, collapsed ones included, in data-axis order.
    assert regions == (
        ((32, 48), (0, 16), (16, 32)),
        ((32, 48), (48, 64), (48, 64)),
    )


def test_nothing_dirty_touches_nothing() -> None:
    paint, _store, calls = _paint_controller(set())
    assert paint._invalidate_dirty_bricks() == []
    assert calls == []
