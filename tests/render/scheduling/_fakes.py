"""Fakes for the chunk scheduler tests: keys, a residency and stores.

Geometry is 2D with one collapsed axis ``t``, as in the design's model
(``scripts/progressive/_chunk_machine.py``): a ``BASE x BASE`` grid of
level-1 cells, levels ``1 .. N_LEVELS`` where a level-``l`` tile covers
``2 ** (l - 1)`` cells a side.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from cellier.render.scheduling import ChunkClass, DesiredSet

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from cellier.render.scheduling import RegistryView

BASE = 16
N_LEVELS = 4


def pack(level: int, t: int, gy: int, gx: int) -> int:
    """``level | t | gy | gx``, one byte each."""
    return (level << 24) | (t << 16) | (gy << 8) | gx


def unpack(key: int) -> tuple[int, int, int, int]:
    """``(level, t, gy, gx)``."""
    key = int(key)
    return (key >> 24) & 0xFF, (key >> 16) & 0xFF, (key >> 8) & 0xFF, key & 0xFF


def span(level: int) -> int:
    """Base cells covered by one tile of *level* along each axis."""
    return 1 << (level - 1)


def grid_dim(level: int) -> int:
    """Tiles along one axis at *level*."""
    return -(-BASE // span(level))


def cells_of(key: int) -> list[tuple[int, int]]:
    """The base cells a key's tile covers."""
    level, _, gy, gx = unpack(key)
    s = span(level)
    return [
        (y, x)
        for y in range(gy * s, min((gy + 1) * s, BASE))
        for x in range(gx * s, min((gx + 1) * s, BASE))
    ]


class FakeResidency:
    """A mosaic adapter: slots in a dict, a LUT of ``cell -> key``.

    ``rebuild_draw`` paints ``view.paint_groups()`` in order, each group
    coarsest to finest, as the image adapter will.
    """

    def __init__(self, n_slots: int) -> None:
        self.n_slots = n_slots
        self.slots: dict[int, tuple[int, Any]] = {}
        self.writes: list[tuple[int, int]] = []
        self.n_rebuilds = 0
        self.lut: dict[tuple[int, int], int] = {}
        self.lut_slot: dict[tuple[int, int], int] = {}
        self.last_complete: bool | None = None

    def write(self, slot: int, key: int, data: Any) -> None:
        assert 0 <= slot < self.n_slots, slot
        self.slots[slot] = (key, data)
        self.writes.append((slot, key))

    def rebuild_draw(self, view: RegistryView) -> None:
        self.n_rebuilds += 1
        self.last_complete = view.complete
        lut: dict[tuple[int, int], int] = {}
        lut_slot: dict[tuple[int, int], int] = {}
        for group in view.paint_groups():
            levels = np.array([unpack(k)[0] for k in view.key[group]], dtype=np.int64)
            for i in group[np.argsort(-levels, kind="stable")].tolist():
                key = int(view.key[i])
                for cell in cells_of(key):
                    lut[cell] = key
                    lut_slot[cell] = int(view.slot[i])
        self.lut = lut
        self.lut_slot = lut_slot

    def keys_in_region(self, keys: np.ndarray, region: Any) -> np.ndarray:
        return np.fromiter(
            (bool(region(int(k))) for k in keys), dtype=bool, count=len(keys)
        )


class FakeStore:
    """A store the synchronous tests never await; requests are the keys."""

    def __init__(self) -> None:
        self.id = uuid4()

    async def get_data(self, request: Any) -> Any:  # pragma: no cover
        return request


class AsyncStore:
    """An asyncio store: ``get_data(key)`` sleeps, then returns ``key * 10``.

    Parameters
    ----------
    latency : float
        Seconds per read.
    fail : Callable[[int, int], bool] | None
        ``(key, attempt) -> raise?``; attempts count from 1.
    gate : asyncio.Event | None
        When given, every read waits for it to be set before its latency,
        so a test decides when reads land instead of racing the clock.
    """

    def __init__(
        self,
        latency: float = 0.002,
        fail: Callable[[int, int], bool] | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.id = uuid4()
        self.latency = latency
        self.fail = fail
        self.gate = gate
        self.calls: list[int] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def get_data(self, request: Any) -> Any:
        key = int(request)
        self.calls.append(key)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(self.latency)
            if self.fail is not None and self.fail(key, self.calls.count(key)):
                raise OSError(f"read of {key} failed")
            return key * 10
        finally:
            self.concurrent -= 1


def desired(
    cache_id: int,
    backstop: Iterable[int] = (),
    target: Iterable[int] = (),
    store: Any = None,
    **kwargs: Any,
) -> DesiredSet:
    """A desired set: backstop keys first, then target keys; slice id = t."""
    backstop, target = list(backstop), list(target)
    keys = np.array(backstop + target, dtype=np.int64)
    cls = np.array(
        [ChunkClass.BACKSTOP] * len(backstop) + [ChunkClass.TARGET] * len(target),
        dtype=np.uint8,
    )
    slice_ids = np.array([unpack(k)[1] for k in keys], dtype=np.int32)
    return DesiredSet(
        cache_id=cache_id,
        keys=keys,
        cls=cls,
        slice_ids=slice_ids,
        build_request=lambda ks: [int(k) for k in ks],
        store=store if store is not None else FakeStore(),
        **kwargs,
    )


def coarse_keys(t: int = 0) -> list[int]:
    """Every tile of the coarsest level at *t*."""
    n = grid_dim(N_LEVELS)
    return [pack(N_LEVELS, t, gy, gx) for gy in range(n) for gx in range(n)]


def fine_keys(level: int, t: int = 0, n: int | None = None) -> list[int]:
    """Tiles of *level* at *t*, row major; the first *n* if given."""
    g = grid_dim(level)
    keys = [pack(level, t, gy, gx) for gy in range(g) for gx in range(g)]
    return keys if n is None else keys[:n]
