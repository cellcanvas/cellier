"""An in-memory image or labels store, served as a two-level brick pyramid.

Oblique 2D design v3, 5.4 (I-iii+, U8).  A tilted 2D view of an in-memory
visual is drawn by the multiscale brick route, whose chunk scheduler reads
bricks from a multiscale source.  :class:`InMemoryBrickSource` is that
source for an in-memory store:

- **level 1** (index 0) is the array itself, read by the store's own
  ``get_data``;
- **level 2** (index 1) is a stored coarse level, built once, block-wise, on
  a worker thread (E, G1): a ``4 x`` block mean for images, a stride at
  phase 1 for labels (never a mean, which would invent label ids).

Only the data axes section cutting would cut are downsampled (D9): every
world axis the axis maps to, and the axis itself, typed ``space``, and its
sampling ``"continuous"``.  Time and discrete axes keep every sample.  The
level transform states exactly what the reduction did: scale 4 and
translation 1.5 fine voxels for a mean, translation 1 for the stride.

Until the coarse level is built the visual plans the target only, and no
read waits for the build (U8).  A contents change marks the coarse blocks it
touches stale and recomputes them on the worker; a coarse read that overlaps
a stale block waits for that recompute, so a stale coarse brick is never
drawn after a stroke (U3).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.data._axes import level_systems
from cellier.transform import AffineTransform

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from uuid import UUID

    from cellier.data.image._image_requests import ChunkRequest

__all__ = ["COARSE_FACTOR", "InMemoryBrickSource", "downsampled_axes"]

#: The coarse level's factor on every downsampled axis (D9).
COARSE_FACTOR = 4

#: Level-0 planes per block of the coarse build (5.4.2), along the first
#: downsampled axis: small enough to keep a block's temporaries modest.
_BUILD_PLANES = 64

#: One worker for every source: builds and recomputes are short, and one
#: thread keeps two large builds from competing for memory bandwidth.
_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="cellier-coarse"
            )
        return _EXECUTOR


def downsampled_axes(transform: Any, world: Any, data: Any) -> tuple[bool, ...]:
    """Which data axes the coarse level downsamples (D9).

    A data axis is downsampled when it is typed ``space`` with
    ``"continuous"`` sampling and every world axis it maps to is typed
    ``space``: the predicate of section cutting
    (``render/_spaces.py``, ``section_planes``).  An axis that reaches no
    world axis (broadcast away) is left alone.

    Parameters
    ----------
    transform : BaseTransform
        The visual's ``data -> world`` transform.
    world : WorldCoordinateSystem
        Its output system.
    data : DataCoordinateSystem
        The store's level-0 system.

    Returns
    -------
    tuple[bool, ...]
        One flag per data axis.
    """
    coupling = np.asarray(transform.axis_coupling(), dtype=bool)
    flags = []
    for index, axis in enumerate(data.axes):
        rows = np.flatnonzero(coupling[:, index])
        flags.append(
            bool(len(rows))
            and axis.axis_type == "space"
            and axis.sampling == "continuous"
            and all(world.axes[int(row)].axis_type == "space" for row in rows)
        )
    return tuple(flags)


def _read_padded(array: np.ndarray, selections, dtype) -> np.ndarray:
    """A padded brick of *array*, as the stores' ``get_data`` reads one.

    ``int`` selections pick a plane (clamped, dropped from the output);
    ``(start, stop)`` windows are kept, clamped to the array and
    zero-padded outside it.
    """
    out_shape = [stop - start for sel in selections if isinstance(sel, tuple)
                 for start, stop in [sel]]  # fmt: skip
    out = np.zeros(out_shape, dtype=dtype)
    src: list[int | slice] = []
    dst: list[slice] = []
    for axis, sel in enumerate(selections):
        size = array.shape[axis]
        if isinstance(sel, tuple):
            start, stop = sel
            lo, hi = max(0, start), min(size, stop)
            if hi <= lo:
                return out
            src.append(slice(lo, hi))
            dst.append(slice(lo - start, hi - start))
        else:
            src.append(int(np.clip(sel, 0, size - 1)))
    out[tuple(dst)] = array[tuple(src)]
    return out


def _coarse_shape(shape: Sequence[int], downsample: Sequence[bool]) -> tuple[int, ...]:
    return tuple(
        math.ceil(n / COARSE_FACTOR) if down else int(n)
        for n, down in zip(shape, downsample, strict=True)
    )


def _block_mean(block: np.ndarray, downsample: Sequence[bool]) -> np.ndarray:
    """The ``4 x`` block mean of *block* over the downsampled axes.

    A partial block at an edge averages only the voxels that exist (D9).
    """
    total = np.asarray(block, dtype=np.float64)
    count = np.ones((1,) * block.ndim)
    for axis, down in enumerate(downsample):
        if not down:
            continue
        n = total.shape[axis]
        starts = np.arange(0, n, COARSE_FACTOR)
        total = np.add.reduceat(total, starts, axis=axis)
        sizes = np.minimum(COARSE_FACTOR, n - starts).astype(np.float64)
        shape = [1] * block.ndim
        shape[axis] = len(sizes)
        count = count * sizes.reshape(shape)
    return (total / count).astype(np.float32)


def _stride_indices(n: int) -> np.ndarray:
    """The fine voxels a phase-1 stride keeps: ``4 k + 1``, clamped at the end."""
    return np.minimum(
        np.arange(math.ceil(n / COARSE_FACTOR)) * COARSE_FACTOR + 1, n - 1
    )


def _block_stride(block: np.ndarray, downsample: Sequence[bool]) -> np.ndarray:
    """The phase-1 stride of *block* over the downsampled axes (labels)."""
    out = block
    for axis, down in enumerate(downsample):
        if down:
            out = np.take(out, _stride_indices(out.shape[axis]), axis=axis)
    return np.ascontiguousarray(out, dtype=np.int32)


class InMemoryBrickSource:
    """An in-memory store as a two-level pyramid for the chunk scheduler.

    Parameters
    ----------
    store : ImageMemoryStore or LabelMemoryStore
        The store.  Its ``data`` is read in place; its ``id`` is this
        source's, so a ``StoreChange`` of the store invalidates the bricks
        read through here (5.4.4).
    labels : bool
        Labels take the phase-1 stride; images the block mean (D9).
    downsample : Sequence[bool]
        Which data axes the coarse level shrinks (:func:`downsampled_axes`).

    Attributes
    ----------
    level_shapes : list[tuple[int, ...]]
        The array's shape, then the coarse level's.
    level_scales, level_translations : list[tuple[float, ...]]
        Level ``k`` voxel to level-0 voxel, per data axis.
    data_coordinate_systems : list[DataCoordinateSystem]
        The store's level-0 system and a coarse one.
    level_transforms : list[AffineTransform]
        The identity, then the coarse level's ``scale, translation``.
    """

    def __init__(self, store: Any, *, labels: bool, downsample: Sequence[bool]) -> None:
        self.store = store
        self.id: UUID = store.id
        self.labels = bool(labels)
        self.downsample = tuple(bool(flag) for flag in downsample)
        shape = tuple(int(n) for n in store.data.shape)
        if len(self.downsample) != len(shape):
            raise ValueError(
                f"downsample names {len(self.downsample)} axes; the store has "
                f"{len(shape)}."
            )
        self._dtype = np.int32 if self.labels else np.float32
        self.level_shapes = [shape, _coarse_shape(shape, self.downsample)]
        offset = 1.0 if self.labels else (COARSE_FACTOR - 1) / 2.0
        self.level_scales = [
            (1.0,) * len(shape),
            tuple(float(COARSE_FACTOR) if d else 1.0 for d in self.downsample),
        ]
        self.level_translations = [
            (0.0,) * len(shape),
            tuple(offset if d else 0.0 for d in self.downsample),
        ]
        level_zero = store.data_coordinate_systems[0]
        self.data_coordinate_systems = level_systems(
            level_zero, 2, f"{store.name}_bricks"
        )
        self.data_coordinate_systems[0] = level_zero
        self.level_transforms = self._transforms()
        self._coarse: np.ndarray | None = None
        self._build: Future | None = None
        # Coarse blocks being recomputed after a contents change, each with
        # the future its recompute completes.
        self._stale: list[tuple[tuple[tuple[int, int], ...], Future]] = []
        self._lock = threading.Lock()
        self._closed = False
        #: Wall time of the coarse build, and of the latest recompute.
        self.build_seconds: float | None = None
        self.recompute_seconds: float | None = None

    @property
    def n_levels(self) -> int:
        """Two: the array and its coarse level."""
        return 2

    def _transforms(self) -> list[AffineTransform]:
        systems = self.data_coordinate_systems
        ndim = systems[0].ndim
        out = []
        for level, system in enumerate(systems):
            out.append(
                AffineTransform.from_axis_map(
                    system,
                    systems[0],
                    axis_map={
                        system.axes[i].id: systems[0].axes[i].id for i in range(ndim)
                    },
                    scale={
                        system.axes[i].id: self.level_scales[level][i]
                        for i in range(ndim)
                    },
                    translation={
                        system.axes[i].id: self.level_translations[level][i]
                        for i in range(ndim)
                    },
                    name=f"level{level}_to_level0",
                )
            )
        return out

    # ── The coarse level ────────────────────────────────────────────────

    @property
    def built(self) -> bool:
        """Whether the coarse level exists (its first build has landed)."""
        return self._coarse is not None

    def start_build(self, on_built: Callable[[], None] | None = None) -> None:
        """Build the coarse level on the worker, once.

        *on_built* is called on the event loop that was running when the
        build started, once it lands; with no loop running it is not called
        (the next plan sees :attr:`built`).
        """
        if self._build is not None or self._closed:
            return
        loop = None
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_running_loop()
        self._build = _executor().submit(self._build_coarse)
        if on_built is None or loop is None:
            return

        def _done(_future: Future) -> None:
            if self._closed or loop.is_closed():
                return
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(on_built)

        self._build.add_done_callback(_done)

    async def wait_built(self) -> None:
        """Wait for the coarse level (starting its build if need be)."""
        self.start_build()
        if self._build is not None:
            await asyncio.wrap_future(self._build)

    def _reduce(self, block: np.ndarray) -> np.ndarray:
        if self.labels:
            return _block_stride(block, self.downsample)
        return _block_mean(block, self.downsample)

    def _build_coarse(self) -> None:
        import time

        start = time.perf_counter()
        data = self.store.data
        coarse = np.zeros(self.level_shapes[1], dtype=self._dtype)
        axis = next((i for i, d in enumerate(self.downsample) if d), None)
        if axis is None:
            coarse[...] = data
        else:
            n = data.shape[axis]
            for lo in range(0, n, _BUILD_PLANES):
                hi = min(n, lo + _BUILD_PLANES)
                index = [slice(None)] * data.ndim
                index[axis] = slice(lo, hi)
                out = [slice(None)] * data.ndim
                out[axis] = slice(lo // COARSE_FACTOR, -(-hi // COARSE_FACTOR))
                coarse[tuple(out)] = self._reduce(np.asarray(data[tuple(index)]))
                if self._closed:
                    return
        self._coarse = coarse
        self.build_seconds = time.perf_counter() - start

    def contents_changed(self, regions) -> None:
        """Recompute the coarse blocks a contents change touched (U3).

        Parameters
        ----------
        regions : tuple of per-axis ``(start, stop)`` or None
            Level-0 regions, as ``StoreChange.regions``; ``None`` rebuilds
            the whole coarse level.
        """
        if self._coarse is None:
            # Not built yet: the build in flight may have read old values.
            if self._build is not None and not self._build.done():
                self._build.add_done_callback(lambda _f: self.contents_changed(None))
            return
        if regions is None:
            boxes = [tuple((0, n) for n in self.level_shapes[1])]
        else:
            boxes = [self._coarse_box(region) for region in regions]
        for box in boxes:
            future = _executor().submit(self._recompute, box)
            with self._lock:
                self._stale.append((box, future))
            future.add_done_callback(lambda f, b=box: self._clear_stale(b, f))

    def _coarse_box(self, region) -> tuple[tuple[int, int], ...]:
        box = []
        for axis, (lo, hi) in enumerate(region):
            n = self.level_shapes[1][axis]
            lo, hi = math.floor(lo), math.ceil(hi)
            if self.downsample[axis]:
                lo, hi = lo // COARSE_FACTOR, -(-hi // COARSE_FACTOR)
            box.append((max(0, min(n, lo)), max(0, min(n, hi))))
        return tuple(box)

    def _recompute(self, box: tuple[tuple[int, int], ...]) -> None:
        import time

        start = time.perf_counter()
        try:
            self._recompute_box(box)
        finally:
            self.recompute_seconds = time.perf_counter() - start

    def _recompute_box(self, box: tuple[tuple[int, int], ...]) -> None:
        data = self.store.data
        fine = []
        for axis, (lo, hi) in enumerate(box):
            if self.downsample[axis]:
                fine.append(
                    slice(lo * COARSE_FACTOR, min(data.shape[axis], hi * COARSE_FACTOR))
                )
            else:
                fine.append(slice(lo, hi))
        if any(s.stop <= s.start for s in fine):
            return
        block = np.asarray(data[tuple(fine)])
        if self.labels:
            # The stride's phase is relative to the block's start, which is
            # a multiple of 4 on every downsampled axis.
            reduced = _block_stride(block, self.downsample)
        else:
            reduced = _block_mean(block, self.downsample)
        target = tuple(
            slice(lo, lo + reduced.shape[a]) for a, (lo, _) in enumerate(box)
        )
        self._coarse[target] = reduced

    def _clear_stale(self, box, future: Future) -> None:
        with self._lock:
            self._stale = [entry for entry in self._stale if entry[1] is not future]

    def _stale_for(self, selections) -> list[Future]:
        with self._lock:
            entries = list(self._stale)
        hits = []
        for box, future in entries:
            overlaps = True
            for (lo, hi), sel in zip(box, selections, strict=True):
                if isinstance(sel, tuple):
                    if sel[1] <= lo or sel[0] >= hi:
                        overlaps = False
                        break
                elif not lo <= sel < hi:
                    overlaps = False
                    break
            if overlaps:
                hits.append(future)
        return hits

    # ── Reads ───────────────────────────────────────────────────────────

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """One padded brick: level 0 from the store, level 1 from the coarse level.

        A coarse read never waits for the build (none is planned before it
        lands); it waits for the recompute of a stale block it overlaps.
        """
        if request.scale_index == 0:
            return await self.store.get_data(request)
        coarse = self._coarse
        if coarse is None:
            # Planned before the build landed (a race with the plan's
            # check): wait rather than serve zeros.
            await self.wait_built()
            coarse = self._coarse
        for future in self._stale_for(request.axis_selections):
            await asyncio.wrap_future(future)
        return _read_padded(coarse, request.axis_selections, self._dtype)

    def close(self) -> None:
        """Stop a build in progress and drop the coarse level."""
        self._closed = True
        self._coarse = None
