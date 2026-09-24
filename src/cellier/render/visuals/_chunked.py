"""Planner tails shared by the chunked multiscale visuals (design v3, 5.3).

Image and labels keep their own LOD selection, sort and cull, and change only
their tail: the planned brick array becomes a :class:`DesiredSet` of packed
keys for the scheduler, instead of staged slots and store requests.  A
planner never touches the atlas or the LUT (invariant 10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellier.logging import _CACHE_LOGGER, _PERF_LOGGER
from cellier.render._backstop import backstop_cap
from cellier.render.block_cache._image_residency import (
    ImageResidency2D,
    ImageResidency3D,
    pack_keys,
)
from cellier.render.scheduling import ChunkClass, DesiredSet

if TYPE_CHECKING:
    from collections.abc import Callable

    from cellier.render._requests import ReslicingRequest
    from cellier.render.block_cache._block_cache import BlockCache3D
    from cellier.render.block_cache._block_cache_2d import BlockCache2D
    from cellier.render.lut_indirection import LutIndirectionManager3D
    from cellier.render.lut_indirection._lut_indirection_manager_2d import (
        LutIndirectionManager2D,
    )
    from cellier.render.visuals._image import MultiscaleRegionPlanner
    from cellier.visuals._loading import ProgressiveLoadingConfig


def brick_budget(block_cache: BlockCache3D | BlockCache2D) -> int:
    """Bricks (or tiles) a plan may want: the scheduler's slots minus one (5.3).

    The atlas's slot 0 is reserved, so the scheduler has ``n_slots - 1``
    slots, and a plan wants at most one fewer: while a wanted brick is being
    committed a free or unwanted slot always exists.
    """
    return int(block_cache.info.n_slots) - 2


def make_residency_3d(
    block_cache: BlockCache3D,
    lut_manager: LutIndirectionManager3D,
    block_size: int,
    level_transforms: list,
    ndim: int,
    brick_max: Callable[[np.ndarray], float],
    on_write: Callable[[], None] | None,
) -> ImageResidency3D:
    """An adapter over *block_cache* and *lut_manager*, with a fresh cache id."""
    from cellier.render.visuals._image import _level_scale_and_translation

    scales, translations = _level_scale_and_translation(
        level_transforms, tuple(range(ndim))
    )
    return ImageResidency3D(
        block_cache,
        lut_manager,
        block_size,
        np.asarray(scales),
        np.asarray(translations),
        brick_max=brick_max,
        on_write=on_write,
    )


def make_residency_2d(
    block_cache: BlockCache2D,
    lut_manager: LutIndirectionManager2D,
    block_size: int,
    level_transforms: list,
    ndim: int,
    on_write: Callable[[], None] | None,
) -> ImageResidency2D:
    """A 2D adapter over *block_cache* and *lut_manager*, with a fresh cache id."""
    from cellier.render.visuals._image import _level_scale_and_translation

    scales, translations = _level_scale_and_translation(
        level_transforms, tuple(range(ndim))
    )
    return ImageResidency2D(
        block_cache,
        lut_manager,
        block_size,
        np.asarray(scales),
        np.asarray(translations),
        on_write=on_write,
    )


def viewport_2d(request: ReslicingRequest) -> tuple[np.ndarray, np.ndarray]:
    """A 2D request's viewport, ``(view_min, view_max)`` in world ``(x, y)``."""
    world_width, world_height = request.world_extent
    cx = float(request.camera_pos[0])
    cy = float(request.camera_pos[1])
    half = np.array([world_width / 2.0, world_height / 2.0])
    center = np.array([cx, cy])
    return center - half, center + half


def _pack(
    planner: MultiscaleRegionPlanner,
    residency: ImageResidency3D | ImageResidency2D,
    arr: np.ndarray | None,
    fill: dict[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(keys, slice_ids)`` for a brick array, interning per-level slices."""
    n_cols = 1 + (3 if isinstance(residency, ImageResidency3D) else 2)
    arr = np.asarray(
        np.empty((0, n_cols)) if arr is None else arr, dtype=np.int64
    ).reshape(-1, n_cols)
    levels = arr[:, 0]
    slice_ids = np.zeros(len(levels), dtype=np.int32)
    for level in np.unique(levels).tolist():
        sid = residency.intern(planner._level_slice_selection(level - 1, fill))
        slice_ids[levels == level] = sid
    return pack_keys(levels, slice_ids, arr[:, 1:]), slice_ids


def desired_bricks(
    planner: MultiscaleRegionPlanner,
    residency: ImageResidency3D | ImageResidency2D,
    target_arr: np.ndarray | None,
    *,
    backstop_arr: np.ndarray | None = None,
    backstop_cap: int = 0,
    fill: dict[int, int] | None = None,
) -> DesiredSet:
    """Turn planned brick or tile arrays into the atlas's desired set (5.3).

    The backstop block comes first, then the target:

    1. the backstop is truncated to *backstop_cap* (nearest first, as it is
       ordered) and to the budget;
    2. a target key equal to a kept backstop key keeps the backstop class,
       so it appears once;
    3. the target is truncated to the room left in :func:`brick_budget`.

    Parameters
    ----------
    planner : MultiscaleRegionPlanner
        The visual (or slot) that planned, with its region already adopted
        (``_begin_region_planning``), so each level's collapsed selection is
        known.
    residency : ImageResidency3D or ImageResidency2D
        The atlas the bricks are for; it interns the per-level selections.
    target_arr, backstop_arr : np.ndarray or None
        ``(N, 4)`` rows ``[level, g0, g1, g2]`` (3D) or ``(N, 3)`` rows
        ``[level, g0, g1]`` (2D), each in load order (nearest the camera or
        canvas centre first).  ``None`` is empty: no target
        (``PlanMode.BACKSTOP_ONLY``), or the backstop is off.
    backstop_cap : int
        At most this many backstop keys (``backstop_max_slot_fraction``).
    fill : dict[int, int] or None
        Per-axis selection overrides (a composite channel's index).

    Returns
    -------
    DesiredSet
        With ``store=None``: the coordinator attaches the visual's store.
    """
    budget = max(0, int(residency.n_slots) - 1)
    b_keys, b_sids = _pack(planner, residency, backstop_arr, fill)
    t_keys, t_sids = _pack(planner, residency, target_arr, fill)

    n_backstop = min(len(b_keys), max(0, int(backstop_cap)), budget)
    truncated_backstop = len(b_keys) - n_backstop
    b_keys, b_sids = b_keys[:n_backstop], b_sids[:n_backstop]

    if n_backstop:
        target_only = ~np.isin(t_keys, b_keys)
        t_keys, t_sids = t_keys[target_only], t_sids[target_only]
    room = budget - n_backstop
    truncated_target = max(0, len(t_keys) - room)
    if truncated_target:
        _PERF_LOGGER.warning(
            "budget_exceeded  required=%d  budget=%d  dropped=%d  "
            "(consider larger cache or tighter LOD thresholds)",
            len(t_keys),
            room,
            truncated_target,
        )
        t_keys, t_sids = t_keys[:room], t_sids[:room]

    return DesiredSet(
        cache_id=residency.cache_id,
        keys=np.concatenate([b_keys, t_keys]),
        cls=np.concatenate(
            [
                np.full(len(b_keys), int(ChunkClass.BACKSTOP), dtype=np.uint8),
                np.full(len(t_keys), int(ChunkClass.TARGET), dtype=np.uint8),
            ]
        ),
        slice_ids=np.concatenate([b_sids, t_sids]),
        build_request=residency.build_request,
        store=None,
        n_truncated_target=int(truncated_target),
        n_truncated_backstop=int(truncated_backstop),
    )


def backstop_cap_for(
    loading: ProgressiveLoadingConfig | None,
    residency: ImageResidency3D | ImageResidency2D,
) -> int:
    """The backstop's slot cap on *residency*'s atlas; 0 when it is off."""
    if loading is None or not loading.backstop:
        return 0
    return backstop_cap(loading, residency.n_slots)


def log_backstop_cap_once(owner: object, desired: list[DesiredSet]) -> None:
    """Log, once per visual, that the backstop cap truncated its backstop (5.9).

    Parameters
    ----------
    owner : object
        The visual; it remembers that it has logged.
    desired : list[DesiredSet]
        The plan just made.
    """
    dropped = max((ds.n_truncated_backstop for ds in desired), default=0)
    if not dropped or getattr(owner, "_backstop_cap_logged", False):
        return
    owner._backstop_cap_logged = True
    kept = max(int((ds.cls == ChunkClass.BACKSTOP).sum()) for ds in desired)
    _CACHE_LOGGER.info(
        "backstop_capped  visual=%s  kept=%d  dropped=%d  (the backstop "
        "outgrew backstop_max_slot_fraction; set backstop_extent='view' or a "
        "coarser backstop_level, or raise the fraction)",
        getattr(owner, "visual_model_id", owner),
        kept,
        dropped,
    )
