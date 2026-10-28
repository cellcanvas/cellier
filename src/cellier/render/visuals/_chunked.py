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
from cellier.render._backstop import (
    backstop_cap,
    backstop_level,
    intermediate_bricks,
    intermediate_cap,
)
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


#: Half-depth of the 2D slice node's culling prism, in rendered units: far
#: beyond any slab, so the 2D camera's own near and far planes never cull
#: (oblique 2D design v3, 5.3.4, C2).
SLICE_PRISM_DEPTH = 1.0e7


def viewport_prism(request: ReslicingRequest) -> np.ndarray:
    """A 2D request's viewport as frustum corners unbounded in depth.

    The 2D slice node is planned by the 3D brick planner (design v3,
    5.3.4), which culls with frustum corners.  A 2D request carries none,
    and the 2D camera's near and far planes must not cull: they would drop
    a slab's bricks to the backstop (Phase 0c P-b).  So the corners are the
    viewport in ``(x, y)`` and +-``SLICE_PRISM_DEPTH`` in depth, in pygfx's
    order: ``[near, far] x [(l, b), (r, b), (r, t), (l, t)]``, the near
    side at ``+z`` (the camera looks down ``-z``).
    """
    (x0, y0), (x1, y1) = viewport_2d(request)
    corners = np.empty((2, 4, 3), dtype=np.float64)
    for side, depth in enumerate((SLICE_PRISM_DEPTH, -SLICE_PRISM_DEPTH)):
        corners[side] = [
            (x0, y0, depth),
            (x1, y0, depth),
            (x1, y1, depth),
            (x0, y1, depth),
        ]
    return corners


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


def _current_slice_ids(
    planner: MultiscaleRegionPlanner,
    residency: ImageResidency2D,
    fill: dict[int, int] | None,
) -> np.ndarray | None:
    """The slice id of every level at the plan's dims position, not interned.

    What :attr:`ImageResidency2D.current_slice_ids` holds.  Every level and
    not only the planned ones: an earlier view may have loaded any of them.
    Looked up and never interned, so a level no plan has drawn at this
    position (-1) does not spend the atlas's table of selections.  ``None``
    for a plan with no region, which has no selection to look up.
    """
    n_levels = len(residency.brick_rule()[1])
    if planner._level_box(0) is None:
        return None
    ids = np.full(n_levels, -1, dtype=np.int64)
    for level_index in range(n_levels):
        if planner._level_box(level_index) is not None:
            ids[level_index] = residency.find(
                planner._level_slice_selection(level_index, fill)
            )
    return ids


def target_room(
    residency: ImageResidency3D | ImageResidency2D | None,
    backstop_arr: np.ndarray | None,
    loading: ProgressiveLoadingConfig | None,
) -> int | None:
    """How many target bricks an atlas keeps beside its backstop and stages.

    The room :func:`desired_bricks` truncates the target to, for a planner
    that would rather plan less than be truncated (an orthographic plane
    plan, plane rendering design 6.3).

    It is asked before the target exists, and the intermediate stages are
    derived from the target, so their count is an estimate: per stage, the
    room divided by ``4 ** offset``, the share a plane's stage is of its
    target where level ratios are 2, in 3D as well (intermediate-level
    design v2, section 6); summed, and limited by the stages' cap.  Where
    the estimate is low, :func:`desired_bricks` drops the farthest target
    bricks, which the stage still covers.

    Parameters
    ----------
    residency : ImageResidency3D or ImageResidency2D or None
        The atlas.
    backstop_arr : np.ndarray or None
        The planned backstop.
    loading : ProgressiveLoadingConfig or None
        For the backstop cap, and the stages and their cap.

    Returns
    -------
    int or None
        ``None`` without an atlas.
    """
    if residency is None:
        return None
    budget = max(0, int(residency.n_slots) - 1)
    n_backstop = 0 if backstop_arr is None else len(backstop_arr)
    cap = max(0, int(backstop_cap_for(loading, residency)))
    room = budget - min(n_backstop, cap, budget)
    if loading is None or not loading.intermediate_offsets:
        return room
    estimate = sum(room // 4**offset for offset in loading.intermediate_offsets)
    return room - min(estimate, intermediate_cap(loading, residency.n_slots), room)


def desired_bricks(
    planner: MultiscaleRegionPlanner,
    residency: ImageResidency3D | ImageResidency2D,
    target_arr: np.ndarray | None,
    *,
    backstop_arr: np.ndarray | None = None,
    backstop_cap: int = 0,
    fill: dict[int, int] | None = None,
    loading: ProgressiveLoadingConfig | None = None,
) -> DesiredSet:
    """Turn planned brick or tile arrays into the atlas's desired set (5.3).

    The backstop block comes first, then each intermediate stage, then the
    target (intermediate-level design v2, section 6):

    1. the backstop is truncated to *backstop_cap* (nearest first, as it is
       ordered) and to the budget;
    2. each stage of ``loading.intermediate_offsets``, coarse to fine, is the
       ancestors of the whole target array (before it is truncated), less
       the keys already taken, nearest first.  One cap
       (``intermediate_max_slot_fraction``) holds for all the stages and is
       filled coarsest stage first; a stage that outgrows it keeps its
       nearest bricks.  A stage brick's class is its offset;
    3. a target key equal to a kept backstop or stage key keeps the higher
       class, so it appears once;
    4. the target is truncated to the room the backstop and stage bricks
       actually kept leave in :func:`brick_budget`.

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
        (``PlanMode.BACKSTOP_ONLY``), or no geometry to plan a backstop on.
    backstop_cap : int
        At most this many backstop keys (``backstop_max_slot_fraction``).
    fill : dict[int, int] or None
        Per-axis selection overrides (a composite channel's index).
    loading : ProgressiveLoadingConfig or None
        For the intermediate stages and their cap.  ``None``, or no offset,
        plans no stage.

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

    blocks = [(b_keys, b_sids, int(ChunkClass.BACKSTOP))]
    taken = b_keys
    truncated_intermediate = 0
    offsets = () if loading is None else loading.intermediate_offsets
    if offsets and target_arr is not None and len(target_arr):
        grid, spans, counts = residency.brick_rule()
        # From the config and not the backstop array: an empty backstop must
        # not let a stage be planned at the backstop's level.
        top_level = backstop_level(loading, len(spans))
        stage_room = min(
            intermediate_cap(loading, residency.n_slots), budget - n_backstop
        )
        for offset in offsets:
            stage_arr = intermediate_bricks(
                target_arr, offset, top_level, grid, spans, counts
            )
            s_keys, s_sids = _pack(planner, residency, stage_arr, fill)
            new = ~np.isin(s_keys, taken)
            s_keys, s_sids = s_keys[new], s_sids[new]
            truncated_intermediate += max(0, len(s_keys) - stage_room)
            s_keys, s_sids = s_keys[:stage_room], s_sids[:stage_room]
            stage_room -= len(s_keys)
            blocks.append((s_keys, s_sids, int(offset)))
            taken = np.concatenate([taken, s_keys])

    if len(taken):
        target_only = ~np.isin(t_keys, taken)
        t_keys, t_sids = t_keys[target_only], t_sids[target_only]
    room = budget - len(taken)
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
    blocks.append((t_keys, t_sids, int(ChunkClass.TARGET)))

    if isinstance(residency, ImageResidency2D):
        # After every _pack: the planned levels' selections are interned.
        residency.current_slice_ids = _current_slice_ids(planner, residency, fill)

    return DesiredSet(
        cache_id=residency.cache_id,
        keys=np.concatenate([keys for keys, _, _ in blocks]),
        cls=np.concatenate(
            [np.full(len(keys), cls, dtype=np.uint8) for keys, _, cls in blocks]
        ),
        slice_ids=np.concatenate([sids for _, sids, _ in blocks]),
        build_request=residency.build_request,
        store=None,
        n_truncated_target=int(truncated_target),
        n_truncated_backstop=int(truncated_backstop),
        n_truncated_intermediate=int(truncated_intermediate),
        intermediate_offsets=tuple(int(offset) for offset in offsets),
    )


def backstop_cap_for(
    loading: ProgressiveLoadingConfig | None,
    residency: ImageResidency3D | ImageResidency2D,
) -> int:
    """The backstop's slot cap on *residency*'s atlas; 0 with no *loading*."""
    if loading is None:
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


def log_intermediate_cap_once(owner: object, desired: list[DesiredSet]) -> None:
    """Log, once per visual, that the stages' cap truncated a stage.

    The counterpart of :func:`log_backstop_cap_once` for
    ``intermediate_max_slot_fraction``.

    Parameters
    ----------
    owner : object
        The visual; it remembers that it has logged.
    desired : list[DesiredSet]
        The plan just made.
    """
    dropped = max((ds.n_truncated_intermediate for ds in desired), default=0)
    if not dropped or getattr(owner, "_intermediate_cap_logged", False):
        return
    owner._intermediate_cap_logged = True
    kept = max(
        int(((ds.cls != ChunkClass.BACKSTOP) & (ds.cls != ChunkClass.TARGET)).sum())
        for ds in desired
    )
    _CACHE_LOGGER.info(
        "intermediate_capped  visual=%s  kept=%d  dropped=%d  (the intermediate "
        "stages outgrew intermediate_max_slot_fraction; use a larger offset or "
        "fewer stages, or raise the fraction)",
        getattr(owner, "visual_model_id", owner),
        kept,
        dropped,
    )
