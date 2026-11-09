"""In-memory geometry on the chunk scheduler: one whole result per visual.

The points, lines and graph visuals load their whole slice (one filtered
result) through a single-level
:class:`~cellier.render._level_residency.LevelResidency`:

- the request key is everything the read depends on except the store's
  contents (:func:`freeze_request`); an unchanged request reads nothing;
- at most two reads are in flight per visual, the newest plan wins, and the
  filter runs in the loop's executor (``MEMORY_GEOMETRY_POLICY``);
- the previous result stays on screen while the next one loads when the two
  keys place their data the same way (``keeps_previous``);
- a contents change keeps the old result drawn until its re-read lands.

An upload is placed by its own request key, never by state the newest plan
set: the key carries the world-frame flag, the collapsed data positions and
the fold axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.render._level_residency import MEMORY_GEOMETRY_POLICY, LevelResidency
from cellier.render._spaces import element_fold
from cellier.render.scheduling import PlanMode

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cellier.render._requests import ReslicingRequest
    from cellier.render._scene_config import VisualRenderConfig
    from cellier.render.scheduling import DesiredSet

#: Request fields no read depends on: the ids are fresh on every plan, the
#: level is always 0, and the read takes the retained axes, not the
#: displayed ones.
_NOT_IN_KEY = frozenset(
    {"slice_request_id", "chunk_request_id", "scale_index", "displayed_axes"}
)

# Positions in a geometry request key.
_RETAINED = 0
_FOLD_AXES = 2
_WORLD_FRAME = 3
_DATA_POSITIONS = 4


def freeze(value: Any) -> Hashable:
    """A hashable stand-in for *value*: dicts, sequences and arrays included."""
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    return value


def freeze_request(request: Any) -> tuple:
    """Every field of a geometry request a read depends on, hashable.

    The request is a ``NamedTuple``; the ids, the level and the displayed
    axes are left out (:data:`_NOT_IN_KEY`).  A graph request's trail dicts
    and arrays are frozen by value.
    """
    return tuple(
        (name, freeze(value))
        for name, value in request._asdict().items()
        if name not in _NOT_IN_KEY
    )


def same_placement(held: Hashable, planned: Hashable) -> bool:
    """Whether a held result sits where the planned one will (``keeps_previous``).

    The same retained axes, fold axes and world-frame flag: a slider move
    or a trail change then leaves the old result in the right frame, only
    behind.
    """
    return (
        held[_RETAINED] == planned[_RETAINED]
        and held[_FOLD_AXES] == planned[_FOLD_AXES]
        and held[_WORLD_FRAME] == planned[_WORLD_FRAME]
    )


class WholeItemGeometryMixin:
    """``plan`` / ``residencies`` for an in-memory geometry visual.

    The host provides ``_build_request(dims_state, selection)`` (which sets
    ``_request_world_frame``, ``_last_data_positions`` and ``_fold``),
    ``_update_node_matrix``, ``_commit(data, is_2d)``, ``_last_displayed_axes``,
    ``_spaces``, ``_transform``, ``node`` and :meth:`_show_empty`.  It calls
    :meth:`_init_whole_item` from its constructor.
    """

    #: Loads through the chunk scheduler (``is_chunked_visual``).
    chunked: bool = True

    def _init_whole_item(self) -> None:
        self._residency = LevelResidency(
            1,
            upload=self.upload,
            release=self._release_whole,
            keeps_previous=same_placement,
            policy=MEMORY_GEOMETRY_POLICY,
            keeps_through_invalidation=True,
        )

    # -- planning ------------------------------------------------------------

    def request_key(self, request: Any) -> tuple:
        """Everything the read of *request* depends on except the store.

        ``(retained axes, frozen request, fold axes, world frame, data
        positions)``: the last two as the plan that built *request* left
        them, so an upload can be placed by its own key.
        """
        return (
            tuple(request.retained_axes),
            freeze_request(request),
            tuple(request.fold_axes),
            bool(self._request_world_frame),
            tuple(sorted(self._last_data_positions.items())),
        )

    def plan(
        self,
        request: ReslicingRequest,
        config: VisualRenderConfig | None = None,
        mode: PlanMode = PlanMode.FULL,
    ) -> list[DesiredSet]:
        """One desired set: the whole filtered slice, whatever the mode.

        A whole-item cache has no backstop, so a ``BACKSTOP_ONLY`` plan (a
        dims drag) still plans the one result.
        """
        displayed = request.dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        slice_request = self._build_request(request.dims_state, request.selection)
        key = self.request_key(slice_request)
        return [self._residency.desired({0: key}, {0}, {0: slice_request})]

    def residencies(self) -> dict[int, LevelResidency]:
        """``cache_id -> adapter``: one cache per visual."""
        return {self._residency.cache_id: self._residency}

    @property
    def awaits_data(self) -> bool:
        """Whether the visual draws nothing only because its read has not landed.

        True only before the first result, and after a placement change
        (``keeps_previous`` keeps everything else on screen).
        """
        return self._residency.awaiting and bool(self.node.visible)

    # -- the residency's callbacks -------------------------------------------

    def upload(self, level: int, request_key: tuple, data: Any) -> None:
        """Put a read's result on the GPU, placed by its own request key.

        Called by the residency from ``rebuild_draw``; a test may call it
        directly with ``request_key(request)``.
        """
        world_frame = bool(request_key[_WORLD_FRAME])
        self._request_world_frame = world_frame
        self._last_data_positions = dict(request_key[_DATA_POSITIONS])
        if world_frame or not request_key[_FOLD_AXES] or self._spaces is None:
            self._fold = None
        else:
            self._fold = element_fold(self._spaces, self._transform)
        self._commit(data, is_2d=len(request_key[_RETAINED]) == 2)

    def _release_whole(self, level: int) -> None:
        """The residency holds nothing: draw nothing."""
        self._show_empty()

    def _show_empty(self) -> None:  # pragma: no cover - every host overrides it
        raise NotImplementedError
