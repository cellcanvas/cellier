"""Shared immutable state snapshots — used by events and render layer."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The empty thickness mapping, shared because it is immutable.
NO_THICKNESS: Mapping[int, float] = MappingProxyType({})


class AxisAlignedSelectionState(NamedTuple):
    """Immutable snapshot of an axis-aligned selection.

    ``thickness`` is a per-axis world-unit **half**-thickness; an axis absent
    from it uses ``cellier.scene.dims.DEFAULT_HALF_THICKNESS``.

    **Where the slice positions went.**  This carried ``slice_indices`` -- a
    mapping of world axis to world position -- until Phase 8 (D5, deferred by
    D4.1 and D6.1).  Every consumer now takes the ``RegionSelection`` the
    controller emits instead: it says the same thing in a form that survives a
    slab, a viewport crop or an oblique plane, and it is pulled back through
    each visual's own transform rather than compared against data coordinates
    as though it were already in them.  The editable positions are still on
    ``cellier.scene.dims.AxisAlignedSelection``, which is what the sliders
    write to and what the region is built from.
    """

    displayed_axes: tuple[int, ...]
    thickness: Mapping[int, float] = NO_THICKNESS


class PlaneSelectionState(NamedTuple):
    """Stub — not yet implemented."""

    pass


SelectionState = AxisAlignedSelectionState | PlaneSelectionState


class DimsState(NamedTuple):
    """Current dimension display state for a scene."""

    axis_labels: tuple[str, ...]
    selection: SelectionState


class CameraState(NamedTuple):
    """Immutable snapshot of the active camera's logical state."""

    camera_type: Literal["perspective", "orthographic"]
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    up: tuple[float, float, float]
    fov: float
    zoom: float
    extent: tuple[float, float]
    depth_range: tuple[float, float]
