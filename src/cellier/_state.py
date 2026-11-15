"""Shared immutable state snapshots — used by events and render layer."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The empty thickness mapping, shared because it is immutable.
NO_THICKNESS: Mapping[int, float] = MappingProxyType({})


class AxisAlignedSelectionState(NamedTuple):
    """Immutable snapshot of a scene's displayed axes and slab thickness.

    **There are no slice positions here.**  A ``DimsChangedEvent`` carries
    them in its own ``slice_indices`` field, and the editable positions are
    on the scene's ``dims.selection`` (an ``AxisAlignedSelection``).

    Attributes
    ----------
    displayed_axes : tuple[int, ...]
        The world axes the view shows, in display order.
    thickness : Mapping[int, float]
        Per world axis, the half-thickness of the slab around the slice, in
        world units.  An axis absent from it has thickness 0, a plane.
    """

    displayed_axes: tuple[int, ...]
    thickness: Mapping[int, float] = NO_THICKNESS


class DimsState(NamedTuple):
    """A scene's dims: axis names and which axes are shown where.

    Attributes
    ----------
    axis_labels : tuple[str, ...]
        The name of each world axis, in world order.
    selection : AxisAlignedSelectionState
        The displayed axes and the per-axis slab thickness.  The slice
        positions are not here: see ``DimsChangedEvent.slice_indices``.
    """

    axis_labels: tuple[str, ...]
    selection: AxisAlignedSelectionState


class OrbitControllerState(NamedTuple):
    """The state of a 3D camera's orbit controller, in a `CameraState`.

    It holds nothing yet: the point the controller turns the camera about
    follows from the camera's ``fov`` and ``extent`` (see
    ``cellier.scene.orbit_distance``).

    Attributes
    ----------
    controller_type : Literal["orbit"]
        Discriminator.  Always ``"orbit"``.
    """

    controller_type: Literal["orbit"] = "orbit"


class PanZoomControllerState(NamedTuple):
    """The state of a 2D camera's pan/zoom controller, in a `CameraState`.

    It holds nothing yet: the 2D camera's ``extent`` and ``position`` say
    everything the controller works from.

    Attributes
    ----------
    controller_type : Literal["pan_zoom"]
        Discriminator.  Always ``"pan_zoom"``.
    """

    controller_type: Literal["pan_zoom"] = "pan_zoom"


class CameraState(NamedTuple):
    """A snapshot of a canvas's active camera.

    Positions and directions are in the renderer's ``(x, y, z)`` frame: x is
    the last displayed world axis, so a scene showing ``("z", "y", "x")``
    has renderer x = world x and renderer z = world z.  Pass it back to
    ``CellierController.set_camera_state`` to restore a view.

    Attributes
    ----------
    camera_type : str
        ``"perspective"`` for a 3D view, ``"orthographic"`` for a 2D one.
    position : tuple[float, float, float]
        Camera position.
    rotation : tuple[float, float, float, float]
        Orientation as a quaternion ``(x, y, z, w)``.
    up : tuple[float, float, float]
        The camera's up vector; always ``(0, 1, 0)`` for a 2D camera.
    fov : float
        Vertical field of view in degrees; ``0`` for an orthographic camera.
    extent : tuple[float, float]
        The ``(width, height)`` of the view in world units.  For a
        perspective camera it is the size at the point the orbit controller
        turns the camera about, which is ``cellier.scene.orbit_distance``
        ahead of it, so a zoom changes it and restoring it restores that
        point; at a field of view of 0 it is the size of the whole view.
        A component of 0 or less, as in states captured before 3D extents
        were recorded, keeps the camera's current width and height.
    depth_range : tuple[float, float] or None
        The ``(near, far)`` clipping distances, or ``None`` when the range
        is automatic (worked out from what the camera is fitted to).
    controller_state : OrbitControllerState or PanZoomControllerState or None
        The state of the camera's mouse controller, told apart by its
        ``controller_type``: an `OrbitControllerState` for a 3D camera, a
        `PanZoomControllerState` for a 2D one.  A captured state always
        has one; applying one of the other kind raises ``ValueError``.
        ``None`` (a state made by hand, or saved before the field existed)
        leaves the controller as it is.  Settings such as
        whether the controller takes input are not part of it: they are
        on the camera model's ``controller``.
    """

    camera_type: Literal["perspective", "orthographic"]
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    up: tuple[float, float, float]
    fov: float
    extent: tuple[float, float]
    # ``None`` is an automatic range: pygfx works it out from what the
    # camera is fitted to.  A 2D camera has one until a range is set.
    depth_range: tuple[float, float] | None
    # Last, and with a default, so positional construction keeps working.
    controller_state: OrbitControllerState | PanZoomControllerState | None = None
