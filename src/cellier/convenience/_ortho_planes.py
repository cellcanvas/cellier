"""Slice planes in the 3D panel of an OrthoViewer.

See ``plans/plane_rendering_design_v3.md`` section 10.2.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

import numpy as np
from psygnal import Signal

from cellier.visuals import PlaneOutline, RenderPlane

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cellier.controller import CellierController
    from cellier.events import (
        DimsChangedEvent,
        DimsInteractionEvent,
        RenderPlanesChangedEvent,
        VisualRemovedEvent,
    )
    from cellier.scene.scene import Scene

OrthoPlaneMode = Literal["slices"] | None
"""How the 3D panel's render planes are tied to the 2D panels."""

#: The 2D panels in the order of the spatial axis each one slices.
_SLICE_PANELS = ("xy", "xz", "yz")
#: How far a slice plane's normal may be from its axis (1 - |cosine|).
_NORMAL_TOL = 1e-9

SLICES_REASON = (
    "The planes follow the 2D views' sliders "
    '(plane_controller.mode is "slices"). Move a slider to move a plane.'
)
"""Why a render planes control is read-only in ``"slices"`` mode."""


def _three_outlines(
    outlines: PlaneOutline | Sequence[PlaneOutline] | None,
) -> tuple[PlaneOutline, PlaneOutline, PlaneOutline]:
    """One outline per slice plane, from one for all, three, or none."""
    if outlines is None:
        return (PlaneOutline(), PlaneOutline(), PlaneOutline())
    if isinstance(outlines, PlaneOutline):
        return (outlines, outlines, outlines)
    given = tuple(outlines)
    if len(given) != 3 or not all(isinstance(o, PlaneOutline) for o in given):
        raise ValueError(
            "The slice planes' outlines are one PlaneOutline, for all three "
            "planes, or three: one each for the xy, xz and yz panels' planes."
        )
    return given


def _check_mode(mode: object) -> None:
    if mode not in ("slices", None):
        raise ValueError(f'Unknown plane mode {mode!r}. Expected "slices" or None.')


class OrthoPlaneController:
    """Tie the 3D panel's render planes to the 2D panels' slices.

    ``mode`` says how:

    ============ ========================================================
    ``None``     No link (the default).  The 3D panel's visuals keep
                 whatever render planes they are given.
    ``"slices"`` Every image and labels visual of the 3D panel carries
                 exactly three render planes, one per 2D panel, at that
                 panel's slice position and facing its sliced axis.
                 Moving a 2D slider moves the matching plane.
    ============ ========================================================

    In a visual in the ``"plane"`` render mode the three planes show, in
    3D, the three slices the 2D panels show.  A visual in another render
    mode carries the planes too and draws them as soon as it is put in
    plane mode.  The mode sets no render mode itself.

    This holds **no plane state**.  Each visual's ``render_planes`` stays
    the source of truth and is what is serialized; in ``"slices"`` mode
    this keeps the planes of the 3D panel's visuals and the 2D sliders in
    step, through the controller's public API:

    - **A slider moves** (or anything else changes a slice position): each
      plane is moved along its normal to its panel's position, on every
      visual.
    - **A plane is changed on one visual** (from code: there is no gizmo
      on a slice plane, and the render planes control shows them
      read-only): the change is carried over.  A plane moved along its
      normal moves its panel's slider, on every panel when the panels'
      axes are linked.  Its extents, its ``enabled`` flag, its spin and
      its outline are copied to the other visuals.  A tuple that is not the three slice
      planes (a plane tilted off its axis, added, removed or replaced)
      cannot be carried over: the visual is given the slice planes back,
      with a warning.

    A 2D panel's **scrub** is forwarded: while a slider is dragged this
    holds a plane interaction scope on each visual, so a multiscale visual
    keeps the bricks it has while the planes move and loads once when the
    slider is released.

    Turning the mode on replaces the visuals' planes with the three slice
    planes.  Turning it off leaves them, free to edit.

    Parameters
    ----------
    controller : CellierController
        The controller owning the scenes and the visuals.
    scenes : Mapping[str, Scene]
        The panel scenes keyed ``"xy"``, ``"xz"``, ``"yz"``, ``"vol"``.
    spatial_axes : Sequence[int]
        The three world axes the 3D panel displays, in order.  The ``xy``
        panel slices the first, ``xz`` the second and ``yz`` the third.
    dims_controller : Callable[[], OrthoDimsController | None] or None
        Returns the controller mirroring the panels' dims while their axes
        are linked, else ``None``.  A plane moved along its normal then
        moves its slider on every panel; with no link it moves the 2D panel
        that slices that axis, and the 3D panel.
    mode : {"slices"} or None
        The starting mode.
    outlines : PlaneOutline, three of them, or None
        The outlines of the slice planes (:attr:`outlines`).  ``None`` is
        no outline.

    Attributes
    ----------
    mode_changed : psygnal.Signal
        Emitted with the new mode after :attr:`mode` changed.
    """

    mode_changed = Signal(object)

    def __init__(
        self,
        controller: CellierController,
        scenes: Mapping[str, Scene],
        spatial_axes: Sequence[int],
        dims_controller: Callable[[], Any] | None = None,
        mode: OrthoPlaneMode = None,
        outlines: PlaneOutline | Sequence[PlaneOutline] | None = None,
    ) -> None:
        _check_mode(mode)
        self._outlines = _three_outlines(outlines)
        self._id: UUID = uuid4()
        self._controller = controller
        self._scenes = dict(scenes)
        self._axes = tuple(int(axis) for axis in spatial_axes)
        if len(self._axes) != 3:
            raise ValueError("The 3D panel displays three world axes.")
        self._dims_controller = dims_controller
        self._mode: OrthoPlaneMode = mode
        #: The 3D panel's visuals that have render planes, in the order added.
        self._visuals: list[UUID] = []
        #: The ids of the three slice planes, by sliced axis.  One triple
        #: for every visual, so their tuples can be equal.
        self._plane_ids = (uuid4(), uuid4(), uuid4())
        self._syncing = False
        self._warned = False
        # One id per visual: the owner of its subscriptions.
        self._owners: dict[UUID, UUID] = {}
        # The 2D panels being scrubbed, and the scope source of each.
        self._scope_ids: dict[UUID, UUID] = {}
        self._scrubbing: set[UUID] = set()
        for key in _SLICE_PANELS:
            scene = self._scenes[key]
            self._scope_ids[scene.id] = uuid4()
            controller.on_dims_changed(scene.id, self._on_dims, owner_id=self._id)
            controller.on_dims_interaction(
                scene.id, self._on_dims_interaction, owner_id=self._id
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def id(self) -> UUID:
        """The source id stamped on the planes this writes."""
        return self._id

    @property
    def mode(self) -> OrthoPlaneMode:
        """How the 3D panel's render planes are tied to the 2D panels.

        Setting ``"slices"`` replaces the render planes of every image and
        labels visual of the 3D panel with the three slice planes,
        unbounded, at the panels' current positions.  Setting ``None``
        writes nothing: the visuals keep the three planes, free to edit.
        ``mode_changed`` is emitted once, after the writes.  Setting the
        current mode does nothing.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: OrthoPlaneMode) -> None:
        _check_mode(mode)
        if mode == self._mode:
            return
        self._end_scrubs()
        self._mode = mode
        if mode == "slices":
            planes = self.slice_planes()
            for visual_id in self._visuals:
                self._write(visual_id, planes)
        self.mode_changed.emit(mode)

    @property
    def outlines(self) -> tuple[PlaneOutline, PlaneOutline, PlaneOutline]:
        """The outlines of the slice planes of the ``xy``, ``xz`` and ``yz`` panels.

        A line around each slice where it crosses the data, in the 3D
        panel.  Set one ``PlaneOutline`` for all three planes, or three, or
        ``None`` for no outline.  In ``"slices"`` mode the planes of every
        visual are given them at once; in mode ``None`` they are what the
        slice planes are built with when the mode is next turned on.

        The render planes control shows the slice planes read-only, so
        this is where their outlines are set.  An outline assigned to a
        slice plane of one visual from code is copied to the others, as a
        plane's extents are, and is then what this returns.
        """
        return self._outlines

    @outlines.setter
    def outlines(self, outlines: PlaneOutline | Sequence[PlaneOutline] | None) -> None:
        self._outlines = _three_outlines(outlines)
        if self._mode != "slices" or not self._visuals:
            return
        planes = tuple(
            plane.model_copy(update={"outline": outline})
            for plane, outline in zip(
                self._current_planes(), self._outlines, strict=True
            )
        )
        for visual_id in self._visuals:
            self._write(visual_id, planes)

    @property
    def visual_ids(self) -> tuple[UUID, ...]:
        """The 3D panel's visuals whose planes the mode writes."""
        return tuple(self._visuals)

    @property
    def plane_ids(self) -> tuple[UUID, UUID, UUID]:
        """The ids of the slice planes of the ``xy``, ``xz`` and ``yz`` panels."""
        return self._plane_ids

    def blocked_reason(self) -> str:
        """Why the slice planes cannot be edited by hand now, or ``""``."""
        return SLICES_REASON if self._mode == "slices" else ""

    def positions(self) -> tuple[float, float, float]:
        """The slice positions of the ``xy``, ``xz`` and ``yz`` panels.

        Each is read from its own panel: the position of the world axis
        that panel slices.
        """
        return tuple(
            float(self._scenes[key].dims.selection.slice_indices.get(axis, 0.0))
            for key, axis in zip(_SLICE_PANELS, self._axes, strict=True)
        )

    def slice_planes(
        self, like: Sequence[RenderPlane] | None = None
    ) -> tuple[RenderPlane, RenderPlane, RenderPlane]:
        """The three slice planes at the panels' current positions.

        Parameters
        ----------
        like : Sequence[RenderPlane] or None
            Slice planes to keep everything of but the position along the
            normal (their extents, ``enabled``, spin and outline).  ``None``
            builds new ones: through the point the three slices meet at,
            unbounded, with :attr:`outlines`.

        Returns
        -------
        tuple[RenderPlane, RenderPlane, RenderPlane]
            The planes of the ``xy``, ``xz`` and ``yz`` panels.
        """
        positions = self.positions()
        if like is not None:
            by_id = {plane.id: plane for plane in like}
            out = []
            for k, plane_id in enumerate(self._plane_ids):
                plane = by_id[plane_id]
                origin = list(plane.origin)
                if origin[k] != positions[k]:
                    origin[k] = positions[k]
                    plane = plane.model_copy(update={"origin": tuple(origin)})
                out.append(plane)
            return tuple(out)
        world = self._scenes["vol"].dims.world_coordinate_system
        axis_ids = tuple(world.axes[axis].id for axis in self._axes)
        unit = np.eye(3)
        return tuple(
            RenderPlane(
                id=self._plane_ids[k],
                coordinate_system=world.id,
                axes=axis_ids,
                origin=positions,
                # The other two axes, in the order whose cross product is
                # axis k.
                in_plane_axis_0=tuple(unit[(k + 1) % 3]),
                in_plane_axis_1=tuple(unit[(k + 2) % 3]),
                outline=self._outlines[k],
            )
            for k in range(3)
        )

    def add_visual(self, visual: Any) -> None:
        """Register a visual of the 3D panel.

        A visual with no render planes (a mesh, points) is ignored.  In
        ``"slices"`` mode the visual is given the slice planes: those the
        other visuals carry, so extents set on them are kept.

        Parameters
        ----------
        visual : visual model or UUID
            The 3D panel's visual of one ``add_*`` call.
        """
        visual_id = getattr(visual, "id", visual)
        model = self._controller.get_visual_model(visual_id)
        if not hasattr(model, "render_planes") or visual_id in self._visuals:
            return
        owner_id = self._owners[visual_id] = uuid4()
        self._visuals.append(visual_id)
        self._controller.on_render_planes_changed(
            visual_id, self._on_planes, owner_id=owner_id
        )
        self._controller.on_visual_removed(
            visual_id, self._on_removed, owner_id=owner_id
        )
        if self._mode == "slices":
            self._write(visual_id, self._current_planes(skip=visual_id))

    def close(self) -> None:
        """Stop following: close forwarded scopes and drop the subscriptions."""
        self._end_scrubs()
        self._controller.unsubscribe_owner(self._id)
        for owner_id in self._owners.values():
            self._controller.unsubscribe_owner(owner_id)
        self._owners.clear()
        self._visuals.clear()

    # ------------------------------------------------------------------
    # Planes
    # ------------------------------------------------------------------

    def _planes_of(self, visual_id: UUID) -> tuple[RenderPlane, ...]:
        return self._controller.get_visual_model(visual_id).render_planes

    def _slice_like(
        self, planes: Sequence[RenderPlane]
    ) -> tuple[RenderPlane, RenderPlane, RenderPlane] | None:
        """*planes* in panel order if they are three slice planes, else ``None``.

        Three planes with the slice planes' ids, on the 3D panel's axes,
        each with its normal along its own axis.  Where a plane sits along
        its normal, its extents, ``enabled`` and spin are free.
        """
        if len(planes) != 3:
            return None
        by_id = {plane.id: plane for plane in planes}
        world = self._scenes["vol"].dims.world_coordinate_system
        out = []
        for k, plane_id in enumerate(self._plane_ids):
            plane = by_id.get(plane_id)
            if plane is None:
                return None
            try:
                axes = tuple(world.resolve(axis) for axis in plane.axes)
            except (KeyError, ValueError):
                return None
            if axes != self._axes:
                return None
            normal = np.cross(plane.in_plane_axis_0, plane.in_plane_axis_1)
            if 1.0 - abs(float(normal[k])) > _NORMAL_TOL:
                return None
            out.append(plane)
        return tuple(out)

    def _current_planes(
        self, skip: UUID | None = None
    ) -> tuple[RenderPlane, RenderPlane, RenderPlane]:
        """The slice planes now: a visual's, moved to the panels' positions."""
        for visual_id in self._visuals:
            if visual_id == skip:
                continue
            like = self._slice_like(self._planes_of(visual_id))
            if like is not None:
                return self.slice_planes(like)
        return self.slice_planes()

    def _write(self, visual_id: UUID, planes: Sequence[RenderPlane]) -> None:
        """Give *visual_id* *planes* if it does not carry them already."""
        if self._planes_of(visual_id) == tuple(planes):
            return
        was, self._syncing = self._syncing, True
        try:
            self._controller.set_render_planes(visual_id, planes, source_id=self._id)
        finally:
            self._syncing = was

    def _on_dims(self, _event: DimsChangedEvent) -> None:
        """A slice position may have changed: move the planes to it."""
        if self._mode != "slices" or self._syncing or not self._visuals:
            return
        planes = self._current_planes()
        for visual_id in self._visuals:
            self._write(visual_id, planes)

    def _on_planes(self, event: RenderPlanesChangedEvent) -> None:
        """A visual's planes changed from outside: carry the change over."""
        if self._mode != "slices" or self._syncing:
            return
        visual_id = event.visual_id
        if visual_id not in self._owners:
            return
        planes = self._planes_of(visual_id)
        if planes != tuple(event.render_planes):
            # A subscriber called before this one changed the visual again;
            # that change's own event carries the tuple to act on.
            return
        like = self._slice_like(planes)
        if like is None:
            if not self._warned:
                self._warned = True
                warnings.warn(
                    "In plane_controller.mode 'slices' the 3D panel's render "
                    "planes are the three slice planes: one per 2D panel, "
                    "facing its sliced axis.  A change that tilts, adds, "
                    "removes or replaces one cannot be kept and was put "
                    "back.  Set plane_controller.mode = None to edit the "
                    "planes freely.",
                    UserWarning,
                    stacklevel=2,
                )
            self._write(visual_id, self._current_planes(skip=visual_id))
            return
        # An outline given to a plane by hand is the planes' outline now.
        self._outlines = tuple(plane.outline for plane in like)
        # The planes' positions become the panels' slice positions...
        moved = {
            axis: float(plane.origin[k])
            for k, (axis, plane) in enumerate(zip(self._axes, like, strict=True))
            if float(plane.origin[k]) != self.positions()[k]
        }
        was, self._syncing = self._syncing, True
        try:
            if moved:
                self._move_sliders(moved)
        finally:
            self._syncing = was
        # ... and every visual takes the tuple: extents, ``enabled`` and
        # spin with it.  Read back from the sliders, in case a position was
        # not taken as given.
        planes = self.slice_planes(like)
        for other in self._visuals:
            self._write(other, planes)

    def _move_sliders(self, positions: Mapping[int, float]) -> None:
        linked = None if self._dims_controller is None else self._dims_controller()
        if linked is not None:
            linked.set_slice_positions(positions, source_id=self._id)
            return
        for key, axis in zip(_SLICE_PANELS, self._axes, strict=True):
            if axis in positions:
                for scene in (self._scenes[key], self._scenes["vol"]):
                    self._controller.update_slice_indices(
                        scene.id, {axis: positions[axis]}, source_id=self._id
                    )

    # ------------------------------------------------------------------
    # Scrub forwarding
    # ------------------------------------------------------------------

    def _on_dims_interaction(self, event: DimsInteractionEvent) -> None:
        """Hold a plane scope on the visuals while a 2D slider is dragged."""
        scene_id = event.scene_id
        if event.phase == "end":
            self._end_scrub(scene_id)
            return
        if self._mode != "slices" or scene_id in self._scrubbing:
            return
        self._scrubbing.add(scene_id)
        for visual_id in self._visuals:
            self._controller.begin_plane_interaction(
                visual_id, source_id=self._scope_ids[scene_id]
            )

    def _end_scrub(self, scene_id: UUID) -> None:
        if scene_id not in self._scrubbing:
            return
        self._scrubbing.discard(scene_id)
        for visual_id in self._visuals:
            self._controller.end_plane_interaction(
                visual_id, source_id=self._scope_ids[scene_id]
            )

    def _end_scrubs(self) -> None:
        for scene_id in list(self._scrubbing):
            self._end_scrub(scene_id)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def _on_removed(self, event: VisualRemovedEvent) -> None:
        removed = event.visual_id
        owner_id = self._owners.pop(removed, None)
        if owner_id is None:
            return
        self._visuals.remove(removed)
        self._controller.unsubscribe_owner(owner_id)
