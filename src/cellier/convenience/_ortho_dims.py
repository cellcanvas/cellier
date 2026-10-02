"""Mirror dims state across the four OrthoViewer panels (design 3.6)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.controller import CellierController
    from cellier.events import DimsInteractionEvent
    from cellier.scene.scene import Scene


class OrthoDimsController:
    """Keep every panel's slice positions, thicknesses and overrides equal.

    The four OrthoViewer scenes show one world, and every scene keeps a
    position for every axis (D36), so mirrored scenes hold identical
    ``slice_indices``: one world point, the crosshair.  The XY panel slices at
    its ``z``, and that same ``z`` is the stored position the XZ and YZ panels
    keep for their displayed ``z``.

    This holds **no dims state**.  Each scene's ``DimsManager`` stays the
    source of truth and is what gets serialized; this only copies between
    them.  ``displayed_axes`` is per panel and is never mirrored.

    A **scrub** is forwarded too (interaction tracker design 4.8).  A
    mirrored position reaches the other panels as a plain
    ``update_slice_indices``, which by itself would be a jump and plan each
    of them in full on every tick.  So when a scrub starts on one panel, this
    opens a dims interaction scope on every other panel, and closes them when
    that scrub ends: the mirrored ticks are then scrub ticks, and the other
    panels end with ``"release"``.  A panel that displays the scrubbed axis
    receives no tick, and its forwarded scope opens and closes silently.

    Parameters
    ----------
    controller : CellierController
        The controller owning the scenes.
    scenes : Sequence[Scene]
        The panels, first panel first.  When they disagree at construction,
        the first panel's values are copied to the others.

    Attributes
    ----------
    enabled : bool
        While ``False``, edits are not mirrored.  Re-enabling copies the first
        panel's values to the others so they agree again.
    """

    def __init__(self, controller: CellierController, scenes: Sequence[Scene]) -> None:
        self._id: UUID = uuid4()
        self._controller = controller
        self._scenes: list[Scene] = list(scenes)
        self._syncing = False
        self._enabled = True
        self._handlers: list[tuple[Any, Any]] = []
        # One scope source per panel a scrub can start on, so two panels
        # scrubbed at once hold two scopes on the others; and the panels
        # each origin currently holds a scope on.
        self._scope_ids: dict[UUID, UUID] = {scene.id: uuid4() for scene in scenes}
        self._forwarded: dict[UUID, list[UUID]] = {}
        for scene in self._scenes:
            handler = self._make_handler(scene.id)
            scene.dims.events.connect(handler)
            self._handlers.append((scene.dims.events, handler))
            controller.on_dims_interaction(
                scene.id, self._on_dims_interaction, owner_id=self._id
            )
        if self._scenes:
            self._mirror_from(self._scenes[0].id, source_id=self._id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def id(self) -> UUID:
        """The source id stamped on every mirrored write."""
        return self._id

    @property
    def scene_ids(self) -> tuple[UUID, ...]:
        """The mirrored scenes, in panel order."""
        return tuple(scene.id for scene in self._scenes)

    @property
    def enabled(self) -> bool:
        """Whether edits are mirrored across the panels."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        was_enabled = self._enabled
        self._enabled = bool(value)
        if self._enabled and not was_enabled and self._scenes:
            self._mirror_from(self._scenes[0].id, source_id=self._id)

    def set_slice_position(
        self,
        axis: int,
        value: float,
        *,
        source_id: UUID | None = None,
        interactive: bool = False,
    ) -> None:
        """Move one world axis's slice position on every panel.

        Parameters
        ----------
        axis : int
            World axis index.
        value : float
            World position.
        source_id : UUID or None
            Stamped on every emitted ``DimsChangedEvent``.  Defaults to this
            controller's id.
        interactive : bool
            Whether the move is a tick of a scrub; see
            ``CellierController.update_slice_indices``.
        """
        self.set_slice_positions(
            {axis: value}, source_id=source_id, interactive=interactive
        )

    def set_slice_positions(
        self,
        positions: Mapping[int, float],
        *,
        source_id: UUID | None = None,
        interactive: bool = False,
    ) -> None:
        """Move several world axes' slice positions on every panel at once.

        Parameters
        ----------
        positions : Mapping[int, float]
            World axis index -> world position.
        source_id : UUID or None
            Stamped on every emitted ``DimsChangedEvent``.  Defaults to this
            controller's id.
        interactive : bool
            Whether the move is a tick of a scrub; see
            ``CellierController.update_slice_indices``.
        """
        resolved = source_id if source_id is not None else self._id
        with self._guard():
            for scene in self._scenes:
                self._controller.update_slice_indices(
                    scene.id, positions, source_id=resolved, interactive=interactive
                )

    def set_slider_override(
        self, axis: int, value: bool | None, *, source_id: UUID | None = None
    ) -> None:
        """Force a slider shown or hidden, or clear that, on every panel.

        Parameters
        ----------
        axis : int
            World axis index.
        value : bool or None
            ``True`` force-shows, ``False`` force-hides, ``None`` clears.
        source_id : UUID or None
            Stamped on any emitted ``SliderAxesChangedEvent``.
        """
        resolved = source_id if source_id is not None else self._id
        with self._guard():
            for scene in self._scenes:
                self._controller.set_slider_override(
                    scene.id, axis, value, source_id=resolved
                )

    def close(self) -> None:
        """Stop mirroring, close forwarded scopes and drop the connections."""
        for signal, handler in self._handlers:
            signal.disconnect(handler)
        self._handlers.clear()
        self._controller.unsubscribe_owner(self._id)
        for origin in list(self._forwarded):
            self._end_forwarded(origin)

    # ------------------------------------------------------------------
    # Inbound mirroring
    # ------------------------------------------------------------------

    def _make_handler(self, scene_id: UUID):
        def _on_dims(_info: Any) -> None:
            if not self._enabled or self._syncing:
                return
            self._mirror_from(scene_id, source_id=self._id)

        return _on_dims

    def _mirror_from(self, source_scene_id: UUID, *, source_id: UUID) -> None:
        """Copy positions, thicknesses and overrides from one panel to the rest.

        Only the differences are written, so an edit reaches each other panel
        as one change and a panel already in agreement emits nothing.
        """
        source = next(scene for scene in self._scenes if scene.id == source_scene_id)
        positions = dict(source.dims.selection.slice_indices)
        thickness = dict(source.dims.selection.thickness)
        overrides = dict(source.dims.slider_overrides)
        with self._guard():
            for scene in self._scenes:
                if scene.id == source_scene_id:
                    continue
                current = scene.dims.selection.slice_indices
                moved = {
                    axis: value
                    for axis, value in positions.items()
                    if current.get(axis) != value
                }
                if moved:
                    self._controller.update_slice_indices(
                        scene.id, moved, source_id=source_id
                    )
                self._controller.update_thickness(
                    scene.id, thickness, source_id=source_id
                )
                for axis in set(scene.dims.slider_overrides) | set(overrides):
                    self._controller.set_slider_override(
                        scene.id, axis, overrides.get(axis), source_id=source_id
                    )

    # ------------------------------------------------------------------
    # Scrub forwarding
    # ------------------------------------------------------------------

    def _on_dims_interaction(self, event: DimsInteractionEvent) -> None:
        """Hold a scope on the other panels for as long as one is scrubbed."""
        origin = event.scene_id
        if event.phase == "end":
            # Whatever ended it, and whoever: only an origin has an entry.
            self._end_forwarded(origin)
            return
        # Echo guard: a scrub this mirror caused (a mirrored tick, or a move
        # through ``set_slice_positions``) is not forwarded back.
        if not self._enabled or event.source_id == self._id:
            return
        if origin in self._forwarded or origin not in self._scope_ids:
            return
        others = [scene.id for scene in self._scenes if scene.id != origin]
        self._forwarded[origin] = others
        for other in others:
            self._controller.begin_dims_interaction(
                other, source_id=self._scope_ids[origin]
            )

    def _end_forwarded(self, origin: UUID) -> None:
        for other in self._forwarded.pop(origin, ()):
            self._controller.end_dims_interaction(
                other, source_id=self._scope_ids[origin]
            )

    def _guard(self):
        return _Reentrancy(self)


class _Reentrancy:
    """Set ``_syncing`` for the duration of a mirrored write."""

    def __init__(self, owner: OrthoDimsController) -> None:
        self._owner = owner
        self._previous = False

    def __enter__(self) -> None:
        self._previous = self._owner._syncing
        self._owner._syncing = True

    def __exit__(self, *_exc: object) -> None:
        self._owner._syncing = self._previous
