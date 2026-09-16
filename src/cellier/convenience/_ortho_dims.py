"""Mirror dims state across the four OrthoViewer panels (design 3.6)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.controller import CellierController
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
        for scene in self._scenes:
            handler = self._make_handler(scene.id)
            scene.dims.events.connect(handler)
            self._handlers.append((scene.dims.events, handler))
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
        self, axis: int, value: float, *, source_id: UUID | None = None
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
        """
        self.set_slice_positions({axis: value}, source_id=source_id)

    def set_slice_positions(
        self, positions: Mapping[int, float], *, source_id: UUID | None = None
    ) -> None:
        """Move several world axes' slice positions on every panel at once.

        Parameters
        ----------
        positions : Mapping[int, float]
            World axis index -> world position.
        source_id : UUID or None
            Stamped on every emitted ``DimsChangedEvent``.  Defaults to this
            controller's id.
        """
        resolved = source_id if source_id is not None else self._id
        with self._guard():
            for scene in self._scenes:
                self._controller.update_slice_indices(
                    scene.id, positions, source_id=resolved
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
        """Stop mirroring and drop the model connections."""
        for signal, handler in self._handlers:
            signal.disconnect(handler)
        self._handlers.clear()

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
