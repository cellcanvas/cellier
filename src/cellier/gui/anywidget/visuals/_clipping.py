"""The anywidget "Clipping planes" control of a visual.

One widget whose synced trait holds the whole list of planes; its front
end draws a row per entry.  Adding a plane constructs no widget, so
marimo's rule against building widgets outside a running cell never
applies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._clipping_planes import (
    CLIPPING_PLANES_TITLE,
    ClippingPlanesEditor,
)
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec

_STATIC = Path(__file__).parent / "static"


class AnywidgetClippingPlanesControls(VisualIdGroup, anywidget.AnyWidget):
    """A visual's clipping planes: one row per plane, and an add button.

    Each row has an enabled checkbox, a flip button, a remove button, the
    normal and a position slider along the normal.  The normal has one
    column per data axis: two buttons named after the axis as the store's
    coordinate system gives it (``+z`` and ``-z``) that face the plane
    along the axis, and under them the normal's entry on it.  Values are in
    the visual's data coordinates.  Given *gizmo*, a row also has a "Gizmo"
    toggle that puts a gizmo on its plane in the 3D canvas; one plane of a
    canvas has it at a time.

    The front end reports an action by setting ``edit`` to
    ``{"action", "index", "value", "serial"}``; the Python side applies it
    and syncs ``rows`` back.  A refused edit comes back as ``error`` with
    ``rows`` unchanged.  ``rows`` and ``error`` are written by the Python
    side only; a value a host writes into them is put back.  An update is sent only when the planes differ from
    the ones held, so a host that delivers one edit twice sends one.

    Wire to the controller after construction::

        seed = get_clipping_planes_data_from_visual(visual, store)
        controls = AnywidgetClippingPlanesControls(visual.id, **seed)
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The visual, or a group of them (an ``OrthoViewer``'s panel
        siblings), edited together.
    coordinate_system :
        The level-0 data coordinate system of the store they read (its id).
    axis_names :
        The data axis names, in order.
    bounds :
        ``(low, high)`` of the data on each axis.
    planes :
        The current rows (``rows_from_planes(visual.clipping_planes)``).
    data_store_id :
        The store the visuals read.
    bounds_source :
        Reads the store's current ``(low, high)`` per axis.  Given with
        *data_store_id*, the position ranges follow the store's extent;
        without it they stay at *bounds*.
    gizmo, gizmo_plane, gizmo_blocked :
        Where a plane's gizmo is drawn, the plane that has it now, and why
        a plane cannot have one: ``gizmo_seed(controller, visual_ids)``.
        Without *gizmo* the rows have no gizmo toggle.
    """

    _esm = _STATIC / "clipping_planes.js"
    _css = _STATIC / "clipping_planes.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = CLIPPING_PLANES_TITLE
    """Name shown when no ``title=`` is given."""

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    #: The data axis names, in the order of a normal's entries.
    axis_names = traitlets.List([]).tag(sync=True)
    #: One entry per plane: id, enabled, normal, position, facing, low, high;
    #: with a gizmo toggle, also gizmo and gizmo_blocked.
    rows = traitlets.List([]).tag(sync=True)
    #: The reason the last edit was refused, or "".
    error = traitlets.Unicode("").tag(sync=True)
    #: Set by the front end: ``{"action", "index", "value", "serial"}``.
    edit = traitlets.Dict({}).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        coordinate_system: UUID | str,
        axis_names: Sequence[str],
        bounds: Sequence[Sequence[float]],
        planes: Sequence[Mapping[str, Any]] = (),
        data_store_id: UUID | str | None = None,
        bounds_source: Callable[[], Sequence[Sequence[float]]] | None = None,
        gizmo: Mapping[str, Any] | None = None,
        gizmo_plane: str | None = None,
        gizmo_blocked: Callable[[str], str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(axis_names=[*map(str, axis_names)], **kwargs)
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._editor = ClippingPlanesEditor(
            self._visual_ids,
            coordinate_system,
            axis_names,
            bounds,
            planes,
            self._id,
            self.changed.emit,
            self._show,
            data_store_id=data_store_id,
            bounds_source=bounds_source,
            gizmo=gizmo,
            gizmo_plane=gizmo_plane,
            gizmo_blocked=gizmo_blocked,
        )
        self._shown_error = ""
        self._show(self._editor.rows, "")
        self.observe(self._on_edit, names="edit")
        self.observe(self._keep_shown, names=["rows", "error"])

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self) -> AnywidgetClippingPlanesControls:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    @property
    def editor(self) -> ClippingPlanesEditor:
        """The toolkit-neutral half (for tests and scripting)."""
        return self._editor

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``ClippingPlanesChangedEvent`` per visual."""
        return self._editor.subscription_specs()

    # -- widget -> model -------------------------------------------------------

    def _on_edit(self, change) -> None:
        edit = change["new"] or {}
        action = edit.get("action")
        editor = self._editor
        if action == "add":
            editor.add()
            return
        index = edit.get("index")
        if not isinstance(index, int) or not 0 <= index < len(editor.rows):
            return
        value = edit.get("value")
        if action == "remove":
            editor.remove(index)
        elif action == "enabled":
            editor.set_enabled(index, bool(value))
        elif action == "position":
            editor.set_position(index, float(value))
        elif action == "flip":
            editor.flip(index)
        elif action == "gizmo":
            editor.set_gizmo(index, bool(value))
        elif action in ("facing", "component"):
            # ``[axis index, sign]`` or ``[axis index, entry]``.
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return
            axis, amount = value
            if not isinstance(axis, int) or not 0 <= axis < len(editor.axis_names):
                return
            if action == "facing":
                editor.set_facing(index, axis, -1 if float(amount) < 0 else 1)
            else:
                editor.set_component(index, axis, float(amount))

    # -- model -> widget -------------------------------------------------------

    def _show(self, rows: list[dict[str, Any]], error: str) -> None:
        self._shown_error = error
        with self.hold_sync():
            self.rows = self._editor.describe()
            self.error = error

    def _keep_shown(self, change) -> None:
        """Put back ``rows`` and ``error`` when a host overwrites them.

        They are written by this side only.  marimo sends a widget's whole
        state with every front-end edit, so the front end's copy of them,
        which is the one from before the edit, arrives with the edit and
        would replace what the edit just produced.
        """
        if change["name"] == "rows":
            shown = self._editor.describe()
            if change["new"] != shown:
                self.rows = shown
        elif change["new"] != self._shown_error:
            self.error = self._shown_error
