"""The anywidget "Render planes" control of a visual.

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
from cellier.gui._render_planes import RENDER_PLANES_TITLE, RenderPlanesEditor
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec
    from cellier.gui._render_planes import RenderPlaneGizmoTarget

_STATIC = Path(__file__).parent / "static"


def _pair(value: Any, size: int) -> list | None:
    """*value* as a list of *size* entries, or ``None`` if it is not one."""
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    return list(value)


class AnywidgetRenderPlanesControls(VisualIdGroup, anywidget.AnyWidget):
    """A visual's render planes: one row per plane, and an add button.

    The planes a visual in the ``"plane"`` render mode draws its data on.
    Each row is a "Clipping planes" row with the extents added: an enabled
    checkbox, a flip button, a remove button, the normal (two buttons and an
    entry per axis of the plane), a position slider along the normal, and
    for each of the plane's two in-plane axes a minimum and a maximum, each
    with an "unbounded" box.  Values are in world units.  Given
    *gizmo_target*, a row also has a "Gizmo" toggle that puts the canvas's
    gizmo on its plane; one plane of a canvas has it at a time, clipping
    planes included.

    The front end reports an action by setting ``edit`` to
    ``{"action", "index", "value", "serial"}``; the Python side applies it
    and syncs ``rows`` back.  A refused edit comes back as ``error`` with
    ``rows`` unchanged.  ``rows``, ``error``, ``blocked`` and ``can_add``
    are written by the Python side only; a value a host writes into them is
    put back.  An update is sent only when the planes differ from the ones
    held, so a host that delivers one edit twice sends one.

    Wire to the controller after construction::

        data = get_render_planes_data_from_visual(controller, visual.id)
        controls = AnywidgetRenderPlanesControls(visual.id, **data)
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The visual, or a group of them, edited together.
    coordinate_system :
        The scene's world coordinate system (its id).
    axis_ids, axis_names :
        The world axes, in order: their ids and names.
    planes :
        The visual's current ``render_planes``.
    displayed_axes, bounds, state_source, scene_id, data_store_id :
        What the scene and the visual's data are now, and a reader of it:
        ``get_render_planes_data_from_visual(controller, visual_id)``.
    gizmo_target, gizmo_plane, gizmo_blocked :
        Where a plane's gizmo is drawn (a ``RenderPlaneGizmoTarget``), the
        plane that has it now, and why a plane cannot have one:
        ``get_render_plane_gizmo_data(controller, visual_id, canvas_id)``.
        Without *gizmo_target* the rows have no gizmo toggle.
    """

    _esm = _STATIC / "render_planes.js"
    _css = _STATIC / "render_planes.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = RENDER_PLANES_TITLE
    """Name shown when no ``title=`` is given."""

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    #: One entry per plane: id, enabled, axes, normal, position, extent_0,
    #: extent_1, facing, low, high; with a gizmo toggle, also gizmo and
    #: gizmo_blocked.
    rows = traitlets.List([]).tag(sync=True)
    #: The reason the last edit was refused, or "".
    error = traitlets.Unicode("").tag(sync=True)
    #: Why the control is disabled now (the planes are not drawn), or "".
    blocked = traitlets.Unicode("").tag(sync=True)
    #: Whether a plane can be added: not disabled, and under four planes.
    can_add = traitlets.Bool(True).tag(sync=True)
    #: Set by the front end: ``{"action", "index", "value", "serial"}``.
    edit = traitlets.Dict({}).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        coordinate_system: UUID | str,
        axis_ids: Sequence[str],
        axis_names: Sequence[str],
        planes: Sequence[Any] = (),
        displayed_axes: Sequence[str] | None = None,
        bounds: Sequence[Sequence[float]] | None = None,
        state_source: Callable[[], Mapping[str, Any]] | None = None,
        scene_id: UUID | str | None = None,
        data_store_id: UUID | str | None = None,
        gizmo_target: RenderPlaneGizmoTarget | None = None,
        gizmo_plane: str | None = None,
        gizmo_blocked: Callable[[str], str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._shown: dict[str, Any] = {}
        self._editor = RenderPlanesEditor(
            self._visual_ids,
            coordinate_system,
            axis_ids,
            axis_names,
            planes,
            self._id,
            self.changed.emit,
            self._show,
            displayed_axes=displayed_axes,
            bounds=bounds,
            state_source=state_source,
            scene_id=scene_id,
            data_store_id=data_store_id,
            gizmo_target=gizmo_target,
            gizmo_plane=gizmo_plane,
            gizmo_blocked=gizmo_blocked,
        )
        self._show(self._editor.rows, "")
        self.observe(self._on_edit, names="edit")
        self.observe(self._keep_shown, names=["rows", "error", "blocked", "can_add"])

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self) -> AnywidgetRenderPlanesControls:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    @property
    def editor(self) -> RenderPlanesEditor:
        """The toolkit-neutral half (for tests and scripting)."""
        return self._editor

    def refresh(self) -> None:
        """Read the scene's state again (whether the control is disabled)."""
        self._editor.refresh()

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """What the control follows on the bus (see the editor)."""
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
        if not isinstance(index, int) or not 0 <= index < len(editor.planes):
            return
        value = edit.get("value")
        if action == "remove":
            editor.remove(index)
        elif action == "enabled":
            editor.set_enabled(index, bool(value))
        elif action == "depth":
            editor.set_depth(index, float(value))
        elif action == "flip":
            editor.flip(index)
        elif action == "gizmo":
            editor.set_gizmo(index, bool(value))
        elif action == "outline":
            editor.set_outline_enabled(index, bool(value))
        elif action == "outline_color":
            # ``#rrggbb``, as a colour input gives it.
            editor.set_outline_color(index, str(value))
        elif action in ("facing", "component"):
            # ``[axis index, sign]`` or ``[axis index, entry]``.
            pair = _pair(value, 2)
            if pair is None or pair[0] not in (0, 1, 2):
                return
            axis, amount = pair
            if action == "facing":
                editor.set_facing(index, axis, -1 if float(amount) < 0 else 1)
            else:
                editor.set_component(index, axis, float(amount))
        elif action in ("side", "bounded"):
            # ``[in-plane axis, side, value]`` or ``[..., bounded]``.
            triple = _pair(value, 3)
            if triple is None or triple[0] not in (0, 1) or triple[1] not in (0, 1):
                return
            axis, side, amount = triple
            if action == "side":
                editor.set_side(index, axis, side, float(amount))
            else:
                editor.set_side_bounded(index, axis, side, bool(amount))

    # -- model -> widget -------------------------------------------------------

    def _show(self, rows: list[dict[str, Any]], error: str) -> None:
        self._shown = {
            "rows": self._editor.describe(),
            "error": error,
            "blocked": self._editor.blocked,
            "can_add": self._editor.can_add,
        }
        with self.hold_sync():
            for name, value in self._shown.items():
                setattr(self, name, value)

    def _keep_shown(self, change) -> None:
        """Put back a trait this side owns when a host overwrites it.

        They are written by this side only.  marimo sends a widget's whole
        state with every front-end edit, so the front end's copy of them,
        which is the one from before the edit, arrives with the edit and
        would replace what the edit just produced.
        """
        name = change["name"]
        if name in self._shown and change["new"] != self._shown[name]:
            setattr(self, name, self._shown[name])
