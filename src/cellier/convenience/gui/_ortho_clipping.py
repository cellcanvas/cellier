"""The clipping planes widget of an ``OrthoViewer``.

See ``plans/clipping_planes_linking_design.md`` (section 6).

One widget holds the link mode selector and every clipping planes control
of the viewer: for each dataset added with ``clipping_controls=True``, one
control per link group of the current mode.

Built from what ``ControlsDock`` is built from: the layout host's
``live_slot`` and the backend's ``target_selector``.  So it is one class
for Qt, Jupyter and marimo, with no toolkit widget of its own.

**Every control any mode can need is built up front; a mode change only
swaps what is shown.**  A clipping planes control's visuals are fixed when
it is built, and on marimo a widget can only be constructed while a cell is
running, which a mode picked in the selector is not.  So per dataset the six
controls of :data:`CONTROL_PANELS` are built when the dataset is added (cell
code), stay wired to the bus while hidden, and are correct the moment they
are shown.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.convenience._hosts import LayoutHost
    from cellier.convenience._ortho_viewer import OrthoViewer

SELECTOR_TITLE = "Link"
"""The mode selector's label."""

PLACEHOLDER = "No visuals with clipping controls"
"""What the widget says, under the selector, while no dataset has controls."""

MODES: tuple = ("all", "2d", None)
"""The link modes, in the selector's order."""

MODE_LABELS: dict = {"all": "All views", "2d": "2D views", None: "Not linked"}
"""What the selector calls each mode."""

CONTROL_PANELS: dict[str, tuple[str, ...]] = {
    "all": ("xy", "xz", "yz", "vol"),
    "2d": ("xy", "xz", "yz"),
    "xy": ("xy",),
    "xz": ("xz",),
    "yz": ("yz",),
    "vol": ("vol",),
}
"""The six controls built per dataset: name -> the panels it edits."""

CONTROL_TITLES: dict[str, str] = {
    "all": "All views",
    "2d": "2D views",
    "xy": "XY",
    "xz": "XZ",
    "yz": "YZ",
    "vol": "3D view",
}
"""What each control is called, after its dataset's name.

A control of several panels that has lost some of them is called by the
panels it has left (:func:`control_title`).
"""

SHOWN: dict = {
    "all": ("all",),
    "2d": ("2d", "vol"),
    None: ("xy", "xz", "yz", "vol"),
}
"""Mode -> the controls shown per dataset: one per link group."""


class OrthoClippingWidget:
    """The link mode selector over an ``OrthoViewer``'s clipping controls.

    The selector is bound both ways to ``viewer.clipping_controller.mode``.
    Under it, each dataset added with ``clipping_controls=True`` has one
    clipping planes control per link group of the mode, titled with the
    dataset's name: "All views" in ``"all"``; "2D views" and "3D view" in
    ``"2d"``; "XY", "XZ", "YZ" and "3D view" in ``None``.  Only a control
    that edits the 3D panel's visual has a gizmo toggle.

    The widget follows the viewer: a dataset added later appears, a removed
    one goes.  A dataset that has lost some of its panels stays, with
    controls for the panels that are left; a control of several panels is
    then titled with the panels it still edits ("XY, XZ, 3D view").

    Parameters
    ----------
    viewer : OrthoViewer
        The viewer.  Its 3D panel must have its canvas: build the canvases
        before this widget.
    host : LayoutHost
        The layout host composing the widget.
    """

    def __init__(self, viewer: OrthoViewer, host: LayoutHost) -> None:
        self._viewer = viewer
        self._host = host
        self._controller = viewer.controller
        self._linker = viewer.clipping_controller
        self._slot = host.live_slot()
        #: Dataset key -> {control name: control}, in dataset order.
        self._built: dict[tuple, dict[str, object]] = {}
        self._shown: tuple | None = None
        self._closed = False
        self._selector = host.backend.target_selector(
            [MODE_LABELS[mode] for mode in MODES],
            MODES.index(self._linker.mode),
            title=SELECTOR_TITLE,
        )
        self._selector.selected.connect(self._on_selected)
        self._linker.mode_changed.connect(self._on_mode)
        self._linker.groups_changed.connect(self.refresh)
        viewer._controls_changed.connect(self.refresh)
        try:
            self.refresh()
        except Exception:
            # Not half built: nothing is left following the viewer.
            self.close()
            raise

    # -- Public interface ------------------------------------------------------

    @property
    def root(self) -> object:
        """The host item to place in the layout."""
        return self._slot.root

    @property
    def selector(self) -> object:
        """The mode selector."""
        return self._selector

    @property
    def datasets(self) -> list[str]:
        """The names of the datasets with controls, in the order shown."""
        return [label for label, _items in self._built]

    @property
    def controls(self) -> list[dict[str, object]]:
        """Every control built, shown or not: control name -> control, per dataset."""
        return [dict(controls) for controls in self._built.values()]

    @property
    def widgets(self) -> list:
        """The controls the current mode shows, dataset by dataset."""
        return [
            controls[name]
            for controls in self._built.values()
            for name in SHOWN[self._linker.mode]
            if name in controls
        ]

    def refresh(self) -> None:
        """Build the controls of new datasets and release those of gone ones.

        Runs when the viewer's configured visuals or the linker's groups
        change: from an ``add_*`` or a removal, both cell code on marimo.
        """
        if self._closed:
            return
        datasets = self._datasets()
        # Released before anything is built, so the controller drops their
        # subscriptions before a replacement subscribes.
        for key in [key for key in self._built if key not in datasets]:
            _close_all(self._built.pop(key).values())
        self._built = {
            key: self._built.get(key) or self._build(key[0], group)
            for key, group in datasets.items()
        }
        self._render()

    def close(self) -> None:
        """Stop following the viewer and release every widget built."""
        if self._closed:
            return
        self._closed = True
        self._viewer._controls_changed.disconnect(self.refresh, missing_ok=True)
        self._linker.groups_changed.disconnect(self.refresh, missing_ok=True)
        self._linker.mode_changed.disconnect(self._on_mode, missing_ok=True)
        for controls in self._built.values():
            _close_all(controls.values())
        self._built = {}
        _close_all([self._selector, self._slot])

    # -- Internals -------------------------------------------------------------

    def _datasets(self) -> dict[tuple, dict[str, UUID]]:
        """The datasets to show: ``(label, panel items)`` -> panel -> visual id.

        A dataset is one of the linker's groups whose controls config has
        ``clipping_controls``.  The key holds what a control is built from,
        so a dataset that loses a panel, or whose label changes, is rebuilt.
        """
        from cellier.convenience.layout._shared import unique_labels

        controller = self._controller
        configs = self._viewer._controls_configs
        rep_of = {
            visual_id: rep
            for rep, visual_ids in self._viewer._visual_groups.items()
            for visual_id in visual_ids
        }
        found = []
        for group in self._linker.groups:
            # The registry announces a removal before the linker has
            # forgotten the visual: keep what the controller still has.
            group = {
                key: visual_id
                for key, visual_id in group.items()
                if _is_registered(controller, visual_id)
            }
            flagged = any(
                getattr(configs.get(rep_of.get(visual_id)), "clipping_controls", False)
                for visual_id in group.values()
            )
            if group and flagged:
                key, visual_id = next(iter(group.items()))
                name = str(controller.get_visual_model(visual_id).name)
                suffix = f"_{key}"
                found.append(
                    (name[: -len(suffix)] if name.endswith(suffix) else name, group)
                )
        labels = unique_labels([name for name, _group in found])
        return {
            (label, tuple(group.items())): group
            for label, (_name, group) in zip(labels, found)
        }

    def _build(self, label: str, group: dict[str, UUID]) -> dict[str, object]:
        """The controls of one dataset: every one a mode can show.

        A control none of whose panels is left is not built.
        """
        from cellier.convenience.layout._shared import (
            ControlSpec,
            _resolve_data_store,
        )
        from cellier.gui._clipping_planes import get_clipping_planes_data_from_visual

        controller = self._controller
        builder = self._host.backend.builders["clipping_planes"]
        controls: dict[str, object] = {}
        try:
            for name in CONTROL_PANELS:
                visual_ids = [
                    group[key] for key in CONTROL_PANELS[name] if key in group
                ]
                if not visual_ids:
                    continue
                visual = controller.get_visual_model(visual_ids[0])
                store = _resolve_data_store(controller, visual)
                spec = ControlSpec(
                    "clipping_planes",
                    f"{label}: {control_title(name, group)}",
                    get_clipping_planes_data_from_visual(visual, store),
                )
                control = builder(
                    spec,
                    visual_ids,
                    controller,
                    self._viewer._clipping_gizmo_target(visual_ids),
                )
                # Held before it is wired, so a failure from here on
                # releases it with the others.
                controls[name] = control
                controller.connect_widget(
                    control, subscription_specs=control.subscription_specs()
                )
        except Exception:
            # Not half built: the controls built so far are wired to the
            # bus and nothing else holds them.
            _close_all(controls.values())
            raise
        return controls

    def _on_selected(self, index: int) -> None:
        # Never builds (see the module docstring).
        if not self._closed and 0 <= index < len(MODES):
            self._linker.mode = MODES[index]

    def _on_mode(self, mode) -> None:
        if self._closed:
            return
        self._selector.set_choices([MODE_LABELS[m] for m in MODES], MODES.index(mode))
        self._render()

    def _render(self) -> None:
        items = [self._selector, *self.widgets]
        placeholder = None if self._built else PLACEHOLDER
        shown = (tuple(id(item) for item in items), placeholder)
        if shown != self._shown:
            self._slot.set(items, placeholder=placeholder)
            self._shown = shown


def control_title(name: str, group: dict[str, UUID]) -> str:
    """What the control *name* of a dataset is called, after the dataset's name.

    Parameters
    ----------
    name : str
        A key of :data:`CONTROL_PANELS`.
    group : dict[str, UUID]
        Panel key -> visual id: the panels the dataset has left.

    Returns
    -------
    str
        The entry of :data:`CONTROL_TITLES` while the control has every one
        of its panels.  For a control of several panels that has lost some,
        the titles of the panels left ("XY, XZ, 3D view"): "All views"
        would name a panel it does not edit.
    """
    panels = CONTROL_PANELS[name]
    left = [key for key in panels if key in group]
    if len(left) == len(panels):
        return CONTROL_TITLES[name]
    return ", ".join(CONTROL_TITLES[key] for key in left)


def _is_registered(controller: object, visual_id: UUID) -> bool:
    try:
        controller.get_visual_scene_id(visual_id)
    except KeyError:
        return False
    return True


def _close_all(widgets) -> None:
    for widget in list(widgets):
        close = getattr(widget, "close", None)
        if close is not None:
            with suppress(Exception):
                close()
