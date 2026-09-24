"""Jupyter manager-rendered anywidget container.

Jupyter has no native multi-pane layout without a container that renders child
widget views.  To keep stock ipywidgets controls out, the container is itself a
tiny anywidget that asks the widget manager to mount its children (design doc
section 10).

This is the one place the anywidget path touches the ipywidgets layer
(``widget_manager.create_view``, ``ipywidgets.widget_serialization``,
``DOMWidget`` as the child type); marimo avoids it entirely via
``mo.vstack`` / ``mo.hstack``.
"""

from __future__ import annotations

from pathlib import Path

import anywidget
import ipywidgets
import traitlets

from cellier.gui.anywidget._teardown import close_aux_widgets

_STATIC = Path(__file__).parent / "static"


class AnywidgetBox(anywidget.AnyWidget):
    """A flexbox container that mounts child widget views via the manager.

    Parameters
    ----------
    children : list of ipywidgets.DOMWidget
        The child widgets to mount, in order.
    direction : "v" or "h"
        Stacking direction: ``"v"`` (column, default) or ``"h"`` (row).
    align : str
        Cross-axis alignment applied as CSS ``align-items`` (e.g. ``"center"``);
        empty string leaves the flexbox default (``stretch``).
    min_width : int
        When set (> 0), the box grows to fill available space but never
        narrower than this many pixels (``flex: 1 1 <min_width>px``); ``0``
        leaves the flexbox default (content-sized, no grow).
    gap : int
        Spacing between children in pixels.  Defaults to ``4``, tuned for
        macro layout blocks (canvas/dims/docks).  Pass a smaller value to
        tightly group sibling controls that used to live inside one widget
        (see ``LayoutHost.dock_panel``).
    padding : int
        Inner padding in pixels, on all four sides.  Defaults to ``0``.  Used
        by :class:`~cellier.convenience._hosts.JupyterHost` to keep the
        outermost box from touching the notebook cell / sidecar tab edges;
        nested boxes leave this at ``0`` so only the outer border shows.
    title : str
        A heading drawn above the children, naming what the box holds.  Empty
        (the default) draws nothing.  The anywidget answer to Qt's
        ``titled_group``: the render dock uses it to say whose settings these
        are, since "Outline" beside "Outlines" is not a distinction a reader
        should have to make.
    scroll : bool
        Scroll the children vertically within the height of the row the box
        sits in, instead of making that row as tall as the children.  For a
        side dock: the canvas column beside it sets the height, and a long
        controls column scrolls rather than stretching the output (or the
        sidecar) past the canvas.  The box keeps its content width, and never
        gets shorter than ``SCROLL_MIN_HEIGHT`` pixels.  Only meaningful for a
        box that is an item in a horizontal box.
    """

    #: The least height a scrolling box takes, so a small canvas does not
    #: squeeze a dock to a sliver; a shorter row grows to it.
    SCROLL_MIN_HEIGHT = 300

    _esm = _STATIC / "container.js"
    _css = _STATIC / "container.css"

    children = traitlets.List(traitlets.Instance(ipywidgets.DOMWidget)).tag(
        sync=True, **ipywidgets.widget_serialization
    )
    direction = traitlets.Unicode("v").tag(sync=True)
    align = traitlets.Unicode("").tag(sync=True)
    min_width = traitlets.Int(0).tag(sync=True)
    gap = traitlets.Int(4).tag(sync=True)
    padding = traitlets.Int(0).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    scroll = traitlets.Bool(False).tag(sync=True)
    scroll_min_height = traitlets.Int(SCROLL_MIN_HEIGHT).tag(sync=True)

    def close(self) -> None:
        """Close this box and every widget beneath it.

        A container is the only thing holding the composed tree, so closing it
        has to close the children too or the whole panel stays registered with
        ``ipywidgets`` (see ``cellier.gui.anywidget._teardown``).  ``children``
        is cleared first so the box does not keep them reachable afterwards.

        ``Widget.close()`` no-ops once the comm is gone, so a child that is
        also closed through another path -- a control tracked as a leaf by
        ``_RenderView`` -- is safe to reach twice.
        """
        for child in list(self.children):
            child.close()
        self.children = []
        close_aux_widgets(self)
        super().close()


def _refs_to_json(value: list, _obj: object) -> list[str]:
    """Serialize children as anywidget composition references."""
    return [f"anywidget:{child.model_id}" for child in value]


def _refs_from_json(value: object, _obj: object) -> object:
    """Incoming state is never written by ``slot.js``; pass it through."""
    return value


class AnywidgetSlot(anywidget.AnyWidget):
    """A column whose children can be replaced after it is displayed.

    ``AnywidgetBox`` mounts children through the Jupyter widget manager's
    ``create_view``, renders once, and does not exist on marimo.  This mounts
    them through anywidget's composition API (``host.getWidget``), which both
    Jupyter and marimo implement, and remounts whenever ``children`` changes.
    That is what lets a dock follow the viewer on either host.

    Children must be anywidgets, and must not be ``AnywidgetBox`` instances:
    a box nested in a slot on marimo would fall back to ``create_view``.

    Parameters
    ----------
    children : list of anywidget.AnyWidget
        The widgets to show, top to bottom.  Assign a new list to replace them;
        the slot does not close the widgets it stops showing.
    gap : int
        Spacing between children in pixels.
    title : str
        A heading drawn above the children.  Empty draws nothing.
    """

    _esm = _STATIC / "slot.js"
    _css = _STATIC / "container.css"

    children = traitlets.List(traitlets.Instance(anywidget.AnyWidget)).tag(
        sync=True, to_json=_refs_to_json, from_json=_refs_from_json
    )
    gap = traitlets.Int(4).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)

    def close(self) -> None:
        """Close this slot and the widgets it currently shows."""
        for child in list(self.children):
            child.close()
        self.children = []
        close_aux_widgets(self)
        super().close()
