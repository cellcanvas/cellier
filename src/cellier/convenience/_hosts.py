"""Host-aware placement for the anywidget front-end (the ``LayoutHost`` seam).

The leaves are host-uniform (the per-control appearance anywidgets, the
``rendercanvas.anywidget`` canvas, the toggle).  Only composition and
presentation differ between Jupyter and marimo, isolated behind one injected
object (design doc section 10).

``gui="anywidget"`` selects the toolkit; the *host* (Jupyter vs marimo) is
detected by default with an explicit ``host=`` override via
:func:`resolve_host`.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from cellier.convenience.layout._spec import DOCK_MIN_WIDTH

if TYPE_CHECKING:
    from collections.abc import Sequence


class LiveSlot(Protocol):
    """A dock region whose contents can be replaced after it is presented.

    Everything else a host composes is fixed once presented -- on marimo
    literally, since ``mo.vstack`` output is static.  A controls dock has to
    follow its viewer, so it composes into one of these instead.
    """

    #: The host item to place in the layout, once.
    root: object

    def set(self, widgets: Sequence[object], *, placeholder: str | None = None) -> None:
        """Show *widgets* top to bottom, replacing what was shown.

        *widgets* are backend widgets (not host leaves); the slot composes them
        itself.  With no widgets, *placeholder* is shown instead.  The slot does
        not close the widgets it stops showing.
        """
        ...

    def close(self) -> None:
        """Release the slot."""
        ...


@runtime_checkable
class LayoutHost(Protocol):
    """Composition + presentation seam for one front end.

    A *host* answers "how are widgets composed and presented"; the
    :class:`~cellier.convenience._backend.GuiBackend` it carries answers "which
    widget class serves this control".  Splitting the two is what lets one
    anywidget backend serve both the Jupyter and marimo hosts, and lets the
    layout walk be written once.
    """

    #: The widget classes this host composes.
    backend: object

    def leaf(self, widget: object) -> object:
        """Wrap one anywidget leaf for this host."""
        ...

    def stack(
        self,
        items: Sequence[object],
        *,
        direction: str = "v",
        align: str | None = None,
        min_width: int | None = None,
        gap: int | None = None,
        title: str | None = None,
    ) -> object:
        """Stack *items* vertically (``"v"``) or horizontally (``"h"``).

        *align* sets the cross-axis alignment of the items (e.g. ``"center"``
        to centre a fixed-width canvas over a wider control panel); ``None``
        leaves the host default.

        *min_width* makes the stack grow to fill available width but never
        narrower than *min_width* pixels; ``None`` leaves the host default
        (content-sized, no grow).

        *gap* sets the spacing between items in pixels; ``None`` leaves the
        host default (tuned for macro layout blocks like canvas/dims/docks).
        Pass a small explicit value to tightly group sibling controls that
        used to live inside one widget (see
        :meth:`dock_panel`).

        *title* draws a heading above the items, naming what the stack holds;
        ``None`` draws none.  The anywidget answer to the Qt renderer's
        ``titled_group``, so a dock can say whose settings it carries on
        either toolkit.
        """
        ...

    def dock_panel(
        self, widgets: Sequence[object], *, title: str | None = None
    ) -> object:
        """Compose *widgets* into a column shaped for a dock area.

        Separate from :meth:`stack` because a dock column carries per-toolkit
        sizing a plain stack must not: Qt wants a minimum width and a trailing
        stretch so the controls sit at the top of a resizable dock, and
        anywidget wants neither.  *title* names the whole column when the dock
        needs to say what its contents are scoped to.
        """
        ...

    def live_slot(self) -> LiveSlot:
        """Return a dock region whose contents can be replaced later."""
        ...

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Arrange *rows* (a list of rows of items) as a grid.

        A ``None`` in a row is an empty cell and must hold its column: the
        items after it keep their position rather than shifting left.  Both
        hosts substitute an invisible placeholder, which is the cheapest way
        to say "nothing here" in a flexbox row.
        """
        ...

    def assemble(
        self,
        center: object,
        docks: dict,
        closeables: list,
        *,
        dock_min_widths: dict | None = None,
    ) -> object:
        """Compose the center and the four docks into one root.

        The genuinely different half of a layout: Qt builds a ``QMainWindow``
        with real ``QDockWidget``s, while anywidget has no dock concept and
        hand-assembles ``[left | center | right]`` inside
        ``[top / middle / bottom]``.

        *docks* is keyed ``"left"``, ``"right"``, ``"top"``, ``"bottom"``,
        with ``None`` for a dock that built nothing.  *closeables* is handed
        over so a host whose root owns teardown -- Qt's window does -- can take
        the list with it.  *dock_min_widths* maps ``"left"`` / ``"right"`` to
        the narrowest that dock may be, in logical pixels; a missing or
        ``None`` entry uses :data:`~cellier.convenience.layout._spec.DOCK_MIN_WIDTH`
        (260 px) on every host.  It is a floor: a dock whose controls need more
        is as wide as they need.  Sizing is placement, so it happens here
        rather than in ``dock_panel``.

        The left and right docks scroll vertically on every host, so a dock
        is never what sets the layout's height: on Qt the window's height
        does (the screen's, at most), on anywidget the center column's.
        """
        ...

    def present(self, root: object) -> object | None:
        """Render *root*, or return it for ``display()`` to yield.

        Two host conventions, distinguished by the return value:

        * Imperative hosts (Jupyter) render *root* as a side effect (e.g.
          ``IPython.display.display``) and return ``None``; ``display()`` then
          returns an inert handle so the cell shows a single copy.
        * Return-value hosts (marimo) render the cell's *returned* value, so
          ``present`` returns *root* and ``display()`` yields it as the cell
          output (an imperative ``mo.output.replace`` is overridden by the
          cell's last expression, so it cannot be used here).
        """
        ...


class QtLayoutHost:
    """Qt host -- composition with ``QBoxLayout`` / ``QGridLayout``.

    The Qt implementation of :class:`LayoutHost`, which is what lets the layout
    walk in :mod:`cellier.convenience.layout._walk` be written once rather than
    once per toolkit.  ``present`` shows the window; the blocking event loop is
    :func:`cellier.convenience.launch`'s job, not the host's.
    """

    #: Spacing between stacked items, matching what the Qt renderer used.
    DEFAULT_GAP = 4

    def __init__(self, backend=None) -> None:
        from cellier.convenience._backend import QT_BACKEND

        self.backend = QT_BACKEND if backend is None else backend

    def leaf(self, widget: object) -> object:
        """Unwrap a leaf to the ``QWidget`` it exposes."""
        return widget.widget if hasattr(widget, "widget") else widget

    def stack(
        self,
        items: Sequence[object],
        *,
        direction: str = "v",
        align: str | None = None,
        min_width: int | None = None,
        gap: int | None = None,
        title: str | None = None,
    ) -> object:
        """Compose into a ``QWidget`` with a horizontal or vertical box layout.

        *align* is accepted for interface parity and ignored -- Qt expresses
        cross-axis alignment through size policies rather than a layout flag.
        """
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

        container = QWidget()
        box = QHBoxLayout(container) if direction == "h" else QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(self.DEFAULT_GAP if gap is None else gap)
        for item in items:
            box.addWidget(item)
        if min_width:
            container.setMinimumWidth(min_width)
        if title:
            # A plain centred heading, not a ``titled_group``: ``stack`` names
            # a block of content (an ortho panel), where ``dock_panel`` names a
            # whole dock and wants the group box.  Matches the header
            # ``_build_qt_ortho_grid`` drew by hand, and matches what the
            # anywidget hosts already draw for the same call.
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

            titled = QWidget()
            outer = QVBoxLayout(titled)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            label = QLabel(title)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-weight: bold; font-size: 11px; padding: 2px;")
            outer.addWidget(label)
            outer.addWidget(container, stretch=1)
            return titled
        return container

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Arrange rows in a ``QGridLayout``, ``None`` leaving a cell empty."""
        from qtpy.QtWidgets import QGridLayout, QWidget

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(self.DEFAULT_GAP)
        for row_index, row in enumerate(rows):
            for column_index, cell in enumerate(row):
                if cell is not None:
                    grid.addWidget(cell, row_index, column_index)
        return container

    def dock_panel(
        self, widgets: Sequence[object], *, title: str | None = None
    ) -> object:
        """Stack dock contents into a column sized for a dock area."""
        from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

        from cellier.convenience.layout._shared import APPEARANCE_DOCK_GAP_PX

        inner = QWidget()
        box = QVBoxLayout(inner)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(APPEARANCE_DOCK_GAP_PX)
        for widget in widgets:
            box.addWidget(widget)
        box.addStretch()

        if title is not None:
            from cellier.gui.qt.visuals._chrome import titled_group

            container = titled_group(title, inner)
        else:
            container = inner
        container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        # No minimum width here: the floor belongs to the whole dock (see
        # ``assemble``), where a caller's larger ``*_dock_min_width``
        # replaces the default.
        return container

    def live_slot(self) -> _QtLiveSlot:
        """A ``QWidget`` whose content column ``set`` replaces."""
        return _QtLiveSlot(self)

    #: The narrowest a dock may be when the layout does not say; the shared
    #: :data:`~cellier.convenience.layout._spec.DOCK_MIN_WIDTH`, kept here
    #: under its old name.
    DEFAULT_DOCK_MIN_WIDTH = DOCK_MIN_WIDTH

    def assemble(
        self,
        center: object,
        docks: dict,
        closeables: list,
        *,
        dock_min_widths: dict | None = None,
    ) -> object:
        """Build the ``QMainWindow``: center plus a ``QDockWidget`` per area.

        Each dock's content is floored at its ``dock_min_widths`` entry, or
        :attr:`DEFAULT_DOCK_MIN_WIDTH`.  A minimum rather than a fixed width:
        the dock separator can still drag it wider, never narrower, and a
        left or right dock whose controls need more than the floor starts
        as wide as they need (see ``_scroll_dock_widget``).

        The left and right docks scroll vertically, so a dock is never what
        sets the window's height: that is the screen's (see
        ``_scroll_dock_widget``).
        """
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow

        from cellier.convenience.layout._qt_renderer import (
            _scroll_dock_widget,
            _wrap_dock_widget,
            make_window,
        )

        areas = {
            "left": Qt.DockWidgetArea.LeftDockWidgetArea,
            "right": Qt.DockWidgetArea.RightDockWidgetArea,
            "top": Qt.DockWidgetArea.TopDockWidgetArea,
            "bottom": Qt.DockWidgetArea.BottomDockWidgetArea,
        }
        window = make_window(QMainWindow)()
        # The window owns teardown on this toolkit, so it takes the list the
        # walk filled rather than keeping one of its own.
        window._cellier_closeables = closeables
        window.setCentralWidget(center)

        for name, area in areas.items():
            widget = docks.get(name)
            if widget is None:
                continue
            dock = QDockWidget(name.capitalize(), window)
            content = _wrap_dock_widget(widget, name)
            content.setMinimumWidth(
                (dock_min_widths or {}).get(name) or self.DEFAULT_DOCK_MIN_WIDTH
            )
            if name in ("left", "right"):
                content = _scroll_dock_widget(content)
            dock.setWidget(content)
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
            window.addDockWidget(area, dock)

        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            window.resize(
                min(int(available.width() * 2 / 3), 1600),
                min(int(available.height() * 2 / 3), 1000),
            )
        return window

    def present(self, root: object) -> None:
        """Show the window.  Running the loop is ``launch``'s job."""
        root.show()
        return None


class _QtLiveSlot:
    """``LiveSlot`` on Qt: a container holding one ``dock_panel`` at a time."""

    def __init__(self, host: QtLayoutHost) -> None:
        from qtpy.QtWidgets import QVBoxLayout, QWidget

        self._host = host
        self.root = QWidget()
        self._box = QVBoxLayout(self.root)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._content = None
        self._leaves: list = []

    def set(self, widgets: Sequence[object], *, placeholder: str | None = None) -> None:
        leaves = [self._host.leaf(widget) for widget in widgets]
        shown = list(leaves)
        if not leaves and placeholder:
            from qtpy.QtWidgets import QLabel

            leaves = [QLabel(placeholder)]
        # Built before the old content goes: ``addWidget`` reparents a widget
        # carried over from the old column (the selector, or controls kept
        # across a relabel), so deleting the old column cannot take it along.
        content = self._host.dock_panel(leaves)
        self._box.addWidget(content)
        # A widget shown before and not now is still owned by its caller (a
        # dock keeps every target's controls built), so it is detached rather
        # than deleted with the old column.  Parentless, Qt hides it.
        for leaf in self._leaves:
            if not any(leaf is kept for kept in shown):
                leaf.setParent(None)
        self._leaves = shown
        old, self._content = self._content, content
        if old is not None:
            self._box.removeWidget(old)
            old.setParent(None)
            old.deleteLater()

    def close(self) -> None:
        """Nothing to do: the window that holds the slot deletes it."""


class _AnywidgetLiveSlot:
    """``LiveSlot`` on the anywidget hosts: an ``AnywidgetSlot``, leafed once.

    ``AnywidgetSlot`` mounts its children through anywidget's composition API,
    which Jupyter and marimo both implement, so the same slot serves both.
    """

    def __init__(self, host: _AnywidgetDockPanel) -> None:
        from cellier.convenience.layout._shared import APPEARANCE_DOCK_GAP_PX
        from cellier.gui.anywidget._container import AnywidgetSlot

        self._slot = AnywidgetSlot(gap=APPEARANCE_DOCK_GAP_PX)
        self.root = host.leaf(self._slot)

    @property
    def slot(self) -> object:
        """The ``AnywidgetSlot`` behind ``root``."""
        return self._slot

    def set(self, widgets: Sequence[object], *, placeholder: str | None = None) -> None:
        children = [getattr(widget, "widget", widget) for widget in widgets]
        with self._slot.hold_sync():
            self._slot.title = "" if children else (placeholder or "")
            self._slot.children = children

    def close(self) -> None:
        self._slot.close()


class _AnywidgetDockPanel:
    """``dock_panel`` for the anywidget hosts, which compose it from ``stack``.

    Unlike Qt there is no sizing to add: the column is a stack with the shared
    gap, and a single widget needs no container at all.
    """

    def assemble(
        self,
        center: object,
        docks: dict,
        closeables: list,
        *,
        dock_min_widths: dict | None = None,
    ) -> object:
        """Hand-assemble ``[left | center | right]`` inside the outer column.

        anywidget has no dock concept, so the arrangement is built from
        stacks.  *closeables* is unused here: on this toolkit the caller's
        ``_RenderView`` owns teardown, not the root.  A side dock is floored
        at its ``dock_min_widths`` entry, or at
        :data:`~cellier.convenience.layout._spec.DOCK_MIN_WIDTH` when it has
        none; there is no splitter to drag it wider, so above the floor it is
        as wide as its content.

        The left and right docks scroll vertically within the height of the
        center column (the canvas and its sliders), so a long dock never
        makes the output taller than the canvas needs.
        """
        docks = dict(docks)
        widths = dock_min_widths or {}
        # Side docks scroll within the center's height rather than setting
        # it, so a long controls column never pushes the canvas row past
        # what the canvas needs.  The dock width floor rides on the same
        # wrapper.
        for side in ("left", "right"):
            if docks.get(side) is not None:
                docks[side] = self._scroll_dock(
                    docks[side], widths.get(side) or DOCK_MIN_WIDTH
                )
        middle_items = [
            item
            for item in (docks.get("left"), center, docks.get("right"))
            if item is not None
        ]
        middle = (
            # "stretch": a scrolling dock fills the height the center sets.
            self.stack(middle_items, direction="h", align="stretch")
            if len(middle_items) > 1
            else middle_items[0]
        )
        outer_items = [
            item
            for item in (docks.get("top"), middle, docks.get("bottom"))
            if item is not None
        ]
        if len(outer_items) == 1:
            return outer_items[0]
        # No explicit align: the default cross-axis "stretch" is what lets the
        # tree fill the notebook cell / sidecar tab width.
        return self.stack(outer_items, direction="v")

    def dock_panel(self, widgets, *, title: str | None = None) -> object:
        from cellier.convenience.layout._shared import APPEARANCE_DOCK_GAP_PX

        widgets = list(widgets)
        if not widgets:
            return None
        if len(widgets) == 1 and title is None:
            return widgets[0]
        return self.stack(
            widgets, direction="v", gap=APPEARANCE_DOCK_GAP_PX, title=title
        )

    def live_slot(self) -> _AnywidgetLiveSlot:
        """An ``AnywidgetSlot`` whose children ``set`` replaces."""
        return _AnywidgetLiveSlot(self)


class MarimoHost(_AnywidgetDockPanel):
    """marimo host -- native anywidget + layout primitives."""

    def __init__(self, backend=None) -> None:
        import marimo as mo

        from cellier.convenience._backend import ANYWIDGET_BACKEND

        self._mo = mo
        self.backend = ANYWIDGET_BACKEND if backend is None else backend

    def leaf(self, widget: object) -> object:
        """Wrap a leaf with ``marimo.ui.anywidget``."""
        return self._mo.ui.anywidget(widget)

    # marimo's own vstack/hstack ``gap=`` is in rem, not pixels; this converts
    # our pixel-based LayoutHost.stack(gap=...) contract to marimo's unit
    # assuming the standard 16px-per-rem base.
    _REM_PX = 16

    def stack(
        self,
        items: Sequence[object],
        *,
        direction: str = "v",
        align: str | None = None,
        min_width: int | None = None,
        gap: int | None = None,
        title: str | None = None,
    ) -> object:
        """Stack with ``marimo.vstack`` / ``marimo.hstack``.

        *min_width* is accepted for interface parity with :class:`JupyterHost`
        but ignored -- marimo has its own layout/width primitives.

        *gap* (pixels, matching :class:`JupyterHost`) is converted to
        marimo's own rem-based ``gap=`` only when given; omitted, marimo's
        own default is used unchanged.  ``mo.vstack``/``mo.hstack`` default to
        a non-trivial gap meant for spacing distinct layout blocks apart,
        which is too loose for grouping sibling controls that used to live
        inside one widget.
        """
        stacker = self._mo.vstack if direction == "v" else self._mo.hstack
        kwargs = {} if gap is None else {"gap": gap / self._REM_PX}
        stacked = stacker(list(items), align=align, **kwargs)
        if not title:
            return stacked
        # marimo has no titled container, so the heading is markdown above the
        # stack -- the same two elements the Jupyter box draws, composed with
        # marimo's own primitives.
        return self._mo.vstack([self._mo.md(f"**{title}**"), stacked])

    def _scroll_dock(self, item: object, width: int | None) -> object:
        """Scroll a side dock within the row's height, floored at *width*.

        Two styled wrappers carrying the rules ``container.css`` gives the
        Jupyter box: the outer one is stretched to the row's height and never
        shorter than ``AnywidgetBox.SCROLL_MIN_HEIGHT``; the inner one takes
        no height of its own (``height: 0``), fills the outer
        (``min-height: 100%``) and scrolls.
        """
        from cellier.gui.anywidget import AnywidgetBox

        inner_style = {
            "height": "0",
            "min-height": "100%",
            "overflow-y": "auto",
            "overflow-x": "hidden",
            "overscroll-behavior": "contain",
        }
        if width:
            inner_style["min-width"] = f"{width}px"
        outer_style = {
            "align-self": "stretch",
            "min-height": f"{AnywidgetBox.SCROLL_MIN_HEIGHT}px",
        }
        return self._mo.style(self._mo.style(item, inner_style), outer_style)

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Arrange rows with nested ``vstack`` / ``hstack``.

        An empty cell becomes empty markdown, which occupies its column
        without drawing anything.
        """
        return self._mo.vstack(
            [
                self._mo.hstack(
                    [self._mo.md("") if item is None else item for item in row]
                )
                for row in rows
            ]
        )

    def present(self, root: object) -> object | None:
        """Return *root* so ``display()`` yields it as the cell output.

        marimo renders a cell's *last expression*, which overrides an
        imperative ``mo.output.replace``; so instead of rendering here, we hand
        *root* back and let ``display()`` return it.
        """
        return root


class JupyterHost(_AnywidgetDockPanel):
    """Jupyter host -- manager-rendered anywidget container (``AnywidgetBox``)."""

    def __init__(self, backend=None) -> None:
        from cellier.convenience._backend import ANYWIDGET_BACKEND

        # Widgets rendered by ``present``; see ``close_presented``.
        self._presented: list[object] = []
        self.backend = ANYWIDGET_BACKEND if backend is None else backend

    def leaf(self, widget: object) -> object:
        """A ``DOMWidget`` is directly displayable; pass it through."""
        return widget

    def stack(
        self,
        items: Sequence[object],
        *,
        direction: str = "v",
        align: str | None = None,
        min_width: int | None = None,
        gap: int | None = None,
        title: str | None = None,
    ) -> object:
        """Compose into an ``AnywidgetBox`` flexbox."""
        from cellier.gui.anywidget import AnywidgetBox

        kwargs = {} if gap is None else {"gap": gap}
        return AnywidgetBox(
            children=list(items),
            direction=direction,
            align=align or "",
            min_width=min_width or 0,
            title=title or "",
            **kwargs,
        )

    def _scroll_dock(self, item: object, width: int | None) -> object:
        """Scroll a side dock within the row's height, floored at *width*."""
        from cellier.gui.anywidget import AnywidgetBox

        return AnywidgetBox(children=[item], min_width=width or 0, scroll=True)

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Compose rows of ``AnywidgetBox`` (horizontal) inside an outer one.

        An empty cell becomes an empty ``AnywidgetBox``: ``children`` only
        accepts ``DOMWidget``s, so the hole has to be a widget, and an
        childless box renders as nothing while still taking its place in the
        row.
        """
        from cellier.gui.anywidget import AnywidgetBox

        return AnywidgetBox(
            children=[
                AnywidgetBox(
                    children=[AnywidgetBox() if item is None else item for item in row],
                    direction="h",
                )
                for row in rows
            ],
            direction="v",
        )

    #: Outer padding (pixels) so the composed tree doesn't touch the notebook
    #: cell / sidecar tab edges; applies to both ``sidecar=True`` and
    #: ``sidecar=False`` since both flow through this one ``present()``.
    _OUTER_PADDING = 12

    def present(self, root: object) -> object | None:
        """Render *root* imperatively via ``IPython.display.display``.

        Wraps *root* in an outer ``AnywidgetBox`` with a small padding so the
        canvas/docks don't touch the cell (or sidecar tab) boundary; nested
        boxes built by cellier.convenience.layout._anywidget_renderer.render_anywidget
        keep ``padding=0``, so this is the only border added.

        Returns ``None`` so ``display()`` yields an inert handle (the viewer is
        already shown), avoiding a duplicate copy from the cell's return value.
        """
        from IPython.display import display as ipy_display

        from cellier.gui.anywidget import AnywidgetBox

        wrapped = AnywidgetBox(children=[root], padding=self._OUTER_PADDING)
        # Kept so it can be closed later.  Because this host renders as a side
        # effect and returns ``None``, the caller never sees this wrapper, so
        # without a reference here nothing could ever release it -- and it is
        # an ``ipywidgets`` widget like any other (see
        # ``cellier.gui.anywidget._teardown``).
        self._presented.append(wrapped)
        ipy_display(wrapped)
        return None

    def close_presented(self) -> None:
        """Close everything this host has rendered.

        Optional part of the ``LayoutHost`` contract, called by
        ``DisplayHandle.close``.  A host that hands its root back to the caller
        (``MarimoHost``) does not need it: whoever owns the root closes it.
        """
        for widget in self._presented:
            close = getattr(widget, "close", None)
            if close is not None:
                with suppress(Exception):
                    close()
        self._presented.clear()


def _marimo_running() -> bool:
    try:
        import marimo as mo

        return bool(mo.running_in_notebook())
    except Exception:
        return False


def _ipython_running() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def resolve_host(host: str | None = None) -> LayoutHost:
    """Resolve the anywidget :class:`LayoutHost`.

    The explicit *host* override wins; otherwise the host is detected (marimo
    first, then IPython / Jupyter).

    Parameters
    ----------
    host : "marimo", "jupyter", or None
        Explicit host override, or ``None`` to auto-detect.

    Returns
    -------
    LayoutHost

    Raises
    ------
    RuntimeError
        If no host is given and none can be detected.
    """
    if host == "marimo" or (host is None and _marimo_running()):
        return MarimoHost()
    if host == "jupyter" or (host is None and _ipython_running()):
        return JupyterHost()
    raise RuntimeError(
        "No anywidget host detected; pass host='jupyter' or host='marimo'."
    )
