"""Qt visual-acceptance helpers (``plans/convenience_cleanup.md`` section 6.3).

``QWidget.grab()`` works under ``QT_QPA_PLATFORM=offscreen`` and returns real
pixels, so a rendered Qt dock can be checked by test rather than by eye.  These
helpers turn "run the dock once and look at it" into two assertions:

* :func:`control_labels` -- what the panel calls each control, **in order**, so
  a stage that adds, renames or reorders one shows up as a diff;
* :func:`assert_panel_renders` -- that the panel actually paints something, so
  a panel of correctly-titled but empty controls fails.

Import from a test module::

    from tests.convenience._qt_acceptance import assert_panel_renders, control_labels
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QLayout, QWidget

# Sampling stride for the rendered pixmap.  The panel is a few hundred pixels
# on a side, so every third pixel is plenty to tell "painted" from "blank"
# while keeping the scan cheap.
_SAMPLE_STRIDE = 3

# A blank panel is one flat background colour (plus, at most, antialiasing on
# a border).  Anything that drew a label, a border and a control clears this
# comfortably -- a fully populated appearance dock scores ~190.
_MIN_DISTINCT_COLORS = 8


def control_labels(container: QWidget) -> list[str]:
    """Return what *container* calls each control it holds, in layout order.

    The toolkit-neutral question "which controls does this panel contain, and
    what does it call them?" has two Qt answers, because a control is named in
    one of two ways (``plans/label_ownership_unification.md``):

    * a **single-control widget** carries its own label row, stamped with
      ``LABELLED_ROW_OBJECT_NAME``.  The row's first item is that label, and
      the walk does not descend past it -- a slider's own value readout and a
      colour picker's ``Alpha:`` are parts of the control, not names for it.
    * a **multi-row widget** (the AABB and volume-render groups) carries a
      ``QGroupBox``, whose title names the group.  The walk *does* descend
      into it, so a group holding labelled rows reports both.
    * a **static display block** (dataset info) carries a ``QCollapsible``,
      whose header names it.  The walk does *not* descend: the rows inside are
      values on display, not controls with names of their own.

    Anything else is scaffolding -- containers, stretches, the sub-labels
    inside a composite -- and contributes nothing.
    """
    from PySide6.QtWidgets import QGroupBox, QLabel
    from superqt import QCollapsible

    from cellier.gui.qt.visuals._chrome import LABELLED_ROW_OBJECT_NAME

    labels: list[str] = []

    def _row_label(row: QWidget) -> str:
        layout = row.layout()
        if layout is None or layout.count() == 0:
            return ""
        first = layout.itemAt(0).widget()
        return first.text() if isinstance(first, QLabel) else ""

    def _walk(layout: QLayout | None) -> None:
        if layout is None:
            return
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            if widget is None:
                _walk(item.layout())
                continue
            if widget.objectName() == LABELLED_ROW_OBJECT_NAME:
                labels.append(_row_label(widget))
                continue
            if isinstance(widget, QCollapsible):
                labels.append(widget.text())
                continue
            if isinstance(widget, QGroupBox):
                labels.append(widget.title())
            _walk(widget.layout())

    _walk(container.layout())
    return labels


def control_labels_anywidget(widgets: list) -> list[str]:
    """Return what each built anywidget control calls itself, in order.

    The anywidget twin of :func:`control_labels`, and the reason the two front
    ends can be compared at all: both sides answer the same question about the
    same controls, so ``control_labels(qt_panel) == control_labels_anywidget(
    anywidget_widgets)`` is a real cross-toolkit assertion rather than two
    lists that happen to be written down side by side.

    A multi-row widget names itself with ``title``; a single-control widget
    with ``label``.  A widget carrying neither is one that cannot name itself,
    which is the state this plan removed -- it reports ``""`` rather than
    being skipped, so a regression shows up as a diff and not as a shorter
    list.
    """
    return [
        str(getattr(widget, "title", "") or getattr(widget, "label", ""))
        for widget in widgets
    ]


def panel_distinct_colors(container: QWidget) -> int:
    """Grab *container* offscreen and count the distinct sampled pixel colours.

    ``grab()`` renders through the real paint pipeline even on the offscreen
    platform plugin, so this reports what the panel would actually look like.
    """
    container.resize(container.sizeHint())
    pixmap = container.grab()
    assert not pixmap.isNull(), "QWidget.grab() returned a null pixmap"

    image = pixmap.toImage()
    colors = {
        image.pixel(x, y)
        for y in range(0, image.height(), _SAMPLE_STRIDE)
        for x in range(0, image.width(), _SAMPLE_STRIDE)
    }
    return len(colors)


def assert_panel_renders(container: QWidget | None) -> None:
    """Assert *container* exists, has a non-zero size, and paints content."""
    assert container is not None, "expected a rendered panel, got None"

    size = container.sizeHint()
    assert size.width() > 0 and size.height() > 0, f"panel has empty size {size}"

    n_colors = panel_distinct_colors(container)
    assert n_colors >= _MIN_DISTINCT_COLORS, (
        f"panel painted only {n_colors} distinct colours "
        f"(< {_MIN_DISTINCT_COLORS}); it looks blank"
    )


def qt_control_name(native: QWidget) -> str:
    """What one built Qt control calls itself.

    A built Qt control has no ``title`` or ``label`` attribute: its name is
    on its own chrome -- the labelled row, the group box or the collapsible.
    """
    from qtpy.QtWidgets import QGroupBox, QLabel
    from superqt import QCollapsible

    from cellier.gui.qt.visuals._chrome import LABELLED_ROW_OBJECT_NAME

    if native.objectName() == LABELLED_ROW_OBJECT_NAME:
        first = native.layout().itemAt(0).widget()
        return first.text() if isinstance(first, QLabel) else ""
    if isinstance(native, QCollapsible):
        return native.text()
    if isinstance(native, QGroupBox):
        return native.title()
    return (control_labels(native) or [""])[0]


def qt_stacking(root: QWidget, widgets: list) -> list[int]:
    """Indices of *widgets* sorted by where each sits in *root*, top first.

    ``list(range(len(widgets)))`` when the dock stacks its controls in the
    order it built them.
    """
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QApplication

    root.resize(400, 4000)
    root.show()
    QApplication.instance().processEvents()
    tops = []
    for index, widget in enumerate(widgets):
        native = getattr(widget, "widget", widget)
        assert root.isAncestorOf(native), f"control {index} is not in the dock"
        tops.append((native.mapTo(root, QPoint(0, 0)).y(), index))
    return [index for _top, index in sorted(tops)]


def anywidget_stacking(dock: object, root: object, widgets: list) -> list[int]:
    """Indices of *widgets* in the order of the slot's (or section's) children."""
    natives = [getattr(widget, "widget", widget) for widget in widgets]
    holder = dock.sections[0].widget if dock.sections else root
    return [
        next(index for index, native in enumerate(natives) if native is child)
        for child in holder.children
    ]
