"""How a Qt appearance control shows its own name.

A control names itself; whatever lays it out only stacks.  The two shapes a
name takes are here so that both the single-field base
(``_base.QtAppearanceField``) and the hand-written multi-control widgets draw
them the same way, and so the anywidget front end has one thing to mirror
rather than five:

* `labelled_row` -- a single control, named by a label beside it.  The
  Qt twin of the anywidget ``.cellier-app-row``.
* `titled_group` -- several rows that only mean something together (the
  bounding box, the volume-render block), named by a frame around them.  Qt
  has a first-class widget for this and the anywidget side draws a heading;
  that is the one place the two front ends deliberately differ, because a
  ``QGroupBox`` frame is the Qt idiom for exactly this.

Nothing here talks to the bus or knows what a visual is.
"""

from __future__ import annotations

LABELLED_ROW_OBJECT_NAME = "cellier-app-row"
"""``objectName`` stamped on every row `labelled_row` builds.

The Qt twin of the anywidget ``.cellier-app-row`` class.  It is what lets a
caller -- notably the visual-acceptance helper
``tests/convenience/_qt_acceptance.control_labels`` -- tell "this widget names
itself" from "this widget is named by a frame around it", without importing a
Qt class or matching on label text.

The row's **first layout item is its label**; that is the invariant readers
rely on to recover a control's name from the widget tree alone.
"""

_LABEL_MIN_EM = 5.5
"""Minimum label-column width, in em.

Mirrors ``.cellier-app-label { min-width: 5.5em }``.  Each control is its own
top-level widget in the dock's column, so nothing aligns them for free -- a
``QFormLayout`` per widget would align a widget with itself and no more.  A
shared minimum width in em is what actually lines up "Size" with "Edge
thickness space" down a panel, and it tracks the user's font the way the CSS
does.
"""

_ROW_GAP_PX = 6
"""Gap between a label and its control, matching ``.cellier-app-row``'s."""


def labelled_row(label_text: str, control, parent=None):
    """Return a ``<label> <control>`` row naming *control* as *label_text*.

    Parameters
    ----------
    label_text :
        The control's name, e.g. ``"Wireframe thickness"``.
    control :
        The Qt widget to name.  It is reparented into the row.
    parent :
        Optional Qt parent for the row.

    Returns
    -------
    QWidget
        The row, with ``objectName`` ``"cellier-app-row"`` (for styling).
    """
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QFontInfo
    from qtpy.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

    row = QWidget(parent)
    row.setObjectName(LABELLED_ROW_OBJECT_NAME)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(_ROW_GAP_PX)

    label = QLabel(label_text, row)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    # QFontInfo resolves the font actually in use, so pixelSize() is 1em by
    # definition; fall back to the line height if it is unavailable.
    em = QFontInfo(label.font()).pixelSize()
    if em <= 0:
        em = label.fontMetrics().height()
    label.setMinimumWidth(round(_LABEL_MIN_EM * em))
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
    layout.addWidget(label)

    # addWidget reparents, which is what moves the control off *parent* and
    # into the row.
    layout.addWidget(control, stretch=1)
    return row


def titled_group(title: str, content, parent=None):
    """Return a ``QGroupBox`` titled *title* wrapping *content*.

    For a widget whose rows only mean something together -- ``Show`` / ``Width``
    / ``Color`` is three anonymous rows until a frame says "Bounding box".  A
    single control should use `labelled_row` instead.

    Parameters
    ----------
    title :
        The group's name, e.g. ``"Bounding box"``.
    content :
        The Qt widget holding the rows.  It is reparented into the group.
    parent :
        Optional Qt parent for the group.

    Returns
    -------
    QGroupBox
    """
    from qtpy.QtWidgets import QGroupBox, QVBoxLayout

    group = QGroupBox(title, parent)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(12, 4, 12, 4)
    layout.addWidget(content)
    return group
