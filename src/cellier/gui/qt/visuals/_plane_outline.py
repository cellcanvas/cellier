"""The "Outline" line of a plane's row, shared by the two Qt plane controls.

A check box that draws the plane's outline and a button, filled with the
outline's colour, that opens a colour dialog (plane outline design,
section 7).  The width of the line is not here: it is set from code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cellier.gui._clipping_planes import OUTLINE_COLOR_TOOLTIP, OUTLINE_TOOLTIP

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class OutlineInputs:
    """The check box and the colour button of one plane's outline.

    Parameters
    ----------
    parent : QWidget
        The row's widget.
    on_enabled : Callable[[bool], None]
        Called when the box is ticked or cleared by the user.
    on_color : Callable[[str], None]
        Called with ``#rrggbb`` when the user chooses a colour.
    """

    def __init__(
        self,
        parent: Any,
        on_enabled: Callable[[bool], None],
        on_color: Callable[[str], None],
    ) -> None:
        from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QPushButton

        self._parent = parent
        self._on_color = on_color
        #: The colour shown, ``#rrggbb``.
        self.hex = "#ffffff"

        self.enabled = QCheckBox("Outline", parent)
        self.enabled.setToolTip(OUTLINE_TOOLTIP)
        self.enabled.toggled.connect(on_enabled)

        self.color = QPushButton(parent)
        self.color.setFixedSize(36, 18)
        self.color.setToolTip(OUTLINE_COLOR_TOOLTIP)
        self.color.clicked.connect(self.pick)

        #: The colour button and the space after it, for the row's grid.
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.color)
        self.layout.addStretch(1)

    def pick(self) -> None:
        """Open a colour dialog; a chosen colour is sent, a cancel is not."""
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import QColorDialog

        chosen = QColorDialog.getColor(
            QColor(self.hex), self._parent, "Choose the outline's colour"
        )
        if chosen.isValid():
            self._on_color(chosen.name())

    def show(self, row: Mapping[str, Any]) -> None:
        """Show a described row's outline; emits nothing."""
        self.enabled.blockSignals(True)
        try:
            self.enabled.setChecked(bool(row["outline"]["enabled"]))
        finally:
            self.enabled.blockSignals(False)
        self.hex = str(row["outline_hex"])
        # A style sheet here is on a button nothing else styles: it is the
        # only way to fill one with a colour on every platform style.
        self.color.setStyleSheet(
            f"background-color: {self.hex}; border: 1px solid #808080;"
        )
