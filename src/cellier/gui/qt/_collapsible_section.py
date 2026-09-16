"""A titled section a controls dock can collapse, one per configured visual."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class QtCollapsibleSection:
    """A ``QCollapsible`` holding a column of controls.

    Local UI state rather than a model control: it puts nothing on the bus and
    is never passed to ``connect_widget``.  A controls dock presented as
    collapsible sections builds one per configured visual.

    Parameters
    ----------
    title :
        The text on the section's toggle, e.g. the visual's name.
    widgets :
        The controls to stack inside, each exposing ``widget``.  They are
        reparented into the section.
    expanded :
        Whether the section starts open.
    gap :
        Spacing between the stacked controls, in pixels.
    parent :
        Optional Qt parent for the section.
    """

    def __init__(
        self,
        title: str,
        widgets: Sequence[object] = (),
        *,
        expanded: bool = False,
        gap: int = 6,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import QVBoxLayout, QWidget
        from superqt import QCollapsible

        self._content = QWidget()
        layout = QVBoxLayout(self._content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(gap)
        for item in widgets:
            layout.addWidget(getattr(item, "widget", item))

        self._collapsible = QCollapsible(title, parent)
        self._collapsible.addWidget(self._content)
        self.set_expanded(expanded)

    @property
    def widget(self):
        """The collapsible to embed."""
        return self._collapsible

    @property
    def content(self):
        """The column holding the controls."""
        return self._content

    @property
    def title(self) -> str:
        """The text on the section's toggle."""
        return str(self._collapsible.text())

    @property
    def expanded(self) -> bool:
        """Whether the section is open."""
        return bool(self._collapsible.isExpanded())

    def set_title(self, title: str) -> None:
        """Retitle the section."""
        self._collapsible.setText(title)

    def set_expanded(self, expanded: bool) -> None:
        """Open or close the section, without animating."""
        if expanded:
            self._collapsible.expand(animate=False)
        else:
            self._collapsible.collapse(animate=False)

    def close(self) -> None:
        """Release the section."""
        self._collapsible.setParent(None)
        self._collapsible.deleteLater()
