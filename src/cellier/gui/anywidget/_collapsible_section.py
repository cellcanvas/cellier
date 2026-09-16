"""A titled section a controls dock can collapse, one per configured visual."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import traitlets

from cellier.gui.anywidget._container import _refs_from_json, _refs_to_json
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Sequence

_STATIC = Path(__file__).parent / "static"


class AnywidgetCollapsibleSection(anywidget.AnyWidget):
    """A ``<details>`` section whose body mounts child anywidgets.

    Mirrors ``QtCollapsibleSection``.  Local UI state rather than a model
    control: it puts nothing on the bus and is never passed to
    ``connect_widget``.

    Children are mounted through anywidget's composition API, like
    ``AnywidgetSlot``, so a section works inside a live slot on both Jupyter
    and marimo.  ``expanded`` is written back from the browser, so a section
    keeps its state when the dock re-renders around it.

    Parameters
    ----------
    title :
        The text on the section's summary line, e.g. the visual's name.
    widgets :
        The controls to stack inside, each exposing ``widget``.
    expanded :
        Whether the section starts open.
    gap :
        Spacing between the stacked controls, in pixels.
    """

    _esm = _STATIC / "collapsible_section.js"
    _css = _STATIC / "collapsible_section.css"

    children = traitlets.List(traitlets.Instance(anywidget.AnyWidget)).tag(
        sync=True, to_json=_refs_to_json, from_json=_refs_from_json
    )
    title = traitlets.Unicode("").tag(sync=True)
    expanded = traitlets.Bool(False).tag(sync=True)
    gap = traitlets.Int(4).tag(sync=True)

    def __init__(
        self,
        title: str,
        widgets: Sequence[object] = (),
        *,
        expanded: bool = False,
        gap: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            children=[getattr(item, "widget", item) for item in widgets],
            title=str(title),
            expanded=bool(expanded),
            gap=int(gap),
            **kwargs,
        )

    @property
    def widget(self) -> AnywidgetCollapsibleSection:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def set_title(self, title: str) -> None:
        """Retitle the section."""
        self.title = str(title)

    def set_expanded(self, expanded: bool) -> None:
        """Open or close the section."""
        self.expanded = bool(expanded)

    def close(self) -> None:
        """Close this section and the widgets it shows."""
        for child in list(self.children):
            child.close()
        self.children = []
        close_aux_widgets(self)
        super().close()
