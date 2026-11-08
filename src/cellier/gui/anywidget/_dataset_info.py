"""anywidget widget for displaying dataset metadata.

Mirrors ``QtDatasetInfo``: a read-only display, not part of the ``WidgetView``
bus contract (no ``changed``/``closed``/``subscription_specs``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import numpy as np
import traitlets

from cellier.gui._dataset_info import (
    DatasetInfo,
    MatrixSection,
    RowSection,
    dataset_info_from_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cellier.gui._dataset_info import Section

_STATIC = Path(__file__).parent / "static"

__all__ = ["AnywidgetDatasetInfo", "DatasetInfo", "dataset_info_from_path"]


def _section_payload(section: Section) -> dict:
    """Convert one section to the JSON-safe dict the front end draws.

    Numbers are formatted here rather than in JavaScript so both toolkits
    render a matrix identically -- Qt's ``:.4g`` and a hand-rolled JS
    formatter would have drifted.
    """
    if isinstance(section, RowSection):
        return {
            "kind": "rows",
            "label": section.label,
            "collapsed": bool(section.collapsed),
            "rows": [[label, value] for label, value in section.rows],
        }
    if isinstance(section, MatrixSection):
        matrix = np.asarray(section.matrix)
        return {
            "kind": "matrix",
            "label": section.label,
            "row_labels": [str(label) for label in section.row_labels],
            "col_labels": [str(label) for label in section.col_labels],
            "values": [[f"{value:.4g}" for value in row] for row in matrix],
        }
    raise TypeError(f"Unknown dataset-info section: {section!r}")


class AnywidgetDatasetInfo(anywidget.AnyWidget):
    """Read-only display widget for dataset metadata.

    Shows ``(label, value)`` rows as a table inside a collapsible
    ``<details>`` block.  A sectioned :class:`DatasetInfo` passed to
    :meth:`from_info` additionally draws nested blocks and matrix tables, so
    the anywidget front end renders everything Qt does -- before this it
    could only draw a flat row list, and a store's affine or per-level
    shapes were reachable from Qt alone.

    The rows cross to the front end as data and are written with
    ``textContent``, not as markup: a value carrying ``<`` or ``&`` -- a path,
    a dtype, anything read off a store -- is displayed, never parsed.

    Parameters
    ----------
    rows :
        ``(label, value)`` pairs to display, in order.  Both halves are
        coerced to ``str``.  An empty sequence hides the block.
    """

    _esm = _STATIC / "dataset_info.js"
    _css = _STATIC / "dataset_info.css"

    DEFAULT_TITLE = "Dataset info"
    """Name shown when no ``title=`` is given.

    The renderer passes the title from the shared control vocabulary; this is
    what a directly-constructed widget calls itself, and
    ``test_composite_default_titles_match_the_shared_vocabulary`` pins the two
    together.
    """

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    """What this control calls itself, drawn by its own front end.

    A control names itself rather than being named by whatever lays it out,
    which is what lets the dock stack controls and stop.
    """

    rows = traitlets.List().tag(sync=True)
    """``[[label, value], ...]``, the synced form of the ``rows`` argument."""

    sections = traitlets.List().tag(sync=True)
    """The synced form of a :class:`DatasetInfo`'s sections.

    Empty for a widget built from plain ``rows``, which the front end then
    draws as a single flat table.  When both are set, ``sections`` wins:
    it is the richer description of the same data.
    """

    def __init__(self, rows: Sequence[tuple[str, str]] = (), **kwargs) -> None:
        super().__init__(
            rows=[[str(label), str(value)] for label, value in rows], **kwargs
        )

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetDatasetInfo:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def section_labels(self) -> list[str]:
        """The sections drawn, in the order they are drawn.

        The twin of :meth:`cellier.gui.qt.QtDatasetInfo.section_labels`, and
        it exists for the same reason: one ``DatasetInfo`` must draw the same
        way on both front ends, and nothing else makes that checkable.

        An unlabelled section is named by its first row's label, matching
        what a reader actually sees at the top of the block.
        """
        labels: list[str] = []
        for section in self.sections:
            label = section.get("label")
            if label is None:
                rows = section.get("rows") or []
                label = str(rows[0][0]) if rows else ""
            labels.append(label)
        return labels

    @classmethod
    def from_info(
        cls, info: DatasetInfo, *, title: str | None = None
    ) -> AnywidgetDatasetInfo:
        """Build from a sectioned :class:`DatasetInfo`.

        The anywidget twin of ``QtDatasetInfo.from_info``; the two draw the
        same sections in the same order.
        """
        kwargs = {} if title is None else {"title": title}
        widget = cls(**kwargs)
        widget.sections = [_section_payload(section) for section in info.sections]
        return widget

    @classmethod
    def from_store(cls, store, *, title: str | None = None) -> AnywidgetDatasetInfo:
        """Construct from any data store, by asking it to describe itself."""
        return cls.from_info(store.dataset_info(), title=title)

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        series_index: int = 0,
        title: str | None = None,
    ) -> AnywidgetDatasetInfo:
        """Construct directly from an OME-Zarr URI."""
        info = dataset_info_from_path(
            zarr_path,
            multiscale_index=multiscale_index,
            series_index=series_index,
        )
        return cls.from_info(info, title=title)
