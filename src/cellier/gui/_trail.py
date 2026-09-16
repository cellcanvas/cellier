"""What the two trail-window controls share.

``QtTrailControls`` and ``AnywidgetTrailControls`` draw the same controls per
axis from the vocabulary here, so their labels and tooltips cannot drift
apart, and both turn an edit into a ``TrailConfig`` the same way.
"""

from __future__ import annotations

from typing import Any

from cellier.visuals._graph_memory import TrailConfig

TRAIL_TITLE = "Trail"
"""What a trail control calls itself."""

TRAIL_FIELDS = ("before", "after", "fade")
"""The ``TrailConfig`` fields a trail control shows, in display order.

The rest (``fade_before``, ``fade_after``, ``min_alpha``) are not shown and
ride along unchanged through every edit.
"""

TRAIL_LABELS = {"before": "Before", "after": "After", "fade": "Fade"}
"""The label beside each shown field."""

TRAIL_TOOLTIPS = {
    "enabled": (
        "Whether the graph shows a window of this axis around the slice "
        "position rather than the slice alone.  Changing it refetches the graph."
    ),
    "before": "How far the window reaches below the slice position, in world units.",
    "after": "How far the window reaches above the slice position, in world units.",
    "fade": "Fade nodes and edges with their distance from the slice position.",
}
"""Why each control exists, shown on hover."""


def enabled_label(axis_label: str) -> str:
    """The on/off checkbox's text for the axis called *axis_label*."""
    return f"Show on {axis_label}"


def edited_window(previous: TrailConfig | None, field: str, value: Any) -> TrailConfig:
    """Return *previous* with *field* set to *value*, as a new object.

    Never mutates *previous*: the result travels in a ``TrailUpdateEvent`` and
    the controller may adopt it.  With no previous window the defaults are
    the starting point.
    """
    base = previous if previous is not None else TrailConfig()
    return base.model_copy(update={field: value})
