"""Widgets for interacting with the Cellier models.

Currently there are two implementations: qt and anywidget.  What is
exported here is toolkit-neutral, for building custom controls: the colour
helpers read and write the RGBA tuples cellier's models use, and the
clipping planes helpers read what a clipping planes control of either
toolkit is built from (its planes, its store's bounds, and where its gizmo
is drawn).
"""

from cellier.gui._appearance_fields import as_rgba, hex_to_rgba, rgba_to_hex
from cellier.gui._clipping_planes import (
    ClippingPlaneGizmoTarget,
    get_axis_bounds_from_store,
    get_clipping_plane_gizmo_data,
    get_clipping_planes_data_from_visual,
)

__all__ = [
    "ClippingPlaneGizmoTarget",
    "as_rgba",
    "get_axis_bounds_from_store",
    "get_clipping_plane_gizmo_data",
    "get_clipping_planes_data_from_visual",
    "hex_to_rgba",
    "rgba_to_hex",
]
