"""Widgets for interacting with the Cellier models.

Currently there are two implementations: qt and anywidget.  What is
exported here is toolkit-neutral, for building custom controls: the
clipping planes helpers read what a clipping planes control of either
toolkit is built from (its planes, its store's bounds, and where its gizmo
is drawn); the render planes helpers do the same for a render planes
control.  The axis values models say which values each dims slider can take.
`image_control_values` builds the starting state of an image control, and
`dataset_info_from_path` reads what a dataset-info widget shows.
"""

from cellier.gui._axis_values import ContinuousAxisValues, DiscreteAxisValues
from cellier.gui._clipping_planes import (
    ClippingPlaneGizmoTarget,
    get_axis_bounds_from_store,
    get_clipping_plane_gizmo_data,
    get_clipping_planes_data_from_visual,
)
from cellier.gui._dataset_info import dataset_info_from_path
from cellier.gui._image_controls import image_control_values
from cellier.gui._render_planes import (
    RenderPlaneGizmoTarget,
    get_render_plane_gizmo_data,
    get_render_planes_data_from_visual,
)

__all__ = [
    "ClippingPlaneGizmoTarget",
    "ContinuousAxisValues",
    "DiscreteAxisValues",
    "RenderPlaneGizmoTarget",
    "dataset_info_from_path",
    "get_axis_bounds_from_store",
    "get_clipping_plane_gizmo_data",
    "get_clipping_planes_data_from_visual",
    "get_render_plane_gizmo_data",
    "get_render_planes_data_from_visual",
    "image_control_values",
]
