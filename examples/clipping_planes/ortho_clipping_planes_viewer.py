"""Clipping planes in an OrthoViewer, linked across its panels.

The right dock has the clipping plane controls.  The "Link" combobox
selects how the clipping planes are shared between canvases:

- "All views": one control per dataset moves the cut in all four panels,
  and its "Gizmo" toggle puts a gizmo on a plane in the 3D panel.
- "2D views": the three 2D panels are linked and the 3D panel has its own
  planes and its own control.
- "Not linked": one control per panel.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from cellier.convenience import (
    InMemoryImageControlsConfig,
    LabelsControlsConfig,
    Layout,
    OrthoClippingControls,
    OrthoViewer,
    axis_values_from_ortho,
    run,
)
from cellier.convenience.gui import build_ortho_grid_widget
from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import ClippingPlane, InMemoryImageSingleAppearance

SIDE = 96
AXES = ("z", "y", "x")
BALL_RADIUS = 11
BALL_OFFSET = 30

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Six balls on the axes through the centre, one label each.
grid = np.indices((SIDE, SIDE, SIDE))
centre = SIDE // 2
balls = np.zeros((SIDE, SIDE, SIDE), dtype=np.int32)
ball_label = 0
for axis in range(3):
    for sign in (-1, 1):
        ball_label += 1
        ball_centre = [centre, centre, centre]
        ball_centre[axis] += sign * BALL_OFFSET
        distance = np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, ball_centre)))
        balls[distance <= BALL_RADIUS] = ball_label

# A soft ball at the centre, between the six: no voxel is in both datasets.
radius = np.sqrt(sum((g - centre) ** 2 for g in grid))
core = np.exp(-((radius / 14.0) ** 2)).astype(np.float32)


def coordinate_system(name: str) -> DataCoordinateSystem:
    """The system a store's clipping planes are written in."""
    return DataCoordinateSystem(
        name=name,
        datastore_id=uuid4(),
        axes=tuple(Axis(name=n, axis_type="space", sampling="discrete") for n in AXES),
    )


core_system, balls_system = coordinate_system("core"), coordinate_system("balls")
core_store = ImageMemoryStore(
    data=core, name="core", data_coordinate_systems=[core_system]
)
balls_store = LabelMemoryStore(data=balls, data_coordinate_systems=[balls_system])


def plane_across_x(system: DataCoordinateSystem) -> ClippingPlane:
    """Keep ``x >= centre``."""
    return ClippingPlane.from_point_normal(system, (centre,), (1.0,), axes=("x",))


# ---------------------------------------------------------------------------
# Orthoviewer model
# ---------------------------------------------------------------------------

# "all" is the default; "2d" and None are the other modes.
viewer = OrthoViewer(spatial_axes(*AXES), link_clipping_planes="all")

# clipping_controls=True puts a dataset in the OrthoClippingControls() dock.
# The labels are added before the image: added after it, the image's black
# background covers them in the 2D panels.
viewer.add_labels(
    balls_store,
    name="balls",
    clipping_planes=(plane_across_x(balls_system),),
    controls=LabelsControlsConfig(appearance=False, clipping_controls=True),
)
viewer.add_image(
    core_store,
    single=InMemoryImageSingleAppearance(
        color_map="gray", clim=(0.0, 1.0), render_mode="iso", iso_threshold=0.5
    ),
    name="core",
    clipping_planes=(plane_across_x(core_system),),
    controls=InMemoryImageControlsConfig(appearance=False, clipping_controls=True),
)
viewer.center_slices()

# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

canvas_widgets = build_ortho_grid_widget(viewer, axis_values_from_ortho(viewer))
layout = Layout(
    center=canvas_widgets,
    right_dock=OrthoClippingControls(),
    right_dock_min_width=380,
)

if __name__ == "__main__":
    run(viewer, layout, fit="ready")
