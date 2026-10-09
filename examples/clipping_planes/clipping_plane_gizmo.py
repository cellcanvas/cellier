"""Move a clipping plane by dragging a gizmo in the 3D view."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
from skimage.data import binary_blobs

from cellier.convenience import (
    AppearanceControls,
    InMemoryImageControlsConfig,
    Layout,
    Viewer,
    axis_values_from_viewer,
    run,
)
from cellier.convenience.gui import build_canvas_widget
from cellier.data import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import ClippingPlane, InMemoryImageSingleAppearance

SIDE = 128
AXES = ("z", "y", "x")

blobs = binary_blobs(length=SIDE, n_dim=3, blob_size_fraction=0.2, rng=0)

# The store owns the coordinate system its planes are written in.
store = ImageMemoryStore(
    data=blobs.astype(np.float32),
    name="image",
    data_coordinate_systems=[
        DataCoordinateSystem(
            name="image",
            datastore_id=uuid4(),
            axes=tuple(
                Axis(name=n, axis_type="space", sampling="discrete") for n in AXES
            ),
        )
    ],
)
system = store.data_coordinate_systems[0]
centre = (SIDE / 2, SIDE / 2, SIDE / 2)
planes = (
    ClippingPlane.from_point_normal(system, centre, (0, 0, 1), axes=AXES),
    # Off at first: its gizmo can place it before it clips.
    ClippingPlane.from_point_normal(
        system, centre, (0, 1, 0), axes=AXES, enabled=False
    ),
)

viewer = Viewer(spatial_axes(*AXES), dim="3d")
visual = viewer.add_image(
    store,
    single=InMemoryImageSingleAppearance(
        color_map="viridis", clim=(0.0, 1.0), render_mode="iso", iso_threshold=0.5
    ),
    name="image (ISO)",
    clipping_planes=planes,
    controls=InMemoryImageControlsConfig(appearance=True, clipping_controls=True),
)
visual.aabb.enabled = True

canvas_widget = build_canvas_widget(viewer, axis_values_from_viewer(viewer))
layout = Layout(
    center=canvas_widget,
    right_dock=AppearanceControls(),
    right_dock_min_width=360,
)

# The gizmo needs the plane on screen: add it once the first data is drawn.
viewer.on_ready(lambda: viewer.add_clipping_plane_gizmo(visual, planes[0]))

if __name__ == "__main__":
    run(viewer, layout, fit="ready")
