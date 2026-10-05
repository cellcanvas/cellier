"""Clipping planes on an image, labels, a mesh and points.

Each visual has a clipping plane attached. Each visual controls widget
has a widget for updating the clipping plane.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
from skimage.data import binary_blobs
from skimage.measure import label, marching_cubes

from cellier.convenience import (
    AppearanceControls,
    InMemoryImageControlsConfig,
    LabelsControlsConfig,
    Layout,
    MeshControlsConfig,
    PointsControlsConfig,
    Viewer,
    axis_values_from_viewer,
    run,
)
from cellier.convenience.gui import build_canvas_widget
from cellier.data import (
    ImageMemoryStore,
    LabelMemoryStore,
    MeshMemoryStore,
    PointsMemoryStore,
)
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageSingleAppearance,
    MeshFlatAppearance,
    PointsMarkerAppearance,
)

SIDE = 128
AXES = ("z", "y", "x")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

blobs = binary_blobs(length=SIDE, n_dim=3, blob_size_fraction=0.2, rng=0)
image = blobs.astype(np.float32)
labels = label(blobs).astype(np.int32)
vertices, faces, _normals, _values = marching_cubes(image, level=0.5)
rng = np.random.default_rng(0)
points = (rng.random((3000, 3)) * SIDE).astype(np.float32)

# ---------------------------------------------------------------------------
# Stores first: each owns the coordinate system its planes are written in
# ---------------------------------------------------------------------------

image_store = ImageMemoryStore(
    data=image,
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
labels_store = LabelMemoryStore(
    data=labels,
    name="labels",
    data_coordinate_systems=[
        DataCoordinateSystem(
            name="labels",
            datastore_id=uuid4(),
            axes=tuple(
                Axis(name=n, axis_type="space", sampling="discrete") for n in AXES
            ),
        )
    ],
)
mesh_store = MeshMemoryStore(
    positions=vertices.astype(np.float32),
    indices=faces.astype(np.int32),
    name="mesh",
    data_coordinate_systems=[
        DataCoordinateSystem(
            name="mesh",
            datastore_id=uuid4(),
            axes=tuple(
                Axis(name=n, axis_type="space", sampling="continuous") for n in AXES
            ),
        )
    ],
)
points_store = PointsMemoryStore(
    positions=points,
    name="points",
    data_coordinate_systems=[
        DataCoordinateSystem(
            name="points",
            datastore_id=uuid4(),
            axes=tuple(
                Axis(name=n, axis_type="space", sampling="continuous") for n in AXES
            ),
        )
    ],
)


def half(store, normal) -> tuple[ClippingPlane, ...]:
    """One plane through the middle of the data, keeping *normal*'s side."""
    centre = (SIDE / 2, SIDE / 2, SIDE / 2)
    return (
        ClippingPlane.from_point_normal(
            store.data_coordinate_systems[0], centre, normal, axes=AXES
        ),
    )


# ---------------------------------------------------------------------------
# Viewer: the four visuals side by side along x
# ---------------------------------------------------------------------------

viewer = Viewer(spatial_axes(*AXES), dim="3d")


def beside(store, column: int):
    """A transform that puts a store in its own column along x."""
    from cellier.transform import AffineTransform

    world = viewer.scene.dims.world_coordinate_system
    return AffineTransform.from_axis_map(
        store.data_coordinate_systems[0],
        world,
        axis_map={name: name for name in AXES},
        translation={"x": column * SIDE * 1.2},
    )


image_visual = viewer.add_image(
    image_store,
    single=InMemoryImageSingleAppearance(
        color_map="viridis", clim=(0.0, 1.0), render_mode="iso", iso_threshold=0.5
    ),
    name="image (ISO)",
    transform=beside(image_store, 0),
    clipping_planes=half(image_store, (0, 0, 1)),
    controls=InMemoryImageControlsConfig(appearance=True, clipping_controls=True),
)
labels_visual = viewer.add_labels(
    labels_store,
    name="labels",
    transform=beside(labels_store, 1),
    clipping_planes=half(labels_store, (0, 1, 1)),
    controls=LabelsControlsConfig(appearance=True, clipping_controls=True),
)
mesh_visual = viewer.add_mesh(
    mesh_store,
    MeshFlatAppearance(color=(0.9, 0.6, 0.2, 1.0), side="both"),
    name="mesh",
    transform=beside(mesh_store, 2),
    clipping_planes=half(mesh_store, (0, 0, 1)),
    controls=MeshControlsConfig(appearance=True, clipping_controls=True),
)
points_visual = viewer.add_points(
    points_store,
    PointsMarkerAppearance(size=4.0, size_space="screen"),
    name="points",
    transform=beside(points_store, 3),
    clipping_planes=half(points_store, (1, 0, 1)),
    controls=PointsControlsConfig(appearance=True, clipping_controls=True),
)

# The bounding box of each visual's whole data.  It is not clipped, so it
# shows how much of the visual a plane has removed.
for visual in (image_visual, labels_visual, mesh_visual, points_visual):
    visual.aabb.enabled = True

# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

canvas_widget = build_canvas_widget(viewer, axis_values_from_viewer(viewer))
layout = Layout(
    center=canvas_widget,
    right_dock=AppearanceControls(presentation="collapsible_sections"),
    right_dock_min_width=360,
)

if __name__ == "__main__":
    run(viewer, layout, fit="ready")
