"""Labels and a two-channel composite image drawn on planes, in memory."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from skimage.data import binary_blobs
from skimage.measure import label

from cellier.convenience import (
    AppearanceControls,
    InMemoryImageControlsConfig,
    LabelsControlsConfig,
    Layout,
    Viewer,
    axis_values_from_viewer,
    run,
)
from cellier.convenience.gui import build_canvas_widget
from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform, Axis, DataCoordinateSystem
from cellier.visuals import (
    InMemoryImageChannelAppearance,
    InMemoryLabelsAppearance,
    RenderPlane,
)

SIDE = 96
AXES = ("z", "y", "x")
#: Where the labels sit along x, in world units.
LABELS_X = 1.2 * SIDE

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Sparse enough that the blobs are separate and get a label each.
blobs = binary_blobs(
    length=SIDE, n_dim=3, blob_size_fraction=0.25, volume_fraction=0.2, rng=0
).astype(np.float32)


def normalised(array: np.ndarray) -> np.ndarray:
    """Scale to 0..1."""
    return ((array - array.min()) / (array.max() - array.min())).astype(np.float32)


# Channel 0: the blob interiors.  Channel 1: the blob boundaries.
interiors = normalised(gaussian_filter(blobs, 1.5))
boundaries = normalised(gaussian_gradient_magnitude(blobs, 1.5))
cells = np.stack([interiors, boundaries])  # (c, z, y, x)

image_store = ImageMemoryStore(data=cells, name="cells")
labels_store = LabelMemoryStore(
    data=label(blobs > 0.5).astype(np.int32),
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

# ---------------------------------------------------------------------------
# Viewer and planes
# ---------------------------------------------------------------------------

# The channel axis is a world axis: a composite draws along it, and a plane
# on (z, y, x) does not constrain it.
viewer = Viewer([("c", "channel"), *spatial_axes(*AXES)], dim="3d")
world = viewer.scene.dims.world_coordinate_system


def planes_at(x_offset: float) -> tuple[RenderPlane, RenderPlane]:
    """Two planes through the middle of a volume that starts at *x_offset*."""
    centre = (SIDE / 2, SIDE / 2, x_offset + SIDE / 2)
    return (
        RenderPlane.from_point_normal(world, centre, (1, 0, 0), axes=AXES),
        RenderPlane.from_point_normal(world, centre, (0, 1, 1), axes=AXES),
    )


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

image_visual = viewer.add_image(
    image_store,
    name="cells",
    channel_axis=0,
    composite=True,
    # One render mode for every channel: a mix is refused.
    channels={
        0: InMemoryImageChannelAppearance(
            color_map="magenta", clim=(0.0, 1.0), render_mode="plane"
        ),
        1: InMemoryImageChannelAppearance(
            color_map="green", clim=(0.0, 1.0), render_mode="plane"
        ),
    },
    render_planes=planes_at(0.0),
    controls=InMemoryImageControlsConfig(appearance=True, render_plane_controls=True),
)

labels_visual = viewer.add_labels(
    labels_store,
    appearance=InMemoryLabelsAppearance(render_mode="plane"),
    name="labels",
    transform=AffineTransform.from_axis_map(
        labels_store.data_coordinate_systems[0],
        world,
        axis_map={name: name for name in AXES},
        translation={"x": LABELS_X},
        # The labels have no channel axis: they are the same on every channel.
        broadcast_output_axes=("c",),
    ),
    render_planes=planes_at(LABELS_X),
    controls=LabelsControlsConfig(appearance=True, render_plane_controls=True),
)

for _visual in (image_visual, labels_visual):
    _visual.aabb.enabled = True

# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

canvas_widget = build_canvas_widget(viewer, axis_values_from_viewer(viewer))
layout = Layout(
    center=canvas_widget,
    right_dock=AppearanceControls(presentation="collapsible_sections"),
    right_dock_min_width=380,
)

if __name__ == "__main__":
    run(viewer, layout, fit="ready")
