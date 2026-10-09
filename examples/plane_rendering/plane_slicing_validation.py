"""Plane rendering validation example.

A `tczyx` scene with three time points and three visuals that read three
different stores:

- points: eight markers per time point, one in each octant
  of the volume;
- labels: a sphere centred on each point;
- image, multiscale, three channels rendered as a composite: a larger
  sphere centred on each point.

Each octant has a colour of its own at each time point. The points,
labels, and image should be aligned and have the same color at each time point.
"""

from __future__ import annotations

import colorsys
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import tensorstore as ts

from cellier.convenience import (
    AppearanceControls,
    Layout,
    MultiscaleImageControlsConfig,
    MultiscaleLabelsControlsConfig,
    PointsControlsConfig,
    Viewer,
    axis_values_from_viewer,
    run,
)
from cellier.convenience.gui import build_canvas_widget
from cellier.data import PointsMemoryStore
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.render import OutlineConfig, RenderManagerConfig
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform, Axis, DataCoordinateSystem
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    PlaneOutline,
    PointsMarkerAppearance,
    RenderPlane,
    VisualOutline,
)

SPACE = ("z", "y", "x")
N_TIMES = 3
#: Level-0 voxels along z, y and x, and the world size of one of them.
SHAPE = (64, 128, 128)
VOXEL_SIZE = (2.0, 1.0, 1.0)
#: The volume's side in world units, the same on every axis.
SIDE = 128.0
#: Downscale factor of each level along z, y and x.  z is never downsampled.
FACTORS = ((1, 1, 1), (1, 2, 2), (1, 4, 4))
#: Sphere radii, in world units.
LABEL_RADIUS = 10.0
IMAGE_RADIUS = 18.0
#: The image holds its octant's colour at this fraction of full intensity,
#: so that a label disc can be told from the image disc around it.
IMAGE_INTENSITY = 0.5
#: Where the points sit, per time point, from the centre of their octant
#: (z, y, x; world units).  Even numbers, so that a plane through a point is
#: through voxel centres and not along a voxel face.
OFFSETS = ((-8.0, -6.0, 8.0), (8.0, 10.0, -6.0), (0.0, -10.0, 10.0))

# ---------------------------------------------------------------------------
# What is where: the one table the three stores are built from
# ---------------------------------------------------------------------------


def octant_colors() -> np.ndarray:
    """A colour per time point and octant, shape ``(N_TIMES, 8, 3)``.

    24 hues around the colour wheel, dealt out so that the eight octants of
    a time point are far apart in hue and no octant keeps its colour from
    one time point to the next.
    """
    colors = np.empty((N_TIMES, 8, 3), dtype=np.float32)
    for t in range(N_TIMES):
        for octant in range(8):
            hue = (octant * N_TIMES + t) / (8 * N_TIMES)
            colors[t, octant] = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return colors


def point_positions() -> np.ndarray:
    """The points' world ``(z, y, x)``, shape ``(N_TIMES, 8, 3)``.

    Octant ``k`` is the low or high half of z, y and x by the bits of ``k``.
    """
    positions = np.empty((N_TIMES, 8, 3), dtype=np.float32)
    for t in range(N_TIMES):
        for octant in range(8):
            for axis in range(3):
                high = (octant >> (2 - axis)) & 1
                centre = SIDE * (0.25 + 0.5 * high)
                positions[t, octant, axis] = centre + OFFSETS[t][axis]
    return positions


COLORS = octant_colors()
POSITIONS = point_positions()


def label_id(t: int, octant: int) -> int:
    """The label of an octant's sphere: a value of its own per time point."""
    return 1 + 8 * t + octant


def srgb_to_linear(color: np.ndarray) -> np.ndarray:
    """Undo the sRGB encoding of a colour.

    A label's ``color_dict`` entry is taken as linear light and encoded for
    the screen, where a point's colour and a colour map's are taken as they
    are.  The labels are given the decoded colour, so that the three look
    the same.
    """
    color = np.asarray(color, dtype=np.float64)
    low = color / 12.92
    high = ((color + 0.055) / 1.055) ** 2.4
    return np.where(color <= 0.04045, low, high)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def level_world_coordinates(factor: tuple[int, int, int]) -> list[np.ndarray]:
    """The world coordinate of every voxel centre of a level, per axis.

    A level voxel covers ``factor`` level-0 voxels, so its centre is at
    level-0 index ``i * factor + (factor - 1) / 2``.
    """
    grids = []
    for size, f, voxel in zip(SHAPE, factor, VOXEL_SIZE):
        index = np.arange(size // f, dtype=np.float32) * f + (f - 1) / 2.0
        grids.append(index * voxel)
    return grids


def render_level(factor: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """One level: the image ``(t, c, z, y, x)`` and the labels ``(t, z, y, x)``.

    Each level is drawn from the table, not downsampled from the level
    below, so every level holds spheres about the same world centres.
    """
    z, y, x = np.meshgrid(*level_world_coordinates(factor), indexing="ij")
    shape = z.shape
    image = np.zeros((N_TIMES, 3, *shape), dtype=np.float32)
    labels = np.zeros((N_TIMES, *shape), dtype=np.int32)
    for t in range(N_TIMES):
        for octant in range(8):
            cz, cy, cx = POSITIONS[t, octant]
            distance = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
            inside = distance <= IMAGE_RADIUS
            for channel in range(3):
                value = IMAGE_INTENSITY * COLORS[t, octant, channel]
                image[t, channel][inside] = value
            labels[t][distance <= LABEL_RADIUS] = label_id(t, octant)
    return image, labels


def write_pyramid(root: Path, levels: list[np.ndarray]) -> list[str]:
    """Write one zarr v3 array per level under *root*; returns their names."""
    root.mkdir()
    names = []
    for index, data in enumerate(levels):
        name = f"s{index}"
        # One time point and one channel a chunk; 16 voxels a side in space.
        chunks = [1] * (data.ndim - 3) + [16, 16, 16]
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / name)},
            "metadata": {
                "shape": list(data.shape),
                "data_type": str(data.dtype),
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": chunks},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        ts.open(spec).result()[...].write(data).result()
        names.append(name)
    return names


def data_system(name: str, axes: tuple[str, ...]) -> DataCoordinateSystem:
    """A store's level-0 coordinate system: voxel indices on *axes*."""
    kinds = {"t": "time", "c": "channel"}
    return DataCoordinateSystem(
        name=name,
        datastore_id=uuid4(),
        axes=tuple(
            Axis(name=n, axis_type=kinds.get(n, "space"), sampling="discrete")
            for n in axes
        ),
    )


def pyramid_store(
    root: Path, levels: list[np.ndarray], name: str, axes: tuple[str, ...]
) -> MultiscaleZarrDataStore:
    """A multiscale zarr store over *levels*, whose last three axes are space."""
    lead = len(axes) - 3
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=write_pyramid(root, levels),
        level_scales=[(1.0,) * lead + tuple(float(f) for f in fs) for fs in FACTORS],
        level_translations=[
            (0.0,) * lead + tuple((f - 1) / 2.0 for f in fs) for fs in FACTORS
        ],
        data_coordinate_system=data_system(name, axes),
        name=name,
    )


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

rendered = [render_level(factor) for factor in FACTORS]
tmpdir = Path(tempfile.mkdtemp())
image_store = pyramid_store(
    tmpdir / "image", [image for image, _ in rendered], "image", ("t", "c", *SPACE)
)
labels_store = pyramid_store(
    tmpdir / "labels", [labels for _, labels in rendered], "labels", ("t", *SPACE)
)

# One row per point: (t, z, y, x), in world units already.
times = np.repeat(np.arange(N_TIMES, dtype=np.float32), 8)[:, None]
points_store = PointsMemoryStore(
    positions=np.hstack([times, POSITIONS.reshape(-1, 3)]),
    colors=np.hstack([COLORS.reshape(-1, 3), np.ones((N_TIMES * 8, 1), np.float32)]),
    name="points",
    data_coordinate_systems=[data_system("points", ("t", *SPACE))],
)

# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

# A 3D view displays the last three axes; t has a slider, and the composite
# draws along c.
# The outline pass is on for the markers alone (below): a marker has its
# label's colour, and without a contour it could not be seen on the label.
viewer = Viewer(
    [("t", "time"), ("c", "channel"), *spatial_axes(*SPACE)],
    dim="3d",
    render_config=RenderManagerConfig(
        outline=OutlineConfig(enabled=True, palette=[(1.0, 1.0, 1.0, 1.0)])
    ),
)
world = viewer.scene.dims.world_coordinate_system


def to_world(store, *, voxels: bool) -> AffineTransform:
    """A store's data axes onto the world axes of the same names.

    The pyramids are in voxels, so their space axes are scaled by the voxel
    size; the points are in world units already.  A store with no channel
    axis is the same on every channel.
    """
    system = store.data_coordinate_systems[0]
    names = [axis.name for axis in system.axes]
    return AffineTransform.from_axis_map(
        system,
        world,
        axis_map={name: name for name in names},
        scale=dict(zip(SPACE, VOXEL_SIZE)) if voxels else None,
        broadcast_output_axes=() if "c" in names else ("c",),
    )


# One plane for the image and the labels: a plane is a value in world
# coordinates, so two visuals of a scene can be given the same one.
#
# Its origin is on a point: the one in the octant that is the high half of
# z, y and x, at t = 0.  The control turns a plane about its origin, and the
# points of a time point are at one place in every octant, so the plane
# facing z, y or x still passes through four points.  (An origin at the
# centre of the volume is between the spheres: a plane turned there cuts
# none of them, and an empty plane is black on black.)
#
# The plane has an outline: a yellow line where it ends at the data's box.
# Most of the plane is background, black on black; the outline shows where
# it is.  The two visuals have the same box, so there is one line.
plane = RenderPlane.from_point_normal(
    world,
    point=POSITIONS[0, 7],
    normal=(1, 0, 0),
    axes=SPACE,
    outline=PlaneOutline(enabled=True, color=(1.0, 0.8, 0.2, 1.0)),
)

# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

image_visual = viewer.add_image_multiscale(
    image_store,
    appearance=MultiscaleImageAppearance(),
    name="image",
    render_config=MultiscaleImageRenderConfig(block_size=16),
    transform=to_world(image_store, voxels=True),
    channel_axis=1,
    composite=True,
    # A channel per colour component: added, the three give the colour back.
    channels={
        channel: MultiscaleImageChannelAppearance(
            color_map=color_map, clim=(0.0, 1.0), render_mode="plane"
        )
        for channel, color_map in enumerate(("red", "green", "blue"))
    },
    render_planes=(plane,),
    controls=MultiscaleImageControlsConfig(
        appearance=["visible", "render_mode", "level_of_detail"],
        render_plane_controls=True,
    ),
)

labels_visual = viewer.add_labels_multiscale(
    labels_store,
    MultiscaleLabelsAppearance(
        render_mode="plane",
        colormap_mode="direct",
        color_dict={
            label_id(t, octant): (*srgb_to_linear(COLORS[t, octant]).tolist(), 1.0)
            for t in range(N_TIMES)
            for octant in range(8)
        },
    ),
    name="labels",
    render_config=MultiscaleLabelRenderConfig(block_size=16),
    transform=to_world(labels_store, voxels=True),
    render_planes=(plane,),
    controls=MultiscaleLabelsControlsConfig(
        appearance=True, render_plane_controls=True
    ),
)

points_visual = viewer.add_points(
    points_store,
    PointsMarkerAppearance(size=10.0, size_space="screen", color_mode="vertex"),
    name="points",
    transform=to_world(points_store, voxels=False),
    # A white contour, in the palette's first colour.
    outline=VisualOutline(slot=1),
    controls=PointsControlsConfig(appearance=True),
)

image_visual.aabb.enabled = True

# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

canvas_widget = build_canvas_widget(viewer, axis_values_from_viewer(viewer))
layout = Layout(
    center=canvas_widget,
    right_dock=AppearanceControls(presentation="collapsible_sections"),
    right_dock_min_width=380,
)

# The gizmo needs the plane on screen: add it once the first data is drawn.
# It is on the labels' plane; the image's plane is kept equal to it below.
viewer.on_ready(lambda: viewer.add_render_plane_gizmo(labels_visual, plane))

# The gizmo and each "Render planes" control edit the planes of one visual.
# Carry every change of either visual's planes over to the other, so that
# the two stay on one plane whichever is edited.  This cannot loop:
# assigning planes equal to the ones a visual has does nothing.
link_id = uuid4()
for source, target in (
    (labels_visual, image_visual),
    (image_visual, labels_visual),
):
    viewer.controller.on_render_planes_changed(
        source.id,
        lambda event, target=target: viewer.controller.set_render_planes(
            target.id, event.render_planes
        ),
        owner_id=link_id,
    )

if __name__ == "__main__":
    run(viewer, layout, fit="ready")
