"""View a large remote image in 2d and 3d.

The dataset is the 20.07 um overview scan of the left hemisphere of brain
I74, imaged by HiP-CT from the the Human Organ Atlas (https://human-organ-atlas.esrf.fr).
The image is stored as an OME-Zarr file on Google Cloud Storage.

There are widgets to adding and controlling clipping planes. Try using the Gizmo to
interactively adjust the clipping plane!

Performance settings
--------------------
When viewing large images, you may need to tune some of the scheduler and shader
settings to optimize the responsiveness of the viewer.

``BLOCK_SIZE = 64``
    The size of the 3D blocks in the shader is set to match the chunk size of the data.
``MAX_IN_FLIGHT = 128`` and ``REQUEST_CONCURRENCY = 128``
    The number of reads the scheduler keeps outstanding, and requests tensorstore sends
    for them.
``CACHE_POOL_BYTES = 4 GiB``
    A 2D tile is one voxel thick but its chunk is 64 thick, so one screen of
    tiles decodes to about 0.5 GB.  Keeping those chunks means the next 63
    slices of the same slab need no requests at all.
``GPU_BUDGET_3D_BYTES = 2 GiB``
    This is the size of the brick cache on the GPU.
``coarsest_while_moving_2d = True``
    Dragging the z slider loads only the coarse backstop of each slice. This
    makes drags responsive while still giving some information about the slice.
``INTERMEDIATE_OFFSETS = (2,)``
    Loading is coarse, medium, fine. This specifies on intermediate request,
    2 pyramid levels less than the target. This gives a more informative visual
    more quickly.
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.table import Table

from cellier.convenience import (
    AppearanceControls,
    ClippingPlanesControl,
    DatasetInfoControl,
    ImageControl,
    Layout,
    LevelOfDetailControl,
    LoadingIndicator,
    RenderPlanesControl,
    Viewer,
    axis_values_from_viewer,
    run,
)
from cellier.convenience.gui import build_canvas_widget
from cellier.data.image._ome_zarr_image_store import OMEZarrImageDataStore
from cellier.render import (
    AmbientOcclusionConfig,
    CameraConfig,
    RenderManagerConfig,
    SchedulerConfig,
)
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    PlaneOutline,
    ProgressiveLoadingConfig,
    RenderPlane,
)

URL = (
    "https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b"
    "/I74/brain-left_hemisphere/20.07um_overview_bm18.ome.zarr"
)

# The data store's axes
AXES = ("x", "y", "z")
# The axes a 2D view shows.  The remaining one, z, is sliced.
DISPLAYED_2D = ("x", "y")

GIB = 1024**3
BLOCK_SIZE = 64
MAX_IN_FLIGHT = 128
REQUEST_CONCURRENCY = 128
CACHE_POOL_BYTES = 4 * GIB
GPU_BUDGET_3D_BYTES = 2 * GIB
GPU_BUDGET_2D_BYTES = 128 * 1024**2

# One intermediate loading stage, two levels coarser than the target.
INTERMEDIATE_OFFSETS = (2,)

# Ambient occlusion contrast exponent; the other settings are the defaults.
AO_POWER = 2.5

# Default level-of-detail bias once the view settles; above 1 picks coarser levels.
SETTLED_LOD_BIAS = 1.5

# Isosurface threshold of the "smooth_iso" render mode.
ISO_THRESHOLD = 32000.0

# The 3D camera's depth range, as fractions of the volume's diagonal
DEPTH_NEAR_FRACTION = 1e-3
DEPTH_FAR_FRACTION = 8.0

# Seconds without a camera change before the view counts as settled.
# The scene is redrawn when the camera is settled.
CAMERA_SETTLE_S = 0.15

# Contrast limits
CLIM = (27000.0, 36000.0)
CLIM_RANGE = (0.0, 65535.0)

# The colour of the render plane's outline (RGBA): purple.
PLANE_OUTLINE_COLOR = (0.6, 0.2, 0.8, 1.0)

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

console = Console()
console.print(f"[bold]Opening[/bold] {URL}")
store = OMEZarrImageDataStore.from_path(
    URL,
    cache_pool_bytes=CACHE_POOL_BYTES,
    request_concurrency=REQUEST_CONCURRENCY,
    name="I74 brain, left hemisphere",
)


def print_dataset_info() -> None:
    """Pretty-print the store's levels, with the full-resolution size in GB."""
    itemsize = np.dtype(store.dtype).itemsize
    table = Table(title=store.name, title_style="bold")
    table.add_column("Level", justify="right")
    table.add_column("Shape (x, y, z)", justify="right")
    table.add_column("Size (GB)", justify="right")
    for level, shape in enumerate(store.level_shapes):
        size_gb = int(np.prod(shape)) * itemsize / 1e9
        style = "bold green" if level == 0 else None
        table.add_row(
            str(level),
            " x ".join(str(n) for n in shape),
            f"{size_gb:,.1f}",
            style=style,
        )
    console.print(table)
    level0_gb = int(np.prod(store.level_shapes[0])) * itemsize / 1e9
    console.print(
        f"[bold]Scale 0:[/bold] [green]{level0_gb:,.1f} GB[/green] decoded "
        f"({np.dtype(store.dtype)}, {store.n_levels} levels, "
        f"voxel size {store.physical_scale} um)"
    )


print_dataset_info()

# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

viewer = Viewer(
    spatial_axes(*AXES),
    displayed_axes=DISPLAYED_2D,
    render_config=RenderManagerConfig(
        scheduler=SchedulerConfig(max_in_flight=MAX_IN_FLIGHT),
        camera=CameraConfig(settle_threshold_s=CAMERA_SETTLE_S),
        ambient_occlusion=AmbientOcclusionConfig(enabled=True, power=AO_POWER),
    ),
)

# Map the data to the world scene in micrometers.
to_micrometers = AffineTransform.from_axis_map(
    store.data_coordinate_systems[0],
    viewer.scene.dims.world_coordinate_system,
    axis_map={name: name for name in AXES},
    scale=dict(zip(AXES, store.physical_scale)),
    translation=dict(zip(AXES, store.physical_translation)),
)

# Make a default rendering plane centered on the volume
world = viewer.scene.dims.world_coordinate_system
center_um = [
    size // 2 * scale + translation
    for size, scale, translation in zip(
        store.level_shapes[0], store.physical_scale, store.physical_translation
    )
]
render_plane = RenderPlane.from_point_normal(
    world,
    point=center_um,
    normal=(1.0, 0.0, 0.0),
    axes=AXES,
    outline=PlaneOutline(enabled=True, color=PLANE_OUTLINE_COLOR),
)

image = viewer.add_image_multiscale(
    store,
    name="I74 brain",
    transform=to_micrometers,
    appearance=MultiscaleImageAppearance(
        coarsest_while_moving_2d=True,
        settled_lod_bias=SETTLED_LOD_BIAS,
    ),
    single=MultiscaleImageSingleAppearance(
        color_map="gray",
        clim=CLIM,
        render_mode="smooth_iso",
        iso_threshold=ISO_THRESHOLD,
    ),
    render_config=MultiscaleImageRenderConfig(
        block_size=BLOCK_SIZE,
        gpu_budget_bytes=GPU_BUDGET_3D_BYTES,
        gpu_budget_bytes_2d=GPU_BUDGET_2D_BYTES,
        loading=ProgressiveLoadingConfig(intermediate_offsets=INTERMEDIATE_OFFSETS),
    ),
    render_planes=(render_plane,),
    # The dock shows these controls in this order: move a line to move its
    # control, delete a line to drop it.
    controls=[
        LoadingIndicator(),
        DatasetInfoControl(),
        ImageControl(
            fields=[
                "visible",
                "color_map",
                "clim",
                "render_mode",
                "iso_threshold",
                "attenuation",
            ],
            colormap_names=[
                "gray",
                "grays",
                "viridis",
                "magma",
                "inferno",
                "cividis",
                "turbo",
            ],
            clim_range=CLIM_RANGE,
            decimals=0,
        ),
        LevelOfDetailControl(),
        ClippingPlanesControl(),
        RenderPlanesControl(),
    ],
)

# The bounding box of the whole volume.
image.aabb.enabled = True

# Start on the middle z slice, looking down z.  Slice positions are world
# positions, so the middle voxel is converted to micrometers.
Z = AXES.index("z")
middle_z_um = (
    store.level_shapes[0][Z] // 2 * store.physical_scale[Z]
    + store.physical_translation[Z]
)
viewer.set_slice_positions({Z: middle_z_um})

# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

# Set the camera depth range based on the dataset
extent_um = [
    size * scale for size, scale in zip(store.level_shapes[0], store.physical_scale)
]
diagonal_um = float(np.linalg.norm(extent_um))

canvas_widget = build_canvas_widget(
    viewer,
    axis_values_from_viewer(viewer),
    depth_range_3d=(
        diagonal_um * DEPTH_NEAR_FRACTION,
        diagonal_um * DEPTH_FAR_FRACTION,
    ),
)

layout = Layout(
    center=canvas_widget,
    left_dock=AppearanceControls(),
    left_dock_min_width=360,
)

if __name__ == "__main__":
    run(viewer, layout, fit="ready")
