"""Move a render plane and a clipping plane from a script, as one drag."""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import tensorstore as ts
from skimage.data import binary_blobs

from cellier.convenience import (
    AppearanceControls,
    Layout,
    MultiscaleImageControlsConfig,
    Viewer,
    axis_values_from_viewer,
    run,
)
from cellier.convenience.gui import build_canvas_widget
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import (
    ClippingPlane,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    RenderPlane,
)

SIDE = 128
AXES = ("z", "y", "x")
#: Downscale factor of each level, finest first.
FACTORS = (1, 2, 4)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def block_average(array: np.ndarray, factor: int) -> np.ndarray:
    """Downscale by averaging ``factor``-sized blocks."""
    if factor == 1:
        return array
    n = array.shape[0] // factor
    blocks = array.reshape(n, factor, n, factor, n, factor)
    return blocks.mean(axis=(1, 3, 5)).astype(array.dtype)


def write_pyramid(root: Path, levels: list[np.ndarray]) -> list[str]:
    """Write one zarr v3 array per level under *root*; returns their names."""
    names = []
    for index, data in enumerate(levels):
        name = f"s{index}"
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / name)},
            "metadata": {
                "shape": list(data.shape),
                "data_type": str(data.dtype),
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [16, 16, 16]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        ts.open(spec).result()[...].write(data).result()
        names.append(name)
    return names


#: Steps of each of the two motions, and the pause between two steps.
STEPS = 20
STEP_S = 1 / 30
#: False moves the planes with no interaction scope: every step is a jump.
USE_SCOPE = True

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

blobs = binary_blobs(length=SIDE, n_dim=3, blob_size_fraction=0.2, rng=0)
levels = [block_average(blobs.astype(np.float32), factor) for factor in FACTORS]

# The store owns the coordinate system its clipping planes are written in.
system = DataCoordinateSystem(
    name="image",
    datastore_id=uuid4(),
    axes=tuple(Axis(name=n, axis_type="space", sampling="discrete") for n in AXES),
)
root = Path(tempfile.mkdtemp()) / "image"
root.mkdir()
store = MultiscaleZarrDataStore.from_scale_and_translation(
    zarr_path=str(root),
    scale_names=write_pyramid(root, levels),
    level_scales=[(float(f),) * 3 for f in FACTORS],
    # A level-k voxel centre sits at (k_factor - 1) / 2 level-0 voxels.
    level_translations=[((f - 1) / 2.0,) * 3 for f in FACTORS],
    data_coordinate_system=system,
    name="image",
)

# ---------------------------------------------------------------------------
# Viewer, planes and visual
# ---------------------------------------------------------------------------

viewer = Viewer(spatial_axes(*AXES), dim="3d")
controller = viewer.controller

# The render plane is in world coordinates and faces z; it runs to the edge
# of the data on every side (the extents are left unbounded).
world = viewer.scene.dims.world_coordinate_system
Z_START, Z_END = 0.25 * SIDE, 0.75 * SIDE
render_plane = RenderPlane.from_point_normal(
    world, point=(Z_START, SIDE / 2, SIDE / 2), normal=(1, 0, 0), axes=AXES
)

# The clipping plane is in the store's data coordinates and keeps the side
# its normal points to: x >= X_START.  It cuts the render plane as it cuts
# a volume rendering.
X_START, X_END = 0.0, 0.5 * SIDE
clipping_plane = ClippingPlane.from_point_normal(
    system, (0, 0, X_START), (0, 0, 1), axes=AXES
)

visual = viewer.add_image_multiscale(
    store,
    appearance=MultiscaleImageAppearance(),
    single=MultiscaleImageSingleAppearance(
        color_map="viridis", clim=(0.0, 1.0), render_mode="plane"
    ),
    render_config=MultiscaleImageRenderConfig(block_size=16),
    name="image (plane)",
    render_planes=(render_plane,),
    clipping_planes=(clipping_plane,),
    controls=MultiscaleImageControlsConfig(
        appearance=["visible", "level_of_detail"],
        render_plane_controls=True,
        clipping_controls=True,
        loading_indicator=True,
    ),
)
visual.aabb.enabled = True

# ---------------------------------------------------------------------------
# The motion
# ---------------------------------------------------------------------------


async def move_planes() -> None:
    """Slide the render plane along z, then the clipping plane along x."""
    scope = (
        controller.plane_interaction(visual.id)
        if USE_SCOPE
        else contextlib.nullcontext()
    )
    with scope:
        for z in np.linspace(Z_START, Z_END, STEPS):
            # A plane is frozen: a move is a copy, which keeps the id.
            moved = render_plane.model_copy(
                update={"origin": (float(z), SIDE / 2, SIDE / 2)}
            )
            controller.set_render_plane(visual.id, render_plane.id, moved)
            # Give the event loop a frame to draw.
            await asyncio.sleep(STEP_S)
        for x in np.linspace(X_START, X_END, STEPS):
            moved = clipping_plane.plane.model_copy(update={"offset": float(x)})
            controller.set_clipping_plane(visual.id, clipping_plane.id, moved)
            await asyncio.sleep(STEP_S)


# ---------------------------------------------------------------------------
# Attach a callback to the plane and slicing events
# ---------------------------------------------------------------------------

owner_id = uuid4()


def print_interaction(event) -> None:
    """Print the start and the end of each interaction."""
    reason = "" if event.reason is None else f" ({event.reason})"
    print(f"plane interaction: {event.phase}{reason}")


def print_plan(event) -> None:
    """Print each plan that includes the visual."""
    if visual.id in event.visual_ids:
        print("plan: the visual's bricks are planned")


controller.on_plane_interaction(visual.id, print_interaction, owner_id=owner_id)
controller.on_reslice_started(viewer.scene.id, print_plan, owner_id=owner_id)

# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

canvas_widget = build_canvas_widget(viewer, axis_values_from_viewer(viewer))
layout = Layout(
    center=canvas_widget,
    right_dock=AppearanceControls(),
    right_dock_min_width=380,
)

#: The running motion: a task is kept, or the event loop may drop it.
motions: list[asyncio.Task] = []


def start_motion() -> None:
    """Start the motion on the running event loop."""
    print("-- the motion starts --")
    motions.append(asyncio.ensure_future(move_planes()))


# Move once the first data is drawn.
viewer.on_ready(start_motion)

if __name__ == "__main__":
    run(viewer, layout, fit="ready")
