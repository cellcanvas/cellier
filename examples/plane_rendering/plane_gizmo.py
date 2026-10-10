"""Control the plane being rendered in a multiscale image with a gizmo."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
from cellier.visuals import (
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


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

blobs = binary_blobs(length=SIDE, n_dim=3, blob_size_fraction=0.2, rng=0)
levels = [block_average(blobs.astype(np.float32), factor) for factor in FACTORS]

root = Path(tempfile.mkdtemp()) / "image"
root.mkdir()
store = MultiscaleZarrDataStore.from_scale_and_translation(
    zarr_path=str(root),
    scale_names=write_pyramid(root, levels),
    level_scales=[(float(f),) * 3 for f in FACTORS],
    # A level-k voxel centre sits at (k_factor - 1) / 2 level-0 voxels.
    level_translations=[((f - 1) / 2.0,) * 3 for f in FACTORS],
    name="image",
)

# ---------------------------------------------------------------------------
# Viewer, plane and visual
# ---------------------------------------------------------------------------

viewer = Viewer(spatial_axes(*AXES), dim="3d")

# A render plane is written in the scene's world coordinates, on three named
# world axes.  Its rectangle is 96 world units a side, about the plane's
# origin; an extent side left as None would run to the edge of the data.
world = viewer.scene.dims.world_coordinate_system
plane = RenderPlane.from_point_normal(
    world,
    point=(SIDE / 2, SIDE / 2, SIDE / 2),
    normal=(1.0, 0.0, 0.5),
    axes=AXES,
    extent_0=(-48.0, 48.0),
    extent_1=(-48.0, 48.0),
)

visual = viewer.add_image_multiscale(
    store,
    appearance=MultiscaleImageAppearance(),
    single=MultiscaleImageSingleAppearance(
        color_map="viridis", clim=(0.0, 1.0), render_mode="plane"
    ),
    render_config=MultiscaleImageRenderConfig(block_size=16),
    name="image (plane)",
    render_planes=(plane,),
    controls=MultiscaleImageControlsConfig(
        appearance=["visible", "render_mode", "color_map", "level_of_detail"],
        render_plane_controls=True,
        loading_indicator=True,
    ),
)
# The box of the whole data: the plane is cut by it.
visual.aabb.enabled = True


# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

canvas_widget = build_canvas_widget(viewer, axis_values_from_viewer(viewer))
layout = Layout(
    center=canvas_widget,
    right_dock=AppearanceControls(),
    right_dock_min_width=380,
)

# The gizmo needs the plane on screen: add it once the first data is drawn.
viewer.on_ready(lambda: viewer.add_render_plane_gizmo(visual, plane))

if __name__ == "__main__":
    run(viewer, layout, fit="ready")
