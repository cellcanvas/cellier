"""A mulitscale orthoviewer where the 3d view has each 2d view rendered as a plane."""

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
    OrthoViewer,
    axis_values_from_ortho,
    run,
)
from cellier.convenience.gui import build_ortho_grid_widget
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    PlaneOutline,
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
# Orthoviewer model
# ---------------------------------------------------------------------------

viewer = OrthoViewer(spatial_axes(*AXES))

# One call makes four visuals, one per panel.  render_mode="plane" is given
# to all four and acts in the 3D panel alone.
visuals = viewer.add_image_multiscale(
    store,
    appearance=MultiscaleImageAppearance(),
    single=MultiscaleImageSingleAppearance(
        color_map="viridis", clim=(0.0, 1.0), render_mode="plane"
    ),
    render_config=MultiscaleImageRenderConfig(block_size=16),
    name="image",
    controls=MultiscaleImageControlsConfig(
        appearance=["visible", "color_map", "render_mode", "level_of_detail"],
        render_plane_controls=True,
    ),
)
vol_visual = visuals["vol"]
vol_visual.aabb.enabled = True

# Put the 2D slices in the middle, then tie the 3D panel's planes to them.
# The mode writes the three planes; it sets no render mode.
#
# The positions are whole voxels on purpose.  viewer.center_slices() would
# give 63.5 here, a voxel face: a 2D panel rounds that to one slice, but a
# plane drawn with nearest interpolation picks the voxel on either side
# pixel by pixel, and looks speckled.
viewer.set_slice_positions(dict.fromkeys(range(3), SIDE / 2))
viewer.plane_controller.mode = "slices"

# An outline round each slice in the 3D panel, in a colour of its own: the
# planes of the xy, xz and yz panels.  The "Render planes" control shows the
# slice planes read-only, so their outlines are set here.
viewer.plane_controller.outlines = (
    PlaneOutline(enabled=True, color=(1.0, 0.8, 0.2, 1.0)),
    PlaneOutline(enabled=True, color=(0.3, 0.9, 1.0, 1.0)),
    PlaneOutline(enabled=True, color=(1.0, 0.4, 0.8, 1.0)),
)

# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

canvas_widgets = build_ortho_grid_widget(viewer, axis_values_from_ortho(viewer))
layout = Layout(
    center=canvas_widgets,
    right_dock=AppearanceControls(),
    right_dock_min_width=380,
)

if __name__ == "__main__":
    run(viewer, layout, fit="ready")
