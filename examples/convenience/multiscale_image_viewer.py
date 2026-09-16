"""Multiscale image viewer using the cellier Viewer convenience class.

Creates three scale levels from a (64, 128, 128) float32 concentric-shell
volume, writes them as zarr v3 arrays, and displays the multiscale image in
3-D ISO mode.  A button in the bottom dock toggles between 2D and 3D.

Requires tensorstore::

    uv pip install tensorstore

Run::

    .venv/bin/python examples/convenience/multiscale_image_viewer.py

Controls
--------
Visible : checkbox in the appearance dock.  Known issue: the canvas does not
          fully update until you drag it -- temporal accumulation blends each
          frame into a history discarded only on camera movement, so the image
          lingers as a ghost.  See plans/visibility_debugging.md.
3D mode : orbit (left-drag), zoom (scroll), pan (right-drag)
2D mode : pan (left-drag), zoom (scroll)
Toggle  : "Switch to 2D / 3D" button in the bottom dock
Z slice : dims slider (2D mode only)
Dataset info : collapsible block at the bottom of the appearance dock,
          built by the store from its own metadata (``dataset_info=True``)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import tensorstore as ts

from cellier.convenience import (
    AppearanceControls,
    Layout,
    Viewer,
    axis_values_from_viewer,
    run,
)
from cellier.convenience.gui import MultiscaleImageControlsConfig, build_canvas_widget
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import MultiscaleImageAppearance, MultiscaleImageSingleAppearance

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _concentric_shells(nz, ny, nx):
    z, y, x = np.ogrid[-1 : 1 : nz * 1j, -1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    r = np.sqrt(x**2 + y**2 + z**2)
    vol = (
        np.exp(-(((r - 0.3) / 0.05) ** 2))
        + np.exp(-(((r - 0.6) / 0.05) ** 2))
        + np.exp(-(((r - 0.9) / 0.05) ** 2))
    )
    return (vol / vol.max()).astype(np.float32)


def _block_average(arr, factor):
    s = arr.shape
    d = arr.ndim
    slices = tuple(slice(None, (s[i] // factor) * factor) for i in range(d))
    trimmed = arr[slices]
    shape = []
    for i in range(d):
        shape.extend([trimmed.shape[i] // factor, factor])
    return (
        trimmed.reshape(shape).mean(axis=tuple(range(1, 2 * d, 2))).astype(np.float32)
    )


def _write_zarr3(base_path, name, data):
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(base_path / name)},
        "metadata": {
            "shape": list(data.shape),
            "data_type": "float32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [16, 16, 16]},
            },
        },
        "create": True,
        "delete_existing": True,
    }
    ts_store = ts.open(spec).result()
    ts_store[...].write(data).result()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

tmpdir = Path(tempfile.mkdtemp())

volume = _concentric_shells(64, 128, 128)
s1 = _block_average(volume, 2)
s2 = _block_average(s1, 2)

_write_zarr3(tmpdir, "s0", volume)
_write_zarr3(tmpdir, "s1", s1)
_write_zarr3(tmpdir, "s2", s2)

# ---------------------------------------------------------------------------
# Multiscale store + viewer model
# ---------------------------------------------------------------------------

store = MultiscaleZarrDataStore(
    zarr_path=str(tmpdir),
    scale_names=["s0", "s1", "s2"],
    level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (4.0, 4.0, 4.0)],
    level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.5, 1.5, 1.5)],
)

viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d")

viewer.add_image_multiscale(
    store,
    appearance=MultiscaleImageAppearance(
        lod_bias=1.0,
        attenuation=1.0,
    ),
    single=MultiscaleImageSingleAppearance(
        color_map="viridis",
        clim=(0.0, 1.0),
        render_mode="iso",
        iso_threshold=0.45,
    ),
    controls=MultiscaleImageControlsConfig(
        appearance=[
            # First in the panel: the group order follows the config class's
            # control map, not this list.
            "visible",
            "color_map",
            "clim",
            "render_mode",
            "iso_threshold",
            "attenuation",
            "lod_bias",
        ],
        colormap_names=[
            "grays",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "hot",
            "bwr",
            "RdYlBu",
        ],
        clim_range=(0.0, 1.0),
        # The store describes itself: path, dtype, axes, and the per-level
        # shapes and scales read off its own level_transforms.
        dataset_info=True,
    ),
)

# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

axis_values = axis_values_from_viewer(viewer)
canvas_view = build_canvas_widget(viewer, axis_values)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run(
    viewer,
    Layout(
        center=canvas_view,
        left_dock=AppearanceControls(),
    ),
    fit="ready",
)
