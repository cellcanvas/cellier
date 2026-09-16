"""Simple 4-panel orthoviewer using the cellier OrthoViewer convenience class.

Creates a (200, 200, 200) float32 image from scikit-image blobs and displays it
in a four-panel orthoviewer: three orthogonal slice planes (XY, XZ, YZ) and one
3D volume.  A single ``add_image`` call fans the visual out to all four panels.
The 3D panel renders the volume as an isosurface (``render_mode="iso"``).

Run::

    .venv/bin/python examples/convenience/simple_ortho_viewer.py

Controls
--------
2D panels : pan (left-drag), zoom (scroll); slice slider below each canvas
3D panel  : orbit (left-drag), zoom (scroll), pan (right-drag)
"""

from __future__ import annotations

import numpy as np
from skimage.data import binary_blobs

from cellier.convenience import (
    Layout,
    OrthoViewer,
    axis_values_from_ortho,
    run,
)
from cellier.convenience.gui import build_ortho_grid_widget
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageSingleAppearance

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

blobs_3d = binary_blobs(length=200, n_dim=3, rng=0).astype(np.float32)

# ---------------------------------------------------------------------------
# Orthoviewer model
# ---------------------------------------------------------------------------

viewer = OrthoViewer(spatial_axes("z", "y", "x"))

store = ImageMemoryStore(data=blobs_3d, name="blobs")
viewer.controller.add_data_store(store)

viewer.add_image(
    store,
    single=InMemoryImageSingleAppearance(
        color_map="viridis",
        clim=(0.0, 1.0),
        render_mode="iso",
        iso_threshold=0.5,
    ),
    name="blobs",
)

# Center the 2D slice planes in the middle of the volume.
viewer.center_slices()

# ---------------------------------------------------------------------------
# Canvas + layout
# ---------------------------------------------------------------------------

axis_values = axis_values_from_ortho(viewer)
canvas_widgets = build_ortho_grid_widget(viewer, axis_values)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

run(viewer, Layout(center=canvas_widgets), fit="ready")
