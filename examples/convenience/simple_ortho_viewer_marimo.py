"""Simple orthoviewer marimo app."""

import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple orthoviewer (anywidget)

    A simple orthoviewer rendered with an Anywidget frontend..
    A single `add_image` call fans the volume out to four panels: three
    orthogonal slice planes (XY, XZ, YZ) and one 3D volume. Runs in both
    Jupyter and marimo via the anywidget GUI.

    - 2D panels: pan (left-drag), zoom (scroll); slice slider below each canvas.
    - 3D panel: orbit (left-drag), zoom (scroll), pan (right-drag).

    Note that this requires installing rendercanvas from main:

    ```bash
    uv pip install git+https://github.com/pygfx/rendercanvas.git
    ```
    """)
    return


@app.cell
def _():
    import numpy as np
    from skimage.data import binary_blobs

    from cellier.convenience import (
        Layout,
        OrthoViewer,
        axis_values_from_ortho,
        display,
    )
    from cellier.convenience.gui import build_ortho_grid_widget
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.visuals import InMemoryImageSingleAppearance

    return (
        ImageMemoryStore,
        InMemoryImageSingleAppearance,
        Layout,
        OrthoViewer,
        axis_values_from_ortho,
        binary_blobs,
        build_ortho_grid_widget,
        display,
        np,
    )


@app.cell
def _(binary_blobs, np):
    blobs_3d = binary_blobs(length=200, n_dim=3, rng=0).astype(np.float32)
    return (blobs_3d,)


@app.cell
def _(ImageMemoryStore, InMemoryImageSingleAppearance, OrthoViewer, blobs_3d):
    from cellier.scene.dims import spatial_axes

    viewer = OrthoViewer(spatial_axes("z", "y", "x"), gui="anywidget")

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
    return (viewer,)


@app.cell
def _(axis_values_from_ortho, build_ortho_grid_widget, viewer):
    axis_values = axis_values_from_ortho(viewer)
    canvas_widgets = build_ortho_grid_widget(
        viewer, axis_values, canvas_size=(200, 200)
    )
    return (canvas_widgets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `display()` resolves the host (Jupyter vs marimo), renders the layout spec,
    presents the result, and arms the first-frame fit + initial reslice.
    It is non-blocking.
    """)
    return


@app.cell
def _(Layout, canvas_widgets, display, viewer):
    display(viewer, Layout(center=canvas_widgets), fit="ready")
    return


if __name__ == "__main__":
    app.run()
