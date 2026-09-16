"""Simple blob viewer marimo app."""

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
    # Simple blob viewer (anywidget)

    An example of making a simple image viewer with the anywidget GUI frontend.

    - 2D mode: pan (left-drag), zoom (scroll); use the `z` slider to reslice.
    - 3D mode: orbit (left-drag), zoom (scroll), pan (right-drag).
    - The "Switch to 3D / 2D" button toggles the displayed dimensions.

    Note that this requires installing render canvas from main:

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
        Viewer,
        axis_values_from_viewer,
        display,
    )
    from cellier.convenience.gui import build_canvas_widget
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.visuals import InMemoryImageSingleAppearance

    return (
        ImageMemoryStore,
        InMemoryImageSingleAppearance,
        Layout,
        Viewer,
        axis_values_from_viewer,
        binary_blobs,
        build_canvas_widget,
        display,
        np,
    )


@app.cell
def _(binary_blobs, np):
    blobs_3d = binary_blobs(length=200, n_dim=3, rng=0).astype(np.float32)
    return (blobs_3d,)


@app.cell
def _(ImageMemoryStore, InMemoryImageSingleAppearance, Viewer, blobs_3d):
    from cellier.scene.dims import spatial_axes

    viewer = Viewer(spatial_axes("z", "y", "x"), dim="2d", gui="anywidget")

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
    return (viewer,)


@app.cell
def _(axis_values_from_viewer, build_canvas_widget, viewer):
    axis_values = axis_values_from_viewer(viewer)
    canvas_widget = build_canvas_widget(
        viewer,
        axis_values,
        canvas_size=(520, 420),
    )
    return (canvas_widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `display()` resolves the host (Jupyter vs marimo), renders the layout spec,
    presents the result, and arms the first-frame fit + initial reslice.
    It is non-blocking.
    """)
    return


@app.cell
def _(Layout, canvas_widget, display, viewer):
    display(viewer, Layout.single(canvas_widget), fit="ready")
    return


if __name__ == "__main__":
    app.run()
