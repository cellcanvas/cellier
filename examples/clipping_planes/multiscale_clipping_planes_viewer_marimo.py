"""Clipping planes on multiscale visuals, as a marimo app."""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clipping planes on multiscale visuals

    A multiscale image, multiscale labels and a multiscale mesh, side by side
    along x.  Each has one clipping plane attached, and a **Clipping planes**
    group in its section of the dock.

    A plane keeps the half-space its normal points into.  It is written in
    the data coordinates of the store the visual reads, so the store is
    built first, then the plane, then the visual:

    ```python
    plane = ClippingPlane.from_point_normal(
        store.data_coordinate_systems[0],
        point=(64, 64, 64),
        normal=(0, 0, 1),
        axes=("z", "y", "x"),
    )
    visual = viewer.add_image_multiscale(store, clipping_planes=(plane,))
    ```

    **The control.**  Each plane has a row:

    - `+z` `-z` `+y` `-y` `+x` `-x` face the plane along a data axis and keep
      the side toward higher (`+`) or lower (`-`) values.  The axis names are
      the store's own.
    - The entries under the buttons are the normal, one per axis, for an
      oblique plane.
    - **Position** moves the plane along its normal, in data units (level-0
      voxels here).
    - **Flip** changes the sign on the normal.
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path
    from uuid import uuid4

    import numpy as np
    import tensorstore as ts
    from skimage.data import binary_blobs
    from skimage.measure import label, marching_cubes

    from cellier.convenience import (
        AppearanceControls,
        Layout,
        MeshControlsConfig,
        MultiscaleImageControlsConfig,
        MultiscaleLabelsControlsConfig,
        Viewer,
        axis_values_from_viewer,
        display,
    )
    from cellier.convenience.gui import build_canvas_widget
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
    from cellier.data.mesh import MeshLevel, MultiscaleMeshStore
    from cellier.scene.dims import spatial_axes
    from cellier.transform import AffineTransform, Axis, DataCoordinateSystem
    from cellier.visuals import (
        ClippingPlane,
        GeometryLodConfig,
        MeshFlatAppearance,
        MultiscaleImageAppearance,
        MultiscaleImageSingleAppearance,
        MultiscaleLabelsAppearance,
    )

    return (
        AffineTransform,
        AppearanceControls,
        Axis,
        ClippingPlane,
        DataCoordinateSystem,
        GeometryLodConfig,
        Layout,
        MeshControlsConfig,
        MeshFlatAppearance,
        MeshLevel,
        MultiscaleImageAppearance,
        MultiscaleImageControlsConfig,
        MultiscaleImageSingleAppearance,
        MultiscaleLabelsAppearance,
        MultiscaleLabelsControlsConfig,
        MultiscaleMeshStore,
        MultiscaleZarrDataStore,
        Path,
        Viewer,
        axis_values_from_viewer,
        binary_blobs,
        build_canvas_widget,
        display,
        label,
        marching_cubes,
        np,
        spatial_axes,
        tempfile,
        ts,
        uuid4,
    )


@app.cell(hide_code=True)
def _(
    Axis,
    DataCoordinateSystem,
    MeshLevel,
    MultiscaleZarrDataStore,
    Path,
    marching_cubes,
    np,
    ts,
    uuid4,
):
    SIDE = 128
    AXES = ("z", "y", "x")
    #: Downscale factor of each level, finest first.
    FACTORS = (1, 2, 4)

    def block_average(array: np.ndarray, factor: int) -> np.ndarray:
        """Downscale by averaging ``factor``-sized blocks (an image level)."""
        if factor == 1:
            return array
        n = array.shape[0] // factor
        blocks = array.reshape(n, factor, n, factor, n, factor)
        return blocks.mean(axis=(1, 3, 5)).astype(array.dtype)

    def subsample(array: np.ndarray, factor: int) -> np.ndarray:
        """Downscale by taking every ``factor``-th voxel (a labels level)."""
        return np.ascontiguousarray(array[::factor, ::factor, ::factor])

    def write_pyramid(root: Path, levels: list) -> list:
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

    def data_system(name: str, sampling: str) -> DataCoordinateSystem:
        """A zyx data coordinate system for a store built here."""
        return DataCoordinateSystem(
            name=name,
            datastore_id=uuid4(),
            axes=tuple(
                Axis(name=n, axis_type="space", sampling=sampling) for n in AXES
            ),
        )

    def pyramid_store(root: Path, levels: list, name: str):
        """A multiscale zarr store over *levels*, with its level-0 system."""
        root.mkdir()
        return MultiscaleZarrDataStore.from_scale_and_translation(
            zarr_path=str(root),
            scale_names=write_pyramid(root, levels),
            level_scales=[(float(f),) * 3 for f in FACTORS],
            # A level-k voxel centre sits at (k_factor - 1) / 2 level-0 voxels.
            level_translations=[((f - 1) / 2.0,) * 3 for f in FACTORS],
            data_coordinate_system=data_system(name, "discrete"),
            name=name,
        )

    def surface(volume: np.ndarray, factor: int) -> MeshLevel:
        """The 0.5 isosurface of a level, in level-0 voxel coordinates."""
        vertices, faces, _normals, _values = marching_cubes(volume, level=0.5)
        vertices = vertices * factor + (factor - 1) / 2.0
        return MeshLevel(
            positions=vertices.astype(np.float32), indices=faces.astype(np.int32)
        )

    return (
        AXES,
        FACTORS,
        SIDE,
        block_average,
        data_system,
        pyramid_store,
        subsample,
        surface,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stores first

    One blob volume, as an image pyramid, a labels pyramid and a two-level
    surface.  Each store owns the coordinate system its planes are written in.
    """)
    return


@app.cell
def _(
    FACTORS,
    MultiscaleMeshStore,
    Path,
    SIDE,
    binary_blobs,
    block_average,
    data_system,
    label,
    np,
    pyramid_store,
    subsample,
    surface,
    tempfile,
):
    blobs = binary_blobs(length=SIDE, n_dim=3, blob_size_fraction=0.2, rng=0)
    image = blobs.astype(np.float32)
    labels = label(blobs).astype(np.int32)

    image_levels = [block_average(image, factor) for factor in FACTORS]
    label_levels = [subsample(labels, factor) for factor in FACTORS]

    tmpdir = Path(tempfile.mkdtemp())
    image_store = pyramid_store(tmpdir / "image", image_levels, "image")
    labels_store = pyramid_store(tmpdir / "labels", label_levels, "labels")
    # The mesh levels are listed finest first and share one coordinate system.
    mesh_store = MultiscaleMeshStore(
        levels=[
            surface(image_levels[0], FACTORS[0]),
            surface(image_levels[2], FACTORS[2]),
        ],
        name="mesh",
        data_coordinate_systems=[data_system("mesh", "continuous")],
    )
    return image_store, labels_store, mesh_store


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Planes, visuals and the dock

    Open a visual's section in the dock on the right and move its plane.  The
    image and labels sections also show how much of the plan is loaded: watch
    it while a plane moves.
    """)
    return


@app.cell
def _(
    AXES,
    AffineTransform,
    AppearanceControls,
    ClippingPlane,
    GeometryLodConfig,
    Layout,
    MeshControlsConfig,
    MeshFlatAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageControlsConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelsAppearance,
    MultiscaleLabelsControlsConfig,
    SIDE,
    Viewer,
    axis_values_from_viewer,
    build_canvas_widget,
    display,
    image_store,
    labels_store,
    mesh_store,
    spatial_axes,
):
    viewer = Viewer(spatial_axes(*AXES), dim="3d", gui="anywidget")

    def half(store, normal):
        """One plane through the middle of the data, keeping *normal*'s side."""
        centre = (SIDE / 2, SIDE / 2, SIDE / 2)
        return (
            ClippingPlane.from_point_normal(
                store.data_coordinate_systems[0], centre, normal, axes=AXES
            ),
        )

    def beside(store, column: int):
        """A transform that puts a store in its own column along x."""
        world = viewer.scene.dims.world_coordinate_system
        return AffineTransform.from_axis_map(
            store.data_coordinate_systems[0],
            world,
            axis_map={name: name for name in AXES},
            translation={"x": column * SIDE * 1.2},
        )

    image_visual = viewer.add_image_multiscale(
        image_store,
        appearance=MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="iso", iso_threshold=0.5
        ),
        name="image (ISO)",
        transform=beside(image_store, 0),
        clipping_planes=half(image_store, (0, 0, 1)),
        controls=MultiscaleImageControlsConfig(
            appearance=["visible", "render_mode", "iso_threshold", "lod_bias"],
            clipping_controls=True,
            loading_indicator=True,
        ),
    )
    labels_visual = viewer.add_labels_multiscale(
        labels_store,
        MultiscaleLabelsAppearance(),
        name="labels",
        transform=beside(labels_store, 1),
        clipping_planes=half(labels_store, (0, 1, 1)),
        controls=MultiscaleLabelsControlsConfig(
            appearance=True, clipping_controls=True, loading_indicator=True
        ),
    )
    mesh_visual = viewer.add_multiscale_mesh(
        mesh_store,
        MeshFlatAppearance(color=(0.9, 0.6, 0.2, 1.0), side="both"),
        name="mesh",
        transform=beside(mesh_store, 2),
        clipping_planes=half(mesh_store, (0, 0, 1)),
        # Keep the coarsest level beside the finest.
        lod=GeometryLodConfig(coarse_level=None),
        controls=MeshControlsConfig(
            appearance=True, clipping_controls=True, section_controls=True
        ),
    )

    # The bounding box of each visual's whole data.  It is not clipped, so it
    # shows how much of the visual a plane has removed.
    for _visual in (image_visual, labels_visual, mesh_visual):
        _visual.aabb.enabled = True

    display(
        viewer,
        Layout(
            center=build_canvas_widget(
                viewer, axis_values_from_viewer(viewer), canvas_size=(480, 480)
            ),
            right_dock=AppearanceControls(presentation="collapsible_sections"),
            right_dock_min_width=360,
        ),
        fit="ready",
    )
    return


if __name__ == "__main__":
    app.run()
