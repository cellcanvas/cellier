# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cellier[pyside]",
#     "scipy",
#     "tensorstore",
# ]
# ///

"""A MaskedImage rendering of an ellipsoid structure.

The mask is the ellipsoid's distance map and the value is a set of concentric
bands, so the iso threshold lights a surface of one band.  Run the file to open
the viewer; :func:`build_masked` takes the scene's options as keyword
arguments for tests and scripts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

#: The ellipsoid's semi-axes, ``(z, y, x)``, as fractions of the side.
SEMI_AXES = (110 / 256, 90 / 256, 70 / 256)
#: Width of one value band, in mask voxels.
BAND_WIDTH = 8.0
#: Pyramid levels of a lazy store.
N_LEVELS = 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def ellipsoid_inside(side: int, scale: int = 1) -> np.ndarray:
    """The ellipsoid as a bool volume of ``side * scale`` voxels a side.

    Voxel ``i`` of the scaled grid sits at mask coordinate
    ``(i + 0.5) / scale - 0.5``, so every scale describes the same shape.
    """
    n = side * scale
    centre = (side - 1) / 2.0
    coords = (np.arange(n, dtype=np.float32) + 0.5) / scale - 0.5 - centre
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij", sparse=True)
    a, b, c = (s * side for s in SEMI_AXES)
    return (z / a) ** 2 + (y / b) ** 2 + (x / c) ** 2 <= 1.0


def example_arrays(
    side: int = 256, value_scale: int = 1, band_width: float = BAND_WIDTH
) -> tuple[np.ndarray, np.ndarray]:
    """``(mask, value)``, ``(z, y, x)`` float32, each on its own grid.

    Distances are in mask voxels on both grids, so a value voxel's band is
    the band of the mask's distance at the same point.

    Parameters
    ----------
    side : int
        Voxels a side of the mask.
    value_scale : int
        The value's voxels per mask voxel on every axis.
    band_width : float
        Width of one band, in mask voxels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The mask (``side`` cubed) and the value (``side * value_scale``
        cubed).
    """
    from scipy import ndimage as ndi

    inside = ellipsoid_inside(side)
    mask = ndi.distance_transform_edt(inside).astype(np.float32)
    if value_scale == 1:
        distance = mask
        value_inside = inside
    else:
        value_inside = ellipsoid_inside(side, value_scale)
        distance = (
            ndi.distance_transform_edt(value_inside).astype(np.float32) / value_scale
        )
    value = np.where(value_inside, np.floor(distance / band_width) + 1.0, 0.0)
    return mask, value.astype(np.float32)


def n_bands(side: int, band_width: float = BAND_WIDTH) -> int:
    """The number of bands the deepest point of the ellipsoid reaches."""
    return int(np.ceil(min(SEMI_AXES) * side / band_width)) + 1


def value_to_mask_matrix(value_scale: int) -> np.ndarray:
    """Value level-0 voxels to mask level-0 voxels, 4x4, ``(z, y, x)``.

    A scale of ``1 / value_scale`` with the half-voxel shift that keeps voxel
    centres on the same points.
    """
    matrix = np.eye(4)
    if value_scale != 1:
        factor = 1.0 / value_scale
        matrix[:3, :3] *= factor
        matrix[:3, 3] = 0.5 * factor - 0.5
    return matrix


def _block_mean(array: np.ndarray) -> np.ndarray:
    """*array* downsampled 2x on every axis by mean, odd edges rounded up.

    An odd edge is padded with its last voxel before halving, so every level
    covers the whole extent of the finest.
    """
    pad = [(0, n % 2) for n in array.shape]
    padded = np.pad(array, pad, mode="edge")
    d, h, w = (n // 2 for n in padded.shape)
    return padded.reshape(d, 2, h, 2, w, 2).mean(axis=(1, 3, 5), dtype=np.float32)


def write_pyramid(root: Path, data: np.ndarray, n_levels: int = N_LEVELS) -> None:
    """Write *data* as a zarr v3 pyramid ``s0 .. s<n-1>``, mean downsampled."""
    import tensorstore as ts

    level = np.asarray(data, dtype=np.float32)
    for k in range(n_levels):
        array = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(root / f"s{k}")},
                "metadata": {
                    "shape": list(level.shape),
                    "data_type": "float32",
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {
                            "chunk_shape": [min(n, 64) for n in level.shape]
                        },
                    },
                },
                "create": True,
                "delete_existing": True,
            }
        ).result()
        array[...].write(level).result()
        if k + 1 < n_levels:
            level = _block_mean(level)


def open_pyramid(root: Path, name: str, n_levels: int = N_LEVELS):
    """Open a pyramid of :func:`write_pyramid` as a multiscale store."""
    from cellier.data import MultiscaleZarrDataStore

    factors = [2.0**k for k in range(n_levels)]
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=[f"s{k}" for k in range(n_levels)],
        level_scales=[(f, f, f) for f in factors],
        level_translations=[tuple([(f - 1.0) / 2.0] * 3) for f in factors],
        name=name,
    )


def build_stores(side: int, value_scale: int, lazy: bool):
    """The mask and value stores, in memory or as temporary zarr pyramids."""
    from cellier.data import ImageMemoryStore

    mask, value = example_arrays(side, value_scale)
    if not lazy:
        return (
            ImageMemoryStore(data=mask, name="mask"),
            ImageMemoryStore(data=value, name="value"),
        )
    root = Path(tempfile.mkdtemp(prefix="cellier_masked_image_"))
    write_pyramid(root / "mask", mask)
    write_pyramid(root / "value", value)
    return open_pyramid(root / "mask", "mask"), open_pyramid(root / "value", "value")


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


#: The MaskedImage's rows of the image control: every one it has.
MASKED_FIELDS = [
    "visible",
    "color_map",
    "clim",
    "opacity",
    "iso_threshold",
    "surface_thickness",
    "surface_placement",
    "surface_projection",
    "surface_min_width_px",
    "band_rays",
    "band_cap",
    "target_bricks",
]


def build_masked(
    *,
    side: int = 256,
    lazy: bool = False,
    value_scale: int = 1,
    threshold: float = 4.0,
    thickness: float = 0.0,
    target_bricks: str = "visible",
    budget_mib: int = 1100,
    gui: str = "qt",
):
    """The viewer, with the two images as one MaskedImage.

    Parameters
    ----------
    side : int
        Voxels a side of the mask.
    lazy : bool
        Write both images to temporary zarr pyramids and open those, instead
        of holding them in memory.
    value_scale : int
        The value's voxels per mask voxel on every axis.
    threshold : float
        The mask's iso threshold, in voxels of distance.
    thickness : float
        The surface thickness, in world units.
    target_bricks : str
        ``"visible"`` loads every brick in view, ``"surface"`` only the
        bricks near the surface (needs ``lazy`` for a coarse level to plan
        from).
    budget_mib : int
        Each image's GPU budget, in MiB.
    gui : str
        The GUI toolkit.
    """
    from cellier.convenience import Viewer
    from cellier.convenience.gui import (
        ClippingPlanesControl,
        DatasetInfoControl,
        ImageControl,
        LevelOfDetailControl,
        LoadingIndicator,
    )
    from cellier.scene.dims import spatial_axes
    from cellier.visuals import (
        MaskedImageAppearance,
        MaskedImageRenderConfig,
        MaskedImageSingleAppearance,
    )

    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui=gui
    )
    mask_store, value_store = build_stores(side, value_scale, lazy)
    bands = n_bands(side)
    viewer.add_masked_image(
        mask_store,
        value_store,
        value_to_mask=value_to_mask_matrix(value_scale),
        appearance=MaskedImageAppearance(target_bricks=target_bricks),
        single=MaskedImageSingleAppearance(
            color_map="tab10",
            clim=(0.0, float(bands)),
            iso_threshold=threshold,
            surface_thickness=thickness,
        ),
        render_config=MaskedImageRenderConfig(
            gpu_budget_bytes=budget_mib * 1024**2,
            value_gpu_budget_bytes=budget_mib * 1024**2,
        ),
        name="masked image",
        controls=[
            ImageControl(
                fields=MASKED_FIELDS,
                decimals=1,
                # The threshold is a distance in mask voxels and the
                # thickness a length in world units (one voxel here).
                threshold_range=(0.0, float(bands * BAND_WIDTH)),
                thickness_range=(0.0, 4.0 * BAND_WIDTH),
            ),
            LevelOfDetailControl(),
            LoadingIndicator(),
            ClippingPlanesControl(),
            DatasetInfoControl(),
        ],
    )
    viewer.set_slice_positions({0: (side - 1) / 2.0})
    return viewer


def main() -> None:
    """Open the example in a Qt window."""
    from cellier.convenience import (
        AppearanceControls,
        Layout,
        axis_values_from_viewer,
        run,
    )
    from cellier.convenience.gui import build_canvas_widget

    viewer = build_masked()
    canvas = build_canvas_widget(viewer, axis_values_from_viewer(viewer))
    run(viewer, Layout(center=canvas, left_dock=AppearanceControls()), fit="ready")


if __name__ == "__main__":
    main()
