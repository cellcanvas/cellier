"""Large OME-Zarr viewer with a clipping plane, for testing clipping at scale.

Opens a multiscale OME-Zarr store (by default ``ExpA_VIP_ASLM_on.zarr`` next to
this script, a ~1937 x 2048 x 2048 uint16 light-sheet volume) as one multiscale
image.  A clipping plane is attached through the middle of the volume, and a
gizmo is put on it once the first data is drawn.

- Switch the render mode to ``iso`` or ``smooth_iso`` to use the threshold.
- The "Clipping planes" control adds, enables and moves planes.
- Drag the arrow of the gizmo to slide the plane, the rings to tilt it.
- The 2D / 3D toggle is part of the dims control under the canvas.
- Slicing follows the camera and dims on its own once they settle.

Run::

    .venv/bin/python scripts/v2/integration_2d_3d/example_combined_camera_redraw.py

Options: ``--zarr-path PATH``, ``--debug-log [SPEC]``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from uuid import uuid4

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
from cellier.transform import AffineTransform, Axis, DataCoordinateSystem
from cellier.visuals import (
    ClippingPlane,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
)

AXES = ("z", "y", "x")
BLOCK_SIZE = 32
GPU_BUDGET = 1024 * 1024**2  # 1 GiB brick / tile cache
LOD_BIAS = 1.0

ZARR_PATH = pathlib.Path(__file__).parent / "ExpA_VIP_ASLM_on.zarr"


# ---------------------------------------------------------------------------
# Debug-logging CLI helper (power-user per-category:level syntax)
# ---------------------------------------------------------------------------


def _setup_debug_logging(spec: str) -> None:
    """Parse ``--debug-log`` spec and configure cellier loggers.

    Supports several forms::

        "all"                           -> all categories at DEBUG
        "all:info"                      -> all categories at INFO
        "perf,cache"                    -> perf+cache at DEBUG
        "perf:info,cache:debug"         -> per-category levels

    A bare category name (no colon) defaults to DEBUG.
    """
    import logging as _logging

    from cellier.logging import _CATEGORY_MAP, enable_debug_logging

    level_names = {
        "debug": _logging.DEBUG,
        "info": _logging.INFO,
        "warning": _logging.WARNING,
    }

    overrides: dict[str, int] = {}
    default_level = _logging.DEBUG

    if spec in ("", "all"):
        pass
    elif spec.startswith("all:"):
        level_str = spec.split(":", 1)[1].strip().lower()
        default_level = level_names.get(level_str, _logging.DEBUG)
    else:
        for token in spec.split(","):
            token = token.strip()
            if ":" in token:
                cat, level_str = token.split(":", 1)
                overrides[cat.strip()] = level_names.get(
                    level_str.strip().lower(), _logging.DEBUG
                )
            else:
                overrides[token] = _logging.DEBUG

    cats = tuple(overrides) if overrides else tuple(_CATEGORY_MAP)
    enable_debug_logging(categories=cats)
    for cat in cats:
        logger = _CATEGORY_MAP.get(cat)
        if logger is not None:
            logger.setLevel(overrides.get(cat, default_level))


# ---------------------------------------------------------------------------
# Data store
# ---------------------------------------------------------------------------


def open_ome_zarr(zarr_path: pathlib.Path) -> tuple[MultiscaleZarrDataStore, tuple]:
    """Open an OME-Zarr multiscale group as a ``MultiscaleZarrDataStore``.

    The store's level-0 coordinates are voxel indices.  The OME physical scale
    of level 0 is returned separately, to be applied as the visual's transform.

    Returns
    -------
    store : MultiscaleZarrDataStore
    voxel_size : tuple of float
        Level-0 physical size of a voxel along each of ``AXES``.
    """
    meta = json.loads((zarr_path / "zarr.json").read_text())
    datasets = meta["attributes"]["ome"]["multiscales"][0]["datasets"]
    names = [d["path"] for d in datasets]
    scales = [
        next(t["scale"] for t in d["coordinateTransformations"] if t["type"] == "scale")
        for d in datasets
    ]
    voxel_size = tuple(scales[0])
    # Level-k scale relative to level 0, in level-0 voxels.
    level_scales = [tuple(s / s0 for s, s0 in zip(sc, voxel_size)) for sc in scales]
    # A level-k voxel centre sits at (scale - 1) / 2 level-0 voxels.
    level_translations = [tuple((f - 1.0) / 2.0 for f in sc) for sc in level_scales]

    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(zarr_path),
        scale_names=names,
        level_scales=level_scales,
        level_translations=level_translations,
        data_coordinate_system=DataCoordinateSystem(
            name="ome_zarr",
            datastore_id=uuid4(),
            axes=tuple(
                Axis(name=n, axis_type="space", sampling="discrete") for n in AXES
            ),
        ),
        name=zarr_path.stem,
    )
    return store, voxel_size


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args, build the viewer and show it."""
    parser = argparse.ArgumentParser(
        description="Large OME-Zarr viewer with a clipping plane"
    )
    parser.add_argument(
        "--zarr-path",
        type=pathlib.Path,
        default=ZARR_PATH,
        help="Path to the OME-Zarr multiscale group.",
    )
    parser.add_argument(
        "--debug-log",
        metavar="SPEC",
        default=None,
        const="all",
        nargs="?",
        help=(
            "Enable debug logging: comma-separated categories "
            "(perf, gpu, cache, slicer), optionally with :LEVEL, e.g. "
            "'perf:info,cache:debug'.  Omit the value for all at DEBUG."
        ),
    )
    args = parser.parse_args()

    if not (args.zarr_path / "zarr.json").exists():
        print(f"Error: OME-Zarr group not found at '{args.zarr_path}'")
        print("Fetch it with scripts/v2/integration_2d_3d/download_ome_zarr.py")
        sys.exit(1)

    if args.debug_log is not None:
        _setup_debug_logging(args.debug_log)

    print("Opening tensorstore stores via MultiscaleZarrDataStore ...")
    store, voxel_size = open_ome_zarr(args.zarr_path)
    print(f"  {store.n_levels} levels opened.")
    for i, shape in enumerate(store.level_shapes):
        print(f"  level {i}: shape={shape}")

    viewer = Viewer(spatial_axes(*AXES), dim="3d")

    # Voxel indices -> physical units, so the anisotropic volume is not squashed.
    transform = AffineTransform.from_axis_map(
        store.data_coordinate_systems[0],
        viewer.scene.dims.world_coordinate_system,
        axis_map={name: name for name in AXES},
        scale=dict(zip(AXES, voxel_size)),
    )

    # A plane through the middle of the volume, keeping the +x half.  It is
    # written in the store's level-0 (voxel) coordinates.
    shape0 = store.level_shapes[0]
    centre = tuple(n / 2.0 for n in shape0)
    plane = ClippingPlane.from_point_normal(
        store.data_coordinate_systems[0], centre, (0, 0, 1), axes=AXES
    )

    visual = viewer.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(lod_bias=LOD_BIAS),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis",
            clim=(0.0, 1500.0),
            render_mode="mip",
            iso_threshold=500.0,
        ),
        render_config=MultiscaleImageRenderConfig(
            block_size=BLOCK_SIZE, gpu_budget_bytes=GPU_BUDGET
        ),
        name="ExpA_VIP_ASLM_on",
        transform=transform,
        clipping_planes=(plane,),
        controls=MultiscaleImageControlsConfig(
            appearance=[
                "visible",
                "color_map",
                "clim",
                "render_mode",
                "iso_threshold",
                "lod_bias",
            ],
            clim_range=(0.0, 4000.0),
            clipping_controls=True,
            # How much of the plan is loaded: watch it while the plane moves.
            loading_indicator=True,
            dataset_info=True,
        ),
    )
    visual.aabb.enabled = True

    # The canvas default depth ranges (3D 1..8000, 2D -500..500) assume a small
    # scene.  Size them from the world extent of the volume instead: the far
    # plane must clear the camera distance plus the whole scene.
    extent = [n * v for n, v in zip(shape0, voxel_size)]
    diagonal = float(sum(e * e for e in extent) ** 0.5)
    canvas_widget = build_canvas_widget(
        viewer,
        axis_values_from_viewer(viewer),
        depth_range_3d=(diagonal / 1000.0, diagonal * 4.0),
        depth_range_2d=(-diagonal, diagonal),
    )
    layout = Layout(
        center=canvas_widget,
        right_dock=AppearanceControls(),
        right_dock_min_width=360,
    )

    # The gizmo needs the plane on screen: add it once the first data is drawn.
    viewer.on_ready(lambda: viewer.add_clipping_plane_gizmo(visual, plane))

    run(viewer, layout, fit="ready")


if __name__ == "__main__":
    main()
