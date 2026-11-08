"""Pre-refactor snapshots for the nD axis-handling refactor.

These tests capture the values produced by today's hardcoded axis
handling at the visual / geometry layer, for the two scenarios the
existing demos cover:

    - orthoviewer_3d : 3D OME-Zarr-shaped dataset, displayed_axes=(0,1,2)
    - multiscale_paint_2d : 2D dataset, displayed_axes=(0,1)

Each test reproduces the exact construction that
``GFXMultiscaleImageVisual.from_cellier_model`` performs (without going
through pygfx) and captures the resulting per-level shape tuples, scale
and translation vectors fed to the shader, the LOD-projected sub-
transform, and the base block-layout dimensions.

The snapshot files in ``snapshots/*.json`` are byte-compared against
fresh runs.  Any drift after a refactor phase signals either a
regression or an intentional value change requiring snapshot update.

These snapshots are a temporary refactor scaffold and will be deleted
at the end of Phase 6 once the unit tests and end-to-end nD test
provide long-term coverage.

Run with the env var ``CELLIER_UPDATE_SNAPSHOTS=1`` to (re)write the
JSON files in-place.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from tests._v2 import level_transforms as _levels
from tests._v2 import scale_and_translation

from cellier.render.visuals._image import (
    ImageGeometry3D,
    MultiscaleBrickLayout3D,
    _displayed_submatrix,
)

SNAPSHOT_DIR = Path(__file__).parent
UPDATE = os.environ.get("CELLIER_UPDATE_SNAPSHOTS", "") not in ("", "0", "false")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round(values, decimals: int = 6):
    """Recursively round numpy values for stable JSON comparison."""
    if isinstance(values, np.ndarray):
        return np.round(values, decimals).tolist()
    if isinstance(values, (list, tuple)):
        return [_round(v, decimals) for v in values]
    if isinstance(values, (np.floating, np.integer)):
        return float(np.round(float(values), decimals))
    if isinstance(values, float):
        return float(np.round(values, decimals))
    return values


def _assert_or_write(name: str, snapshot: dict) -> None:
    """Compare snapshot to the on-disk JSON, or rewrite it if UPDATE is set."""
    path = SNAPSHOT_DIR / f"{name}.json"
    serialised = json.dumps(snapshot, indent=2, sort_keys=True)
    if UPDATE or not path.exists():
        path.write_text(serialised + "\n")
        if not UPDATE:
            pytest.fail(
                f"Snapshot {name}.json did not exist; wrote initial copy. "
                f"Re-run to verify."
            )
        return
    expected = path.read_text().rstrip("\n")
    assert serialised == expected, (
        f"Snapshot mismatch for {name}.\n"
        f"--- expected ---\n{expected}\n--- actual ---\n{serialised}"
    )


def _replicate_from_cellier_model(
    level_shapes,
    level_transforms,
    displayed_axes,
    block_size,
    render_modes,
):
    """Mirror the geometry-construction half of ``from_cellier_model``."""
    if len(displayed_axes) == 3:
        axes_3d = displayed_axes
        axes_2d = displayed_axes[-2:]
    else:
        axes_3d = None
        axes_2d = displayed_axes

    volume_geometry = None
    if "3d" in render_modes and axes_3d is not None:
        shapes_3d = [tuple(s[ax] for ax in axes_3d) for s in level_shapes]
        volume_geometry = MultiscaleBrickLayout3D(
            level_shapes=shapes_3d,
            level_transforms=level_transforms,
            block_size=block_size,
            fetch_axes=axes_3d,
        )

    image_geometry_2d = None
    if "2d" in render_modes:
        shapes_2d_full = [tuple(s[ax] for ax in axes_2d) for s in level_shapes]
        level_shapes_2d = [(s[0], s[1]) for s in shapes_2d_full]
        image_geometry_2d = ImageGeometry3D(
            level_shapes=level_shapes_2d,
            block_size=block_size,
            n_levels=len(level_shapes),
            level_transforms=level_transforms,
            fetch_axes=axes_2d,
        )

    return axes_3d, axes_2d, volume_geometry, image_geometry_2d


def _capture_volume_geometry(geom: MultiscaleBrickLayout3D) -> dict:
    return {
        "level_shapes": [list(s) for s in geom.level_shapes],
        "scale_vecs_data": _round(geom._scale_vecs_data),
        "translation_vecs_data": _round(geom._translation_vecs_data),
        "scale_vecs_shader": _round(geom._scale_vecs_shader),
        "translation_vecs_shader": _round(geom._translation_vecs_shader),
        "level_scale_factors": _round(geom._level_scale_factors),
        "base_layout_volume_shape": list(geom.base_layout.volume_shape),
        "base_layout_block_size": int(geom.base_layout.block_size),
        "base_layout_grid_dims": list(geom.base_layout.grid_dims),
    }


def _capture_image_geometry_2d(geom: ImageGeometry3D) -> dict:
    return {
        "level_shapes": [list(s) for s in geom.level_shapes],
        "scale_vecs_data": _round(geom._scale_vecs_data),
        "translation_vecs_data": _round(geom._translation_vecs_data),
        "scale_vecs_shader": _round(geom._scale_vecs_shader),
        "translation_vecs_shader": _round(geom._translation_vecs_shader),
        "level_scale_factors": _round(geom._level_scale_factors),
        "base_layout_volume_shape": list(geom.base_layout.volume_shape),
        "base_layout_block_size": int(geom.base_layout.block_size),
        "base_layout_grid_dims": list(geom.base_layout.grid_dims),
    }


# ---------------------------------------------------------------------------
# Scenario: orthoviewer_3d
# ---------------------------------------------------------------------------


def test_snapshot_orthoviewer_3d():
    """3D dataset, 3D-displayed, anisotropic z to flush out reversal bugs."""
    # 3-level pyramid with anisotropic per-axis scales and per-axis
    # translations so that data-order and shader-order vectors differ
    # (i.e. the [2,1,0] reversal is observable in the snapshot).
    level_shapes = [(64, 256, 256), (32, 128, 128), (16, 64, 64)]
    level_transforms = _levels(
        [(1.0, 1.0, 1.0), (2.0, 4.0, 8.0), (4.0, 16.0, 64.0)],
        [(0.0, 0.0, 0.0), (0.5, 1.5, 2.5), (1.0, 3.0, 5.0)],
    )
    displayed_axes = (0, 1, 2)
    block_size = 32

    axes_3d, axes_2d, vol, img2d = _replicate_from_cellier_model(
        level_shapes=level_shapes,
        level_transforms=level_transforms,
        displayed_axes=displayed_axes,
        block_size=block_size,
        render_modes={"2d", "3d"},
    )

    # Slice-request 3D sub-transform.  ``select_axes`` went with v1 in
    # Phase 8; ``_displayed_submatrix`` reads the same entries off the v2
    # transform and returns the same square block.
    full_transform = scale_and_translation((4.0, 1.0, 1.0), (10.0, 20.0, 30.0))
    sub_3d = _displayed_submatrix(full_transform, displayed_axes[-3:])

    snapshot = {
        "scenario": "orthoviewer_3d",
        "displayed_axes": list(displayed_axes),
        "axes_3d": list(axes_3d),
        "axes_2d": list(axes_2d),
        "volume_geometry": _capture_volume_geometry(vol),
        "image_geometry_2d": _capture_image_geometry_2d(img2d),
        "slice_request_sub_3d_matrix": _round(sub_3d),
    }
    _assert_or_write("orthoviewer_3d", snapshot)


# ---------------------------------------------------------------------------
# Scenario: multiscale_paint_2d
# ---------------------------------------------------------------------------


def test_snapshot_multiscale_paint_2d():
    """2D-only multiscale paint demo state."""
    # Anisotropic per-axis scale + translation so the [1, 0] reversal in
    # ImageGeometry2D is observable in the snapshot.
    level_shapes = [(256, 256), (128, 128), (64, 64)]
    level_transforms = _levels(
        [(1.0, 1.0), (2.0, 4.0), (4.0, 16.0)],
        [(0.0, 0.0), (0.5, 1.5), (1.0, 3.0)],
    )
    displayed_axes = (0, 1)
    block_size = 64

    axes_3d, axes_2d, vol, img2d = _replicate_from_cellier_model(
        level_shapes=level_shapes,
        level_transforms=level_transforms,
        displayed_axes=displayed_axes,
        block_size=block_size,
        render_modes={"2d"},
    )
    assert axes_3d is None
    assert vol is None

    # Slice-request 2D sub-transform; see the 3D note above.
    full_transform = scale_and_translation((1.0, 1.0), (5.0, 7.0))
    sub_2d = _displayed_submatrix(full_transform, displayed_axes)

    snapshot = {
        "scenario": "multiscale_paint_2d",
        "displayed_axes": list(displayed_axes),
        "axes_2d": list(axes_2d),
        "image_geometry_2d": _capture_image_geometry_2d(img2d),
        "slice_request_sub_2d_matrix": _round(sub_2d),
    }
    _assert_or_write("multiscale_paint_2d", snapshot)


# ---------------------------------------------------------------------------
# Scenario: orthoviewer_2d_in_3d, 2D mode on 3D data (the [-2:] fallback)
# ---------------------------------------------------------------------------


def test_snapshot_orthoviewer_2d_in_3d():
    """3D dataset rendered in 2D mode: displayed_axes=(1, 2)."""
    # Anisotropic transforms so the [1, 0] reversal applied to the
    # selected (y, x) sub-transform is observable in the snapshot.
    level_shapes = [(64, 256, 256), (32, 128, 128), (16, 64, 64)]
    level_transforms = _levels(
        [(1.0, 1.0, 1.0), (2.0, 4.0, 8.0), (4.0, 16.0, 64.0)],
        [(0.0, 0.0, 0.0), (0.5, 1.5, 2.5), (1.0, 3.0, 5.0)],
    )
    displayed_axes = (1, 2)
    block_size = 32

    axes_3d, axes_2d, vol, img2d = _replicate_from_cellier_model(
        level_shapes=level_shapes,
        level_transforms=level_transforms,
        displayed_axes=displayed_axes,
        block_size=block_size,
        render_modes={"2d"},
    )
    assert axes_3d is None
    assert vol is None

    snapshot = {
        "scenario": "orthoviewer_2d_in_3d",
        "displayed_axes": list(displayed_axes),
        "axes_2d": list(axes_2d),
        "image_geometry_2d": _capture_image_geometry_2d(img2d),
    }
    _assert_or_write("orthoviewer_2d_in_3d", snapshot)
