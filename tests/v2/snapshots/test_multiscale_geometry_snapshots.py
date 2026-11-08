"""Per-level geometry of a multiscale image visual, pinned for three views.

Each test builds a slot's geometry the way ``GFXMultiscaleImageVisual``
does, through ``_slot_geometries``, for an anisotropic pyramid whose levels
carry per-axis scales and translations, and compares the result with a JSON
file beside this module:

    - orthoviewer_3d : 3D data, displayed_axes=(0, 1, 2)
    - multiscale_paint_2d : 2D data, displayed_axes=(0, 1)
    - orthoviewer_2d_in_3d : 3D data seen in 2D, displayed_axes=(1, 2)

Pinned: the data axes the geometry is built over, the level shapes, the
per-level scale and translation vectors in data order and in shader order
(reversed), the level scale factors, the base block layout, and the square
displayed block of the visual's transform (``_displayed_submatrix``).
Anisotropic values make an axis mix-up or a missing reversal show.

The JSON files are maintained by hand and are never regenerated: a mismatch
is a change in the geometry to explain, not a file to rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tests._v2 import level_transforms as _levels
from tests._v2 import scale_and_translation

from cellier.render.visuals._image import (
    ImageGeometry3D,
    MultiscaleBrickLayout3D,
    _displayed_submatrix,
    _slot_geometries,
)

SNAPSHOT_DIR = Path(__file__).parent

# Shared by the two scenarios on 3D data.
_LEVEL_SHAPES_3D = [(64, 256, 256), (32, 128, 128), (16, 64, 64)]
_LEVEL_SCALES_3D = [(1.0, 1.0, 1.0), (2.0, 4.0, 8.0), (4.0, 16.0, 64.0)]
_LEVEL_TRANSLATIONS_3D = [(0.0, 0.0, 0.0), (0.5, 1.5, 2.5), (1.0, 3.0, 5.0)]


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


def _assert_matches(name: str, snapshot: dict) -> None:
    """Compare *snapshot* with ``<name>.json``, byte for byte."""
    path = SNAPSHOT_DIR / f"{name}.json"
    serialised = json.dumps(snapshot, indent=2, sort_keys=True)
    expected = path.read_text().rstrip("\n")
    assert serialised == expected, (
        f"Snapshot mismatch for {name}.\n"
        f"--- expected ---\n{expected}\n--- actual ---\n{serialised}"
    )


def _capture_geometry(geom: MultiscaleBrickLayout3D | ImageGeometry3D) -> dict:
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
    """3D data displayed in 3D builds the volume geometry only."""
    transform = scale_and_translation((4.0, 1.0, 1.0), (10.0, 20.0, 30.0))
    displayed_axes = (0, 1, 2)

    volume, image_2d = _slot_geometries(
        transform,
        _LEVEL_SHAPES_3D,
        _levels(_LEVEL_SCALES_3D, _LEVEL_TRANSLATIONS_3D),
        {"2d", "3d"},
        displayed_axes,
        32,
    )
    assert image_2d is None

    snapshot = {
        "scenario": "orthoviewer_3d",
        "displayed_axes": list(displayed_axes),
        "axes_3d": list(volume.fetch_axes),
        "volume_geometry": _capture_geometry(volume),
        "slice_request_sub_3d_matrix": _round(
            _displayed_submatrix(transform, displayed_axes)
        ),
    }
    _assert_matches("orthoviewer_3d", snapshot)


# ---------------------------------------------------------------------------
# Scenario: multiscale_paint_2d
# ---------------------------------------------------------------------------


def test_snapshot_multiscale_paint_2d():
    """2D data builds the tile geometry only."""
    transform = scale_and_translation((1.0, 1.0), (5.0, 7.0))
    displayed_axes = (0, 1)

    volume, image_2d = _slot_geometries(
        transform,
        [(256, 256), (128, 128), (64, 64)],
        _levels(
            [(1.0, 1.0), (2.0, 4.0), (4.0, 16.0)],
            [(0.0, 0.0), (0.5, 1.5), (1.0, 3.0)],
        ),
        {"2d"},
        displayed_axes,
        64,
    )
    assert volume is None

    snapshot = {
        "scenario": "multiscale_paint_2d",
        "displayed_axes": list(displayed_axes),
        "axes_2d": list(image_2d.fetch_axes),
        "image_geometry_2d": _capture_geometry(image_2d),
        "slice_request_sub_2d_matrix": _round(
            _displayed_submatrix(transform, displayed_axes)
        ),
    }
    _assert_matches("multiscale_paint_2d", snapshot)


# ---------------------------------------------------------------------------
# Scenario: orthoviewer_2d_in_3d
# ---------------------------------------------------------------------------


def test_snapshot_orthoviewer_2d_in_3d():
    """3D data displayed in 2D builds the tile geometry over the (y, x) axes."""
    transform = scale_and_translation((4.0, 1.0, 1.0), (10.0, 20.0, 30.0))
    displayed_axes = (1, 2)

    volume, image_2d = _slot_geometries(
        transform,
        _LEVEL_SHAPES_3D,
        _levels(_LEVEL_SCALES_3D, _LEVEL_TRANSLATIONS_3D),
        {"2d", "3d"},
        displayed_axes,
        32,
    )
    assert volume is None

    snapshot = {
        "scenario": "orthoviewer_2d_in_3d",
        "displayed_axes": list(displayed_axes),
        "axes_2d": list(image_2d.fetch_axes),
        "image_geometry_2d": _capture_geometry(image_2d),
    }
    _assert_matches("orthoviewer_2d_in_3d", snapshot)
