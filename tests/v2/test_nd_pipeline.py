"""End-to-end nD correctness tests for the multiscale image pipeline.

These tests build ``MultiscaleBrickLayout3D`` / ``ImageGeometry3D`` and
fetch slices from a synthetic 5D dataset (T, C, Z, Y, X) where each voxel
encodes its own indices.  Decoding the fetched array verifies that:

- ``displayed_axes`` is honoured at the data store layer (correct shape
  and content for non-contiguous displayed axes like ``(0, 4)``).
- ``GFXMultiscaleImageVisual``-style geometry construction works for
  3D display where the displayed data axes are non-contiguous, e.g.
  ``displayed_axes=(0, 3, 4)`` selecting (T, Y, X) from a 5D dataset.
- The shader-order vectors built by MultiscaleBrickLayout3D / ImageGeometry3D
  are correct reversals of the displayed-axis-order inputs.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.image._image_requests import ChunkRequest
from cellier.render._spaces import select_axes
from cellier.render.visuals._image import (
    ImageGeometry3D,
    MultiscaleBrickLayout3D,
)
from tests._v2 import level_transforms

# Encode (t, c, z, y, x) into a single float so any extracted slice can
# be decoded back to its original indices.  Bases must exceed dimension
# sizes; we use a 5-axis Horner-style code with strict bases.  Values
# stay under 100_000 so they fit precisely in float32.
_BASES = (10_000, 1_000, 100, 10, 1)


def _encode(t: int, c: int, z: int, y: int, x: int) -> float:
    return float(t * _BASES[0] + c * _BASES[1] + z * _BASES[2] + y * _BASES[3] + x)


def _decode(value: float) -> tuple[int, int, int, int, int]:
    v = round(value)
    t = v // _BASES[0]
    v -= t * _BASES[0]
    c = v // _BASES[1]
    v -= c * _BASES[1]
    z = v // _BASES[2]
    v -= z * _BASES[2]
    y = v // _BASES[3]
    v -= y * _BASES[3]
    return t, c, z, y, v


@pytest.fixture(scope="module")
def encoded_5d_data() -> np.ndarray:
    """Build a (T=2, C=3, Z=4, Y=5, X=6) array where each voxel encodes its index."""
    shape = (2, 3, 4, 5, 6)
    data = np.zeros(shape, dtype=np.float32)
    for t in range(shape[0]):
        for c in range(shape[1]):
            for z in range(shape[2]):
                for y in range(shape[3]):
                    for x in range(shape[4]):
                        data[t, c, z, y, x] = _encode(t, c, z, y, x)
    return data


# ---------------------------------------------------------------------------
# Data store: nD axis selection round-trip
# ---------------------------------------------------------------------------


def _make_request(axis_selections):
    return ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=tuple(axis_selections),
    )


def test_5d_fetch_with_2d_displayed_axes_TX(encoded_5d_data):
    """Fetch (T, X) plane from a 5D store, slicing C=2, Z=1, Y=3."""
    store = ImageMemoryStore(data=encoded_5d_data)
    request = _make_request(
        [(0, 2), 2, 1, 3, (0, 6)]  # T full, C=2, Z=1, Y=3, X full
    )
    out = asyncio.run(store.get_data(request))
    # Expected shape: (T=2, X=6) in data axis order.
    assert out.shape == (2, 6)
    for t in range(2):
        for x in range(6):
            assert _decode(out[t, x]) == (t, 2, 1, 3, x)


def test_5d_fetch_with_3d_displayed_axes_TYX(encoded_5d_data):
    """Fetch (T, Y, X) volume from a 5D store, slicing C=1, Z=2."""
    store = ImageMemoryStore(data=encoded_5d_data)
    request = _make_request(
        [(0, 2), 1, 2, (0, 5), (0, 6)]  # T full, C=1, Z=2, Y full, X full
    )
    out = asyncio.run(store.get_data(request))
    assert out.shape == (2, 5, 6)
    # Spot-check a few cells.
    assert _decode(out[0, 0, 0]) == (0, 1, 2, 0, 0)
    assert _decode(out[1, 4, 5]) == (1, 1, 2, 4, 5)
    assert _decode(out[0, 2, 3]) == (0, 1, 2, 2, 3)


def test_5d_fetch_with_3d_non_contiguous_TZX(encoded_5d_data):
    """Fetch (T, Z, X) volume — non-contiguous selection skipping C and Y."""
    store = ImageMemoryStore(data=encoded_5d_data)
    request = _make_request(
        [(0, 2), 0, (0, 4), 4, (0, 6)]  # T full, C=0, Z full, Y=4, X full
    )
    out = asyncio.run(store.get_data(request))
    assert out.shape == (2, 4, 6)
    for t in range(2):
        for z in range(4):
            for x in range(6):
                assert _decode(out[t, z, x]) == (t, 0, z, 4, x)


# ---------------------------------------------------------------------------
# Geometry construction with non-contiguous displayed_axes
# ---------------------------------------------------------------------------


def _build_5d_level_setup():
    """Per-axis anisotropic 5D scales so axis-handling bugs are observable."""
    # Distinct scale per axis so the displayed-subset is clearly identifiable.
    full_scale = (10.0, 100.0, 1000.0, 10000.0, 100000.0)  # T, C, Z, Y, X
    full_translation = (1.0, 2.0, 3.0, 4.0, 5.0)
    # 3-level pyramid: level k has scale sv * 2^k.
    scales = [tuple(s * (2**k) for s in full_scale) for k in range(3)]
    translations = [tuple(t * (1 + k) for t in full_translation) for k in range(3)]
    # The geometry classes assert that level 0 is the identity.
    scales[0] = (1.0,) * 5
    translations[0] = (0.0,) * 5
    full_level_shapes = [(2, 3, 4, 5, 6), (2, 3, 4, 5, 6), (1, 2, 2, 3, 3)]
    return level_transforms(scales, translations), full_level_shapes


def test_volume_geometry_3d_non_contiguous_displayed_axes():
    """3D display with displayed_axes=(0, 3, 4) selects T, Y, X from 5D."""
    transforms, level_shapes = _build_5d_level_setup()
    displayed_axes = (0, 3, 4)  # T, Y, X

    # Replicate from_cellier_model's 3D geometry construction.  The transforms
    # are handed over whole and projected inside: a transform cannot be
    # rank-reduced without inventing two coordinate systems.
    shapes_3d = [select_axes(s, displayed_axes) for s in level_shapes]
    vg = MultiscaleBrickLayout3D(
        level_shapes=shapes_3d,
        level_transforms=transforms,
        block_size=4,
        fetch_axes=displayed_axes,
    )

    # Expected shapes per level: pick (T, Y, X) from each full shape.
    assert vg.level_shapes[0] == (2, 5, 6)
    assert vg.level_shapes[1] == (2, 5, 6)
    assert vg.level_shapes[2] == (1, 3, 3)

    # Level-1 transform is full_scale * 2 = (20, 200, 2000, 20000, 200000).
    # Displayed (T, Y, X) selects (20, 20000, 200000) in displayed-axis order.
    np.testing.assert_allclose(
        vg._scale_vecs_data[1], [20.0, 20000.0, 200000.0], atol=1e-3
    )
    # Shader order reverses to (X, Y, T) = (200000, 20000, 20).
    np.testing.assert_allclose(
        vg._scale_vecs_shader[1], [200000.0, 20000.0, 20.0], atol=1e-3
    )

    # Level-1 translation = full_translation * 2 = (2, 4, 6, 8, 10).
    # Selected (T, Y, X) = (2, 8, 10); shader order = (10, 8, 2).
    np.testing.assert_allclose(vg._translation_vecs_data[1], [2.0, 8.0, 10.0])
    np.testing.assert_allclose(vg._translation_vecs_shader[1], [10.0, 8.0, 2.0])


def test_image_geometry_2d_non_contiguous_displayed_axes():
    """2D display with displayed_axes=(0, 4) selects T, X from 5D."""
    transforms, level_shapes = _build_5d_level_setup()
    displayed_axes = (0, 4)  # T, X

    shapes_2d = [select_axes(s, displayed_axes) for s in level_shapes]
    ig = ImageGeometry3D(
        level_shapes=shapes_2d,
        block_size=2,
        n_levels=3,
        level_transforms=transforms,
        fetch_axes=displayed_axes,
    )

    assert ig.level_shapes[0] == (2, 6)
    assert ig.level_shapes[1] == (2, 6)
    assert ig.level_shapes[2] == (1, 3)

    # Level-2 transform is full_scale * 4 = (40, 400, 4000, 40000, 400000).
    # Selected (T, X) = (40, 400000); shader-order reverses to (400000, 40).
    np.testing.assert_allclose(ig._scale_vecs_data[2], [40.0, 400000.0], atol=1e-3)
    np.testing.assert_allclose(ig._scale_vecs_shader[2], [400000.0, 40.0], atol=1e-3)


# ---------------------------------------------------------------------------
# Dims validator: rejects display rank outside {2, 3}
# ---------------------------------------------------------------------------


def test_axis_aligned_selection_rejects_4_displayed_axes():
    from cellier.scene.dims import AxisAlignedSelection

    with pytest.raises(ValueError, match="length 2 or 3"):
        AxisAlignedSelection(displayed_axes=(0, 1, 2, 3))


def test_axis_aligned_selection_rejects_1_displayed_axis():
    from cellier.scene.dims import AxisAlignedSelection

    with pytest.raises(ValueError, match="length 2 or 3"):
        AxisAlignedSelection(displayed_axes=(0,))


def test_axis_aligned_selection_rejects_duplicates():
    from cellier.scene.dims import AxisAlignedSelection

    with pytest.raises(ValueError, match="distinct"):
        AxisAlignedSelection(displayed_axes=(0, 0))


def test_axis_aligned_selection_accepts_non_contiguous():
    from cellier.scene.dims import AxisAlignedSelection

    # (0, 4) is fine — non-contiguous but valid.
    sel = AxisAlignedSelection(displayed_axes=(0, 4))
    assert sel.displayed_axes == (0, 4)
