"""The synthetic MaskedImage example's data.

The mask is an ellipsoid's distance transform, zero outside; the value is one
integer per ``BAND_WIDTH`` of that distance, counted from 1 at the boundary.
The expected numbers are analytic: the deepest point of the ellipsoid is its
smallest semi-axis deep.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = (
    Path(__file__).parents[2]
    / "examples"
    / "masked_image"
    / "masked_image_synthetic.py"
)


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("masked_image_synthetic", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arrays(example):
    return example.example_arrays(64)


def test_mask_is_zero_outside_and_deepest_at_centre(example, arrays):
    mask, _ = arrays
    inside = example.ellipsoid_inside(64)
    assert mask.dtype == np.float32
    assert mask.shape == (64, 64, 64)
    assert np.all(mask[~inside] == 0.0)
    assert np.all(mask[inside] >= 1.0)
    # The smallest semi-axis (x) is 70/256 of the side: 17.5 voxels at 64.
    deepest = min(example.SEMI_AXES) * 64
    assert abs(float(mask.max()) - deepest) <= 1.0
    # The deepest points are a ridge along the long axis (z), centred in y, x.
    peak = np.unravel_index(np.argmax(mask), mask.shape)
    assert all(abs(p - 31.5) <= 1.0 for p in peak[1:])
    assert mask[:, 31:33, 31:33].max() == mask.max()


def test_value_is_the_band_of_the_distance(example, arrays):
    mask, value = arrays
    inside = mask > 0
    expected = np.floor(mask[inside] / example.BAND_WIDTH) + 1.0
    np.testing.assert_array_equal(value[inside], expected)
    assert np.all(value[~inside] == 0.0)
    assert set(np.unique(value)) == {0.0, 1.0, 2.0, 3.0}
    assert example.n_bands(64) == 4


@pytest.mark.parametrize("iso", [1.0, 4.0, 9.5, 12.0])
def test_surface_at_iso_lies_in_one_band(example, arrays, iso):
    """Voxels just inside the iso surface have band ``floor(iso / 8) + 1``."""
    mask, value = arrays
    shell = (mask >= iso) & (mask < iso + 0.5)
    assert shell.any()
    band = np.floor(iso / example.BAND_WIDTH) + 1.0
    np.testing.assert_array_equal(np.unique(value[shell]), [band])


def test_value_scale_two_shares_the_shape(example):
    mask, value = example.example_arrays(32, value_scale=2)
    assert mask.shape == (32, 32, 32)
    assert value.shape == (64, 64, 64)
    matrix = example.value_to_mask_matrix(2)
    # Value voxel j sits at mask coordinate (j + 0.5) / 2 - 0.5.
    np.testing.assert_allclose(matrix[:3, :3], np.eye(3) * 0.5)
    np.testing.assert_allclose(matrix[:3, 3], [-0.25] * 3)
    # The two grids agree on inside and outside at the mask's voxel centres
    # away from the boundary, and on the band deep inside.
    coarse = value.reshape(32, 2, 32, 2, 32, 2).max(axis=(1, 3, 5))
    deep = mask >= 2.0
    assert np.all(coarse[deep] >= 1.0)
    outside = example.ellipsoid_inside(32, 1) == 0
    far = outside & (coarse == 0)
    assert far.sum() > 0.9 * outside.sum()
    # Deep inside both grids agree on the band within one.
    band = np.floor(mask[deep] / example.BAND_WIDTH) + 1.0
    assert np.all(np.abs(coarse[deep] - band) <= 1.0)


def test_value_to_mask_is_identity_at_scale_one(example):
    np.testing.assert_array_equal(example.value_to_mask_matrix(1), np.eye(4))


def test_pyramid_levels_round_up(example):
    """A coarse level covers the finest level's whole extent."""
    level = np.ones((5, 7, 9), np.float32)
    half = example._block_mean(level)
    assert half.shape == (3, 4, 5)
    np.testing.assert_array_equal(half, 1.0)
