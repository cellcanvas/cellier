"""Full affine design v2, Phase 2: foundations with no axis-aligned change.

* the mirror fix in the in-memory image volume shader (K14);
* ``norm_size`` and the 2D multiscale LOD off the axis-aligned branch
  (K9, K10), which equal today's values for scaled permutations.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render.visuals._image import (
    _displayed_submatrix,
    _norm_size_from_transform,
    _world_to_voxel_scale_2d,
)
from cellier.transform import AffineTransform, Axis, CoordinateSystem
from tests.render._affine import (
    BLOCK_HALF,
    LANDMARK_PX,
    LINEARS,
    P0,
    landmark_error,
    rot,
    to_world,
    world_matrix,
)


def _affine(linear: np.ndarray) -> AffineTransform:
    axes = tuple(Axis(name=n, axis_type="space") for n in "zyx")
    data = CoordinateSystem(name="data", axes=axes)
    world = CoordinateSystem(name="world", axes=axes)
    return AffineTransform.from_matrix(world_matrix(linear), data, world)


# ---------------------------------------------------------------------------
# K14: a mirrored in-memory image volume draws in 3D
# ---------------------------------------------------------------------------


def _camera_inside(camera, matrix) -> None:
    """Put the camera inside the volume's box, looking at the landmark."""
    eye = to_world(matrix, P0 + np.array([0.0, -6.0, -8.0]))
    target = to_world(matrix, P0)
    camera.local.position = tuple(eye[::-1])
    camera.look_at(tuple(target[::-1]))


@pytest.mark.parametrize("inside", [False, True], ids=["outside", "inside"])
def test_a_mirrored_image_volume_draws_its_landmark(tmp_path, inside):
    """Before the fix the box culled every face and nothing drew (inv. 4.4)."""
    error = landmark_error(
        "image",
        "flip_x",
        "3d",
        tmp_path,
        camera_setup=_camera_inside if inside else None,
    )
    assert error < LANDMARK_PX


def test_the_inside_camera_really_is_inside_the_box():
    """The ``inside`` case above is meaningful only if the eye is in the data."""
    eye_data = P0 + np.array([0.0, -6.0, -8.0])
    assert np.all(eye_data > 0) and np.all(eye_data < np.array([24, 32, 40]) - 1)
    assert BLOCK_HALF < 6.0


# ---------------------------------------------------------------------------
# K9: norm_size
# ---------------------------------------------------------------------------

DATASET_XYZ = np.array([40.0, 32.0, 24.0])
SCALED_PERMUTATIONS = {
    "identity": np.eye(3),
    "scale": np.diag([2.0, 1.0, 1.5]),
    "anisotropic": np.diag([4.0, 0.25, 1.0]),
    "flip_x": np.diag([1.0, 1.0, -1.0]),
}


@pytest.mark.parametrize("name", sorted(SCALED_PERMUTATIONS))
def test_norm_size_general_branch_equals_the_axis_aligned_one(name, monkeypatch):
    """For a scaled permutation the column norms are today's ``|entry|``."""
    import cellier.render.visuals._image as image_module

    transform = _affine(SCALED_PERMUTATIONS[name])
    today = _norm_size_from_transform(transform, (0, 1, 2), DATASET_XYZ)
    monkeypatch.setattr(image_module, "_is_axis_aligned_view", lambda *a: False)
    general = _norm_size_from_transform(transform, (0, 1, 2), DATASET_XYZ)
    assert np.array_equal(today, general)


def test_norm_size_under_rotation_is_the_unrotated_one():
    """A rotation is a similarity of the proxy box: column norms ignore it."""
    scale = np.diag([2.0, 1.0, 1.5])
    unrotated = _norm_size_from_transform(_affine(scale), (0, 1, 2), DATASET_XYZ)
    rotated = _norm_size_from_transform(
        _affine(LINEARS["general"] @ np.diag([1.0, 1.0, 1.5])), (0, 1, 2), DATASET_XYZ
    )
    np.testing.assert_allclose(rotated, unrotated, rtol=1e-12)
    for angle in (30.0, 60.0):
        turned = _norm_size_from_transform(
            _affine(rot(1, 2, angle) @ scale), (0, 1, 2), DATASET_XYZ
        )
        np.testing.assert_allclose(turned, unrotated, rtol=1e-12)


# ---------------------------------------------------------------------------
# K10: the 2D multiscale LOD
# ---------------------------------------------------------------------------


def _homogeneous(block: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2, :2] = block
    return matrix


@pytest.mark.parametrize("name", sorted(SCALED_PERMUTATIONS))
def test_lod_general_branch_equals_the_axis_aligned_one(name):
    sub = _displayed_submatrix(_affine(SCALED_PERMUTATIONS[name]), (1, 2))
    today = _world_to_voxel_scale_2d(sub, axis_aligned=True)
    general = _world_to_voxel_scale_2d(sub, axis_aligned=False)
    assert general == pytest.approx(today, rel=1e-6)


def test_the_axis_aligned_branch_is_todays_expression():
    sub = _displayed_submatrix(_affine(np.diag([2.0, 1.0, 1.5])), (1, 2))
    nd2 = sub.shape[0] - 1
    today = float(np.prod(1.0 / np.abs(np.diag(sub[:nd2, :nd2]))) ** (1.0 / nd2))
    assert _world_to_voxel_scale_2d(sub, axis_aligned=True) == today


@pytest.mark.parametrize("angle", [30.0, 60.0])
def test_lod_under_in_plane_rotation_equals_the_unrotated_lod(angle):
    """The diagonal would read ``cos(theta)`` and pick too fine a level."""
    scale = np.diag([3.0, 0.5])
    theta = np.deg2rad(angle)
    turn = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    level0 = _world_to_voxel_scale_2d(_homogeneous(scale), axis_aligned=True)
    rotated = _world_to_voxel_scale_2d(_homogeneous(turn @ scale), axis_aligned=False)
    assert rotated == pytest.approx(level0, rel=1e-6)
