"""Volume shading follows the world normal (full affine design v2, Phase 3, D9).

The three cellier volume shaders march their rays in object space (voxel
space for in-memory labels, the normalized proxy box for the multiscale
families) and light the surface hit in world space.  The oracle is
analytic (D10): the data is a ball of radius ``R``, so under ``world = L (d
- c) + c`` its surface is the ellipsoid ``|A (w - c)| = R`` with ``A =
L^-1``.  Each pixel's ray is cast against that ellipsoid and the world
normal there, ``A^T A (w - c)``, is compared with the rendered brightness.
The shaders light with a headlight, so brightness is monotone in ``|n .
v|``: the Spearman correlation must reach ``SHADING_MIN``.

Hypotheses compared against the correct one, for the negative controls and
the multiscale image's ``iso`` (whose Phong highlight keeps its absolute
correlation lower):

* ``object_data`` -- the data normal lit against the view direction carried
  into data space (what in-memory labels did before);
* ``object_proxy`` -- the same in the multiscale proxy space, data scaled
  per axis by ``norm_size`` (what the brick shaders did before).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    LINEARS,
    bind,
    load,
    multiscale_store,
    new_viewer,
    render,
)

SHAPE = (36, 44, 52)
C = (np.asarray(SHAPE, dtype=np.float64) - 1.0) / 2.0
R = 12.0
SIZE = 320
#: D10's shading threshold, from the Phase 0 readout (measured 0.94-0.96).
SHADING_MIN = 0.90

ANISO_Z4 = np.diag([4.0, 1.0, 1.0])


def _matrix(linear: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = linear
    m[:3, 3] = C - linear @ C
    return m


def _grid() -> np.ndarray:
    axes = [np.arange(n, dtype=np.float64) for n in SHAPE]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1)


def _ball() -> np.ndarray:
    return (np.linalg.norm(_grid() - C, axis=-1) < R).astype(np.int32)


def _ramp() -> np.ndarray:
    """An image whose 0.5 iso-surface is the sphere of radius ``R``."""
    distance = np.linalg.norm(_grid() - C, axis=-1)
    return np.clip((R - distance) / 3.0 + 0.5, 0.0, 1.0).astype(np.float32)


def _add(viewer, kind: str, mode: str, matrix: np.ndarray, tmp) -> None:
    if kind == "labels":
        store = LabelMemoryStore(data=_ball(), name="l")
        viewer.add_labels(
            store,
            appearance=InMemoryLabelsAppearance(render_mode=mode),
            transform=bind(viewer, store, matrix),
        )
    elif kind == "labels_multiscale":
        store = multiscale_store(tmp, f"labels_{mode}", _ball())
        viewer.add_labels_multiscale(
            store,
            appearance=MultiscaleLabelsAppearance(force_level=1, render_mode=mode),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=bind(viewer, store, matrix),
        )
    elif kind == "image_multiscale":
        store = multiscale_store(tmp, f"image_{mode}", _ramp())
        viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(force_level=1),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode=mode, iso_threshold=0.5
            ),
            transform=bind(viewer, store, matrix),
        )
    elif kind == "image":
        store = ImageMemoryStore(data=_ramp(), name="i")
        viewer.add_image(
            store,
            single=InMemoryImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode=mode, iso_threshold=0.5
            ),
            transform=bind(viewer, store, matrix),
        )
    else:  # pragma: no cover - a typo in a parametrisation
        raise ValueError(kind)


def _frame(kind: str, mode: str, linear: np.ndarray, tmp):
    matrix = _matrix(linear)
    viewer = new_viewer("3d")
    try:
        _add(viewer, kind, mode, matrix, tmp)
        load(viewer)
        frame, _, camera = render(viewer, SIZE)
    finally:
        viewer.controller.close()
    return frame, camera, matrix


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _hypotheses(frame: np.ndarray, camera, matrix: np.ndarray) -> dict[str, float]:
    """Spearman of brightness with ``|n . v|`` under each normal hypothesis."""
    height, width = frame.shape[:2]
    rows, cols = np.mgrid[:height, :width]
    ndc = np.stack(
        [(cols + 0.5) / width * 2.0 - 1.0, 1.0 - (rows + 0.5) / height * 2.0], -1
    ).reshape(-1, 2)
    inverse = np.linalg.inv(np.asarray(camera.camera_matrix, dtype=np.float64))

    def unproject(z: float) -> np.ndarray:
        ones = np.ones((len(ndc), 1))
        h = np.concatenate([ndc, np.full((len(ndc), 1), z), ones], 1) @ inverse.T
        return h[:, :3] / h[:, 3:]

    near = unproject(0.0)
    direction = _unit(unproject(1.0) - near)
    # cellier zyx -> pygfx xyz: reverse both axes of the linear block.
    linear = matrix[:3, :3][::-1, ::-1]
    a = np.linalg.inv(linear)
    p = (near - C[::-1]) @ a.T
    q = direction @ a.T
    qa = (q * q).sum(1)
    qb = 2.0 * (p * q).sum(1)
    disc = qb * qb - 4.0 * qa * ((p * p).sum(1) - R * R)
    hit = disc > 0
    t = (-qb - np.sqrt(np.where(hit, disc, 0.0))) / (2.0 * qa)
    sphere = p + t[:, None] * q  # A (w - c): the point on the data sphere

    luminance = frame[..., :3].astype(np.float64).mean(-1).reshape(-1)
    mask = frame[..., :3].max(-1) > 20
    for _ in range(3):
        mask = (
            mask
            & np.roll(mask, 1, 0)
            & np.roll(mask, -1, 0)
            & np.roll(mask, 1, 1)
            & np.roll(mask, -1, 1)
        )
    use = hit & mask.reshape(-1)
    assert use.sum() > 2000, f"only {use.sum()} usable pixels drew"

    def rho(term: np.ndarray) -> float:
        return float(spearmanr(luminance[use], term[use])[0])

    out = {"correct": rho(np.abs((_unit(sphere @ a) * direction).sum(1)))}
    data_view = _unit(direction @ a.T)
    out["object_data"] = rho(np.abs((_unit(sphere) * data_view).sum(1)))
    # The proxy box scales each data axis by its column norm (norm_size).
    per_axis = np.linalg.norm(linear, axis=0)
    proxy_view = _unit((direction @ a.T) * per_axis)
    out["object_proxy"] = rho(np.abs((_unit(sphere / per_axis) * proxy_view).sum(1)))
    return out


# ---------------------------------------------------------------------------
# In-memory labels on z-anisotropic voxels: every shaded mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth_iso", "smooth_geometry"])
def test_anisotropic_labels_smooth_modes_follow_the_world_normal(tmp_path, mode):
    frame, camera, matrix = _frame("labels", mode, ANISO_Z4, tmp_path)
    result = _hypotheses(frame, camera, matrix)
    assert result["correct"] >= SHADING_MIN, result
    assert result["correct"] > result["object_data"], result


def test_anisotropic_labels_iso_categorical_faces_take_world_brightness(tmp_path):
    """Voxel faces are lit by their world normal and the world view direction.

    A face of data normal ``e_k`` has the world normal ``A^T e_k`` and is lit
    ``color * (0.3 + |n . v|)`` (``label_volume.wgsl``).  Under a diagonal
    ``L`` that normal is ``e_k`` again, so the faceted brightness takes one
    of three values per pixel -- each of which must be the world one; the
    voxel-space alternative (view direction carried into data space) gives
    three other values.
    """
    frame, camera, matrix = _frame("labels", "iso_categorical", ANISO_Z4, tmp_path)
    world, voxel = _face_errors(frame, camera, matrix)
    # Measured: world 0.24 levels, voxel 3.1 (the committed shaders: the
    # reverse, voxel 0.25).
    assert world < 1.0, (world, voxel)
    assert voxel > 2.0, (world, voxel)


def _face_errors(frame, camera, matrix) -> tuple[float, float]:
    """Median 8-bit error of the best per-pixel face prediction, world and voxel."""
    height, width = frame.shape[:2]
    rows, cols = np.mgrid[:height, :width]
    ndc = np.stack(
        [(cols + 0.5) / width * 2.0 - 1.0, 1.0 - (rows + 0.5) / height * 2.0], -1
    ).reshape(-1, 2)
    inverse = np.linalg.inv(np.asarray(camera.camera_matrix, dtype=np.float64))

    def unproject(z: float) -> np.ndarray:
        ones = np.ones((len(ndc), 1))
        h = np.concatenate([ndc, np.full((len(ndc), 1), z), ones], 1) @ inverse.T
        return h[:, :3] / h[:, 3:]

    direction = _unit(unproject(1.0) - unproject(0.0))
    a = np.linalg.inv(matrix[:3, :3][::-1, ::-1])
    world_normals = _unit(a.T)  # rows: A^T e_k, k over pygfx x, y, z
    world_terms = np.abs(direction @ world_normals.T)
    voxel_terms = np.abs(_unit(direction @ a.T))  # e_k . unit(A d)

    rgb = frame[..., :3].astype(np.float64).reshape(-1, 3)
    mask = (frame[..., :3].max(-1) > 20).reshape(-1)
    observed = rgb[mask]
    unclipped = [c for c in range(3) if np.percentile(observed[:, c], 99) < 250]
    channel = max(unclipped, key=lambda c: observed[:, c].mean())
    value = observed[:, channel]

    def srgb_to_linear(v):
        c = v / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(v):
        c = np.clip(v, 0.0, 1.0)
        return 255.0 * np.where(
            c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055
        )

    def best(terms: np.ndarray) -> float:
        # The label colour is the one free parameter: a 1-D search for the
        # one that best explains the frame, each pixel by its best face.
        terms = terms[mask]
        errors = [
            np.median(
                np.abs(linear_to_srgb(color * (0.3 + terms)) - value[:, None]).min(1)
            )
            for color in np.linspace(0.05, 1.5, 300)
        ]
        return float(min(errors))

    return best(world_terms), best(voxel_terms)


# ---------------------------------------------------------------------------
# Multiscale image and labels under shear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth_iso", "smooth_geometry"])
def test_sheared_multiscale_labels_follow_the_world_normal(tmp_path, mode):
    frame, camera, matrix = _frame(
        "labels_multiscale", mode, LINEARS["shear_yx"], tmp_path
    )
    result = _hypotheses(frame, camera, matrix)
    assert result["correct"] >= SHADING_MIN, result
    assert result["correct"] > result["object_proxy"], result


def test_sheared_multiscale_image_iso_follows_the_world_normal(tmp_path):
    """Phong's highlight keeps the absolute correlation near 0.7 for ``iso``.

    So the check is relative: the world normal explains the brightness
    better than the proxy-space one the shader lit with before, as it does
    for pygfx's own in-memory volume shader (the control).
    """
    frame, camera, matrix = _frame(
        "image_multiscale", "iso", LINEARS["shear_yx"], tmp_path
    )
    result = _hypotheses(frame, camera, matrix)
    control, control_camera, _ = _frame("image", "iso", LINEARS["shear_yx"], tmp_path)
    reference = _hypotheses(control, control_camera, matrix)
    assert result["correct"] > result["object_proxy"] + 0.02, result
    assert result["correct"] >= reference["correct"] - 0.03, (result, reference)


# ---------------------------------------------------------------------------
# Isotropic, axis-aligned: unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["labels", "labels_multiscale"])
def test_a_similarity_does_not_change_the_shading(tmp_path, kind):
    """World and object space agree up to scale for a similarity.

    An isotropic ball drawn at unit voxels and at twice that size, each with
    its own fitted camera, frames the same picture: lighting depends on
    directions only, which a uniform scale leaves alone.
    """
    one, _, _ = _frame(kind, "smooth_iso", np.eye(3), tmp_path / "one")
    two, _, _ = _frame(kind, "smooth_iso", 2.0 * np.eye(3), tmp_path / "two")
    both = (one[..., :3].max(-1) > 20) & (two[..., :3].max(-1) > 20)
    diff = np.abs(one[..., :3].astype(int) - two[..., :3].astype(int)).max(-1)
    assert np.percentile(diff[both], 99) <= 2
