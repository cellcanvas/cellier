"""Labels volumes are lit in world space, not in their voxel space.

The volume shaders march their rays in object space -- voxel space for the
in-memory labels, the normalized proxy box for the multiscale ones -- and
used to light the surface there too.  That is right only when object ->
world is a similarity, so a labels sphere drawn with anisotropic voxels was
shaded as the sphere it is in voxel space, not as the ellipsoid it is in the
world.

The check is analytic and needs no reference render: each pixel's ray is
cast against the world ellipsoid, and the brightness the labels shader
writes there, ``color * (ambient + |n . v|)`` with a headlight, is predicted
from the ellipsoid's own world normal.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.visuals._label_memory import InMemoryLabelsAppearance
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._planes import scale_translation_transform, shoot

SHAPE = (20, 40, 40)
RADIUS = 9.0
LABEL = 5
#: ``data -> world`` voxel size, ``(z, y, x)``: z voxels twice as deep.
SCALE = (2.0, 1.0, 1.0)
#: The labels shader's ambient term (``label_volume.wgsl``).
AMBIENT = 0.3

KINDS = ["memory", "multiscale"]
MODES = ["smooth_iso", "smooth_geometry"]


def _sphere() -> np.ndarray:
    centre = (np.asarray(SHAPE, dtype=np.float64) - 1.0) / 2.0
    grid = np.stack(np.mgrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]], axis=-1)
    data = np.zeros(SHAPE, dtype=np.int32)
    data[np.linalg.norm(grid - centre, axis=-1) <= RADIUS] = LABEL
    return data


def _store(root, data: np.ndarray) -> MultiscaleZarrDataStore:
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(root / "s0")},
        "metadata": {
            "shape": list(data.shape),
            "data_type": "int32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(data.shape)},
            },
        },
        "create": True,
        "delete_existing": True,
    }
    ts.open(spec).result()[...].write(np.ascontiguousarray(data)).result()
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0"],
        level_scales=[(1.0, 1.0, 1.0)],
        level_translations=[(0.0, 0.0, 0.0)],
        name="labels-lighting",
    )


async def _shot(controller, reslice, root, kind: str, render_mode: str):
    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="lighting")
    data = _sphere()
    if kind == "memory":
        store = LabelMemoryStore(data=data, name="labels")
        transform = scale_translation_transform(
            controller, scene.id, store, SCALE, (0.0, 0.0, 0.0)
        )
        controller.add_labels(
            data=store,
            scene_id=scene.id,
            appearance=InMemoryLabelsAppearance(render_mode=render_mode),
            transform=transform,
        )
    else:
        root.mkdir(parents=True, exist_ok=True)
        store = _store(root, data)
        transform = scale_translation_transform(
            controller, scene.id, store, SCALE, (0.0, 0.0, 0.0)
        )
        controller.add_labels_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleLabelsAppearance(
                force_level=0, render_mode=render_mode
            ),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
            transform=transform,
        )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    return shoot(controller, scene, size=(320, 320), view_dir=(-0.6, -0.5, -1.0))


def _analytic(shot):
    """Per pixel: hit, ``|n . v|`` with the world normal, and in voxel space.

    pygfx world is ``(x, y, z)``: cellier's ``(z, y, x)`` reversed.  The
    surface is ``|A (w - c)| = r`` with ``A = diag(1 / scale)``.
    """
    scale = np.asarray(SCALE[::-1], dtype=np.float64)
    a = 1.0 / scale
    centre = (np.asarray(SHAPE[::-1], dtype=np.float64) - 1.0) / 2.0 * scale
    origin = shot.ray_origin.reshape(-1, 3)
    direction = shot.ray_vector.reshape(-1, 3)
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    p = (origin - centre) * a
    q = direction * a
    qa = (q * q).sum(1)
    qb = 2.0 * (p * q).sum(1)
    qc = (p * p).sum(1) - RADIUS * RADIUS
    disc = qb * qb - 4.0 * qa * qc
    hit = disc > 0
    t = (-qb - np.sqrt(np.where(hit, disc, 0.0))) / (2.0 * qa)
    on_sphere = p + t[:, None] * q  # A (w - c), a point on the voxel sphere

    def unit(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    world_normal = unit(on_sphere * a)  # A^T A (w - c)
    world_term = np.abs((world_normal * direction).sum(1))
    # What a shader lighting in voxel space computes: the voxel-space normal
    # against the view direction carried into voxel space.
    voxel_term = np.abs((unit(on_sphere) * unit(q)).sum(1))
    height, width = shot.frame.shape[:2]
    return (
        hit.reshape(height, width),
        world_term.reshape(height, width),
        voxel_term.reshape(height, width),
    )


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    c = value / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    c = np.clip(value, 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _prediction_error(observed: np.ndarray, term: np.ndarray) -> np.ndarray:
    """|frame - predicted| in 8-bit levels, for one channel.

    The shader writes ``color * (ambient + term)`` in linear light and the
    frame stores it sRGB-encoded.  The label colour is the one free
    parameter, fitted as the median of the per-pixel estimates.
    """
    color = np.median(_srgb_to_linear(observed) / (AMBIENT + term))
    return np.abs(_linear_to_srgb(color * (AMBIENT + term)) - observed)


def _interior(mask: np.ndarray, erosion: int = 4) -> np.ndarray:
    for _ in range(erosion):
        mask = (
            mask
            & np.roll(mask, 1, 0)
            & np.roll(mask, -1, 0)
            & np.roll(mask, 1, 1)
            & np.roll(mask, -1, 1)
        )
    return mask


@pytest.mark.parametrize("render_mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
async def test_anisotropic_labels_sphere_is_lit_by_its_world_normal(
    controller, reslice, tmp_path, kind, render_mode
):
    """Brightness follows the world ellipsoid's normal, not the voxel sphere's."""
    shot = await _shot(controller, reslice, tmp_path / "store", kind, render_mode)
    hit, world_term, voxel_term = _analytic(shot)
    use = _interior(hit & shot.drawn)
    assert use.sum() > 3000, f"the ellipsoid drew only {use.sum()} usable pixels"

    rgb = shot.frame[..., :3].astype(np.float64)[use]
    # The brightest channel that never clips (the label colour's red does).
    unclipped = [c for c in range(3) if np.percentile(rgb[:, c], 99) < 250]
    channel = max(unclipped, key=lambda c: rgb[:, c].mean())
    observed = rgb[:, channel]
    world_error = _prediction_error(observed, world_term[use])
    voxel_error = _prediction_error(observed, voxel_term[use])

    # Expected values, measured on the fixed shaders: the world normal
    # predicts the frame to a median of 2 levels (the smooth normals are
    # not quite the analytic one); the voxel-space normal, what the shaders
    # lit with before, misses by a median of about 13.
    assert np.median(world_error) < 4.0, (
        f"brightness strays a median {np.median(world_error):.1f} levels "
        "from the world-normal shading"
    )
    assert np.percentile(world_error, 90) < 11.0, (
        f"brightness strays {np.percentile(world_error, 90):.1f} levels "
        "from the world-normal shading at the 90th percentile"
    )
    assert np.median(voxel_error) > 8.0, (
        f"voxel-space shading fits too ({np.median(voxel_error):.1f} levels): "
        "the test no longer tells the two apart"
    )
