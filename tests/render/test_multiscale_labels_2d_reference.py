"""2D multiscale labels, pixel by pixel, against the reference sampler.

``plans/multiscale_level_transform_v2.md``, R2 and R3.  The centroid harness
(``test_multiscale_level_alignment.py``) needs integer level ratios, so it
cannot see tile seams on a non-integer pyramid, which is where the ghost
border and the cell -> tile rule matter.  2D labels are nearest-sampled, so
the reference sampler (``_level_reference.sample_nearest``) predicts every
pixel: map each pixel centre to data coordinates through the camera, take
the level voxel ``floor(u + 0.5)``, and compare its label with the colour
drawn there.  Only pixels on a label boundary are ignored.

The pyramid has non-integer ratios in y (98 -> 49 -> 24: 2 and 4.083), a
translation inside the contract on every coarse level, an 8-voxel block so
there are several seams, and labels (7 x 5 voxel blocks) that cross them.
Each level's labels are the level-0 pattern evaluated at that level's voxel
centres, so a correctly placed level agrees with the pattern everywhere
except where a coarse voxel straddles a pattern boundary -- and the oracle
reads the level's own array, so even those pixels must match.

R3: at level 0, a pick one pixel inside a label edge returns the label drawn
under that pixel.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.visuals import ProgressiveLoadingConfig
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests.render._level_reference import sample_nearest

#: (z, y, x) per level; z is never downsampled.
SHAPES = [(4, 98, 60), (4, 49, 30), (4, 24, 15)]
SCALES = [tuple(s0 / sk for s0, sk in zip(SHAPES[0], shape)) for shape in SHAPES]
#: Level translations inside C3 (``-0.5 <= t <= s - 0.5``), level-0 voxels.
TRANSLATIONS = [
    (0.0, 0.4 * s[1] if s[1] > 1 else 0.0, 0.2 * s[2] if s[2] > 1 else 0.0)
    for s in SCALES
]
BLOCK_SIZE = 8
SIZE = (512, 512)

#: Six labels with well separated colours, so a drawn colour decodes to one.
COLORS = {
    1: (1.0, 0.0, 0.0, 1.0),
    2: (0.0, 1.0, 0.0, 1.0),
    3: (0.0, 0.0, 1.0, 1.0),
    4: (1.0, 1.0, 0.0, 1.0),
    5: (1.0, 0.0, 1.0, 1.0),
    6: (0.0, 1.0, 1.0, 1.0),
}


def _pattern(py: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Label at data position ``(py, px)``: 7 x 5 voxel blocks, 6 labels."""
    return (1 + (np.floor(py / 7) + 2 * np.floor(px / 5)) % 6).astype(np.int32)


def _level_array(level: int) -> np.ndarray:
    """Level ``level``'s labels: the pattern at its voxel centres, ``s i + t``."""
    _, ny, nx = SHAPES[level]
    _, sy, sx = SCALES[level]
    _, ty, tx = TRANSLATIONS[level]
    py = sy * np.arange(ny) + ty
    px = sx * np.arange(nx) + tx
    plane = _pattern(py[:, None], px[None, :])
    return np.broadcast_to(plane, SHAPES[level]).copy()


def _store(root) -> MultiscaleZarrDataStore:
    for index, shape in enumerate(SHAPES):
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / f"s{index}")},
            "metadata": {
                "shape": list(shape),
                "data_type": "int32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(shape)},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        ts.open(spec).result()[...].write(_level_array(index)).result()
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=[f"s{index}" for index in range(len(SHAPES))],
        level_scales=SCALES,
        level_translations=TRANSLATIONS,
        name="labels-2d-reference",
    )


async def _render(controller, render_scene, reslice, root, level):
    """Render *level*; return ``(frame, camera, gfx scene, gfx visual)``."""
    scene = controller.add_scene(dim="2d", name=f"labels-2d-{level}")
    visual = controller.add_labels_multiscale(
        data=_store(root),
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(
            force_level=level + 1, colormap_mode="direct", color_dict=COLORS
        ),
        render_config=MultiscaleLabelRenderConfig(
            block_size=BLOCK_SIZE, loading=ProgressiveLoadingConfig(backstop=False)
        ),
    )
    np.testing.assert_allclose(
        np.asarray(visual.transform.matrix), np.eye(visual.transform.matrix.shape[0])
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id, size=SIZE)
    canvas = controller._render_manager._canvases[
        controller.get_canvas_ids(scene.id)[0]
    ]
    gfx_scene = canvas._get_scene_fn(scene.id)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    return frame, canvas.camera, gfx_scene, gfx_visual


def _pixel_data_positions(camera, shape) -> np.ndarray:
    """``(H, W, 2)`` data ``(y, x)`` of every pixel centre (identity transform)."""
    h, w = shape
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    ndc = np.stack(
        [
            cols / w * 2.0 - 1.0,
            1.0 - rows / h * 2.0,
            np.zeros_like(rows),
            np.ones_like(rows),
        ],
        axis=-1,
    )
    inverse = np.linalg.inv(np.asarray(camera.camera_matrix, dtype=np.float64))
    world = ndc.reshape(-1, 4) @ inverse.T
    world = world[:, :3] / world[:, 3:4]
    # pygfx (x, y) is data (x, y): the visual's transform is the identity.
    return np.stack([world[:, 1], world[:, 0]], axis=-1).reshape(h, w, 2)


def _decode(frame: np.ndarray) -> np.ndarray:
    """Label drawn at each pixel (0 where nothing was drawn)."""
    rgb = frame[..., :3].astype(np.float64) / 255.0
    ids = np.array(list(COLORS))
    palette = np.array([COLORS[i][:3] for i in ids])
    distance = ((rgb[..., None, :] - palette) ** 2).sum(axis=-1)
    out = ids[np.argmin(distance, axis=-1)]
    out[frame[..., 3] == 0] = 0
    return out


def _boundary(labels: np.ndarray) -> np.ndarray:
    """Pixels whose 4-neighbourhood holds more than one label."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[1:] |= labels[:-1] != labels[1:]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return edge


@pytest.mark.parametrize("level", range(len(SHAPES)))
async def test_every_pixel_matches_the_reference(
    controller, render_scene, reslice, tmp_path, level
):
    frame, camera, _, _ = await _render(
        controller, render_scene, reslice, tmp_path, level
    )
    drawn = _decode(frame)
    positions = _pixel_data_positions(camera, frame.shape[:2])
    _, sy, sx = SCALES[level]
    _, ty, tx = TRANSLATIONS[level]
    plane = _level_array(level)[0]
    expected = sample_nearest(
        plane, positions.reshape(-1, 2), (sy, sx), (ty, tx), outside=0
    ).reshape(drawn.shape)
    # Inside level 0's extent only (D3: the rendering domain).
    inside = (
        (positions[..., 0] > -0.5)
        & (positions[..., 0] < SHAPES[0][1] - 0.5)
        & (positions[..., 1] > -0.5)
        & (positions[..., 1] < SHAPES[0][2] - 0.5)
    )
    check = inside & ~_boundary(expected) & (expected != 0)
    assert check.sum() > 0.5 * inside.sum()
    wrong = check & (drawn != expected)
    assert not wrong.any(), (
        f"level {level}: {int(wrong.sum())} of {int(check.sum())} pixels differ"
    )


async def test_a_pick_just_inside_a_label_edge_returns_the_drawn_label(
    controller, render_scene, reslice, tmp_path
):
    """R3: picking agrees with drawing, one pixel inside each label edge."""
    import pygfx as gfx
    from rendercanvas.offscreen import RenderCanvas

    _, camera, gfx_scene, gfx_visual = await _render(
        controller, render_scene, reslice, tmp_path, 0
    )
    level0 = _level_array(0)[0]

    # Draw and pick through one renderer without anti-aliasing, so the drawn
    # label at a pixel is exactly what the shader wrote there.
    canvas = RenderCanvas(size=SIZE, pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    canvas.request_draw(lambda: renderer.render(gfx_scene, camera))
    drawn = _decode(np.asarray(canvas.draw()))

    # One pixel inside each vertical label edge between columns c and c + 1:
    # column c - 1 on the left, c + 2 on the right.
    edge = (drawn[:, 1:-2] != drawn[:, 2:-1]) & (drawn[:, 1:-2] != 0)
    rows, cols = np.nonzero(edge)
    cols = cols + 1
    targets = [(r, c - 1) for r, c in zip(rows, cols)] + [
        (r, c + 2) for r, c in zip(rows, cols)
    ]
    rng = np.random.default_rng(0)
    picks = rng.choice(len(targets), size=min(60, len(targets)), replace=False)
    mismatches = []
    checked = 0
    for index in picks:
        r, c = targets[index]
        if drawn[r, c] == 0:
            continue
        info = renderer.get_pick_info((c + 0.5, r + 0.5))
        coord = gfx_visual.pick_data_coordinate(info.get("world_object"), info)
        assert coord is not None
        # pygfx order; the decode is in the [i, i + 1) convention, so floor
        # names the voxel under the cursor.
        x, y = coord[0], coord[1]
        picked = level0[int(np.floor(y)), int(np.floor(x))]
        checked += 1
        if picked != drawn[r, c]:
            mismatches.append((int(r), int(c), int(drawn[r, c]), int(picked)))
    assert checked > 20
    assert not mismatches, mismatches[:5]
