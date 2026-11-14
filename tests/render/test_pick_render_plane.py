"""A render-plane pick reads the voxel the plane drew there.

In ``"plane"`` render mode the four volume shaders draw one sample where the
ray meets the plane, and report that point as the pick.  They encoded it
truncated to 14 bits per axis, so a point within ``dim / 16383`` voxels above a
voxel face came back below it and the controller read the voxel next door:
0.1-0.8% of picks on a 40-voxel volume, 7-9% at 8192 voxels and about 20% at
16000.  The shaders now clamp the point one quantum inside the drawn-level
voxel the plane sampled (at most to its centre) and round the encode, as the
surface picks do (``test_pick_large_volume``).

A 3-voxel block is drawn on a black background and cut by a plane through its
centre, axis-aligned or oblique, then picked from twelve directions.  The
truth is the screen: a bright pixel shows the block and must read it, and a
dark one shows the background and must read that.  The value is read the way
the controller reads it: the drawn level the render layer reports at the hit
and that level's voxel.  Half the picks are near the block's outline, where a
voxel off would show.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import _level_voxel_index, _voxel_index
from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.visuals import (
    InMemoryImageAppearance,
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    RenderPlane,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    P0,
    SHAPE,
    bind,
    canvas_view,
    load,
    multiscale_store,
    render,
)
from tests.render._pick import PickFrame
from tests.render.test_labels_pick_face import DIRECTIONS, DISTANCE, _viewer
from tests.render.test_pick_large_volume import _budgets

DISPLAYED = ("z", "y", "x")
PICKS_PER_VIEW = 60
#: Brightness (0-255) above which a pixel shows the block, and below which it
#: shows the background; pixels in between are on the outline and skipped.
BRIGHT, DARK = 128, 20
NORMALS = {"z": (1.0, 0.0, 0.0), "oblique": (1.0, 0.6, 0.35)}


def _block(shape, centre, dtype) -> np.ndarray:
    data = np.zeros(shape, dtype=dtype)
    z, y, x = (int(v) for v in centre)
    data[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 1
    return data


def _add(viewer, family, data, tmp_path, level, block_size, budgets):
    """Add *data* as *family* in plane mode; ``(visual, store, level arrays)``."""
    identity = np.eye(4)
    if family == "image":
        store = ImageMemoryStore(data=data, name="b")
        visual = viewer.add_image(
            store,
            appearance=InMemoryImageAppearance(interpolation="nearest"),
            single=InMemoryImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="plane"
            ),
            transform=bind(viewer, store, identity),
        )
        return visual, store, [data]
    if family == "labels":
        store = LabelMemoryStore(data=data, name="b")
        visual = viewer.add_labels(
            store,
            appearance=InMemoryLabelsAppearance(render_mode="plane"),
            transform=bind(viewer, store, identity),
        )
        return visual, store, [data]
    store = multiscale_store(tmp_path, "b", data)
    arrays = [data, data[::2, ::2, ::2]]
    if family == "labels_multiscale":
        visual = viewer.add_labels_multiscale(
            store,
            appearance=MultiscaleLabelsAppearance(
                render_mode="plane", force_level=level
            ),
            render_config=MultiscaleLabelRenderConfig(**budgets, block_size=block_size),
            transform=bind(viewer, store, identity),
        )
    else:
        visual = viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(force_level=level),
            render_config=MultiscaleImageRenderConfig(**budgets, block_size=block_size),
            single=MultiscaleImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="plane"
            ),
            transform=bind(viewer, store, identity),
        )
    return visual, store, arrays


def _read(full, level, store, arrays) -> float:
    """What the controller reads at the level-0 coordinate *full*."""
    index = _voxel_index(tuple(full), arrays[0].shape)
    if index is None:
        return 0.0
    if (level or 1) > 1:
        coarse = _level_voxel_index(tuple(full), store, level)
        if coarse is not None:
            return float(arrays[level - 1][coarse])
    return float(arrays[0][index])


def _drawn_level(gfx_visual, hit, coord) -> int | None:
    single = getattr(gfx_visual, "pick_drawn_level", None)
    if single is not None:
        return single(hit, coord)
    many = getattr(gfx_visual, "pick_drawn_levels", None)
    if many is None:
        return None
    return next(iter(dict(many(hit, coord)).values()), None)


def _picks(viewer, visual, store, arrays, centre, level):
    """Pick every view; ``(number of picks, wrong picks)``."""
    controller = viewer.controller
    gfx_visual = controller._render_manager._scenes[viewer.scene.id].get_visual(
        visual.id
    )
    target = np.asarray(centre, float)[::-1]
    rng = np.random.default_rng(0)
    n_picks = 0
    wrong = []
    for direction in DIRECTIONS:
        camera = canvas_view(viewer).camera
        camera.world.position = tuple(target + DISTANCE * direction)
        camera.look_at(tuple(target))
        frame, renderer, _ = render(viewer)
        pick_frame = PickFrame(renderer)
        brightness = frame[..., :3].astype(float).max(axis=-1)
        hit_mask = (pick_frame.values & ((1 << 20) - 1)) != 0
        bright = hit_mask & (brightness > BRIGHT)
        rows, cols = np.nonzero(bright | (hit_mask & (brightness < DARK)))
        if rows.size == 0:
            continue
        chosen = rng.choice(
            rows.size, size=min(PICKS_PER_VIEW // 2, rows.size), replace=False
        )
        bright_rows, bright_cols = np.nonzero(bright)
        if bright_rows.size:
            near = np.nonzero(
                (rows >= bright_rows.min() - 6)
                & (rows <= bright_rows.max() + 6)
                & (cols >= bright_cols.min() - 6)
                & (cols <= bright_cols.max() + 6)
            )[0]
            chosen = np.concatenate(
                [
                    chosen,
                    rng.choice(
                        near, size=min(PICKS_PER_VIEW // 2, near.size), replace=False
                    ),
                ]
            )
        for index in chosen:
            row, col = rows[index], cols[index]
            info = pick_frame.info((col + 0.5, row + 0.5))
            hit = info.get("world_object")
            assert hit is not None
            coord = gfx_visual.pick_data_coordinate(hit, info)
            assert coord is not None
            full = controller._promote_pick_coordinate(
                coord,
                displayed_axes=(0, 1, 2),
                slice_indices={},
                world_ndim=3,
                hit_visual_id=visual.id,
            )
            drawn = _drawn_level(gfx_visual, hit, coord)
            if level is not None:
                assert drawn == level
            expected = 1.0 if bright[row, col] else 0.0
            n_picks += 1
            if _read(full, drawn, store, arrays) != expected:
                wrong.append((direction.round(2).tolist(), full, expected))
    return n_picks, wrong


def _run(tmp_path, family, normal, *, shape, centre, level, block_size, budgets):
    viewer = _viewer(DISPLAYED)
    try:
        dtype = np.int32 if family.startswith("labels") else np.float32
        data = _block(shape, centre, dtype)
        visual, store, arrays = _add(
            viewer, family, data, tmp_path, level, block_size, budgets
        )
        world = viewer.scene.dims.world_coordinate_system
        plane = RenderPlane.from_point_normal(
            world, np.asarray(centre, float), NORMALS[normal]
        )
        viewer.controller.set_render_planes(visual.id, [plane])
        load(viewer)
        n_picks, wrong = _picks(viewer, visual, store, arrays, centre, level)
        assert n_picks > 300
        assert not wrong, f"{len(wrong)} of {n_picks} picks misread: {wrong[:3]}"
    finally:
        viewer.controller.close()


_SMALL = [
    ("labels", None),
    ("image", None),
    ("labels_multiscale", 1),
    ("labels_multiscale", 2),
    ("image_multiscale", 1),
    ("image_multiscale", 2),
]


@pytest.mark.parametrize("normal", sorted(NORMALS))
@pytest.mark.parametrize(("family", "level"), _SMALL)
def test_plane_picks_read_the_drawn_voxel(tmp_path, family, level, normal):
    _run(
        tmp_path,
        family,
        normal,
        shape=SHAPE,
        centre=P0,
        level=level,
        block_size=16,
        budgets=SMALL_BUDGETS,
    )


#: The in-memory families are bounded by the 2048-voxel 3D texture.
_LONG = [
    ("labels", 2048),
    ("image", 2048),
    *[
        (family, n)
        for family in ("labels_multiscale", "image_multiscale")
        for n in (2048, 8192, 16000)
    ],
]


@pytest.mark.parametrize("normal", sorted(NORMALS))
@pytest.mark.parametrize(("family", "length"), _LONG)
def test_plane_picks_on_a_long_axis(tmp_path, family, length, normal):
    # Off the middle, where the faces would sit on the quantisation grid.
    centre = (7, 20, int(length * 0.4137))
    _run(
        tmp_path,
        family,
        normal,
        shape=(24, 32, length),
        centre=centre,
        level=None if family in ("labels", "image") else 1,
        block_size=64,
        budgets=_budgets(length),
    )
