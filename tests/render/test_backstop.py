"""The backstop helpers and the planner tail that packs it (design v3, 5.3/5.9)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.render._backstop import (
    backstop_bricks_3d,
    backstop_cap,
    backstop_level,
    backstop_tiles_2d,
)
from cellier.render._frustum import frustum_planes_from_corners
from cellier.render._level_of_detail_2d import build_tile_grids_2d
from cellier.render.block_cache import BlockCache3D, compute_block_cache_parameters_3d
from cellier.render.block_cache._image_residency import ImageResidency3D, unpack_keys
from cellier.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from cellier.render.lut_indirection._layout_2d import BlockLayout2D
from cellier.render.scheduling import ChunkClass
from cellier.render.visuals._chunked import (
    backstop_cap_for,
    desired_bricks,
    log_backstop_cap_once,
)
from cellier.visuals import ProgressiveLoadingConfig

BLOCK = 4


# -- the config -------------------------------------------------------------------


def test_defaults_are_the_designed_ones() -> None:
    cfg = ProgressiveLoadingConfig()
    assert cfg.backstop is True
    assert cfg.backstop_level is None
    assert cfg.backstop_extent == "full"
    assert cfg.backstop_max_slot_fraction == 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backstop_level": 0},
        {"backstop_extent": "all"},
        {"backstop_max_slot_fraction": 0.0},
        {"backstop_max_slot_fraction": 0.6},
    ],
)
def test_bad_settings_are_refused(kwargs) -> None:
    with pytest.raises(ValidationError):
        ProgressiveLoadingConfig(**kwargs)


def test_the_config_is_frozen() -> None:
    with pytest.raises(ValidationError):
        ProgressiveLoadingConfig().backstop = False


# -- level and cap --------------------------------------------------------------------


@pytest.mark.parametrize("level, expected", [(None, 4), (1, 1), (3, 3), (9, 4)])
def test_backstop_level_defaults_to_the_coarsest_and_clamps(level, expected) -> None:
    assert backstop_level(ProgressiveLoadingConfig(backstop_level=level), 4) == expected


def test_the_cap_is_a_share_of_the_slots() -> None:
    cfg = ProgressiveLoadingConfig(backstop_max_slot_fraction=0.25)
    assert backstop_cap(cfg, 100) == 25
    assert backstop_cap(cfg, 3) == 0
    residency = SimpleNamespace(n_slots=40)
    assert backstop_cap_for(cfg, residency) == 10
    assert backstop_cap_for(None, residency) == 0
    assert backstop_cap_for(ProgressiveLoadingConfig(backstop=False), residency) == 0


# -- the helpers ----------------------------------------------------------------------


def _grids_3d(level: int):
    """A 4 x 4 x 4 brick grid at *level*, identity placement."""
    g = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)
    arr = np.column_stack([np.full(len(g), level), g]).astype(np.int32)
    grids = [{"arr": arr[:0]}] * (level - 1) + [{"arr": arr}]
    scales = np.ones((level, 3))
    translations = np.zeros((level, 3))
    return grids, scales, translations


def test_3d_full_extent_is_every_brick_nearest_first() -> None:
    grids, scales, translations = _grids_3d(2)
    camera = np.array([0.0, 0.0, 0.0])
    arr = backstop_bricks_3d(grids, 2, camera, BLOCK, scales, translations)
    assert arr.shape == (64, 4)
    assert (arr[:, 0] == 2).all()
    assert arr[0, 1:].tolist() == [0, 0, 0]
    centres = (arr[:, 1:] + 0.5) * BLOCK
    distances = np.linalg.norm(centres - camera, axis=1)
    assert (np.diff(distances) >= -1e-9).all()


def test_3d_view_extent_culls_to_the_frustum() -> None:
    grids, scales, translations = _grids_3d(1)
    # A box around the first brick corner only: x, y, z in [0, 5].
    lo, hi = 0.0, 5.0
    # (near, far) x (left-bottom, right-bottom, right-top, left-top), for a
    # camera looking down -z from z = 5.
    corners = np.array(
        [[[lo, lo, z], [hi, lo, z], [hi, hi, z], [lo, hi, z]] for z in (hi, lo)]
    )
    planes = frustum_planes_from_corners(corners)
    full = backstop_bricks_3d(grids, 1, np.zeros(3), BLOCK, scales, translations)
    view = backstop_bricks_3d(
        grids, 1, np.zeros(3), BLOCK, scales, translations, frustum_planes=planes
    )
    assert 0 < len(view) < len(full)


def test_2d_view_extent_keeps_one_tile_of_margin() -> None:
    layout = BlockLayout2D.from_shape(shape=(64, 64), block_size=8)
    scales = [np.array([1.0, 1.0])]
    grids = build_tile_grids_2d(layout, 1, [(64, 64)], scales)
    shader_scale = np.ones((1, 2))
    shader_translation = np.zeros((1, 2))
    centre = np.array([4.0, 4.0, 0.0])
    full = backstop_tiles_2d(grids, 1, centre, 8, shader_scale, shader_translation)
    assert len(full) == 64
    assert full[0, 1:].tolist() == [0, 0]
    # A viewport over one tile (0..8): the tile and a ring of one tile.
    view = backstop_tiles_2d(
        grids,
        1,
        centre,
        8,
        shader_scale,
        shader_translation,
        view_min=np.array([0.5, 0.5]),
        view_max=np.array([7.5, 7.5]),
    )
    assert sorted(map(tuple, view[:, 1:].tolist())) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


# -- the planner tail -------------------------------------------------------------


SHAPES = [(16, 16, 16), (8, 8, 8), (4, 4, 4)]
SCALES = np.array([[1.0] * 3, [2.0] * 3, [4.0] * 3])


class _Planner:
    """Stands in for a visual: every level reads plane 0 of nothing collapsed."""

    def _level_slice_selection(self, level_index, fill=None):
        return None


def _residency(n_bricks: int) -> ImageResidency3D:
    params = compute_block_cache_parameters_3d(
        block_size=BLOCK,
        gpu_budget_bytes=n_bricks * (BLOCK + 2) ** 3 * 4,
        overlap=1,
    )
    lut = LutIndirectionManager3D(
        BlockLayout3D(volume_shape=SHAPES[0], block_size=BLOCK),
        n_levels=3,
        level_scale_vecs_data=list(SCALES),
        level_shapes=SHAPES,
    )
    return ImageResidency3D(BlockCache3D(params), lut, BLOCK, SCALES, np.zeros((3, 3)))


def _rows(level: int, n: int) -> np.ndarray:
    g = np.stack(np.unravel_index(np.arange(n), (4, 4, 4)), -1)
    return np.column_stack([np.full(n, level), g])


def test_the_backstop_leads_and_keeps_its_class() -> None:
    residency = _residency(64)
    target = _rows(1, 10)
    backstop = _rows(3, 1)
    ds = desired_bricks(
        _Planner(), residency, target, backstop_arr=backstop, backstop_cap=5
    )
    assert ds.cls.tolist() == [ChunkClass.BACKSTOP] + [ChunkClass.TARGET] * 10
    levels, _, _ = unpack_keys(ds.keys)
    assert levels.tolist() == [3] + [1] * 10
    assert ds.n_truncated_target == ds.n_truncated_backstop == 0


def test_a_target_key_equal_to_a_backstop_key_is_listed_once_as_backstop() -> None:
    residency = _residency(64)
    shared = _rows(3, 1)
    ds = desired_bricks(
        _Planner(),
        residency,
        np.concatenate([shared, _rows(1, 2)]),
        backstop_arr=shared,
        backstop_cap=5,
    )
    assert len(ds.keys) == 3
    assert len(np.unique(ds.keys)) == 3
    assert ds.cls.tolist() == [
        ChunkClass.BACKSTOP,
        ChunkClass.TARGET,
        ChunkClass.TARGET,
    ]


def test_the_cap_truncates_the_backstop_nearest_first() -> None:
    residency = _residency(64)
    backstop = _rows(2, 8)
    ds = desired_bricks(
        _Planner(), residency, None, backstop_arr=backstop, backstop_cap=3
    )
    assert ds.n_truncated_backstop == 5
    _, _, grids = unpack_keys(ds.keys)
    assert grids.tolist() == backstop[:3, 1:].tolist()


def test_the_target_gets_the_room_the_backstop_leaves() -> None:
    residency = _residency(10)
    budget = residency.n_slots - 1
    ds = desired_bricks(
        _Planner(),
        residency,
        _rows(1, 40),
        backstop_arr=_rows(3, 1),
        backstop_cap=5,
    )
    assert len(ds.keys) == budget
    assert int((ds.cls == ChunkClass.BACKSTOP).sum()) == 1
    assert ds.n_truncated_target == 40 - (budget - 1)


def test_no_backstop_is_the_target_alone() -> None:
    residency = _residency(64)
    ds = desired_bricks(_Planner(), residency, _rows(1, 4))
    assert (ds.cls == ChunkClass.TARGET).all()
    assert len(ds.keys) == 4


def test_the_cap_is_logged_once_per_visual(caplog) -> None:
    residency = _residency(64)
    ds = desired_bricks(
        _Planner(), residency, None, backstop_arr=_rows(2, 8), backstop_cap=3
    )
    owner = SimpleNamespace(visual_model_id="v")
    with caplog.at_level(logging.INFO, logger="cellier.render.cache"):
        log_backstop_cap_once(owner, [ds])
        log_backstop_cap_once(owner, [ds])
    lines = [r for r in caplog.records if "backstop_capped" in r.getMessage()]
    assert len(lines) == 1
    assert "backstop_extent='view'" in lines[0].getMessage()


@pytest.mark.parametrize("module", ["image", "labels"])
def test_loading_round_trips_with_the_render_config(module) -> None:
    if module == "image":
        from cellier.visuals import MultiscaleImageRenderConfig as Config
    else:
        from cellier.visuals import MultiscaleLabelRenderConfig as Config
    config = Config(
        loading=ProgressiveLoadingConfig(backstop_level=2, backstop_extent="view")
    )
    again = Config.model_validate_json(config.model_dump_json())
    assert again == config
    assert again.loading.backstop_extent == "view"
    # Older files, written before the field existed, load with the defaults.
    legacy = config.model_dump()
    del legacy["loading"]
    assert Config.model_validate(legacy).loading == ProgressiveLoadingConfig()
