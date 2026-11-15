"""Tests for the shared pick → data-coordinate helpers.

These lock in the invariant the whole picking pipeline relies on: the decoded
coordinate is in level-0 data space and ``floor`` of each component recovers the
integer voxel index — uniformly for memory and multiscale, image and labels,
2-D and 3-D.
"""

from __future__ import annotations

import math

import pytest

from cellier.render.visuals._pick import (
    memory_image_data_coordinate,
    multiscale_image_data_coordinate,
    multiscale_volume_data_coordinate,
)

# ── Memory image / labels (real data texture) ──────────────────────────────


def test_memory_image_2d_center_floors_to_index() -> None:
    # pygfx reports pixel center at integer index, fractional in [-0.5, 0.5).
    pick = {"index": (3, 7), "pixel_coord": (0.0, 0.0)}
    coord = memory_image_data_coordinate(pick, ndim=2)
    assert coord == (3.5, 7.5)
    assert tuple(math.floor(c) for c in coord) == (3, 7)


def test_memory_image_2d_subpixel_stays_in_voxel() -> None:
    # Anywhere within the pixel (frac in [-0.5, 0.5)) must floor to the index.
    for fx in (-0.5, -0.1, 0.0, 0.49):
        pick = {"index": (3, 7), "pixel_coord": (fx, fx)}
        coord = memory_image_data_coordinate(pick, ndim=2)
        assert tuple(math.floor(c) for c in coord) == (3, 7)


def test_memory_volume_3d_center_floors_to_index() -> None:
    pick = {"index": (2, 5, 9), "voxel_coord": (0.0, 0.0, 0.0)}
    coord = memory_image_data_coordinate(pick, ndim=3)
    assert coord == (2.5, 5.5, 9.5)
    assert tuple(math.floor(c) for c in coord) == (2, 5, 9)


def test_memory_image_missing_index_returns_none() -> None:
    assert memory_image_data_coordinate({}, ndim=2) is None


def test_memory_image_missing_frac_defaults_to_center() -> None:
    coord = memory_image_data_coordinate({"index": (1, 1)}, ndim=2)
    assert coord == (1.5, 1.5)


# ── Multiscale volume (NormSizedVolume voxel_pos) ──────────────────────────


def test_multiscale_volume_reads_the_decoded_voxel_position() -> None:
    # The node decodes the hit with its own layout; the helper reads it as is,
    # in pygfx (x, y, z) order.
    coord = multiscale_volume_data_coordinate({"voxel_pos": (1.25, 3.5, 10.875)})
    assert coord == (1.25, 3.5, 10.875)
    assert tuple(math.floor(c) for c in coord) == (1, 3, 10)


def test_multiscale_volume_far_face_hit_floors_inside() -> None:
    # A hit on the far face (v == n) decodes inside the last voxel: the
    # layout's decode clamps it, so the helper needs no clamp of its own.
    from cellier.render._pick_layout import pick_bit_layout

    layout = pick_bit_layout((4, 4, 4))
    q = [int(4 * 2.0**f) for f in layout.frac_bits]
    voxel = tuple(layout.decode(q).tolist())
    coord = multiscale_volume_data_coordinate({"voxel_pos": voxel})
    assert tuple(math.floor(c) for c in coord) == (3, 3, 3)


def test_multiscale_volume_missing_voxel_pos_returns_none() -> None:
    # A node with no dataset size gives no position, nor does a stale
    # ``norm_pos``-only payload.
    assert multiscale_volume_data_coordinate({}) is None
    assert multiscale_volume_data_coordinate({"norm_pos": (0.0, 0.0, 0.0)}) is None


# ── Multiscale 2-D image / labels (tile proxy) ─────────────────────────────


def test_multiscale_image_resolves_within_tile_not_to_tile_center() -> None:
    # Level-0 shape 1000x2000 (H, W) split into 4x8 tiles of ~256 px. A pick in
    # proxy tile (gx=5, gy=2) at sub-tile offset must land on a real pixel, not
    # the tile centre — this is the bug the change fixes.
    level0 = (1000, 2000)  # (H, W)
    grid = (4, 8)  # (gh, gw)
    pick = {"index": (5, 2), "pixel_coord": (0.25, -0.25)}
    coord = multiscale_image_data_coordinate(pick, level0, grid)
    # data-x per proxy texel = W/gw = 250; data-y per texel = H/gh = 250.
    # x = (5 + 0.25 + 0.5) * 250 = 1437.5 ; y = (2 - 0.25 + 0.5) * 250 = 562.5
    assert coord == pytest.approx((1437.5, 562.5))
    assert tuple(math.floor(c) for c in coord) == (1437, 562)


def test_multiscale_image_first_tile_origin() -> None:
    level0 = (512, 512)
    grid = (2, 2)  # 256 px per texel
    pick = {"index": (0, 0), "pixel_coord": (-0.5, -0.5)}
    coord = multiscale_image_data_coordinate(pick, level0, grid)
    # (0 - 0.5 + 0.5) * 256 = 0 on both axes.
    assert coord == pytest.approx((0.0, 0.0))
    assert tuple(math.floor(c) for c in coord) == (0, 0)


def test_multiscale_image_clamps_upper_edge() -> None:
    level0 = (256, 256)
    grid = (1, 1)  # one tile, 256 px per texel
    # frac just under 0.5 -> (0 + 0.4999 + 0.5) * 256 ≈ 255.97 -> floor 255.
    pick = {"index": (0, 0), "pixel_coord": (0.4999, 0.4999)}
    coord = multiscale_image_data_coordinate(pick, level0, grid)
    assert all(0.0 <= c < 256.0 for c in coord)
    assert tuple(math.floor(c) for c in coord) == (255, 255)


def test_multiscale_image_missing_index_returns_none() -> None:
    assert multiscale_image_data_coordinate({}, (10, 10), (1, 1)) is None
