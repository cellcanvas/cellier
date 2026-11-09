"""Clipping, SSAO and outlines on the masked path.

* **The cross section**: a clipping plane keeping ``z >= CUT`` cuts the
  example's ellipsoid, and the camera looks at the cut face.  A cut face is a
  surface whose inward normal is the plane's (into the kept side), so with the
  value a ramp ``value = z``, an inward ``max`` over ``THICKNESS`` reads
  ``CUT + THICKNESS - 1/2`` (the midpoint rule) and a ``mean``
  ``CUT + THICKNESS / 2``.  The wrong sign would read ``CUT - 1/2`` and
  ``CUT - THICKNESS / 2``.  The colormap's three primaries over ``CLIM`` put
  each of the four answers in its own colour.
* **The value side's cull**: the value side plans with the mask's planes
  pulled back to its own data coordinates, so its target bricks on the
  clipped side are not fetched.
* **The targets**: the pick, depth and normal targets of a masked frame are
  byte-identical to the plain iso frame's, so SSAO (which reads depth and
  normal) and outlines (which read the pick target) treat a MaskedImage as an
  image.  The masked branch only replaces the colour.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pygfx as gfx
import pytest
from pygfx.renderers.wgpu import get_shared
from rendercanvas.offscreen import RenderCanvas

from cellier.convenience.capture import _wait_until_loaded
from cellier.render._cellier_blender import NORMAL_TARGET, ensure_extra_targets
from cellier.render.scheduling import ChunkClass
from cellier.render.visuals._masked_image import value_clipping_planes
from cellier.visuals import ClippingPlane, MaskedImageSingleAppearance
from tests.render._affine import canvas_view
from tests.render._masked import (
    band_colormap,
    band_counts,
    band_pair,
    gfx_visual,
    load,
    masked_config,
    new_viewer,
    store,
    value_to_mask_matrix,
)
from tests.render._pick import PickFrame

SIDE = 64
SIZE = 256
CUT = 32.0
THICKNESS = 6.0
#: Red below 33, green to 36, blue above: max 37.5 is blue and mean 35
#: green; the wrong sign's 31.5 and 29 are both red.
CLIM = (30.0, 39.0)


def _ramp(side: int, value_scale: int = 1) -> np.ndarray:
    """``value = z`` in mask voxels, on a grid of ``value_scale`` a mask voxel."""
    n = side * value_scale
    z = (np.arange(n, dtype=np.float32) + 0.5) / value_scale - 0.5
    return np.broadcast_to(z[:, None, None], (n, n, n)).copy()


def _plane(data_store, cut: float = CUT, keep: float = 1.0) -> ClippingPlane:
    """Keeps ``z >= cut`` (``keep = 1``) or ``z <= cut`` (``keep = -1``).

    In the mask's data coordinates.
    """
    centre = (SIDE - 1) / 2.0
    return ClippingPlane.from_point_normal(
        data_store.data_coordinate_systems[0], (cut, centre, centre), (keep, 0.0, 0.0)
    )


def _render(viewer, targets=()):
    """Draw the scene with its own camera; ``(frame, renderer)``."""
    viewer.controller.update_background_field(viewer.scene.id, "visible", False)
    view = canvas_view(viewer)
    scene = view._get_scene_fn(viewer.scene.id)
    canvas = RenderCanvas(size=(SIZE, SIZE), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    if targets:
        assert ensure_extra_targets(renderer, list(targets)) is True
    canvas.request_draw(lambda: renderer.render(scene, view.camera))
    return np.asarray(canvas.draw()), renderer


def _read_target(renderer, name: str) -> np.ndarray:
    """The raw bytes of a blender target, one row per pixel row."""
    texture = renderer._blender.get_texture(name)
    width, height, _ = texture.size
    fmt = str(texture.format).split(".")[-1]
    size = {"depth32float": 4, "rgba8unorm": 4}.get(fmt, 8)
    stride = (size * width + 255) // 256 * 256
    raw = get_shared().device.queue.read_texture(
        {
            "texture": texture,
            "mip_level": 0,
            "origin": (0, 0, 0),
            "aspect": "depth-only" if "depth" in fmt else "all",
        },
        {"offset": 0, "bytes_per_row": stride, "rows_per_image": height},
        (width, height, 1),
    )
    rows = np.frombuffer(raw, np.uint8).reshape(height, stride)
    return rows[:, : size * width].copy()


# -- the cross section ------------------------------------------------------------


def _cut_viewer(tmp_path, kind, op, value_scale=1, keep=1.0):
    """The ellipsoid cut at ``z = CUT``, seen from the clipped side, loaded."""
    mask_data, _ = band_pair(SIDE)
    viewer = new_viewer("3d")
    mask = store(kind, tmp_path, "mask", mask_data)
    value = store(kind, tmp_path, "value", _ramp(SIDE, value_scale))
    value_budget = 48 if kind == "memory" and value_scale == 2 else 16
    model = viewer.controller.add_masked_image(
        mask,
        value,
        viewer.scene.id,
        value_to_mask=value_to_mask_matrix(value_scale),
        single=MaskedImageSingleAppearance(
            color_map=band_colormap(),
            clim=CLIM,
            iso_threshold=4.0,
            surface_thickness=THICKNESS,
            surface_projection=op,
            surface_placement="inward",
        ),
        render_config=masked_config(value_gpu_budget_bytes=value_budget * 1024**2),
    )
    # A store's data system exists once it is added.
    model.clipping_planes = [_plane(mask, keep=keep)]
    load(viewer, size=(SIZE, SIZE))
    view = canvas_view(viewer)
    # pygfx (x, y, z) is data (z, y, x) reversed: look into the kept side.
    view.camera.show_object(
        view._get_scene_fn(viewer.scene.id), view_dir=(0.0, 0.0, keep)
    )
    return viewer, model


def _face_counts(viewer, model) -> np.ndarray:
    """Band counts of the pixels whose pick lies on the cut face."""
    frame, renderer = _render(viewer)
    pick = PickFrame(renderer)
    visual = gfx_visual(viewer, model.id)
    rgb = frame[..., :3].astype(np.int64)
    counts = np.zeros(3, np.int64)
    for y in range(1, SIZE, 2):
        for x in range(1, SIZE, 2):
            if rgb[y, x].max() <= 8:
                continue
            info = pick.info((x + 0.5, y + 0.5))
            if info.get("world_object") is None:
                continue
            coordinate = visual.pick_data_coordinate(info["world_object"], info)
            # pygfx (x, y, z), voxel centres at i + 0.5: data z is the last.
            if coordinate is None or abs(coordinate[2] - 0.5 - CUT) > 0.6:
                continue
            counts[int(np.argmax(rgb[y, x]))] += 1
    return counts


#: The projections each cut-face scene is drawn with.
CUT_OPS = ("max", "mean")


@pytest.fixture(scope="module")
def face_counts(tmp_path_factory):
    """``face_counts(kind, keep)``: ``op -> band counts`` on the cut face.

    The ops differ only in the single appearance, so each store kind and
    kept side is built once and drawn with each projection.
    """
    cache: dict[tuple[str, float], dict[str, np.ndarray]] = {}

    def get(kind: str, keep: float = 1.0) -> dict[str, np.ndarray]:
        if (kind, keep) not in cache:
            root = tmp_path_factory.mktemp(f"cut_{kind}_{keep}")
            viewer, model = _cut_viewer(root, kind, CUT_OPS[0], keep=keep)
            out = {}
            try:
                for op in CUT_OPS:
                    model.single.surface_projection = op
                    asyncio.run(_wait_until_loaded(viewer.controller))
                    out[op] = _face_counts(viewer, model)
            finally:
                viewer.controller.close()
            cache[kind, keep] = out
        return cache[kind, keep]

    return get


@pytest.mark.parametrize("kind", ["memory", "lazy"])
@pytest.mark.parametrize(("op", "band"), [("max", 2), ("mean", 1)])
def test_a_cut_face_projects_into_the_kept_side(face_counts, kind, op, band):
    """Max reads 37.5 (blue) and mean 35 (green); the wrong sign would be red."""
    counts = face_counts(kind)[op]
    assert counts.sum() > 300, counts
    assert counts[band] == counts.sum(), counts


@pytest.mark.parametrize("op", CUT_OPS)
def test_a_cut_face_keeping_the_other_side_projects_into_it(face_counts, op):
    """Keep ``z <= CUT``, seen from above: max reads 31.5 and mean 29, both red.

    The wrong sign would read 37.5 (blue) and 35 (green): the projection
    follows the plane's kept side, not a fixed direction.
    """
    counts = face_counts("memory", keep=-1.0)[op]
    assert counts.sum() > 300, counts
    assert counts[0] == counts.sum(), counts


def test_a_cut_face_projects_on_the_value_grid(tmp_path):
    """The same on a value at twice the mask's resolution."""
    viewer, model = _cut_viewer(tmp_path, "memory", "max", value_scale=2)
    try:
        counts = _face_counts(viewer, model)
        assert counts.sum() > 300, counts
        assert counts[2] == counts.sum(), counts
    finally:
        viewer.controller.close()


def test_the_cut_face_is_a_section_of_the_bands(tmp_path):
    """On the example's own value, a cut through the centre shows every band.

    Thickness 0, threshold 1: the face's value is the band at each point,
    so the section is concentric rings.  The cut's semi-axes are 22.5 and
    17.5 voxels: band 1 is the outer ring 8 wide, band 2 the next; band 3
    (16 voxels deep) is a sliver at the centre, too thin to count on.
    """
    mask_data, value_data = band_pair(SIDE)
    viewer = new_viewer("3d")
    mask = store("memory", tmp_path, "mask", mask_data)
    model = viewer.controller.add_masked_image(
        mask,
        store("memory", tmp_path, "value", value_data),
        viewer.scene.id,
        single=MaskedImageSingleAppearance(
            color_map=band_colormap(), clim=(0.5, 3.5), iso_threshold=1.0
        ),
        render_config=masked_config(),
    )
    model.clipping_planes = [_plane(mask, cut=(SIDE - 1) / 2.0)]
    try:
        load(viewer, size=(SIZE, SIZE))
        view = canvas_view(viewer)
        view.camera.show_object(
            view._get_scene_fn(viewer.scene.id), view_dir=(0.0, 0.0, 1.0)
        )
        counts = band_counts(_render(viewer)[0])
        assert counts[0] > counts[1] > 100, counts
    finally:
        viewer.controller.close()


# -- the value side's cull --------------------------------------------------------


def test_the_value_side_gets_the_planes_in_its_own_coordinates(tmp_path):
    """Pulled back through ``value_to_mask``: the same half space, ids kept."""
    mask_data, _ = band_pair(SIDE)
    viewer = new_viewer("3d")
    mask = store("memory", tmp_path, "mask", mask_data)
    value = store("memory", tmp_path, "value", _ramp(SIDE, 2))
    model = viewer.controller.add_masked_image(
        mask,
        value,
        viewer.scene.id,
        value_to_mask=value_to_mask_matrix(2),
        render_config=masked_config(value_gpu_budget_bytes=48 * 1024**2),
    )
    plane = _plane(mask)
    model.clipping_planes = [plane]
    try:
        visual = gfx_visual(viewer, model.id)
        (pulled,) = visual.value_side._clipping_planes
        assert pulled.id == plane.id
        assert pulled == value_clipping_planes([plane], model)[0]
        # Value voxel z maps to mask z = (z + 0.5) / 2 - 0.5: the cut at mask
        # 32 is value 64.5, and the kept side is still +z.
        rng = np.random.default_rng(0)
        points = rng.uniform(0.0, 2 * SIDE, (200, 3))
        kept = pulled.plane.signed_distance(points) >= 0
        assert np.array_equal(kept, points[:, 0] >= 2 * CUT + 0.5)
    finally:
        viewer.controller.close()


def test_the_value_side_does_not_fetch_clipped_targets(tmp_path):
    """A lazy value plans fewer target bricks with the plane than without.

    The backstop is never culled; the target bricks wholly on the clipped
    side are.
    """
    mask_data, _ = band_pair(SIDE)

    def value_targets(planes) -> int:
        viewer = new_viewer("3d")
        mask = store("lazy", tmp_path / str(planes), "mask", mask_data)
        value = store("lazy", tmp_path / str(planes), "value", _ramp(SIDE))
        model = viewer.controller.add_masked_image(
            mask,
            value,
            viewer.scene.id,
            single=MaskedImageSingleAppearance(iso_threshold=4.0),
            render_config=masked_config(),
        )
        if planes:
            model.clipping_planes = [_plane(mask)]
        visual = gfx_visual(viewer, model.id)
        counted: list[int] = []
        plan = visual.value_side.plan

        def counting(*args, **kwargs):
            sets = plan(*args, **kwargs)
            counted.append(sum(int((s.cls == ChunkClass.TARGET).sum()) for s in sets))
            return sets

        visual.value_side.plan = counting
        try:
            load(viewer, size=(SIZE, SIZE))
            assert counted
            return counted[-1]
        finally:
            viewer.controller.close()

    whole, clipped = value_targets(False), value_targets(True)
    assert whole > 0
    # The kept half is z >= 32 of 64: 16-voxel bricks, 2 of 4 layers kept.
    assert clipped <= 0.6 * whole, (clipped, whole)


# -- the targets ------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["memory", "lazy"])
def test_pick_depth_and_normal_targets_equal_plain_iso(tmp_path, kind):
    """The masked frame's targets, byte for byte, are plain iso's."""
    mask_data, value_data = band_pair(SIDE)
    viewer = new_viewer("3d")
    model = viewer.controller.add_masked_image(
        store(kind, tmp_path, "mask", mask_data),
        store(kind, tmp_path, "value", value_data),
        viewer.scene.id,
        single=MaskedImageSingleAppearance(
            color_map=band_colormap(),
            clim=(0.5, 3.5),
            iso_threshold=4.0,
            surface_thickness=6.0,
        ),
        render_config=masked_config(),
    )
    names = ("pick", "depth", NORMAL_TARGET)
    try:
        load(viewer, size=(SIZE, SIZE))
        material = gfx_visual(viewer, model.id)._slots[0].material_3d
        assert material.masked
        masked_frame, renderer = _render(viewer, [NORMAL_TARGET])
        masked = {name: _read_target(renderer, name) for name in names}
        # The plain iso shader of the same material and bricks.  The extent
        # clamp is under ``masked`` too; it changes nothing on this pyramid
        # (level 1 covers level 0's extent: 64 halves evenly).
        material._store.masked = False
        plain_frame, renderer = _render(viewer, [NORMAL_TARGET])
        plain = {name: _read_target(renderer, name) for name in names}
        assert (masked_frame[..., :3].max(axis=-1) > 8).sum() > 1_000
        assert not np.array_equal(masked_frame, plain_frame)
        for name in names:
            different = int((masked[name] != plain[name]).sum())
            assert different == 0, (name, different)
        assert np.any(masked["pick"]) and np.any(masked[NORMAL_TARGET])
    finally:
        viewer.controller.close()
