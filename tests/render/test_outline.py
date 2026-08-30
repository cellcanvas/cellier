"""Tests for the screen-space outline pass.

Three groups:

* the pygfx coupling canary -- ``test_pygfx_coupling_still_valid`` is what
  fails when a pygfx bump invalidates the pick-texture workaround, so it
  should break CI rather than silently disabling outlines;
* the LUT's encoding and full-range indexing;
* the edge operator, driven over a *fabricated* pick buffer so band widths
  can be measured exactly.  Going through the renderer instead would put
  the output pass's reconstruction filter between the shader and the
  assertion.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest
import wgpu

from cellier.render._config import (
    MAX_OUTLINE_SLOT,
    OutlineConfig,
    OutlineLayerConfig,
    RenderManagerConfig,
)
from cellier.render._pick_buffer import enable_pick_texture_binding, get_pick_view
from cellier.render._visual_lut import (
    KIND_WHOLE_OBJECT,
    LUT_HEIGHT,
    LUT_WIDTH,
    PLACEMENT_INWARD,
    PLACEMENT_OUTWARD,
    VisualLut,
    decode_entry,
    encode_entry,
    lut_index,
)

PICK_ID_MAX = 2**20 - 1


# ---------------------------------------------------------------------------
# pygfx coupling canary
# ---------------------------------------------------------------------------


def test_pygfx_coupling_still_valid(offscreen_renderer):
    """The pick texture can still be granted TEXTURE_BINDING.

    This is the canary for pygfx bumps.  The outline pass reads ids out of
    ``renderer._blender``'s pick target, which pygfx allocates without
    ``TEXTURE_BINDING``; ``enable_pick_texture_binding`` raises the usage
    before the texture is created.  If this test fails after a pygfx
    upgrade, outlines are broken -- check ``cellier.render._pick_buffer``
    against the new ``Blender``, and re-run ``scripts/v2/outline_spike.py``.
    """
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(32, 32), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)

    assert enable_pick_texture_binding(renderer) is True

    # A second grant fails only once the texture exists, so draw first.
    scene = gfx.Scene()
    scene.add(gfx.Mesh(gfx.box_geometry(), gfx.MeshBasicMaterial(pick_write=True)))
    camera = gfx.OrthographicCamera()
    camera.show_object(scene)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    canvas.draw()

    assert get_pick_view(renderer) is not None
    assert enable_pick_texture_binding(renderer) is False


def test_pick_buffer_helpers_degrade_on_unexpected_objects():
    """Neither helper raises when the pygfx internals are not there."""

    class _NoBlender:
        pass

    assert enable_pick_texture_binding(_NoBlender()) is False
    assert get_pick_view(_NoBlender()) is None


# ---------------------------------------------------------------------------
# LUT encoding and indexing
# ---------------------------------------------------------------------------


def test_encode_decode_entry_round_trips():
    """Every field survives a pack/unpack cycle.

    The ambient occlusion bit is included: it shares the byte with the
    outline fields, and the two features must stay independent inside it.
    """
    for slot in range(MAX_OUTLINE_SLOT + 1):
        for kind in (0, 1, 2):
            for placement in (PLACEMENT_INWARD, PLACEMENT_OUTWARD):
                for ao_excluded in (False, True):
                    value = encode_entry(slot, kind, placement, ao_excluded=ao_excluded)
                    assert 0 <= value <= 255
                    assert decode_entry(value) == (
                        slot,
                        kind,
                        placement,
                        ao_excluded,
                    )


def test_encode_entry_rejects_out_of_range():
    """Out-of-range fields raise rather than silently truncating."""
    with pytest.raises(ValueError):
        encode_entry(MAX_OUTLINE_SLOT + 1)
    with pytest.raises(ValueError):
        encode_entry(1, kind=3)
    with pytest.raises(ValueError):
        encode_entry(1, placement=2)


def test_lut_index_is_injective_over_the_full_id_range():
    """Ids across the whole 2^20 space land on distinct texels.

    Sized for how pygfx actually allocates ids -- ``random.randint(1,
    1_048_575)``, not a counter -- so this deliberately uses full-range
    draws.  Small sequential ids would pass a broken index.
    """
    rng = np.random.default_rng(0)
    ids = rng.integers(1, PICK_ID_MAX, size=4096, dtype=np.int64)
    ids = np.unique(ids)
    texels = {lut_index(int(i)) for i in ids}
    assert len(texels) == len(ids)
    for x, y in texels:
        assert 0 <= x < LUT_WIDTH
        assert 0 <= y < LUT_HEIGHT


def test_lut_round_trips_full_range_ids(offscreen_renderer):
    """Entries written for full-range ids read back from the right texel."""
    lut = VisualLut()
    rng = np.random.default_rng(1)
    ids = np.unique(rng.integers(1, PICK_ID_MAX, size=64, dtype=np.int64))

    expected = {}
    for index, object_id in enumerate(ids):
        slot = (index % MAX_OUTLINE_SLOT) + 1
        placement = PLACEMENT_OUTWARD if index % 2 else PLACEMENT_INWARD
        value = encode_entry(slot, KIND_WHOLE_OBJECT, placement)
        expected[int(object_id)] = value
        assert lut.set_entry(int(object_id), value) is True

    for object_id, value in expected.items():
        assert lut.get_entry(object_id) == value
        assert decode_entry(lut.get_entry(object_id))[1] == KIND_WHOLE_OBJECT

    # Writing the same value again is a no-op.
    for object_id, value in expected.items():
        assert lut.set_entry(object_id, value) is False


def test_lut_entry_zero_stays_inert(offscreen_renderer):
    """Background (id 0) can never be given an entry."""
    lut = VisualLut()
    assert lut.set_entry(0, encode_entry(1)) is False
    assert lut.get_entry(0) == 0


def test_lut_apply_clears_stale_entries(offscreen_renderer):
    """``apply`` makes the table match exactly, dropping what is gone."""
    lut = VisualLut()
    value = encode_entry(1, KIND_WHOLE_OBJECT, PLACEMENT_INWARD)
    lut.apply({12345: value, 987654: value})
    assert lut.entries == {12345: value, 987654: value}

    lut.apply({987654: value})
    assert lut.entries == {987654: value}
    assert lut.get_entry(12345) == 0

    lut.clear()
    assert lut.entries == {}
    assert lut.get_entry(987654) == 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_outline_config_round_trips_through_json():
    """The config survives model_dump_json / model_validate_json."""
    config = RenderManagerConfig(
        outline=OutlineConfig(
            enabled=True,
            boundaries=OutlineLayerConfig(
                enabled=True, inward_thickness=1, color=(1.0, 1.0, 1.0, 0.4)
            ),
            selection=OutlineLayerConfig(
                enabled=True, inward_thickness=2, outward_thickness=2
            ),
            inner_thickness=2,
            inner_color=(0.0, 0.0, 0.0, 1.0),
            palette=[(1.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0)],
        )
    )
    restored = RenderManagerConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_outline_config_defaults_to_disabled():
    """Nothing changes for a scene that has not opted in."""
    assert RenderManagerConfig().outline.enabled is False


def test_outline_palette_is_capped_at_the_slot_field_width():
    """The LUT's slot field is 4 bits, so the palette cannot exceed 15."""
    with pytest.raises(ValueError, match="at most"):
        OutlineConfig(palette=[(1.0, 1.0, 1.0, 1.0)] * (MAX_OUTLINE_SLOT + 1))


# ---------------------------------------------------------------------------
# Edge operator, over a fabricated pick buffer
# ---------------------------------------------------------------------------

_SIZE = 64
_FILL = (0.2, 0.2, 0.2, 1.0)
_SELECT = (1.0, 0.0, 1.0, 1.0)


class _SyntheticOutline:
    """Run ``_OutlineQuadPass`` over a hand-built pick buffer.

    Bypasses the renderer entirely so band widths can be measured on the
    exact pixels the shader wrote: the renderer's output pass applies a
    reconstruction filter that would smear a one-pixel band across several
    intermediate values.
    """

    def __init__(self, ids: np.ndarray) -> None:
        from pygfx.renderers.wgpu.engine.shared import get_shared

        from cellier.render._outline import _OutlineQuadPass

        self._device = get_shared().device
        height, width = ids.shape
        self._size = (width, height)

        # rgba16uint, with global_id packed into bits 0..19 exactly as
        # pick_pack writes it.
        pick = np.zeros((height, width, 4), dtype=np.uint16)
        pick[..., 0] = ids & 0xFFFF
        pick[..., 1] = (ids >> 16) & 0xF
        self._pick_view = self._upload(
            pick, wgpu.TextureFormat.rgba16uint, 8 * width, height
        )

        color = np.zeros((height, width, 4), dtype=np.uint8)
        color[:] = np.round(np.array(_FILL) * 255).astype(np.uint8)
        self._color_view = self._upload(
            color, wgpu.TextureFormat.rgba8unorm, 4 * width, height
        )

        self._target = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
            dimension="2d",
        )
        self._lut = VisualLut()
        self._quad = _OutlineQuadPass()

    def _upload(self, data, fmt, bytes_per_row, height):
        texture = self._device.create_texture(
            size=(self._size[0], self._size[1], 1),
            format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d",
        )
        self._device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"offset": 0, "bytes_per_row": bytes_per_row, "rows_per_image": height},
            (self._size[0], self._size[1], 1),
        )
        return texture.create_view()

    def configure(self, config: OutlineConfig, *, has_inward, has_outward) -> None:
        """Apply an OutlineConfig and declare which placements are present."""
        quad = self._quad
        quad.set_template_vars(
            b_t_in=config.boundaries.inward_thickness,
            b_t_out=config.boundaries.outward_thickness,
            s_t_in=config.selection.inward_thickness,
            s_t_out=config.selection.outward_thickness,
            inner_t=config.inner_thickness,
            has_inward=has_inward,
            has_outward=has_outward,
        )
        quad.set_uniform("boundaries_enabled", int(config.boundaries.enabled))
        quad.set_uniform("selection_enabled", int(config.selection.enabled))
        quad.set_uniform("boundary_color", tuple(config.boundaries.color))
        quad.set_uniform("inner_color", tuple(config.inner_color))
        quad.set_palette([tuple(c) for c in config.palette])

    def set_entry(self, object_id: int, value: int) -> None:
        """Write one LUT entry."""
        self._lut.set_entry(object_id, value)

    def render(self) -> np.ndarray:
        """Draw and read the target back as an ``(h, w, 4)`` uint8 array."""
        width, height = self._size
        encoder = self._device.create_command_encoder()
        self._quad.render(
            encoder,
            self._color_view,
            self._pick_view,
            self._lut.view,
            self._target.create_view(),
        )
        self._device.queue.submit([encoder.finish()])
        raw = self._device.queue.read_texture(
            {"texture": self._target, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": 4 * width, "rows_per_image": height},
            (width, height, 1),
        )
        return np.frombuffer(raw, np.uint8).reshape(height, width, 4)


def _selection_config(*, t_in: int, t_out: int, inner_t: int = 0) -> OutlineConfig:
    """Selection layer only, so the assertions see one band."""
    return OutlineConfig(
        enabled=True,
        boundaries=OutlineLayerConfig(enabled=False),
        selection=OutlineLayerConfig(
            enabled=True, inward_thickness=t_in, outward_thickness=t_out
        ),
        inner_thickness=inner_t,
        palette=[_SELECT],
    )


def _count_exact(frame: np.ndarray, rgba) -> int:
    """Count pixels whose RGB is exactly *rgba*'s, 8-bit."""
    expected = np.round(np.array(rgba[:3]) * 255).astype(np.uint8)
    return int(np.count_nonzero(np.all(frame[..., :3] == expected, axis=-1)))


def _painted_columns(frame: np.ndarray, row: int) -> list[int]:
    """Columns in *row* holding the selection colour rather than the fill."""
    expected = np.round(np.array(_SELECT) * 255).astype(np.uint8)
    match = np.all(frame[row] == expected, axis=-1)
    return np.flatnonzero(match).tolist()


@pytest.mark.parametrize("thickness", [1, 2, 4])
def test_straight_edge_produces_a_band_of_exactly_t_in_pixels(
    offscreen_renderer, thickness
):
    """A straight edge under inward placement yields a ``t_in``-wide band.

    Thickness is a sampling *offset*, not a dilation radius: a pixel at
    distance ``d`` inside the region has its outward neighbour at
    ``d - t``, which is outside whenever ``d < t``.  So four taps produce a
    filled band exactly ``t`` pixels wide, not a ring.
    """
    object_id = 700_123
    ids = np.zeros((_SIZE, _SIZE), dtype=np.uint32)
    ids[:, : _SIZE // 2] = object_id  # region on the left, background right

    harness = _SyntheticOutline(ids)
    harness.configure(
        _selection_config(t_in=thickness, t_out=0),
        has_inward=True,
        has_outward=False,
    )
    harness.set_entry(object_id, encode_entry(1, KIND_WHOLE_OBJECT, PLACEMENT_INWARD))
    frame = harness.render()

    painted = _painted_columns(frame, row=_SIZE // 2)
    edge = _SIZE // 2  # first background column
    assert painted == list(range(edge - thickness, edge))


def test_thin_region_is_consumed_inward(offscreen_renderer):
    """A region narrower than ``2 * t_in`` is fully covered by its outline."""
    object_id = 42_000
    stripe = slice(30, 33)  # 3 px wide, thinner than 2 * 2
    ids = np.zeros((_SIZE, _SIZE), dtype=np.uint32)
    ids[:, stripe] = object_id

    harness = _SyntheticOutline(ids)
    harness.configure(
        _selection_config(t_in=2, t_out=0), has_inward=True, has_outward=False
    )
    harness.set_entry(object_id, encode_entry(1, KIND_WHOLE_OBJECT, PLACEMENT_INWARD))
    frame = harness.render()

    assert _painted_columns(frame, row=_SIZE // 2) == [30, 31, 32]


def test_thin_region_survives_outward(offscreen_renderer):
    """Outward placement leaves the region intact and puts a halo outside.

    This is why lines and points default to outward: their screen-space
    defaults are 2 px and 5 px, so an inward band consumes them entirely.
    """
    object_id = 999_001
    stripe = slice(30, 33)
    ids = np.zeros((_SIZE, _SIZE), dtype=np.uint32)
    ids[:, stripe] = object_id

    harness = _SyntheticOutline(ids)
    harness.configure(
        _selection_config(t_in=0, t_out=2), has_inward=False, has_outward=True
    )
    harness.set_entry(object_id, encode_entry(1, KIND_WHOLE_OBJECT, PLACEMENT_OUTWARD))
    frame = harness.render()

    painted = _painted_columns(frame, row=_SIZE // 2)
    assert painted == [28, 29, 33, 34]
    # The stripe itself keeps the fill.
    fill = np.round(np.array(_FILL) * 255).astype(np.uint8)
    assert np.all(frame[_SIZE // 2, 30:33] == fill)


def test_unassigned_region_is_not_outlined(offscreen_renderer):
    """An id with no LUT entry produces no outline at all."""
    ids = np.zeros((_SIZE, _SIZE), dtype=np.uint32)
    ids[:, : _SIZE // 2] = 555_555

    harness = _SyntheticOutline(ids)
    harness.configure(
        _selection_config(t_in=2, t_out=2), has_inward=True, has_outward=True
    )
    frame = harness.render()

    fill = np.round(np.array(_FILL) * 255).astype(np.uint8)
    assert np.all(frame[..., :4] == fill)


def test_contrast_band_sits_immediately_inside_the_selection_outline(
    offscreen_renderer,
):
    """``inner_thickness`` adds a contrasting band just inside the outline."""
    object_id = 321_000
    ids = np.zeros((_SIZE, _SIZE), dtype=np.uint32)
    ids[:, : _SIZE // 2] = object_id

    harness = _SyntheticOutline(ids)
    config = _selection_config(t_in=2, t_out=0, inner_t=2)
    config.inner_color = (0.0, 1.0, 0.0, 1.0)
    harness.configure(config, has_inward=True, has_outward=False)
    harness.set_entry(object_id, encode_entry(1, KIND_WHOLE_OBJECT, PLACEMENT_INWARD))
    frame = harness.render()

    row = frame[_SIZE // 2]
    edge = _SIZE // 2
    selection = np.round(np.array(_SELECT) * 255).astype(np.uint8)
    inner = np.array([0, 255, 0, 255], dtype=np.uint8)
    assert np.all(row[edge - 2 : edge] == selection)
    assert np.all(row[edge - 4 : edge - 2] == inner)


def test_boundaries_layer_outlines_regions_without_a_slot(offscreen_renderer):
    """Slot 0 with a nonzero kind is still drawn by the boundaries layer.

    The edge test runs on the raw key, never on the resolved slot, so two
    regions that both resolve to "not selected" stay distinguishable.
    """
    object_id = 88_888
    ids = np.zeros((_SIZE, _SIZE), dtype=np.uint32)
    ids[:, : _SIZE // 2] = object_id

    harness = _SyntheticOutline(ids)
    config = OutlineConfig(
        enabled=True,
        boundaries=OutlineLayerConfig(
            enabled=True, inward_thickness=1, outward_thickness=0, color=(1, 1, 1, 1)
        ),
        selection=OutlineLayerConfig(enabled=False),
        inner_thickness=0,
    )
    harness.configure(config, has_inward=True, has_outward=False)
    harness.set_entry(object_id, encode_entry(0, KIND_WHOLE_OBJECT, PLACEMENT_INWARD))
    frame = harness.render()

    row = frame[_SIZE // 2]
    edge = _SIZE // 2
    assert np.all(row[edge - 1] == np.array([255, 255, 255, 255], dtype=np.uint8))
    assert np.all(row[edge - 2] == np.round(np.array(_FILL) * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Controller / RenderManager plumbing
# ---------------------------------------------------------------------------


def _sphere_mesh(size: int = 24, radius: int = 8):
    """Return ``(positions, indices)`` for a small sphere."""
    from skimage.measure import marching_cubes

    centre = size // 2
    z, y, x = np.mgrid[:size, :size, :size]
    volume = (x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2
    verts, faces, _, _ = marching_cubes(volume, level=radius**2)
    return verts.astype(np.float32), faces.astype(np.int32)


@pytest.fixture
def outline_controller(qtbot, offscreen_renderer):
    """A controller with outlines enabled, one mesh visual, one canvas."""
    from cellier.controller import CellierController
    from cellier.data import MeshMemoryStore
    from cellier.visuals import MeshPhongAppearance

    controller = CellierController(
        render_config=RenderManagerConfig(outline=OutlineConfig(enabled=True))
    )
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(dim="3d", name="scene", lighting="default")
    positions, indices = _sphere_mesh()
    visual = controller.add_mesh(
        data=MeshMemoryStore(positions=positions, indices=indices, name="sphere"),
        scene_id=scene.id,
        appearance=MeshPhongAppearance(color=(0.4, 0.7, 1.0, 1.0), side="both"),
        name="sphere",
    )
    controller.add_canvas(scene_id=scene.id)
    return controller, scene, visual


def test_set_visual_outline_populates_every_world_object(outline_controller):
    """One cellier visual maps to several pygfx objects; all get the entry."""
    from cellier.render._visual_lut import get_shared_visual_lut

    controller, scene, visual = outline_controller
    get_shared_visual_lut().clear()

    controller.set_visual_outline(visual.id, slot=2, placement="inward")

    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    expected_ids = {
        obj.id
        for mode in ("2d", "3d")
        if gfx_visual.get_node(mode) is not None
        for obj in gfx_visual.get_node(mode).iter()
    }
    assert expected_ids  # the mesh node exists at registration time

    entries = get_shared_visual_lut().entries
    assert set(entries) == expected_ids
    expected_value = encode_entry(2, KIND_WHOLE_OBJECT, PLACEMENT_INWARD)
    assert set(entries.values()) == {expected_value}

    assert controller.get_visual_outline(visual.id) == (2, "inward")


def test_set_visual_outline_slot_zero_clears(outline_controller):
    """Slot 0 removes the assignment and the LUT entries with it."""
    from cellier.render._visual_lut import get_shared_visual_lut

    controller, _scene, visual = outline_controller
    controller.set_visual_outline(visual.id, slot=1)
    assert get_shared_visual_lut().entries

    controller.set_visual_outline(visual.id, slot=0)
    assert get_shared_visual_lut().entries == {}
    assert controller.get_visual_outline(visual.id) is None


def test_set_visual_outline_forces_pick_write(outline_controller):
    """Outlines come from the pick buffer, so picking is turned back on."""
    controller, _scene, visual = outline_controller
    visual.pick_write = False

    controller.set_visual_outline(visual.id, slot=1)

    assert visual.pick_write is True


def test_lines_default_to_outward_placement(qtbot, offscreen_renderer):
    """Lines are 2 px in screen space; an inward band would consume them."""
    from cellier.controller import CellierController
    from cellier.data import LinesMemoryStore
    from cellier.visuals import LinesMemoryAppearance

    controller = CellierController(
        render_config=RenderManagerConfig(outline=OutlineConfig(enabled=True))
    )
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(dim="3d", name="scene")
    coordinates = np.array(
        [[0, 0, 0], [0, 10, 10], [0, 10, 0], [0, 0, 10]], dtype=np.float32
    )
    visual = controller.add_lines(
        data=LinesMemoryStore(positions=coordinates, name="lines"),
        scene_id=scene.id,
        appearance=LinesMemoryAppearance(),
        name="lines",
    )
    controller.add_canvas(scene_id=scene.id)

    controller.set_visual_outline(visual.id, slot=1)
    assert controller.get_visual_outline(visual.id) == (1, "outward")

    # An explicit placement still wins.
    controller.set_visual_outline(visual.id, slot=1, placement="inward")
    assert controller.get_visual_outline(visual.id) == (1, "inward")


def test_set_visual_outline_validates_arguments(outline_controller):
    """Out-of-range slots and unknown placements raise."""
    controller, _scene, visual = outline_controller
    with pytest.raises(ValueError, match="slot"):
        controller.set_visual_outline(visual.id, slot=MAX_OUTLINE_SLOT + 1)
    with pytest.raises(ValueError, match="placement"):
        controller.set_visual_outline(visual.id, slot=1, placement="sideways")


def test_live_layer_setters_reach_every_canvas(outline_controller):
    """The RenderManager setters follow the temporal_* pattern."""
    controller, _scene, _visual = outline_controller
    manager = controller._render_manager
    canvas = next(iter(manager._canvases.values()))
    quad = canvas._outline_pass._quad_pass

    manager.outline_enabled = False
    assert manager.config.outline.enabled is False
    assert canvas._outline_pass.enabled is False

    manager.outline_enabled = True
    assert canvas._outline_pass.enabled is True

    manager.outline_boundaries_enabled = False
    assert manager.config.outline.boundaries.enabled is False
    assert int(quad._uniform_data["boundaries_enabled"]) == 0

    manager.outline_selection_enabled = False
    assert manager.config.outline.selection.enabled is False
    assert int(quad._uniform_data["selection_enabled"]) == 0


def test_placement_template_vars_collapse_to_what_is_present(outline_controller):
    """A scene with only inward visuals compiles the outward branch away."""
    controller, _scene, visual = outline_controller
    canvas = next(iter(controller._render_manager._canvases.values()))
    quad = canvas._outline_pass._quad_pass

    controller.set_visual_outline(visual.id, slot=1, placement="inward")
    assert quad._template_vars["has_inward"] is True
    assert quad._template_vars["has_outward"] is False

    controller.set_visual_outline(visual.id, slot=1, placement="outward")
    assert quad._template_vars["has_inward"] is False
    assert quad._template_vars["has_outward"] is True


def test_outline_survives_a_world_object_rebuild(outline_controller):
    """A rebuilt node gets a fresh global_id; the sync must re-populate it.

    ``wobject.id`` is per-WorldObject, and cellier rebuilds world objects on
    2D/3D switches, multiscale brick updates and channel changes.  A
    write-once LUT would silently lose its entries; the per-frame re-sync
    from ``_visual_flags`` is what keeps them alive.
    """
    import pygfx as gfx

    from cellier.render._visual_lut import get_shared_visual_lut

    controller, scene, visual = outline_controller
    controller.set_visual_outline(visual.id, slot=1)
    original = set(get_shared_visual_lut().entries)
    assert original

    # Stand in for a rebuild: swap the node for a new one, which claims a
    # new id from the same provider.
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    replacement = gfx.Mesh(gfx.box_geometry(), gfx.MeshBasicMaterial(pick_write=True))
    gfx_visual.node_3d = replacement
    gfx_visual.node_2d = replacement
    gfx_visual.node = replacement

    controller._render_manager._sync_visual_lut()

    entries = get_shared_visual_lut().entries
    assert set(entries) == {replacement.id}
    assert original.isdisjoint(entries)


def test_disabled_pass_leaves_the_frame_pixel_identical(offscreen_renderer):
    """With outlines off, the frame matches one rendered without the pass.

    ``flush()`` skips disabled effect passes entirely, which is what lets
    every canvas install the pass unconditionally and still leave scenes
    that never opted in untouched.
    """
    from rendercanvas.offscreen import RenderCanvas

    from cellier.render._outline import OutlinePass

    def _frame(install_pass: bool) -> np.ndarray:
        canvas = RenderCanvas(size=(96, 96), pixel_ratio=1)
        renderer = gfx.WgpuRenderer(canvas)
        enable_pick_texture_binding(renderer)
        scene = gfx.Scene()
        mesh = gfx.Mesh(
            gfx.sphere_geometry(1.0, 32, 16),
            gfx.MeshBasicMaterial(color="#ff8800", pick_write=True),
        )
        scene.add(mesh)
        camera = gfx.OrthographicCamera()
        camera.show_object(scene)
        if install_pass:
            outline = OutlinePass(renderer, VisualLut())
            outline.enabled = False
            renderer.effect_passes = (outline, *renderer.effect_passes)
        canvas.request_draw(lambda: renderer.render(scene, camera))
        return np.asarray(canvas.draw())

    np.testing.assert_array_equal(_frame(install_pass=True), _frame(install_pass=False))


# ---------------------------------------------------------------------------
# Placement mixing, in one pass
# ---------------------------------------------------------------------------


def test_inward_and_outward_visuals_are_both_correct_in_one_pass(offscreen_renderer):
    """One shader invocation serves an inward region and an outward one.

    Also pins the documented precedence: where an inward region touches an
    outward one, the shared border pixels belong to the *inward* region,
    because a pixel inside a region wins over a halo cast onto it by a
    neighbour.
    """
    inward_id = 640_001
    outward_id = 91_234
    ids = np.zeros((_SIZE, _SIZE), dtype=np.uint32)
    ids[:, 20:31] = inward_id  # columns 20..30
    ids[:, 31:41] = outward_id  # columns 31..40, sharing a border at 30/31

    harness = _SyntheticOutline(ids)
    config = _selection_config(t_in=2, t_out=2)
    config.palette = [(1.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0)]
    harness.configure(config, has_inward=True, has_outward=True)
    harness.set_entry(inward_id, encode_entry(1, KIND_WHOLE_OBJECT, PLACEMENT_INWARD))
    harness.set_entry(outward_id, encode_entry(2, KIND_WHOLE_OBJECT, PLACEMENT_OUTWARD))
    frame = harness.render()

    row = frame[_SIZE // 2]
    magenta = np.round(np.array(config.palette[0]) * 255).astype(np.uint8)
    cyan = np.round(np.array(config.palette[1]) * 255).astype(np.uint8)
    fill = np.round(np.array(_FILL) * 255).astype(np.uint8)

    inward_cols = np.flatnonzero(np.all(row == magenta, axis=-1)).tolist()
    outward_cols = np.flatnonzero(np.all(row == cyan, axis=-1)).tolist()

    # Inward: a band inside its own footprint on both borders, including
    # the border it shares with the outward region.
    assert inward_cols == [20, 21, 29, 30]
    # Outward: a halo only on its free side.  The border it shares with the
    # inward region is owned by that region, and an inward neighbour casts
    # no halo back onto it.
    assert outward_cols == [41, 42]
    # The outward region's own pixels keep the fill.
    assert np.all(row[31:41] == fill)


# ---------------------------------------------------------------------------
# Instanced meshes
# ---------------------------------------------------------------------------


class _SceneOutline:
    """Render a pygfx scene through an ``OutlinePass``, 1:1 and unfiltered.

    ``pixel_scale = 1`` and ``ppaa = "none"`` so an outline band lands on
    exactly the pixels the shader wrote.  The plan asks for
    ``PYGFX_DEFAULT_PPAA=none``; the property does the same thing without
    depending on process-wide state set before pygfx imports.
    """

    def __init__(self, lut: VisualLut | None = None, size=(128, 128)) -> None:
        from rendercanvas.offscreen import RenderCanvas

        from cellier.render._outline import OutlinePass

        self.canvas = RenderCanvas(size=size, pixel_ratio=1)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.renderer.pixel_scale = 1
        self.renderer.ppaa = "none"
        enable_pick_texture_binding(self.renderer)

        self.lut = lut if lut is not None else VisualLut()
        self.pass_ = OutlinePass(self.renderer, self.lut)
        # ppaa is set first: its setter rebuilds effect_passes, which would
        # otherwise drop the outline pass again.
        self.renderer.effect_passes = (self.pass_, *self.renderer.effect_passes)

    def configure(self, config: OutlineConfig, *, has_inward, has_outward) -> None:
        """Apply a config and declare which placements are present."""
        self.pass_.apply_config(config)
        self.pass_.set_placements(has_inward=has_inward, has_outward=has_outward)

    def render(self, scene: gfx.Scene, camera: gfx.Camera) -> np.ndarray:
        """Draw *scene* and return the frame as an ``(h, w, 4)`` uint8 array."""
        errors: list[BaseException] = []

        def _draw() -> None:
            try:
                self.renderer.render(scene, camera)
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)
                raise

        self.canvas.request_draw(_draw)
        image = self.canvas.draw()
        if errors:  # pragma: no cover - failure path
            raise RuntimeError(
                f"offscreen draw failed -- {type(errors[0]).__name__}: {errors[0]}"
            ) from errors[0]
        return np.asarray(image)


def _instanced_scene(count: int = 3):
    """Return ``(scene, camera, instanced_mesh)`` with *count* spread boxes."""
    scene = gfx.Scene()
    mesh = gfx.InstancedMesh(
        gfx.box_geometry(1.0, 1.0, 1.0),
        gfx.MeshBasicMaterial(color="#ff8800", pick_write=True),
        count,
    )
    for index in range(count):
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = (index - (count - 1) / 2) * 2.0
        mesh.set_matrix_at(index, matrix)
    scene.add(mesh)
    camera = gfx.OrthographicCamera()
    camera.show_object(scene)
    return scene, camera, mesh


def _instance_ids(mesh: gfx.InstancedMesh) -> list[int]:
    return [int(v) for v in mesh.instance_buffer.data["global_id"].ravel()]


def test_instanced_mesh_outlines_from_instance_ids(offscreen_renderer):
    """``mesh.wgsl`` writes the *instance* id to pick, not the object's own.

    Populating the LUT with ``wobject.id`` leaves an instanced mesh with no
    outline at all, which is a silent failure -- hence both halves here.
    """
    scene, camera, mesh = _instanced_scene()
    config = _selection_config(t_in=2, t_out=0)

    harness = _SceneOutline()
    harness.configure(config, has_inward=True, has_outward=False)
    entry = encode_entry(1, KIND_WHOLE_OBJECT, PLACEMENT_INWARD)

    # Wrong: the world object's own id is never written to pick.
    harness.lut.apply({mesh.id: entry})
    wrong = _count_exact(harness.render(scene, camera), _SELECT)
    assert wrong == 0

    # Right: the per-instance ids, claimed from the same id provider.
    instance_ids = _instance_ids(mesh)
    assert mesh.id not in instance_ids
    harness.lut.apply(dict.fromkeys(instance_ids, entry))
    outlined = _count_exact(harness.render(scene, camera), _SELECT)
    assert outlined > 0


def test_instanced_mesh_outlines_individual_instances(offscreen_renderer):
    """Per-instance ids mean per-instance outlining comes for free."""
    scene, camera, mesh = _instanced_scene()
    config = _selection_config(t_in=2, t_out=0)
    config.palette = [(1.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0)]

    harness = _SceneOutline()
    harness.configure(config, has_inward=True, has_outward=False)

    instance_ids = _instance_ids(mesh)
    harness.lut.apply(
        {
            instance_ids[0]: encode_entry(1, KIND_WHOLE_OBJECT, PLACEMENT_INWARD),
            instance_ids[2]: encode_entry(2, KIND_WHOLE_OBJECT, PLACEMENT_INWARD),
        }
    )
    frame = harness.render(scene, camera)

    first = _count_exact(frame, config.palette[0])
    third = _count_exact(frame, config.palette[1])
    assert first > 0
    assert third > 0
    # The middle instance was left out and picks up nothing.
    assert first == third  # identical boxes, identical band


def test_world_object_ids_collects_instance_ids(offscreen_renderer):
    """``RenderManager._world_object_ids`` must prefer instance ids."""
    from cellier.render.render_manager import RenderManager

    _scene, _camera, mesh = _instanced_scene()

    class _Visual:
        def get_node(self, mode):
            return mesh if mode == "3d" else None

    found = RenderManager._world_object_ids(_Visual())
    assert found == set(_instance_ids(mesh))
    assert mesh.id not in found


# ---------------------------------------------------------------------------
# Render image tests: every layer combination, in 2D and 3D
# ---------------------------------------------------------------------------

_BOUNDARY = (0.0, 0.0, 1.0, 1.0)

# The boundaries band is deliberately *wider* than the selection band, so
# both layers are visible at once when both are on.  With the reverse the
# thin boundary band would sit entirely inside the selection band and the
# "both" case would be indistinguishable from "selection only" -- which is
# correct behaviour (selection > boundaries) but tests nothing.
_BOUNDARY_T = 5
_SELECTION_T = 2


def _layer_config(*, boundaries: bool, selection: bool) -> OutlineConfig:
    return OutlineConfig(
        enabled=True,
        boundaries=OutlineLayerConfig(
            enabled=boundaries,
            inward_thickness=_BOUNDARY_T,
            outward_thickness=0,
            color=_BOUNDARY,
        ),
        selection=OutlineLayerConfig(
            enabled=selection,
            inward_thickness=_SELECTION_T,
            outward_thickness=0,
        ),
        inner_thickness=0,
        palette=[_SELECT],
    )


def _layer_counts(controller, scene_id, *, boundaries: bool, selection: bool):
    """Render one layer combination; return ``(boundary px, selection px)``."""
    from cellier.render._visual_lut import get_shared_visual_lut

    canvas_id = controller.get_canvas_ids(scene_id)[0]
    canvas_view = controller.get_canvas_view(canvas_id)

    harness = _SceneOutline(lut=get_shared_visual_lut())
    harness.configure(
        _layer_config(boundaries=boundaries, selection=selection),
        has_inward=True,
        has_outward=False,
    )
    gfx_scene = controller._render_manager.get_scene(scene_id)
    frame = harness.render(gfx_scene, canvas_view.camera)
    return _count_exact(frame, _BOUNDARY), _count_exact(frame, _SELECT), frame


def _assert_layer_combinations(controller, scene_id) -> None:
    """All four layer combinations behave, and precedence holds.

    Precedence is selection > boundaries > fill.  With the boundaries band
    the wider of the two, "both on" must show the selection band at its
    full width and the boundaries band only in the strip the selection band
    does not already cover.
    """
    neither_b, neither_s, neither_frame = _layer_counts(
        controller, scene_id, boundaries=False, selection=False
    )
    assert (neither_b, neither_s) == (0, 0)

    bounds_b, bounds_s, _ = _layer_counts(
        controller, scene_id, boundaries=True, selection=False
    )
    assert bounds_b > 0
    assert bounds_s == 0

    select_b, select_s, _ = _layer_counts(
        controller, scene_id, boundaries=False, selection=True
    )
    assert select_b == 0
    assert select_s > 0

    both_b, both_s, both_frame = _layer_counts(
        controller, scene_id, boundaries=True, selection=True
    )
    # Selection is untouched by the layer underneath it.
    assert both_s == select_s
    # Boundaries survive only outside the selection band, so strictly fewer
    # pixels than boundaries-only -- but not none, since it is the wider band.
    assert 0 < both_b < bounds_b
    # The wider band really is wider.
    assert bounds_b > select_s
    # And nothing is drawn at all with both layers off.
    assert np.count_nonzero(np.any(neither_frame != both_frame, axis=-1)) > 0


async def test_layer_combinations_in_3d(outline_controller, reslice):
    """Each layer combination renders correctly on a 3D mesh."""
    controller, scene, visual = outline_controller
    await reslice(controller, scene.id)
    # Re-fit *after* the data lands.  A mesh visual's bounding box comes from
    # its positions buffer, which the async reslice uploads, so the fit inside
    # the ``reslice`` fixture frames a placeholder box and leaves the real
    # mesh mostly out of shot.
    controller.fit_camera(scene.id)
    controller.set_visual_outline(visual.id, slot=1, placement="inward")

    _assert_layer_combinations(controller, scene.id)


async def test_layer_combinations_in_2d(
    qtbot, offscreen_renderer, reslice, gradient_image
):
    """Each layer combination renders correctly on a 2D image.

    2D matters on its own: a flat image plane has near-constant depth, so a
    depth/normal edge detector would find nothing there.  An id-based one
    produces the same contour from the same code path.
    """
    from cellier.controller import CellierController
    from cellier.visuals import InMemoryImageAppearance

    controller = CellierController(
        render_config=RenderManagerConfig(outline=OutlineConfig(enabled=True))
    )
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_image(
        data=gradient_image,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    controller.set_visual_outline(visual.id, slot=1, placement="inward")

    _assert_layer_combinations(controller, scene.id)


def test_outline_survives_a_resize(offscreen_renderer):
    """Resizing keeps the pick binding, so outlines keep drawing.

    ``ensure_target_size`` drops the blender's textures but keeps
    ``_texture_info``, which is the whole reason granting the usage once,
    before the first draw, is sufficient.  A pygfx change that rebuilt
    ``_texture_info`` on resize would break outlining silently at the first
    window drag; this is the test that would catch it.
    """
    scene, camera, mesh = _instanced_scene(count=1)
    harness = _SceneOutline(size=(128, 128))
    harness.configure(
        _selection_config(t_in=2, t_out=0), has_inward=True, has_outward=False
    )
    harness.lut.apply(
        dict.fromkeys(
            _instance_ids(mesh), encode_entry(1, KIND_WHOLE_OBJECT, PLACEMENT_INWARD)
        )
    )

    before = _count_exact(harness.render(scene, camera), _SELECT)
    assert before > 0

    harness.canvas.set_logical_size(96, 160)
    camera.show_object(scene)
    after = _count_exact(harness.render(scene, camera), _SELECT)
    assert after > 0


def test_outline_pass_precedes_temporal_accumulation(outline_controller):
    """Ordering in the effect chain, which the design pins deliberately.

    The volume raymarcher jitters per frame, so silhouette pixels shift
    sub-pixel between frames.  Compositing the outline *before*
    accumulation lets the EMA average that into a free antialiased edge
    instead of a flicker.  If outlines ever flicker on a jittered volume,
    this ordering is the first thing to check.
    """
    from cellier.render._outline import OutlinePass
    from cellier.render._temporal_accumulation import TemporalAccumulationPass

    controller, _scene, _visual = outline_controller
    canvas = next(iter(controller._render_manager._canvases.values()))
    passes = list(canvas._renderer.effect_passes)

    outline_index = next(i for i, p in enumerate(passes) if isinstance(p, OutlinePass))
    accum_index = next(
        i for i, p in enumerate(passes) if isinstance(p, TemporalAccumulationPass)
    )
    assert outline_index < accum_index
    # And the antialiasing pass stays last, so it antialiases the outline.
    assert accum_index < len(passes) - 1


def test_canvas_with_outlines_off_never_allocates_the_lut(qtbot, offscreen_renderer):
    """A canvas that never opts in pays nothing, not even the 1 MB table.

    The pass is installed on every canvas so runtime toggling works, but the
    table is resolved on first draw of an *enabled* pass, and the per-frame
    sync short-circuits while nothing is assigned.
    """
    import cellier.render._visual_lut as lut_module
    from cellier.controller import CellierController

    saved = lut_module._SHARED_LUT
    lut_module._SHARED_LUT = None
    try:
        controller = CellierController()
        controller.camera_reslice_enabled = False
        scene = controller.add_scene(dim="3d", name="scene")
        controller.add_canvas(scene_id=scene.id)

        canvas = next(iter(controller._render_manager._canvases.values()))
        assert canvas._outline_pass.enabled is False
        controller._render_manager._sync_visual_lut()
        assert lut_module.peek_shared_visual_lut() is None

        # Touching the property is what allocates it.
        assert canvas._outline_pass.lut is lut_module.peek_shared_visual_lut()
        assert lut_module.peek_shared_visual_lut() is not None
    finally:
        lut_module._SHARED_LUT = saved
