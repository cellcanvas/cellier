"""Tests for the depth edges of the screen-space outline pass.

A depth edge is a band where two pixels of the *same* key lie on surfaces a
depth jump apart: one part of a visual passing in front of another.

Most of these drive ``_OutlineQuadPass`` over *fabricated* pick, label-key
and depth buffers, as ``test_outline.py`` does for the identity edges, so
the band can be named pixel by pixel.  Every fabricated scene is constant
along y: the columns of one row say everything.  The last group draws a
real labels volume through a canvas.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pygfx as gfx
import pytest
import wgpu

from cellier.render._config import OutlineConfig, OutlineLayerConfig
from cellier.render._visual_lut import (
    KIND_LABEL_ALL,
    KIND_WHOLE_OBJECT,
    PLACEMENT_INWARD,
    PLACEMENT_OUTWARD,
    VisualLut,
    decode_depth_range,
    encode_depth_range,
    encode_entry,
)
from tests.render.conftest import drain_loading

_SIZE = 64
_ROW = _SIZE // 2
_FILL = (0.2, 0.2, 0.2, 1.0)
_SELECT = (1.0, 0.0, 1.0, 1.0)
_BOUNDARY = (0.0, 1.0, 0.0, 1.0)
_INNER = (0.0, 0.0, 1.0, 1.0)

#: The camera the fabricated depth is written for.
_NEAR, _FAR = 5.0, 40.0
_CAMERA = gfx.PerspectiveCamera(50, 1, depth_range=(_NEAR, _FAR))
_PROJECTION = np.asarray(_CAMERA.projection_matrix, dtype=np.float64)
_PROJECTION_INV = np.asarray(_CAMERA.projection_matrix_inverse, dtype=np.float64)

_A = 300_001
_B = 700_002


def _raw_of(z) -> np.ndarray:
    """The raw depth of a distance *z* from the camera plane."""
    z = np.asarray(z, dtype=np.float64)
    clip_z = _PROJECTION[2, 2] * -z + _PROJECTION[2, 3]
    clip_w = _PROJECTION[3, 2] * -z + _PROJECTION[3, 3]
    return clip_z / clip_w


def _z_of(raw) -> np.ndarray:
    """The distance of a raw depth, as the shader recovers it."""
    raw = np.asarray(raw, dtype=np.float64)
    z = _PROJECTION_INV[2, 2] * raw + _PROJECTION_INV[2, 3]
    w = _PROJECTION_INV[3, 2] * raw + _PROJECTION_INV[3, 3]
    return -z / w


def _columns(values) -> np.ndarray:
    """A ``(_SIZE, _SIZE)`` image holding *values* in every row."""
    return np.tile(np.asarray(values)[None, :], (_SIZE, 1))


class _SyntheticDepthOutline:
    """Run ``_OutlineQuadPass`` over hand-built pick, key and depth buffers.

    Parameters
    ----------
    ids : np.ndarray
        ``(h, w)`` pygfx object ids; 0 is the background.
    z : np.ndarray
        ``(h, w)`` distance of each pixel's surface from the camera plane.
    label_keys : np.ndarray or None
        ``(h, w)`` label keys, for visuals of a label kind.  ``None`` binds
        no label-key target.
    """

    def __init__(self, ids, z, label_keys=None) -> None:
        from pygfx.renderers.wgpu.engine.shared import get_shared

        from cellier.render._outline import _OutlineQuadPass

        self._device = get_shared().device
        height, width = ids.shape
        self._size = (width, height)

        pick = np.zeros((height, width, 4), dtype=np.uint16)
        pick[..., 0] = ids & 0xFFFF
        pick[..., 1] = (ids >> 16) & 0xF
        self._pick_view = self._upload(pick, wgpu.TextureFormat.rgba16uint, 8)

        color = np.zeros((height, width, 4), dtype=np.uint8)
        color[:] = np.round(np.array(_FILL) * 255).astype(np.uint8)
        self._color_view = self._upload(color, wgpu.TextureFormat.rgba8unorm, 4)

        # A depth32float texture cannot be written from the CPU; 16 bits is
        # ample here.  ``raw`` is the depth as stored.
        codes = np.round(_raw_of(z) * 65535.0).astype(np.uint16)
        self.raw = codes / 65535.0
        self._depth_view = self._upload(codes, wgpu.TextureFormat.depth16unorm, 2)

        self._key_view = None
        if label_keys is not None:
            self._key_view = self._upload(
                np.ascontiguousarray(label_keys, dtype=np.uint32),
                wgpu.TextureFormat.r32uint,
                4,
            )

        self._target = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
            dimension="2d",
        )
        self._lut = VisualLut()
        self._quad = _OutlineQuadPass()
        self._quad.set_uniform(
            "projection_transform_inv", _PROJECTION_INV.astype(np.float32).T
        )

    def _upload(self, data, fmt, bytes_per_texel):
        width, height = self._size
        texture = self._device.create_texture(
            size=(width, height, 1),
            format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d",
        )
        self._device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(data),
            {
                "offset": 0,
                "bytes_per_row": bytes_per_texel * width,
                "rows_per_image": height,
            },
            (width, height, 1),
        )
        return texture.create_view()

    def configure(self, config: OutlineConfig, *, placement: int) -> None:
        """Apply an ``OutlineConfig`` for visuals of one *placement*."""
        quad = self._quad
        quad.set_template_vars(
            b_t_in=config.boundaries.inward_thickness,
            b_t_out=config.boundaries.outward_thickness,
            s_t_in=config.selection.inward_thickness,
            s_t_out=config.selection.outward_thickness,
            inner_t=config.inner_thickness,
            has_inward=placement == PLACEMENT_INWARD,
            has_outward=placement == PLACEMENT_OUTWARD,
        )
        quad.set_uniform("boundaries_enabled", int(config.boundaries.enabled))
        quad.set_uniform("selection_enabled", int(config.selection.enabled))
        quad.set_uniform("boundary_color", tuple(config.boundaries.color))
        quad.set_uniform("inner_color", tuple(config.inner_color))
        quad.set_palette([tuple(c) for c in config.palette])

    def set_visual(
        self, object_id, *, placement, depth_range=None, kind=KIND_WHOLE_OBJECT, slot=1
    ) -> None:
        """Write one visual's entry, with or without a depth range."""
        start_code, ramp_code = (
            (0, 0) if depth_range is None else encode_depth_range(depth_range)
        )
        self._lut.set_entry(
            object_id,
            encode_entry(
                slot,
                kind,
                placement,
                depth_start_code=start_code,
                depth_ramp_code=ramp_code,
            ),
        )

    def render(self, *, depth_edges: bool = True) -> np.ndarray:
        """Draw and read the target back as an ``(h, w, 4)`` uint8 array."""
        width, height = self._size
        encoder = self._device.create_command_encoder()
        self._quad.render(
            encoder,
            self._color_view,
            self._pick_view,
            self._lut.view,
            self._target.create_view(),
            self._key_view,
            self._depth_view if depth_edges else None,
        )
        self._device.queue.submit([encoder.finish()])
        raw = self._device.queue.read_texture(
            {"texture": self._target, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": 4 * width, "rows_per_image": height},
            (width, height, 1),
        )
        return np.frombuffer(raw, np.uint8).reshape(height, width, 4)


def _config(t: int, *, boundaries: int = 0, inner_t: int = 0) -> OutlineConfig:
    """The selection layer at thickness *t* on both sides; other layers opt in."""
    return OutlineConfig(
        enabled=True,
        boundaries=OutlineLayerConfig(
            enabled=boundaries > 0,
            inward_thickness=max(boundaries, 1),
            outward_thickness=max(boundaries, 1),
            color=_BOUNDARY,
        ),
        selection=OutlineLayerConfig(
            enabled=True, inward_thickness=t, outward_thickness=t
        ),
        inner_thickness=inner_t,
        inner_color=_INNER,
        palette=[_SELECT],
    )


def _rgb8(rgba) -> np.ndarray:
    return np.round(np.array(rgba[:3]) * 255).astype(np.uint8)


def _changed_columns(frame: np.ndarray, row: int = _ROW) -> list[int]:
    """Columns of *row* that are not the fill."""
    return np.flatnonzero(np.any(frame[row, :, :3] != _rgb8(_FILL), axis=-1)).tolist()


def _is_select(frame: np.ndarray, columns, row: int = _ROW) -> bool:
    return bool(np.all(frame[row, columns, :3] == _rgb8(_SELECT)))


def _step_scene(z_near: float = 10.0, z_far: float = 20.0, edge: int = _SIZE // 2):
    """One object filling the frame: near left of *edge*, far from it on."""
    ids = np.full((_SIZE, _SIZE), _A, dtype=np.uint32)
    z = _columns(np.where(np.arange(_SIZE) < edge, z_near, z_far))
    return ids, z


def _band_columns(placement: int, edge: int, t: int = 2) -> list[int]:
    """The band of a jump at *edge*: on the near side inward, the far outward."""
    if placement == PLACEMENT_INWARD:
        return list(range(edge - t, edge))
    return list(range(edge, edge + t))


def _over(colour, alpha: float, below) -> np.ndarray:
    """*colour* at *alpha* over *below*, as floats in [0, 1]."""
    return np.asarray(below[:3]) * (1.0 - alpha) + np.asarray(colour[:3]) * alpha


def _smoothstep(low: float, high: float, value: float) -> float:
    x = min(max((value - low) / (high - low), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


# ---------------------------------------------------------------------------
# Where the band lies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thickness", [1, 2, 4])
@pytest.mark.parametrize(
    "placement, side", [(PLACEMENT_INWARD, "near"), (PLACEMENT_OUTWARD, "far")]
)
def test_a_depth_jump_inside_one_object_is_outlined(
    offscreen_renderer, thickness, placement, side
):
    """Inward the band is on the nearer surface, outward on the farther.

    The object fills the frame, so it has no identity edge at all: with no
    depth range nothing is drawn.
    """
    edge = _SIZE // 2
    harness = _SyntheticDepthOutline(*_step_scene(edge=edge))
    harness.configure(_config(thickness), placement=placement)

    harness.set_visual(_A, placement=placement)
    assert _changed_columns(harness.render()) == []

    harness.set_visual(_A, placement=placement, depth_range=(0.02, 0.02))
    frame = harness.render()
    expected = (
        list(range(edge - thickness, edge))
        if side == "near"
        else list(range(edge, edge + thickness))
    )
    assert _changed_columns(frame) == expected
    # A hard range draws the line at its full alpha.
    assert _is_select(frame, expected)
    # Constant along y: every row is the same.
    assert np.all(frame == frame[_ROW])


@pytest.mark.parametrize("placement", [PLACEMENT_INWARD, PLACEMENT_OUTWARD])
def test_a_jump_under_the_start_is_not_an_edge(offscreen_renderer, placement):
    """The start is relative to the distance of the nearer surface."""
    harness = _SyntheticDepthOutline(*_step_scene(z_near=10.0, z_far=10.4))
    harness.configure(_config(2), placement=placement)

    harness.set_visual(_A, placement=placement, depth_range=(0.05, 0.05))
    assert _changed_columns(harness.render()) == []

    harness.set_visual(_A, placement=placement, depth_range=(0.03, 0.03))
    assert len(_changed_columns(harness.render())) == 2


@pytest.mark.parametrize("placement", [PLACEMENT_INWARD, PLACEMENT_OUTWARD])
def test_without_the_depth_texture_the_range_draws_nothing(
    offscreen_renderer, placement
):
    """The term is compiled away when no depth is handed to the pass."""
    harness = _SyntheticDepthOutline(*_step_scene())
    harness.configure(_config(2), placement=placement)
    harness.set_visual(_A, placement=placement, depth_range=(0.02, 0.02))
    assert _changed_columns(harness.render(depth_edges=False)) == []
    assert harness._quad._template_vars["has_depth_edges"] is False


# ---------------------------------------------------------------------------
# The second condition: the near pixel in front of the far surface carried on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thickness", [1, 2, 4])
@pytest.mark.parametrize("placement", [PLACEMENT_INWARD, PLACEMENT_OUTWARD])
def test_a_slanted_surface_is_not_an_edge(offscreen_renderer, thickness, placement):
    """One flat surface seen at a slant has depth jumps and no edge.

    The plain gap between two pixels a thickness apart is over the start
    all the way across; what stops a band is that the near pixel lies *on*
    the far surface carried on to it.  The surface runs off the frame on
    both sides, so this also holds the rule at the frame's edge: there the
    surface cannot be carried on from a texel outside the frame, and
    reading the clamped texel instead would draw a band along the edge.
    """
    start = 0.004
    # Raw depth is linear across a plane on screen.
    raw = np.linspace(float(_raw_of(10.0)), float(_raw_of(30.0)), _SIZE)
    z = _z_of(raw)
    # The test has teeth: the plain gap is over the start everywhere.
    assert np.all((z[thickness:] - z[:-thickness]) > start * z[:-thickness])

    ids = np.full((_SIZE, _SIZE), _A, dtype=np.uint32)
    harness = _SyntheticDepthOutline(ids, _columns(z))
    harness.configure(_config(thickness), placement=placement)
    harness.set_visual(_A, placement=placement, depth_range=(start, start))
    assert _changed_columns(harness.render()) == []


@pytest.mark.parametrize("placement", [PLACEMENT_INWARD, PLACEMENT_OUTWARD])
def test_a_near_surface_over_a_slanted_one_is_an_edge(offscreen_renderer, placement):
    """A real overlap is still drawn when the far surface is slanted."""
    edge = _SIZE // 2
    raw = np.linspace(float(_raw_of(18.0)), float(_raw_of(30.0)), _SIZE)
    z = np.where(np.arange(_SIZE) < edge, 10.0, _z_of(raw))
    ids = np.full((_SIZE, _SIZE), _A, dtype=np.uint32)
    harness = _SyntheticDepthOutline(ids, _columns(z))
    harness.configure(_config(2), placement=placement)
    harness.set_visual(_A, placement=placement, depth_range=(0.004, 0.004))
    expected = _band_columns(placement, edge)
    assert _changed_columns(harness.render()) == expected


# ---------------------------------------------------------------------------
# A far surface too narrow to be carried on from a thickness away (T2Fs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "far_width, expected_width", [(1, 0), (2, 1), (3, 2), (4, 2), (8, 2)]
)
def test_outward_a_narrow_far_surface_is_carried_on_from_one_pixel(
    offscreen_renderer, far_width, expected_width
):
    """The band is lost only where the far surface is one pixel wide.

    Past a far pixel the surface is carried on from the pixel a thickness
    beyond it, and where that is off the surface, from the pixel one step
    beyond.  With only the first of the two a far surface 3 pixels wide
    would get a band 1 pixel wide at thickness 2, and one 2 wide none.
    """
    edge = 30
    columns = np.arange(_SIZE)
    ids_row = np.where(columns < edge + far_width, _A, 0).astype(np.uint32)
    z = np.where(columns < edge, 10.0, 20.0)
    harness = _SyntheticDepthOutline(_columns(ids_row), _columns(z))
    harness.configure(_config(2), placement=PLACEMENT_OUTWARD)

    # Outward, the object's own silhouette is drawn on the background.
    harness.set_visual(_A, placement=PLACEMENT_OUTWARD)
    silhouette = _changed_columns(harness.render())
    assert silhouette == [edge + far_width, edge + far_width + 1]

    harness.set_visual(_A, placement=PLACEMENT_OUTWARD, depth_range=(0.02, 0.02))
    band = sorted(set(_changed_columns(harness.render())) - set(silhouette))
    assert band == list(range(edge, edge + expected_width))


@pytest.mark.parametrize(
    "far_width, expected", [(1, []), (2, [28]), (3, [28, 29]), (4, [28, 29])]
)
def test_inward_a_narrow_far_surface_is_carried_on_from_one_pixel(
    offscreen_renderer, far_width, expected
):
    """The same rule with the centre as the near pixel."""
    edge = 30
    columns = np.arange(_SIZE)
    ids_row = np.where(columns < edge + far_width, _A, 0).astype(np.uint32)
    z = np.where(columns < edge, 10.0, 20.0)
    harness = _SyntheticDepthOutline(_columns(ids_row), _columns(z))
    harness.configure(_config(2), placement=PLACEMENT_INWARD)

    harness.set_visual(_A, placement=PLACEMENT_INWARD)
    silhouette = _changed_columns(harness.render())

    harness.set_visual(_A, placement=PLACEMENT_INWARD, depth_range=(0.02, 0.02))
    band = sorted(set(_changed_columns(harness.render())) - set(silhouette))
    assert band == expected


# ---------------------------------------------------------------------------
# Identity edges are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("placement", [PLACEMENT_INWARD, PLACEMENT_OUTWARD])
def test_two_objects_at_one_depth_keep_their_identity_edge(
    offscreen_renderer, placement
):
    """A range adds nothing where the keys differ and the depth does not."""
    ids = _columns(np.where(np.arange(_SIZE) < _SIZE // 2, _A, _B).astype(np.uint32))
    z = np.full((_SIZE, _SIZE), 12.0)
    frames = []
    for depth_range in (None, (0.01, 0.06)):
        harness = _SyntheticDepthOutline(ids, z)
        harness.configure(_config(2), placement=placement)
        for object_id in (_A, _B):
            harness.set_visual(object_id, placement=placement, depth_range=depth_range)
        frames.append(harness.render())
    assert len(_changed_columns(frames[0])) == 4
    np.testing.assert_array_equal(frames[0], frames[1])


@pytest.mark.parametrize("placement", [PLACEMENT_INWARD, PLACEMENT_OUTWARD])
def test_an_identity_edge_is_not_faded(offscreen_renderer, placement):
    """A silhouette over a far background of another key has full alpha.

    The jump across it is under the solid value, so a depth edge there
    would be faint; an identity edge has a strength of 1.
    """
    edge = _SIZE // 2
    ids = _columns(np.where(np.arange(_SIZE) < edge, _A, _B).astype(np.uint32))
    z = _columns(np.where(np.arange(_SIZE) < edge, 10.0, 10.3))
    harness = _SyntheticDepthOutline(ids, z)
    harness.configure(_config(2), placement=placement)
    harness.set_visual(_A, placement=placement, depth_range=(0.02, 0.12))
    frame = harness.render()
    expected = _band_columns(placement, edge)
    assert _changed_columns(frame) == expected
    assert _is_select(frame, expected)


# ---------------------------------------------------------------------------
# The strength of an edge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("placement", [PLACEMENT_INWARD, PLACEMENT_OUTWARD])
@pytest.mark.parametrize("jump", [0.015, 0.03, 0.05, 0.09, 0.12, 0.3])
def test_the_alpha_of_a_depth_line_grows_with_the_jump(
    offscreen_renderer, placement, jump
):
    """No band under the start, the line's colour from the solid value up.

    Between the two the line is blended over the fill at
    ``smoothstep(start, solid, jump)``.
    """
    depth_range = (0.02, 0.1)
    start, solid = decode_depth_range(*encode_depth_range(depth_range))
    edge = _SIZE // 2
    harness = _SyntheticDepthOutline(*_step_scene(z_near=10.0, z_far=10.0 * (1 + jump)))
    harness.configure(_config(2), placement=placement)
    harness.set_visual(_A, placement=placement, depth_range=depth_range)
    frame = harness.render()

    # The jump as the pass sees it, from the depth as stored.
    z_near, z_far = _z_of(harness.raw[_ROW, edge - 1]), _z_of(harness.raw[_ROW, edge])
    seen = (z_far - z_near) / z_near
    columns = _band_columns(placement, edge)

    if seen <= start:
        assert _changed_columns(frame) == []
        return
    assert _changed_columns(frame) == columns
    alpha = _smoothstep(start, solid, seen)
    expected = _over(_SELECT, alpha, _FILL) * 255
    for column in columns:
        np.testing.assert_allclose(frame[_ROW, column, :3], expected, atol=1.5)
    if seen >= solid:
        assert _is_select(frame, columns)
    else:
        assert not _is_select(frame, columns)


@pytest.mark.parametrize("placement", [PLACEMENT_INWARD, PLACEMENT_OUTWARD])
def test_the_strongest_of_the_taps_is_drawn(offscreen_renderer, placement):
    """A pixel between a shallow and a deep jump takes the deep one's alpha.

    A strip 2 pixels wide with a shallow jump on its left and a deep one on
    its right: at thickness 2 every pixel of it has both in reach, and the
    left tap is examined first.
    """
    depth_range = (0.02, 0.2)
    left, right = 30, 32
    columns = np.arange(_SIZE)
    if placement == PLACEMENT_INWARD:
        # The strip is near; what lies to its right is much farther.
        z = np.select([columns < left, columns < right], [10.5, 10.0], 14.0)
    else:
        # The strip is far; what lies to its right is much nearer.
        z = np.select([columns < left, columns < right], [13.5, 14.0], 10.0)
    ids = np.full((_SIZE, _SIZE), _A, dtype=np.uint32)
    harness = _SyntheticDepthOutline(ids, _columns(z))
    harness.configure(_config(2), placement=placement)
    harness.set_visual(_A, placement=placement, depth_range=depth_range)
    frame = harness.render()
    # The deep jump is over the solid value: full alpha on the whole strip.
    assert _is_select(frame, [left, left + 1])


def test_every_layer_is_faded_with_the_line(offscreen_renderer):
    """The boundaries layer and the contrast band take the edge's strength."""
    depth_range = (0.02, 0.1)
    start, solid = decode_depth_range(*encode_depth_range(depth_range))
    edge = _SIZE // 2
    harness = _SyntheticDepthOutline(*_step_scene(z_near=10.0, z_far=10.5))
    harness.configure(_config(2, boundaries=1, inner_t=2), placement=PLACEMENT_OUTWARD)
    harness.set_visual(_A, placement=PLACEMENT_OUTWARD, depth_range=depth_range)
    frame = harness.render()

    z_near, z_far = _z_of(harness.raw[_ROW, edge - 1]), _z_of(harness.raw[_ROW, edge])
    alpha = _smoothstep(start, solid, (z_far - z_near) / z_near)
    assert 0.2 < alpha < 0.8

    assert _changed_columns(frame) == [edge, edge + 1, edge + 2, edge + 3]
    line = _over(_SELECT, alpha, _FILL)
    # The boundaries layer, 1 pixel, lies under the line's first pixel.
    line_over_boundary = _over(_SELECT, alpha, _over(_BOUNDARY, alpha, _FILL))
    contrast = _over(_INNER, alpha, _FILL)
    expected = [line_over_boundary, line, contrast, contrast]
    for offset, colour in enumerate(expected):
        np.testing.assert_allclose(
            frame[_ROW, edge + offset, :3], np.asarray(colour) * 255, atol=1.5
        )


# ---------------------------------------------------------------------------
# The range is per visual
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("placement", [PLACEMENT_INWARD, PLACEMENT_OUTWARD])
def test_each_visual_is_tested_against_its_own_range(offscreen_renderer, placement):
    """The same jump is an edge for one visual and not for the other."""
    edge = _SIZE // 2
    ids = np.full((_SIZE, _SIZE), _A, dtype=np.uint32)
    ids[_SIZE // 2 :] = _B
    z = _columns(np.where(np.arange(_SIZE) < edge, 10.0, 10.3))
    harness = _SyntheticDepthOutline(ids, z)
    harness.configure(_config(2), placement=placement)
    harness.set_visual(_A, placement=placement, depth_range=(0.02, 0.02))
    harness.set_visual(_B, placement=placement, depth_range=(0.05, 0.05))
    frame = harness.render()
    assert len(_changed_columns(frame, row=8)) == 2
    assert _changed_columns(frame, row=_SIZE - 8) == []


def _same_key_scene():
    """Visual A near on the left, visual B far on the right, one label key."""
    edge = _SIZE // 2
    columns = np.arange(_SIZE)
    ids = _columns(np.where(columns < edge, _A, _B).astype(np.uint32))
    z = _columns(np.where(columns < edge, 10.0, 20.0))
    keys = np.full((_SIZE, _SIZE), 19_209_143, dtype=np.uint32)
    return ids, z, keys


@pytest.mark.parametrize(
    "placement, ranged, expected",
    [
        # Outward: the range and the style are those of the tap, the nearer.
        (PLACEMENT_OUTWARD, _A, [32, 33]),
        (PLACEMENT_OUTWARD, _B, []),
        # Inward: those of the centre, which is the nearer pixel.
        (PLACEMENT_INWARD, _A, [30, 31]),
        (PLACEMENT_INWARD, _B, []),
        (PLACEMENT_INWARD, None, []),
        (PLACEMENT_OUTWARD, None, []),
    ],
)
def test_two_label_visuals_with_one_key_get_a_depth_edge_between_them(
    offscreen_renderer, placement, ranged, expected
):
    """Two labels visuals holding the same label have no identity edge.

    The depth test is consulted between them, with the range of the nearer
    visual.  With no range on it nothing is drawn, as before.
    """
    harness = _SyntheticDepthOutline(*_same_key_scene())
    harness.configure(_config(2), placement=placement)
    for object_id in (_A, _B):
        harness.set_visual(
            object_id,
            placement=placement,
            kind=KIND_LABEL_ALL,
            depth_range=(0.02, 0.02) if object_id == ranged else None,
        )
    frame = harness.render()
    assert _changed_columns(frame) == expected
    assert _is_select(frame, expected)


# ---------------------------------------------------------------------------
# Through a canvas
# ---------------------------------------------------------------------------


def _crossing_tubes() -> np.ndarray:
    """One label: a tube along x, and a tube along y in front of it."""
    shape = (48, 64, 64)
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    data = np.zeros(shape, dtype=np.int32)
    data[(z - 12) ** 2 + (y - 32) ** 2 < 25] = 1  # far, along x
    data[(z - 36) ** 2 + (x - 32) ** 2 < 25] = 1  # near, along y
    return data


@pytest.fixture
def crossing_scene(qtbot, offscreen_renderer):
    """A 3D labels visual of two crossing tubes, outlined; one canvas."""
    from cellier.controller import CellierController
    from cellier.data import LabelMemoryStore
    from cellier.render import RenderManagerConfig
    from cellier.visuals import InMemoryLabelsAppearance, VisualOutline

    controller = CellierController(
        render_config=RenderManagerConfig(outline=_config(2))
    )
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="scene")
    visual = controller.add_labels(
        data=LabelMemoryStore(data=_crossing_tubes(), name="labels"),
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(
            colormap_mode="random", render_mode="iso_categorical"
        ),
        name="labels",
        outline=VisualOutline(slot=1, placement="outward"),
        outline_mode="all_boundaries",
    )
    controller.add_canvas(scene_id=scene.id, depth_range_3d=(1.0, 20000.0))
    # A pixel is then either the label or the line: no background gradient.
    controller.update_background_field(scene.id, "visible", False)

    async def _load() -> None:
        controller.fit_camera(scene.id)
        controller.reslice_all()
        await drain_loading(controller)

    asyncio.run(_load())
    yield controller, scene, visual
    controller.close()


def _draw_scene(controller, scene, placement, depth_range, size=(192, 192)):
    """Draw the live scene through an outline pass of its own renderer."""
    from pygfx.renderers.wgpu.engine.effectpasses import PPAAPass
    from rendercanvas.offscreen import RenderCanvas

    from cellier.render._cellier_blender import (
        OUTLINE_ID_TARGET,
        install_cellier_blender,
    )
    from cellier.render._outline import OutlinePass
    from cellier.render._pick_buffer import enable_pick_texture_binding
    from cellier.render._visual_lut import get_shared_visual_lut

    canvas_view = controller.get_canvas_view(controller.get_canvas_ids(scene.id)[0])
    camera = canvas_view.camera
    gfx_scene = controller._render_manager.get_scene(scene.id)
    # Along the data's z axis: the tubes cross at the middle of the frame.
    camera.show_object(gfx_scene, view_dir=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0))
    canvas = RenderCanvas(size=size, pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    install_cellier_blender(renderer, [OUTLINE_ID_TARGET])
    enable_pick_texture_binding(renderer)

    outline = OutlinePass(renderer, get_shared_visual_lut(), lambda: camera)
    outline.apply_config(controller.render_config.outline)
    outline.set_placements(
        has_inward=placement == "inward", has_outward=placement == "outward"
    )
    outline.set_depth_edges(depth_range is not None)
    renderer.effect_passes = (
        outline,
        *(p for p in renderer.effect_passes if not isinstance(p, PPAAPass)),
    )
    canvas.request_draw(lambda: renderer.render(gfx_scene, camera))
    return np.asarray(canvas.draw())


def _draw_mode(controller, scene, visual, placement, mode, depth_range=None):
    """Draw the labels in an outline mode, set through the controller."""
    controller.update_visual_render_field(visual.id, "outline.placement", placement)
    controller.update_visual_render_field(visual.id, "outline_mode", mode)
    if depth_range is not None:
        controller.update_visual_render_field(
            visual.id, "outline_depth_range", depth_range
        )
    sent = controller._render_manager.get_visual_outline(visual.id)[3] > 0
    return _draw_scene(controller, scene, placement, depth_range if sent else None)


@pytest.mark.parametrize("placement", ["inward", "outward"])
def test_depth_threshold_outlines_a_crossing_of_one_label(crossing_scene, placement):
    """The mode, end to end: model, controller, table, pass.

    Two tubes of one label cross.  ``all_boundaries`` draws the silhouette
    only; ``depth_threshold`` draws that and more, and the more is at the
    crossing.
    """
    controller, scene, visual = crossing_scene

    def select_pixels(mode, depth_range=None) -> np.ndarray:
        frame = _draw_mode(controller, scene, visual, placement, mode, depth_range)
        return np.all(frame[..., :3] == _rgb8(_SELECT), axis=-1)

    plain = select_pixels("all_boundaries")
    assert plain.sum() > 0
    # The start is well over a voxel's depth, which at this size is a few
    # percent of the camera's distance: the steps of the voxel surface are
    # real depth jumps too, and are not what is tested here.
    ranged = select_pixels("depth_threshold", (0.1, 0.1))
    added = ranged & ~plain
    assert added.sum() > 0
    assert not np.any(plain & ~ranged)
    # The tubes cross at the middle of the frame, and only there does one
    # part of the label pass in front of another.
    rows, cols = np.nonzero(added)
    height, width = added.shape
    assert np.all(np.abs(rows - height / 2) < height / 4)
    assert np.all(np.abs(cols - width / 2) < width / 4)

    # Back in all_boundaries the range is no longer sent.
    np.testing.assert_array_equal(select_pixels("all_boundaries"), plain)


def test_depth_threshold_on_a_render_plane_is_all_boundaries(crossing_scene):
    """Labels drawn on a plane have no jump in depth inside them.

    The plane is seen at a slant, so its depth changes from pixel to pixel
    by more than the start; it is one flat surface and gets no band.
    """
    from tests.render import _planes as planes

    controller, scene, visual = crossing_scene
    plane = planes.plane_zyx(
        planes.world_of(controller, scene), (24.0, 32.0, 32.0), (1.0, 0.6, 0.0)
    )

    async def _to_plane_mode() -> None:
        # Inside the loop: the change schedules a reslice.
        controller.set_render_planes(visual.id, [plane])
        controller.update_appearance_field(visual.id, "render_mode", "plane")
        await drain_loading(controller)

    asyncio.run(_to_plane_mode())

    expected = _draw_mode(controller, scene, visual, "inward", "all_boundaries")
    assert np.any(np.all(expected[..., :3] == _rgb8(_SELECT), axis=-1))
    frame = _draw_mode(
        controller, scene, visual, "inward", "depth_threshold", (0.001, 0.001)
    )
    np.testing.assert_array_equal(frame, expected)


def test_the_canvas_pass_follows_the_ranges_of_the_visuals(crossing_scene):
    """A canvas's pass has the canvas's camera and is told when to read depth.

    While no visual has a range the depth term is off, which is what keeps
    the frame of every existing scene as it was.
    """
    controller, scene, visual = crossing_scene
    manager = controller._render_manager
    canvas_view = controller.get_canvas_view(controller.get_canvas_ids(scene.id)[0])
    outline_pass = canvas_view._outline_pass
    assert outline_pass.USES_DEPTH is True
    assert outline_pass._get_camera() is canvas_view.camera
    assert outline_pass._has_depth_edges is False

    manager.set_visual_outline(
        visual.id, slot=1, kind=KIND_LABEL_ALL, depth_range=(0.01, 0.06)
    )
    assert outline_pass._has_depth_edges is True

    manager.set_visual_outline(visual.id, slot=1, kind=KIND_LABEL_ALL)
    assert outline_pass._has_depth_edges is False

    manager.set_visual_outline(
        visual.id, slot=1, kind=KIND_LABEL_ALL, depth_range=(0.01, 0.06)
    )
    manager.set_visual_outline(visual.id, slot=0)
    assert outline_pass._has_depth_edges is False
