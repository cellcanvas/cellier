"""Shared helpers for the plane rendering pixel tests.

A visual in ``"plane"`` render mode is drawn through an offscreen renderer
and compared, pixel by pixel, with a CPU reference built here from nothing
but the camera's matrices, the planes and the data: the ray through each
pixel centre is met with every plane and the voxel at the nearest hit is
looked up.  The reference shares no code with the render layer.

A pixel whose ray passes within a small margin of any edge the picture
depends on (the data box, a plane's extent, a clipping plane, a voxel face,
the line where two planes cross) is "unsure": float32 on the GPU and float64
here may fall on different sides of it.  Tests count wrong pixels among the
sure ones, and check that the sure ones are most of the picture.

Coordinates here are pygfx ``(x, y, z)`` unless a name says ``zyx``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

from cellier.transform import AffineTransform
from cellier.visuals import RenderPlane
from tests.render._pick import PickFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Grey levels a voxel value is drawn as: ``(value + 1) / (N_VALUES + 1)``.
N_VALUES = 9
#: How far (in level-0 voxels) a ray must stay from an edge to be sure.
MARGIN = 0.06
#: A colour channel differing by more than this is another grey level.
COLOUR_TOL = 9
#: The same for a label colour, after its display encoding.
LINEAR_COLOUR_TOL = 4


def voxel_values(shape: Sequence[int]) -> np.ndarray:
    """An integer in ``[0, N_VALUES)`` per voxel, different from all 26 neighbours."""
    z, y, x = np.indices(tuple(shape), dtype=np.int64)
    return (z + 2 * y + 4 * x) % N_VALUES


def grey_of(values: np.ndarray) -> np.ndarray:
    """The 8-bit grey a voxel value is drawn as by a grey colormap on ``[0, 1]``."""
    return np.round(255.0 * (values + 1) / (N_VALUES + 1))


def image_data(shape: Sequence[int]) -> np.ndarray:
    """float32 image whose every voxel differs from its neighbours."""
    return ((voxel_values(shape) + 1) / (N_VALUES + 1)).astype(np.float32)


def label_data(shape: Sequence[int]) -> np.ndarray:
    """int32 labels ``1 .. N_VALUES - 1`` with background (0) scattered through."""
    return voxel_values(shape).astype(np.int32)


def label_colours() -> dict[int, tuple[float, float, float, float]]:
    """A grey per label, so a pixel's colour names its label."""
    return {
        label: (g, g, g, 1.0)
        for label in range(1, N_VALUES)
        for g in [(label + 1) / (N_VALUES + 1)]
    }


def scale_translation_transform(controller, scene_id, store, scale_zyx, shift_zyx):
    """``data -> world`` scaling and shifting each of ``z, y, x``."""
    controller._ensure_data_coordinate_systems(scene_id, store)
    data = store.data_coordinate_systems[0]
    world = controller._model.scenes[scene_id].dims.world_coordinate_system
    names = ("z", "y", "x")
    return AffineTransform.from_axis_map(
        data,
        world,
        {data.axis_by_name(n).id: world.axis_by_name(n).id for n in names},
        scale={
            data.axis_by_name(n).id: float(s)
            for n, s in zip(names, scale_zyx, strict=True)
        },
        translation={
            data.axis_by_name(n).id: float(t)
            for n, t in zip(names, shift_zyx, strict=True)
        },
    )


def world_of(controller, scene):
    return controller._model.scenes[scene.id].dims.world_coordinate_system


def plane_zyx(world, point, normal, **kwargs) -> RenderPlane:
    """A render plane from a world ``(z, y, x)`` point and normal."""
    return RenderPlane.from_point_normal(
        world, point, normal, axes=("z", "y", "x"), **kwargs
    )


@dataclass
class Shot:
    """One offscreen frame and what is needed to trace its pixels."""

    frame: np.ndarray
    renderer: gfx.WgpuRenderer
    camera: gfx.Camera
    #: World position of each pixel centre on the near plane, and the vector
    #: from there to the far plane: ``(H, W, 3)`` each.
    ray_origin: np.ndarray
    ray_vector: np.ndarray
    canvas: object = field(repr=False, default=None)
    _picks: PickFrame | None = field(repr=False, default=None)

    @property
    def drawn(self) -> np.ndarray:
        return self.frame[..., 3] > 0

    def pick_info(self, row: int, col: int) -> dict:
        """What ``renderer.get_pick_info`` gives at a pixel, less ``"rgba"``.

        The pick target is read back once, at the first pick, and every pick
        after that is decoded from the copy: a round trip to the GPU per
        pixel made the pick tests the slowest on a real GPU (see
        ``tests/render/_pick.py``).  A shot is of one frame; drawing again
        with its renderer does not update the copy.
        """
        if self._picks is None:
            self._picks = PickFrame(self.renderer)
        return self._picks.info((col + 0.5, row + 0.5))


def shoot(
    controller,
    scene,
    *,
    size: tuple[int, int] = (320, 320),
    view_dir=(-0.5, -0.4, -1.0),
    up=(0.0, 1.0, 0.0),
    fov: float | None = None,
    distance_factor: float = 1.0,
    blender_targets: Sequence[str] = (),
) -> Shot:
    """Draw *scene* from *view_dir* with no anti-aliasing; return the frame.

    The background is hidden, so a pixel with alpha above zero is one a
    visual drew.  *distance_factor* moves the camera along its view line
    after the fit: 2.0 is twice as far from the scene's centre.
    """
    from rendercanvas.offscreen import RenderCanvas

    from cellier.render._cellier_blender import install_cellier_blender

    controller.update_background_field(scene.id, "visible", False)
    canvas_view = controller._render_manager._canvases[
        controller.get_canvas_ids(scene.id)[0]
    ]
    gfx_scene = canvas_view._get_scene_fn(scene.id)
    camera = canvas_view.camera
    if fov is not None:
        camera.fov = fov
    camera.show_object(gfx_scene, view_dir=view_dir, up=up)
    if distance_factor != 1.0:
        centre = np.asarray(gfx_scene.get_world_bounding_sphere()[:3], dtype=float)
        position = np.asarray(camera.world.position, dtype=float)
        camera.world.position = tuple(centre + (position - centre) * distance_factor)

    canvas = RenderCanvas(size=size, pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    if blender_targets:
        assert install_cellier_blender(renderer, list(blender_targets)) is True
    errors: list[BaseException] = []

    def _draw() -> None:
        try:
            renderer.render(gfx_scene, camera)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)
            raise

    canvas.request_draw(_draw)
    frame = canvas.draw()
    if errors:  # pragma: no cover - failure path
        raise RuntimeError(
            f"offscreen draw failed -- {type(errors[0]).__name__}: {errors[0]}"
        ) from errors[0]
    frame = np.asarray(frame)

    width, height = size
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    ndc_x = (cols + 0.5) / width * 2.0 - 1.0
    ndc_y = 1.0 - (rows + 0.5) / height * 2.0
    inverse = np.linalg.inv(np.asarray(camera.camera_matrix, dtype=np.float64))

    def unproject(depth: float) -> np.ndarray:
        ndc = np.stack(
            [ndc_x, ndc_y, np.full_like(ndc_x, depth), np.ones_like(ndc_x)], axis=-1
        )
        world = ndc @ inverse.T
        return world[..., :3] / world[..., 3:4]

    near = unproject(0.0)
    far = unproject(1.0)
    return Shot(
        frame=frame,
        renderer=renderer,
        camera=camera,
        ray_origin=near,
        ray_vector=far - near,
        canvas=canvas,
    )


@dataclass
class Reference:
    """What the CPU says each pixel shows."""

    #: The pixel shows a plane.
    hit: np.ndarray
    #: The voxel shown, data ``(z, y, x)``; meaningless where not ``hit``.
    voxel: np.ndarray
    #: Which plane, ``-1`` where not ``hit``.
    plane: np.ndarray
    #: World position of the hit.
    position: np.ndarray
    #: Level-0 data position of the hit, ``(x, y, z)``, continuous.
    data_position: np.ndarray
    #: The pixel is clear of every edge.
    sure: np.ndarray


def trace(
    shot: Shot,
    planes: Sequence[RenderPlane],
    shape_zyx: Sequence[int],
    *,
    scale_zyx=(1.0, 1.0, 1.0),
    shift_zyx=(0.0, 0.0, 0.0),
    clipping: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
    background: np.ndarray | None = None,
    margin: float = MARGIN,
) -> Reference:
    """Trace every pixel of *shot* against *planes*.

    Parameters
    ----------
    shot : Shot
        The frame.
    planes : Sequence[RenderPlane]
        Planes on world ``("z", "y", "x")``; disabled ones are skipped.
    shape_zyx : Sequence[int]
        Level-0 data shape.
    scale_zyx, shift_zyx : Sequence[float]
        The visual's transform: ``world = scale * data + shift``.
    clipping : sequence of ``(point, normal)``
        Clipping planes in data ``(z, y, x)``; the normal's side is kept.
    background : np.ndarray or None
        Boolean per voxel, data ``(z, y, x)``: voxels a plane does not draw
        (label background), so the next plane along the ray shows instead.
    margin : float
        The unsure band, in level-0 voxels.
    """
    scale = np.asarray(scale_zyx, dtype=np.float64)[::-1]
    shift = np.asarray(shift_zyx, dtype=np.float64)[::-1]
    size = np.asarray(shape_zyx, dtype=np.float64)[::-1]
    origin, vector = shot.ray_origin, shot.ray_vector
    shape = origin.shape[:2]
    # One voxel as a fraction of the ray, to compare distances along it.
    voxel_world = float(scale.min())

    best_t = np.full(shape, np.inf)
    hit = np.zeros(shape, dtype=bool)
    plane_index = np.full(shape, -1, dtype=np.int64)
    sure = np.ones(shape, dtype=bool)
    hit_ts = []
    for index, plane in enumerate(planes):
        if not plane.enabled:
            continue
        o = np.asarray(plane.origin, dtype=np.float64)[::-1]
        a0 = np.asarray(plane.in_plane_axis_0, dtype=np.float64)[::-1]
        a1 = np.asarray(plane.in_plane_axis_1, dtype=np.float64)[::-1]
        normal = np.cross(a0, a1)
        denom = vector @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((o - origin) @ normal) / denom
        ok = np.isfinite(t) & (t >= 0.0) & (t <= 1.0)
        t = np.where(ok, t, 0.0)
        world = origin + t[..., None] * vector
        data = (world - shift) / scale
        # Distances (in voxels, positive inside) to every edge of the plane.
        edges = [data + 0.5, size - 0.5 - data]
        distance = np.minimum(edges[0], edges[1]).min(axis=-1)
        rel = world - o
        for axis, (low, high) in ((a0, plane.extent_0), (a1, plane.extent_1)):
            along = rel @ axis
            if low is not None:
                distance = np.minimum(distance, (along - low) / voxel_world)
            if high is not None:
                distance = np.minimum(distance, (high - along) / voxel_world)
        for point, clip_normal in clipping:
            n = np.asarray(clip_normal, dtype=np.float64)[::-1]
            p = np.asarray(point, dtype=np.float64)[::-1]
            distance = np.minimum(distance, ((data - p) @ n) / np.linalg.norm(n))
        inside = ok & (distance > 0.0)
        sure &= ~ok | (np.abs(distance) > margin)
        voxel = np.clip(np.floor(data + 0.5), 0, size - 1).astype(np.int64)
        face = np.abs((data + 0.5) - np.round(data + 0.5)).min(axis=-1)
        if background is not None:
            is_background = background[voxel[..., 2], voxel[..., 1], voxel[..., 0]]
            # A plane is passed over on background, so which side of a
            # voxel face the hit is on matters to every plane, not only
            # the one that is shown.
            sure &= ~inside | (face > margin / 2)
            inside &= ~is_background
        hit_ts.append(np.where(inside, t, np.inf))
        nearer = inside & (t < best_t)
        best_t = np.where(nearer, t, best_t)
        plane_index = np.where(nearer, index, plane_index)
        hit |= inside

    # Two planes at nearly the same distance: the line where they cross.
    ray_length = np.linalg.norm(vector, axis=-1)
    for t in hit_ts:
        other = np.isfinite(t) & np.isfinite(best_t) & (t != best_t)
        gap = np.zeros(shape)
        gap[other] = np.abs(t[other] - best_t[other]) * ray_length[other] / voxel_world
        sure &= ~other | (gap > margin)

    t = np.where(hit, best_t, 0.0)
    position = origin + t[..., None] * vector
    data = (position - shift) / scale
    voxel_xyz = np.clip(np.floor(data + 0.5), 0, size - 1).astype(np.int64)
    face = np.abs((data + 0.5) - np.round(data + 0.5)).min(axis=-1)
    sure &= ~hit | (face > margin / 2)
    return Reference(
        hit=hit,
        voxel=voxel_xyz[..., ::-1],
        plane=plane_index,
        position=position,
        data_position=data,
        sure=sure,
    )


def srgb_encode(grey: np.ndarray) -> np.ndarray:
    """8-bit linear grey to the 8-bit sRGB value a display is sent."""
    linear = np.asarray(grey, dtype=np.float64) / 255.0
    encoded = np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )
    return np.round(255.0 * encoded)


def compare_greys(
    shot: Shot,
    reference: Reference,
    values: np.ndarray,
    *,
    linear_colours: bool = False,
    channel: int = 0,
) -> dict:
    """Count the sure pixels that disagree with the reference.

    Parameters
    ----------
    shot, reference :
        The frame and its trace.
    values : np.ndarray
        The voxel values (``voxel_values`` of the level-0 shape, or a label
        array), data ``(z, y, x)``.
    linear_colours : bool
        ``True`` for labels: a label colour is taken as linear light and
        encoded for the display, where an image colormap is sRGB already.
    channel : int
        The colour channel of the frame the grey is read from.

    Returns
    -------
    dict
        ``sure_hit`` and ``sure_empty`` (pixels compared), ``missing`` (the
        reference shows a plane, nothing was drawn), ``extra`` (the
        reverse) and ``wrong`` (drawn, another grey).
    """
    drawn = shot.drawn
    hit, sure = reference.hit, reference.sure
    voxel = reference.voxel
    expected = grey_of(values[voxel[..., 0], voxel[..., 1], voxel[..., 2]])
    tolerance = COLOUR_TOL
    if linear_colours:
        expected = srgb_encode(expected)
        # The encoding crowds the bright greys: neighbours are 12 apart.
        tolerance = LINEAR_COLOUR_TOL
    red = shot.frame[..., channel].astype(np.float64)
    both = sure & hit & drawn
    return {
        "sure_hit": int((sure & hit).sum()),
        "sure_empty": int((sure & ~hit).sum()),
        "missing": int((sure & hit & ~drawn).sum()),
        "extra": int((sure & ~hit & drawn).sum()),
        "wrong": int((both & (np.abs(red - expected) > tolerance)).sum()),
    }


def pick_voxel(shot: Shot, gfx_visual, row: int, col: int):
    """The data voxel ``(z, y, x)`` a pick at a pixel names, or ``None``."""
    info = shot.pick_info(row, col)
    coordinate = gfx_visual.pick_data_coordinate(info.get("world_object"), info)
    if coordinate is None:
        return None
    # pygfx (x, y, z) reverses onto data (z, y, x); [i, i + 1) convention.
    return tuple(int(np.floor(v)) for v in tuple(coordinate)[::-1])


def pick_coordinate(shot: Shot, gfx_visual, row: int, col: int):
    """The continuous level-0 data position ``(x, y, z)`` of a pick.

    In the centre convention (voxel ``i`` spans ``[i - 0.5, i + 0.5)``).
    """
    info = shot.pick_info(row, col)
    coordinate = gfx_visual.pick_data_coordinate(info.get("world_object"), info)
    if coordinate is None:
        return None
    return np.asarray(coordinate, dtype=np.float64) - 0.5


def read_target(shot: Shot, name: str, dtype, channels: int) -> np.ndarray:
    """Read an extra render target of *shot*'s blender as ``(H, W, channels)``."""
    from pygfx.renderers.wgpu.engine.shared import get_shared

    texture = shot.renderer._blender.get_texture(name)
    width, height = texture.size[:2]
    itemsize = np.dtype(dtype).itemsize * channels
    raw = get_shared().device.queue.read_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": itemsize * width, "rows_per_image": height},
        (width, height, 1),
    )
    return np.frombuffer(raw, dtype).reshape(height, width, channels)
