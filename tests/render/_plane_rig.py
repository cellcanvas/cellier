"""A multiscale visual in ``"plane"`` render mode, for the Phase 7 tests.

Small pyramids whose levels hold a known pattern, a rig that places the
camera, plans, loads and draws, and the level-aware comparison with the CPU
reference of ``tests/render/_planes.py``.
"""

from __future__ import annotations

import numpy as np
import tensorstore as ts

from cellier.render.scheduling import ChunkClass, Tier
from cellier.visuals import (
    ClippingPlane,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests import _plane_fixtures as fx
from tests._gpu_budget import SMALL_BUDGETS
from tests.render import _planes as h
from tests.render.conftest import drain_loading

SIZE = (320, 320)
BLOCK = 16

#: Small pyramids whose every level is filled by ``_level_values``.
ISO = fx.PyramidSpec(
    shape0=(48, 64, 80),
    factors=fx.halving(3, (1, 1, 1)),
    voxel_size=(1.0, 1.0, 1.0),
    dtype="float32",
)
ANISO = fx.PyramidSpec(
    shape0=(40, 96, 96),
    factors=fx.halving(3, (0, 1, 1)),
    voxel_size=(2.5, 1.0, 1.0),
    dtype="float32",
)
NONPOW2 = fx.PyramidSpec(
    shape0=(49, 76, 66),
    factors=(
        (1.0, 1.0, 1.0),
        (49 / 25, 76 / 38, 66 / 33),
        (49 / 13, 76 / 19, 66 / 17),
    ),
    voxel_size=(1.0, 1.0, 1.0),
    dtype="float32",
)
#: Levels placed with no translation: a level's bricks then sit most of a
#: level voxel off the lookup table's cells, so the brick a place is drawn
#: from is not always the one whose own box holds it.
UNALIGNED = fx.PyramidSpec(
    shape0=(48, 64, 80),
    factors=fx.halving(3, (1, 1, 1)),
    voxel_size=(1.0, 1.0, 1.0),
    dtype="float32",
    translations=((0.0, 0.0, 0.0),) * 3,
)
SPECS = {"iso": ISO, "aniso": ANISO, "nonpow2": NONPOW2, "unaligned": UNALIGNED}

#: ``name -> normal`` over world ``(z, y, x)``.
POSES = {
    "xy": (1.0, 0.0, 0.0),
    "xz": (0.0, 1.0, 0.0),
    "oblique": (0.6, -0.5, 1.0),
}


def level_values(spec: fx.PyramidSpec, level: int) -> np.ndarray:
    """The voxel pattern of ``_planes.voxel_values`` on level *level*'s grid."""
    return h.voxel_values(spec.level_shape(level))


def write_levels(
    root, spec: fx.PyramidSpec, *, labels: bool = False, fill=None
) -> None:
    """Write *spec* with each level holding ``fill(level)`` (default the pattern)."""
    for level in range(spec.n_levels):
        shape = spec.level_shape(level)
        values = level_values(spec, level) if fill is None else fill(level)
        if labels:
            data = values.astype(np.int32)
        else:
            data = ((values + 1) / (h.N_VALUES + 1)).astype(np.float32)
        array = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(root / f"s{level}")},
                "metadata": {
                    "shape": list(shape),
                    "data_type": "int32" if labels else "float32",
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [min(s, 32) for s in shape]},
                    },
                },
                "create": True,
                "delete_existing": True,
            }
        ).result()
        array[...].write(data).result()


def centre_of(spec: fx.PyramidSpec) -> np.ndarray:
    """A point near the data's centre, world ``(z, y, x)``, off every voxel face
    of every level."""
    return ((np.array(spec.shape0) - 1) / 2 + 0.3) * np.array(spec.voxel_size)


class Rig:
    """One multiscale visual in plane mode, a camera and an offscreen frame."""

    def __init__(self, controller, scene, visual, spec, planes, labels, size=SIZE):
        self.size = size
        self.controller = controller
        self.scene = scene
        self.visual = visual
        self.spec = spec
        self.planes = tuple(planes)
        self.labels = labels
        self.gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
        canvas_id = controller.get_canvas_ids(scene.id)[0]
        self.view = controller._render_manager._canvases[canvas_id]
        # The Qt canvas is never shown, so it reports 1 x 1; plan for the
        # size the frames are drawn at.
        self.view._canvas.get_logical_size = lambda: (float(size[0]), float(size[1]))

    def shoot(self, **kwargs) -> h.Shot:
        return h.shoot(self.controller, self.scene, size=self.size, **kwargs)

    async def settle(self, **camera) -> h.Shot:
        """Place the camera, plan for it, load, and draw."""
        # A frame first: the planner culls to the camera's frustum, which a
        # draw updates.
        self.shoot(**camera)
        self.controller.reslice_all()
        await drain_loading(self.controller)
        return self.shoot(**camera)

    def set_planes(self, planes) -> None:
        self.planes = tuple(planes)
        self.controller.set_render_planes(self.visual.id, self.planes)

    def trace(self, shot: h.Shot, **kwargs) -> h.Reference:
        return h.trace(
            shot,
            self.planes,
            self.spec.shape0,
            scale_zyx=self.spec.voxel_size,
            **kwargs,
        )

    def plan(self, **camera):
        """Plan for the camera as placed, through the production path.

        Draws a frame first (the planner culls to the camera's frustum, which
        a draw updates), then builds the canvas's request and calls the
        visual's planner.  Nothing is loaded.

        Returns
        -------
        desired : list[DesiredSet]
            One per drawn channel.
        request : ReslicingRequest
        """
        self.shoot(**camera)
        canvas_id = self.controller.get_canvas_ids(self.scene.id)[0]
        selection = self.controller._selections_for_scene(self.scene.id)[canvas_id]
        request = self.view.capture_reslicing_request(
            self.scene.dims.to_state(), selection, None
        )
        config = self.controller._render_config_for(self.scene.id, self.visual)
        return self.gfx.plan(request, config), request

    @property
    def planner(self):
        """The object whose ``_last_plane_plan`` and geometry the plan used."""
        return self.gfx if self.labels else self.gfx.slots[0]

    def brick_centres_world(self, rows: np.ndarray, block: int = BLOCK) -> np.ndarray:
        """World ``(x, y, z)`` centres of ``[level, gz, gy, gx]`` rows."""
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
        factors = np.array([self.spec.factors[int(k) - 1] for k in rows[:, 0]])
        shifts = np.array([self.spec.level_translation(int(k) - 1) for k in rows[:, 0]])
        level_centre = (rows[:, 1:] + 0.5) * block - 0.5
        data = level_centre * factors + shifts
        return (data * np.asarray(self.spec.voxel_size))[:, ::-1]

    def planned(self) -> np.ndarray:
        """``[level, gz, gy, gx]`` rows of the target the scheduler holds."""
        from cellier.render.block_cache._image_residency import unpack_keys

        residency = (
            self.gfx.residency_3d() if self.labels else self.gfx.slots[0].residency_3d()
        )
        registry = self.controller._render_manager.scheduler.core.registry(
            residency.cache_id
        )
        keep = (registry.tier == Tier.VISIBLE) & (registry.cls == ChunkClass.TARGET)
        levels, _sids, grids = unpack_keys(registry.key[keep])
        return np.column_stack([levels, grids]).astype(np.int64)


async def make_rig(
    controller,
    tmp_path,
    spec: fx.PyramidSpec,
    make_planes,
    *,
    labels: bool = False,
    fill=None,
    clipping=(),
    appearance=None,
    single=None,
    budget=None,
    fov: float | None = None,
    size: tuple[int, int] = SIZE,
    block_size: int = BLOCK,
) -> Rig:
    root = tmp_path / "pyramid"
    root.mkdir(parents=True, exist_ok=True)
    write_levels(root, spec, labels=labels, fill=fill)
    store, voxel = fx.open_pyramid(root, spec)
    scene = controller.add_scene(dim="3d", name="scene")
    world = h.world_of(controller, scene)
    planes = tuple(make_planes(world, centre_of(spec)))
    transform = h.scale_translation_transform(
        controller, scene.id, store, voxel, (0.0, 0.0, 0.0)
    )
    system = store.data_coordinate_systems[0]
    clip = tuple(
        ClippingPlane.from_point_normal(system, p, n, axes=("z", "y", "x"))
        for p, n in clipping
    )
    budgets = dict(SMALL_BUDGETS)
    if budget is not None:
        budgets["gpu_budget_bytes"] = budget
    if labels:
        visual = controller.add_labels_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleLabelsAppearance(
                **{
                    "render_mode": "plane",
                    "colormap_mode": "direct",
                    "color_dict": h.label_colours(),
                }
                | (appearance or {})
            ),
            render_config=MultiscaleLabelRenderConfig(block_size=block_size, **budgets),
            transform=transform,
            render_planes=planes,
            clipping_planes=clip,
        )
    else:
        visual = controller.add_image_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleImageAppearance(**(appearance or {})),
            render_config=MultiscaleImageRenderConfig(block_size=block_size, **budgets),
            single=MultiscaleImageSingleAppearance(
                **{"render_mode": "plane", "color_map": "gray", "clim": (0.0, 1.0)}
                | (single or {})
            ),
            transform=transform,
            render_planes=planes,
            clipping_planes=clip,
        )
    controller.add_canvas(scene_id=scene.id, canvas_size=size)
    if fov is not None:
        canvas_id = controller.get_canvas_ids(scene.id)[0]
        controller._render_manager._canvases[canvas_id].camera.fov = fov
    return Rig(controller, scene, visual, spec, planes, labels, size=size)


def level_voxel(spec: fx.PyramidSpec, level: int, data_position: np.ndarray):
    """The level-*level* voxel ``(z, y, x)`` a level-0 position falls in.

    Parameters
    ----------
    data_position : np.ndarray
        ``(..., 3)`` centred level-0 positions, ``(x, y, z)``.

    Returns
    -------
    voxel : np.ndarray
        ``(..., 3)`` integer ``(z, y, x)``, clamped to the level's shape.
    clear : np.ndarray
        ``(...)`` bool: the position is clear of every face of that voxel,
        and inside the part of the volume the level has voxels for.  (A
        level placed with no translation ends short of the level-0 extent;
        past its last voxel there is nothing to draw but fill.)
    """
    factor = np.asarray(spec.factors[level], dtype=np.float64)[::-1]
    shift = np.asarray(spec.level_translation(level), dtype=np.float64)[::-1]
    u = (data_position - shift) / factor
    shape = np.asarray(spec.level_shape(level))[::-1]
    index = np.floor(u + 0.5)
    covered = ((index >= 0) & (index <= shape - 1)).all(axis=-1)
    voxel = np.clip(index, 0, shape - 1).astype(np.int64)
    face = np.abs((u + 0.5) - np.round(u + 0.5)).min(axis=-1)
    return voxel[..., ::-1], covered & (face > h.MARGIN / 2)


def compare_level(rig: Rig, shot, reference, level: int, values, **kwargs) -> dict:
    """``_planes.compare_greys`` against level *level*'s voxel grid."""
    voxel, clear = level_voxel(rig.spec, level, reference.data_position)
    at_level = h.Reference(
        hit=reference.hit,
        voxel=voxel,
        plane=reference.plane,
        position=reference.position,
        data_position=reference.data_position,
        sure=reference.sure & clear,
    )
    return h.compare_greys(shot, at_level, values, **kwargs)


def assert_matches(counts: dict, least: int = 2000) -> None:
    assert counts["sure_hit"] > least, counts
    assert counts["missing"] == 0, counts
    assert counts["extra"] == 0, counts
    assert counts["wrong"] == 0, counts


def target_rows(desired) -> np.ndarray:
    """``[level, gz, gy, gx]`` rows of a desired set's target, in plan order."""
    from cellier.render.block_cache._image_residency import unpack_keys

    keys = np.asarray(desired.keys)[
        np.asarray(desired.cls).ravel() == int(ChunkClass.TARGET)
    ]
    levels, _sids, grids = unpack_keys(keys)
    return np.column_stack([levels, grids]).astype(np.int64)
