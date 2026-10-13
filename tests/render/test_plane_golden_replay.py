"""Golden plans that the plane rendering work must not change.

``plans/plane_rendering_design_v3.md`` 14.1 step 0.2.  Captured once from the
shipping code (HEAD ``ec887f3b``) and replayed here: the cull-first planner
(Phase 1) and the renames (Phase 2) must leave every one of these equal.

* ``plans.npz``: for each case, the desired sets of a multiscale visual's
  plan (packed brick or tile keys, their request class, truncation counts),
  from the production path (canvas request -> ``plan``).

The golden files are written only with ``PLANE_GOLDEN_WRITE=1`` and must
never be regenerated to make a test pass: a difference is the finding.

One exception is on record: the two ``2d_*/fit::0_keys`` arrays were captured
again when the 2D tile sort became stable.  They held the same keys in the tie
order of the machine that wrote them (256 tiles at 32 distinct distances), an
order no other platform reproduced.

The ``*_cls`` arrays hold the backstop's class as 1; it is 255 now.  They
are compared through ``_recorded_class`` until they are captured again.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cellier.render.scheduling import ChunkClass
from cellier.transform import AffineTransform
from cellier.visuals import (
    ClippingPlane,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests._plane_fixtures import ANISO, ISO, open_pyramid
from tests.render.scheduling._trace import RECORDED_BACKSTOP

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "plane_golden"
WRITE = os.environ.get("PLANE_GOLDEN_WRITE") == "1"

#: ``(point, normal)`` over data ``(z, y, x)`` of ``ANISO`` (level-0 voxels).
CLIPPING = {
    "none": [],
    "axis": [((0, 0, 128), (0, 0, 1))],
    "oblique": [((32, 128, 128), (-0.6, 0.5, 1.0))],
    "along_z": [((32, 0, 0), (1, 0, 0))],
    "slab": [((0, 0, 96), (0, 0, 1)), ((0, 0, 160), (0, 0, -1))],
    "oblique_and_axis": [((32, 128, 128), (0.3, 0.5, 1.0)), ((0, 100, 0), (0, 1, 0))],
}


def _bias(appearance_cls, value: float) -> dict:
    """The appearance's bias field, whichever name it has at this commit."""
    fields = appearance_cls.model_fields
    name = "settled_lod_bias" if "settled_lod_bias" in fields else "lod_bias"
    return {name: value}


def _transform(controller, scene_id, store, voxel):
    controller._ensure_data_coordinate_systems(scene_id, store)
    data = store.data_coordinate_systems[0]
    world = controller._model.scenes[scene_id].dims.world_coordinate_system
    names = [a.name for a in data.axes]
    return AffineTransform.from_axis_map(
        data,
        world,
        {data.axis_by_name(n).id: world.axis_by_name(n).id for n in names},
        scale={data.axis_by_name(n).id: v for n, v in zip(names, voxel, strict=True)},
    )


def _add(controller, pyramid_root, spec, *, dim, labels=False, bias=1.0, planes=()):
    store, voxel = open_pyramid(pyramid_root(spec, labels=labels), spec)
    scene = controller.add_scene(
        displayed_axes=("z", "y", "x") if dim == "3d" else ("y", "x"), name="s"
    )
    transform = _transform(controller, scene.id, store, voxel)
    system = store.data_coordinate_systems[0]
    clipping = tuple(
        ClippingPlane.from_point_normal(system, p, n, axes=("z", "y", "x"))
        for p, n in planes
    )
    if labels:
        visual = controller.add_labels_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleLabelsAppearance(
                **_bias(MultiscaleLabelsAppearance, bias)
            ),
            render_config=MultiscaleLabelRenderConfig(block_size=16, **SMALL_BUDGETS),
            transform=transform,
            clipping_planes=clipping,
        )
    else:
        visual = controller.add_image_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleImageAppearance(
                **_bias(MultiscaleImageAppearance, bias)
            ),
            render_config=MultiscaleImageRenderConfig(block_size=16, **SMALL_BUDGETS),
            single=MultiscaleImageSingleAppearance(
                color_map="viridis", render_mode="mip"
            ),
            transform=transform,
            clipping_planes=clipping,
        )
    controller.add_canvas(scene_id=scene.id, canvas_size=(400, 300))
    # The Qt canvas is never shown, so it reports 1 x 1; plan for a fixed size.
    view = controller._render_manager._canvases[controller.get_canvas_ids(scene.id)[0]]
    view._canvas.get_logical_size = lambda: (400.0, 300.0)
    controller.fit_camera(scene.id)
    return scene, visual, spec


def _plan(controller, scene, visual):
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    view = controller._render_manager._canvases[canvas_id]
    selection = controller._selections_for_scene(scene.id)[canvas_id]
    request = view.capture_reslicing_request(scene.dims.to_state(), selection, None)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    desired = gfx.plan(request, controller._render_config_for(scene.id, visual))
    out = {}
    for i, ds in enumerate(desired):
        out[f"{i}_keys"] = np.asarray(ds.keys, dtype=np.int64)
        out[f"{i}_cls"] = np.asarray(ds.cls, dtype=np.int64).ravel()
        out[f"{i}_trunc"] = np.array(
            [ds.n_truncated_target, ds.n_truncated_backstop], dtype=np.int64
        )
    return out


def _centre(spec):
    size = np.array(spec.shape0[-3:], dtype=float) * np.array(spec.voxel_size[-3:])
    return (size / 2.0)[::-1]  # pygfx (x, y, z)


def _place(view, spec, which):
    camera = view.camera
    centre = _centre(spec)
    fitted = np.asarray(camera.local.position, dtype=float)
    offset = fitted - centre
    if which == "side":
        direction = np.array([-0.7, 0.45, 0.55])
        offset = direction / np.linalg.norm(direction) * np.linalg.norm(offset)
    elif which == "close":
        offset = offset * 0.35
    camera.local.position = tuple(centre + offset)
    camera.look_at(tuple(centre))


def _cases(controller, pyramid_root):
    """Yield ``(name, arrays)`` for every plan case."""
    # 3D perspective: image, clipped and not, two biases, three cameras.
    for clip, planes in CLIPPING.items():
        for bias in (1.0, 0.5):
            for which in ("fit", "side", "close"):
                scene, visual, spec = _add(
                    controller, pyramid_root, ANISO, dim="3d", bias=bias, planes=planes
                )
                view = controller._render_manager._canvases[
                    controller.get_canvas_ids(scene.id)[0]
                ]
                _place(view, spec, which)
                yield (
                    f"3d_image/{clip}/bias{bias}/{which}",
                    _plan(controller, scene, visual),
                )
                controller.remove_scene(scene.id)
    # 3D labels, with and without a clip.
    for clip in ("none", "oblique"):
        for which in ("fit", "side"):
            scene, visual, spec = _add(
                controller,
                pyramid_root,
                ANISO,
                dim="3d",
                labels=True,
                planes=CLIPPING[clip],
            )
            view = controller._render_manager._canvases[
                controller.get_canvas_ids(scene.id)[0]
            ]
            _place(view, spec, which)
            yield f"3d_labels/{clip}/{which}", _plan(controller, scene, visual)
            controller.remove_scene(scene.id)
    # An isotropic pyramid.
    scene, visual, spec = _add(controller, pyramid_root, ISO, dim="3d")
    view = controller._render_manager._canvases[controller.get_canvas_ids(scene.id)[0]]
    _place(view, spec, "side")
    yield "3d_iso/none/side", _plan(controller, scene, visual)
    controller.remove_scene(scene.id)
    # Orthographic 3D at three zooms (view height scales the camera).
    for zoom in (1.0, 2.0, 4.0):
        for clip in ("none", "oblique"):
            scene, visual, spec = _add(
                controller, pyramid_root, ANISO, dim="3d", planes=CLIPPING[clip]
            )
            view = controller._render_manager._canvases[
                controller.get_canvas_ids(scene.id)[0]
            ]
            view.camera.fov = 0.0
            controller.fit_camera(scene.id)
            view.camera.zoom = view.camera.zoom * zoom
            yield f"3d_ortho/{clip}/zoom{zoom}", _plan(controller, scene, visual)
            controller.remove_scene(scene.id)
    # 2D slices.
    for labels in (False, True):
        scene, visual, spec = _add(
            controller, pyramid_root, ANISO, dim="2d", labels=labels
        )
        yield (
            f"2d_{'labels' if labels else 'image'}/fit",
            _plan(controller, scene, visual),
        )
        controller.remove_scene(scene.id)


def _flatten(cases):
    flat = {}
    for name, arrays in cases:
        for key, value in arrays.items():
            flat[f"{name}::{key}"] = value
    return flat


def _recorded_class(cls: np.ndarray) -> np.ndarray:
    """A plan's class array as the golden recorded it.

    The request class became a priority with the backstop at 255
    (``plans/progressive_intermediate_level_design_v2.md`` 5.1); the golden
    holds it as 1.  The integer changed and no plan did, so it is translated
    back here and the golden is untouched, as ``scheduling/_trace.py`` does
    for the trace baseline.  Both translations go when that plan's Phase 7
    recaptures the class arrays.
    """
    out = np.array(cls)
    out[out == int(ChunkClass.BACKSTOP)] = RECORDED_BACKSTOP
    return out


def test_plans_equal_the_golden(controller, pyramid_root):
    flat = _flatten(_cases(controller, pyramid_root))
    path = GOLDEN / "plans.npz"
    if WRITE:
        GOLDEN.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **flat)
        pytest.skip(f"wrote {len(flat)} arrays to {path}")
    golden = np.load(path)
    assert set(golden.files) == set(flat), sorted(set(golden.files) ^ set(flat))[:10]
    flat = {k: _recorded_class(v) if k.endswith("_cls") else v for k, v in flat.items()}
    wrong = [k for k in flat if not np.array_equal(golden[k], flat[k])]
    assert not wrong, f"{len(wrong)} of {len(flat)} differ, e.g. {wrong[:5]}"
