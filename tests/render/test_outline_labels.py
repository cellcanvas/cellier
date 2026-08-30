"""Label outlines: the per-label key and the selection layer (Stage 7b/7c).

Covers the two demo checklist items Phase 1 could not reach:

13. two touching labels show a boundary between them;
14. selecting one label outlines exactly that label.

A labels volume is a *single* pygfx object, so `global_id` gives it one
silhouette and nothing more.  These tests are about the `outline_id`
target, which carries a per-pixel label key so the edge test can see
boundaries *inside* one visual.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pygfx as gfx
import pytest
from pygfx.renderers.wgpu.engine.effectpasses import PPAAPass
from pygfx.renderers.wgpu.engine.shared import get_shared

from cellier.render._cellier_blender import (
    OUTLINE_ID_TARGET,
    install_cellier_blender,
)
from cellier.render._config import (
    OutlineConfig,
    OutlineLayerConfig,
    RenderManagerConfig,
)
from cellier.render._outline import OutlinePass
from cellier.render._pick_buffer import enable_pick_texture_binding, get_pick_view
from cellier.render._visual_lut import KIND_LABEL, get_shared_visual_lut
from cellier.render.shaders._label_colormap import (
    OUTLINE_SELECTION_CAPACITY,
    build_outline_selection_texture,
    update_outline_selection,
)

_BOUNDARY = (0.0, 0.0, 1.0, 1.0)
_SLOT1 = (1.0, 0.0, 1.0, 1.0)
_SLOT2 = (0.0, 1.0, 1.0, 1.0)
_SIZE = 48


def _label_volume() -> np.ndarray:
    """Three discs on one Z slice; labels 1 and 2 touch."""
    _z, y, x = np.mgrid[:_SIZE, :_SIZE, :_SIZE]
    data = np.zeros((_SIZE, _SIZE, _SIZE), dtype=np.int32)
    data[(y - 16) ** 2 + (x - 16) ** 2 < 121] = 1
    data[(y - 16) ** 2 + (x - 34) ** 2 < 121] = 2
    data[(y - 34) ** 2 + (x - 24) ** 2 < 81] = 3
    return data


def _config() -> RenderManagerConfig:
    return RenderManagerConfig(
        outline=OutlineConfig(
            enabled=True,
            boundaries=OutlineLayerConfig(
                enabled=True,
                inward_thickness=1,
                outward_thickness=0,
                color=_BOUNDARY,
            ),
            selection=OutlineLayerConfig(
                enabled=True, inward_thickness=3, outward_thickness=0
            ),
            inner_thickness=0,
            palette=[_SLOT1, _SLOT2],
        )
    )


@pytest.fixture
def label_scene(qtbot, offscreen_renderer):
    """A 2D labels visual with outlines enabled; returns (controller, scene, visual)."""
    from cellier.controller import CellierController
    from cellier.data import LabelMemoryStore
    from cellier.visuals import InMemoryLabelsAppearance

    controller = CellierController(render_config=_config())
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_labels(
        data=LabelMemoryStore(data=_label_volume(), name="labels"),
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(colormap_mode="random"),
        name="labels",
    )
    controller.add_canvas(scene_id=scene.id)

    async def _load() -> None:
        # Inside the loop: mutating dims schedules asyncio reslice tasks, so
        # doing it out here raises "no current event loop".
        scene.dims.selection.slice_indices = {0: 24}
        controller.fit_camera(scene.id)
        controller.reslice_all()
        slicer = controller._render_manager._slicer
        for _ in range(20):
            tasks = list(slicer._tasks.values())
            if not tasks:
                return
            await asyncio.gather(*tasks)

    asyncio.run(_load())
    controller.fit_camera(scene.id)
    return controller, scene, visual


def _render(controller, scene, *, with_target: bool = True, size=(160, 160)):
    """Render the live scene through an outline pass; return (frame, renderer)."""
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=size, pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    # Order matters and is the subject of its own test below: installing
    # replaces the blender, so a pick grant made first would be discarded.
    if with_target:
        install_cellier_blender(renderer, [OUTLINE_ID_TARGET])
    enable_pick_texture_binding(renderer)

    outline = OutlinePass(renderer, get_shared_visual_lut())
    outline.apply_config(controller.render_config.outline)
    controller._render_manager._sync_visual_lut()
    outline.set_placements(has_inward=True, has_outward=False)
    renderer.effect_passes = (
        outline,
        *(p for p in renderer.effect_passes if not isinstance(p, PPAAPass)),
    )

    gfx_scene = controller._render_manager.get_scene(scene.id)
    camera = controller.get_canvas_view(controller.get_canvas_ids(scene.id)[0]).camera
    errors: list[BaseException] = []

    def _draw() -> None:
        try:
            renderer.render(gfx_scene, camera)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)
            raise

    canvas.request_draw(_draw)
    image = canvas.draw()
    if errors:  # pragma: no cover - failure path
        raise RuntimeError(
            f"draw failed -- {type(errors[0]).__name__}: {errors[0]}"
        ) from errors[0]
    return np.asarray(image), renderer


def _count(frame: np.ndarray, rgba, tol: int = 8) -> int:
    target = np.round(np.array(rgba[:3]) * 255)
    delta = np.abs(frame[..., :3].astype(np.int32) - target)
    return int(np.count_nonzero(np.all(delta <= tol, axis=-1)))


def _read_outline_id(renderer) -> np.ndarray:
    texture = renderer._blender.get_texture(OUTLINE_ID_TARGET)
    width, height = texture.size[:2]
    raw = get_shared().device.queue.read_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": 4 * width, "rows_per_image": height},
        (width, height, 1),
    )
    return np.frombuffer(raw, np.uint32).reshape(height, width)


# ---------------------------------------------------------------------------
# The key itself
# ---------------------------------------------------------------------------


def test_each_label_gets_its_own_key(label_scene):
    """Three labels produce three distinct keys, all in the unselected range."""
    controller, scene, visual = label_scene
    controller.set_visual_outline(visual.id, slot=1)
    _frame, renderer = _render(controller, scene)

    keys = np.unique(_read_outline_id(renderer))
    non_background = keys[keys != 0]
    assert len(non_background) == 3
    # 1..15 is reserved for selected labels; nothing is selected yet.
    assert np.all(non_background >= 16)
    assert 0 in keys  # background stays 0


def test_labels_visual_is_registered_as_a_label_kind(label_scene):
    """The LUT entry says LABEL, which is what routes the key lookup."""
    controller, _scene, visual = label_scene
    controller.set_visual_outline(visual.id, slot=1)
    entry = controller._render_manager.get_visual_outline(visual.id)
    assert entry is not None
    assert entry[2] == KIND_LABEL


# ---------------------------------------------------------------------------
# Checklist item 13 -- boundaries between touching labels
# ---------------------------------------------------------------------------


def test_touching_labels_show_a_boundary_between_them(label_scene):
    """The boundaries layer draws inside one visual, not just around it.

    This is the whole point of the label key: with `global_id` alone the
    volume is one region and only its silhouette is found.
    """
    controller, scene, visual = label_scene
    controller.set_visual_outline(visual.id, slot=1)

    with_target, _r = _render(controller, scene, with_target=True)
    without_target, _r2 = _render(controller, scene, with_target=False)

    # Both draw the visual's outer silhouette; only the keyed one finds the
    # internal boundaries, so it must paint strictly more.
    assert _count(with_target, _BOUNDARY) > _count(without_target, _BOUNDARY) > 0


def test_without_the_target_a_labels_visual_still_gets_a_silhouette(label_scene):
    """Graceful degradation: no target means whole-object outlining.

    A canvas built without outlines enabled has no `outline_id`, and a
    labels visual must still be outlined rather than vanishing.
    """
    controller, scene, visual = label_scene
    controller.set_visual_outline(visual.id, slot=1)
    frame, renderer = _render(controller, scene, with_target=False)

    assert renderer._blender.get_texture(OUTLINE_ID_TARGET) is None
    assert _count(frame, _BOUNDARY) > 0
    assert _count(frame, _SLOT1) > 0


# ---------------------------------------------------------------------------
# Checklist item 14 -- exact selection
# ---------------------------------------------------------------------------


def test_selecting_one_label_outlines_exactly_that_label(label_scene):
    """One label in slot 1; nothing else picks up a selection colour.

    The range partition guarantees this -- an unselected label hashes into
    16.. and can never land in the 1..15 selection range -- so a failure
    here means the partition is wrong, not that a hash collided.
    """
    controller, scene, visual = label_scene
    controller.set_visual_outline(visual.id, slot=1)

    before, _r = _render(controller, scene)
    assert _count(before, _SLOT1) == 0

    controller.set_label_selection(visual.id, {2: 1})
    after, _r2 = _render(controller, scene)

    assert _count(after, _SLOT1) > 0
    assert _count(after, _SLOT2) == 0


def test_two_labels_can_take_different_palette_slots(label_scene):
    """Slots index the palette, so selections can be distinguished."""
    controller, scene, visual = label_scene
    controller.set_visual_outline(visual.id, slot=1)
    controller.set_label_selection(visual.id, {2: 1, 3: 2})
    frame, _r = _render(controller, scene)

    assert _count(frame, _SLOT1) > 0
    assert _count(frame, _SLOT2) > 0


def test_clearing_the_selection_leaves_the_boundaries_layer(label_scene):
    """An empty selection is not the same as no outline at all."""
    controller, scene, visual = label_scene
    controller.set_visual_outline(visual.id, slot=1)
    controller.set_label_selection(visual.id, {2: 1})
    assert _count(_render(controller, scene)[0], _SLOT1) > 0

    controller.set_label_selection(visual.id, {})
    frame, _r = _render(controller, scene)
    assert _count(frame, _SLOT1) == 0
    assert _count(frame, _BOUNDARY) > 0


def test_selection_takes_precedence_over_boundaries(label_scene):
    """Selecting a label consumes boundary pixels rather than adding to them.

    Precedence is selection > boundaries, and the selection band here is
    wider, so the boundary count must *fall*.
    """
    controller, scene, visual = label_scene
    controller.set_visual_outline(visual.id, slot=1)
    unselected = _count(_render(controller, scene)[0], _BOUNDARY)

    controller.set_label_selection(visual.id, {2: 1})
    selected = _count(_render(controller, scene)[0], _BOUNDARY)

    assert 0 < selected < unselected


# ---------------------------------------------------------------------------
# The selection texture
# ---------------------------------------------------------------------------


def test_selection_texture_is_sorted_and_clamped():
    """The shader binary-searches, so entries must be sorted by label."""
    texture = build_outline_selection_texture()
    count = update_outline_selection(texture, {9: 3, 2: 99, 5: 0})

    assert count == 3
    labels = texture.data[0, :3, 0].tolist()
    slots = texture.data[0, :3, 1].tolist()
    assert labels == [2, 5, 9]
    # Slots are clamped into the 1..15 range the partition reserves.
    assert slots == [15, 1, 3]


def test_selection_texture_capacity_is_enforced_with_a_warning():
    """Overflow drops entries loudly rather than corrupting the search."""
    texture = build_outline_selection_texture()
    oversized = dict.fromkeys(range(OUTLINE_SELECTION_CAPACITY + 10), 1)

    with pytest.warns(UserWarning, match="capacity"):
        count = update_outline_selection(texture, oversized)

    assert count == OUTLINE_SELECTION_CAPACITY


def test_selection_update_is_in_place(label_scene):
    """Changing the selection must not swap the texture object.

    The texture is bound when the shader is built, so replacing it would
    force a pipeline rebuild on every selection change.
    """
    controller, _scene, visual = label_scene
    materials = list(
        controller._render_manager._label_materials(
            controller._render_manager._scenes[
                controller._visual_to_scene[visual.id]
            ].get_visual(visual.id)
        )
    )
    assert materials, "the labels visual exposes no material with a key texture"
    before = [m.outline_selection_texture for m in materials]

    controller.set_label_selection(visual.id, {2: 1})

    after = [m.outline_selection_texture for m in materials]
    assert all(a is b for a, b in zip(after, before))
    assert all(
        int(m.label_params_buffer.data["n_outline_entries"]) == 1 for m in materials
    )


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_installing_the_blender_preserves_the_pick_grant(offscreen_renderer):
    """Installing must not discard TEXTURE_BINDING on pick, in either order.

    ``install_cellier_blender`` replaces the whole blender, so a grant made
    beforehand lived on the object being thrown away.  The symptom was
    silent: ``get_pick_view`` returned None, the outline pass took its
    passthrough branch, and enabling outlines made them stop working
    entirely.
    """
    from rendercanvas.offscreen import RenderCanvas

    scene = gfx.Scene()
    scene.add(
        gfx.Mesh(
            gfx.sphere_geometry(1.0, 16, 8),
            gfx.MeshBasicMaterial(color="#ff8800", pick_write=True),
        )
    )
    camera = gfx.OrthographicCamera()
    camera.show_object(scene)

    for grant_first in (True, False):
        canvas = RenderCanvas(size=(32, 32), pixel_ratio=1)
        renderer = gfx.WgpuRenderer(canvas)
        if grant_first:
            assert enable_pick_texture_binding(renderer) is True
            assert install_cellier_blender(renderer, [OUTLINE_ID_TARGET]) is True
        else:
            assert install_cellier_blender(renderer, [OUTLINE_ID_TARGET]) is True
            assert enable_pick_texture_binding(renderer) is True

        canvas.request_draw(lambda r=renderer: r.render(scene, camera))
        canvas.draw()
        assert (
            get_pick_view(renderer) is not None
        ), f"pick binding lost when grant_first={grant_first}"


def test_canvas_view_keeps_both_couplings_when_outlines_are_enabled(
    qtbot, offscreen_renderer
):
    """The wiring order in CanvasView must leave both flags true."""
    from cellier.controller import CellierController

    controller = CellierController(render_config=_config())
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_canvas(scene_id=scene.id)

    canvas = next(iter(controller._render_manager._canvases.values()))
    assert canvas._outline_id_available is True
    assert canvas._outline_available is True
