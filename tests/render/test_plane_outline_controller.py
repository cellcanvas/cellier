"""Plane outlines on the controller: a change of an outline is not a move.

Plane outline design (``plans/plane_outline_design.md``), section 4.3.  An
outline is a field of its plane, so a new outline is a new tuple; the
controller must not take it for a new pose.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.visuals import (
    ClippingPlane,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    PlaneOutline,
    RenderPlane,
)
from tests import _plane_fixtures as fx
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading

RED = PlaneOutline(enabled=True, color=(1.0, 0.0, 0.0, 1.0), width=3.0)
SETTLE_S = 0.03


async def _image(controller, tmp_path, *, render_mode="plane"):
    """A loaded multiscale image with one render plane and one clipping plane."""
    controller._render_manager.config.scheduler.dims_settle_s = SETTLE_S
    fx.write_pyramid(tmp_path, fx.ISO)
    store, _ = fx.open_pyramid(tmp_path, fx.ISO)
    scene = controller.add_scene(dim="3d", name="scene")
    world = controller._model.scenes[scene.id].dims.world_coordinate_system
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 65535.0), render_mode=render_mode
        ),
        render_planes=(
            RenderPlane.from_point_normal(
                world, (60, 0, 0), (1, 0, 0), axes=("z", "y", "x")
            ),
        ),
        clipping_planes=(
            ClippingPlane.from_point_normal(
                store.data_coordinate_systems[0],
                (10, 0, 0),
                (1, 0, 0),
                axes=("z", "y", "x"),
            ),
        ),
    )
    controller.add_canvas(scene_id=scene.id, canvas_size=(200, 125))
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    return scene, visual


class _Spy:
    """What the controller did about one change of a visual's planes."""

    def __init__(self, controller, visual, monkeypatch) -> None:
        self.changed: list = []
        self.interactions: list = []
        self.reslices: list = []
        self.draws: list = []
        self.ticks: list = []
        owner = uuid4()
        controller.on_render_planes_changed(
            visual.id, self.changed.append, owner_id=owner
        )
        controller.on_clipping_planes_changed(
            visual.id, self.changed.append, owner_id=owner
        )
        controller.on_plane_interaction(
            visual.id, self.interactions.append, owner_id=owner
        )
        reslice = controller.reslice_visual
        monkeypatch.setattr(
            controller,
            "reslice_visual",
            lambda vid: (self.reslices.append(vid), reslice(vid))[1],
        )
        draw = controller._request_draw_for_visual
        monkeypatch.setattr(
            controller,
            "_request_draw_for_visual",
            lambda vid: (self.draws.append(vid), draw(vid))[1],
        )
        driver = controller._plane_interaction_driver
        tick = driver.tick
        monkeypatch.setattr(
            driver,
            "tick",
            lambda *args, **kwargs: (self.ticks.append(args), tick(*args, **kwargs))[1],
        )


def _planes(visual, kind):
    return visual.render_planes if kind == "render" else visual.clipping_planes


def _set(controller, visual, kind, planes) -> None:
    if kind == "render":
        controller.set_render_planes(visual.id, planes)
    else:
        controller.set_clipping_planes(visual.id, planes)


@pytest.mark.parametrize("kind", ["render", "clipping"])
async def test_a_new_outline_is_announced_and_drawn_and_reads_nothing(
    controller, tmp_path, monkeypatch, kind
):
    _scene, visual = await _image(controller, tmp_path)
    spy = _Spy(controller, visual, monkeypatch)
    plane = _planes(visual, kind)[0]

    _set(controller, visual, kind, (plane.model_copy(update={"outline": RED}),))

    assert _planes(visual, kind)[0].outline == RED
    assert _planes(visual, kind)[0].id == plane.id
    assert len(spy.changed) == 1
    announced = getattr(spy.changed[0], f"{kind}_planes")
    assert announced[0].outline == RED
    assert spy.reslices == []
    assert spy.ticks == []
    assert spy.interactions == []
    assert spy.draws == [visual.id]


@pytest.mark.parametrize("kind", ["render", "clipping"])
async def test_the_render_visual_is_given_the_outline(controller, tmp_path, kind):
    scene, visual = await _image(controller, tmp_path)
    plane = _planes(visual, kind)[0]
    _set(controller, visual, kind, (plane.model_copy(update={"outline": RED}),))
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    assert getattr(gfx_visual, f"{kind}_planes")[0].outline == RED


@pytest.mark.parametrize("kind", ["render", "clipping"])
async def test_a_move_still_reads_again(controller, tmp_path, monkeypatch, kind):
    """The other half of the rule: a pose change is treated as before."""
    _scene, visual = await _image(controller, tmp_path)
    spy = _Spy(controller, visual, monkeypatch)
    plane = _planes(visual, kind)[0]
    if kind == "render":
        moved = plane.model_copy(update={"origin": (70.0, 0.0, 0.0)})
    else:
        moved = plane.model_copy(
            update={
                "plane": plane.plane.model_copy(
                    update={"offset": plane.plane.offset + 10.0}
                )
            }
        )

    _set(controller, visual, kind, (moved,))

    assert spy.reslices == [visual.id]
    assert len(spy.ticks) == 1
    await drain_loading(controller)


@pytest.mark.parametrize("kind", ["render", "clipping"])
async def test_a_move_with_a_new_outline_is_a_move(
    controller, tmp_path, monkeypatch, kind
):
    _scene, visual = await _image(controller, tmp_path)
    spy = _Spy(controller, visual, monkeypatch)
    plane = _planes(visual, kind)[0]
    _set(
        controller,
        visual,
        kind,
        (plane.model_copy(update={"outline": RED, "enabled": False}),),
    )
    assert spy.reslices == [visual.id]
    assert len(spy.ticks) == 1
    await drain_loading(controller)


@pytest.mark.parametrize("kind", ["render", "clipping"])
async def test_a_new_outline_inside_a_drag_is_not_a_tick_of_it(
    controller, tmp_path, monkeypatch, kind
):
    """A drag's scope is open (a gizmo is held): restyling does not extend
    the drag or plan."""
    _scene, visual = await _image(controller, tmp_path)
    spy = _Spy(controller, visual, monkeypatch)
    plane = _planes(visual, kind)[0]
    with controller.plane_interaction(visual.id):
        _set(controller, visual, kind, (plane.model_copy(update={"outline": RED}),))
        assert spy.ticks == []
        assert spy.reslices == []
    await drain_loading(controller)


async def test_a_new_outline_on_a_plane_that_is_not_drawn_is_only_announced(
    controller, tmp_path, monkeypatch
):
    """Outside plane mode the render planes are stored and nothing more."""
    _scene, visual = await _image(controller, tmp_path, render_mode="mip")
    spy = _Spy(controller, visual, monkeypatch)
    plane = visual.render_planes[0]
    controller.set_render_planes(
        visual.id, (plane.model_copy(update={"outline": RED}),)
    )
    assert len(spy.changed) == 1
    assert spy.reslices == []
    assert spy.draws == []


async def test_a_direct_assignment_follows_the_same_rule(
    controller, tmp_path, monkeypatch
):
    _scene, visual = await _image(controller, tmp_path)
    spy = _Spy(controller, visual, monkeypatch)
    visual.render_planes = (
        visual.render_planes[0].model_copy(update={"outline": RED}),
    )
    visual.clipping_planes = (
        visual.clipping_planes[0].model_copy(update={"outline": RED}),
    )
    assert len(spy.changed) == 2
    assert spy.reslices == []
    assert spy.ticks == []
