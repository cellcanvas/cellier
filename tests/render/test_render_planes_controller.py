"""Render planes on the controller: checks, API, events, planning.

Plane rendering design v3, sections 4, 7.4 and 8.  What a plane draws is
tested in ``test_plane_rendering_memory.py`` and
``test_plane_rendering_multiscale.py``.
"""

from __future__ import annotations

import warnings
from uuid import uuid4

import numpy as np
import pytest
from psygnal import EmitLoopError

from cellier.data import ImageMemoryStore, LabelMemoryStore, PointsMemoryStore
from cellier.events import (
    ChannelAppearanceChangedEvent,
    PlaneInteractionEvent,
    RenderPlanesChangedEvent,
    RenderPlanesUpdateEvent,
)
from cellier.scene import spatial_axes
from cellier.visuals import (
    InMemoryImageChannelAppearance,
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    RenderPlane,
)
from tests import _plane_fixtures as fx
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading

SETTLE_S = 0.03
#: A refused direct assignment: the handler's ``ValueError``, which psygnal
#: wraps in ``EmitLoopError`` on its way out of the model.
REFUSED = (ValueError, EmitLoopError)


def _world(controller, scene):
    return controller._model.scenes[scene.id].dims.world_coordinate_system


def _plane(world, z: float = 10.0, **kwargs) -> RenderPlane:
    return RenderPlane.from_point_normal(
        world, (z, 0, 0), (1, 0, 0), axes=("z", "y", "x"), **kwargs
    )


def _record_plans(monkeypatch) -> list[tuple]:
    """Record ``(visual_id, plan mode, sliced)`` for every visual a pass plans."""
    from cellier.render.scene_manager import SceneManager

    plans: list[tuple] = []
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        for visual_id, cfg in visual_configs.items():
            targets = request.target_visual_ids
            if targets is None or visual_id in targets:
                plans.append((visual_id, cfg.plan_mode.name, cfg.slicing_enabled))
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    return plans


def _scene(controller, spec=fx.ISO, dim="3d"):
    if "t" in spec.axes:
        return controller.add_scene(
            coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
            dim=dim,
            name="scene",
        )
    return controller.add_scene(dim=dim, name="scene")


def _multiscale_image(controller, tmp_path, *, spec=fx.ISO, dim="3d", **kwargs):
    """A multiscale image in ``"plane"`` mode; returns ``(scene, visual)``."""
    controller._render_manager.config.scheduler.dims_settle_s = SETTLE_S
    fx.write_pyramid(tmp_path, spec)
    store, _ = fx.open_pyramid(tmp_path, spec)
    scene = _scene(controller, spec, dim)
    kwargs.setdefault(
        "single",
        MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 65535.0), render_mode="plane"
        ),
    )
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        **kwargs,
    )
    controller.add_canvas(scene_id=scene.id, canvas_size=(200, 125))
    return scene, visual


def _memory_image(controller, *, channels=None, **kwargs):
    """An in-memory image; channel axis 0 when *channels* is given."""
    scene = controller.add_scene(dim="3d", name="scene")
    rng = np.random.default_rng(0)
    if channels is None:
        store = ImageMemoryStore(data=rng.random((8, 8, 8), dtype=np.float32))
    else:
        store = ImageMemoryStore(data=rng.random((3, 8, 8, 8), dtype=np.float32))
        kwargs.update(channel_axis=0, channels=channels)
        scene = controller.add_scene(
            coordinate_system=[("c", "channel"), *spatial_axes("z", "y", "x")],
            dim="3d",
            name="channels",
        )
    visual = controller.add_image(data=store, scene_id=scene.id, **kwargs)
    controller.add_canvas(scene_id=scene.id, canvas_size=(200, 125))
    return scene, visual


def _events(controller, visual) -> list:
    log: list = []
    owner = uuid4()
    controller.on_render_planes_changed(visual.id, log.append, owner_id=owner)
    controller.on_plane_interaction(visual.id, log.append, owner_id=owner)
    return log


async def _loaded(controller, scene) -> None:
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)


# -- the fields on the four visuals ---------------------------------------------


def test_the_four_visuals_take_render_planes_and_plane_mode(
    controller, tmp_path, image_volume, labels_volume
):
    fx.write_pyramid(tmp_path / "i", fx.ISO)
    fx.write_pyramid(tmp_path / "l", fx.ISO, labels=True)
    image_store, _ = fx.open_pyramid(tmp_path / "i", fx.ISO)
    label_store, _ = fx.open_pyramid(tmp_path / "l", fx.ISO)
    scene = controller.add_scene(dim="3d", name="scene")
    plane = _plane(_world(controller, scene))

    visuals = [
        controller.add_image(
            data=image_volume,
            scene_id=scene.id,
            single=InMemoryImageSingleAppearance(render_mode="plane"),
            render_planes=(plane,),
        ),
        controller.add_labels(
            data=labels_volume,
            scene_id=scene.id,
            appearance=InMemoryLabelsAppearance(render_mode="plane"),
            render_planes=(plane,),
        ),
        controller.add_image_multiscale(
            data=image_store,
            scene_id=scene.id,
            single=MultiscaleImageSingleAppearance(render_mode="plane"),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            render_planes=(plane,),
        ),
        controller.add_labels_multiscale(
            data=label_store,
            scene_id=scene.id,
            appearance=MultiscaleLabelsAppearance(render_mode="plane"),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            render_planes=(plane,),
        ),
    ]
    for visual in visuals:
        assert visual.render_planes == (plane,)
        assert visual.plane_mode() is True
        assert controller.get_render_plane(visual.id, plane.id) == plane


def test_render_planes_default_to_empty_and_are_validated_on_assignment(
    controller, image_volume
):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_image(data=image_volume, scene_id=scene.id)
    assert visual.render_planes == ()
    assert visual.plane_mode() is False
    plane = _plane(_world(controller, scene))
    visual.render_planes = [plane]  # a list is coerced
    assert visual.render_planes == (plane,)
    with pytest.raises(REFUSED):
        visual.render_planes = ("not a plane",)
    assert visual.render_planes == (plane,)


def test_a_visual_without_render_planes_is_refused(controller):
    scene = controller.add_scene(dim="3d", name="scene")
    points = controller.add_points(
        data=PointsMemoryStore(positions=np.zeros((4, 3), dtype=np.float32)),
        scene_id=scene.id,
    )
    with pytest.raises(TypeError, match="image and labels"):
        controller.set_render_planes(points.id, ())
    with pytest.raises(KeyError):
        controller.get_render_plane(points.id, uuid4())


# -- the rules (4.3) --------------------------------------------------------------


def test_bad_tuples_are_refused_and_the_last_good_one_put_back(
    controller, image_volume
):
    scene = controller.add_scene(dim="3d", name="scene")
    other = controller.add_scene(dim="3d", name="other")
    visual = controller.add_image(data=image_volume, scene_id=scene.id)
    world = _world(controller, scene)
    good = _plane(world)
    controller.set_render_planes(visual.id, (good,))
    log = _events(controller, visual)

    cases = {
        "world coordinate system": (_plane(_world(controller, other)),),
        "share an id": (good, good.model_copy(update={"origin": (1.0, 0.0, 0.0)})),
        "at most 4": tuple(_plane(world, z) for z in range(5)),
        "not one axis": (good.model_copy(update={"axes": ("z", "y", "q")}),),
    }
    for match, planes in cases.items():
        # Through the method: checked before anything changes.
        with pytest.raises(ValueError, match=match):
            controller.set_render_planes(visual.id, planes)
        assert visual.render_planes == (good,)
        # By direct assignment: put back, and the error raised.
        with pytest.raises(REFUSED, match=match):
            visual.render_planes = planes
        assert visual.render_planes == (good,)
    assert log == []

    # Four are allowed.
    four = tuple(_plane(world, z) for z in range(4))
    assert controller.set_render_planes(visual.id, four) == four


def test_a_plane_on_an_axis_the_data_does_not_span_is_refused(controller, image_volume):
    """P16: each plane axis must be one data axis, scaled and shifted."""
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="scene",
    )
    visual = controller.add_image(data=image_volume, scene_id=scene.id)  # zyx data
    world = _world(controller, scene)
    on_data_axes = RenderPlane.from_point_normal(
        world, (4, 0, 0), (1, 0, 0), axes=("z", "y", "x")
    )
    controller.set_render_planes(visual.id, (on_data_axes,))
    off_data_axes = RenderPlane.from_point_normal(
        world, (0, 0, 0), (1, 0, 0), axes=("t", "y", "x")
    )
    with pytest.raises(ValueError, match="scale and a translation per axis"):
        controller.set_render_planes(visual.id, (off_data_axes,))
    with pytest.raises(REFUSED, match="scale and a translation per axis"):
        visual.render_planes = (off_data_axes,)
    assert visual.render_planes == (on_data_axes,)


def test_a_visual_added_with_bad_planes_is_refused(controller, image_volume):
    scene = controller.add_scene(dim="3d", name="scene")
    other = controller.add_scene(dim="3d", name="other")
    with pytest.raises(ValueError, match="world coordinate system"):
        controller.add_image(
            data=image_volume,
            scene_id=scene.id,
            render_planes=(_plane(_world(controller, other)),),
        )
    assert controller._model.scenes[scene.id].visuals == []


# -- C9, as decided 2026-10-06: every channel has the same render mode ------------


def _channel_events(controller, visual) -> list:
    log: list = []
    controller._outgoing_events.subscribe(
        ChannelAppearanceChangedEvent,
        log.append,
        entity_id=visual.id,
        owner_id=uuid4(),
    )
    return log


def _channels(*modes):
    return {
        index: InMemoryImageChannelAppearance(render_mode=mode)
        for index, mode in enumerate(modes)
    }


def test_channels_with_different_modes_are_refused_when_the_visual_is_built(
    controller,
):
    """Plane beside volume, and one volume mode beside another, alike."""
    for modes in (("plane", "mip", "mip"), ("mip", "iso", "mip")):
        with pytest.raises(ValueError, match="different render modes"):
            _memory_image(controller, channels=_channels(*modes), composite=True)
        # In single mode too, and for a hidden channel.
        channels = _channels(*modes)
        channels[0].visible = False
        with pytest.raises(ValueError, match="different render modes"):
            _memory_image(controller, channels=channels)


async def test_one_channel_cannot_be_given_a_mode_alone(controller):
    _scene, visual = _memory_image(
        controller, channels=_channels("mip", "mip", "mip"), composite=True
    )
    log = _channel_events(controller, visual)
    for mode in ("plane", "iso"):
        with pytest.raises(REFUSED, match="set_image_render_mode"):
            visual.channels[1].render_mode = mode
        assert [c.render_mode for c in visual.channels.values()] == ["mip"] * 3
    assert log == []  # a refused mode is not announced

    # A replaced dict is checked by the model: the old one stays.
    previous = visual.channels
    with pytest.raises(ValueError, match="different render modes"):
        visual.channels = _channels("plane", "mip", "plane")
    assert visual.channels is previous
    visual.channels = _channels("iso", "iso", "iso")
    assert visual.plane_mode() is False


async def test_set_image_render_mode_sets_every_channel(controller):
    _scene, visual = _memory_image(
        controller, channels=_channels("mip", "mip", "mip"), composite=True
    )
    visual.channels[2].visible = False  # hidden channels are set too
    log = _channel_events(controller, visual)
    source = uuid4()

    controller.set_image_render_mode(visual.id, "plane", source_id=source)
    assert [c.render_mode for c in visual.channels.values()] == ["plane"] * 3
    assert visual.plane_mode() is True
    assert [(e.channel_index, e.field_name, e.new_value) for e in log] == [
        (index, "render_mode", "plane") for index in range(3)
    ]
    assert {e.source_id for e in log} == {source}
    # The single appearance is separate.
    assert visual.single.render_mode == "mip"

    # The same mode again changes nothing and announces nothing.
    log.clear()
    controller.set_image_render_mode(visual.id, "plane")
    assert log == []

    # An unknown mode is refused before any channel changes.
    with pytest.raises(ValueError):
        controller.set_image_render_mode(visual.id, "no_such_mode")
    assert [c.render_mode for c in visual.channels.values()] == ["plane"] * 3
    assert log == []

    # After a sync a lone assignment is still refused, and put back.
    with pytest.raises(REFUSED, match="set_image_render_mode"):
        visual.channels[0].render_mode = "mip"
    assert visual.channels[0].render_mode == "plane"


async def test_a_channel_field_update_of_the_mode_sets_every_channel(controller):
    """What a GUI's per-channel combo sends."""
    _scene, visual = _memory_image(
        controller, channels=_channels("mip", "mip", "mip"), composite=True
    )
    log = _channel_events(controller, visual)
    widget = uuid4()
    controller.update_channel_appearance_field(
        visual.id, 1, "render_mode", "iso", source_id=widget
    )
    assert [c.render_mode for c in visual.channels.values()] == ["iso"] * 3
    # The widget's own channel carries its id; the others the controller's,
    # so the widget's echo filter lets them through.
    assert {e.channel_index: e.source_id for e in log} == {
        0: controller._id,
        1: widget,
        2: controller._id,
    }
    with pytest.raises(KeyError):
        controller.update_channel_appearance_field(visual.id, 7, "render_mode", "mip")


async def test_single_and_channel_modes_are_independent(controller):
    scene, visual = _memory_image(
        controller,
        channels=_channels("plane", "plane", "plane"),
        single=InMemoryImageSingleAppearance(render_mode="mip"),
    )
    assert visual.plane_mode() is False  # single mode reads ``single``
    visual.composite = True
    assert visual.plane_mode() is True
    visual.composite = False
    visual.single.render_mode = "plane"
    assert visual.plane_mode() is True
    controller.set_image_render_mode(visual.id, "iso")
    assert visual.single.render_mode == "plane"
    assert scene.id in controller._model.scenes


def test_only_an_image_has_a_channel_render_mode(controller, labels_volume):
    scene = controller.add_scene(dim="3d", name="scene")
    labels = controller.add_labels(data=labels_volume, scene_id=scene.id)
    with pytest.raises(TypeError, match="image visual"):
        controller.set_image_render_mode(labels.id, "plane")


async def test_the_multiscale_channels_follow_the_rule_and_reslice_once(
    controller, tmp_path, monkeypatch
):
    spec = fx.channel_spec(2, fx.ISO)
    fx.write_pyramid(tmp_path, spec)
    store, _ = fx.open_pyramid(tmp_path, spec)
    scene = controller.add_scene(
        coordinate_system=spatial_axes("c", "z", "y", "x"), dim="3d", name="scene"
    )
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        channel_axis=0,
        composite=True,
        channels={
            0: MultiscaleImageChannelAppearance(render_mode="mip"),
            1: MultiscaleImageChannelAppearance(render_mode="mip"),
        },
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
    )
    controller.add_canvas(scene_id=scene.id, canvas_size=(200, 125))
    world = _world(controller, scene)
    controller.set_render_planes(visual.id, (_plane(world, 30.0),))
    await _loaded(controller, scene)
    with pytest.raises(REFUSED, match="set_image_render_mode"):
        visual.channels[0].render_mode = "plane"
    assert visual.channels[0].render_mode == "mip"

    plans = _record_plans(monkeypatch)
    controller.set_image_render_mode(visual.id, "iso")  # volume to volume
    assert plans == []
    controller.set_image_render_mode(visual.id, "plane")
    assert len(plans) == 1  # entering plane mode: one reslice, not one a channel
    controller.set_image_render_mode(visual.id, "mip")
    assert len(plans) == 2
    await drain_loading(controller)


# -- events -----------------------------------------------------------------------


def test_one_changed_event_per_assignment_with_the_source_stamped(
    controller, image_volume
):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_image(data=image_volume, scene_id=scene.id)
    world = _world(controller, scene)
    log = _events(controller, visual)
    first, second = _plane(world, 2.0), _plane(world, 4.0)
    source = uuid4()

    controller.set_render_planes(visual.id, (first,), source_id=source)
    controller.set_render_planes(visual.id, (first,), source_id=source)  # equal
    visual.render_planes = (first, second)
    assert [type(event) for event in log] == [RenderPlanesChangedEvent] * 2
    assert [event.source_id for event in log] == [source, controller._id]
    assert [event.render_planes for event in log] == [(first,), (first, second)]
    assert {event.visual_id for event in log} == {visual.id}

    # A GUI's request goes the same way.
    widget = uuid4()
    controller.incoming_events.emit(
        RenderPlanesUpdateEvent(
            source_id=widget, visual_id=visual.id, render_planes=(second,)
        )
    )
    assert visual.render_planes == (second,)
    assert log[-1].source_id == widget


def test_set_render_plane_replaces_one_plane_and_keeps_its_id(controller, image_volume):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_image(data=image_volume, scene_id=scene.id)
    world = _world(controller, scene)
    first, second = _plane(world, 2.0), _plane(world, 4.0)
    controller.set_render_planes(visual.id, (first, second))

    replacement = _plane(world, 6.0, extent_0=(-1.0, 1.0), enabled=False)
    moved = controller.set_render_plane(visual.id, second.id, replacement)
    assert moved.id == second.id
    assert moved.origin == (6.0, 0.0, 0.0)
    assert moved.extent_0 == (-1.0, 1.0)
    assert moved.enabled is False
    assert visual.render_planes == (first, moved)
    assert controller.get_render_plane(visual.id, second.id) == moved
    with pytest.raises(KeyError):
        controller.set_render_plane(visual.id, uuid4(), replacement)


# -- 8.2: a change while not in plane mode is stored, announced, inert -------------


async def test_a_change_outside_plane_mode_is_inert(controller, tmp_path, monkeypatch):
    scene, visual = _multiscale_image(
        controller,
        tmp_path,
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 65535.0), render_mode="mip"
        ),
    )
    await _loaded(controller, scene)
    world = _world(controller, scene)
    log = _events(controller, visual)
    plans = _record_plans(monkeypatch)
    with controller.plane_interaction(visual.id):
        controller.set_render_planes(visual.id, (_plane(world),))
        controller.set_render_planes(visual.id, (_plane(world, 20.0),))
        assert controller.plane_interaction_state(visual.id) == "idle"
    assert plans == []
    assert [type(event) for event in log] == [RenderPlanesChangedEvent] * 2
    await drain_loading(controller)


async def test_plane_mode_is_ignored_in_a_2d_view(controller, tmp_path, monkeypatch):
    """D-P19: a 2D view shows the normal slice, planes or not."""
    scene, visual = _multiscale_image(controller, tmp_path, dim="2d")
    assert visual.plane_mode() is True
    assert controller._draws_planes(visual) is False
    world = _world(controller, scene)
    plans = _record_plans(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        controller.set_render_planes(visual.id, (_plane(world),))
    assert plans == []  # inert
    await _loaded(controller, scene)
    assert [(mode, sliced) for _, mode, sliced in plans] == [("FULL", True)]
    await drain_loading(controller)


# -- 4.2 (P17): plane mode with nothing to draw draws nothing ----------------------


async def test_plane_mode_with_no_plane_is_not_sliced_until_one_appears(
    controller, tmp_path, monkeypatch
):
    scene, visual = _multiscale_image(controller, tmp_path)
    world = _world(controller, scene)
    plans = _record_plans(monkeypatch)
    await _loaded(controller, scene)
    assert [(mode, sliced) for _, mode, sliced in plans] == [("FULL", False)]
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    cache_id = gfx.slots[0].residency_3d().cache_id
    progress = controller._render_manager.scheduler.progress(cache_id)
    assert progress.needed_backstop == 0  # no backstop either
    assert progress.needed_target == 0

    plans.clear()
    plane = _plane(world, 30.0)
    controller.set_render_planes(visual.id, (plane,))
    assert [(mode, sliced) for _, mode, sliced in plans] == [("FULL", True)]
    await drain_loading(controller)
    progress = controller._render_manager.scheduler.progress(cache_id)
    assert progress.needed_backstop > 0

    # Disabled, it is as if there were none.
    plans.clear()
    controller.set_render_planes(
        visual.id, (plane.model_copy(update={"enabled": False}),)
    )
    assert [(mode, sliced) for _, mode, sliced in plans] == [("FULL", False)]
    await drain_loading(controller)


async def test_entering_and_leaving_plane_mode_reslices(
    controller, tmp_path, monkeypatch
):
    scene, visual = _multiscale_image(
        controller,
        tmp_path,
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 65535.0), render_mode="mip"
        ),
    )
    controller.set_render_planes(visual.id, (_plane(_world(controller, scene)),))
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    visual.single.render_mode = "iso"  # volume to volume: the same bricks
    assert plans == []
    visual.single.render_mode = "plane"
    assert [(mode, sliced) for _, mode, sliced in plans] == [("FULL", True)]
    visual.single.render_mode = "mip"
    assert len(plans) == 2
    await drain_loading(controller)


async def test_an_in_memory_visual_is_sliced_when_its_first_plane_appears(
    controller, image_volume, monkeypatch
):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_image(
        data=image_volume,
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(render_mode="plane"),
    )
    controller.add_canvas(scene_id=scene.id)
    world = _world(controller, scene)
    assert controller._render_config_for(scene.id, visual).slicing_enabled is False
    reslices: list = []
    original = controller.reslice_visual
    monkeypatch.setattr(
        controller,
        "reslice_visual",
        lambda vid: (reslices.append(vid), original(vid))[1],
    )
    controller.set_render_planes(visual.id, (_plane(world, 2.0),))
    assert reslices == [visual.id]
    assert controller._render_config_for(scene.id, visual).slicing_enabled is True
    # A moved plane is a uniform write: nothing is read again.
    controller.set_render_planes(visual.id, (_plane(world, 3.0),))
    assert reslices == [visual.id]


# -- C10 and 4.4: a plane whose axes are not displayed -----------------------------


async def test_a_plane_off_the_displayed_axes_is_not_drawn_and_warns_once(
    controller, tmp_path, monkeypatch
):
    scene, visual = _multiscale_image(controller, tmp_path, spec=fx.TZYX)
    world = _world(controller, scene)
    zyx = RenderPlane.from_point_normal(
        world, (8, 0, 0), (1, 0, 0), axes=("z", "y", "x")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # drawable: no warning
        controller.set_render_planes(visual.id, (zyx,))
    await _loaded(controller, scene)

    plans = _record_plans(monkeypatch)
    # Display t, y, x in 3D: the plane is on z, y, x.
    with pytest.warns(UserWarning, match="not drawn") as caught:
        controller.set_displayed_axes(scene.id, (0, 2, 3))
    assert len(caught) == 1
    await drain_loading(controller)
    assert plans
    assert all(sliced is False for _, _, sliced in plans)  # nothing planned

    # One warning per visual: further changes are quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        controller.set_render_planes(
            visual.id, (zyx.model_copy(update={"origin": (9.0, 0.0, 0.0)}),)
        )
        controller.reslice_all()

    # A plane on the displayed axes is drawn; the other still is not.
    tyx = RenderPlane.from_point_normal(
        world, (2, 0, 0), (0, 1, 0), axes=("t", "y", "x")
    )
    plans.clear()
    controller.set_render_planes(visual.id, (zyx, tyx))
    assert [(mode, sliced) for _, mode, sliced in plans] == [("FULL", True)]

    # Back on z, y, x the other plane is the one not drawn: still one warning
    # per visual, so this is quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        controller.set_displayed_axes(scene.id, (1, 2, 3))
    await drain_loading(controller)

    # Once every enabled plane is on the displayed axes the warning is armed
    # again.
    controller.set_render_planes(visual.id, (zyx,))
    with pytest.warns(UserWarning, match="not drawn") as caught:
        controller.set_displayed_axes(scene.id, (0, 2, 3))
    assert len(caught) == 1
    await drain_loading(controller)


# -- 7.4: a render plane change is a tick, and holds the plan ----------------------


async def test_a_render_plane_drag_holds_the_plan(controller, tmp_path, monkeypatch):
    scene, visual = _multiscale_image(controller, tmp_path)
    world = _world(controller, scene)
    plane = _plane(world, 20.0)
    controller.set_render_planes(visual.id, (plane,))
    await _loaded(controller, scene)
    log = _events(controller, visual)
    plans = _record_plans(monkeypatch)
    reads = controller._render_manager.scheduler.core
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    cache_id = gfx.slots[0].residency_3d().cache_id
    before = controller._render_manager.scheduler.progress(cache_id)

    with controller.plane_interaction(visual.id):
        for z in (30.0, 40.0, 50.0):
            controller.set_render_plane(
                visual.id, plane.id, plane.model_copy(update={"origin": (z, 0.0, 0.0)})
            )
        assert controller.plane_interaction_state(visual.id) == "active"
        assert plans == []  # nothing planned: no reads
        assert reads.idle()
        assert controller._render_manager.scheduler.progress(cache_id) == before
    assert [(mode, sliced) for _, mode, sliced in plans] == [("FULL", True)]

    interactions = [
        (event.phase, event.reason)
        for event in log
        if isinstance(event, PlaneInteractionEvent)
    ]
    assert interactions == [("start", None), ("end", "release")]
    # The start is announced ahead of the drag's first change.
    assert isinstance(log[0], PlaneInteractionEvent)
    assert isinstance(log[1], RenderPlanesChangedEvent)
    assert sum(isinstance(e, RenderPlanesChangedEvent) for e in log) == 3
    await drain_loading(controller)


async def test_a_render_plane_and_a_clipping_plane_share_one_drag(
    controller, tmp_path, monkeypatch
):
    """One tracker per visual serves both kinds of plane (D-P17)."""
    from cellier.visuals import ClippingPlane

    scene, visual = _multiscale_image(controller, tmp_path)
    world = _world(controller, scene)
    plane = _plane(world, 20.0)
    controller.set_render_planes(visual.id, (plane,))
    await _loaded(controller, scene)
    system = controller._model.data.stores[
        __import__("uuid").UUID(visual.data_store_id)
    ].data_coordinate_systems[0]
    log = _events(controller, visual)
    plans = _record_plans(monkeypatch)

    with controller.plane_interaction(visual.id):
        controller.set_render_planes(visual.id, (_plane(world, 30.0),))
        controller.set_clipping_planes(
            visual.id,
            (
                ClippingPlane.from_point_normal(
                    system, (0, 0, 40), (0, 0, 1), axes=("z", "y", "x")
                ),
            ),
        )
        assert plans == []
    assert len(plans) == 1  # the target loads once, at the end
    interactions = [e for e in log if isinstance(e, PlaneInteractionEvent)]
    assert [(e.phase, e.reason) for e in interactions] == [
        ("start", None),
        ("end", "release"),
    ]
    await drain_loading(controller)


async def test_an_eager_render_plane_drag_plans_every_tick(
    controller, tmp_path, monkeypatch
):
    scene, visual = _multiscale_image(controller, tmp_path)
    visual.appearance.coarsest_while_moving_3d = False
    world = _world(controller, scene)
    controller.set_render_planes(visual.id, (_plane(world, 20.0),))
    await _loaded(controller, scene)
    plans = _record_plans(monkeypatch)
    with controller.plane_interaction(visual.id):
        controller.set_render_planes(visual.id, (_plane(world, 30.0),))
        controller.set_render_planes(visual.id, (_plane(world, 40.0),))
        assert len(plans) == 2
    assert len(plans) == 2  # nothing more at the end
    await drain_loading(controller)


def test_an_in_memory_plane_drag_is_announced_and_reads_nothing(
    controller, image_volume, monkeypatch
):
    scene = controller.add_scene(dim="3d", name="scene")
    world = _world(controller, scene)
    visual = controller.add_image(
        data=image_volume,
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(render_mode="plane"),
        render_planes=(_plane(world, 2.0),),
    )
    controller.add_canvas(scene_id=scene.id)
    log = _events(controller, visual)
    reslices: list = []
    monkeypatch.setattr(controller, "reslice_visual", reslices.append)
    with controller.plane_interaction(visual.id):
        controller.set_render_planes(visual.id, (_plane(world, 3.0),))
    assert reslices == []
    assert [type(e) for e in log] == [
        PlaneInteractionEvent,
        RenderPlanesChangedEvent,
        PlaneInteractionEvent,
    ]


# -- nothing is drawn differently yet ---------------------------------------------


def test_the_gui_combos_offer_plane(controller, image_volume, labels_volume):
    """Design 9.1: "plane" is a choice of every render-mode combo."""
    from cellier.gui._appearance_fields import literal_choices
    from cellier.gui._image_controls import image_control_values

    scene = controller.add_scene(dim="3d", name="scene")
    image = controller.add_image(data=image_volume, scene_id=scene.id)
    labels = controller.add_labels(data=labels_volume, scene_id=scene.id)
    assert (
        "plane" in image_control_values(image, fields=["render_mode"])["render_modes"]
    )
    assert "plane" in literal_choices(labels.appearance, "render_mode")


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_a_plane_mode_visual_builds_and_draws(
    controller, render_scene, image_volume, labels_volume, kind
):
    """A visual added in plane mode with a plane draws it on its first frame.

    The pixels are checked in ``test_plane_rendering_memory.py``.
    """
    scene = controller.add_scene(dim="3d", name="scene")
    world = _world(controller, scene)
    if kind == "image":
        assert isinstance(image_volume, ImageMemoryStore)
        controller.add_image(
            data=image_volume,
            scene_id=scene.id,
            single=InMemoryImageSingleAppearance(render_mode="plane"),
            render_planes=(_plane(world, 2.0),),
        )
    else:
        assert isinstance(labels_volume, LabelMemoryStore)
        controller.add_labels(
            data=labels_volume,
            scene_id=scene.id,
            appearance=InMemoryLabelsAppearance(render_mode="plane"),
            render_planes=(_plane(world, 2.0),),
        )
    controller.add_canvas(scene_id=scene.id)
    await _loaded(controller, scene)
    frame = np.asarray(render_scene(controller, scene.id))
    assert (frame[..., 3] > 0).any()
