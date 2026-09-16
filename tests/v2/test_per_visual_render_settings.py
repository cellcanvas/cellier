"""Tests for the per-visual screen-space render settings.

Outline slot and placement, the ambient-occlusion tri-state, and the labels
visuals' per-label selection all live on the **visual model** now, alongside
``aabb`` and ``pick_write``, rather than only in the render layer.  That move
is what makes them serialize, travel on the bus, and die with the visual they
belong to; these tests pin each of those.

The render layer keeps a flag map, but it is a *cache* derived from the
models -- ``test_the_render_layer_cache_*`` are the tests that say so.
"""

from __future__ import annotations

import json
from uuid import uuid4

import numpy as np
import pytest

from cellier.controller import VISUAL_RENDER_FIELDS, CellierController
from cellier.data import (
    ImageMemoryStore,
    LabelMemoryStore,
    LinesMemoryStore,
    MeshMemoryStore,
    PointsMemoryStore,
)
from cellier.events import VisualRenderChangedEvent, VisualRenderUpdateEvent
from cellier.render import OutlineConfig, RenderManagerConfig
from cellier.visuals import (
    InMemoryImageAppearance,
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    LinesMemoryAppearance,
    MeshFlatAppearance,
    PointsMarkerAppearance,
)
from cellier.visuals._base_visual import VisualOutline
from cellier.visuals._label_memory import BaseLabelsVisual


@pytest.fixture
def controller():
    """A controller with the outline pass on, so outlining warns about less."""
    ctrl = CellierController(
        render_config=RenderManagerConfig(outline=OutlineConfig(enabled=True))
    )
    ctrl.camera_reslice_enabled = False
    yield ctrl
    ctrl.close()


@pytest.fixture
def scene(controller):
    return controller.add_scene(dim="3d", name="scene")


def _add_mesh(controller, scene, name="mesh"):
    return controller.add_mesh(
        data=MeshMemoryStore(
            positions=np.zeros((3, 3), dtype=np.float32),
            indices=np.array([[0, 1, 2]], dtype=np.int32),
            name=name,
        ),
        scene_id=scene.id,
        appearance=MeshFlatAppearance(color=(1.0, 1.0, 1.0, 1.0)),
        name=name,
    )


def _add_image(controller, scene, name="image", render_mode="iso"):
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[2:6, 2:6, 2:6] = 1.0
    return controller.add_image(
        data=ImageMemoryStore(data=data, name=name),
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        name=name,
        single=InMemoryImageSingleAppearance(
            color_map="gray", clim=(0.0, 1.0), render_mode=render_mode
        ),
    )


def _add_labels(controller, scene, name="labels"):
    data = np.zeros((8, 8, 8), dtype=np.int32)
    data[2:6, 2:6, 2:6] = 1
    return controller.add_labels(
        data=LabelMemoryStore(data=data, name=name),
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(
            colormap_mode="random", render_mode="iso_categorical"
        ),
        name=name,
    )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def test_a_visual_defaults_to_unoutlined_and_automatic(controller, scene):
    """Both features stay off by default, so nothing changes for existing code."""
    visual = _add_mesh(controller, scene)
    assert visual.outline.slot == 0
    assert visual.outline.placement is None
    assert visual.ambient_occlusion is None


def test_the_outline_slot_is_bounded(controller, scene):
    """The 4-bit LUT field cannot carry more than 15 slots."""
    visual = _add_mesh(controller, scene)
    with pytest.raises(ValueError, match="slot"):
        visual.outline.slot = 16
    with pytest.raises(ValueError, match="slot"):
        visual.outline.slot = -1


def test_an_unknown_placement_is_rejected(controller, scene):
    visual = _add_mesh(controller, scene)
    with pytest.raises(ValueError, match="placement"):
        visual.outline.placement = "sideways"


def test_placement_stays_none_until_asked_for(controller, scene):
    """Left unresolved so a visual follows the rule if the rule changes.

    Freezing the derived value into every serialized visual would mean a
    lines visual that never asked for a placement kept an old default
    forever.
    """
    visual = _add_mesh(controller, scene)
    visual.outline.slot = 1
    assert visual.outline.placement is None
    assert controller.get_visual_outline(visual.id) == (1, "inward")


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        ("mesh", "inward"),
        ("points", "outward"),
        ("lines", "outward"),
        ("graph", "outward"),
    ],
)
def test_placement_defaults_by_visual_type(controller, scene, factory, expected):
    """Outward for anything a few screen pixels wide; inward for the rest.

    A graph is in the outward group because its nodes and edges are both
    ``"screen"``-spaced by default -- an inward band twice the thickness of
    the thing it outlines consumes it entirely.
    """
    from cellier.data import GraphMemoryStore
    from cellier.visuals import GraphAppearance

    if factory == "mesh":
        visual = _add_mesh(controller, scene)
    elif factory == "points":
        visual = controller.add_points(
            data=PointsMemoryStore(
                positions=np.zeros((3, 3), dtype=np.float32), name="p"
            ),
            scene_id=scene.id,
            appearance=PointsMarkerAppearance(),
            name="points",
        )
    elif factory == "lines":
        visual = controller.add_lines(
            data=LinesMemoryStore(
                positions=np.zeros((4, 3), dtype=np.float32), name="l"
            ),
            scene_id=scene.id,
            appearance=LinesMemoryAppearance(),
            name="lines",
        )
    else:
        visual = controller.add_graph(
            data=GraphMemoryStore(
                positions=np.zeros((3, 3), dtype=np.float32),
                edges=np.array([[0, 1]], dtype=np.int64),
                name="g",
            ),
            scene_id=scene.id,
            appearance=GraphAppearance(),
            name="graph",
        )

    controller.set_visual_outline(visual.id, slot=1)
    assert controller.get_visual_outline(visual.id) == (1, expected)


def test_the_outline_mode_selects_the_shaders_kind(controller, scene):
    """Both modes already exist in the shader; the mode chooses between them.

    ``KIND_WHOLE_OBJECT`` keys on the pick id, so a region is one object --
    the same path a mesh or an intensity volume takes.  ``KIND_LABEL`` keys
    on the per-pixel label, so touching labels keep a band between them.
    """
    from cellier.render._visual_lut import KIND_LABEL, KIND_WHOLE_OBJECT

    labels = _add_labels(controller, scene)
    labels.outline.slot = 1
    assert labels.outline_mode == "per_label"
    assert controller._render_manager.get_visual_outline(labels.id)[2] == KIND_LABEL

    labels.outline_mode = "whole_object"

    assert (
        controller._render_manager.get_visual_outline(labels.id)[2] == KIND_WHOLE_OBJECT
    )


def test_a_non_labels_visual_is_always_one_object(controller, scene):
    """There is nothing to choose: a mesh has no labels to key on."""
    from cellier.render._visual_lut import KIND_WHOLE_OBJECT

    visual = _add_mesh(controller, scene)
    visual.outline.slot = 1

    assert controller._render_manager.get_visual_outline(visual.id)[2] == (
        KIND_WHOLE_OBJECT
    )
    with pytest.raises(ValueError, match="only available on labels visuals"):
        controller.update_visual_render_field(visual.id, "outline_mode", "whole_object")


def test_the_outline_mode_round_trips_through_a_file(controller, scene, tmp_path):
    labels = _add_labels(controller, scene)
    labels.outline.slot = 1
    labels.outline_mode = "whole_object"

    path = tmp_path / "viewer.json"
    controller.to_file(path)
    restored = CellierController.from_file(path)
    try:
        by_name = {
            v.name: v for v in next(iter(restored._model.scenes.values())).visuals
        }
        assert by_name["labels"].outline_mode == "whole_object"
    finally:
        restored.close()


def test_only_labels_visuals_carry_a_label_selection(controller, scene):
    """Every other visual type is outlined as one silhouette."""
    labels = _add_labels(controller, scene)
    mesh = _add_mesh(controller, scene)
    assert isinstance(labels, BaseLabelsVisual)
    assert not isinstance(mesh, BaseLabelsVisual)
    assert labels.outline_selected_labels == {}

    with pytest.raises(ValueError, match="only available on labels visuals"):
        controller.set_label_selection(mesh.id, {1: 1})


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_the_settings_round_trip_through_a_file(controller, scene, tmp_path):
    """The whole reason for moving them onto the model.

    Before this, reloading a saved viewer silently dropped every outline
    while the global outline config came back intact.
    """
    image = _add_image(controller, scene)
    labels = _add_labels(controller, scene)
    image.outline.slot = 2
    image.outline.placement = "outward"
    image.ambient_occlusion = True
    labels.outline_selected_labels = {1: 3}

    path = tmp_path / "viewer.json"
    controller.to_file(path)
    blob = json.loads(path.read_text())
    visuals = next(iter(blob["scenes"].values()))["visuals"]
    saved = {v["name"]: v for v in visuals}
    assert saved["image"]["outline"] == {"slot": 2, "placement": "outward"}
    assert saved["image"]["ambient_occlusion"] is True
    assert saved["labels"]["outline_selected_labels"] == {"1": 3}

    restored = CellierController.from_file(path)
    try:
        by_name = {
            v.name: v for v in next(iter(restored._model.scenes.values())).visuals
        }
        assert by_name["image"].outline.slot == 2
        assert by_name["image"].outline.placement == "outward"
        assert by_name["image"].ambient_occlusion is True
        assert by_name["labels"].outline_selected_labels == {1: 3}
    finally:
        restored.close()


def test_a_restored_visual_is_outlined_on_its_first_frame(controller, scene, tmp_path):
    """The render layer is seeded from the model, not left to a later write."""
    image = _add_image(controller, scene)
    image.outline.slot = 1

    path = tmp_path / "viewer.json"
    controller.to_file(path)
    restored = CellierController.from_file(path)
    try:
        assert len(restored._render_manager._visual_flags) == 1
    finally:
        restored.close()


def test_a_visual_added_with_an_outline_needs_no_second_call(controller, scene):
    """``add_visual`` seeds the render layer, so there is no unstyled frame.

    This is the path ``from_file`` takes, and the path anyone building visual
    models directly takes.  The ``add_mesh``-style convenience methods do not
    expose the field yet; setting it on the returned model is the one-liner
    equivalent, and is what every other test here does.
    """
    from cellier.visuals import MeshVisual

    store = MeshMemoryStore(
        positions=np.zeros((3, 3), dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int32),
        name="m",
    )
    controller.add_data_store(store)
    visual = MeshVisual(
        name="mesh",
        data_store_id=str(store.id),
        appearance=MeshFlatAppearance(color=(1.0, 1.0, 1.0, 1.0)),
        outline=VisualOutline(slot=1),
        ambient_occlusion=True,
    )
    controller.add_visual(scene.id, visual, data_store=store)

    assert controller._render_manager.get_visual_outline(visual.id) is not None
    assert controller._render_manager.get_visual_ambient_occlusion(visual.id) is True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_removing_a_visual_drops_its_render_flags(controller, scene):
    """Otherwise the map leaks, and a re-added id inherits the old outline."""
    visual = _add_mesh(controller, scene)
    visual.outline.slot = 1
    assert visual.id in controller._render_manager._visual_flags

    controller.remove_visual(visual.id)

    assert visual.id not in controller._render_manager._visual_flags


def test_removing_a_scene_drops_its_visuals_render_flags(controller, scene):
    visual = _add_mesh(controller, scene)
    visual.outline.slot = 1

    controller.remove_scene(scene.id)

    assert controller._render_manager._visual_flags == {}


def test_the_automatic_occlusion_rule_follows_the_render_mode(controller, scene):
    """MIP writes the depth of an extremum sample, so it is excluded.

    Re-derived per frame from the material rather than stored, which is what
    lets it track a render-mode change without anyone re-applying it.
    """
    manager = controller._render_manager
    image = _add_image(controller, scene, render_mode="iso")
    gfx_visual = manager._scenes[scene.id].get_visual(image.id)
    assert manager._mip_object_ids(gfx_visual) == set()

    image.single.render_mode = "mip"

    assert len(manager._mip_object_ids(gfx_visual)) > 0


def test_an_explicit_occlusion_setting_survives_a_render_mode_change(controller, scene):
    image = _add_image(controller, scene, render_mode="iso")
    image.ambient_occlusion = True

    image.single.render_mode = "mip"

    assert image.ambient_occlusion is True


# ---------------------------------------------------------------------------
# The pick_write coupling
# ---------------------------------------------------------------------------


def test_outlining_forces_pick_write_and_says_so(controller, scene):
    """Outlines are derived from the pick buffer, so picking is turned on.

    The most recent explicit action wins: asking for an outline is a request
    the renderer can satisfy by turning picking on, so it does -- out loud,
    rather than leaving the user to discover it.
    """
    visual = _add_mesh(controller, scene)
    visual.pick_write = False

    with pytest.warns(
        RuntimeWarning,
        match=r"outlines require pick_write=True\. pick_write set to True",
    ):
        visual.outline.slot = 1

    assert visual.pick_write is True


def test_turning_picking_off_warns_but_stands(controller, scene):
    """The other half: a decision about picking is not the renderer's to undo.

    The outline stops drawing -- there is no other per-pixel identity
    channel -- but the setting the user just made is left alone.
    """
    visual = _add_mesh(controller, scene)
    visual.outline.slot = 1

    with pytest.warns(RuntimeWarning, match=r"outlines require pick_write=True\.$"):
        visual.pick_write = False

    assert visual.pick_write is False
    assert visual.outline.slot == 1


def test_excluding_from_occlusion_forces_pick_write_and_says_so(controller, scene):
    """An occlusion *exclusion* needs identity for the same reason.

    An occlusion *inclusion* does not, which is why the normal target is
    deliberately not gated on pick.
    """
    visual = _add_mesh(controller, scene)
    visual.pick_write = False

    with pytest.warns(RuntimeWarning, match="ambient occlusion exclusions require"):
        visual.ambient_occlusion = False

    assert visual.pick_write is True


def test_turning_picking_off_warns_for_an_excluded_visual(controller, scene):
    visual = _add_mesh(controller, scene)
    visual.ambient_occlusion = False

    with pytest.warns(RuntimeWarning, match="ambient occlusion exclusions require"):
        visual.pick_write = False

    assert visual.pick_write is False


def test_including_a_visual_in_occlusion_needs_no_picking(controller, scene):
    """Only the exclusion needs identity, so this must not warn or force."""
    visual = _add_mesh(controller, scene)
    visual.pick_write = False

    with warnings_are_errors():
        visual.ambient_occlusion = True

    assert visual.pick_write is False


# ---------------------------------------------------------------------------
# The other two warnings
# ---------------------------------------------------------------------------


def test_outlining_while_the_pass_is_off_warns(scene):
    """Decision: do not switch the pass on implicitly -- say it is off."""
    controller = CellierController()  # outline pass defaults to off
    controller.camera_reslice_enabled = False
    try:
        own_scene = controller.add_scene(dim="3d", name="scene")
        visual = _add_mesh(controller, own_scene)
        with pytest.warns(RuntimeWarning, match="outline pass is off"):
            visual.outline.slot = 1
        assert controller.render_config.outline.enabled is False
    finally:
        controller.close()


def test_a_slot_past_the_palette_warns(controller, scene):
    """The pass pads unused slots with transparent, so it draws nothing."""
    visual = _add_mesh(controller, scene)
    n_slots = len(controller.render_config.outline.palette)

    with pytest.warns(RuntimeWarning, match="has no palette entry"):
        visual.outline.slot = n_slots + 1


def test_a_slot_inside_the_palette_does_not_warn(controller, scene):
    visual = _add_mesh(controller, scene)
    with warnings_are_errors():
        visual.outline.slot = len(controller.render_config.outline.palette)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@pytest.fixture
def seen(controller):
    events: list[VisualRenderChangedEvent] = []
    controller._outgoing_events.subscribe(
        VisualRenderChangedEvent, events.append, owner_id=uuid4()
    )
    return events


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outline.slot", 1),
        ("outline.placement", "outward"),
        ("ambient_occlusion", True),
    ],
)
def test_each_field_announces_itself(controller, scene, seen, field, value):
    visual = _add_mesh(controller, scene)
    controller.update_visual_render_field(visual.id, field, value)

    assert len(seen) == 1
    assert seen[0].visual_id == visual.id
    assert seen[0].field_name == field
    assert seen[0].new_value == value


def test_the_label_selection_announces_itself(controller, scene, seen):
    labels = _add_labels(controller, scene)
    controller.update_visual_render_field(labels.id, "outline_selected_labels", {1: 2})

    assert seen[-1].field_name == "outline_selected_labels"
    assert seen[-1].new_value == {1: 2}


def test_a_direct_model_write_announces_itself_too(controller, scene, seen):
    """The bridge hangs off the field, not off the setter.

    So ``visual.outline.slot = 1`` in a notebook cell reaches a subscribed
    widget exactly as the seam does.
    """
    visual = _add_mesh(controller, scene)
    visual.outline.slot = 1

    assert [e.field_name for e in seen] == ["outline.slot"]


def test_a_widget_update_event_drives_the_model(controller, scene):
    visual = _add_mesh(controller, scene)
    widget_id = uuid4()

    controller.incoming_events.emit(
        VisualRenderUpdateEvent(
            source_id=widget_id,
            visual_id=visual.id,
            field="ambient_occlusion",
            value=False,
        )
    )

    assert visual.ambient_occlusion is False


def test_the_widget_source_id_is_stamped_on_the_echo(controller, scene, seen):
    """So a widget can ignore the echo of its own change."""
    visual = _add_mesh(controller, scene)
    widget_id = uuid4()

    controller.update_visual_render_field(
        visual.id, "outline.slot", 1, source_id=widget_id
    )

    assert seen[-1].source_id == widget_id


def test_the_event_is_filterable_by_visual(controller, scene):
    """Two visuals, one subscription each; neither sees the other's change."""
    first = _add_mesh(controller, scene, name="first")
    second = _add_mesh(controller, scene, name="second")
    for_first: list = []
    controller._outgoing_events.subscribe(
        VisualRenderChangedEvent,
        for_first.append,
        entity_id=first.id,
        owner_id=uuid4(),
    )

    second.outline.slot = 1

    assert for_first == []


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


#: A valid value per settable field, so the whole table can be exercised.
_FIELD_VALUES = {
    "outline.slot": 1,
    "outline.placement": "outward",
    "ambient_occlusion": True,
    "pick_write": True,
}


def test_the_field_table_matches_what_the_seam_accepts(controller, scene):
    """Every field in the table is settable, and the table lists them all."""
    visual = _add_mesh(controller, scene)
    # The two labels-only fields are exercised in their own tests below.
    covered = {*_FIELD_VALUES, "outline_selected_labels", "outline_mode"}
    assert covered == set(VISUAL_RENDER_FIELDS)

    for field, value in _FIELD_VALUES.items():
        controller.update_visual_render_field(visual.id, field, value)
        target = visual.outline if field.startswith("outline.") else visual
        assert getattr(target, field.split(".")[-1]) == value


def test_an_unknown_field_raises_with_a_suggestion(controller, scene):
    visual = _add_mesh(controller, scene)
    with pytest.raises(ValueError, match=r"Did you mean 'outline\.slot'"):
        controller.update_visual_render_field(visual.id, "outline.slots", 1)


def test_the_legacy_setters_still_work(controller, scene):
    """``set_visual_*`` are wrappers now, and must behave identically."""
    visual = _add_mesh(controller, scene)

    controller.set_visual_outline(visual.id, slot=2, placement="outward")
    assert controller.get_visual_outline(visual.id) == (2, "outward")
    assert visual.outline.slot == 2

    controller.set_visual_ambient_occlusion(visual.id, False)
    assert controller.get_visual_ambient_occlusion(visual.id) is False
    assert visual.ambient_occlusion is False

    controller.set_visual_outline(visual.id, slot=0)
    assert controller.get_visual_outline(visual.id) is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def warnings_are_errors():
    """Context manager asserting the block emits no warning at all."""
    import warnings as _warnings

    class _Strict:
        def __enter__(self):
            self._ctx = _warnings.catch_warnings()
            self._ctx.__enter__()
            _warnings.simplefilter("error")
            return self

        def __exit__(self, *exc):
            return self._ctx.__exit__(*exc)

    return _Strict()


# ---------------------------------------------------------------------------
# The add_* keyword arguments
# ---------------------------------------------------------------------------

#: Every ``add_*`` on the controller, with the arguments it needs beyond the
#: render settings.  Parametrised rather than spot-checked because the
#: keywords were added to ten signatures mechanically, and a method missed in
#: that pass would look exactly like one that works until someone used it.
_ADD_METHODS = (
    "add_image",
    "add_labels",
    "add_mesh",
    "add_points",
    "add_lines",
    "add_graph",
    "add_image_with_channels",
)


def _call_add(controller, scene, method: str, **render_kwargs):
    """Call one ``add_*`` with a minimal valid payload plus *render_kwargs*."""
    from cellier.data import GraphMemoryStore
    from cellier.visuals import GraphAppearance, InMemoryImageChannelAppearance

    common = {"scene_id": scene.id, **render_kwargs}
    if method == "add_image":
        data = np.zeros((8, 8, 8), dtype=np.float32)
        return controller.add_image(
            data=ImageMemoryStore(data=data, name="i"),
            appearance=InMemoryImageAppearance(),
            **common,
            single=InMemoryImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
        )
    if method == "add_labels":
        data = np.zeros((8, 8, 8), dtype=np.int32)
        return controller.add_labels(
            data=LabelMemoryStore(data=data, name="l"),
            appearance=InMemoryLabelsAppearance(colormap_mode="random"),
            **common,
        )
    if method == "add_mesh":
        return controller.add_mesh(
            data=MeshMemoryStore(
                positions=np.zeros((3, 3), dtype=np.float32),
                indices=np.array([[0, 1, 2]], dtype=np.int32),
                name="m",
            ),
            appearance=MeshFlatAppearance(color=(1.0, 1.0, 1.0, 1.0)),
            **common,
        )
    if method == "add_points":
        return controller.add_points(
            data=PointsMemoryStore(
                positions=np.zeros((3, 3), dtype=np.float32), name="p"
            ),
            appearance=PointsMarkerAppearance(),
            **common,
        )
    if method == "add_lines":
        return controller.add_lines(
            data=LinesMemoryStore(
                positions=np.zeros((4, 3), dtype=np.float32), name="ln"
            ),
            appearance=LinesMemoryAppearance(),
            **common,
        )
    if method == "add_graph":
        return controller.add_graph(
            data=GraphMemoryStore(
                positions=np.zeros((3, 3), dtype=np.float32),
                edges=np.array([[0, 1]], dtype=np.int64),
                name="g",
            ),
            appearance=GraphAppearance(),
            **common,
        )
    data = np.zeros((2, 8, 8), dtype=np.float32)
    return controller.add_image(
        data=ImageMemoryStore(data=data, name="mc"),
        channel_axis=0,
        channels={0: InMemoryImageChannelAppearance(color_map="red")},
        **common,
    )


@pytest.mark.parametrize("method", _ADD_METHODS)
def test_add_methods_accept_the_render_settings(controller, scene, method):
    """The settings arrive with the visual rather than needing a second call."""
    visual = _call_add(
        controller,
        scene,
        method,
        outline=VisualOutline(slot=2, placement="outward"),
        ambient_occlusion=False,
    )

    assert visual.outline.slot == 2
    assert visual.outline.placement == "outward"
    assert visual.ambient_occlusion is False


@pytest.mark.parametrize("method", _ADD_METHODS)
def test_add_methods_seed_the_render_layer(controller, scene, method):
    """Seeded at registration, so there is no unstyled first frame."""
    visual = _call_add(controller, scene, method, outline=VisualOutline(slot=1))

    assert controller._render_manager.get_visual_outline(visual.id) is not None


@pytest.mark.parametrize("method", _ADD_METHODS)
def test_add_methods_default_to_no_render_settings(controller, scene, method):
    """Omitting them leaves every visual exactly as it was before."""
    visual = _call_add(controller, scene, method)

    assert visual.outline.slot == 0
    assert visual.ambient_occlusion is None
    assert visual.id not in controller._render_manager._visual_flags


def test_add_labels_accepts_a_label_selection(controller, scene):
    """The labels-only third keyword."""
    data = np.zeros((8, 8, 8), dtype=np.int32)
    visual = controller.add_labels(
        data=LabelMemoryStore(data=data, name="l"),
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(colormap_mode="random"),
        outline=VisualOutline(slot=1),
        outline_selected_labels={1: 2, 3: 4},
    )

    assert visual.outline_selected_labels == {1: 2, 3: 4}


def test_only_labels_add_methods_take_a_label_selection(controller, scene):
    """Every other visual type is outlined as one silhouette."""
    with pytest.raises(TypeError, match="outline_selected_labels"):
        controller.add_mesh(
            data=MeshMemoryStore(
                positions=np.zeros((3, 3), dtype=np.float32),
                indices=np.array([[0, 1, 2]], dtype=np.int32),
                name="m",
            ),
            scene_id=scene.id,
            appearance=MeshFlatAppearance(color=(1.0, 1.0, 1.0, 1.0)),
            outline_selected_labels={1: 1},
        )


def test_the_add_warnings_fire_once(controller, scene):
    """From the seed, not from the seed *and* a later write.

    The settings are written onto the model before ``add_visual`` registers
    it, so the psygnal bridge is not yet listening -- which is what keeps a
    single ``add_*`` from warning twice about the same thing.
    """
    controller.outline_enabled = False

    with pytest.warns(RuntimeWarning) as record:
        _call_add(controller, scene, "add_mesh", outline=VisualOutline(slot=1))

    off_warnings = [w for w in record if "outline pass is off" in str(w.message)]
    assert len(off_warnings) == 1
