"""Controller wiring for graph visuals: add_graph, trail, D21 (D11, D21, D23).

Every test here attaches a canvas.  A reslice fans out across the canvases
attached to a scene, so with none attached ``_build_request`` never runs and
a warn-once test would pass for entirely the wrong reason.  ``qtbot``
supplies the ``QApplication`` that ``add_canvas`` needs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.graph import GraphMemoryStore
from cellier.events import TrailChangedEvent
from cellier.scene.dims import CoordinateSystem
from cellier.transform import AffineTransform
from cellier.visuals import GraphVisual, TrailConfig


def _store(**kwargs) -> GraphMemoryStore:
    """A 4-D tzyx lineage: five tracks x ten timepoints, edges spanning t."""
    n_tracks, n_time = 5, 10
    positions = np.zeros((n_tracks * n_time, 4), dtype=np.float32)
    edges = []
    for track in range(n_tracks):
        for t in range(n_time):
            row = track * n_time + t
            positions[row] = (t, 4.0, 5.0 + track, 6.0 + t)
            if t > 0:
                edges.append((row - 1, row))
    return GraphMemoryStore.from_arrays(
        positions, np.asarray(edges, dtype=np.int32), **kwargs
    )


@pytest.fixture
def graph_setup(qtbot):
    """Return ``(controller, scene, store)`` with a canvas attached."""
    controller = CellierController()
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=CoordinateSystem(
            name="world", axis_labels=("t", "z", "y", "x")
        ),
        name="main",
        render_modes={"2d", "3d"},
    )
    controller.add_canvas(scene_id=scene.id)
    return controller, scene, _store()


def _gfx(controller, visual):
    scene_id = controller._visual_to_scene[visual.id]
    return controller._render_manager._scenes[scene_id].get_visual(visual.id)


# ── Registration ───────────────────────────────────────────────────────────


def test_add_graph_registers_store_and_visual(graph_setup):
    controller, scene, store = graph_setup
    visual = controller.add_graph(store, scene.id, name="lineage")

    assert isinstance(visual, GraphVisual)
    assert store.id in controller._model.data.stores
    assert visual in scene.visuals
    assert controller._visual_to_scene[visual.id] == scene.id
    assert _gfx(controller, visual) is not None


def test_add_graph_uses_store_transform(graph_setup):
    """A geff-derived transform becomes the default; explicit still wins (D23)."""
    controller, scene, _ = graph_setup
    scaled = _store(
        transform=AffineTransform.from_scale_and_translation(
            (1.0, 4.0, 0.26, 0.26), (0.0, 10.0, 0.0, 0.0)
        )
    )
    visual = controller.add_graph(scaled, scene.id, name="from_file")
    assert np.allclose(np.diag(visual.transform.matrix)[:4], [1.0, 4.0, 0.26, 0.26])

    explicit = AffineTransform.identity(ndim=4)
    override = controller.add_graph(
        scaled, scene.id, name="override", transform=explicit
    )
    assert np.allclose(override.transform.matrix, np.eye(5))


def test_add_graph_defaults_to_identity_without_store_transform(graph_setup):
    controller, scene, store = graph_setup
    visual = controller.add_graph(store, scene.id)
    assert np.allclose(visual.transform.matrix, np.eye(5))


# ── Trail wiring ───────────────────────────────────────────────────────────


def test_trail_reaches_the_render_layer_at_add(graph_setup):
    controller, scene, store = graph_setup
    visual = controller.add_graph(
        store, scene.id, trail={0: TrailConfig(before=3.0, after=1.0)}
    )
    assert _gfx(controller, visual)._trail[0].before == 3.0


async def test_trail_field_change_triggers_reslice(graph_setup):
    """Editing a nested TrailConfig field reaches the render layer.

    The nested-psygnal regression: psygnal does not propagate a child
    ``EventedModel``'s field change to the parent's event group, so without
    ``_wire_trail``'s per-config handler this edit would be invisible.
    """
    controller, scene, store = graph_setup
    visual = controller.add_graph(store, scene.id, trail={0: TrailConfig(before=1.0)})

    events: list[TrailChangedEvent] = []
    controller._outgoing_events.subscribe(TrailChangedEvent, events.append)

    visual.trail[0].before = 6.0

    assert len(events) == 1
    assert events[0].field_name == "before"
    assert events[0].axis == 0
    assert _gfx(controller, visual)._trail[0].before == 6.0


async def test_trail_dict_replacement_rewires(graph_setup):
    """``visual.trail = {...}`` rewires the per-config handlers and reslices."""
    controller, scene, store = graph_setup
    visual = controller.add_graph(store, scene.id, trail={0: TrailConfig(before=1.0)})

    events: list[TrailChangedEvent] = []
    controller._outgoing_events.subscribe(TrailChangedEvent, events.append)

    visual.trail = {1: TrailConfig(before=2.0)}
    assert len(events) == 1
    assert events[0].field_name is None
    assert set(_gfx(controller, visual)._trail) == {1}

    # The *new* config's fields are wired too -- this is what "rewires" means.
    visual.trail[1].after = 9.0
    assert len(events) == 2
    assert _gfx(controller, visual)._trail[1].after == 9.0


# ── D21: out-of-range axes raise ───────────────────────────────────────────


def test_out_of_range_trail_axis_raises_at_add(graph_setup):
    controller, scene, store = graph_setup
    with pytest.raises(ValueError, match="out of range"):
        controller.add_graph(store, scene.id, trail={7: TrailConfig()})


def test_out_of_range_trail_axis_message_names_axis_and_range(graph_setup):
    controller, scene, store = graph_setup
    with pytest.raises(ValueError, match=r"axis 7 .*4-axis graph.*0 to 3"):
        controller.add_graph(store, scene.id, trail={7: TrailConfig()})


def test_out_of_range_trail_axis_raises_on_assignment(graph_setup):
    """Assignment validates too, but psygnal wraps the error.

    The upper bound needs the store's ``ndim``, so the check can only run
    in the psygnal handler for ``visual.events.trail`` -- and psygnal wraps
    anything a callback raises in ``EmitLoopError``.  The ValueError is
    preserved verbatim as ``__cause__`` and its text is inlined in the
    wrapper's message, so the axis and the valid range still reach the
    user; only the exception *type* differs from the ``add_graph`` path.
    """
    from psygnal import EmitLoopError

    controller, scene, store = graph_setup
    visual = controller.add_graph(store, scene.id)
    with pytest.raises(EmitLoopError, match="out of range") as excinfo:
        visual.trail = {9: TrailConfig()}
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "valid axes are 0 to 3" in str(excinfo.value.__cause__)


def test_negative_trail_axis_raises(graph_setup):
    controller, scene, store = graph_setup
    with pytest.raises(ValueError, match="out of range"):
        controller.add_graph(store, scene.id, trail={-1: TrailConfig()})


# ── D21: displayed axes warn once ──────────────────────────────────────────


def _displayed_trail_setup(graph_setup, axis: int = 3):
    """Add a graph whose trail sits on a currently *displayed* axis."""
    controller, scene, store = graph_setup
    visual = controller.add_graph(store, scene.id, trail={axis: TrailConfig(before=2)})
    return controller, scene, visual


async def test_displayed_axis_warns_once(graph_setup):
    """A window on an axis you are looking down warns, once."""
    controller, scene, _ = _displayed_trail_setup(graph_setup)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        controller.reslice_scene(scene.id)
        controller.reslice_scene(scene.id)

    trail_warnings = [w for w in caught if "Trail configured on axis" in str(w.message)]
    assert len(trail_warnings) == 1
    assert "currently displayed" in str(trail_warnings[0].message)


async def test_warn_once_survives_dims_flapping(graph_setup):
    """Flip dims back and forth repeatedly: still exactly one warning.

    The set is sticky across *every* dims change; only a trail edit clears
    it.  Without that, toggling 2D/3D would spam.
    """
    controller, scene, _ = _displayed_trail_setup(graph_setup)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            scene.dims.selection.displayed_axes = (2, 3)
            scene.dims.selection.slice_indices = {0: 0, 1: 4}
            controller.reslice_scene(scene.id)
            scene.dims.selection.displayed_axes = (1, 2)
            scene.dims.selection.slice_indices = {0: 0, 3: 6}
            controller.reslice_scene(scene.id)

    trail_warnings = [w for w in caught if "Trail configured on axis" in str(w.message)]
    assert len(trail_warnings) == 1


async def test_warn_resets_on_trail_edit(graph_setup):
    """Editing any TrailConfig field produces one more warning next reslice."""
    controller, scene, visual = _displayed_trail_setup(graph_setup)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        controller.reslice_scene(scene.id)
        controller.reslice_scene(scene.id)
        visual.trail[3].before = 5.0
        controller.reslice_scene(scene.id)

    trail_warnings = [w for w in caught if "Trail configured on axis" in str(w.message)]
    assert len(trail_warnings) == 2


async def test_two_visuals_same_bad_axis_both_warn(graph_setup):
    """Guards against relying on the warnings module's process-global dedup.

    That dedup keys on (message, category, module, lineno), so two visuals
    with the same misconfigured axis would produce only one warning.  D21
    requires one per (visual, axis), which is why the set lives on the GFX
    visual.
    """
    controller, scene, store = graph_setup
    controller.add_graph(store, scene.id, name="a", trail={3: TrailConfig(before=2)})
    controller.add_graph(store, scene.id, name="b", trail={3: TrailConfig(before=2)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        controller.reslice_scene(scene.id)

    trail_warnings = [w for w in caught if "Trail configured on axis" in str(w.message)]
    assert len(trail_warnings) == 2


async def test_sliced_axis_trail_does_not_warn(graph_setup):
    """The normal case is silent."""
    controller, scene, store = graph_setup
    controller.add_graph(store, scene.id, trail={0: TrailConfig(before=3.0)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        controller.reslice_scene(scene.id)

    assert not [w for w in caught if "Trail configured on axis" in str(w.message)]
