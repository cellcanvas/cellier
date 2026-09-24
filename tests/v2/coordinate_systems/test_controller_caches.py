"""The controller's runtime coordinate systems, and when they are rebuilt.

Design 3.1's invalidation table, asserted.  None of these caches exists for
speed -- composing affines is microseconds -- they exist so that **axis ids
are stable within a session**, which they are not if ``from_world`` /
``from_data`` are called per frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.transform import RenderedCoordinateSystem
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


@pytest.fixture
def controller_with_canvas():
    """A 4-D ``TZYX`` scene showing ``ZYX``, with one canvas and one image."""
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
        dim="3d",
    )
    controller.add_canvas(scene.id)
    store = ImageMemoryStore(data=np.zeros((4, 5, 6), dtype=np.float32))
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
    )
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    return controller, scene, canvas_id, visual, store


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_every_system_is_reachable_by_id(controller_with_canvas):
    """Three v2 methods take system objects while transforms store only ids."""
    controller, scene, canvas_id, visual, store = controller_with_canvas
    world = scene.dims.world_coordinate_system
    rendered, _ = controller._rendered[canvas_id]
    assert controller.coordinate_system(world.id) is world
    assert controller.coordinate_system(rendered.id) is rendered
    assert (
        controller.coordinate_system(store.data_coordinate_system.id)
        is store.data_coordinate_system
    )
    space = controller._visual_spaces[(visual.id, "3d")]
    assert controller.coordinate_system(space.id) is space


def test_an_unknown_id_raises_with_an_explanation(controller_with_canvas):
    controller = controller_with_canvas[0]
    from uuid import uuid4

    with pytest.raises(KeyError, match="not in this viewer"):
        controller.coordinate_system(uuid4())


# ---------------------------------------------------------------------------
# The rendered system
# ---------------------------------------------------------------------------


def test_the_rendered_system_is_in_cellier_displayed_order(controller_with_canvas):
    """Part 5 D1: the ``(z, y, x) -> (x, y, z)`` reversal stays at the pygfx
    boundary, so the rendered system reads against the world."""
    controller, _scene, canvas_id, _visual, _store = controller_with_canvas
    rendered, _ = controller._rendered[canvas_id]
    assert isinstance(rendered, RenderedCoordinateSystem)
    assert rendered.axis_names() == ("z", "y", "x")
    assert rendered.canvas_id == canvas_id


async def test_the_embedding_records_the_slice_position_as_a_constant(
    controller_with_canvas,
):
    """D35: a world axis the canvas does not display sits at a definite
    position, carried in the translation column."""
    controller, scene, canvas_id, _visual, _store = controller_with_canvas
    controller.update_slice_indices(scene.id, {0: 1.5})
    _, embedding = controller._rendered[canvas_id]
    # rows are (t, z, y, x); t is the collapsed one.
    assert embedding.matrix.shape == (5, 4)
    assert embedding.translation[0] == pytest.approx(1.5)
    assert not embedding.broadcast_axes


async def test_a_slice_move_keeps_the_rendered_system_and_moves_the_embedding(
    controller_with_canvas,
):
    """Same axes, same ids -- which is what keeps ids stable across a drag."""
    controller, scene, canvas_id, _visual, _store = controller_with_canvas
    before_system, before_embedding = controller._rendered[canvas_id]
    controller.update_slice_indices(scene.id, {0: 2.0})
    after_system, after_embedding = controller._rendered[canvas_id]
    assert after_system is before_system
    assert after_embedding is not before_embedding
    assert after_embedding.translation[0] == pytest.approx(2.0)


async def test_a_slice_move_rebuilds_the_embedding_from_the_kept_system(
    controller_with_canvas,
):
    """The moved embedding still starts at the rendered system the canvas holds.

    ``RenderedCoordinateSystem.from_world`` mints fresh ids on every call, so
    an embedding built from a new system paired the kept system with an
    embedding of a system nothing else knew.  Every later node matrix on the
    scene then failed its id check -- in the orthoviewer, on the first camera
    change after a channel-slider move in the 3D panel.
    """
    from cellier.render._spaces import node_matrix

    controller, scene, canvas_id, visual, _store = controller_with_canvas
    controller.update_slice_indices(scene.id, {0: 2.0})

    system, embedding = controller._rendered[canvas_id]
    assert embedding.input_coordinate_system == system.id

    # The composition that raised: data -> world -> rendered.
    spaces = controller.render_spaces(visual.id)
    assert spaces.world_to_rendered.output_coordinate_system == spaces.rendered.id
    matrix = node_matrix(spaces, controller.get_visual_model(visual.id).transform, {})
    assert matrix.shape == (4, 4)


async def test_changing_the_displayed_set_rebuilds_the_rendered_system(
    controller_with_canvas,
):
    controller, scene, canvas_id, _visual, _store = controller_with_canvas
    before, _ = controller._rendered[canvas_id]
    # Every axis must stay covered by displayed | sliced at all times, so the
    # newly hidden axis gains a position before it stops being displayed.
    controller.update_slice_indices(scene.id, {0: 0.0, 1: 0.0})
    controller.update_displayed_axes(scene.id, (2, 3))
    after, _ = controller._rendered[canvas_id]
    assert after is not before
    assert after.axis_names() == ("y", "x")


async def test_a_pure_reorder_still_rebuilds_the_rendered_system(
    controller_with_canvas,
):
    """Design 3.14: a transpose fetches identical data but is a different
    rendered system, and the node matrices derived from it change."""
    controller, scene, canvas_id, _visual, _store = controller_with_canvas
    before, _ = controller._rendered[canvas_id]
    assert before.axis_names() == ("z", "y", "x")
    controller.update_displayed_axes(scene.id, (3, 2, 1))
    after, _ = controller._rendered[canvas_id]
    assert after is not before
    assert after.axis_names() == ("x", "y", "z")


def test_a_removed_canvas_takes_its_rendered_system_with_it(controller_with_canvas):
    controller, _scene, canvas_id, _visual, _store = controller_with_canvas
    rendered, _ = controller._rendered[canvas_id]
    controller.remove_canvas(canvas_id)
    assert canvas_id not in controller._rendered
    with pytest.raises(KeyError):
        controller.coordinate_system(rendered.id)


# ---------------------------------------------------------------------------
# The visual system
# ---------------------------------------------------------------------------


async def test_the_visual_system_is_in_ascending_data_axis_order(
    controller_with_canvas,
):
    """Design 3.14's invariant: ``displayed_axes`` is a display order, never a
    fetch order.  ``get_data`` returns axes ascending and numpy will not
    negotiate, so the permutation lives in the transform."""
    controller, scene, _canvas_id, visual, _store = controller_with_canvas
    controller.update_displayed_axes(scene.id, (3, 2, 1))
    space = controller._visual_spaces[(visual.id, "3d")]
    assert space.axis_names() == ("z", "y", "x")


async def test_the_visual_system_drops_the_collapsed_axes(controller_with_canvas):
    controller, scene, _canvas_id, visual, _store = controller_with_canvas
    # Every axis must stay covered by displayed | sliced at all times, so the
    # newly hidden axis gains a position before it stops being displayed.
    controller.update_slice_indices(scene.id, {0: 0.0, 1: 0.0})
    controller.update_displayed_axes(scene.id, (2, 3))
    space = controller._visual_spaces[(visual.id, "2d")]
    assert space.axis_names() == ("y", "x")
    assert space.visual_id == visual.id


def test_two_visuals_over_one_store_get_distinct_spaces(controller_with_canvas):
    """D45: the system is per visual, so two visuals never share axis ids."""
    controller, scene, _canvas_id, first, store = controller_with_canvas
    second = controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        name="second",
        single=InMemoryImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
    )
    one = controller._visual_spaces[(first.id, "3d")]
    two = controller._visual_spaces[(second.id, "3d")]
    assert one.id != two.id
    assert {axis.id for axis in one.axes}.isdisjoint({axis.id for axis in two.axes})


def test_a_removed_visual_takes_its_spaces_with_it(controller_with_canvas):
    controller, _scene, _canvas_id, visual, _store = controller_with_canvas
    space = controller._visual_spaces[(visual.id, "3d")]
    controller.remove_visual(visual.id)
    assert not [key for key in controller._visual_spaces if key[0] == visual.id]
    with pytest.raises(KeyError):
        controller.coordinate_system(space.id)


def test_a_removed_scene_takes_its_world_with_it(controller_with_canvas):
    controller, scene, _canvas_id, _visual, _store = controller_with_canvas
    world = scene.dims.world_coordinate_system
    controller.remove_scene(scene.id)
    with pytest.raises(KeyError):
        controller.coordinate_system(world.id)
