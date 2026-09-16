"""Tests for get_node_for_dims on each GFX visual type."""

import numpy as np
import pytest

from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.render.visuals import GFXImageMemoryVisual
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import ImageVisual, InMemoryImageAppearance

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def store_3d():
    return ImageMemoryStore(data=np.zeros((4, 4, 4), dtype=np.float32))


@pytest.fixture
def visual_model_3d(store_3d):
    return ImageVisual(
        name="test",
        data_store_id=str(store_3d.id),
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="viridis"),
    )


# ── GFXImageMemoryVisual ──────────────────────────────────────────────────────


class TestGFXImageMemoryVisualGetNodeForDims:
    def test_returns_node_3d_for_3d_axes(self, visual_model_3d, store_3d):
        visual = GFXImageMemoryVisual(
            visual_model=visual_model_3d,
            data_store=store_3d,
            render_modes={"2d", "3d"},
        )
        node = visual.get_node_for_dims((0, 1, 2))
        assert node is visual.node_3d
        assert node is not None

    def test_returns_node_2d_for_2d_axes(self, visual_model_3d, store_3d):
        visual = GFXImageMemoryVisual(
            visual_model=visual_model_3d,
            data_store=store_3d,
            render_modes={"2d", "3d"},
        )
        node = visual.get_node_for_dims((1, 2))
        assert node is visual.node_2d
        assert node is not None

    def test_returns_none_for_3d_when_only_2d_built(self, visual_model_3d, store_3d):
        visual = GFXImageMemoryVisual(
            visual_model=visual_model_3d,
            data_store=store_3d,
            render_modes={"2d"},
        )
        node = visual.get_node_for_dims((0, 1, 2))
        assert node is None

    def test_updates_last_displayed_axes(self, visual_model_3d, store_3d):
        visual = GFXImageMemoryVisual(
            visual_model=visual_model_3d,
            data_store=store_3d,
            render_modes={"2d", "3d"},
        )
        assert visual._last_displayed_axes is None
        visual.get_node_for_dims((1, 2))
        assert visual._last_displayed_axes == (1, 2)

    def test_node_2d_and_node_3d_are_distinct_objects(self, visual_model_3d, store_3d):
        """Core invariant: image memory visual never has node_2d is node_3d."""
        visual = GFXImageMemoryVisual(
            visual_model=visual_model_3d,
            data_store=store_3d,
            render_modes={"2d", "3d"},
        )
        assert visual.node_2d is not visual.node_3d

    def test_repeated_call_same_axes_is_idempotent(self, visual_model_3d, store_3d):
        visual = GFXImageMemoryVisual(
            visual_model=visual_model_3d,
            data_store=store_3d,
            render_modes={"2d", "3d"},
        )
        node_first = visual.get_node_for_dims((0, 1, 2))
        node_second = visual.get_node_for_dims((0, 1, 2))
        assert node_first is node_second


# ── Integration: _rebuild_visuals_geometry uses get_node_for_dims ─────────────


def test_rebuild_visuals_geometry_delegates_to_get_node_for_dims(
    store_3d, visual_model_3d, qtbot
):
    """_rebuild_visuals_geometry calls get_node_for_dims on every visual.

    Uses a spy on get_node_for_dims to confirm it is called with the
    correct displayed_axes, and confirms swap_node is called on the
    SceneManager.
    """
    from cellier.controller import CellierController
    from cellier.scene.dims import spatial_axes, world_coordinate_system

    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    scene = controller.add_scene(
        dim="3d",
        coordinate_system=cs,
        name="test",
        render_modes={"2d", "3d"},
    )
    from cellier.visuals._image_memory import InMemoryImageAppearance

    controller.add_image(
        data=store_3d,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="viridis"),
    )

    scene_manager = controller._render_manager._scenes[scene.id]
    visual_id = next(iter(scene_manager._visuals.keys()))
    gfx_visual = scene_manager.get_visual(visual_id)

    # Spy on get_node_for_dims.
    calls = []
    original = gfx_visual.get_node_for_dims
    gfx_visual.get_node_for_dims = lambda axes: calls.append(axes) or original(axes)

    # Trigger a dim toggle.
    scene.dims.selection.displayed_axes = (1, 2)

    assert len(calls) == 1
    assert calls[0] == (1, 2)
    # Active node should now be node_2d.
    assert scene_manager.get_active_node(visual_id) is gfx_visual.node_2d
