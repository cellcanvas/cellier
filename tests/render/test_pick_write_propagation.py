"""Audit guard: model ``pick_write`` reaches the pygfx material for every visual.

pygfx materials default ``pick_write=False``, so every render-layer visual must
explicitly forward ``visual_model.pick_write`` or the pick buffer never records
the visual and ``hit_visual_id`` is always ``None``.  These tests build each
in-memory visual through the controller and assert the resulting material(s)
have ``pick_write is True`` by default.
"""

from __future__ import annotations

import numpy as np

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.visuals import InMemoryImageChannelAppearance, MeshFlatAppearance


def _scene(controller, axis_labels=("z", "y", "x")):
    cs = world_coordinate_system(spatial_axes(*axis_labels), name="world")
    return controller.add_scene(
        dim="3d", coordinate_system=cs, name="s", render_modes={"2d", "3d"}
    )


def test_label_memory_pick_write_2d_and_3d():
    controller = CellierController()
    scene = _scene(controller)
    store = LabelMemoryStore(data=np.zeros((4, 4, 4), dtype=np.int32), name="lb")
    visual = controller.add_labels(data=store, scene_id=scene.id, name="labels")

    gv = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    assert gv._inner_node_2d.material.pick_write is True
    assert gv._inner_node_3d.material.pick_write is True


def test_composite_image_memory_pick_write_on_every_slot():
    controller = CellierController()
    scene = _scene(controller, axis_labels=("c", "z", "y", "x"))
    store = ImageMemoryStore(data=np.zeros((2, 4, 4, 4), dtype=np.float32), name="mc")
    channels = {
        0: InMemoryImageChannelAppearance(color_map="red"),
        1: InMemoryImageChannelAppearance(color_map="green"),
    }
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        channel_axis=0,
        composite=True,
        channels=channels,
        name="mci",
    )

    gv = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    assert all(
        node.material.pick_write is True for slot in gv.slots for node in slot.nodes()
    )


# ---------------------------------------------------------------------------
# Runtime toggle tests
# ---------------------------------------------------------------------------


def test_points_pick_write_runtime_toggle():
    controller = CellierController()
    scene = _scene(controller)
    store = PointsMemoryStore(positions=np.zeros((3, 3), dtype=np.float32), name="pts")
    visual = controller.add_points(data=store, scene_id=scene.id)
    gv = controller._render_manager._scenes[scene.id].get_visual(visual.id)

    visual.pick_write = False
    assert gv._material.pick_write is False
    visual.pick_write = True
    assert gv._material.pick_write is True


def test_lines_pick_write_runtime_toggle():
    controller = CellierController()
    scene = _scene(controller)
    positions = np.zeros((4, 3), dtype=np.float32)
    store = LinesMemoryStore(positions=positions, name="ln")
    visual = controller.add_lines(data=store, scene_id=scene.id)
    gv = controller._render_manager._scenes[scene.id].get_visual(visual.id)

    visual.pick_write = False
    assert gv._material.pick_write is False
    visual.pick_write = True
    assert gv._material.pick_write is True


def test_mesh_pick_write_runtime_toggle():
    controller = CellierController()
    scene = _scene(controller)
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    store = MeshMemoryStore(positions=pos, indices=idx, name="msh")
    visual = controller.add_mesh(
        data=store, scene_id=scene.id, appearance=MeshFlatAppearance()
    )
    gv = controller._render_manager._scenes[scene.id].get_visual(visual.id)

    visual.pick_write = False
    assert gv._material_3d.pick_write is False
    assert gv._material_2d.pick_write is False
    visual.pick_write = True
    assert gv._material_3d.pick_write is True
    assert gv._material_2d.pick_write is True


def test_composite_image_memory_pick_write_runtime_toggle():
    controller = CellierController()
    scene = _scene(controller, axis_labels=("c", "z", "y", "x"))
    store = ImageMemoryStore(data=np.zeros((2, 4, 4, 4), dtype=np.float32), name="mc")
    channels = {
        0: InMemoryImageChannelAppearance(color_map="red"),
        1: InMemoryImageChannelAppearance(color_map="green"),
    }
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        channel_axis=0,
        composite=True,
        channels=channels,
        name="mci",
    )
    gv = controller._render_manager._scenes[scene.id].get_visual(visual.id)

    visual.pick_write = False
    assert all(
        node.material.pick_write is False for slot in gv.slots for node in slot.nodes()
    )
    visual.pick_write = True
    assert all(
        node.material.pick_write is True for slot in gv.slots for node in slot.nodes()
    )
