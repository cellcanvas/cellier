"""The bounding-box control for the geometry visuals.

Found by a user running ``examples/convenience/geometry_viewer_marimo.py``:
the AABB checkbox did nothing on mesh, points, lines or graph.  It had never
worked there -- ``_aabb_line`` was declared ``None`` in all four GFX visuals
and never constructed, so only the image and label visuals ever drew a box.
The control was unreachable on geometry panels until ``controls=`` reached
those visual types, which is why nothing caught it.

The first assertion is in **pixels**, because every intermediate step looked
correct while the picture did not change.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data.graph._graph_memory_store import GraphMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.visuals._mesh_memory import MeshFlatAppearance

_POSITIONS = np.array([[1, 1, 1], [5, 1, 1], [1, 5, 1], [1, 1, 5]], dtype=np.float32)


def _add_mesh(controller, scene_id):
    return controller.add_mesh(
        MeshMemoryStore(
            positions=_POSITIONS,
            indices=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32),
        ),
        scene_id,
        MeshFlatAppearance(color=(1.0, 0.2, 0.2, 1.0)),
        "mesh",
    )


def _add_points(controller, scene_id):
    return controller.add_points(
        PointsMemoryStore(positions=_POSITIONS), scene_id, None, "points"
    )


def _add_lines(controller, scene_id):
    return controller.add_lines(
        LinesMemoryStore(positions=_POSITIONS), scene_id, None, "lines"
    )


def _add_graph(controller, scene_id):
    return controller.add_graph(
        GraphMemoryStore.from_arrays(
            _POSITIONS, np.array([[0, 1], [1, 2], [2, 3]], np.int32), name="g"
        ),
        scene_id,
        None,
        "graph",
    )


ADDERS = {
    "mesh": _add_mesh,
    "points": _add_points,
    "lines": _add_lines,
    "graph": _add_graph,
}


def _scene_with(controller, kind):
    """A 3-D scene holding one geometry visual, with a canvas."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = ADDERS[kind](controller, scene.id)
    controller.add_canvas(scene_id=scene.id)
    return scene, visual


def _gfx(controller, scene_id, visual_id):
    return controller._render_manager._scenes[scene_id].get_visual(visual_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(ADDERS))
async def test_the_box_draws_and_undraws(controller, render_scene, reslice, kind):
    """Enabling the box changes the frame; disabling restores it exactly."""
    scene, visual = _scene_with(controller, kind)
    await reslice(controller, scene.id)

    before = render_scene(controller, scene.id)

    controller.update_aabb_field(visual.id, "enabled", True)
    shown = render_scene(controller, scene.id)
    assert not np.array_equal(before, shown), f"{kind}: the box drew nothing"

    controller.update_aabb_field(visual.id, "enabled", False)
    hidden = render_scene(controller, scene.id)
    assert np.array_equal(before, hidden), f"{kind}: turning it off left a trace"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(ADDERS))
async def test_the_box_matches_the_data_extent(controller, reslice, kind):
    """Sized from the committed geometry, not from a placeholder."""
    scene, visual = _scene_with(controller, kind)
    await reslice(controller, scene.id)
    controller.update_aabb_field(visual.id, "enabled", True)

    corners = np.asarray(
        _gfx(controller, scene.id, visual.id)._aabb_line.geometry.positions.data
    )

    # pygfx is (x, y, z) where the data is (z, y, x), so compare as sets.
    assert np.allclose(sorted(corners.min(axis=0)), [1.0, 1.0, 1.0])
    assert np.allclose(sorted(corners.max(axis=0)), [5.0, 5.0, 5.0])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(ADDERS))
async def test_the_box_does_not_grow_when_refreshed(controller, reslice, kind):
    """It is a child of the node it measures, so it must not measure itself.

    ``get_geometry_bounding_box`` excludes children; ``get_bounding_box``
    would not, and the box would expand a little on every commit.
    """
    scene, visual = _scene_with(controller, kind)
    await reslice(controller, scene.id)
    controller.update_aabb_field(visual.id, "enabled", True)
    gfx_visual = _gfx(controller, scene.id, visual.id)
    first = np.asarray(gfx_visual._aabb_line.geometry.positions.data).copy()

    for _ in range(3):
        gfx_visual._refresh_aabb()

    assert np.array_equal(
        first, np.asarray(gfx_visual._aabb_line.geometry.positions.data)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(ADDERS))
async def test_colour_and_width_reach_the_line(controller, reslice, kind):
    scene, visual = _scene_with(controller, kind)
    await reslice(controller, scene.id)
    controller.update_aabb_field(visual.id, "enabled", True)
    controller.update_aabb_field(visual.id, "color", "#ff00ff")
    controller.update_aabb_field(visual.id, "line_width", 6.0)

    material = _gfx(controller, scene.id, visual.id)._aabb_line.material
    assert material.thickness == pytest.approx(6.0)
    assert tuple(material.color)[:3] == pytest.approx((1.0, 0.0, 1.0))


@pytest.mark.asyncio
async def test_the_mesh_box_is_the_store_extent_before_any_data(controller):
    """A mesh sizes its box from the store's extent, not from loaded faces.

    So the box is there while the mesh loads, and does not blink with it
    (``plans/mesh_refactor_v3.md`` M5).
    """
    scene, visual = _scene_with(controller, "mesh")
    controller.update_aabb_field(visual.id, "enabled", True)

    line = _gfx(controller, scene.id, visual.id)._aabb_line
    assert line.visible is True
    corners = np.asarray(line.geometry.positions.data)
    assert np.allclose(sorted(corners.min(axis=0)), [1.0, 1.0, 1.0])
    assert np.allclose(sorted(corners.max(axis=0)), [5.0, 5.0, 5.0])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [kind for kind in ADDERS if kind != "mesh"])
async def test_the_box_stays_hidden_before_any_data(controller, kind):
    """A box around no data would be a box around the origin.

    The image visuals gate on ``_data_ready_2d`` / ``_data_ready_3d`` for the
    same reason; the geometry ones gate on having found real bounds.

    Async only so that adding the visual happens inside pytest-asyncio's loop:
    ``AsyncSlicer.submit`` calls ``asyncio.ensure_future``, and outside a
    running loop that silently creates one nobody closes (see the module
    docstring of ``tests/render/test_multiscale_paint_textures.py``).
    """
    scene, visual = _scene_with(controller, kind)
    controller.update_aabb_field(visual.id, "enabled", True)

    assert _gfx(controller, scene.id, visual.id)._aabb_line.visible is False
