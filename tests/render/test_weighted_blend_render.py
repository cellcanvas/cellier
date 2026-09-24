"""Weighted-blended order-independent transparency for labels and a graph.

A translucent labels volume with a tracking graph running through it (a
nucleus segmentation and its lineage tree, say) cannot be drawn correctly by
``transparency_mode="blend"``: pygfx sorts transparent objects back to front
by one point each (``wobject.world.position``), so which of the two draws
first -- and therefore whether the labels' depth write hides the part of the
graph inside a nucleus -- flips with the camera angle.  No per-object sort
can fix that for objects that interpenetrate.

These tests pin the configuration that removes the order dependence while
keeping the labels translucent: both visuals in ``"weighted_blend"`` with
``depth_write=False``.  (When the labels should instead *occlude* the graph,
make them opaque, keep the depth write, and draw them first with a lower
``render_order``.)  Draw order is swapped here with ``render_order`` rather
than by orbiting the camera, since that is exactly the variable the camera
angle was flipping.

They also cover the risk of adopting it: the cellier shaders and
``CellierBlender`` had only been checked in weighted mode at the
struct-editing level, and a shader that fails to compile there fails
silently inside the draw callback.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest

from cellier.data import GraphMemoryStore
from cellier.render._cellier_blender import EXTRA_TARGETS, install_cellier_blender
from cellier.visuals import GraphAppearance
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS

_SIZE = 96
_TRACK_COLOR = (0.0, 1.0, 0.0, 1.0)
_ALL_TARGETS = tuple(EXTRA_TARGETS)


def _track_store() -> GraphMemoryStore:
    """Two nodes and one edge running along z *inside* the label-3 bar.

    ``multiscale_labels_store`` fills label 3 over rows and columns
    ``[4, 8)`` at every z, so this track is fully enclosed: every ray that
    reaches it crosses a label surface first.
    """
    positions = np.array([[4.0, 6.0, 6.0], [12.0, 6.0, 6.0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)
    return GraphMemoryStore.from_arrays(positions, edges, name="tracks")


async def _build_scene(controller, reslice, labels_store, *, mode, depth_write):
    """A 3D scene holding translucent labels with a track inside them."""
    scene = controller.add_scene(dim="3d", name="scene")
    labels = controller.add_labels_multiscale(
        data=labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(
            force_level=1,
            opacity=0.5,
            transparency_mode=mode,
            depth_write=depth_write,
        ),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    graph = controller.add_graph(
        data=_track_store(),
        scene_id=scene.id,
        appearance=GraphAppearance(
            node_color=_TRACK_COLOR,
            node_size=10.0,
            node_size_space="screen",
            edge_color=_TRACK_COLOR,
            transparency_mode=mode,
            depth_write=depth_write,
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    return scene, labels, graph


def _render(controller, scene_id, *, extra_targets=_ALL_TARGETS):
    """Draw the live scene offscreen; return ``(int32 RGBA frame, renderer)``.

    Installs ``CellierBlender`` with every extra target by default, because
    the label shader writes ``normal`` and ``outline_id`` and the weighted
    ``FragmentOutput`` places them after accum, reveal and pick -- the
    combination that had never compiled in a test.
    """
    from rendercanvas.offscreen import RenderCanvas

    controller.update_background_field(scene_id, "visible", False)
    controller.fit_camera(scene_id)
    canvas_id = controller.get_canvas_ids(scene_id)[0]
    canvas_view = controller._render_manager._canvases[canvas_id]
    gfx_scene = canvas_view._get_scene_fn(scene_id)

    canvas = RenderCanvas(size=(_SIZE, _SIZE), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    if extra_targets:
        assert install_cellier_blender(renderer, extra_targets) is True

    errors: list[BaseException] = []

    def _draw() -> None:
        try:
            renderer.render(gfx_scene, canvas_view.camera)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)
            raise

    canvas.request_draw(_draw)
    image = canvas.draw()
    if errors:  # pragma: no cover - failure path
        raise RuntimeError(
            f"offscreen draw failed -- {type(errors[0]).__name__}: {errors[0]}"
        ) from errors[0]
    assert image is not None, "the offscreen canvas produced no frame"
    return np.asarray(image).astype(np.int32), renderer


def _render_both_orders(controller, scene, labels, graph):
    """Render with the graph drawn first, then with the labels drawn first.

    ``render_order`` lands on each visual's ``gfx.Group``, which pygfx turns
    into the ``group_order`` of every child -- ahead of the camera distance
    in the sort key, so it decides the order outright.
    """
    frames = []
    for labels_order, graph_order in ((1, 0), (0, 1)):
        controller.update_appearance_field(labels.id, "render_order", labels_order)
        controller.update_appearance_field(graph.id, "render_order", graph_order)
        frame, _renderer = _render(controller, scene.id)
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# The shaders compile and draw in weighted mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_targets", [(), _ALL_TARGETS], ids=["stock_blender", "cellier_blender"]
)
async def test_weighted_labels_and_graph_draw(
    offscreen_renderer,
    controller,
    reslice,
    multiscale_labels_store,
    extra_targets,
):
    """Labels and graph in weighted mode render through a weighted pass."""
    scene, _labels, _graph = await _build_scene(
        controller,
        reslice,
        multiscale_labels_store,
        mode="weighted_blend",
        depth_write=False,
    )
    frame, renderer = _render(controller, scene.id, extra_targets=extra_targets)

    # The accum target is only created when a weighted pass attaches it, so
    # its presence proves the objects really went through weighted blending
    # rather than silently falling back to another pass.
    assert renderer._blender.get_texture("accum") is not None
    assert np.count_nonzero(frame[..., 3]) > 0


# ---------------------------------------------------------------------------
# Order independence
# ---------------------------------------------------------------------------


async def test_blend_with_depth_write_depends_on_draw_order(
    offscreen_renderer, controller, reslice, multiscale_labels_store
):
    """The bug: with "blend" the frame depends on which object draws first.

    Drawn first, the labels write their surface depth and the enclosed track
    fails the depth test; drawn second, they blend over it.  This pins the
    harness's sensitivity -- if it stops failing here, the order test below
    proves nothing.
    """
    scene, labels, graph = await _build_scene(
        controller,
        reslice,
        multiscale_labels_store,
        mode="blend",
        depth_write=True,
    )
    graph_first, labels_first = _render_both_orders(controller, scene, labels, graph)

    assert np.count_nonzero(np.abs(graph_first - labels_first) > 32) > 0


async def test_weighted_blend_is_independent_of_draw_order(
    offscreen_renderer, controller, reslice, multiscale_labels_store
):
    """The fix: weighted blending without depth writes gives one frame.

    A tolerance of 2 absorbs rounding in the float accum and 8-bit reveal
    targets, whose accumulation order does change with the draw order.
    """
    scene, labels, graph = await _build_scene(
        controller,
        reslice,
        multiscale_labels_store,
        mode="weighted_blend",
        depth_write=False,
    )
    graph_first, labels_first = _render_both_orders(controller, scene, labels, graph)

    assert np.count_nonzero(graph_first[..., 3]) > 0
    np.testing.assert_allclose(graph_first, labels_first, atol=2)

    # The track must actually contribute, or the two frames could agree
    # merely because it is hidden in both.
    graph_node = controller._render_manager._scenes[scene.id].get_visual(graph.id)
    graph_node.node.visible = False
    labels_only, _renderer = _render(controller, scene.id)
    assert np.count_nonzero(np.abs(graph_first - labels_only) > 32) > 0
