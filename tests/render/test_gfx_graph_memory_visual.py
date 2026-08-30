"""GFXGraphMemoryVisual -- the pygfx compound (D1, D22)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pygfx as gfx
import pytest

from cellier.data.graph import GraphMemoryStore, GraphSliceRequest
from cellier.events._events import AppearanceChangedEvent
from cellier.render.shaders._alpha_modulated import (
    AlphaLineSegmentMaterial,
    AlphaPointsMaterial,
)
from cellier.render.visuals._graph_memory import GFXGraphMemoryVisual
from cellier.transform import AffineTransform
from cellier.visuals import GraphAppearance, GraphVisual


def _appearance_event(field, value):
    return AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=uuid4(),
        field_name=field,
        new_value=value,
        requires_reslice=False,
    )


def _store(**kwargs) -> GraphMemoryStore:
    """A 3-D chain: four nodes along the diagonal, three edges."""
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    return GraphMemoryStore.from_arrays(positions, edges, **kwargs)


def _visual(store, appearance=None, trail=None) -> GFXGraphMemoryVisual:
    model = GraphVisual(
        name="test",
        data_store_id=str(store.id),
        appearance=appearance or GraphAppearance(),
        trail=trail or {},
    )
    return GFXGraphMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )


def _commit(visual, store, displayed=(0, 1, 2), sliced=None, extents=None, fades=None):
    shared = uuid4()
    request = GraphSliceRequest(
        slice_request_id=shared,
        chunk_request_id=shared,
        scale_index=0,
        displayed_axes=displayed,
        slice_indices=dict(sliced or {}),
        extents=dict(extents or {}),
        fades=dict(fades or {}),
    )
    data = asyncio.run(store.get_data(request))
    if len(displayed) == 2:
        visual.on_data_ready_2d([(request, data)])
    else:
        visual.on_data_ready([(request, data)])
    return data


# ── Structure ──────────────────────────────────────────────────────────────


def test_group_structure():
    """Two children, and the node material wins coplanar ties (D22)."""
    visual = _visual(_store())

    assert isinstance(visual.node, gfx.Group)
    assert set(visual.node.children) == {visual.node_points, visual.node_edges}
    assert isinstance(visual.node_points, gfx.Points)
    assert isinstance(visual.node_edges, gfx.Line)
    assert isinstance(visual._node_material, AlphaPointsMaterial)
    assert isinstance(visual._edge_material, AlphaLineSegmentMaterial)
    assert visual._node_material.depth_compare == "<="
    assert visual._edge_material.depth_compare == "<"


def test_single_node_for_both_modes():
    visual = _visual(_store())
    assert visual.node_2d is visual.node
    assert visual.node_3d is visual.node
    assert visual.get_node_for_dims((0, 1, 2)) is visual.node
    assert visual.get_node_for_dims((1, 2)) is visual.node


def test_pick_write_enabled_by_default():
    visual = _visual(_store())
    assert visual._node_material.pick_write is True
    assert visual._edge_material.pick_write is True


def test_sub_visual_toggles_are_independent():
    """D16: node_visible / edge_visible nest under the group's visible."""
    visual = _visual(_store())
    visual.on_appearance_changed(_appearance_event("node_visible", False))
    assert visual.node_points.visible is False
    assert visual.node_edges.visible is True
    assert visual.node.visible is True


# ── Coordinate convention ──────────────────────────────────────────────────


def test_3d_coordinate_convention():
    """Both buffers reverse (z, y, x) -> pygfx (x, y, z)."""
    positions = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    store = GraphMemoryStore.from_arrays(positions, np.array([[0, 1]], dtype=np.int32))
    visual = _visual(store)
    _commit(visual, store, displayed=(0, 1, 2))

    assert np.allclose(
        visual.node_points.geometry.positions.data, positions[:, [2, 1, 0]]
    )
    # The edge buffer is the same two nodes, in vertex-pair order.
    assert np.allclose(
        visual.node_edges.geometry.positions.data, positions[:, [2, 1, 0]]
    )


def test_2d_coordinate_convention():
    """Both buffers zero-pad then swap so (row, col) -> (x=col, y=row, z=0)."""
    positions = np.array([[0.0, 1.0, 2.0], [0.0, 4.0, 5.0]], dtype=np.float32)
    store = GraphMemoryStore.from_arrays(positions, np.array([[0, 1]], dtype=np.int32))
    visual = _visual(store)
    _commit(visual, store, displayed=(1, 2), sliced={0: 0})

    expected = np.array([[2.0, 1.0, 0.0], [5.0, 4.0, 0.0]], dtype=np.float32)
    assert np.allclose(visual.node_points.geometry.positions.data, expected)
    assert np.allclose(visual.node_edges.geometry.positions.data, expected)


# ── Empty slices ───────────────────────────────────────────────────────────


def test_empty_slice_material_swap():
    """Nodes and edges swap independently."""
    store = _store()
    visual = _visual(store)

    _commit(visual, store, displayed=(1, 2), sliced={0: 99})
    assert visual.node_points.material is visual._empty_node_material
    assert visual.node_edges.material is visual._empty_edge_material

    _commit(visual, store, displayed=(0, 1, 2))
    assert visual.node_points.material is visual._node_material
    assert visual.node_edges.material is visual._edge_material


def test_isolated_nodes_swap_only_the_edge_material():
    """Nodes present, no edges -- the common case for isolated nodes."""
    store = GraphMemoryStore.from_arrays(
        np.zeros((3, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    )
    visual = _visual(store)
    _commit(visual, store, displayed=(0, 1, 2))

    assert visual.node_points.material is visual._node_material
    assert visual.node_edges.material is visual._empty_edge_material


# ── Alpha buffers ──────────────────────────────────────────────────────────


def test_alpha_buffer_all_ones_without_fade():
    """No fade configured -> uploaded alphas are all 1.0.

    Uploading ones rather than swapping to a stock material keeps one
    pipeline; a swap would rebuild it on every fade toggle.
    """
    store = _store()
    visual = _visual(store)
    _commit(visual, store, displayed=(0, 1, 2))

    node_alphas = visual.node_points.geometry.alphas.data
    edge_alphas = visual.node_edges.geometry.alphas.data
    assert node_alphas.shape == (4,)
    assert edge_alphas.shape == (6,)
    assert np.all(node_alphas == 1.0)
    assert np.all(edge_alphas == 1.0)


def test_alpha_buffer_carries_the_fade():
    store = _store()
    visual = _visual(store)
    _commit(
        visual,
        store,
        displayed=(1, 2),
        sliced={0: 3},
        extents={0: (3.0, 0.0)},
        fades={0: (3.0, 3.0, 0.0)},
    )
    alphas = visual.node_points.geometry.alphas.data
    assert np.isclose(alphas.max(), 1.0)
    assert alphas.min() < 1.0


def test_alpha_survives_geometry_replacement():
    """Two reslices with different surviving counts keep the fade correct.

    This is the per-frame path: a binding that went stale across a
    geometry replacement would show as a fade that stops updating.
    """
    store = _store()
    visual = _visual(store)

    _commit(
        visual,
        store,
        displayed=(1, 2),
        sliced={0: 3},
        extents={0: (3.0, 0.0)},
        fades={0: (3.0, 3.0, 0.0)},
    )
    first = np.asarray(visual.node_points.geometry.alphas.data).copy()
    assert first.shape == (4,)

    _commit(
        visual,
        store,
        displayed=(1, 2),
        sliced={0: 3},
        extents={0: (1.0, 0.0)},
        fades={0: (1.0, 1.0, 0.0)},
    )
    second = np.asarray(visual.node_points.geometry.alphas.data)
    assert second.shape == (2,)
    assert np.isclose(second.max(), 1.0)

    # And back to no fade at all: an all-ones buffer of the new length.
    _commit(visual, store, displayed=(0, 1, 2))
    third = np.asarray(visual.node_points.geometry.alphas.data)
    assert third.shape == (4,)
    assert np.all(third == 1.0)


# ── color_mode is honoured, never inferred (D20) ───────────────────────────


def test_declared_uniform_survives_store_colors():
    """A store carrying colours does not flip a declared uniform mode."""
    colors = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1))
    store = _store(node_colors=colors)
    visual = _visual(store, appearance=GraphAppearance(node_color_mode="uniform"))
    _commit(visual, store, displayed=(0, 1, 2))

    assert visual._node_material.color_mode == "uniform"
    assert visual._node_color_mode == "uniform"


def test_declared_vertex_without_colors_raises():
    """The mismatch is a misconfiguration, not a silent fallback."""
    store = _store()
    visual = _visual(store, appearance=GraphAppearance(node_color_mode="vertex"))
    with pytest.raises(ValueError, match="node_color_mode='vertex'"):
        _commit(visual, store, displayed=(0, 1, 2))


def test_declared_vertex_edges_without_colors_raises():
    store = _store()
    visual = _visual(store, appearance=GraphAppearance(edge_color_mode="vertex"))
    with pytest.raises(ValueError, match="edge_color_mode='vertex'"):
        _commit(visual, store, displayed=(0, 1, 2))


def test_empty_slice_does_not_raise_on_declared_vertex():
    """An empty slice carries no colours by construction; that is not a bug."""
    store = _store()
    visual = _visual(store, appearance=GraphAppearance(node_color_mode="vertex"))
    _commit(visual, store, displayed=(1, 2), sliced={0: 99})
    assert visual._nodes_empty


def test_declared_vertex_node_size_mode_reaches_the_material():
    """node_size_mode is a declaration, applied at construction.

    The graph visual previously *inferred* this from the data, which is the
    pattern D20 removed for colour.  It is now declared on the appearance
    like every other mode field.
    """
    store = _store()
    visual = _visual(store, appearance=GraphAppearance(node_size_mode="vertex"))
    assert visual._node_material.size_mode == "vertex"
    assert visual._node_size_mode == "vertex"


def test_declared_uniform_survives_store_node_sizes():
    """Store carries sizes, appearance says uniform -> stays uniform."""
    sizes = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    store = _store(node_sizes=sizes)
    visual = _visual(store, appearance=GraphAppearance(node_size_mode="uniform"))
    _commit(visual, store, displayed=(0, 1, 2))

    assert visual._node_material.size_mode == "uniform"
    assert visual._node_size_mode == "uniform"


def test_declared_vertex_node_size_without_sizes_raises():
    """The mismatch is a misconfiguration, not a silent fallback."""
    store = _store()
    visual = _visual(store, appearance=GraphAppearance(node_size_mode="vertex"))
    with pytest.raises(ValueError, match="node_size_mode='vertex'"):
        _commit(visual, store, displayed=(0, 1, 2))


def test_empty_slice_does_not_raise_on_declared_vertex_size():
    """An empty slice carries no sizes by construction; that is not a bug."""
    store = _store()
    visual = _visual(store, appearance=GraphAppearance(node_size_mode="vertex"))
    _commit(visual, store, displayed=(1, 2), sliced={0: 99})
    assert visual._nodes_empty


def test_node_size_mode_appearance_change_reaches_the_material():
    visual = _visual(_store())
    assert visual._node_material.size_mode == "uniform"
    visual.on_appearance_changed(_appearance_event("node_size_mode", "vertex"))
    assert visual._node_material.size_mode == "vertex"
    assert visual._node_size_mode == "vertex"


# ── Appearance events ──────────────────────────────────────────────────────


def test_appearance_events_route_to_the_right_child():
    visual = _visual(_store())

    visual.on_appearance_changed(_appearance_event("node_color", (0.1, 0.2, 0.3, 1.0)))
    visual.on_appearance_changed(_appearance_event("node_size", 11.0))
    visual.on_appearance_changed(_appearance_event("edge_color", (0.4, 0.5, 0.6, 1.0)))
    visual.on_appearance_changed(_appearance_event("edge_thickness", 7.0))
    visual.on_appearance_changed(_appearance_event("node_depth_compare", "<"))

    assert tuple(visual._node_material.color) == pytest.approx((0.1, 0.2, 0.3, 1.0))
    assert visual._node_material.size == 11.0
    assert tuple(visual._edge_material.color) == pytest.approx((0.4, 0.5, 0.6, 1.0))
    assert visual._edge_material.thickness == 7.0
    assert visual._node_material.depth_compare == "<"


def test_space_appearance_changes_reach_the_materials():
    """node_size_space / edge_thickness_space are applied live."""
    visual = _visual(_store())
    assert visual._node_material.size_space == "screen"
    assert visual._edge_material.thickness_space == "screen"

    visual.on_appearance_changed(_appearance_event("node_size_space", "world"))
    visual.on_appearance_changed(_appearance_event("edge_thickness_space", "world"))

    assert visual._node_material.size_space == "world"
    assert visual._edge_material.thickness_space == "world"


def test_shared_appearance_fields_reach_both_materials():
    visual = _visual(_store())
    visual.on_appearance_changed(_appearance_event("opacity", 0.25))
    visual.on_appearance_changed(_appearance_event("depth_write", False))

    assert visual._node_material.opacity == 0.25
    assert visual._edge_material.opacity == 0.25
    assert visual._node_material.depth_write is False
    assert visual._edge_material.depth_write is False


# ── Rendered output ────────────────────────────────────────────────────────


def test_nodes_draw_over_coplanar_edges(offscreen_renderer):
    """The direct regression for D22.

    Under the original design -- rely on insertion order, or set
    ``render_order`` -- the centre pixel came back green (the edge).  It is
    the depth compare that decides, and nodes win only because
    ``node_depth_compare`` is ``"<="``.
    """
    size = (128, 128)
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000000"))

    positions = np.array([[-40, 0, 0], [40, 0, 0]], dtype=np.float32)
    edges = gfx.Line(
        gfx.Geometry(positions=positions, alphas=np.ones(2, dtype=np.float32)),
        AlphaLineSegmentMaterial(
            thickness=20, color=(0.0, 1.0, 0.0, 1.0), depth_compare="<"
        ),
    )
    nodes = gfx.Points(
        gfx.Geometry(
            positions=np.zeros((1, 3), dtype=np.float32),
            alphas=np.ones(1, dtype=np.float32),
        ),
        AlphaPointsMaterial(size=40, color=(1.0, 0.0, 0.0, 1.0), depth_compare="<="),
    )
    group = gfx.Group()
    group.add(edges)
    group.add(nodes)
    scene.add(group)

    image = offscreen_renderer(scene, gfx.OrthographicCamera(*size), size)
    centre = image[size[1] // 2, size[0] // 2]
    assert centre[0] > 200, f"node red should win the tie, got {centre}"
    assert centre[1] < 60, f"edge green should lose the tie, got {centre}"


def test_scene_manager_resolves_both_children():
    """``get_visual_id_for_node`` walks the parent chain to the group."""
    from cellier.render.scene_manager import SceneManager

    store = _store()
    visual = _visual(store)
    manager = SceneManager(scene_id=uuid4())
    manager.add_visual(visual, displayed_axes=(0, 1, 2))

    assert manager.get_visual_id_for_node(visual.node) == visual.visual_model_id
    assert manager.get_visual_id_for_node(visual.node_points) == visual.visual_model_id
    assert manager.get_visual_id_for_node(visual.node_edges) == visual.visual_model_id
    assert (
        manager.get_visual_id_for_node(
            gfx.Points(
                gfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
                gfx.PointsMaterial(),
            )
        )
        is None
    )
