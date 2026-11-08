"""``GraphMemoryStore.from_geff``.

The fixture files are *written* by each test rather than committed: a geff
store is a zarr tree of many small files, and geff is needed to read one
back anyway, so generating them keeps the repo clean and the fixtures
readable at the point of use.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data.graph import GraphMemoryStore
from tests._v2 import data_system

try:  # pragma: no cover - import probe
    import geff as _geff
    import networkx as _nx
except ImportError:  # pragma: no cover
    _geff = None
    _nx = None

requires_geff = pytest.mark.skipif(_geff is None, reason="geff not installed")


def _write_lineage(
    path,
    *,
    node_ids=(0, 1, 2, 3),
    scales=None,
    offsets=None,
    directed=True,
    axis_types=("time", "space", "space", "space"),
):
    """Write a small 4-D tzyx lineage and return ``(path, positions, edges)``.

    Two tracks branching from node 0, which is the shape a tracking file
    actually has: every edge spans ``t`` -> ``t + 1``.
    """
    coords = [
        (0.0, 1.0, 2.0, 3.0),
        (1.0, 1.0, 2.0, 4.0),
        (1.0, 5.0, 6.0, 7.0),
        (2.0, 5.0, 6.0, 8.0),
    ]
    graph = _nx.DiGraph() if directed else _nx.Graph()
    for node_id, (t, z, y, x) in zip(node_ids, coords):
        graph.add_node(node_id, t=t, z=z, y=y, x=x, score=float(node_id))
    edge_pairs = [(0, 1), (0, 2), (2, 3)]
    for u, v in edge_pairs:
        graph.add_edge(node_ids[u], node_ids[v], weight=float(u + v))

    kwargs = {}
    if scales is not None:
        kwargs["axis_scales"] = list(scales)
    if offsets is not None:
        kwargs["axis_offset"] = list(offsets)
    _geff.write(
        graph,
        path,
        axis_names=["t", "z", "y", "x"],
        axis_types=None if axis_types is None else list(axis_types),
        **kwargs,
    )
    return (
        path,
        np.asarray(coords, dtype=np.float32),
        np.asarray(edge_pairs, dtype=np.int32),
    )


@requires_geff
def test_from_geff_round_trip(tmp_path):
    """Positions, edges and node ids match the file."""
    path, positions, edges = _write_lineage(tmp_path / "lineage.geff")
    store = GraphMemoryStore.from_geff(path)

    assert np.allclose(store.positions, positions)
    assert np.array_equal(store.edges, edges)
    assert np.array_equal(store.node_ids, np.array([0, 1, 2, 3]))
    assert store.ndim == 4
    assert store.directed is True


@requires_geff
def test_from_geff_axis_order(tmp_path):
    """Axes land in file order, with ``t`` included as axis 0.

    Time is an ordinary sliceable dimension; nothing special-cases it.
    """
    path, positions, _ = _write_lineage(tmp_path / "lineage.geff")
    store = GraphMemoryStore.from_geff(path)

    assert [axis.name for axis in store.axes] == ["t", "z", "y", "x"]
    assert np.allclose(store.positions[:, 0], positions[:, 0])


@requires_geff
def test_from_geff_axis_names_override(tmp_path):
    """``axis_names`` selects and reorders the loaded axes."""
    path, positions, _ = _write_lineage(tmp_path / "lineage.geff")
    store = GraphMemoryStore.from_geff(path, axis_names=["z", "y", "x"])

    assert store.ndim == 3
    assert [axis.name for axis in store.axes] == ["z", "y", "x"]
    assert np.allclose(store.positions, positions[:, 1:])


@requires_geff
def test_from_geff_builds_a_system_from_the_file_axes(tmp_path):
    """Names and types come from the file, and the store owns the system."""
    path, _, _ = _write_lineage(tmp_path / "lineage.geff")
    store = GraphMemoryStore.from_geff(path)

    system = store.data_coordinate_system
    assert system.axis_names() == ("t", "z", "y", "x")
    assert [axis.axis_type for axis in system.axes] == [
        "time",
        "space",
        "space",
        "space",
    ]
    assert [axis.sampling for axis in system.axes] == ["continuous"] * 4
    assert system.datastore_id == store.id
    assert len(store.level_transforms) == 1


@requires_geff
def test_from_geff_system_follows_the_selected_axes(tmp_path):
    path, _, _ = _write_lineage(tmp_path / "lineage.geff")
    store = GraphMemoryStore.from_geff(path, axis_names=["z", "y", "x"])
    assert store.data_coordinate_system.axis_names() == ("z", "y", "x")


@requires_geff
def test_from_geff_untyped_axes_take_the_name_rule(tmp_path):
    """geff makes ``type`` optional; an unset one is typed from its name."""
    path, _, _ = _write_lineage(tmp_path / "untyped.geff", axis_types=None)
    store = GraphMemoryStore.from_geff(path)
    assert [axis.axis_type for axis in store.data_coordinate_system.axes] == [
        "time",
        "space",
        "space",
        "space",
    ]


@requires_geff
def test_from_geff_uses_a_passed_system(tmp_path):
    """A caller-built system replaces the metadata-derived one outright."""
    path, _, _ = _write_lineage(tmp_path / "lineage.geff")
    system = data_system(("t", "z", "y", "x"), sampling="discrete", name="tracks")
    store = GraphMemoryStore.from_geff(path, data_coordinate_system=system)

    assert store.data_coordinate_system.id == system.id
    assert store.id == system.datastore_id
    assert [axis.sampling for axis in store.data_coordinate_system.axes] == [
        "discrete"
    ] * 4


@requires_geff
def test_from_geff_unknown_axis_name_raises(tmp_path):
    path, _, _ = _write_lineage(tmp_path / "lineage.geff")
    with pytest.raises(KeyError, match="channel"):
        GraphMemoryStore.from_geff(path, axis_names=["channel"])


@requires_geff
def test_from_geff_retains_props(tmp_path):
    """Non-position props are retained and unconsumed."""
    path, _, _ = _write_lineage(tmp_path / "lineage.geff")
    store = GraphMemoryStore.from_geff(path)

    assert "score" in store.node_props
    assert np.allclose(store.node_props["score"], [0.0, 1.0, 2.0, 3.0])
    assert "weight" in store.edge_props
    # Axis columns are not duplicated into node_props.
    assert not {"t", "z", "y", "x"} & set(store.node_props)


@requires_geff
def test_from_geff_builds_transform(tmp_path):
    """The test that would have caught a silent 4x z error.

    A typical anisotropic light-sheet volume: 4 um z spacing, 0.26 um in
    plane, and a 10 um z offset.
    """
    path, _, _ = _write_lineage(
        tmp_path / "scaled.geff",
        scales=[1.0, 4.0, 0.26, 0.26],
        offsets=[0.0, 10.0, 0.0, 0.0],
    )
    store = GraphMemoryStore.from_geff(path)

    # The store keeps the file's raw numbers.  It cannot name the scene's
    # world, so the transform is built by the controller.
    assert store.axis_scales == (1.0, 4.0, 0.26, 0.26)
    assert store.axis_offsets == (0.0, 10.0, 0.0, 0.0)


@requires_geff
def test_from_geff_identity_when_axes_unscaled(tmp_path):
    """Null scale/offset give identity, not zeros."""
    path, _, _ = _write_lineage(tmp_path / "plain.geff")
    store = GraphMemoryStore.from_geff(path)

    assert store.axis_scales == (1.0, 1.0, 1.0, 1.0)
    assert store.axis_offsets == (0.0, 0.0, 0.0, 0.0)


@requires_geff
def test_from_geff_has_no_transform_param(tmp_path):
    """The file is the sole source of the transform: no ``transform=``."""
    path, _, _ = _write_lineage(tmp_path / "lineage.geff")
    with pytest.raises(TypeError, match="transform"):
        GraphMemoryStore.from_geff(path, transform=object())


@requires_geff
def test_from_geff_edge_ids_become_rows(tmp_path):
    """Original id pairs arrive; row indices are stored."""
    path, _, edges = _write_lineage(
        tmp_path / "sparse.geff", node_ids=(10, 20, 45, 900)
    )
    store = GraphMemoryStore.from_geff(path)

    assert np.array_equal(store.node_ids, np.array([10, 20, 45, 900]))
    # Rows, not ids: the largest endpoint is 3, not 900.
    assert np.array_equal(store.edges, edges)
    assert int(store.edges.max()) == store.n_nodes - 1
    # And the round trip back to ids still works.
    assert np.array_equal(store.ids_for_rows([0, 3]), np.array([10, 900]))


@requires_geff
def test_from_geff_undirected(tmp_path):
    path, _, _ = _write_lineage(tmp_path / "undirected.geff", directed=False)
    store = GraphMemoryStore.from_geff(path)
    assert store.directed is False


@requires_geff
def test_from_geff_node_color_and_size_props(tmp_path):
    """The named-prop shortcut resolves into the appearance arrays."""
    path, _, _ = _write_lineage(tmp_path / "lineage.geff")
    store = GraphMemoryStore.from_geff(path, node_size_prop="score")

    assert store.node_sizes is not None
    assert np.allclose(store.node_sizes, [0.0, 1.0, 2.0, 3.0])
    assert store.node_size_mode == "vertex"


@requires_geff
def test_from_geff_missing_prop_raises(tmp_path):
    path, _, _ = _write_lineage(tmp_path / "lineage.geff")
    with pytest.raises(KeyError, match="radius"):
        GraphMemoryStore.from_geff(path, node_size_prop="radius")


@requires_geff
async def test_from_geff_store_slices(tmp_path, make_request):
    """End to end: a geff-loaded store slices on t and renders edges."""
    path, _, _ = _write_lineage(tmp_path / "lineage.geff")
    store = GraphMemoryStore.from_geff(path)

    data = await store.get_data(make_request(displayed=(1, 2, 3), sliced={0: 1}))
    assert np.array_equal(data.original_node_rows, np.array([1, 2]))
    # Every edge touching t=1 survives (either endpoint): (0,1), (0,2), (2,3).
    assert data.original_edge_rows.shape[0] == 3


def test_from_geff_without_extra_raises(tmp_path, monkeypatch):
    """With geff unimportable the error names the extra.

    Simulated through ``sys.modules`` so this runs even when geff *is*
    installed -- otherwise the one message a user without the extra ever
    sees would be the one message never under test.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "geff" or name.startswith("geff."):
            raise ImportError("No module named 'geff'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    monkeypatch.delitem(__import__("sys").modules, "geff", raising=False)

    with pytest.raises(ImportError, match=r"cellier\[graph\]"):
        GraphMemoryStore.from_geff(tmp_path / "nope.geff")
