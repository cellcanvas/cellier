"""``Viewer.add_graph`` and ``OrthoViewer.add_graph``."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.convenience._ortho_viewer import OrthoViewer
from cellier.transform import AffineTransform
from cellier.visuals import GraphAppearance, GraphVisual, TrailConfig

_PANELS = {"xy", "xz", "yz", "vol"}


def _visual_ids(viewer: Viewer) -> set:
    return {v.id for v in viewer.scene.visuals}


def test_add_graph_from_store(graph_store):
    viewer = Viewer(("z", "y", "x"), dim="3d")
    visual = viewer.add_graph(graph_store, appearance=GraphAppearance(), name="g")

    assert isinstance(visual, GraphVisual)
    assert visual.name == "g"
    assert visual.id in _visual_ids(viewer)


def test_add_graph_defaults_appearance_when_none(graph_store):
    viewer = Viewer(("z", "y", "x"), dim="3d")
    visual = viewer.add_graph(graph_store)

    assert visual.id in _visual_ids(viewer)
    assert isinstance(visual.appearance, GraphAppearance)
    assert visual.appearance.node_depth_compare == "<="


def test_add_graph_dict_appearance(graph_store):
    """A plain dict coerces to GraphAppearance."""
    viewer = Viewer(("z", "y", "x"), dim="3d")
    visual = viewer.add_graph(
        graph_store,
        appearance={"node_size": 9.0, "edge_color": (0.1, 0.2, 0.3, 1.0)},
        name="g",
    )

    assert isinstance(visual.appearance, GraphAppearance)
    assert visual.appearance.node_size == 9.0
    assert visual.appearance.edge_color == (0.1, 0.2, 0.3, 1.0)


def test_add_graph_with_trail(graph_store):
    """The trail reaches the visual."""
    viewer = Viewer(("z", "y", "x"), dim="3d")
    visual = viewer.add_graph(
        graph_store, trail={0: TrailConfig(before=3.0, after=1.0, fade=True)}
    )

    assert set(visual.trail) == {0}
    assert visual.trail[0].before == 3.0
    assert visual.trail[0].fade is True


def test_add_graph_rejects_out_of_range_trail_axis(graph_store):
    viewer = Viewer(("z", "y", "x"), dim="3d")
    with pytest.raises(ValueError, match="out of range"):
        viewer.add_graph(graph_store, trail={5: TrailConfig()})


def test_add_graph_by_store_uuid(graph_store):
    """The already-registered-store branch."""
    viewer = Viewer(("z", "y", "x"), dim="3d")
    first = viewer.add_graph(graph_store, name="a")
    second = viewer.add_graph(graph_store.id, name="b")

    assert first.data_store_id == second.data_store_id
    assert len(viewer.controller._model.data.stores) == 1


def test_add_graph_explicit_transform_wins(graph_store):
    viewer = Viewer(("z", "y", "x"), dim="3d")
    transform = AffineTransform.from_scale((2.0, 2.0, 2.0))
    visual = viewer.add_graph(graph_store, transform=transform)

    assert np.allclose(np.diag(visual.transform.matrix)[:3], [2.0, 2.0, 2.0])


def test_ortho_add_graph_fans_out(graph_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_graph(graph_store, appearance=GraphAppearance(), name="g")

    assert set(visuals) == _PANELS
    assert {v.name for v in visuals.values()} == {f"g_{k}" for k in _PANELS}
    assert len(ortho.controller._model.data.stores) == 1
    for scene in ortho.scenes.values():
        assert len(scene.visuals) == 1


def test_ortho_add_graph_dict_appearance_and_trail(graph_store):
    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_graph(
        graph_store,
        appearance={"node_size": 7.0},
        name="g",
        trail={0: TrailConfig(before=2.0)},
    )

    for visual in visuals.values():
        assert isinstance(visual.appearance, GraphAppearance)
        assert visual.appearance.node_size == 7.0
        assert visual.trail[0].before == 2.0
