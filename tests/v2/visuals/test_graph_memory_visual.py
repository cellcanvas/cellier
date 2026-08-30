"""GraphVisual, GraphAppearance and TrailConfig (D9, D11)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from cellier.visuals import GraphAppearance, GraphVisual, TrailConfig
from cellier.visuals._types import VisualType


def _visual(**kwargs) -> GraphVisual:
    return GraphVisual(name="graph", data_store_id="store-id", **kwargs)


# ── Appearance ─────────────────────────────────────────────────────────────


def test_graph_appearance_is_flat():
    """Every node_*/edge_* field emits on ``appearance.events``.

    The direct regression for the psygnal finding behind D9: nested
    ``EventedModel`` changes are *not* propagated to a parent's event
    group, and ``Controller._wire_appearance`` subscribes only to
    ``visual.appearance.events``.  A composed appearance would drop every
    one of these silently.
    """
    appearance = GraphAppearance()
    seen: list[str] = []
    appearance.events.connect(lambda info: seen.append(info.signal.name))

    appearance.node_color = (0.1, 0.2, 0.3, 1.0)
    appearance.node_size = 9.0
    appearance.node_size_space = "world"
    appearance.node_color_mode = "vertex"
    appearance.node_size_mode = "vertex"
    appearance.node_visible = False
    appearance.node_pick_write = False
    appearance.node_depth_compare = "<"
    appearance.edge_color = (0.4, 0.5, 0.6, 1.0)
    appearance.edge_thickness = 7.0
    appearance.edge_thickness_space = "world"
    appearance.edge_color_mode = "vertex"
    appearance.edge_visible = False
    appearance.edge_pick_write = False
    appearance.edge_depth_compare = "<="

    expected = [
        "node_color",
        "node_size",
        "node_size_space",
        "node_color_mode",
        "node_size_mode",
        "node_visible",
        "node_pick_write",
        "node_depth_compare",
        "edge_color",
        "edge_thickness",
        "edge_thickness_space",
        "edge_color_mode",
        "edge_visible",
        "edge_pick_write",
        "edge_depth_compare",
    ]
    assert seen == expected


def test_node_depth_compare_defaults_to_le():
    """Nodes win coplanar ties against edges via the depth test (D22).

    A deliberate divergence from ``BaseAppearance``'s ``"<"``:
    ``render_order`` was measured and does nothing for coplanar children.
    """
    appearance = GraphAppearance()
    assert appearance.node_depth_compare == "<="
    assert appearance.edge_depth_compare == "<"
    # And it is overridable like any other field.
    appearance.node_depth_compare = "<"
    assert appearance.node_depth_compare == "<"


def test_edge_color_default_is_dimmer_than_nodes():
    """So nodes read on top by default without any depth trickery."""
    appearance = GraphAppearance()
    assert appearance.edge_color[:3] < appearance.node_color[:3]


def test_appearance_inherits_base_fields():
    appearance = GraphAppearance()
    assert appearance.visible is True
    assert appearance.opacity == 1.0
    assert appearance.render_order == 0


# ── TrailConfig ────────────────────────────────────────────────────────────


def test_trail_config_defaults():
    """before/after default to 0.5; the fades fall back to them."""
    config = TrailConfig()
    assert config.before == 0.5
    assert config.after == 0.5
    assert config.fade is False
    assert config.fade_before is None
    assert config.fade_after is None
    assert config.min_alpha == 0.0
    assert config.resolved_fade_before == 0.5
    assert config.resolved_fade_after == 0.5


def test_trail_config_fade_fallback_follows_the_window():
    config = TrailConfig(before=10.0, after=3.0)
    assert config.resolved_fade_before == 10.0
    assert config.resolved_fade_after == 3.0

    config.fade_before = 2.0
    assert config.resolved_fade_before == 2.0
    assert config.resolved_fade_after == 3.0


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_trail_config_min_alpha_bounds(bad):
    with pytest.raises(ValidationError):
        TrailConfig(min_alpha=bad)


def test_trail_config_fields_emit():
    """Each field emits, which is what ``_wire_trail`` subscribes to."""
    config = TrailConfig()
    seen: list[str] = []
    config.events.connect(lambda info: seen.append(info.signal.name))

    config.before = 4.0
    config.after = 1.0
    config.fade = True
    config.fade_before = 2.0
    config.fade_after = 1.0
    config.min_alpha = 0.25

    assert seen == [
        "before",
        "after",
        "fade",
        "fade_before",
        "fade_after",
        "min_alpha",
    ]


# ── Visual ─────────────────────────────────────────────────────────────────


def test_graph_visual_defaults():
    visual = _visual()
    assert visual.visual_type == "graph_memory"
    assert visual.trail == {}
    assert visual.requires_camera_reslice is False
    assert isinstance(visual.appearance, GraphAppearance)


def test_trail_keys_are_axis_indices():
    visual = _visual(trail={0: TrailConfig(before=10.0, after=3.0, fade=True)})
    assert set(visual.trail) == {0}
    assert visual.trail[0].before == 10.0


def test_trail_dict_replacement_emits():
    """``visual.trail = {...}`` emits, which is what the rewire hangs off."""
    visual = _visual()
    seen: list[str] = []
    visual.events.connect(lambda info: seen.append(info.signal.name))
    visual.trail = {1: TrailConfig(before=2.0)}
    assert "trail" in seen


def test_graph_visual_serialization_roundtrip():
    visual = _visual(
        trail={0: TrailConfig(before=10.0, after=3.0, fade=True, min_alpha=0.15)},
        appearance=GraphAppearance(node_size=8.0, edge_color=(0.1, 0.2, 0.3, 1.0)),
    )
    restored = GraphVisual.model_validate_json(visual.model_dump_json())

    assert restored.appearance.node_size == 8.0
    assert restored.appearance.edge_color == (0.1, 0.2, 0.3, 1.0)
    assert restored.appearance.node_depth_compare == "<="
    assert restored.trail[0].before == 10.0
    assert restored.trail[0].fade is True
    assert restored.trail[0].min_alpha == 0.15
    assert np.allclose(restored.transform.matrix, visual.transform.matrix)


def test_visual_type_discriminator():
    """GraphVisual resolves through the VisualType union."""
    adapter = TypeAdapter(VisualType)
    restored = adapter.validate_json(_visual().model_dump_json())
    assert isinstance(restored, GraphVisual)
