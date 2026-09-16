"""Tests for GFXMultiscaleImageVisual lazy-init visibility behaviour.

Regression tests for the bug where a node built lazily on first entry into a
mode defaulted to visible=True even when on_visibility_changed(visible=False)
had already been called before it existed.  The visual's per-mode groups
exist from construction; its slots build their nodes lazily, and the group
carries the visibility.
"""

from __future__ import annotations

import uuid

from cellier.events._events import VisualVisibilityChangedEvent
from cellier.render.visuals._image import GFXMultiscaleImageVisual
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageSingleAppearance,
    MultiscaleImageVisual,
)
from tests._v2 import level_transforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(visible: bool = True) -> MultiscaleImageVisual:
    """Minimal dual-mode MultiscaleImageVisual model."""
    return MultiscaleImageVisual(
        name="vol",
        data_store_id=str(uuid.uuid4()),
        level_transforms=level_transforms(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        appearance=MultiscaleImageAppearance(visible=visible),
        single=MultiscaleImageSingleAppearance(color_map="grays", clim=(0.0, 255.0)),
    )


def _make_gfx_visual_starting_2d(
    model: MultiscaleImageVisual,
) -> GFXMultiscaleImageVisual:
    """Build a dual-mode GFX visual starting in 2D (node_3d deferred)."""
    level_shapes = [(8, 8, 8), (4, 4, 4)]
    return GFXMultiscaleImageVisual.from_cellier_model(
        model=model,
        level_shapes=level_shapes,
        render_modes={"2d", "3d"},
        displayed_axes=(1, 2),  # 2D start — node_3d is not built yet
    )


def _make_gfx_visual_starting_3d(
    model: MultiscaleImageVisual,
) -> GFXMultiscaleImageVisual:
    """Build a dual-mode GFX visual starting in 3D (node_2d deferred)."""
    level_shapes = [(8, 8, 8), (4, 4, 4)]
    return GFXMultiscaleImageVisual.from_cellier_model(
        model=model,
        level_shapes=level_shapes,
        render_modes={"2d", "3d"},
        displayed_axes=(0, 1, 2),  # 3D start — node_2d is not built yet
    )


def _visibility_event(visible: bool) -> VisualVisibilityChangedEvent:
    return VisualVisibilityChangedEvent(
        source_id=uuid.uuid4(),
        visual_id=uuid.uuid4(),
        visible=visible,
    )


# ---------------------------------------------------------------------------
# from_cellier_model: initial visible=False
# ---------------------------------------------------------------------------


def test_from_cellier_model_visible_false_2d_start():
    """Nodes built at construction time are hidden when model.visible=False."""
    model = _make_model(visible=False)
    gfx = _make_gfx_visual_starting_2d(model)

    assert gfx.node_2d is not None, "node_2d should be built when starting in 2D"
    assert gfx.node_2d.visible is False
    assert gfx.slots[0].node_3d is None  # deferred -- not yet built


def test_from_cellier_model_visible_false_3d_start():
    """Nodes built at construction time are hidden when model.visible=False."""
    model = _make_model(visible=False)
    gfx = _make_gfx_visual_starting_3d(model)

    assert gfx.node_3d is not None, "node_3d should be built when starting in 3D"
    assert gfx.node_3d.visible is False
    assert gfx.slots[0].node_2d is None  # deferred -- not yet built


def test_from_cellier_model_visible_true_is_default():
    """Nodes built at construction time are visible when model.visible=True."""
    model = _make_model(visible=True)
    gfx = _make_gfx_visual_starting_2d(model)

    assert gfx.node_2d is not None
    assert gfx.node_2d.visible is True


# ---------------------------------------------------------------------------
# Lazy init respects _visible set before the node existed
# ---------------------------------------------------------------------------


def test_lazy_init_3d_respects_hidden_state():
    """node_3d created by _lazy_init_3d is hidden when visibility was set False first.

    Regression: previously node_3d defaulted to visible=True on lazy init,
    causing a hidden single-channel visual to bleed through into 3D renders.
    """
    model = _make_model(visible=True)
    gfx = _make_gfx_visual_starting_2d(model)

    assert gfx.slots[0].node_3d is None  # not built yet

    # Mark as hidden before 3D has been lazily initialized.
    gfx.on_visibility_changed(_visibility_event(visible=False))
    assert gfx._visible is False
    assert gfx.slots[0].node_3d is None  # still not built

    # Trigger lazy 3D init via get_node_for_dims.
    node = gfx.get_node_for_dims((0, 1, 2))

    assert node is gfx.node_3d
    assert gfx.slots[0].node_3d is not None
    assert node.visible is False, (
        "node_3d must inherit _visible=False from lazy init, not default to True"
    )


def test_lazy_init_2d_respects_hidden_state():
    """node_2d created by _lazy_init_2d is hidden when visibility was set False."""
    model = _make_model(visible=True)
    gfx = _make_gfx_visual_starting_3d(model)

    assert gfx.slots[0].node_2d is None  # not built yet

    gfx.on_visibility_changed(_visibility_event(visible=False))
    assert gfx._visible is False
    assert gfx.slots[0].node_2d is None

    node = gfx.get_node_for_dims((1, 2))

    assert node is gfx.node_2d
    assert gfx.slots[0].node_2d is not None
    assert node.visible is False


# ---------------------------------------------------------------------------
# on_visibility_changed updates _visible and already-built nodes
# ---------------------------------------------------------------------------


def test_on_visibility_changed_updates_visible_shadow():
    model = _make_model(visible=True)
    gfx = _make_gfx_visual_starting_2d(model)

    gfx.on_visibility_changed(_visibility_event(visible=False))
    assert gfx._visible is False

    gfx.on_visibility_changed(_visibility_event(visible=True))
    assert gfx._visible is True


def test_on_visibility_changed_updates_existing_nodes():
    model = _make_model(visible=True)
    gfx = _make_gfx_visual_starting_2d(model)

    # Build node_3d while visible.
    gfx.get_node_for_dims((0, 1, 2))
    assert gfx.node_3d.visible is True
    assert gfx.node_2d.visible is True

    gfx.on_visibility_changed(_visibility_event(visible=False))

    assert gfx.node_3d.visible is False
    assert gfx.node_2d.visible is False

    gfx.on_visibility_changed(_visibility_event(visible=True))

    assert gfx.node_3d.visible is True
    assert gfx.node_2d.visible is True


# ---------------------------------------------------------------------------
# Round-trip: hide → switch dims → re-show
# ---------------------------------------------------------------------------


def test_hide_switch_to_3d_then_show():
    """Full round-trip matching the oz-viewer SC-hidden-in-MC use case."""
    model = _make_model(visible=True)
    gfx = _make_gfx_visual_starting_2d(model)

    # Simulate: MC mode hides the SC visual while still in 2D.
    gfx.on_visibility_changed(_visibility_event(visible=False))

    # Switch to 3D — triggers lazy init.
    node_3d = gfx.get_node_for_dims((0, 1, 2))
    assert node_3d.visible is False, "SC visual must stay hidden in 3D"

    # Switch back to 2D.
    node_2d = gfx.get_node_for_dims((1, 2))
    assert node_2d.visible is False, "SC visual must stay hidden back in 2D"

    # Re-show (switching back from MC to SC).
    gfx.on_visibility_changed(_visibility_event(visible=True))
    assert gfx.node_3d.visible is True
    assert gfx.node_2d.visible is True
