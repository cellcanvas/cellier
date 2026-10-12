"""Render + appearance tests for ``GFXLabelMemoryVisual`` (Phase 3).

Exercises the in-memory label commit path (``on_data_ready`` texture upload +
material build) and the colormap/appearance update handlers that the
planning-only tests never reach, reading pixels back through the offscreen
harness.  Uses the shared fixtures (``controller``, ``render_scene``,
``reslice``, ``labels_volume``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest

from cellier.events._events import (
    AppearanceChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.visuals._label_memory import InMemoryLabelsAppearance

if TYPE_CHECKING:
    from cellier.render.visuals._label_memory import GFXLabelMemoryVisual


def _gfx_visual(controller, scene_id, visual_id) -> GFXLabelMemoryVisual:
    return controller._render_manager._scenes[scene_id].get_visual(visual_id)


def _appearance_event(visual_id, field, value):
    return AppearanceChangedEvent(
        source_id=uuid4(),
        visual_id=visual_id,
        field_name=field,
        new_value=value,
        requires_reslice=False,
    )


def _opaque_colors(frame: np.ndarray) -> np.ndarray:
    opaque = frame[..., 3] > 0
    return np.unique(frame[opaque][:, :3], axis=0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_2d(controller, labels_volume):
    """In-memory label visuals build both inner nodes eagerly at construction."""
    scene = controller.add_scene(displayed_axes=("y", "x"), name="scene")
    visual = controller.add_labels(data=labels_volume, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_2d is not None
    assert gfx._inner_node_2d is not None


def test_construction_3d(controller, labels_volume):
    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="scene")
    visual = controller.add_labels(data=labels_volume, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    assert gfx.node_3d is not None
    assert gfx._inner_node_3d is not None


# ---------------------------------------------------------------------------
# Rendered output
# ---------------------------------------------------------------------------


async def test_render_2d_shows_two_labels(
    controller, render_scene, reslice, labels_volume
):
    """The two label blocks upload and render as two distinct colours."""
    scene = controller.add_scene(displayed_axes=("y", "x"), name="scene")
    controller.add_labels(data=labels_volume, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)

    assert np.count_nonzero(frame[..., 3]) > 0
    assert len(_opaque_colors(frame)) >= 2


async def test_render_3d_draws(controller, render_scene, reslice, labels_volume):
    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="scene")
    controller.add_labels(data=labels_volume, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_data_ready_uploads_label_texture(controller, reslice, labels_volume):
    """After reslice the inner node's texture holds the label ids (max == 7)."""
    scene = controller.add_scene(displayed_axes=("y", "x"), name="scene")
    visual = controller.add_labels(data=labels_volume, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    grid = gfx._inner_node_2d.geometry.grid
    assert int(grid.data.max()) == 7


# ---------------------------------------------------------------------------
# Appearance updates
# ---------------------------------------------------------------------------


async def test_opacity_update(controller, reslice, labels_volume):
    scene = controller.add_scene(displayed_axes=("y", "x"), name="scene")
    visual = controller.add_labels(data=labels_volume, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(_appearance_event(visual.id, "opacity", 0.4))
    assert gfx._inner_node_2d.material.opacity == pytest.approx(0.4)


async def test_salt_and_background_label_update_params(
    controller, reslice, labels_volume
):
    """``salt`` / ``background_label`` write into the 0-d label params uniform.

    Regression: ``_update_label_params_uniform`` used ``buf.data[:] = ...`` on a
    0-dimensional structured buffer, which raised ``IndexError`` on every salt /
    background_label / direct-recolor change.  Fields must be written directly.
    """
    scene = controller.add_scene(displayed_axes=("y", "x"), name="scene")
    visual = controller.add_labels(data=labels_volume, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(_appearance_event(visual.id, "salt", 11))
    gfx.on_appearance_changed(_appearance_event(visual.id, "background_label", 3))

    buf = gfx._label_params_buf
    assert int(buf.data["salt"]) == 11
    assert int(buf.data["background_label"]) == 3
    assert gfx._background_label == 3


async def test_render_mode_update_3d(controller, reslice, labels_volume):
    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="scene")
    visual = controller.add_labels(data=labels_volume, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(
        _appearance_event(visual.id, "render_mode", "flat_categorical")
    )
    assert gfx._inner_node_3d.material.render_mode == "flat_categorical"


async def test_color_dict_direct_mode_rebuilds_lut(controller, render_scene, reslice):
    """Direct colormap mode + ``color_dict`` update rebuilds the LUT and renders."""
    from cellier.data.label._label_memory_store import LabelMemoryStore

    data = np.zeros((4, 16, 16), dtype=np.int32)
    data[:, 2:8, 2:8] = 1
    data[:, 9:14, 9:14] = 2
    store = LabelMemoryStore(data=data, name="direct_labels")

    scene = controller.add_scene(displayed_axes=("y", "x"), name="scene")
    visual = controller.add_labels(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(
            colormap_mode="direct",
            color_dict={
                1: (1.0, 0.0, 0.0, 1.0),
                2: (0.0, 0.0, 1.0, 1.0),
            },
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_appearance_changed(
        _appearance_event(
            visual.id,
            "color_dict",
            {1: (0.0, 1.0, 0.0, 1.0), 2: (1.0, 1.0, 0.0, 1.0)},
        )
    )

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_render_3d_direct_mode_binds_lut(controller, render_scene, reslice):
    """A 3D in-memory label volume in direct mode binds the direct-LUT textures.

    Covers the ``colormap_mode == "direct"`` branch of
    ``LabelVolumeShader.get_bindings`` (``_label_volume.py``), which the
    random-mode 3D tests never reach: it binds ``t_label_keys`` /
    ``t_label_colors`` only when direct-mode textures are present.
    """
    from cellier.data.label._label_memory_store import LabelMemoryStore

    data = np.zeros((6, 16, 16), dtype=np.int32)
    data[1:5, 2:8, 2:8] = 1
    data[1:5, 9:14, 9:14] = 2
    store = LabelMemoryStore(data=data, name="direct_labels_3d")

    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="scene")
    controller.add_labels(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(
            colormap_mode="direct",
            color_dict={
                1: (1.0, 0.0, 0.0, 1.0),
                2: (0.0, 0.0, 1.0, 1.0),
            },
        ),
    )
    controller.add_canvas(scene_id=scene.id)

    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_visibility_toggle(controller, render_scene, reslice, labels_volume):
    scene = controller.add_scene(displayed_axes=("y", "x"), name="scene")
    visual = controller.add_labels(data=labels_volume, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    gfx = _gfx_visual(controller, scene.id, visual.id)
    gfx.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=visual.id, visible=False
        )
    )
    hidden = render_scene(controller, scene.id)
    assert np.count_nonzero(hidden[..., 3]) == 0

    gfx.on_visibility_changed(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=visual.id, visible=True
        )
    )
    shown = render_scene(controller, scene.id)
    assert np.count_nonzero(shown[..., 3]) > 0
