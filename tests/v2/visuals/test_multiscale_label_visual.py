"""Tests for MultiscaleLabelsAppearance and MultiscaleLabelVisual model layer."""

from __future__ import annotations

from uuid import uuid4

import pytest

from cellier.visuals import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    MultiscaleLabelVisual,
)
from tests._v2 import level_transforms

# ── MultiscaleLabelsAppearance ────────────────────────────────────────────────


def test_appearance_defaults():
    a = MultiscaleLabelsAppearance()
    assert a.colormap_mode == "random"
    assert a.background_label == 0
    assert a.salt == 0
    assert a.color_dict == {}
    assert a.render_mode == "iso_categorical"
    assert a.opacity == 1.0
    assert a.settled_lod_bias == 1.0
    assert a.force_level is None
    assert a.frustum_cull is True


def test_appearance_json_roundtrip():
    a = MultiscaleLabelsAppearance(
        colormap_mode="direct",
        background_label=-1,
        salt=42,
        color_dict={1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0, 1.0)},
        render_mode="flat_categorical",
        opacity=0.7,
        settled_lod_bias=2.0,
        force_level=1,
        frustum_cull=False,
    )
    json_str = a.model_dump_json()
    b = MultiscaleLabelsAppearance.model_validate_json(json_str)
    assert b.colormap_mode == "direct"
    assert b.background_label == -1
    assert b.salt == 42
    assert b.color_dict == {1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0, 1.0)}
    assert b.render_mode == "flat_categorical"
    assert abs(b.opacity - 0.7) < 1e-6
    assert b.settled_lod_bias == 2.0
    assert b.force_level == 1
    assert b.frustum_cull is False


def test_appearance_mutation_fires_event():
    a = MultiscaleLabelsAppearance()
    received = []
    a.events.background_label.connect(lambda v: received.append(v))
    a.background_label = 5
    assert received == [5]


def test_appearance_salt_mutation_fires_event():
    a = MultiscaleLabelsAppearance()
    received = []
    a.events.salt.connect(lambda v: received.append(v))
    a.salt = 123
    assert received == [123]


def test_colormap_mode_is_frozen():
    from pydantic import ValidationError

    a = MultiscaleLabelsAppearance(colormap_mode="random")
    with pytest.raises(ValidationError):
        a.colormap_mode = "direct"


def test_render_mode_mutation_fires_event():
    a = MultiscaleLabelsAppearance(render_mode="iso_categorical")
    received = []
    a.events.render_mode.connect(lambda v: received.append(v))
    a.render_mode = "flat_categorical"
    assert received == ["flat_categorical"]


# ── MultiscaleLabelRenderConfig ──────────────────────────────────────────────


def test_render_config_defaults():
    rc = MultiscaleLabelRenderConfig()
    assert rc.block_size == 32
    assert rc.gpu_budget_bytes == 1 * 1024**3
    assert rc.gpu_budget_bytes_2d == 64 * 1024**2


def test_render_config_json_roundtrip():
    rc = MultiscaleLabelRenderConfig(block_size=64, gpu_budget_bytes=512 * 1024**2)
    json_str = rc.model_dump_json()
    rc2 = MultiscaleLabelRenderConfig.model_validate_json(json_str)
    assert rc2.block_size == 64
    assert rc2.gpu_budget_bytes == 512 * 1024**2


# ── MultiscaleLabelVisual ────────────────────────────────────────────────────


def _identity_transform(ndim: int = 3):
    return level_transforms([(1.0,) * ndim])[0]


def test_visual_json_roundtrip():
    store_id = str(uuid4())
    t = _identity_transform()
    a = MultiscaleLabelsAppearance(colormap_mode="random", background_label=0)
    v = MultiscaleLabelVisual(
        name="test",
        data_store_id=store_id,
        level_transforms=[t],
        appearance=a,
    )
    json_str = v.model_dump_json()
    v2 = MultiscaleLabelVisual.model_validate_json(json_str)
    assert v2.data_store_id == store_id
    assert v2.appearance.colormap_mode == "random"
    assert v2.name == "test"


def test_visual_json_roundtrip_with_color_dict():
    store_id = str(uuid4())
    t = _identity_transform()
    a = MultiscaleLabelsAppearance(
        colormap_mode="direct",
        color_dict={1: (1.0, 0.0, 0.0, 1.0), 3: (0.0, 0.0, 1.0, 1.0)},
    )
    v = MultiscaleLabelVisual(
        name="labels",
        data_store_id=store_id,
        level_transforms=[t],
        appearance=a,
    )
    json_str = v.model_dump_json()
    v2 = MultiscaleLabelVisual.model_validate_json(json_str)
    assert v2.appearance.color_dict == {
        1: (1.0, 0.0, 0.0, 1.0),
        3: (0.0, 0.0, 1.0, 1.0),
    }


def test_visual_type_discriminator():
    v = MultiscaleLabelVisual(
        name="x",
        data_store_id=str(uuid4()),
        level_transforms=[_identity_transform()],
        appearance=MultiscaleLabelsAppearance(),
    )
    assert v.visual_type == "multiscale_label"


def test_visual_requires_camera_reslice_is_true():
    v = MultiscaleLabelVisual(
        name="x",
        data_store_id=str(uuid4()),
        level_transforms=[_identity_transform()],
        appearance=MultiscaleLabelsAppearance(),
    )
    assert v.requires_camera_reslice is True


# ── Phase 2: MultiscaleLabelsAppearance inherits from BaseLabelsAppearance ────


def test_label_appearance_inherits_colormap_mode_frozen():
    from pydantic import ValidationError

    a = MultiscaleLabelsAppearance()
    with pytest.raises(ValidationError):
        a.colormap_mode = "direct"


def test_label_appearance_inherits_salt_mutation():
    a = MultiscaleLabelsAppearance()
    received = []
    a.events.salt.connect(lambda v: received.append(v))
    a.salt = 99
    assert received == [99]


def test_label_appearance_render_mode_is_live():
    """render_mode must not be frozen — Phase 1 added a live GFX handler for it."""
    a = MultiscaleLabelsAppearance()
    received = []
    a.events.render_mode.connect(lambda v: received.append(v))
    a.render_mode = "flat_categorical"
    assert received == ["flat_categorical"]


def test_label_appearance_wide_render_mode():
    """MultiscaleLabelsAppearance supports gradient_debug and smooth_iso (wider than base)."""  # noqa: E501
    a = MultiscaleLabelsAppearance(render_mode="gradient_debug")
    assert a.render_mode == "gradient_debug"
    a2 = MultiscaleLabelsAppearance(render_mode="smooth_iso")
    assert a2.render_mode == "smooth_iso"
