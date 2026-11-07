"""Tests for RenderManagerConfig and related config models."""

from __future__ import annotations

import json
import warnings

import pytest
from pydantic import ValidationError

from cellier.render import RenderManager
from cellier.render._config import (
    CameraConfig,
    RenderManagerConfig,
    SchedulerConfig,
    TemporalAccumulationConfig,
)


def test_default_config_values():
    config = RenderManagerConfig()
    assert config.temporal.enabled is True
    assert config.temporal.blend_weight == 0.1
    assert config.camera.reslice_enabled is True
    assert config.camera.settle_threshold_s == 0.3


def test_json_roundtrip():
    config = RenderManagerConfig(
        scheduler=SchedulerConfig(max_in_flight=16),
        temporal=TemporalAccumulationConfig(enabled=False, blend_weight=0.05),
        camera=CameraConfig(reslice_enabled=False, settle_threshold_s=0.5),
    )
    json_str = config.model_dump_json()
    config2 = RenderManagerConfig.model_validate_json(json_str)
    assert config2.model_dump_json() == json_str
    assert "slicing" not in config.model_dump()


def test_an_old_slicing_field_is_ignored():
    """``slicing`` configured the retired async slicer (scheduler-only
    loading design v2, D9): it is dropped, without a warning or an error."""
    old = {
        "slicing": {"batch_size": 32, "render_every": 4},
        "temporal": {"enabled": False},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = RenderManagerConfig.model_validate(old)
        from_json = RenderManagerConfig.model_validate_json(json.dumps(old))
        from_kwargs = RenderManagerConfig(slicing={"batch_size": 8})
    assert config == from_json
    assert config.temporal.enabled is False
    assert "slicing" not in config.model_dump()
    assert from_kwargs == RenderManagerConfig()


def test_temporal_blend_weight_setter_updates_config():
    manager = RenderManager(config=RenderManagerConfig())
    manager.temporal_blend_weight = 0.02
    assert manager.config.temporal.blend_weight == 0.02


def test_temporal_enabled_setter_updates_config():
    manager = RenderManager(config=RenderManagerConfig())
    manager.temporal_enabled = False
    assert manager.config.temporal.enabled is False


def test_blend_weight_must_be_in_range():
    with pytest.raises(ValidationError):
        TemporalAccumulationConfig(blend_weight=0.0)
    with pytest.raises(ValidationError):
        TemporalAccumulationConfig(blend_weight=1.1)


def test_settle_threshold_must_be_positive():
    with pytest.raises(ValidationError):
        CameraConfig(settle_threshold_s=0.0)
