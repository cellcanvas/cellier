"""Verify render_config classes are frozen."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellier.visuals import MultiscaleLabelRenderConfig
from cellier.visuals._image import MultiscaleImageRenderConfig


def test_multiscale_image_render_config_frozen():
    rc = MultiscaleImageRenderConfig()
    with pytest.raises((ValidationError, TypeError)):
        rc.block_size = 64


def test_multiscale_label_render_config_frozen():
    rc = MultiscaleLabelRenderConfig()
    with pytest.raises((ValidationError, TypeError)):
        rc.block_size = 64


def test_multiscale_image_render_config_json_roundtrip(tmp_path):
    rc = MultiscaleImageRenderConfig(block_size=64)
    path = tmp_path / "rc.json"
    path.write_text(rc.model_dump_json())
    rc2 = MultiscaleImageRenderConfig.model_validate_json(path.read_text())
    assert rc2.block_size == 64


def test_multiscale_label_render_config_json_roundtrip(tmp_path):
    rc = MultiscaleLabelRenderConfig(block_size=64)
    path = tmp_path / "rc.json"
    path.write_text(rc.model_dump_json())
    rc2 = MultiscaleLabelRenderConfig.model_validate_json(path.read_text())
    assert rc2.block_size == 64
