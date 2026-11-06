"""MaskedImage models (masked image design v2, 4.1 to 4.3, Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from cellier.transform import AffineTransform, CoordinateSystem
from cellier.visuals import (
    MaskedImageAppearance,
    MaskedImageRenderConfig,
    MaskedImageSingleAppearance,
    MaskedImageVisual,
    MultiscaleImageVisual,
    RenderPlane,
)
from cellier.visuals._masked_image import DEFAULT_MASKED_GPU_BUDGET_BYTES
from cellier.visuals._types import VisualType


def _system(name: str) -> CoordinateSystem:
    from cellier.scene.dims import spatial_axes

    return CoordinateSystem(name=name, axes=spatial_axes("z", "y", "x"))


def _model(**fields) -> MaskedImageVisual:
    mask = _system("mask")
    value = _system("value")
    world = _system("world")
    defaults = {
        "name": "masked",
        "data_store_id": "00000000-0000-4000-8000-000000000001",
        "value_store_id": "00000000-0000-4000-8000-000000000002",
        "level_transforms": [],
        "value_level_transforms": [],
        "transform": AffineTransform.from_matrix(np.eye(4), mask, world),
        "value_to_mask": AffineTransform.from_matrix(
            np.diag([0.5, 0.5, 0.5, 1.0]), value, mask
        ),
    }
    defaults.update(fields)
    return MaskedImageVisual(**defaults)


def test_defaults():
    model = _model()
    assert model.visual_type == "masked_image"
    assert isinstance(model, MultiscaleImageVisual)
    assert model.appearance.interpolation == "linear"
    assert model.appearance.target_bricks == "visible"
    assert model.appearance.value_force_level is None
    single = model.single
    assert single.render_mode == "iso"
    assert single.surface_thickness == 0.0
    assert single.surface_projection == "max"
    assert single.surface_placement == "inward"
    assert single.surface_min_width_px == 1.0
    assert (single.band_rays, single.band_cap) == (16, 32)
    config = model.render_config
    assert config.gpu_budget_bytes == DEFAULT_MASKED_GPU_BUDGET_BYTES
    assert config.value_gpu_budget_bytes == DEFAULT_MASKED_GPU_BUDGET_BYTES
    assert DEFAULT_MASKED_GPU_BUDGET_BYTES == 1100 * 1024**2


@pytest.mark.parametrize(
    ("fields", "words"),
    [
        ({"channel_axis": 0}, "channel_axis=0"),
        ({"channel_axis": 0, "composite": True}, "composite=True"),
        ({"render_planes": ["plane"]}, "render_planes"),
    ],
)
def test_refused_fields(fields, words):
    if "render_planes" in fields:
        world = _system("world")
        fields = {
            "render_planes": [
                RenderPlane.from_point_normal(world, (1.0, 1.0, 1.0), (1, 0, 0))
            ]
        }
    with pytest.raises(ValidationError, match=words) as info:
        _model(**fields)
    assert "slider axis" in str(info.value)


@pytest.mark.parametrize(
    ("fields", "words"),
    [
        ({"slab_projection": "mip"}, "does not project a slab"),
        ({"projection_axes": ("z",)}, "does not project a slab"),
    ],
)
def test_appearance_refuses_a_slab(fields, words):
    with pytest.raises(ValidationError, match=words):
        MaskedImageAppearance(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"surface_thickness": -1.0},
        {"surface_min_width_px": -0.5},
        {"band_rays": 0},
        {"band_cap": 0},
        {"surface_projection": "sum"},
        {"surface_placement": "centered"},
        {"render_mode": "mip"},
    ],
)
def test_single_appearance_bounds(fields):
    with pytest.raises(ValidationError):
        MaskedImageSingleAppearance(**fields)


def test_store_ids_and_registration_are_frozen():
    model = _model()
    with pytest.raises(ValidationError):
        model.value_store_id = "00000000-0000-4000-8000-000000000003"
    with pytest.raises(ValidationError):
        model.value_to_mask = None


def test_serialization_round_trip():
    model = _model(
        single=MaskedImageSingleAppearance(
            surface_thickness=2.5,
            surface_projection="mean",
            surface_placement="centred",
            band_rays=8,
            band_cap=24,
            iso_threshold=3.0,
        ),
        appearance=MaskedImageAppearance(target_bricks="surface", value_force_level=2),
        render_config=MaskedImageRenderConfig(
            gpu_budget_bytes=64 * 1024**2, value_gpu_budget_bytes=32 * 1024**2
        ),
    )
    adapter = TypeAdapter(VisualType)
    restored = adapter.validate_json(model.model_dump_json())
    assert type(restored) is MaskedImageVisual
    assert restored.model_dump() == model.model_dump()
    np.testing.assert_array_equal(
        restored.value_to_mask.matrix, model.value_to_mask.matrix
    )
