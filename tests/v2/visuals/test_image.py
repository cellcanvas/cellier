"""Tests for MultiscaleImageAppearance and MultiscaleImageVisual models."""

import uuid

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageSingleAppearance,
    MultiscaleImageVisual,
)
from cellier.visuals._base_visual import BaseAppearance, BaseVisual
from tests._v2 import level_transforms, scale_and_translation


def _make_level_transforms_3d(factors):
    """Build level transforms from integer downscale factors."""
    scales = [(float(f),) * 3 for f in factors]
    translations = [((float(f) - 1) / 2,) * 3 for f in factors]
    translations[0] = (0.0, 0.0, 0.0)
    return level_transforms(scales, translations)


def test_image_appearance_roundtrip(tmp_path):
    # Non-default values including the new LOD/frustum fields
    original = MultiscaleImageAppearance(
        lod_bias=2.0,
        force_level=1,
        frustum_cull=False,
    )
    path = tmp_path / "appearance.json"
    path.write_text(original.model_dump_json())
    deserialized = MultiscaleImageAppearance.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()

    # force_level=None roundtrip
    original_none = MultiscaleImageAppearance(force_level=None)
    path2 = tmp_path / "appearance_none.json"
    path2.write_text(original_none.model_dump_json())
    deserialized_none = MultiscaleImageAppearance.model_validate_json(path2.read_text())
    assert original_none.model_dump_json() == deserialized_none.model_dump_json()


def test_multiscale_image_visual_roundtrip(tmp_path):
    store_id = str(uuid.uuid4())
    transforms = _make_level_transforms_3d([1, 2, 4])
    original = MultiscaleImageVisual(
        name="volume",
        data_store_id=store_id,
        level_transforms=transforms,
        appearance=MultiscaleImageAppearance(
            lod_bias=1.5, force_level=None, frustum_cull=True
        ),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    path = tmp_path / "visual.json"
    path.write_text(original.model_dump_json())
    deserialized = MultiscaleImageVisual.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


# ---------------------------------------------------------------------------
# requires_camera_reslice field tests
# ---------------------------------------------------------------------------


class _MinimalVisual(BaseVisual):
    appearance: BaseAppearance = BaseAppearance()


def test_base_visual_requires_camera_reslice_defaults_false():
    v = _MinimalVisual(
        name="test", data_store_id="00000000-0000-0000-0000-000000000000"
    )
    assert v.requires_camera_reslice is False


def test_multiscale_image_visual_requires_camera_reslice_true():
    v = MultiscaleImageVisual(
        name="vol",
        data_store_id="00000000-0000-0000-0000-000000000000",
        level_transforms=_make_level_transforms_3d([1, 2]),
        appearance=MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    assert v.requires_camera_reslice is True


def test_requires_camera_reslice_is_frozen():
    v = MultiscaleImageVisual(
        name="vol",
        data_store_id="00000000-0000-0000-0000-000000000000",
        level_transforms=_make_level_transforms_3d([1, 2]),
        appearance=MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    with pytest.raises((ValidationError, TypeError)):
        v.requires_camera_reslice = False


# ---------------------------------------------------------------------------
# Transform field tests
# ---------------------------------------------------------------------------


def test_transform_defaults_to_unplaced():
    """D18 forbids a coordinate-system-less identity.

    A transform names its endpoints by id, and a visual built before it is
    added to a scene does not yet know which world it is going into -- so the
    default is "not decided", and ``add_visual`` decides it.
    """
    v = _MinimalVisual(
        name="test", data_store_id="00000000-0000-0000-0000-000000000000"
    )
    assert v.transform is None


def test_transform_field_fires_psygnal():
    v = _MinimalVisual(
        name="test", data_store_id="00000000-0000-0000-0000-000000000000"
    )
    received = []
    v.events.transform.connect(lambda t: received.append(t))
    new_t = scale_and_translation((2.0, 2.0, 2.0))
    v.transform = new_t
    assert len(received) == 1
    np.testing.assert_array_equal(received[0].matrix, new_t.matrix)


def test_visual_roundtrip_with_non_identity_transform():
    t = scale_and_translation((2.0, 3.0, 4.0), (10.0, 20.0, 30.0))
    v = MultiscaleImageVisual(
        name="vol",
        data_store_id="00000000-0000-0000-0000-000000000000",
        level_transforms=_make_level_transforms_3d([1, 2]),
        appearance=MultiscaleImageAppearance(),
        transform=t,
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    json_str = v.model_dump_json()
    v2 = MultiscaleImageVisual.model_validate_json(json_str)
    np.testing.assert_allclose(v2.transform.matrix, t.matrix, atol=1e-6)


def test_single_appearance_defaults_color_map_and_clim():
    a = MultiscaleImageSingleAppearance()
    assert a.clim == (0.0, 1.0)
    assert a.color_map is not None
    assert a.render_mode == "iso"


def test_shared_appearance_carries_interpolation():
    a = MultiscaleImageAppearance(interpolation="linear")
    assert a.interpolation == "linear"


def test_image_appearance_interpolation_default():
    a = MultiscaleImageAppearance()
    assert a.interpolation == "nearest"
