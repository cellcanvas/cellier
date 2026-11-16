"""Tests for Canvas model."""

from cellier.scene import Canvas
from cellier.scene.cameras import (
    OrbitCameraController,
    OrthographicCamera,
    PanZoomCameraController,
    PerspectiveCamera,
)


def test_canvas_single_camera_roundtrip(tmp_path):
    original = Canvas(
        cameras={"3d": PerspectiveCamera(controller=OrbitCameraController())}
    )
    path = tmp_path / "canvas_single.json"
    path.write_text(original.model_dump_json())
    deserialized = Canvas.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_canvas_two_camera_roundtrip(tmp_path):
    original = Canvas(
        cameras={
            "3d": PerspectiveCamera(controller=OrbitCameraController()),
            "2d": OrthographicCamera(controller=PanZoomCameraController()),
        }
    )
    path = tmp_path / "canvas_two.json"
    path.write_text(original.model_dump_json())
    deserialized = Canvas.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_canvas_needs_a_camera():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Canvas(cameras={})


def test_canvas_camera_keys_are_2d_or_3d():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Canvas(cameras={"4d": PerspectiveCamera(controller=OrbitCameraController())})
