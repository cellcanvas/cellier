"""Tests for Scene model."""

import uuid

from cellier.scene import Canvas
from cellier.scene.cameras import OrbitCameraController, PerspectiveCamera
from cellier.scene.dims import (
    AxisAlignedSelection,
    DimsManager,
    spatial_axes,
    world_coordinate_system,
)
from cellier.scene.scene import Scene
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageSingleAppearance,
    MultiscaleImageVisual,
)
from tests._v2 import level_transforms


def test_scene_roundtrip(tmp_path):
    dims = DimsManager(
        world_coordinate_system=world_coordinate_system(
            spatial_axes("z", "y", "x"), name="world"
        ),
        selection=AxisAlignedSelection(
            displayed_axes=(0, 1, 2),
            slice_indices={0: 0, 1: 0, 2: 0},
        ),
    )
    visual = MultiscaleImageVisual(
        name="volume",
        data_store_id=str(uuid.uuid4()),
        level_transforms=level_transforms(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        appearance=MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="viridis"),
    )
    camera = PerspectiveCamera(controller=OrbitCameraController())
    canvas = Canvas(cameras={"3d": camera})
    original = Scene(
        name="main",
        dims=dims,
        visuals=[visual],
        canvases={canvas.id: canvas},
    )
    path = tmp_path / "scene.json"
    path.write_text(original.model_dump_json())
    deserialized = Scene.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()
