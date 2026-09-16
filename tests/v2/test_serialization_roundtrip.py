"""Round-trip serialization test for ViewerModel / CellierController."""

import numpy as np

from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.scene.dims import (
    AxisAlignedSelection,
    DimsManager,
    spatial_axes,
    world_coordinate_system,
)
from cellier.scene.scene import Scene
from cellier.viewer_model import DataManager, ViewerModel
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual
from tests._v2 import level_transforms


def _make_minimal_model() -> ViewerModel:
    world = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    dims = DimsManager(
        world_coordinate_system=world,
        selection=AxisAlignedSelection(
            displayed_axes=(1, 2), slice_indices={0: 0, 1: 0, 2: 0}
        ),
    )
    positions = np.zeros((4, 3), dtype=np.float32)
    colors = np.ones((4, 4), dtype=np.float32)
    store = PointsMemoryStore(positions=positions, colors=colors, name="pts")

    visual = PointsVisual(
        name="points",
        data_store_id=str(store.id),
        appearance=PointsMarkerAppearance(),
    )
    scene = Scene(
        name="xy",
        dims=dims,
        render_modes={"2d"},
        lighting="none",
        visuals=[visual],
    )

    return ViewerModel(
        data=DataManager(stores={store.id: store}),
        scenes={scene.id: scene},
    )


def test_viewer_model_roundtrip_json(tmp_path):
    """ViewerModel serializes to JSON and deserializes without data loss."""
    original = _make_minimal_model()
    path = tmp_path / "viewer.json"
    original.to_file(path)

    loaded = ViewerModel.from_file(path)

    # Structural checks
    assert list(loaded.scenes.keys()) == list(original.scenes.keys())
    original_scene = next(iter(original.scenes.values()))
    loaded_scene = next(iter(loaded.scenes.values()))
    assert loaded_scene.name == original_scene.name
    assert loaded_scene.render_modes == original_scene.render_modes
    assert loaded_scene.lighting == original_scene.lighting
    assert len(loaded_scene.visuals) == len(original_scene.visuals)
    assert loaded_scene.visuals[0].name == original_scene.visuals[0].name

    # Data store checks
    assert list(loaded.data.stores.keys()) == list(original.data.stores.keys())


def test_multiscale_render_config_roundtrip():
    """MultiscaleImageRenderConfig fields survive JSON serialization.

    interpolation is now on MultiscaleImageAppearance, not MultiscaleImageRenderConfig.
    """
    from cmap import Colormap

    from cellier.visuals import MultiscaleImageVisual

    render_config = MultiscaleImageRenderConfig(
        block_size=64,
        gpu_budget_bytes=512 * 1024**2,
    )
    appearance = MultiscaleImageAppearance(
        color_map=Colormap("viridis"), clim=(0.0, 1.0), interpolation="linear"
    )
    visual = MultiscaleImageVisual(
        name="img",
        data_store_id="00000000-0000-0000-0000-000000000001",
        level_transforms=level_transforms(
            [[1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0]],
        ),
        appearance=appearance,
        render_config=render_config,
    )
    reloaded = MultiscaleImageVisual.model_validate_json(visual.model_dump_json())
    assert reloaded.render_config.block_size == 64
    assert reloaded.appearance.interpolation == "linear"
