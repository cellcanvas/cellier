"""Tests for ViewerModel."""

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.scene import Canvas
from cellier.scene.cameras import OrbitCameraController, PerspectiveCamera
from cellier.scene.dims import (
    AxisAlignedSelection,
    DimsManager,
    spatial_axes,
    world_coordinate_system,
)
from cellier.scene.scene import Scene
from cellier.viewer_model import DataManager, ViewerModel
from cellier.visuals import MultiscaleImageAppearance, MultiscaleImageVisual
from tests._v2 import level_transforms, pyramid_levels


def _build_viewer(small_zarr_store):
    store = MultiscaleZarrDataStore(
        zarr_path=str(small_zarr_store),
        scale_names=["s0", "s1"],
        **pyramid_levels(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        name="test_volume",
    )
    data = DataManager(stores={store.id: store})

    appearance = MultiscaleImageAppearance(
        color_map="viridis",
        clim=(0.0, 1.0),
        lod_bias=1.0,
        force_level=None,
        frustum_cull=True,
    )
    visual = MultiscaleImageVisual(
        name="volume",
        data_store_id=str(store.id),
        level_transforms=level_transforms(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        appearance=appearance,
    )

    cs = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    dims = DimsManager(
        world_coordinate_system=cs,
        selection=AxisAlignedSelection(
            displayed_axes=(0, 1, 2),
            slice_indices={0: 0, 1: 0, 2: 0},
        ),
    )
    camera = PerspectiveCamera(
        fov=70.0,
        near_clipping_plane=1.0,
        far_clipping_plane=8000.0,
        controller=OrbitCameraController(enabled=True),
    )
    canvas = Canvas(cameras={"3d": camera})
    scene = Scene(
        name="main",
        dims=dims,
        visuals=[visual],
        canvases={canvas.id: canvas},
    )

    return ViewerModel(data=data, scenes={scene.id: scene}), store, scene, canvas


def test_viewer_model_roundtrip(tmp_path, small_zarr_store):
    viewer, _, _, _ = _build_viewer(small_zarr_store)

    viewer.to_file(tmp_path / "session.json")
    deserialized = ViewerModel.from_file(tmp_path / "session.json")

    assert viewer.model_dump_json() == deserialized.model_dump_json()


def test_viewer_model_worked_example(tmp_path, small_zarr_store):
    viewer, _store, _scene, _canvas = _build_viewer(small_zarr_store)

    viewer.to_file(tmp_path / "session.json")
    deserialized = ViewerModel.from_file(tmp_path / "session.json")

    assert viewer.model_dump_json() == deserialized.model_dump_json()

    # Structural integrity checks
    assert len(deserialized.scenes) == 1
    d_scene = next(iter(deserialized.scenes.values()))
    assert len(d_scene.visuals) == 1
    assert len(d_scene.canvases) == 1

    d_visual = d_scene.visuals[0]
    assert d_visual.appearance.frustum_cull is True
    assert d_visual.appearance.force_level is None

    d_canvas = next(iter(d_scene.canvases.values()))
    assert "3d" in d_canvas.cameras
    assert d_canvas.cameras["3d"].controller.controller_type == "orbit"

    d_store = next(iter(deserialized.data.stores.values()))
    assert d_store.n_levels == 2
