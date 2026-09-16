"""Tests for cellier.convenience.OrthoViewer."""

import numpy as np
import pytest

from cellier.convenience import (
    ContinuousAxisValues,
    OrthoViewer,
    Viewer,
    axis_values_from_ortho,
)
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageChannelAppearance,
    InMemoryImageSingleAppearance,
)
from cellier.visuals._image_memory import InMemoryImageAppearance


@pytest.fixture
def image_store() -> ImageMemoryStore:
    data = np.zeros((8, 16, 24), dtype=np.float32)
    return ImageMemoryStore(data=data, name="test_image")


def test_scene_layout_3d():
    """The four panels get the expected displayed/sliced axes and render modes."""
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))
    assert set(viewer.scenes) == {"xy", "xz", "yz", "vol"}
    assert viewer.spatial_axes == (0, 1, 2)
    assert viewer.extra_axes == set()

    expected_displayed = {
        "xy": (1, 2),
        "xz": (0, 2),
        "yz": (0, 1),
        "vol": (0, 1, 2),
    }
    for key, scene in viewer.scenes.items():
        assert tuple(scene.dims.selection.displayed_axes) == expected_displayed[key]
        assert scene.render_modes == ({"3d"} if key == "vol" else {"2d"})


def test_extra_axis_is_last_three_spatial_by_default():
    """A 5D dataset keeps the last three axes spatial; the rest are extra."""
    viewer = OrthoViewer(
        [
            ("t", "time"),
            ("c", "channel"),
            ("z", "space"),
            ("y", "space"),
            ("x", "space"),
        ]
    )
    assert viewer.spatial_axes == (2, 3, 4)
    assert viewer.extra_axes == {0, 1}
    # Every panel slices both extra axes.
    for scene in viewer.scenes.values():
        sliced = set(scene.dims.selection.slice_indices)
        assert {0, 1} <= sliced


def test_explicit_spatial_axes_by_name():
    """spatial_axes may be given as names in (z, y, x) order."""
    viewer = OrthoViewer(
        [("z", "space"), ("y", "space"), ("x", "space"), ("c", "channel")],
        spatial_axes=("z", "y", "x"),
    )
    assert viewer.spatial_axes == (0, 1, 2)
    assert viewer.extra_axes == {3}


def test_requires_three_axes():
    with pytest.raises(ValueError, match="at least 3 axes"):
        OrthoViewer(spatial_axes("y", "x"))


def test_add_image_fans_out_to_all_panels(image_store):
    """One add_image registers a single store and adds one visual per panel."""
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    visuals = viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        name="blobs",
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    assert set(visuals) == {"xy", "xz", "yz", "vol"}
    assert {v.name for v in visuals.values()} == {
        "blobs_xy",
        "blobs_xz",
        "blobs_yz",
        "blobs_vol",
    }
    # Exactly one data store, one visual per scene.
    assert len(viewer.controller._model.data.stores) == 1
    for scene in viewer.scenes.values():
        assert len(scene.visuals) == 1


def test_center_slices_sets_unrounded_midpoints(image_store):
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    ranges = axis_values_from_ortho(viewer)
    # The edge convention: a (8, 16, 24) store spans half a voxel beyond its
    # first and last voxel centres, because a voxel is centred on its index.
    # The midpoints asserted below are unaffected -- the widening is
    # symmetric -- which is what this test is actually about.
    assert ranges == {
        0: ContinuousAxisValues(min=-0.5, max=7.5),
        1: ContinuousAxisValues(min=-0.5, max=15.5),
        2: ContinuousAxisValues(min=-0.5, max=23.5),
    }

    viewer.center_slices()
    # Slice positions are floats (D3): the midpoint of an even-length axis
    # sits between its two middle voxels and is kept there, not rounded.
    # Every panel keeps a position for every axis (D36), so all four hold the
    # same point, displayed axes included.
    for key in ("xy", "xz", "yz", "vol"):
        assert dict(viewer.scenes[key].dims.selection.slice_indices) == {
            0: 3.5,
            1: 7.5,
            2: 11.5,
        }


def test_axis_sync_propagates_across_panels():
    """Changing an extra axis on one panel updates the others."""
    viewer = OrthoViewer(
        [("c", "channel"), ("z", "space"), ("y", "space"), ("x", "space")]
    )
    assert viewer.axis_sync_enabled

    xy = viewer.scenes["xy"]
    viewer.controller.update_slice_indices(xy.id, {0: 3})

    for scene in viewer.scenes.values():
        assert scene.dims.selection.slice_indices[0] == 3


def test_axis_sync_can_be_disabled():
    viewer = OrthoViewer(
        [("c", "channel"), ("z", "space"), ("y", "space"), ("x", "space")]
    )
    viewer.axis_sync_enabled = False
    assert not viewer.axis_sync_enabled

    xy = viewer.scenes["xy"]
    viewer.controller.update_slice_indices(xy.id, {0: 3})

    # Other panels are untouched.
    assert viewer.scenes["vol"].dims.selection.slice_indices[0] == 0


def test_serialization_roundtrip(tmp_path, image_store):
    """OrthoViewer serializes and restores an equivalent ViewerModel."""
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        name="blobs",
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    viewer.center_slices()

    path = tmp_path / "ortho.json"
    viewer.to_file(path)

    loaded = OrthoViewer.from_file(path)
    assert viewer.controller._model == loaded.controller._model
    assert set(loaded.scenes) == {"xy", "xz", "yz", "vol"}
    assert loaded.spatial_axes == (0, 1, 2)


def test_from_file_rejects_non_ortho_model(tmp_path, image_store):
    """A single-scene Viewer file is not a valid OrthoViewer file."""
    viewer = Viewer(spatial_axes("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    path = tmp_path / "single.json"
    viewer.to_file(path)

    with pytest.raises(ValueError, match="four orthoviewer panels"):
        OrthoViewer.from_file(path)


@pytest.mark.parametrize(
    ("composite", "n_channels"),
    [(False, 2), (True, 2), (True, 0)],
    ids=["single", "composite", "empty_composite"],
)
def test_serialization_roundtrip_unified_image(tmp_path, composite, n_channels):
    """Every panel's image keeps its mode and channels (design 3.11)."""
    world = [("c", "channel"), *spatial_axes("z", "y", "x")]
    store = ImageMemoryStore(data=np.zeros((2, 4, 8, 8), dtype=np.float32), name="c")
    viewer = OrthoViewer(world)
    viewer.add_image(
        store,
        channel_axis=0,
        composite=composite,
        channels={
            index: InMemoryImageChannelAppearance(visible=bool(index))
            for index in range(n_channels)
        },
        name="czyx",
    )

    path = tmp_path / "ortho.json"
    viewer.to_file(path)
    loaded = OrthoViewer.from_file(path)

    assert viewer.controller._model == loaded.controller._model
    for key, scene in loaded.scenes.items():
        (visual,) = scene.visuals
        assert (visual.channel_axis, visual.composite) == (0, composite), key
        assert sorted(visual.channels) == list(range(n_channels)), key
