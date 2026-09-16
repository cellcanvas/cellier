"""Tests for the world coordinate system, DimsManager and AxisAlignedSelection."""

import pytest

from cellier._state import AxisAlignedSelectionState
from cellier.scene.dims import (
    AxisAlignedSelection,
    DimsManager,
    spatial_axes,
    world_coordinate_system,
)
from cellier.transform import WorldCoordinateSystem


def test_coordinate_system_roundtrip(tmp_path):
    original = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    path = tmp_path / "coordinate_system.json"
    path.write_text(original.model_dump_json())
    deserialized = WorldCoordinateSystem.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_dims_manager_roundtrip(tmp_path):
    # 3D case -- all axes displayed; each still keeps a position (D36)
    original_3d = DimsManager(
        world_coordinate_system=world_coordinate_system(
            spatial_axes("z", "y", "x"), name="world"
        ),
        selection=AxisAlignedSelection(
            displayed_axes=(0, 1, 2),
            slice_indices={0: 0.0, 1: 0.0, 2: 0.0},
        ),
    )
    path = tmp_path / "dims_3d.json"
    path.write_text(original_3d.model_dump_json())
    deserialized_3d = DimsManager.model_validate_json(path.read_text())
    assert original_3d.model_dump_json() == deserialized_3d.model_dump_json()

    # 2D slice through 3D volume
    original_2d = DimsManager(
        world_coordinate_system=world_coordinate_system(
            spatial_axes("z", "y", "x"), name="world"
        ),
        selection=AxisAlignedSelection(
            displayed_axes=(1, 2),
            slice_indices={0: 32, 1: 0, 2: 0},
        ),
        slider_overrides={0: False},
    )
    path2 = tmp_path / "dims_2d.json"
    path2.write_text(original_2d.model_dump_json())
    deserialized_2d = DimsManager.model_validate_json(path2.read_text())
    assert original_2d.model_dump_json() == deserialized_2d.model_dump_json()


def test_axis_aligned_selection_to_state():
    sel = AxisAlignedSelection(
        displayed_axes=(1, 2),
        slice_indices={0: 42},
    )
    state = sel.to_state()
    assert isinstance(state, AxisAlignedSelectionState)
    assert state.displayed_axes == (1, 2)


def test_the_snapshot_does_not_carry_the_slice_positions():
    """D5, landed in Phase 8.  The editable positions stay on the selection,
    which the sliders write to and the region is built from; the snapshot the
    render layer receives carries only what is still read from it.  Every
    consumer of the positions takes the ``RegionSelection`` instead."""
    sel = AxisAlignedSelection(displayed_axes=(1, 2), slice_indices={0: 42})
    assert sel.slice_indices == {0: 42}
    assert not hasattr(sel.to_state(), "slice_indices")


def test_dims_manager_validates_axis_coverage():
    """Every world axis needs a position, displayed or not (D36)."""
    with pytest.raises(ValueError, match="Axis coverage mismatch"):
        DimsManager(
            world_coordinate_system=world_coordinate_system(
                spatial_axes("z", "y", "x"), name="world"
            ),
            selection=AxisAlignedSelection(
                displayed_axes=(0, 1),
                slice_indices={2: 0.0},  # the displayed axes need one too
            ),
        )


def test_dims_manager_rejects_positions_outside_the_world():
    with pytest.raises(ValueError, match="Axis coverage mismatch"):
        DimsManager(
            world_coordinate_system=world_coordinate_system(
                spatial_axes("z", "y", "x"), name="world"
            ),
            selection=AxisAlignedSelection(
                displayed_axes=(1, 2), slice_indices={0: 0, 1: 0, 2: 0, 3: 0}
            ),
        )


def test_dims_manager_rejects_overrides_outside_the_world():
    with pytest.raises(ValueError, match="slider_overrides"):
        DimsManager(
            world_coordinate_system=world_coordinate_system(
                spatial_axes("z", "y", "x"), name="world"
            ),
            selection=AxisAlignedSelection(
                displayed_axes=(1, 2), slice_indices={0: 0, 1: 0, 2: 0}
            ),
            slider_overrides={5: True},
        )


def test_to_selection_bounds_only_the_sliced_axes():
    """A displayed axis keeps a stored position, but the region ignores it."""
    import numpy as np

    from cellier.controller import CellierController

    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=spatial_axes("z", "y", "x"), dim="2d"
    )
    controller.update_slice_indices(scene.id, {0: 2.0, 1: 5.0, 2: 6.0})
    controller.add_canvas(scene.id)
    rendered, embedding = controller._rendered[controller.get_canvas_ids(scene.id)[0]]
    box = scene.dims.to_selection(rendered, embedding).region.bounding_box()
    assert box.min_coordinate[0] == box.max_coordinate[0] == 2.0
    for axis in (1, 2):
        assert box.min_coordinate[axis] == -np.inf
        assert box.max_coordinate[axis] == np.inf


def test_dims_manager_to_state():
    dims = DimsManager(
        world_coordinate_system=world_coordinate_system(
            [
                ("t", "time"),
                ("c", "channel"),
                ("z", "space"),
                ("y", "space"),
                ("x", "space"),
            ],
            name="world",
        ),
        selection=AxisAlignedSelection(
            displayed_axes=(2, 3, 4),
            slice_indices={0: 5, 1: 1, 2: 0, 3: 0, 4: 0},
        ),
    )
    state = dims.to_state()
    assert state.axis_labels == ("t", "c", "z", "y", "x")
    assert state.selection.displayed_axes == (2, 3, 4)
