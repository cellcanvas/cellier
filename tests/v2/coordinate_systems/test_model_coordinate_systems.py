"""The world and data coordinate systems, and what they must survive.

Phase 2 of the ``transform`` integration adds coordinate systems to the
model and nothing that reads them.  These tests pin the three things that
would otherwise be discovered much later:

* **ids survive a round trip.**  Every stored transform names its endpoints
  by UUID, so a system rebuilt on load with fresh ids silently orphans them.
* **axis types are stated, never guessed** -- for the world by requiring the
  caller to say, for a store by inheriting from the world the caller already
  declared.
* **the runtime caches invalidate on the right events**, including a pure
  reorder of ``displayed_axes``, which is a different rendered system even
  though it fetches identical data.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier._state import AxisAlignedSelectionState
from cellier.controller import CellierController
from cellier.data._axes import axis_types_from_names, build_axes, data_axes_from_world
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.render.visuals._slicing import (
    axis_selections_from_box,
    round_world_to_voxel,
)
from cellier.scene.dims import (
    DEFAULT_HALF_THICKNESS,
    AxisAlignedSelection,
    DimsManager,
    spatial_axes,
    world_coordinate_system,
)
from cellier.transform import (
    Axis,
    AxisAlignedBoundingBox,
    WorldCoordinateSystem,
)
from cellier.viewer_model import DataManager, ViewerModel
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import ImageVisual, InMemoryImageAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual
from tests._v2 import data_system, systems

# ---------------------------------------------------------------------------
# The world: types are stated
# ---------------------------------------------------------------------------


def test_a_bare_axis_name_is_rejected_and_names_the_fix():
    """``Axis.axis_type`` has no default, so neither does the world (D3)."""
    with pytest.raises(TypeError, match="spatial_axes"):
        world_coordinate_system(("z", "y", "x"))


def test_name_and_type_pairs_build_a_mixed_world():
    world = world_coordinate_system(
        [("t", "time"), ("c", "channel"), *spatial_axes("z", "y", "x")]
    )
    assert world.axis_names() == ("t", "c", "z", "y", "x")
    assert [axis.axis_type for axis in world.axes] == [
        "time",
        "channel",
        "space",
        "space",
        "space",
    ]


def test_an_existing_world_passes_through_with_its_ids_intact():
    """Rebuilding would mint fresh ids and orphan every stored transform."""
    original = world_coordinate_system(spatial_axes("z", "y", "x"))
    assert world_coordinate_system(original) is original


def test_axis_labels_is_derived_and_unchanged_for_the_gui():
    dims = DimsManager(
        world_coordinate_system=world_coordinate_system(spatial_axes("z", "y", "x")),
        selection=AxisAlignedSelection(
            displayed_axes=(0, 1, 2), slice_indices={0: 0.0, 1: 0.0, 2: 0.0}
        ),
    )
    assert dims.axis_labels == ("z", "y", "x")
    assert dims.to_state().axis_labels == ("z", "y", "x")


# ---------------------------------------------------------------------------
# slice_indices is a world position; thickness is a world half-thickness
# ---------------------------------------------------------------------------


def test_a_fractional_slice_position_is_kept(tmp_path):
    """The bug D3 fixes: on a 0.5 unit-per-voxel axis, half the planes were
    unreachable through an integer-valued slider."""
    selection = AxisAlignedSelection(displayed_axes=(1, 2), slice_indices={0: 2.5})
    assert selection.slice_indices[0] == 2.5
    # The snapshot stopped carrying the positions in Phase 8 (D5); what
    # reaches the render layer is the region built from them, and nothing on
    # the way rounds or truncates.
    assert not hasattr(selection.to_state(), "slice_indices")


def test_a_float_slice_position_reaches_the_voxel_mapper_unchanged():
    """``round_world_to_voxel`` must survive this migration untouched.

    Phase 8 deleted ``map_world_slice_to_voxel``, the v1-era wrapper this
    originally went through; the surviving assembler is
    ``axis_selections_from_box``, which calls the same rule.  A world position
    of 2.5 on a 0.5 unit-per-voxel axis is voxel 5.0 exactly, and half-up
    keeps 5.
    """
    data, _ = systems(3)
    box = AxisAlignedBoundingBox(
        coordinate_system=data.id,
        min_coordinate=np.asarray([5.0, -np.inf, -np.inf]),
        max_coordinate=np.asarray([5.0, np.inf, np.inf]),
    )
    selections = axis_selections_from_box(box, (10, 10, 10))
    assert selections[0] == round_world_to_voxel(5.0, 10) == 5


def test_an_absent_axis_gets_the_default_half_thickness():
    selection = AxisAlignedSelection(displayed_axes=(1, 2), slice_indices={0: 0.0})
    assert selection.half_thickness(0) == DEFAULT_HALF_THICKNESS
    assert DEFAULT_HALF_THICKNESS == 0.5


def test_thickness_is_per_axis():
    """One number means three frames on a time axis and a quarter of a voxel
    on a 2 um spatial one, which is why D4 is a mapping."""
    selection = AxisAlignedSelection(
        displayed_axes=(2, 3),
        slice_indices={0: 0.0, 1: 0.0},
        thickness={0: 1.5},
    )
    assert selection.half_thickness(0) == 1.5
    assert selection.half_thickness(1) == DEFAULT_HALF_THICKNESS


def test_a_negative_half_thickness_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        AxisAlignedSelection(
            displayed_axes=(1, 2), slice_indices={0: 0.0}, thickness={0: -1.0}
        )


def test_the_selection_state_carries_thickness_through_to_the_render_layer():
    selection = AxisAlignedSelection(
        displayed_axes=(1, 2), slice_indices={0: 3.0}, thickness={0: 2.0}
    )
    state = selection.to_state()
    assert isinstance(state, AxisAlignedSelectionState)
    assert state.thickness == {0: 2.0}


def test_a_state_built_without_thickness_still_works():
    """Every render-layer construction site predates the field."""
    state = AxisAlignedSelectionState(
        displayed_axes=(1, 2),
    )
    assert state.thickness == {}


# ---------------------------------------------------------------------------
# Data coordinate systems on the stores
# ---------------------------------------------------------------------------


def test_a_store_has_no_coordinate_systems_until_it_is_told():
    """F1.4: no auto-build.  ``Axis.axis_type`` has no honest default, and a
    store with no metadata has nowhere to get one from on its own."""
    store = ImageMemoryStore(data=np.zeros((4, 5, 6), dtype=np.float32))
    assert store.data_coordinate_systems == []
    with pytest.raises(ValueError, match="data_coordinate_systems"):
        _ = store.data_coordinate_system


def test_a_store_adopts_the_id_its_system_names():
    """The system is built before the store, so the store takes its id."""
    system = data_system(("z", "y", "x"))
    store = ImageMemoryStore(
        data=np.zeros((4, 5, 6), dtype=np.float32), data_coordinate_systems=[system]
    )
    assert store.id == system.datastore_id
    assert store.data_coordinate_system.id == system.id
    assert store.level_transforms[0].matrix.shape == (4, 4)


def test_a_system_naming_another_store_is_refused():
    with pytest.raises(ValueError, match="belongs to datastore"):
        ImageMemoryStore(
            data=np.zeros((4, 5, 6), dtype=np.float32),
            id=uuid4(),
            data_coordinate_systems=[data_system(("z", "y", "x"))],
        )


def test_a_system_of_the_wrong_rank_is_refused():
    """One axis per data dimension, checked when the store is built."""
    with pytest.raises(ValueError, match="has 2 axes"):
        PointsMemoryStore(
            positions=np.zeros((4, 3), dtype=np.float32),
            data_coordinate_systems=[data_system(("y", "x"))],
        )


def test_the_name_rule_types_the_conventional_names():
    assert axis_types_from_names(("t", "c", "z", "y", "x")) == (
        "time",
        "channel",
        "space",
        "space",
        "space",
    )


def test_explicit_axis_types_win_over_the_name_rule():
    axes = build_axes(("c", "y", "x"), types=("space", "space", "space"))
    assert [a.axis_type for a in axes] == ["space"] * 3


def test_an_empty_axis_type_raises_and_names_the_axis():
    """A blank ``type`` in NGFF is a dataset defect, not a default (F1.4)."""
    with pytest.raises(ValueError, match="'y' has an empty axis_type"):
        build_axes(("y", "x"), types=("", "space"))


def test_a_store_added_to_a_scene_takes_the_worlds_trailing_axes():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
        dim="3d",
    )
    store = ImageMemoryStore(data=np.zeros((4, 5, 6), dtype=np.float32))
    controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
    )
    system = store.data_coordinate_system
    assert system.axis_names() == ("z", "y", "x")
    assert [axis.axis_type for axis in system.axes] == ["space"] * 3
    # Fresh ids: these are the axes of a different coordinate system.
    world = scene.dims.world_coordinate_system
    assert {axis.id for axis in system.axes}.isdisjoint(
        {axis.id for axis in world.axes}
    )


def test_a_store_that_already_knows_its_axes_is_left_alone():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=spatial_axes("z", "y", "x"))
    store = ImageMemoryStore(
        data=np.zeros((4, 5, 6), dtype=np.float32),
        data_coordinate_systems=[data_system(("depth", "row", "col"))],
    )
    before = store.data_coordinate_system.id
    controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
    )
    assert store.data_coordinate_system.id == before
    assert store.data_coordinate_system.axis_names() == ("depth", "row", "col")


def test_a_declared_axis_sits_alongside_the_inherited_ones():
    """A multichannel store's channel axis has no world counterpart, and the
    visual is the only object that knows which axis it is."""
    world = world_coordinate_system(spatial_axes("z", "y", "x"))
    axes = data_axes_from_world(world, 4, {0: ("c", "channel")})
    assert [axis.name for axis in axes] == ["c", "z", "y", "x"]
    assert [axis.axis_type for axis in axes] == ["channel", "space", "space", "space"]


def test_a_store_wider_than_its_world_is_refused_with_a_reason():
    world = world_coordinate_system(spatial_axes("y", "x"))
    with pytest.raises(ValueError, match="Give the scene more"):
        data_axes_from_world(world, 4)


# ---------------------------------------------------------------------------
# Serialization: the ids are the point
# ---------------------------------------------------------------------------


def _viewer_model_with_every_memory_store() -> ViewerModel:
    dims = DimsManager(
        world_coordinate_system=world_coordinate_system(spatial_axes("z", "y", "x")),
        selection=AxisAlignedSelection(
            displayed_axes=(1, 2),
            slice_indices={0: 2.5, 1: 0.0, 2: 0.0},
            thickness={0: 1.25},
        ),
    )
    from cellier.scene.scene import Scene

    positions = np.zeros((4, 3), dtype=np.float32)
    stores = [
        ImageMemoryStore(
            data=np.zeros((4, 5, 6), dtype=np.float32),
            data_coordinate_systems=[data_system(("z", "y", "x"))],
        ),
        LabelMemoryStore(
            data=np.zeros((4, 5, 6), dtype=np.int32),
            data_coordinate_systems=[data_system(("z", "y", "x"))],
        ),
        PointsMemoryStore(
            positions=positions, data_coordinate_systems=[data_system(("z", "y", "x"))]
        ),
        LinesMemoryStore(
            positions=positions, data_coordinate_systems=[data_system(("z", "y", "x"))]
        ),
        MeshMemoryStore(
            positions=positions,
            indices=np.zeros((1, 3), dtype=np.int32),
            data_coordinate_systems=[data_system(("z", "y", "x"))],
        ),
    ]
    image_visual = ImageVisual(
        name="image",
        data_store_id=str(stores[0].id),
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
    )
    points_visual = PointsVisual(
        name="points",
        data_store_id=str(stores[2].id),
        appearance=PointsMarkerAppearance(),
    )
    scene = Scene(name="s", dims=dims, visuals=[image_visual, points_visual])
    return ViewerModel(
        data=DataManager(stores={store.id: store for store in stores}),
        scenes={scene.id: scene},
    )


def test_every_store_type_round_trips_with_its_axis_ids_intact():
    original = _viewer_model_with_every_memory_store()
    restored = ViewerModel.model_validate_json(original.model_dump_json())

    for store_id, store in original.data.stores.items():
        other = restored.data.stores[store_id]
        assert len(other.data_coordinate_systems) == len(store.data_coordinate_systems)
        for mine, theirs in zip(
            store.data_coordinate_systems, other.data_coordinate_systems
        ):
            assert theirs.id == mine.id
            assert [a.id for a in theirs.axes] == [a.id for a in mine.axes]
            assert [a.axis_type for a in theirs.axes] == [
                a.axis_type for a in mine.axes
            ]


def test_the_world_round_trips_with_its_axis_ids_intact():
    original = _viewer_model_with_every_memory_store()
    restored = ViewerModel.model_validate_json(original.model_dump_json())
    (scene_id,) = original.scenes
    mine = original.scenes[scene_id].dims.world_coordinate_system
    theirs = restored.scenes[scene_id].dims.world_coordinate_system
    assert theirs.id == mine.id
    assert [axis.id for axis in theirs.axes] == [axis.id for axis in mine.axes]


def test_slice_positions_and_thicknesses_survive_a_round_trip():
    original = _viewer_model_with_every_memory_store()
    restored = ViewerModel.model_validate_json(original.model_dump_json())
    (scene_id,) = original.scenes
    selection = restored.scenes[scene_id].dims.selection
    assert selection.slice_indices == {0: 2.5, 1: 0.0, 2: 0.0}
    assert selection.thickness == {0: 1.25}


def test_a_world_built_from_axis_objects_keeps_them():
    axes = (Axis(name="q", axis_type="coordinate"), *spatial_axes("y", "x"))
    world = world_coordinate_system(axes)
    assert isinstance(world, WorldCoordinateSystem)
    assert [axis.id for axis in world.axes] == [axis.id for axis in axes]
