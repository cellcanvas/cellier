"""The OME-Zarr readers hand their axis metadata to a coordinate system.

NGFF already carries names, types and units, and cellier discarded all but
the names.  Populating ``data_coordinate_systems`` from them is close to free
and is a strict information gain -- and it is the one store family that can
answer "what type is this axis" without asking the scene.

The systems are the stores' only record of that metadata, so a caller's own
level-0 system replaces it outright rather than sitting beside it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from cellier.data.image._ome_zarr_image_store import OMEZarrImageDataStore
from cellier.data.label._ome_zarr_label_store import OMEZarrLabelDataStore
from cellier.transform import DataCoordinateSystem
from tests._v2 import data_system, pyramid_levels

readers = pytest.mark.parametrize(
    "reader",
    [OMEZarrImageDataStore, OMEZarrLabelDataStore],
    ids=["image", "labels"],
)


def _blank_the_z_type(uri: str) -> None:
    """Rewrite the fixture's metadata so the ``z`` axis has an empty type."""
    meta_path = pathlib.Path(uri.removeprefix("file://")) / "zarr.json"
    meta = json.loads(meta_path.read_text())
    meta["attributes"]["ome"]["multiscales"][0]["axes"][2]["type"] = ""
    meta_path.write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# Built from the NGFF metadata
# ---------------------------------------------------------------------------


@readers
def test_the_level_zero_system_mirrors_the_ngff_axes(reader, ome_zarr_5d: str) -> None:
    store = reader.from_path(ome_zarr_5d)
    system = store.data_coordinate_system
    assert isinstance(system, DataCoordinateSystem)
    assert system.axis_names() == ("t", "c", "z", "y", "x")
    assert [axis.axis_type for axis in system.axes] == [
        "time",
        "channel",
        "space",
        "space",
        "space",
    ]
    assert [axis.unit for axis in system.axes] == [
        "second",
        None,
        "micrometer",
        "micrometer",
        "micrometer",
    ]
    assert system.datastore_id == store.id


@readers
def test_there_is_one_system_per_pyramid_level(reader, ome_zarr_5d: str) -> None:
    """A level-2 voxel is not a level-0 voxel, and a transform between them is
    exactly what says so -- which needs two distinct systems."""
    store = reader.from_path(ome_zarr_5d)
    assert len(store.data_coordinate_systems) == store.n_levels == 3
    ids = [system.id for system in store.data_coordinate_systems]
    assert len(set(ids)) == 3
    level_zero, level_one = store.data_coordinate_systems[:2]
    assert level_one.axis_names() == level_zero.axis_names()
    # Fresh axis ids per level: index_of must not be ambiguous across systems.
    assert {axis.id for axis in level_one.axes}.isdisjoint(
        {axis.id for axis in level_zero.axes}
    )


@readers
def test_every_level_is_a_discrete_grid_of_the_same_store(
    reader, ome_zarr_5d: str
) -> None:
    """Coarser levels used to drop ``sampling`` and fall back to continuous."""
    store = reader.from_path(ome_zarr_5d)
    for system in store.data_coordinate_systems:
        assert [axis.sampling for axis in system.axes] == ["discrete"] * 5
        assert system.datastore_id == store.id


def test_both_readers_derive_the_same_pyramid(ome_zarr_5d: str) -> None:
    """The label reader parses raw dicts but shares the image reader's maths."""
    image = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    labels = OMEZarrLabelDataStore.from_path(ome_zarr_5d)
    assert labels.level_scales == image.level_scales
    assert labels.level_translations == image.level_translations
    assert labels.physical_scale == image.physical_scale
    assert labels.physical_translation == image.physical_translation


def test_the_systems_survive_a_round_trip_with_their_ids(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    restored = OMEZarrImageDataStore.model_validate_json(store.model_dump_json())
    for mine, theirs in zip(
        store.data_coordinate_systems, restored.data_coordinate_systems
    ):
        assert theirs.id == mine.id
        assert [axis.id for axis in theirs.axes] == [axis.id for axis in mine.axes]


def test_an_empty_axis_type_raises_and_names_the_axis(ome_zarr_5d: str) -> None:
    """The metadata had a slot for the type and left it blank, which is a
    dataset defect.  ``AxisType`` is a closed six-value literal with no
    honest fallback, so "an axis type is never defaulted" is taken
    literally.

    Reached through the label reader: the v0.5 image reader runs NGFF
    metadata through ``ome_zarr_models``, which rejects an untyped axis first
    (it can no longer order the axes).  The label reader parses raw dicts and
    has no such gate, so this is the shape a blank type actually arrives in.
    """
    _blank_the_z_type(ome_zarr_5d)
    with pytest.raises(ValueError, match="'z' has an empty axis_type"):
        OMEZarrLabelDataStore.from_path(ome_zarr_5d)


# ---------------------------------------------------------------------------
# A caller's level-0 system
# ---------------------------------------------------------------------------


@readers
def test_a_passed_level_zero_system_replaces_the_metadata(
    reader, ome_zarr_5d: str
) -> None:
    system = data_system(("t", "c", "depth", "row", "col"), sampling="discrete")
    store = reader.from_path(ome_zarr_5d, data_coordinate_system=system)

    assert store.id == system.datastore_id
    assert store.data_coordinate_systems[0].id == system.id
    assert len(store.data_coordinate_systems) == store.n_levels == 3
    assert len(store.level_transforms) == 3
    for level in store.data_coordinate_systems[1:]:
        assert level.axis_names() == system.axis_names()
        assert [a.axis_type for a in level.axes] == [a.axis_type for a in system.axes]
        assert [a.sampling for a in level.axes] == ["discrete"] * 5
        assert level.datastore_id == store.id
        assert {a.id for a in level.axes}.isdisjoint({a.id for a in system.axes})


def test_a_passed_system_is_the_way_past_a_blank_type(ome_zarr_5d: str) -> None:
    _blank_the_z_type(ome_zarr_5d)
    system = data_system(("t", "c", "z", "y", "x"), sampling="discrete")
    store = OMEZarrLabelDataStore.from_path(ome_zarr_5d, data_coordinate_system=system)
    assert store.data_coordinate_system.id == system.id


@readers
def test_a_passed_system_of_the_wrong_rank_is_refused(reader, ome_zarr_5d: str) -> None:
    with pytest.raises(ValueError, match="has 3 axes"):
        reader.from_path(
            ome_zarr_5d, data_coordinate_system=data_system(("z", "y", "x"))
        )


# ---------------------------------------------------------------------------
# Direct construction
# ---------------------------------------------------------------------------


def _direct_kwargs(uri: str) -> dict:
    return {
        "zarr_path": uri,
        "scale_names": ["0", "1", "2"],
        **pyramid_levels([[1.0] * 5] * 3),
    }


@readers
def test_constructing_without_systems_leaves_them_empty(
    reader, ome_zarr_5d: str
) -> None:
    """Without from_path nothing reads the metadata; the scene fills them in."""
    store = reader(**_direct_kwargs(ome_zarr_5d))
    assert store.data_coordinate_systems == []


@readers
def test_one_system_for_three_levels_is_refused(reader, ome_zarr_5d: str) -> None:
    with pytest.raises(ValueError, match="3 resolution level"):
        reader(
            **_direct_kwargs(ome_zarr_5d),
            data_coordinate_systems=[data_system(("t", "c", "z", "y", "x"))],
        )
