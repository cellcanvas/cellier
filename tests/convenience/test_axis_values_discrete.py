"""Discrete slider values from ``axis_values_from_viewer``.

A non-space world axis that every visual reaching it samples discretely -- an
OME-Zarr channel or time axis -- gets a ``DiscreteAxisValues`` stepping
through those samples, labelled with the store's channel names when they are
known.  Space axes, and anything a continuous store also reaches, keep the
continuous range they always had.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience import (
    ContinuousAxisValues,
    DiscreteAxisValues,
    OrthoViewer,
    Viewer,
    axis_values_from_ortho,
    axis_values_from_viewer,
)
from cellier.convenience import _geometry as geometry
from cellier.data import ImageMemoryStore, OMEZarrImageDataStore, PointsMemoryStore
from cellier.transform import AffineTransform, Axis, DataCoordinateSystem
from cellier.visuals import (
    InMemoryImageAppearance,
    InMemoryImageSingleAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageSingleAppearance,
)

if TYPE_CHECKING:
    import pathlib

_WORLD = [
    ("t", "time"),
    ("c", "channel"),
    ("z", "space"),
    ("y", "space"),
    ("x", "space"),
]
_TYPES = dict(_WORLD)
_SHAPE = (3, 2, 4, 6, 8)


def _write_ome_zarr(
    root: pathlib.Path, *, labels: tuple[str, ...] | None = None, shape=_SHAPE
) -> str:
    """A single-level ``tczyx`` OME-Zarr v0.5 image, named channels optional."""
    import zarr

    root.mkdir(parents=True)
    ome = {
        "version": "0.5",
        "multiscales": [
            {
                "name": "test",
                "version": "0.5",
                "axes": [{"name": name, "type": kind} for name, kind in _WORLD],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0] * len(_WORLD)}
                        ],
                    }
                ],
            }
        ],
    }
    if labels is not None:
        ome["omero"] = {
            "channels": [
                {
                    "label": label,
                    "color": "FFFFFF",
                    "window": {"min": 0.0, "max": 1.0, "start": 0.0, "end": 1.0},
                }
                for label in labels
            ]
        }
    (root / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}})
    )
    array = zarr.create(
        store=zarr.storage.LocalStore(str(root / "0")),
        shape=shape,
        dtype="float32",
        chunks=shape,
        zarr_format=3,
    )
    array[...] = np.zeros(shape, dtype=np.float32)
    return f"file://{root}"


def _ome_store(tmp_path, name: str = "image.ome.zarr", **kwargs):
    return OMEZarrImageDataStore.from_path(_write_ome_zarr(tmp_path / name, **kwargs))


def _add_ome(viewer, store, transform=None):
    return viewer.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(),
        transform=transform,
        single=MultiscaleImageSingleAppearance(color_map="grays"),
    )


def test_an_ome_zarr_channel_axis_is_discrete_and_labelled(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path, labels=("mem9", "H2B")))

    values = axis_values_from_viewer(viewer)

    assert values[1] == DiscreteAxisValues(values=(0.0, 1.0), labels=("mem9", "H2B"))


def test_an_ome_zarr_time_axis_is_discrete(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))

    assert axis_values_from_viewer(viewer)[0] == DiscreteAxisValues(
        values=(0.0, 1.0, 2.0)
    )


def test_space_axes_stay_continuous_though_the_grid_is_discrete(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))

    values = axis_values_from_viewer(viewer)

    assert values[2] == ContinuousAxisValues(min=-0.5, max=3.5)
    assert values[3] == ContinuousAxisValues(min=-0.5, max=5.5)
    assert values[4] == ContinuousAxisValues(min=-0.5, max=7.5)


def test_without_channel_names_the_channel_axis_is_unlabelled(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))

    assert axis_values_from_viewer(viewer)[1] == DiscreteAxisValues(values=(0.0, 1.0))


def test_the_transform_places_the_samples(tmp_path):
    viewer = Viewer(_WORLD)
    store = _ome_store(tmp_path)
    world = viewer.scene.dims.world_coordinate_system
    transform = AffineTransform.from_axis_map(
        store.data_coordinate_systems[0],
        world,
        axis_map={name: name for name, _kind in _WORLD},
        scale={"t": 2.5},
    )
    _add_ome(viewer, store, transform=transform)

    assert axis_values_from_viewer(viewer)[0] == DiscreteAxisValues(
        values=(0.0, 2.5, 5.0)
    )


def test_stores_with_different_frame_counts_take_the_union(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path, labels=("mem9", "H2B")))
    longer = ImageMemoryStore(
        data=np.zeros((5, 2, 4, 6, 8), dtype=np.float32),
        data_coordinate_systems=[
            DataCoordinateSystem(
                name="longer",
                axes=tuple(
                    Axis(name=name, axis_type=kind, sampling="discrete")
                    for name, kind in _WORLD
                ),
                datastore_id=uuid4(),
            )
        ],
    )
    viewer.add_image(
        longer,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays"),
    )

    values = axis_values_from_viewer(viewer)

    assert values[0] == DiscreteAxisValues(values=(0.0, 1.0, 2.0, 3.0, 4.0))
    # The unnamed store adds no new channel position, so the names still hold.
    assert values[1] == DiscreteAxisValues(values=(0.0, 1.0), labels=("mem9", "H2B"))


def test_one_continuous_contributor_keeps_the_axis_continuous(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))
    # Axes inherited from the world are continuous, like any bare store's.
    points = PointsMemoryStore(
        positions=np.array([[0.0, 0.0, 1.0, 1.0, 1.0], [2.0, 1.0, 2.0, 2.0, 2.0]])
    )
    viewer.add_points(points)

    values = axis_values_from_viewer(viewer)

    assert values[0] == ContinuousAxisValues(min=-0.5, max=2.5)
    assert values[1] == ContinuousAxisValues(min=-0.5, max=1.5)


def test_disagreeing_channel_names_drop_the_labels(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path, "a.ome.zarr", labels=("mem9", "H2B")))
    _add_ome(viewer, _ome_store(tmp_path, "b.ome.zarr", labels=("dapi", "gfp")))

    assert axis_values_from_viewer(viewer)[1] == DiscreteAxisValues(values=(0.0, 1.0))


def test_too_many_samples_fall_back_to_continuous(tmp_path, monkeypatch):
    monkeypatch.setattr(geometry, "MAX_DISCRETE_VALUES", 2)
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))

    values = axis_values_from_viewer(viewer)

    assert values[0] == ContinuousAxisValues(min=-0.5, max=2.5)  # three frames
    assert isinstance(values[1], DiscreteAxisValues)  # two channels


def test_an_in_memory_image_keeps_continuous_sliders():
    """A bare store inherits continuous axes from the world: nothing changes."""
    viewer = Viewer(_WORLD)
    viewer.add_image(
        ImageMemoryStore(data=np.zeros(_SHAPE, dtype=np.float32)),
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays"),
    )

    values = axis_values_from_viewer(viewer)

    assert all(isinstance(entry, ContinuousAxisValues) for entry in values.values())
    assert _TYPES["t"] == "time"


# ---------------------------------------------------------------------------
# draw_ticks
# ---------------------------------------------------------------------------


def test_no_axis_draws_ticks_by_default(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))

    values = axis_values_from_viewer(viewer)

    assert values[0].draw_ticks is False
    assert values[1].draw_ticks is False


def test_draw_ticks_marks_only_the_named_axes(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path, labels=("mem9", "H2B")))

    values = axis_values_from_viewer(viewer, draw_ticks=["c"])

    # The rest of the entry -- values and channel names -- is unchanged.
    assert values[1] == DiscreteAxisValues(
        values=(0.0, 1.0), labels=("mem9", "H2B"), draw_ticks=True
    )
    assert values[0].draw_ticks is False


def test_draw_ticks_names_a_continuous_axis_raises(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))

    with pytest.raises(ValueError, match=r"'z'.*continuous slider"):
        axis_values_from_viewer(viewer, draw_ticks=["z"])


def test_draw_ticks_raises_when_the_data_makes_the_axis_continuous(tmp_path):
    """Discreteness is the data's call: one continuous store undoes it."""
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))
    viewer.add_points(
        PointsMemoryStore(
            positions=np.array([[0.0, 0.0, 1.0, 1.0, 1.0], [2.0, 1.0, 2.0, 2.0, 2.0]])
        )
    )

    with pytest.raises(ValueError, match=r"'c'.*continuous slider"):
        axis_values_from_viewer(viewer, draw_ticks=["c"])


def test_draw_ticks_rejects_an_unknown_axis_name(tmp_path):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))

    with pytest.raises(ValueError, match=r"'q'.*not a world axis"):
        axis_values_from_viewer(viewer, draw_ticks=["q"])


def test_draw_ticks_checks_names_before_measuring():
    """A bad name is reported even on a viewer with nothing to measure."""
    with pytest.raises(ValueError, match="not a world axis"):
        axis_values_from_viewer(Viewer(_WORLD), draw_ticks=["q"])


@pytest.mark.parametrize(
    ("draw_ticks", "match"),
    [("c", r"draw_ticks=\['c'\]"), ([1], "axis names"), (True, "not iterable")],
)
def test_draw_ticks_takes_names_only(tmp_path, draw_ticks, match):
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))

    with pytest.raises(TypeError, match=match):
        axis_values_from_viewer(viewer, draw_ticks=draw_ticks)


def test_draw_ticks_on_a_long_axis_still_warns(tmp_path, monkeypatch):
    from cellier.gui import _axis_values

    monkeypatch.setattr(_axis_values, "TICK_WARNING_LIMIT", 2)
    viewer = Viewer(_WORLD)
    _add_ome(viewer, _ome_store(tmp_path))  # three frames

    with pytest.warns(UserWarning):
        axis_values_from_viewer(viewer, draw_ticks=["t"])


def test_ortho_draw_ticks(tmp_path):
    ortho = OrthoViewer(_WORLD, gui="offscreen")
    ortho.add_image_multiscale(
        _ome_store(tmp_path),
        appearance=MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="grays"),
    )

    values = axis_values_from_ortho(ortho, draw_ticks=["c"])

    assert values[1].draw_ticks is True
    assert values[0].draw_ticks is False
    # A bad name is not mistaken for "no data on this panel, try the next".
    with pytest.raises(ValueError, match="not a world axis"):
        axis_values_from_ortho(ortho, draw_ticks=["q"])
