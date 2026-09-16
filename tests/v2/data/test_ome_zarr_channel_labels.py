"""``OMEZarrImageDataStore.channel_labels``, read from the ``omero`` block."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from cellier.data import OMEZarrImageDataStore

if TYPE_CHECKING:
    import pathlib

_AXES = [
    {"name": "c", "type": "channel"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]
_SHAPE = (2, 4, 4)


def _channel(label: str | None = None) -> dict:
    """One omero channel; ``color`` and ``window`` are required by the schema."""
    channel = {
        "color": "FF00FF",
        "window": {"min": 0.0, "max": 1.0, "start": 0.0, "end": 1.0},
    }
    if label is not None:
        channel["label"] = label
    return channel


def _write_image(root: pathlib.Path, omero: dict | None = None) -> str:
    """A single-level ``cyx`` OME-Zarr v0.5 image, with an optional omero block."""
    import zarr

    root.mkdir(parents=True)
    ome = {
        "version": "0.5",
        "multiscales": [
            {
                "name": "test",
                "version": "0.5",
                "axes": _AXES,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                        ],
                    }
                ],
            }
        ],
    }
    if omero is not None:
        ome["omero"] = omero
    (root / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}})
    )
    array = zarr.create(
        store=zarr.storage.LocalStore(str(root / "0")),
        shape=_SHAPE,
        dtype="float32",
        chunks=_SHAPE,
        zarr_format=3,
    )
    array[...] = np.zeros(_SHAPE, dtype=np.float32)
    return f"file://{root}"


def test_channel_labels_come_from_the_omero_block(tmp_path):
    uri = _write_image(
        tmp_path / "image.ome.zarr",
        omero={"channels": [_channel("mem9"), _channel("H2B")]},
    )

    store = OMEZarrImageDataStore.from_path(uri)

    assert store.channel_labels == ["mem9", "H2B"]


def test_an_unlabelled_channel_is_called_by_its_index(tmp_path):
    uri = _write_image(
        tmp_path / "image.ome.zarr",
        omero={"channels": [_channel("mem9"), _channel()]},
    )

    store = OMEZarrImageDataStore.from_path(uri)

    assert store.channel_labels == ["mem9", "1"]


def test_an_image_without_an_omero_block_has_no_channel_labels(tmp_path):
    store = OMEZarrImageDataStore.from_path(_write_image(tmp_path / "image.ome.zarr"))

    assert store.channel_labels is None


def test_channel_labels_survive_serialisation(tmp_path):
    uri = _write_image(
        tmp_path / "image.ome.zarr",
        omero={"channels": [_channel("mem9"), _channel("H2B")]},
    )
    store = OMEZarrImageDataStore.from_path(uri)

    restored = OMEZarrImageDataStore.model_validate(store.model_dump())

    assert restored.channel_labels == ["mem9", "H2B"]
