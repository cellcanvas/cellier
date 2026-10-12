"""The visuals the appearance-spec parity golden was captured from.

``plans/ordered_controls_list_implementation.md`` (P0-2).  The golden,
``data/appearance_specs_golden.json``, holds what ``appearance_specs``
returned for each case before the controls list replaced the config classes.
It was captured once and is never regenerated: the parity test builds the same
visuals here, asks for the same controls as a list, and compares.

Each case is a real visual added through a ``Viewer``, so the store and the
palette are the ones a dock would be given.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import UUID

import numpy as np
import tensorstore as ts

if TYPE_CHECKING:
    from collections.abc import Callable

GOLDEN_PATH = Path(__file__).parent / "data" / "appearance_specs_golden.json"

#: What a temporary directory is written as, so the golden does not hold one.
_TMP = "<tmp>"

_UUID_TEXT = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


class ParityCase(NamedTuple):
    """One visual of the golden, and what ``appearance_specs`` is given."""

    viewer: Any
    visual: Any
    store: Any
    palette: Any


def _write_pyramid(root: Path, dtype: str, fill: Callable | None = None) -> None:
    """Write a two-level zarr v3 pyramid (16 cubed, 8 cubed) under *root*."""
    for name, size in (("s0", 16), ("s1", 8)):
        shape = (size, size, size)
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / name)},
            "metadata": {
                "shape": list(shape),
                "data_type": dtype,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [8, 8, 8]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        array = np.zeros(shape, dtype=dtype)
        if fill is not None:
            fill(array)
        ts.open(spec).result()[...].write(array).result()


def _pyramid_store(root: Path, name: str):
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore

    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        name=name,
    )


def _viewer(axes=None):
    from cellier.convenience import Viewer
    from cellier.scene.dims import spatial_axes
    from tests._displayed import trailing_axes

    world = spatial_axes("z", "y", "x") if axes is None else axes
    return Viewer(world, displayed_axes=trailing_axes(world, "2d"), gui="offscreen")


def _tetrahedron():
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return MeshMemoryStore(positions=positions, indices=indices, name="mesh")


def _image(tmp: Path):
    from cellier.data.image._image_memory_store import ImageMemoryStore

    viewer = _viewer()
    store = ImageMemoryStore(data=np.zeros((8, 16, 24), dtype=np.float32), name="image")
    return viewer, viewer.add_image(store)


def _multiscale_image(tmp: Path):
    root = tmp / "multiscale_image"
    _write_pyramid(root, "float32")
    viewer = _viewer()
    return viewer, viewer.add_image_multiscale(_pyramid_store(root, "multiscale_image"))


def _labels(tmp: Path):
    from cellier.data.label._label_memory_store import LabelMemoryStore

    data = np.zeros((8, 16, 16), dtype=np.int32)
    data[:, 4:10, 4:10] = 5
    viewer = _viewer()
    return viewer, viewer.add_labels(LabelMemoryStore(data=data, name="labels"))


def _multiscale_labels(tmp: Path):
    from cellier.visuals._labels import MultiscaleLabelsAppearance

    def fill(array: np.ndarray) -> None:
        _depth, height, width = array.shape
        array[:, height // 4 : height // 2, width // 4 : width // 2] = 3

    root = tmp / "multiscale_labels"
    _write_pyramid(root, "int32", fill)
    viewer = _viewer()
    return viewer, viewer.add_labels_multiscale(
        _pyramid_store(root, "multiscale_labels"),
        appearance=MultiscaleLabelsAppearance(),
    )


def _mesh_flat(tmp: Path):
    from cellier.visuals import MeshFlatAppearance

    viewer = _viewer()
    return viewer, viewer.add_mesh(_tetrahedron(), appearance=MeshFlatAppearance())


def _mesh_phong(tmp: Path):
    from cellier.visuals import MeshPhongAppearance

    viewer = _viewer()
    return viewer, viewer.add_mesh(_tetrahedron(), appearance=MeshPhongAppearance())


def _multiscale_mesh(tmp: Path):
    from cellier.data.mesh import MeshLevel, MultiscaleMeshStore
    from cellier.visuals import MeshFlatAppearance
    from tests._meshes import uv_sphere

    levels = [
        MeshLevel(positions=positions, indices=indices)
        for positions, indices in (
            uv_sphere(5.0, (8.0, 8.0, 8.0), n_lat=n, n_lon=2 * n) for n in (8, 4)
        )
    ]
    viewer = _viewer()
    return viewer, viewer.add_multiscale_mesh(
        MultiscaleMeshStore(levels=levels, name="sphere"), MeshFlatAppearance()
    )


def _points(tmp: Path):
    from cellier.data.points._points_memory_store import PointsMemoryStore

    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    viewer = _viewer()
    return viewer, viewer.add_points(PointsMemoryStore(positions=positions))


def _lines(tmp: Path):
    from cellier.data.lines._lines_memory_store import LinesMemoryStore

    positions = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 2], [1, 1, 3]], dtype=np.float32)
    viewer = _viewer()
    return viewer, viewer.add_lines(LinesMemoryStore(positions=positions))


def _graph(tmp: Path):
    """A chain on ``(t, z, y, x)`` with a trail window on ``y``."""
    from uuid import uuid4

    from cellier.data import GraphMemoryStore
    from cellier.scene.dims import spatial_axes
    from cellier.transform import Axis, DataCoordinateSystem
    from cellier.visuals import GraphAppearance, TrailConfig

    positions = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    axes = (
        Axis(name="t", axis_type="time", sampling="discrete"),
        *(Axis(name=name, axis_type="space") for name in ("z", "y", "x")),
    )
    viewer = _viewer([Axis(name="t", axis_type="time"), *spatial_axes("z", "y", "x")])
    store = GraphMemoryStore.from_arrays(
        positions,
        edges,
        data_coordinate_system=DataCoordinateSystem(
            name="tracks", axes=axes, datastore_id=uuid4()
        ),
        name="tracks",
    )
    return viewer, viewer.add_graph(
        data=store,
        appearance=GraphAppearance(),
        name="tracks",
        trail={2: TrailConfig()},
    )


CASE_BUILDERS: dict[str, Callable[[Path], tuple]] = {
    "image": _image,
    "multiscale_image": _multiscale_image,
    "labels": _labels,
    "multiscale_labels": _multiscale_labels,
    "mesh_flat": _mesh_flat,
    "mesh_phong": _mesh_phong,
    "multiscale_mesh": _multiscale_mesh,
    "points": _points,
    "lines": _lines,
    "graph": _graph,
}
"""Case name to a function that adds its visual to a new offscreen viewer."""


def build_case(name: str, tmp: Path) -> ParityCase:
    """Build the visual of case *name*, writing any data under *tmp*.

    The caller closes ``case.viewer.controller``.
    """
    from cellier.convenience.layout._shared import _resolve_data_store

    viewer, visual = CASE_BUILDERS[name](tmp)
    controller = viewer.controller
    return ParityCase(
        viewer,
        visual,
        _resolve_data_store(controller, visual),
        controller.render_config.outline.palette,
    )


def normalise(value: Any, tmp: Path | None = None) -> Any:
    """Turn spec values into plain JSON data, the same way every time.

    Used when the golden was captured and when a test compares against it,
    so both sides go through one function.  A type it does not know raises:
    a value that is silently turned into its ``repr`` would compare equal to
    nothing, or to everything.
    """
    from pydantic import BaseModel

    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str):
            # Ids are new on every run, as objects or already as text.
            value = _UUID_TEXT.sub("<uuid>", value)
            if tmp is not None:
                value = value.replace(str(tmp), _TMP)
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, enum.Enum):
        return normalise(value.value, tmp)
    if isinstance(value, UUID):
        return "<uuid>"
    if isinstance(value, Path):
        return normalise(str(value), tmp)
    if isinstance(value, np.generic):
        return normalise(value.item(), tmp)
    if isinstance(value, np.ndarray):
        return normalise(value.tolist(), tmp)
    if type(value).__name__ == "Colormap" and hasattr(value, "name"):
        # A ``cmap.Colormap``: its name says which one it is.
        return {"colormap": str(value.name)}
    if isinstance(value, BaseModel):
        return normalise(value.model_dump(), tmp)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: normalise(getattr(value, field.name), tmp)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): normalise(item, tmp) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(item, tmp) for item in value]
    raise TypeError(f"no JSON form for {type(value).__name__}: {value!r}")


def specs_as_json(specs: list, tmp: Path | None = None) -> list[dict]:
    """The ``ControlSpec`` list as the golden stores it."""
    return [
        {
            "kind": spec.kind,
            "title": spec.title,
            "values": normalise(spec.values, tmp),
        }
        for spec in specs
    ]
