"""Toolkit-neutral description of what a data store holds.

A store describes itself as a list of *sections*, each one of the block
shapes a dataset-info widget knows how to draw:

``RowSection``
    ``(label, value)`` pairs, either inline at the top of the block
    (``label=None``) or under their own collapsible heading.
``MatrixSection``
    A labelled 2-D array, drawn as a table.

Two shapes rather than a fixed field list is what lets one widget serve
every store: a points store and an OME-Zarr image have nothing in common
field by field, but both can say "here are some rows".

The vocabulary lives here, beside the stores that speak it, rather than in
``cellier.gui``: a store must be able to describe itself without importing a
front end.  ``cellier.gui._dataset_info`` re-exports these names for the
widget side.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Union
from urllib.parse import urlparse

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "DatasetInfo",
    "MatrixSection",
    "RowSection",
    "Section",
    "array_extent_row",
    "axis_rows",
    "format_bytes",
    "format_extent",
    "format_scale",
    "format_shape",
    "ome_zarr_dataset_info",
    "source_label",
    "uri_file_name",
    "world_to_data_matrix",
]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RowSection:
    """``(label, value)`` rows, optionally under their own heading.

    Parameters
    ----------
    label : str or None
        Heading for the block.  ``None`` puts the rows inline at the top
        level of the widget, which is where a store's identity rows belong.
    rows : list[tuple[str, str]]
        The rows, in display order.  Both halves are coerced to ``str`` on
        construction: a value is *displayed*, never interpreted as markup,
        so anything read off a store is safe to pass verbatim.
    collapsed : bool
        Whether a labelled block starts closed.  Ignored when ``label`` is
        ``None``.  Bulky detail (per-level shapes, geff properties) starts
        collapsed so the block stays scannable.
    """

    label: str | None
    rows: list[tuple[str, str]]
    collapsed: bool = False

    def __post_init__(self) -> None:
        """Coerce both halves of every row to ``str``."""
        object.__setattr__(
            self, "rows", [(str(label), str(value)) for label, value in self.rows]
        )


@dataclasses.dataclass(frozen=True)
class MatrixSection:
    """A labelled 2-D array, drawn as a table.

    Parameters
    ----------
    label : str
        Heading for the table, e.g. ``"World to data"``.
    matrix : np.ndarray
        The values.  Formatted by the widget, not here, so a front end can
        choose its own precision.
    row_labels, col_labels : list[str]
        Header text.  For a homogeneous affine these are the axis names plus
        a trailing entry for the homogeneous coordinate.
    """

    label: str
    matrix: np.ndarray
    row_labels: list[str]
    col_labels: list[str]


Section = Union[RowSection, MatrixSection]


@dataclasses.dataclass(frozen=True)
class DatasetInfo:
    """Everything a dataset-info widget draws for one data store.

    Parameters
    ----------
    sections : list[Section]
        The blocks, in display order.  Conventionally the first is an
        unlabelled :class:`RowSection` carrying the store's identity.
    """

    sections: list[Section]

    @property
    def rows(self) -> list[tuple[str, str]]:
        """Every row in every :class:`RowSection`, flattened.

        The lossy view, for a front end that can only draw a flat table.  A
        labelled section's rows are prefixed with the label so the flattening
        does not silently merge two blocks into one ambiguous list.
        """
        flat: list[tuple[str, str]] = []
        for section in self.sections:
            if not isinstance(section, RowSection):
                continue
            for label, value in section.rows:
                if section.label is None:
                    flat.append((label, value))
                else:
                    flat.append((f"{section.label}: {label}", value))
        return flat


# ---------------------------------------------------------------------------
# Formatting helpers shared by every store's ``dataset_info``
# ---------------------------------------------------------------------------


def format_shape(shape: Iterable[int]) -> str:
    """Render a shape tuple as ``"64 x 128 x 128"``."""
    return " x ".join(str(int(dim)) for dim in shape)


def format_bytes(n_bytes: int) -> str:
    """Render a byte count with a binary unit, e.g. ``"4.0 MiB"``."""
    size = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


def format_scale(scale: Sequence[float]) -> str:
    """Render a per-axis scale factor, collapsing the isotropic case.

    ``(2.0, 2.0, 2.0)`` becomes ``"2x isotropic"``; an anisotropic factor is
    spelled out per axis.  This is the row the examples used to hardcode as
    the string ``"2x isotropic"`` regardless of what the transform said.
    """
    values = [float(value) for value in scale]
    if not values:
        return ""
    if all(abs(value - values[0]) < 1e-9 for value in values):
        return f"{_number(values[0])}x isotropic"
    return " x ".join(_number(value) for value in values)


def format_extent(minimum: Sequence[float], maximum: Sequence[float]) -> str:
    """Render an axis-aligned bounding box as ``"[0, 63] x [0, 127]"``."""
    return " x ".join(
        f"[{_number(lo)}, {_number(hi)}]"
        for lo, hi in zip(minimum, maximum, strict=True)
    )


def world_to_data_matrix(
    scale: Sequence[float], translation: Sequence[float]
) -> np.ndarray:
    """Build the homogeneous world-to-data affine for a level-0 array.

    Data-to-world is ``world[i] = scale[i] * voxel[i] + translation[i]``, so
    the inverse this returns is
    ``voxel[i] = (world[i] - translation[i]) / scale[i]``.

    Parameters
    ----------
    scale, translation : Sequence[float]
        Physical scale and translation per axis, in data axis order.

    Returns
    -------
    np.ndarray
        ``(n + 1, n + 1)`` homogeneous matrix.
    """
    n = len(scale)
    matrix = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n):
        # A zero scale is not invertible; fall back to 1 rather than emitting
        # inf into a table a person is going to read.
        divisor = float(scale[i]) if float(scale[i]) != 0.0 else 1.0
        matrix[i, i] = 1.0 / divisor
        matrix[i, n] = -float(translation[i]) / divisor
    matrix[n, n] = 1.0
    return matrix


def axis_rows(
    names: Sequence[str],
    units: Sequence[str | None],
    types: Sequence[str],
) -> list[tuple[str, str]]:
    """Return one row per axis: ``("z", "space, um")``.

    The unit and type are folded into the value rather than given rows of
    their own, so an 5-axis image costs five rows instead of fifteen.
    """
    rows: list[tuple[str, str]] = []
    for name, unit, axis_type in zip(names, units, types, strict=True):
        parts = [part for part in (axis_type or None, unit) if part]
        rows.append((str(name), ", ".join(parts) if parts else "-"))
    return rows


def _number(value: float) -> str:
    """Render a float without a trailing ``.0`` when it is whole."""
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.4g}"


def _is_windows_drive_path(uri: str) -> bool:
    """Whether *uri* starts with a Windows drive letter (so is not a URI).

    ``urlparse`` reads the drive letter as a one-letter URL scheme.
    """
    return len(uri) >= 2 and uri[1] == ":" and uri[0].isalpha()


def source_label(uri: str) -> str:
    """Map a URI scheme to a human-readable source label."""
    if _is_windows_drive_path(str(uri)):
        return "local file"
    scheme = urlparse(str(uri)).scheme
    return {
        "": "local file",
        "file": "local file",
        "s3": "S3",
        "gs": "Google Cloud Storage",
        "gcs": "Google Cloud Storage",
        "https": "HTTP",
        "http": "HTTP",
    }.get(scheme, scheme)


def uri_file_name(uri: str) -> str:
    """Extract the final path component from a URI."""
    uri = str(uri)
    if _is_windows_drive_path(uri):
        path = uri.replace("\\", "/")
    else:
        path = urlparse(uri).path or uri
    return path.rstrip("/").rsplit("/", 1)[-1]


def ome_zarr_dataset_info(store: object, dtype: object) -> DatasetInfo:
    """Build the :class:`DatasetInfo` shared by the two OME-Zarr stores.

    The image and label stores carry the same metadata fields and differ
    only in what they call themselves and how they report their dtype, so
    the block is assembled once here and specialised by *dtype*.

    Parameters
    ----------
    store : object
        Duck-typed: any object with the OME-Zarr store fields
        (``zarr_path``, ``scale_names``, ``level_transforms``,
        ``data_coordinate_systems``, ``multiscale_index``, ``anonymous``,
        ``physical_scale``, ``physical_translation``) and the ``n_levels`` /
        ``level_shapes`` properties.
    dtype : object
        Rendered into the ``Data type`` row.  Passed in rather than read off
        the store so the label store can report both its on-disk dtype and
        the int32 bricks it actually returns.

    Returns
    -------
    DatasetInfo
    """
    identity = [
        *store._identity_rows(),
        ("Path", store.zarr_path),
        ("Source", source_label(store.zarr_path)),
        ("Data type", str(dtype)),
        ("Scale levels", str(store.n_levels)),
    ]
    # Both are noise at their defaults; a reader only needs to be told when
    # the store is not doing the ordinary thing.
    if store.multiscale_index:
        identity.append(("Multiscale index", str(store.multiscale_index)))
    if store.anonymous:
        identity.append(("Anonymous access", "yes"))

    sections: list[Section] = [RowSection(None, identity)]

    # The level-0 system is the store's only record of its axes.  A store
    # constructed without one has no axis rows to show, and its matrix is
    # labelled by axis position instead.
    systems = store.data_coordinate_systems
    axes = systems[0].axes if systems else ()
    if axes:
        sections.append(
            RowSection(
                "Axes",
                axis_rows(
                    [axis.name for axis in axes],
                    [axis.unit for axis in axes],
                    [axis.axis_type for axis in axes],
                ),
            )
        )

    if store.physical_scale:
        axis_labels = [axis.name for axis in axes] or [
            str(index) for index in range(len(store.physical_scale))
        ]
        headers = [*axis_labels, "1"]
        sections.append(
            MatrixSection(
                "World to data",
                world_to_data_matrix(store.physical_scale, store.physical_translation),
                row_labels=headers,
                col_labels=headers,
            )
        )

    level_rows: list[tuple[str, str]] = []
    for index, (level_name, shape) in enumerate(
        zip(store.scale_names, store.level_shapes)
    ):
        level_scale = np.asarray(store.level_scales[index], dtype=float)
        level_rows.append(
            (level_name, f"{format_shape(shape)}  ({format_scale(level_scale)})")
        )
    sections.append(RowSection("Scale levels", level_rows, collapsed=True))

    return DatasetInfo(sections=sections)


def array_extent_row(positions: np.ndarray) -> list[tuple[str, str]]:
    """Return the ``Extent`` row for a ``(n, ndim)`` position array.

    Empty for an empty array: ``min``/``max`` over zero points raises, and a
    store with no data has no extent to report.
    """
    if positions.size == 0:
        return []
    return [
        ("Extent", format_extent(positions.min(axis=0), positions.max(axis=0))),
    ]
