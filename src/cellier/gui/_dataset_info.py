"""Dataset metadata for display, shared by the Qt and anywidget front ends.

The section vocabulary a widget draws -- `DatasetInfo`,
`RowSection`, `MatrixSection` -- is defined in
``cellier.data._dataset_info`` and re-exported here: a data store has to be
able to describe itself without importing a GUI, so the types live beside the
stores and the widgets import them from this module.

What remains here is `dataset_info_from_path`, which builds the same
description for an OME-Zarr URI when there is no store object to ask.  A
store you already hold is better asked directly -- ``store.dataset_info()``
reads metadata it has parsed already, where this re-opens the group.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

from cellier.data._dataset_info import (
    DatasetInfo,
    MatrixSection,
    RowSection,
    Section,
    axis_rows,
    format_scale,
    format_shape,
    source_label,
    uri_file_name,
    world_to_data_matrix,
)

__all__ = [
    "DatasetInfo",
    "MatrixSection",
    "RowSection",
    "Section",
    "dataset_info_from_path",
]


def _read_level_shapes(zarr_path: str, scale_paths: list[str]) -> list[tuple[int, ...]]:
    """Read per-level array shapes without importing zarr.

    Reads ``.zarray`` (zarr v2) or ``zarr.json`` (zarr v3) metadata files
    directly to avoid zarr's internal asyncio loop, which conflicts with
    a running QtAsyncio event loop.

    Supports ``file://`` URIs (reads from disk) and remote URIs (reads via
    fsspec, which yaozarrs[io] already requires).
    """
    parsed = urlparse(zarr_path)
    shapes: list[tuple[int, ...]] = []

    if parsed.scheme == "file":
        import pathlib

        root = pathlib.Path(parsed.path)
        for rel in scale_paths:
            level = root / rel
            for sentinel in (".zarray", "zarr.json"):
                meta_file = level / sentinel
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                    shapes.append(tuple(int(d) for d in meta["shape"]))
                    break
            else:
                raise FileNotFoundError(
                    f"No zarr metadata (.zarray or zarr.json) found at '{level}'."
                )
    else:
        import fsspec

        fs, root = fsspec.url_to_fs(zarr_path)
        for rel in scale_paths:
            level = root.rstrip("/") + "/" + rel
            for sentinel in (".zarray", "zarr.json"):
                path = level + "/" + sentinel
                if fs.exists(path):
                    with fs.open(path) as f:
                        meta = json.load(f)
                    shapes.append(tuple(int(d) for d in meta["shape"]))
                    break
            else:
                raise FileNotFoundError(
                    f"No zarr metadata (.zarray or zarr.json) found at '{level}'."
                )

    return shapes


def dataset_info_from_path(
    zarr_path: str,
    *,
    multiscale_index: int = 0,
    series_index: int = 0,
) -> DatasetInfo:
    """Extract display metadata from an OME-Zarr store by URI.

    Uses ``yaozarrs`` to validate and read OME metadata, and direct JSON
    reads for the per-level array shapes.

    Prefer ``store.dataset_info()`` when you have a store: this opens the
    group, which for a remote URI is a network round trip.

    Parameters
    ----------
    zarr_path :
        Root URI of the OME-Zarr group (``file://``, ``s3://``, etc.).
    multiscale_index :
        Which ``multiscales[]`` entry to read. Defaults to 0.
    series_index :
        For Bf2Raw containers, which child image series to inspect.

    Returns
    -------
    DatasetInfo
        The same sectioned description a store's ``dataset_info`` returns.
    """
    import yaozarrs
    from yaozarrs import v05 as ome_v05
    from yaozarrs.v05 import ScaleTransformation, TranslationTransformation

    # ── Open and validate OME metadata ──────────────────────────────────────
    group = yaozarrs.open_group(zarr_path)
    metadata = group.ome_metadata()

    zarr_type = type(metadata).__name__

    # Resolve Bf2Raw containers to a child Image.
    if isinstance(metadata, ome_v05.Bf2Raw):
        ome_subgroup = group["OME"]
        ome_meta = ome_subgroup.ome_metadata()
        if isinstance(ome_meta, ome_v05.Series):
            child_path = ome_meta.series[series_index]
            zarr_path = zarr_path.rstrip("/") + "/" + child_path
            group = yaozarrs.open_group(zarr_path)
            metadata = group.ome_metadata()
            zarr_type = f"Bf2Raw/{type(metadata).__name__}"

    if not isinstance(metadata, ome_v05.Image):
        raise TypeError(f"Expected an OME-Zarr Image, got {type(metadata).__name__!r}.")

    ms = metadata.multiscales[multiscale_index]

    # ── Axis metadata ───────────────────────────────────────────────────────
    axis_names = [ax.name for ax in ms.axes]
    axis_units = [getattr(ax, "unit", None) for ax in ms.axes]
    axis_types = [ax.type or "" for ax in ms.axes]
    n = len(axis_names)

    # ── Global coordinate transforms ────────────────────────────────────────
    global_scale = [1.0] * n
    global_translation = [0.0] * n

    if ms.coordinateTransformations is not None:
        for ct in ms.coordinateTransformations:
            if isinstance(ct, ScaleTransformation):
                global_scale = list(ct.scale)
            elif isinstance(ct, TranslationTransformation):
                global_translation = list(ct.translation)

    # Physical scale per axis at level 0: global_scale * dataset0_scale.
    ds0_scale = list(ms.datasets[0].scale_transform.scale)
    phys_scale = [global_scale[i] * ds0_scale[i] for i in range(n)]

    # ── Per-level shapes via direct JSON reads ──────────────────────────────
    # Avoid zarr.open_group: zarr v3 tries to start an asyncio event loop,
    # which raises RuntimeError when QtAsyncio is already running.
    scale_paths = [ds.path for ds in ms.datasets]
    scale_shapes = _read_level_shapes(zarr_path, scale_paths)

    # ── Per-level scale, relative to level 0 ────────────────────────────────
    level_rows: list[tuple[str, str]] = []
    for path, shape, ds in zip(scale_paths, scale_shapes, ms.datasets):
        relative = [
            (global_scale[i] * ds.scale_transform.scale[i]) / phys_scale[i]
            if phys_scale[i]
            else 1.0
            for i in range(n)
        ]
        level_rows.append((path, f"{format_shape(shape)}  ({format_scale(relative)})"))

    headers = [*axis_names, "1"]
    sections: list[Section] = [
        RowSection(
            None,
            [
                ("File name", uri_file_name(zarr_path)),
                ("Type", zarr_type),
                ("Source", source_label(zarr_path)),
                ("Scale levels", str(len(scale_shapes))),
            ],
        ),
        RowSection("Axes", axis_rows(axis_names, axis_units, axis_types)),
        MatrixSection(
            "World to data",
            world_to_data_matrix(phys_scale, global_translation),
            row_labels=headers,
            col_labels=headers,
        ),
        RowSection("Scale levels", level_rows, collapsed=True),
    ]
    return DatasetInfo(sections=sections)
