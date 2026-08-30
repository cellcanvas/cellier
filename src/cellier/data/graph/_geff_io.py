# src/cellier/data/graph/_geff_io.py
"""geff -> ``GraphMemoryStore`` adapter.

``geff`` is an optional dependency (D8) and is imported inside
:func:`read_geff` only, so importing this module costs nothing and the
whole ``mask`` slice path stays dependency-free.

Reading goes through ``geff.GeffReader`` rather than
``geff.read(backend="spatial-graph")``.  The high-level call builds two
R-trees keyed by *original node id*, which D18 requires this code to throw
away and rebuild over row indices; the low-level reader skips that
entirely and measured 5.2x faster at 100k nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.transform import AffineTransform

if TYPE_CHECKING:
    import pathlib


@dataclass(frozen=True)
class GeffPayload:
    """Everything ``from_geff`` needs, already in cellier's own conventions.

    Parameters
    ----------
    positions : np.ndarray
        (n_nodes, ndim) float32, axes stacked in ``metadata.axes`` file
        order (or in ``axis_names`` order when that is given).
    edges : np.ndarray
        (n_edges, 2) int32 endpoint **row indices**.  geff stores original
        id pairs; the one ``searchsorted`` that converts them happens at
        load time and never on the per-frame path (D18).
    node_ids : np.ndarray
        Original node ids, kept for pick payloads.
    transform : AffineTransform
        Built from the axes' ``scale`` / ``offset`` (D23).
    directed : bool
        From ``metadata.directed``.
    axes : list
        The geff ``Axis`` objects, in file order, retained as-is.
    node_props, edge_props : dict[str, np.ndarray]
        Every non-position property, retained but unconsumed.  These are
        what colour-by-property will read.
    node_colors, node_sizes : np.ndarray | None
        Resolved from ``node_color_prop`` / ``node_size_prop`` when asked.
    """

    positions: np.ndarray
    edges: np.ndarray
    node_ids: np.ndarray
    transform: AffineTransform
    directed: bool
    axes: list = field(default_factory=list)
    node_props: dict = field(default_factory=dict)
    edge_props: dict = field(default_factory=dict)
    node_colors: np.ndarray | None = None
    node_sizes: np.ndarray | None = None


def _import_geff() -> Any:
    """Import ``geff``, raising a message that names the extra."""
    try:
        import geff
    except ImportError as e:
        raise ImportError(
            "Reading geff files requires the 'graph' extra: "
            "pip install 'cellier[graph]'"
        ) from e
    return geff


def read_geff(
    path: str | pathlib.Path,
    *,
    axis_names: list[str] | None = None,
    node_color_prop: str | None = None,
    node_size_prop: str | None = None,
) -> GeffPayload:
    """Read a geff file into flat arrays plus its axis transform.

    Every axis in ``metadata.axes`` is loaded, in file order, **including
    ``t``**: time is an ordinary sliceable dimension here and the existing
    dims sliders work on it unchanged.  ``TrailConfig`` is what turns an
    axis into a history/forecast window, and it is not time-specific.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the geff store.
    axis_names : list[str] | None
        Overrides which axes are loaded and in what order.  ``None`` uses
        ``metadata.axes`` order.
    node_color_prop : str | None
        Name of a node property to use as ``node_colors``.  Must resolve to
        an (n, 4) RGBA array.  A convenience shortcut, not the deferred
        colormap machinery.
    node_size_prop : str | None
        Name of a node property to use as ``node_sizes``.

    Returns
    -------
    GeffPayload

    Raises
    ------
    ImportError
        When ``geff`` is not installed.
    KeyError
        When a requested axis or property is not in the file.
    """
    geff = _import_geff()

    reader = geff.GeffReader(path)
    reader.read_node_props()
    reader.read_edge_props()
    payload = reader.build()

    metadata = payload["metadata"]
    node_ids = np.ascontiguousarray(payload["node_ids"])
    node_props_raw = payload["node_props"]
    edge_props_raw = payload["edge_props"]

    axes = list(metadata.axes)
    if axis_names is not None:
        by_name = {axis.name: axis for axis in axes}
        missing = [name for name in axis_names if name not in by_name]
        if missing:
            raise KeyError(
                f"axis_names {missing} are not in the file; it carries "
                f"{[axis.name for axis in axes]}"
            )
        axes = [by_name[name] for name in axis_names]

    # Positions are assembled by stacking the per-axis "values" arrays.
    # Each prop entry is a {"values", "missing"} dict, not a bare array.
    columns = []
    for axis in axes:
        if axis.name not in node_props_raw:
            raise KeyError(
                f"axis '{axis.name}' has no matching node property in the file"
            )
        columns.append(np.asarray(node_props_raw[axis.name]["values"]))
    positions = np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)

    # geff edge_ids are original id PAIRS. One searchsorted at load time
    # converts them to row indices; the per-frame path never pays it (D18).
    edge_ids = np.asarray(payload["edge_ids"])
    if edge_ids.size:
        order = np.argsort(node_ids)
        sorted_ids = node_ids[order]
        located = np.searchsorted(sorted_ids, edge_ids.reshape(-1))
        edges = order[located].reshape(edge_ids.shape).astype(np.int32)
    else:
        edges = np.zeros((0, 2), dtype=np.int32)

    # D23: the file's axes are the sole source of the transform. Ignoring a
    # stated scale renders an anisotropic light-sheet file at the wrong
    # scale with no signal, which is why there is no override to forget
    # to pass.
    scales = tuple(1.0 if a.scale is None else float(a.scale) for a in axes)
    offsets = tuple(0.0 if a.offset is None else float(a.offset) for a in axes)
    transform = AffineTransform.from_scale_and_translation(scales, offsets)

    axis_name_set = {axis.name for axis in axes}
    node_props = {
        name: np.asarray(entry["values"])
        for name, entry in node_props_raw.items()
        if name not in axis_name_set
    }
    edge_props = {
        name: np.asarray(entry["values"]) for name, entry in edge_props_raw.items()
    }

    node_colors = None
    if node_color_prop is not None:
        node_colors = np.asarray(
            _require_prop(node_props_raw, node_color_prop, "node")["values"],
            dtype=np.float32,
        )
    node_sizes = None
    if node_size_prop is not None:
        node_sizes = np.asarray(
            _require_prop(node_props_raw, node_size_prop, "node")["values"],
            dtype=np.float32,
        )

    return GeffPayload(
        positions=positions,
        edges=edges,
        node_ids=node_ids,
        transform=transform,
        directed=bool(metadata.directed),
        axes=axes,
        node_props=node_props,
        edge_props=edge_props,
        node_colors=node_colors,
        node_sizes=node_sizes,
    )


def _require_prop(props: dict, name: str, kind: str) -> dict:
    """Fetch a named property, with a message listing what is available."""
    if name not in props:
        raise KeyError(
            f"{kind} property '{name}' is not in the file; it carries {sorted(props)}"
        )
    return props[name]
