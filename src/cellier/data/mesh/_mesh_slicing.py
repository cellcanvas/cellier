"""Slicing one level of a mesh: the synchronous body of a mesh store's read.

``plans/mesh_refactor_v3.md`` L5 and 5.4.  A mesh store's ``get_data`` runs
:func:`slice_mesh` in an executor thread, on one level's arrays and that
level's :class:`LevelCache`.  The function has no cancellation checkpoints:
the scheduler never cancels a read, and a thread cannot be interrupted.

The inclusion rule is the whole-face rule: a face survives when all three of
its vertices are inside the request's region.

What keeps the read cheap (5.4):

- **the identity path** (S3): when every face passes, the level's projected
  arrays are returned as they are, with no reindex;
- **cached normals** (S2): 3D vertex normals are computed once per level and
  column order, and gathered while no face straddles the slab;
- **the bounds index** (S4): an axis-aligned region asks a per-axis index
  for its candidate faces instead of testing every vertex.

Nothing here calls ``np.unique`` on a large array: numpy 2.4 can hold the
GIL in it, which stalls the event loop from an executor thread (D15).
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.data._bounds_index import AxisBoundsIndex
from cellier.data.mesh._mesh_requests import MeshData, MeshSectionData
from cellier.data.mesh._section import (
    SectionParts,
    clip_parts,
    closure_report,
    section_cut,
    section_slab,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from cellier.data.mesh._mesh_requests import MeshSliceRequest
    from cellier.transform import ConvexRegion

#: Faces (and vertices) per block when computing vertex normals.
_ACCUMULATE_BLOCK = 1_000_000

# Single degenerate triangle used when the slab contains no surviving faces.
PLACEHOLDER_INDICES = np.array([[0, 1, 2]], dtype=np.int32)


@dataclass(frozen=True, slots=True)
class MeshLevelArrays:
    """One level's arrays, as a read sees them.

    Taken on the event loop when the read starts, so a store whose fields
    are reassigned while the read runs is read consistently (the result is
    then dropped: the change invalidated it).

    Parameters
    ----------
    positions : np.ndarray
        ``(n_vertices, N)`` float32.
    indices : np.ndarray
        ``(n_faces, 3)`` int32.
    colors : np.ndarray or None
        ``(n_vertices, 4)`` or ``(n_faces, 4)`` float32.
    colors_layout : str or None
        ``"vertex"`` or ``"face"``; ``None`` without colours.
    """

    positions: np.ndarray
    indices: np.ndarray
    colors: np.ndarray | None = None
    colors_layout: str | None = None


class LevelCache:
    """What one level's reads compute once and share.

    Lives on the store (the ortho viewer's panels read one store) and is
    dropped when the store changes.  Entries are immutable once built.  The
    first read that needs an entry builds it, inside that read; a second
    read that needs the same entry at the same time waits for the first
    instead of building it again.  The lock is per entry, so different
    entries (the bounds indexes of different axes) build side by side, and
    a finished entry is read without a lock.

    Entries, by key:

    - ``("index", axis)``: the :class:`AxisBoundsIndex` of a data axis;
    - ``("projected", output_axes)``: the level's upload-ready positions and
      their bounds;
    - ``("normals", output_axes)``: the level's vertex normals.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[Hashable, threading.Lock] = {}
        self._entries: dict[Hashable, Any] = {}
        #: How many times each entry was built (tests).
        self.builds: dict[Hashable, int] = {}

    def get(self, key: Hashable, build: Callable[[], Any]) -> Any:
        """Return the entry for *key*, building it with *build* if missing."""
        entry = self._entries.get(key)
        if entry is not None:
            return entry
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = build()
                self._entries[key] = entry
                self.builds[key] = self.builds.get(key, 0) + 1
        return entry

    def peek(self, key: Hashable) -> Any:
        """The entry for *key* if it is built, else ``None``."""
        return self._entries.get(key)

    def __deepcopy__(self, memo: dict) -> LevelCache:
        """A copied store starts with an empty cache (locks do not copy)."""
        return LevelCache()


def compute_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area-weighted per-vertex normals of a triangle mesh.

    Parameters
    ----------
    positions : np.ndarray
        (n_vertices, 3) float32 vertex positions.
    indices : np.ndarray
        (n_faces, 3) int32 triangle face indices.

    Returns
    -------
    np.ndarray
        (n_vertices, 3) float32 unit normals, contiguous.  Degenerate
        vertices (zero accumulated normal) get [0, 0, 1].
    """
    n_vertices = positions.shape[0]
    normals = np.zeros((n_vertices, 3), dtype=np.float32)
    # In blocks of faces, for two reasons.  ``np.bincount`` holds the GIL
    # for part of a call (about 30 ms at 48M entries), and this runs in an
    # executor thread beside the event loop: a block keeps each hold to a
    # millisecond or two.  And the edge vectors and face normals of a whole
    # level are several times its size; a block's are a few tens of MB.
    for start in range(0, indices.shape[0], _ACCUMULATE_BLOCK):
        faces = indices[start : start + _ACCUMULATE_BLOCK]
        origin = positions[faces[:, 0]]
        # Area-weighted face normals, (n_block, 3).
        face_normals = np.cross(
            positions[faces[:, 1]] - origin, positions[faces[:, 2]] - origin
        )
        for corner in range(3):
            vertices = faces[:, corner]
            first = int(vertices.min())
            local = vertices - first
            for component in range(3):
                summed = np.bincount(local, weights=face_normals[:, component])
                normals[first : first + len(summed), component] += summed
    for start in range(0, n_vertices, _ACCUMULATE_BLOCK):
        block = normals[start : start + _ACCUMULATE_BLOCK]
        norms = np.linalg.norm(block, axis=1)
        degenerate = norms == 0.0
        norms[degenerate] = 1.0
        block /= norms[:, None]
        block[degenerate] = (0.0, 0.0, 1.0)
    return normals


def _project(
    positions: np.ndarray, rows: np.ndarray | None, output_axes: tuple[int, ...]
) -> np.ndarray:
    """``(n, 3)`` contiguous float32: the *output_axes* columns of *rows*.

    A 2D result (two output axes) gets zeros in the third column.
    """
    n = positions.shape[0] if rows is None else len(rows)
    out = np.zeros((n, 3), dtype=np.float32)
    for column, axis in enumerate(output_axes):
        source = positions[:, axis]
        out[:, column] = source if rows is None else source[rows]
    return out


def _bounds(positions: np.ndarray) -> np.ndarray:
    """``(2, 3)`` float64 minimum and maximum of ``(n, 3)`` positions."""
    return np.stack([positions.min(axis=0), positions.max(axis=0)]).astype(np.float64)


def _empty(request: MeshSliceRequest) -> MeshData:
    return MeshData(
        request_id=request.slice_request_id,
        positions=np.zeros((3, 3), dtype=np.float32),
        indices=PLACEHOLDER_INDICES,
        normals=None,
        colors=None,
        color_mode="vertex",
        is_empty=True,
        level=int(request.scale_index),
    )


def _axis_constraints(
    region: ConvexRegion,
) -> tuple[dict[int, list[tuple[float, float]]], bool] | None:
    """The region's constraints, grouped by the one axis each is along.

    Returns
    -------
    constraints : dict[int, list[tuple[float, float]]]
        ``axis -> [(normal component, offset), ...]``.
    infeasible : bool
        A constraint with a zero normal excludes everything.

    ``None`` when a constraint is oblique (more than one non-zero normal
    component): the caller then tests every vertex.
    """
    constraints: dict[int, list[tuple[float, float]]] = {}
    infeasible = False
    for half_space in region.half_spaces:
        along = np.flatnonzero(half_space.normal)
        if len(along) == 0:
            # A transform with no extent along the axis: vacuous, or empty.
            infeasible = infeasible or half_space.offset < 0
            continue
        if len(along) > 1:
            return None
        axis = int(along[0])
        constraints.setdefault(axis, []).append(
            (float(half_space.normal[axis]), float(half_space.offset))
        )
    return constraints, infeasible


def _axis_interval(terms: list[tuple[float, float]]) -> tuple[float, float]:
    """``[lo, hi]`` of one axis's constraints, one float32 step wide of them.

    The interval only narrows the candidates; the exact test is the
    half-space arithmetic of ``ConvexRegion.contains``.  One step out makes
    sure rounding here never drops a face that test would keep.
    """
    lo, hi = -np.inf, np.inf
    for component, offset in terms:
        bound = offset / component
        if component > 0:
            hi = min(hi, bound)
        else:
            lo = max(lo, bound)
    if lo > -np.inf:
        lo = float(np.nextafter(np.float32(lo), np.float32(-np.inf)))
        lo = float(np.nextafter(np.float32(lo), np.float32(-np.inf)))
    if hi < np.inf:
        hi = float(np.nextafter(np.float32(hi), np.float32(np.inf)))
        hi = float(np.nextafter(np.float32(hi), np.float32(np.inf)))
    return lo, hi


def face_bounds_index(
    arrays: MeshLevelArrays, cache: LevelCache, axis: int
) -> AxisBoundsIndex:
    """The level's face-bounds index along data *axis*, built on first use."""

    def build() -> AxisBoundsIndex:
        column = np.ascontiguousarray(arrays.positions[:, axis])
        indices = arrays.indices
        first, second, third = (column[indices[:, corner]] for corner in range(3))
        return AxisBoundsIndex.build(
            np.minimum(np.minimum(first, second), third),
            np.maximum(np.maximum(first, second), third),
            lambda ids: column[indices[ids]].max(axis=1),
        )

    return cache.get(("index", axis), build)


def _select_faces(
    arrays: MeshLevelArrays, region: ConvexRegion, cache: LevelCache | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The faces wholly inside *region*.

    Returns
    -------
    faces : np.ndarray or None
        Surviving face ids, ascending.  ``None`` when every face survives.
    near : np.ndarray or None
        Faces that do not survive but may share a vertex with one that does
        (a face reaching into the slab).  ``None`` when that is not known,
        in which case the caller must assume some do.
    """
    positions, indices = arrays.positions, arrays.indices
    n_faces = indices.shape[0]
    parsed = _axis_constraints(region)
    if parsed is not None:
        constraints, infeasible = parsed
        if infeasible:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if not constraints:
            return None, np.empty(0, dtype=np.int64)
    if parsed is None or cache is None:
        # Oblique, or no cache to hold an index: test every vertex.
        face_mask = region.contains(positions)[indices].all(axis=1)
        if face_mask.all():
            return None, np.empty(0, dtype=np.int64)
        return np.flatnonzero(face_mask), None

    intervals = {axis: _axis_interval(terms) for axis, terms in constraints.items()}
    indexes = {axis: face_bounds_index(arrays, cache, axis) for axis in constraints}
    # The most selective axis narrows; every axis is then tested exactly.
    chosen = min(
        constraints, key=lambda axis: indexes[axis].count_min_between(*intervals[axis])
    )
    index = indexes[chosen]
    lo, hi = intervals[chosen]
    candidates = index.min_between(lo, hi)
    corners = indices[candidates]  # (n_candidates, 3)
    keep = np.ones(len(candidates), dtype=bool)
    for axis, terms in constraints.items():
        values = positions[:, axis][corners].astype(np.float64)
        for component, offset in terms:
            # The arithmetic of ``ConvexRegion.contains``, on one column.
            keep &= (values * component <= offset).all(axis=1)
    if len(candidates) == n_faces and keep.all():
        return None, np.empty(0, dtype=np.int64)
    faces = np.sort(candidates[keep])

    # Faces that reach into the slab along the chosen axis without being
    # inside it.  When no short face has an extent and there is no long
    # face (a time axis), they are all among the candidates.
    if index.longest == 0.0 and not len(index.long_ids):
        near = candidates[~keep]
    else:
        overlapping = index.overlapping(lo, hi)
        survives = np.zeros(n_faces, dtype=bool)
        survives[faces] = True
        near = overlapping[~survives[overlapping]]
    return faces, near


def slice_mesh(
    arrays: MeshLevelArrays, request: MeshSliceRequest, cache: LevelCache | None = None
) -> MeshData | MeshSectionData:
    """Slice one level of a mesh for *request*; upload-ready.

    A request with a ``section`` is cut (:func:`slice_section`); the rest of
    this describes the whole-face read.

    Synchronous, and safe to run in an executor thread next to other reads
    of the same level.

    Parameters
    ----------
    arrays : MeshLevelArrays
        The level's arrays.
    request : MeshSliceRequest
        The region, in the level's data coordinates, and the columns to
        emit.
    cache : LevelCache or None
        The level's cache.  ``None`` slices with a full pass and caches
        nothing; the result is the same.

    Returns
    -------
    MeshData
        ``is_empty=True`` when no face is inside the region.
    """
    if request.section is not None:
        return slice_section(arrays, request, cache)
    positions, indices = arrays.positions, arrays.indices
    output_axes = tuple(int(axis) for axis in request.output_axes)
    is_3d = len(output_axes) == 3
    level = int(request.scale_index)
    color_mode = arrays.colors_layout or "vertex"

    faces, near = _select_faces(arrays, request.region, cache)

    def whole_level_positions() -> tuple[np.ndarray, np.ndarray]:
        projected = _project(positions, None, output_axes)
        return projected, _bounds(projected)

    def whole_level_normals() -> np.ndarray:
        # The projected level is kept only by the identity path, which
        # uploads it; a sliced read uses it here and lets it go.
        held = None if cache is None else cache.peek(("projected", output_axes))
        whole = whole_level_positions()[0] if held is None else held[0]
        return compute_vertex_normals(whole, indices)

    if faces is None:
        # S3: every face passes, so the output is the level itself.
        if indices.shape[0] == 0:
            return _empty(request)
        if cache is None:
            projected, bounds = whole_level_positions()
            normals = whole_level_normals() if is_3d else None
        else:
            projected, bounds = cache.get(
                ("projected", output_axes), whole_level_positions
            )
            normals = (
                cache.get(("normals", output_axes), whole_level_normals)
                if is_3d
                else None
            )
        return MeshData(
            request_id=request.slice_request_id,
            positions=projected,
            indices=indices,
            normals=normals,
            colors=arrays.colors,
            color_mode=color_mode,
            is_empty=False,
            original_face_indices=None,
            level=level,
            bounds=bounds,
        )

    if len(faces) == 0:
        return _empty(request)

    # Reindex with a vertex mask (S1), not ``np.unique``.
    surviving = indices[faces]
    used = np.zeros(positions.shape[0], dtype=bool)
    used[surviving.ravel()] = True
    kept_vertices = np.flatnonzero(used)
    remap = np.full(positions.shape[0], -1, dtype=np.int32)
    remap[kept_vertices] = np.arange(len(kept_vertices), dtype=np.int32)
    new_indices = np.ascontiguousarray(remap[surviving])
    projected = _project(positions, kept_vertices, output_axes)

    normals = None
    if is_3d:
        # S2: the level's normals hold for a vertex whose faces all
        # survived.  A dropped face that shares a kept vertex changes it.
        straddles = near is None or (len(near) > 0 and bool(used[indices[near]].any()))
        if cache is not None and not straddles:
            normals = np.ascontiguousarray(
                cache.get(("normals", output_axes), whole_level_normals)[kept_vertices]
            )
        else:
            normals = compute_vertex_normals(projected, new_indices)

    colors = None
    if arrays.colors is not None:
        rows = faces if arrays.colors_layout == "face" else kept_vertices
        colors = np.ascontiguousarray(arrays.colors[rows])

    return MeshData(
        request_id=request.slice_request_id,
        positions=projected,
        indices=new_indices,
        normals=normals,
        colors=colors,
        color_mode=color_mode,
        is_empty=False,
        original_face_indices=faces,
        level=level,
        bounds=_bounds(projected),
    )


#: In-plane tolerance, as a fraction of the level's bounding-box diagonal
#: (decision D2).
SECTION_TOLERANCE = 1e-7


def _section_columns(request: MeshSliceRequest) -> tuple[tuple[int, int, int], int]:
    """The three data axes the kernel works in: the outputs, then the cut's.

    Returns ``(columns, section_axis)``.
    """
    output_axes = tuple(int(axis) for axis in request.output_axes)
    if len(output_axes) != 2:
        raise ValueError(
            f"A mesh section needs two output axes, got {output_axes}: only a "
            "2D view is cut."
        )
    normal = np.asarray(request.section.normal, dtype=np.float64)
    outside = [int(a) for a in np.flatnonzero(normal) if int(a) not in output_axes]
    if len(outside) != 1:
        raise ValueError(
            "A mesh section plane must cross exactly one data axis that is "
            f"not displayed; its normal {tuple(normal)} crosses {outside} "
            f"with {output_axes} displayed."
        )
    return (output_axes[0], output_axes[1], outside[0]), outside[0]


def _restrict(candidates: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    """The *candidates* that are among the sorted *faces* (``None``: all)."""
    if faces is None:
        return candidates
    if len(faces) == 0:
        return candidates[:0]
    position = np.minimum(np.searchsorted(faces, candidates), len(faces) - 1)
    return candidates[faces[position] == candidates]


def _flatten(points: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Upload-ready ``(n, 3)`` float32 with a zero third column; its bounds."""
    out = np.zeros((len(points), 3), dtype=np.float32)
    out[:, :2] = points[:, :2]
    return out, (_bounds(out) if len(out) else None)


def level_closure(arrays: MeshLevelArrays, cache: LevelCache):
    """The level's closure report, computed on first use (X5)."""
    return cache.get(
        ("closure",),
        lambda: closure_report(arrays.indices, arrays.positions.shape[0]),
    )


def closure_text(report) -> str:
    """One line for ``dataset_info``: is the surface closed (D14)."""
    if report is None:
        return "not computed"
    if report.closed:
        return "yes"
    return (
        f"no ({report.boundary_edges} boundary, "
        f"{report.nonmanifold_edges} non-manifold edges)"
    )


def slice_section(
    arrays: MeshLevelArrays, request: MeshSliceRequest, cache: LevelCache | None = None
) -> MeshSectionData:
    """Cut one level of a mesh with the request's plane or slab (5.5).

    The request's ``region`` is the filter on the axes the cut does not
    replace (the whole-face rule, as in :func:`slice_mesh`); the faces that
    pass it are then cut.  With an axis-aligned plane the candidates come
    from the level's bounds index (S4), not from a pass over every face.

    Parameters
    ----------
    arrays : MeshLevelArrays
        The level's arrays.
    request : MeshSliceRequest
        With ``section`` set.
    cache : LevelCache or None
        The level's cache; ``None`` cuts with a full pass.

    Returns
    -------
    MeshSectionData
    """
    section = request.section
    columns, section_axis = _section_columns(request)
    positions, indices = arrays.positions, arrays.indices

    def kernel_space() -> tuple[np.ndarray, float]:
        points = np.ascontiguousarray(positions[:, list(columns)], dtype=np.float32)
        if len(points) == 0:
            return points, 0.0
        diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        return points, SECTION_TOLERANCE * diagonal

    if cache is None:
        points, tol = kernel_space()
    else:
        points, tol = cache.get(("section", columns), kernel_space)
        # The closure of the level, for ``dataset_info`` (X5).
        level_closure(arrays, cache)

    # A unit normal pointing up the section axis, so that the float32
    # distance of an axis-aligned plane is one subtraction.
    normal = np.asarray(section.normal, dtype=np.float64)[list(columns)]
    offsets = np.asarray(section.offsets, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    normal, offsets = normal / length, offsets / length
    if normal[2] < 0:
        normal, offsets = -normal, -offsets[::-1]
    slab = section.mode == "slab" and len(offsets) == 2
    low, high = (offsets[0], offsets[-1]) if slab else (offsets.mean(),) * 2

    faces, _near = _select_faces(arrays, request.region, cache)
    axis_aligned = normal[0] == 0.0 and normal[1] == 0.0
    if cache is not None and axis_aligned:
        index = face_bounds_index(arrays, cache, section_axis)
        # One float32 step wider than the tolerance: a superset is safe.
        reach = tol + abs(float(np.spacing(np.float32(max(abs(low), abs(high))))))
        candidates = np.sort(index.overlapping(low - reach, high + reach))
        candidates = _restrict(candidates, faces)
    else:
        candidates = faces

    vertex_colors = face_colors = None
    if arrays.colors is not None:
        if arrays.colors_layout == "face":
            face_colors = arrays.colors
        else:
            vertex_colors = arrays.colors
    options = {
        "candidates": candidates,
        "vertex_colors": vertex_colors,
        "face_colors": face_colors,
        "outline": bool(section.outline),
        "fill": bool(section.fill),
    }
    if candidates is not None and len(candidates) == 0:
        parts: SectionParts = section_cut(
            points[:0], indices[:0], normal, low, tol, outline=False, fill=False
        )
    elif slab:
        parts = section_slab(points, indices, normal, low, high, tol, **options)
        if request.clip_planes:
            # Clipping planes, on the kernel's three columns.  After the
            # slab, not in its face filter: the cut edge must be exact.
            parts = clip_parts(
                parts,
                [
                    (tuple(plane_normal[axis] for axis in columns), offset)
                    for plane_normal, offset in request.clip_planes
                ],
            )
    else:
        parts = section_cut(points, indices, normal, low, tol, **options)

    fill_positions, fill_bounds = _flatten(parts.fill_positions)
    outline_positions, outline_bounds = _flatten(parts.outline_positions)

    def colors_of(values):
        if arrays.colors is None or values is None:
            return None
        return np.ascontiguousarray(values, dtype=np.float32)

    return MeshSectionData(
        request_id=request.slice_request_id,
        level=int(request.scale_index),
        fill_positions=fill_positions,
        fill_indices=np.ascontiguousarray(parts.fill_indices, dtype=np.int32),
        fill_colors=colors_of(parts.fill_colors),
        fill_face_ids=parts.fill_face_ids,
        outline_positions=outline_positions,
        outline_colors=colors_of(parts.outline_colors),
        outline_face_ids=parts.outline_face_ids,
        color_mode=arrays.colors_layout or "vertex",
        is_empty=parts.is_empty,
        fill_bounds=fill_bounds,
        outline_bounds=outline_bounds,
        n_closed_loops=parts.n_closed_loops,
        n_open_segments=parts.n_open_segments,
    )


async def run_slice(
    arrays: MeshLevelArrays, request: MeshSliceRequest, cache: LevelCache | None
) -> MeshData | MeshSectionData:
    """Run :func:`slice_mesh` off the event loop, in the loop's executor.

    How many of these run at once is the scheduler's business
    (``SchedulerConfig.compute_budget``), not the executor's.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, slice_mesh, arrays, request, cache)
