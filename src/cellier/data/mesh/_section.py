"""The mesh section kernel: where a plane, or a slab, cuts a triangle mesh.

``plans/mesh_refactor_v3.md`` 5.5 (X1-X5).  Pure numpy and scipy, with no
store or render imports; it does not care which level it is given.

Everything here works on **three-column positions** and one plane
``normal . p = offset`` (cut mode) or two (slab mode).  The caller picks the
three columns and flattens the result.

Conventions:

- A vertex within ``tol`` of the plane is **on** it.  For the crossing test
  an on-plane vertex counts as positive, so every edge's crossing is decided
  by its two vertices alone, faces sharing an edge agree, and loops close
  through vertices.
- A face with all three vertices on the plane is **in-plane**: it is drawn
  as itself in the fill, and its boundary edges go in the outline.
- Segments are joined into loops by the **exact key** of the mesh edge each
  endpoint lies on: no float welding, and no reliance on face winding.
- The fill follows the faces' winding where it is consistent: each loop is
  the boundary of a solid or of a cavity, by which way its faces point, and
  a region is filled where the count of solids minus cavities around it is
  not zero.  So one object inside another is filled, and a cavity inside an
  object is a hole.  Loops that nest with a loop whose faces disagree fall
  back to even-odd: a loop inside another is a hole, a loop inside a hole
  is filled again.  Open and non-manifold components are outline only.

Nothing here calls ``np.unique`` without ``return_inverse`` on a large
array, nor ``np.bincount`` with weights: both can hold the GIL (D15).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, depth_first_order

# Lone vertex per packed side code (bit i = vertex i on the positive side).
# Codes 0 and 7 do not cross.
_LONE = np.array([-1, 0, 1, 2, 2, 1, 0, -1], dtype=np.int64)

_NO_FACES = np.zeros(0, dtype=np.int64)


def plane_distance(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Signed distance ``normal . p - offset``, float32, in a fixed order.

    An explicit per-component product rather than ``p @ n``: BLAS may round
    a matrix-vector product differently for different array shapes, and a
    vertex must classify the same whether it is reached through every face
    or through a candidate subset, or loops would not close.
    """
    n = np.asarray(normal, dtype=np.float32)
    along = np.flatnonzero(n)
    if len(along) == 1 and n[along[0]] == 1.0:
        return points[..., along[0]] - np.float32(offset)
    return (
        points[..., 0] * n[0] + points[..., 1] * n[1] + points[..., 2] * n[2]
    ) - np.float32(offset)


@dataclass(frozen=True)
class Cut:
    """The segments where one plane crosses a mesh, and its in-plane faces.

    Attributes
    ----------
    p0, p1 : np.ndarray
        ``(S, 3)`` float64 segment endpoints.
    k0, k1 : np.ndarray
        ``(S,)`` int64 key of the mesh edge each endpoint lies on.
    face : np.ndarray
        ``(S,)`` source face of each segment.
    apex_above : np.ndarray
        ``(S,)`` bool: the face's lone vertex is on the positive side.  With
        the plane's normal towards the viewer and the face's winding
        counter-clockwise from outside, the outside of the surface is on
        the right of ``p0 -> p1`` when this is true, on the left otherwise.
    apex, v0, v1 : np.ndarray
        ``(S,)`` vertex ids: endpoint ``i`` lies on the edge ``apex - v_i``.
    t0, t1 : np.ndarray
        ``(S,)`` float64 position of each endpoint along its edge, 0 at the
        apex; a vertex attribute is ``a[apex] + t * (a[v] - a[apex])``.
    inplane_faces : np.ndarray
        ``(F,)`` faces lying in the plane.
    inplane_edges : np.ndarray
        ``(E, 2)`` vertex pairs: edges used by exactly one in-plane face.
    inplane_edge_faces : np.ndarray
        ``(E,)`` the in-plane face each boundary edge belongs to.
    """

    p0: np.ndarray
    p1: np.ndarray
    k0: np.ndarray
    k1: np.ndarray
    face: np.ndarray
    apex_above: np.ndarray
    apex: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    inplane_faces: np.ndarray
    inplane_edges: np.ndarray
    inplane_edge_faces: np.ndarray


def _edge_runs(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs of equal values: ``(order, run starts, run lengths)``.

    A sort and a comparison with the neighbour, not ``np.unique``.
    """
    order = np.argsort(packed, kind="stable")
    ordered = packed[order]
    if len(ordered) == 0:
        return order, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]]))
    counts = np.diff(np.concatenate([starts, [len(ordered)]]))
    return order, starts, counts


def cut_plane(
    positions: np.ndarray,
    indices: np.ndarray,
    normal: np.ndarray,
    offset: float,
    tol: float,
    candidates: np.ndarray | None = None,
) -> Cut:
    """Cut a mesh with the plane ``normal . p = offset`` (X2 steps 2-4).

    Parameters
    ----------
    positions : np.ndarray
        ``(V, 3)`` float32.
    indices : np.ndarray
        ``(F, 3)`` integer faces.
    normal : np.ndarray
        ``(3,)`` plane normal; any magnitude.
    offset : float
        Plane offset, in the units of ``normal . p``.
    tol : float
        A vertex with ``|normal . p - offset| <= tol`` is on the plane.
    candidates : np.ndarray or None
        Face ids to look at; ``None`` is every face.  The cut is the same
        for any superset of the faces the plane touches.

    Returns
    -------
    Cut
    """
    tol32 = np.float32(tol)
    if candidates is None:
        faces = indices
        face_ids = None
        distance = plane_distance(positions, normal, offset)  # per vertex
        side = (distance >= -tol32)[faces]
        any_on = bool((np.abs(distance) <= tol32).any())
    else:
        face_ids = np.asarray(candidates)
        faces = indices[face_ids]
        face_distance = plane_distance(positions[faces], normal, offset)  # (C, 3)
        side = face_distance >= -tol32
        any_on = bool((np.abs(face_distance) <= tol32).any())
    code = (
        side[:, 0].view(np.uint8)
        | (side[:, 1].view(np.uint8) << 1)
        | (side[:, 2].view(np.uint8) << 2)
    )

    cross = np.flatnonzero((code != 0) & (code != 7))
    crossing = faces[cross].astype(np.int64)
    lone = _LONE[code[cross]]
    rows = np.arange(len(crossing))
    apex = crossing[rows, lone]
    v0 = crossing[rows, (lone + 1) % 3]
    v1 = crossing[rows, (lone + 2) % 3]

    # Distances of the crossing vertices only, in float64 for the
    # interpolation.  On-plane is still decided by the float32 rule, so it
    # matches the side code, and counts as exactly 0; ``t`` is clamped in
    # case float64 disagrees with float32 on a sign.
    normal64 = np.asarray(normal, dtype=np.float64)

    def exact_distance(vertices: np.ndarray) -> np.ndarray:
        points = positions[vertices]
        on = np.abs(plane_distance(points, normal, offset)) <= tol32
        return np.where(on, 0.0, points.astype(np.float64) @ normal64 - offset)

    d_apex, d0, d1 = exact_distance(apex), exact_distance(v0), exact_distance(v1)
    p_apex = positions[apex].astype(np.float64)

    def along(d_other: np.ndarray, other: np.ndarray):
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.clip(np.nan_to_num(d_apex / (d_apex - d_other)), 0.0, 1.0)
        return t, p_apex + t[:, None] * (positions[other].astype(np.float64) - p_apex)

    n_vertices = np.int64(len(positions))
    t0, p0 = along(d0, v0)
    t1, p1 = along(d1, v1)
    k0 = np.minimum(apex, v0) * n_vertices + np.maximum(apex, v0)
    k1 = np.minimum(apex, v1) * n_vertices + np.maximum(apex, v1)
    segment_face = cross if face_ids is None else face_ids[cross]

    inplane_faces = _NO_FACES
    boundary = np.zeros((0, 2), dtype=np.int64)
    boundary_faces = _NO_FACES
    if any_on:
        # Only a face whose first vertex is on the plane can be in-plane:
        # one column's gather, then the few survivors are tested in full.
        if candidates is None:
            on_vertex = np.abs(distance) <= tol32
            first = np.flatnonzero(on_vertex[faces[:, 0]])
            rows_in = first[on_vertex[faces[first, 1]] & on_vertex[faces[first, 2]]]
        else:
            rows_in = np.flatnonzero((np.abs(face_distance) <= tol32).all(axis=1))
        if len(rows_in):
            flat = faces[rows_in].astype(np.int64)
            # Degenerate faces (a repeated vertex) draw nothing.
            proper = (
                (flat[:, 0] != flat[:, 1])
                & (flat[:, 1] != flat[:, 2])
                & (flat[:, 2] != flat[:, 0])
            )
            rows_in, flat = rows_in[proper], flat[proper]
        if len(rows_in):
            edges = np.concatenate([flat[:, [0, 1]], flat[:, [1, 2]], flat[:, [2, 0]]])
            low = np.minimum(edges[:, 0], edges[:, 1])
            high = np.maximum(edges[:, 0], edges[:, 1])
            order, starts, counts = _edge_runs(low * n_vertices + high)
            once = order[starts[counts == 1]]
            boundary = np.stack([low[once], high[once]], axis=1)
            inplane_faces = rows_in if face_ids is None else face_ids[rows_in]
            boundary_faces = inplane_faces[once % len(rows_in)]
    return Cut(
        p0=p0,
        p1=p1,
        k0=k0,
        k1=k1,
        face=segment_face,
        apex_above=side[cross, lone],
        apex=apex,
        v0=v0,
        v1=v1,
        t0=t0,
        t1=t1,
        inplane_faces=inplane_faces,
        inplane_edges=boundary,
        inplane_edge_faces=boundary_faces,
    )


@dataclass(frozen=True)
class Stitched:
    """The segments of a cut, joined into loops (X2 step 6).

    A node is one distinct edge key: one point of the cut.

    Attributes
    ----------
    tail, head : np.ndarray
        ``(S,)`` node of each segment's ``p0`` and ``p1``.
    n_nodes : int
        Distinct nodes.
    loops : list[np.ndarray]
        Node ids of each closed loop, in order, the start not repeated.
    open_segments : np.ndarray
        Segments of components that are not closed loops (an open chain, or
        a node with more than two segments): outline only.
    """

    tail: np.ndarray
    head: np.ndarray
    n_nodes: int
    loops: list[np.ndarray]
    open_segments: np.ndarray


def stitch(k0: np.ndarray, k1: np.ndarray) -> Stitched:
    """Join segments into closed loops by their exact edge keys.

    One ``depth_first_order`` from a virtual root joined to one node of each
    closed component: a depth-first traversal of a cycle visits its nodes in
    cycle order, so one traversal orders every loop.  Winding is not used.
    """
    n_segments = len(k0)
    if n_segments == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Stitched(empty, empty, 0, [], empty)
    _keys, inverse = np.unique(np.concatenate([k0, k1]), return_inverse=True)
    tail, head = inverse[:n_segments], inverse[n_segments:]
    n_nodes = int(inverse.max()) + 1
    degree = np.bincount(tail, minlength=n_nodes) + np.bincount(head, minlength=n_nodes)
    graph = coo_matrix(
        (np.ones(n_segments, dtype=np.int8), (tail, head)), shape=(n_nodes, n_nodes)
    ).tocsr()
    n_components, label = connected_components(graph, directed=False)
    # A closed loop: every node has exactly two segment ends.  A zero-length
    # segment (both ends on one node) gives that node two ends by itself and
    # joins nothing, so its component is "closed" with one node; it is
    # dropped below with every loop of fewer than three nodes.
    bad = np.zeros(n_components, dtype=bool)
    bad[label[degree != 2]] = True
    by_label = np.argsort(label, kind="stable")
    starts = np.flatnonzero(
        np.concatenate([[True], label[by_label][1:] != label[by_label][:-1]])
    )
    representatives = by_label[starts][~bad]
    open_segments = np.flatnonzero(bad[label[tail]])
    if len(representatives) == 0:
        return Stitched(tail, head, n_nodes, [], open_segments)

    root = n_nodes
    rows = np.concatenate([tail, np.full(len(representatives), root)])
    cols = np.concatenate([head, representatives])
    rooted = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_nodes + 1, n_nodes + 1),
    ).tocsr()
    order = depth_first_order(rooted, root, directed=False, return_predecessors=False)[
        1:
    ]
    splits = np.flatnonzero(np.diff(label[order])) + 1
    loops = [loop for loop in np.split(order, splits) if len(loop) >= 3]
    return Stitched(tail, head, n_nodes, loops, open_segments)


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-handed ``(u, v)`` spanning the plane, with ``u x v`` along it."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def signed_area(loop: np.ndarray) -> float:
    """Shoelace area of a 2D loop, positive for counter-clockwise."""
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    xs, ys = polygon[:, 0], polygon[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    straddles = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    return bool(np.count_nonzero(straddles & (x < crossing)) % 2)


def _nesting_groups(low: np.ndarray, high: np.ndarray) -> list[np.ndarray]:
    """Group loops that can nest: one's bounding box lies inside another's.

    Neighbouring objects have overlapping boxes but not nested ones, so they
    stay separate groups, each triangulated by itself.  Vectorised: no
    Python loop over loops.
    """
    n = len(low)
    order = np.argsort(low[:, 0], kind="stable")
    low_s, high_s = low[order], high[order]
    stop = np.searchsorted(low_s[:, 0], high_s[:, 0], side="right")
    lengths = np.maximum(stop - (np.arange(n) + 1), 0)
    i = np.repeat(np.arange(n), lengths)
    j = (
        (
            np.arange(int(lengths.sum()))
            - np.repeat(np.cumsum(lengths) - lengths, lengths)
        )
        + i
        + 1
    )
    inside = (
        (high_s[j, 0] <= high_s[i, 0])
        & (low_s[j, 1] >= low_s[i, 1])
        & (high_s[j, 1] <= high_s[i, 1])
    )
    i, j = i[inside], j[inside]
    if len(i) == 0:
        return [np.array([k]) for k in order]
    graph = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    by_label = np.argsort(label, kind="stable")
    splits = np.flatnonzero(np.diff(label[by_label])) + 1
    return [order[group] for group in np.split(by_label, splits)]


def _triangulate_rings(rings: list[np.ndarray]) -> np.ndarray:
    """Earcut one outer ring and its holes; indices into the stacked rings."""
    import mapbox_earcut as earcut

    vertices = np.ascontiguousarray(np.concatenate(rings), dtype=np.float64)
    ends = np.cumsum([len(ring) for ring in rings]).astype(np.uint32)
    return earcut.triangulate_float64(vertices, ends).reshape(-1, 3).astype(np.int64)


def loop_sides(
    stitched: Stitched, apex_above: np.ndarray, node_2d: np.ndarray
) -> np.ndarray:
    """Which side of each closed loop its faces point to.

    Parameters
    ----------
    stitched : Stitched
        The loops of a cut.
    apex_above : np.ndarray
        ``Cut.apex_above``.
    node_2d : np.ndarray
        ``(n_nodes, 2)`` node positions in a right-handed basis of the plane
        (:func:`plane_basis`).

    Returns
    -------
    np.ndarray
        ``(n_loops,)`` int: 1 when the loop's faces point out of it (it
        bounds a solid), -1 when they point into it (a cavity), 0 when they
        disagree or the loop has no area.
    """
    n_loops = len(stitched.loops)
    sides = np.zeros(n_loops, dtype=np.int64)
    if n_loops == 0:
        return sides
    lengths = np.array([len(loop) for loop in stitched.loops])
    nodes = np.concatenate(stitched.loops)
    node_loop = np.full(stitched.n_nodes, -1, dtype=np.int64)
    node_loop[nodes] = np.repeat(np.arange(n_loops), lengths)
    position = np.zeros(stitched.n_nodes, dtype=np.int64)
    position[nodes] = np.arange(len(nodes)) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    segments = np.flatnonzero(node_loop[stitched.tail] >= 0)
    tail, head = stitched.tail[segments], stitched.head[segments]
    loop = node_loop[tail]
    # Along the loop as it was traversed, or against it.
    forward = position[head] == (position[tail] + 1) % lengths[loop]
    # A segment as its face orients it (outside on its right) runs
    # counter-clockwise around a solid.  Is that the traversal's direction?
    agrees = forward == apex_above[segments]
    n_agree = np.bincount(loop[agrees], minlength=n_loops)
    n_segments = np.bincount(loop, minlength=n_loops)
    # Every loop's shoelace area as traversed, in one pass: each node with
    # the next node of its loop, about the loop's first node.
    starts = np.cumsum(lengths) - lengths
    first = np.repeat(starts, lengths)
    following = np.arange(len(nodes)) + 1
    following[starts + lengths - 1] = starts
    here = node_2d[nodes] - node_2d[nodes[first]]
    there = here[following]
    area = np.add.reduceat(here[:, 0] * there[:, 1] - there[:, 0] * here[:, 1], starts)
    traversal = np.sign(area).astype(np.int64)
    all_agree = n_agree == n_segments
    consistent = all_agree | (n_agree == 0)
    sides[consistent] = np.where(all_agree, traversal, -traversal)[consistent]
    return sides


def fill_loops(
    loops: list[np.ndarray], sides: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate closed 2D loops (X2 step 7).

    Parameters
    ----------
    loops : list[np.ndarray]
        ``(L_i, 2)`` loops, in order, the start not repeated.
    sides : np.ndarray or None
        ``(n_loops,)`` from :func:`loop_sides`.  Loops that nest are filled
        by winding number: a region is filled where the sum of the sides of
        the loops around it is not zero.  A nest holding a loop of side 0,
        and every nest when *sides* is ``None``, is filled even-odd.

    Returns
    -------
    vertices : np.ndarray
        ``(M,)`` indices into the loops' points stacked in the given order:
        the points the triangles use.
    triangles : np.ndarray
        ``(T, 3)`` indices into ``vertices``.
    """
    if not loops:
        return _NO_FACES, np.zeros((0, 3), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum([len(loop) for loop in loops])])
    # Every loop's bounding box in one pass over the stacked points.
    stacked = np.concatenate(loops)
    low = np.minimum.reduceat(stacked, starts[:-1], axis=0)
    high = np.maximum.reduceat(stacked, starts[:-1], axis=0)
    vertex_parts: list[np.ndarray] = []
    triangle_parts: list[np.ndarray] = []
    base = 0

    def emit(members: list[int]) -> None:
        nonlocal base
        triangles = _triangulate_rings([loops[k] for k in members])
        vertex_parts.append(
            np.concatenate([np.arange(starts[k], starts[k + 1]) for k in members])
        )
        triangle_parts.append(triangles + base)
        base += sum(len(loops[k]) for k in members)

    for group in _nesting_groups(low, high):
        if len(group) == 1:
            emit([int(group[0])])
            continue
        members = [int(k) for k in group]
        areas = {k: abs(signed_area(loops[k])) for k in members}
        contains = {
            a: [
                b
                for b in members
                if b != a
                and areas[a] > areas[b]
                and _point_in_polygon(loops[b][0], loops[a])
            ]
            for a in members
        }
        around = {k: [a for a in members if k in contains[a]] for k in members}
        if sides is None or any(sides[k] == 0 for k in members):
            # Even-odd: the count of loops around a region decides it.
            outside = {k: len(around[k]) % 2 for k in members}
            inside = {k: 1 - outside[k] for k in members}
        else:
            # Winding number just outside and just inside each loop.
            outside = {k: int(sum(sides[a] for a in around[k])) for k in members}
            inside = {k: outside[k] + int(sides[k]) for k in members}
        # An outer ring has nothing outside it and fill inside; a hole is the
        # reverse.  A loop with fill on both sides bounds nothing.
        outers = [k for k in members if inside[k] and not outside[k]]
        holes: dict[int, list[int]] = {k: [] for k in outers}
        for k in members:
            if inside[k] or not outside[k]:
                continue
            # A hole belongs to the innermost outer ring around it.
            rings = [a for a in around[k] if a in holes]
            if rings:
                holes[min(rings, key=areas.__getitem__)].append(k)
        for outer in outers:
            emit([outer, *holes[outer]])
    if not vertex_parts:
        return _NO_FACES, np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(vertex_parts), np.concatenate(triangle_parts)


def _lerp_attribute(
    values: np.ndarray, apex: np.ndarray, other: np.ndarray, t: np.ndarray
) -> np.ndarray:
    start = values[apex].astype(np.float64)
    return start + t[:, None] * (values[other].astype(np.float64) - start)


@dataclass(frozen=True)
class SectionParts:
    """A section, in the kernel's three-column space (X4, before flattening).

    Attributes
    ----------
    fill_positions : np.ndarray
        ``(M, 3)`` float64.
    fill_indices : np.ndarray
        ``(T, 3)`` int64 into ``fill_positions``.
    fill_colors : np.ndarray or None
        ``(M, 4)`` per vertex.
    fill_face_ids : np.ndarray
        ``(T,)`` source face of each fill triangle; -1 for a cap triangle.
    outline_positions : np.ndarray
        ``(2 * S, 3)`` float64: two vertices per segment.
    outline_colors : np.ndarray or None
        ``(2 * S, 4)``.
    outline_face_ids : np.ndarray
        ``(S,)`` source face of each segment.
    n_closed_loops, n_open_segments : int
        Closed loops, and segments of components that did not close.
    """

    fill_positions: np.ndarray
    fill_indices: np.ndarray
    fill_colors: np.ndarray | None
    fill_face_ids: np.ndarray
    outline_positions: np.ndarray
    outline_colors: np.ndarray | None
    outline_face_ids: np.ndarray
    n_closed_loops: int = 0
    n_open_segments: int = 0

    @property
    def is_empty(self) -> bool:
        """Nothing to draw: no fill triangle and no outline segment."""
        return len(self.fill_indices) == 0 and len(self.outline_face_ids) == 0


class _Soup:
    """Fill geometry gathered piece by piece, then stacked once."""

    def __init__(self, colored: bool) -> None:
        self.colored = colored
        self.positions: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []
        self.indices: list[np.ndarray] = []
        self.face_ids: list[np.ndarray] = []
        self.n_vertices = 0

    def add(self, positions, indices, face_ids, colors=None) -> None:
        if len(indices) == 0:
            return
        self.positions.append(np.asarray(positions, dtype=np.float64))
        self.indices.append(np.asarray(indices, dtype=np.int64) + self.n_vertices)
        self.face_ids.append(np.asarray(face_ids, dtype=np.int64))
        if self.colored:
            self.colors.append(np.asarray(colors, dtype=np.float32))
        self.n_vertices += len(positions)

    def stacked(self):
        if not self.indices:
            return (
                np.zeros((0, 3)),
                np.zeros((0, 3), dtype=np.int64),
                np.zeros((0, 4), dtype=np.float32) if self.colored else None,
                _NO_FACES,
            )
        return (
            np.concatenate(self.positions),
            np.concatenate(self.indices),
            np.concatenate(self.colors) if self.colored else None,
            np.concatenate(self.face_ids),
        )


def _cut_outline(
    cut: Cut, positions: np.ndarray, vertex_colors, face_colors
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """The outline of one cut (X2 step 8): two vertices per segment.

    Zero-length segments (a face touching the plane at one vertex) are left
    out; the boundary edges of in-plane faces are added.
    """
    proper = (cut.p0 != cut.p1).any(axis=1)
    points = np.stack([cut.p0[proper], cut.p1[proper]], axis=1).reshape(-1, 3)
    face_ids = cut.face[proper]
    edge_points = positions[cut.inplane_edges].astype(np.float64).reshape(-1, 3)
    colors = None
    if vertex_colors is not None:
        c0 = _lerp_attribute(vertex_colors, cut.apex, cut.v0, cut.t0)[proper]
        c1 = _lerp_attribute(vertex_colors, cut.apex, cut.v1, cut.t1)[proper]
        colors = np.concatenate(
            [
                np.stack([c0, c1], axis=1).reshape(-1, 4),
                vertex_colors[cut.inplane_edges].reshape(-1, 4),
            ]
        )
    elif face_colors is not None:
        colors = np.concatenate(
            [
                np.repeat(face_colors[face_ids], 2, axis=0),
                np.repeat(face_colors[cut.inplane_edge_faces], 2, axis=0),
            ]
        )
    return (
        np.concatenate([points, edge_points]),
        None if colors is None else colors.astype(np.float32),
        np.concatenate([face_ids, cut.inplane_edge_faces]),
    )


def _cap(
    cut: Cut, normal: np.ndarray, soup: _Soup, vertex_colors, face_colors
) -> tuple[int, int]:
    """Fill the closed loops of *cut* into *soup*; cap triangles carry -1.

    Returns ``(closed loops, open segments)``.
    """
    stitched = stitch(cut.k0, cut.k1)
    if not stitched.loops:
        return 0, len(stitched.open_segments)
    node_position = np.empty((stitched.n_nodes, 3))
    node_position[stitched.tail] = cut.p0
    node_position[stitched.head] = cut.p1
    u, v = plane_basis(normal)
    # Project every node once, then pick each loop's rows.
    node_2d = np.stack([node_position @ u, node_position @ v], axis=1)
    loops_2d = [node_2d[loop] for loop in stitched.loops]
    sides = loop_sides(stitched, cut.apex_above, node_2d)
    used, triangles = fill_loops(loops_2d, sides)
    nodes = np.concatenate(stitched.loops)[used]
    colors = None
    if soup.colored:
        node_color = np.empty((stitched.n_nodes, 4))
        if vertex_colors is not None:
            node_color[stitched.tail] = _lerp_attribute(
                vertex_colors, cut.apex, cut.v0, cut.t0
            )
            node_color[stitched.head] = _lerp_attribute(
                vertex_colors, cut.apex, cut.v1, cut.t1
            )
        else:
            # A cut point lies on an edge between two faces; it takes the
            # colour of one of them.
            node_color[stitched.tail] = face_colors[cut.face]
            node_color[stitched.head] = face_colors[cut.face]
        colors = node_color[nodes]
    soup.add(
        node_position[nodes],
        triangles,
        np.full(len(triangles), -1, dtype=np.int64),
        colors,
    )
    return len(stitched.loops), len(stitched.open_segments)


def _add_whole_faces(
    soup: _Soup, positions, indices, faces, vertex_colors, face_colors
) -> None:
    """Add mesh faces to the fill as they are, three vertices each."""
    if len(faces) == 0:
        return
    corners = indices[faces]
    colors = None
    if vertex_colors is not None:
        colors = vertex_colors[corners].reshape(-1, 4)
    elif face_colors is not None:
        colors = np.repeat(face_colors[faces], 3, axis=0)
    soup.add(
        positions[corners].reshape(-1, 3),
        np.arange(3 * len(faces)).reshape(-1, 3),
        faces,
        colors,
    )


def section_cut(
    positions: np.ndarray,
    indices: np.ndarray,
    normal: np.ndarray,
    offset: float,
    tol: float,
    *,
    candidates: np.ndarray | None = None,
    vertex_colors: np.ndarray | None = None,
    face_colors: np.ndarray | None = None,
    outline: bool = True,
    fill: bool = True,
) -> SectionParts:
    """The section of a mesh by one plane (cut mode).

    Parameters
    ----------
    positions : np.ndarray
        ``(V, 3)`` float32.
    indices : np.ndarray
        ``(F, 3)`` faces.
    normal, offset, tol
        The plane and its tolerance; see :func:`cut_plane`.
    candidates : np.ndarray or None
        Face ids to look at; ``None`` is every face.
    vertex_colors, face_colors : np.ndarray or None
        ``(V, 4)`` or ``(F, 4)``; at most one.  Vertex colours are
        interpolated along the cut edges, face colours copied.
    outline, fill : bool
        The parts to build.

    Returns
    -------
    SectionParts
    """
    cut = cut_plane(positions, indices, normal, offset, tol, candidates)
    colored = vertex_colors is not None or face_colors is not None
    soup = _Soup(colored)
    n_loops = n_open = 0
    if fill:
        n_loops, n_open = _cap(cut, normal, soup, vertex_colors, face_colors)
        _add_whole_faces(
            soup, positions, indices, cut.inplane_faces, vertex_colors, face_colors
        )
    fill_positions, fill_indices, fill_colors, fill_face_ids = soup.stacked()
    if outline:
        points, colors, face_ids = _cut_outline(
            cut, positions, vertex_colors, face_colors
        )
    else:
        points = np.zeros((0, 3))
        colors = np.zeros((0, 4), dtype=np.float32) if colored else None
        face_ids = _NO_FACES
    return SectionParts(
        fill_positions=fill_positions,
        fill_indices=fill_indices,
        fill_colors=fill_colors,
        fill_face_ids=fill_face_ids,
        outline_positions=points,
        outline_colors=colors,
        outline_face_ids=face_ids,
        n_closed_loops=n_loops,
        n_open_segments=n_open,
    )


def _clip(polygons: np.ndarray, counts: np.ndarray, distance_of) -> tuple:
    """Clip convex polygons to ``distance >= 0`` (Sutherland-Hodgman).

    ``polygons`` is ``(N, K, C)`` with ``counts`` used vertices each; every
    column is interpolated, so attributes ride along with the positions.
    """
    n, k, width = polygons.shape
    distance = distance_of(polygons)
    out = np.zeros((n, k + 1, width))
    out_counts = np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    for i in range(k):
        valid = i < counts
        j = np.where(i + 1 < counts, i + 1, 0)
        current, following = polygons[:, i], polygons[rows, j]
        d_current, d_following = distance[:, i], distance[rows, j]
        current_in, following_in = d_current >= 0, d_following >= 0
        keep = valid & current_in
        out[rows[keep], out_counts[keep]] = current[keep]
        out_counts[keep] += 1
        crosses = valid & (current_in != following_in)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = d_current[crosses] / (d_current[crosses] - d_following[crosses])
        out[rows[crosses], out_counts[crosses]] = current[crosses] + t[:, None] * (
            following[crosses] - current[crosses]
        )
        out_counts[crosses] += 1
    return out, out_counts


def section_slab(
    positions: np.ndarray,
    indices: np.ndarray,
    normal: np.ndarray,
    low: float,
    high: float,
    tol: float,
    *,
    candidates: np.ndarray | None = None,
    vertex_colors: np.ndarray | None = None,
    face_colors: np.ndarray | None = None,
    outline: bool = True,
    fill: bool = True,
) -> SectionParts:
    """The section of a mesh by the slab ``low <= normal . p <= high`` (X3).

    The fill is the surface inside the slab (faces inside it kept, faces
    crossing either plane clipped) plus a cap at each plane; the outline is
    the two cuts.  A slab thinner than the tolerance is the cut at its
    middle.

    Parameters are those of :func:`section_cut`, with the slab's two
    offsets.  *candidates* must hold every face that touches the slab.
    """
    if high - low <= 2.0 * tol:
        return section_cut(
            positions,
            indices,
            normal,
            (low + high) / 2.0,
            tol,
            candidates=candidates,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
            outline=outline,
            fill=fill,
        )
    colored = vertex_colors is not None or face_colors is not None
    soup = _Soup(colored)
    cuts = [
        cut_plane(positions, indices, normal, plane, tol, candidates)
        for plane in (low, high)
    ]
    n_loops = n_open = 0
    if fill:
        face_ids = (
            np.arange(len(indices)) if candidates is None else np.asarray(candidates)
        )
        faces = indices[face_ids]
        distance = plane_distance(positions, normal, 0.0)
        # One zone per vertex: 0 below, 1 inside, 2 above.
        zone = (distance >= np.float32(low - tol)).view(np.uint8) + (
            distance > np.float32(high + tol)
        ).view(np.uint8)
        z0, z1, z2 = zone[faces[:, 0]], zone[faces[:, 1]], zone[faces[:, 2]]
        lowest = np.minimum(np.minimum(z0, z1), z2)
        highest = np.maximum(np.maximum(z0, z1), z2)
        inside = face_ids[(lowest == 1) & (highest == 1)]
        _add_whole_faces(soup, positions, indices, inside, vertex_colors, face_colors)

        clipped = face_ids[highest > lowest]
        if len(clipped):
            corners = indices[clipped]
            polygons = positions[corners].astype(np.float64)  # (N, 3, 3)
            if vertex_colors is not None:
                polygons = np.concatenate(
                    [polygons, vertex_colors[corners].astype(np.float64)], axis=2
                )
            normal64 = np.asarray(normal, dtype=np.float64)
            counts = np.full(len(polygons), 3)
            polygons, counts = _clip(
                polygons, counts, lambda p: p[..., :3] @ normal64 - low
            )
            polygons, counts = _clip(
                polygons, counts, lambda p: high - p[..., :3] @ normal64
            )
            # Fan triangulation: (0, k, k + 1).
            for k in range(1, polygons.shape[1] - 1):
                has = counts >= k + 2
                if not has.any():
                    continue
                triangle = np.stack(
                    [polygons[has, 0], polygons[has, k], polygons[has, k + 1]], axis=1
                ).reshape(-1, polygons.shape[2])
                colors = None
                if vertex_colors is not None:
                    colors = triangle[:, 3:]
                elif face_colors is not None:
                    colors = np.repeat(face_colors[clipped[has]], 3, axis=0)
                soup.add(
                    triangle[:, :3],
                    np.arange(len(triangle)).reshape(-1, 3),
                    clipped[has],
                    colors,
                )
        for cut in cuts:
            loops, opened = _cap(cut, normal, soup, vertex_colors, face_colors)
            n_loops += loops
            n_open += opened
    fill_positions, fill_indices, fill_colors, fill_face_ids = soup.stacked()
    if outline:
        parts = [
            _cut_outline(cut, positions, vertex_colors, face_colors) for cut in cuts
        ]
        points = np.concatenate([part[0] for part in parts])
        colors = np.concatenate([part[1] for part in parts]) if colored else None
        face_ids_out = np.concatenate([part[2] for part in parts])
    else:
        points = np.zeros((0, 3))
        colors = np.zeros((0, 4), dtype=np.float32) if colored else None
        face_ids_out = _NO_FACES
    return SectionParts(
        fill_positions=fill_positions,
        fill_indices=fill_indices,
        fill_colors=fill_colors,
        fill_face_ids=fill_face_ids,
        outline_positions=points,
        outline_colors=colors,
        outline_face_ids=face_ids_out,
        n_closed_loops=n_loops,
        n_open_segments=n_open,
    )


@dataclass(frozen=True)
class ClosureReport:
    """Whether a mesh is closed (X5, D14).

    Attributes
    ----------
    boundary_edges : int
        Edges used by exactly one face: where the surface is open.
    nonmanifold_edges : int
        Edges used by more than two faces.
    """

    boundary_edges: int
    nonmanifold_edges: int

    @property
    def closed(self) -> bool:
        """No boundary edge and no non-manifold edge."""
        return self.boundary_edges == 0 and self.nonmanifold_edges == 0


def closure_report(indices: np.ndarray, n_vertices: int) -> ClosureReport:
    """Count boundary and non-manifold edges, from one sort of the edges."""
    faces = indices.astype(np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    low = np.minimum(edges[:, 0], edges[:, 1])
    high = np.maximum(edges[:, 0], edges[:, 1])
    proper = low != high  # a degenerate edge is not an edge
    packed = low[proper] * np.int64(n_vertices) + high[proper]
    packed.sort()
    if len(packed) == 0:
        return ClosureReport(0, 0)
    starts = np.flatnonzero(np.concatenate([[True], packed[1:] != packed[:-1]]))
    counts = np.diff(np.concatenate([starts, [len(packed)]]))
    return ClosureReport(int((counts == 1).sum()), int((counts > 2).sum()))
