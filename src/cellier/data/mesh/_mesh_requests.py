# src/cellier/v2/data/mesh/_mesh_requests.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np

    from cellier.transform import ConvexRegion


class MeshSectionRequest(NamedTuple):
    """The plane, or slab, a 2D view cuts a mesh with.

    ``plans/mesh_refactor_v3.md`` X1.  Hashable: it is the section part of
    the request key, so changing any field is a new request.

    Parameters
    ----------
    normal : tuple[float, ...]
        The plane normal in **data** coordinates, one entry per data axis;
        any magnitude.  It is non-zero on exactly one axis outside the
        request's ``output_axes`` (the section axis), and may be non-zero on
        the output axes (an oblique plane).
    offsets : tuple[float, ...]
        ``(offset,)`` in cut mode: the plane ``normal . p = offset``.
        ``(low, high)`` in slab mode: the slab ``low <= normal . p <= high``.
    mode : str
        ``"cut"`` or ``"slab"``.
    outline : bool
        Build the outline (the curve where the surface crosses the plane).
    fill : bool
        Build the fill (the area closed loops enclose; in slab mode, the
        surface inside the slab and a cap at each face).
    """

    normal: tuple[float, ...]
    offsets: tuple[float, ...]
    mode: str = "cut"
    outline: bool = True
    fill: bool = True


class MeshSliceRequest(NamedTuple):
    """Request for one slab-filtered slice of one level of a mesh.

    Parameters
    ----------
    slice_request_id : UUID
        Shared ID for all requests in one planning event.
    chunk_request_id : UUID
        Per-request ID.  For mesh (never tiled) this equals
        ``slice_request_id``.
    scale_index : int
        The level to read, 0 the finest.  A single-level store has level 0
        only.
    displayed_axes : tuple[int, ...]
        Axis indices rendered in the canvas.
    retained_axes : tuple[int, ...]
        The **data** axes this visual's geometry keeps, ascending.

        Not the same list as ``displayed_axes``, which indexes the **world**:
        a ``zyx`` store in a ``czyx`` world retains ``(0, 1, 2)`` while the
        world displays ``(1, 2, 3)``, and a transform that permutes its axes
        retains a different set again.  Indexing the position array with
        world axes raises on the first and silently uploads the wrong columns
        on the second.
    region : ConvexRegion
        The selected region, already pulled back into **data** coordinates
        (design 3.12).  It is the whole filter: a face survives when all
        three of its vertices are inside.
    output_axes : tuple[int, ...]
        The data axes to emit, in the column order of the result.  The
        render visual passes ``retained_axes`` reversed, which is the
        ``(x, y, z)`` order pygfx draws; a 2D result gets a third column of
        zeros.  The store emits this order directly, so the commit on the UI
        thread copies nothing (``plans/mesh_refactor_v3.md`` L5).
    section : MeshSectionRequest or None
        ``None`` reads whole faces (a 3D view, or a 2D view of a mesh with
        no extent across the slice).  Otherwise the read returns a
        :class:`MeshSectionData`: the mesh cut by the plane or slab, and
        ``region`` holds only the constraints the cut does not replace
        (a ``t`` filter, say).
    """

    slice_request_id: UUID
    chunk_request_id: UUID
    scale_index: int
    displayed_axes: tuple[int, ...]
    retained_axes: tuple[int, ...]
    region: ConvexRegion
    output_axes: tuple[int, ...]
    section: MeshSectionRequest | None = None
    #: Clipping planes a slab section is clipped with on the CPU, as
    #: ``(normal, offset)`` pairs in data coordinates, kept where
    #: ``normal . p >= offset`` (clipping planes design 5.2).  Part of the
    #: request key.  Empty in cut mode and in 3D, where the shader clips.
    clip_planes: tuple = ()


@dataclass(frozen=True)
class MeshData:
    """One level of a mesh at one request, ready to upload.

    Returned by a mesh store's ``get_data``.  Every array is contiguous and
    in the dtype and column order the GPU takes, so the render visual builds
    a ``gfx.Geometry`` from it without a copy.

    Parameters
    ----------
    request_id : UUID
        Echo of MeshSliceRequest.slice_request_id.
    positions : np.ndarray
        ``(n_vertices, 3)`` float32, columns in the request's
        ``output_axes`` order; a 2D result has zeros in the third column.
    indices : np.ndarray
        (n_faces, 3) int32.  Reindexed to reference only the vertices
        in ``positions``.
    normals : np.ndarray | None
        ``(n_vertices, 3)`` float32 unit normals in the same column order,
        or ``None`` for a 2D result (drawn unlit).
    colors : np.ndarray | None
        Per-vertex (n_vertices, 4) or per-face (n_faces, 4) float32
        RGBA.  None when the store carries no color data.
    color_mode : str
        ``"vertex"`` or ``"face"``.  Ignored when colors is None.
    is_empty : bool
        True when the slab contained no surviving faces and a
        placeholder geometry was returned.
    original_face_indices : np.ndarray | None
        (n_faces,) int array mapping each row of ``indices`` back to its face
        index in the level's full face array: the slab filter's
        surviving-face list.  ``None`` when every face of the level passed
        (the rows are the level's faces, in order) and on the
        empty-placeholder path.  Used by the render layer to translate a
        pick's rendered face index into the level's face index.
    level : int
        The level the data is of, 0 the finest.
    bounds : np.ndarray | None
        ``(2, 3)`` float64 minimum and maximum of ``positions``, computed
        with the read so the UI thread does not pass over the vertices.
        ``None`` on the empty-placeholder path.
    """

    request_id: UUID
    positions: np.ndarray
    indices: np.ndarray
    normals: np.ndarray | None
    colors: np.ndarray | None
    color_mode: str = "vertex"
    is_empty: bool = False
    original_face_indices: np.ndarray | None = None
    level: int = 0
    bounds: np.ndarray | None = None

    @property
    def shape(self) -> str:
        """Summary string for DEBUG logging."""
        return f"positions={self.positions.shape} indices={self.indices.shape}"


@dataclass(frozen=True)
class MeshSectionData:
    """One level of a mesh cut by a plane or slab, ready to upload (X4).

    Returned by a mesh store's ``get_data`` for a request with a
    ``section``.  Positions are ``(n, 3)`` float32 in the request's
    ``output_axes`` order with zeros in the third column.

    Parameters
    ----------
    request_id : UUID
        Echo of ``MeshSliceRequest.slice_request_id``.
    level : int
        The level the data is of, 0 the finest.
    fill_positions : np.ndarray
        ``(Mf, 3)`` float32.
    fill_indices : np.ndarray
        ``(Tf, 3)`` int32 into ``fill_positions``.
    fill_colors : np.ndarray or None
        ``(Mf, 4)`` float32, per vertex whatever the store's layout.
    fill_face_ids : np.ndarray
        ``(Tf,)`` the level's face each fill triangle came from; -1 for a
        cap triangle (the area a loop encloses belongs to no face).
    outline_positions : np.ndarray
        ``(2 * S, 3)`` float32: two vertices per segment.
    outline_colors : np.ndarray or None
        ``(2 * S, 4)`` float32.
    outline_face_ids : np.ndarray
        ``(S,)`` the level's face each segment lies on.
    color_mode : str
        The **store's** colour layout, ``"vertex"`` or ``"face"``, so the
        render layer can refuse an appearance that declares the other.  The
        arrays here are per vertex either way.
    is_empty : bool
        Neither part has anything to draw.
    fill_bounds, outline_bounds : np.ndarray or None
        ``(2, 3)`` float64 minimum and maximum of each part's positions;
        ``None`` for an empty part.
    n_closed_loops, n_open_segments : int
        Closed loops of the cut, and segments of components that did not
        close (outline only).
    """

    request_id: UUID
    level: int
    fill_positions: np.ndarray
    fill_indices: np.ndarray
    fill_colors: np.ndarray | None
    fill_face_ids: np.ndarray
    outline_positions: np.ndarray
    outline_colors: np.ndarray | None
    outline_face_ids: np.ndarray
    color_mode: str = "vertex"
    is_empty: bool = False
    fill_bounds: np.ndarray | None = None
    outline_bounds: np.ndarray | None = None
    n_closed_loops: int = 0
    n_open_segments: int = 0

    @property
    def shape(self) -> str:
        """Summary string for DEBUG logging."""
        return f"fill={self.fill_indices.shape} outline={self.outline_face_ids.shape}"
