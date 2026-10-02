# src/cellier/v2/data/mesh/_mesh_memory_store.py
from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import (
    ConfigDict,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from cellier.data._base_data_store import BaseDataStore, geometry_axis_extents
from cellier.data._dataset_info import (
    DatasetInfo,
    RowSection,
    array_extent_row,
    format_bytes,
)
from cellier.data.mesh._mesh_requests import (  # noqa: TC001
    MeshData,
    MeshSliceRequest,
)
from cellier.data.mesh._mesh_slicing import (
    PLACEHOLDER_INDICES,
    LevelCache,
    MeshLevelArrays,
    closure_text,
    compute_vertex_normals,
    run_slice,
)

# Kept under their old names for callers that import them from here.
_PLACEHOLDER_INDICES = PLACEHOLDER_INDICES
_compute_vertex_normals = compute_vertex_normals


class MeshMemoryStore(BaseDataStore):
    """In-memory triangle mesh data store.

    Parameters
    ----------
    positions : np.ndarray
        (n_vertices, N) float32 vertex positions for any world
        dimensionality N ≥ 2.  For a 3-D scene use shape (n_vertices, 3);
        for a 5-D scene (t, z, y, x, c) use shape (n_vertices, 5).
    indices : np.ndarray
        (n_faces, 3) int32 triangle face indices.
        **Must be int32** — pygfx rejects int64 at upload time.
        int64 input is coerced silently by the validator.
    colors : np.ndarray | None
        Per-vertex (n_vertices, 4) or per-face (n_faces, 4) float32 RGBA.
        Which of the two is declared by ``colors_layout``, never inferred.
    colors_layout : str | None
        ``"vertex"`` or ``"face"``.  **Required whenever ``colors`` is
        set**, and rejected when it is not.

        This used to be inferred by comparing ``colors.shape[0]`` against
        ``n_faces``, which is ambiguous whenever a mesh has as many
        vertices as faces -- a tetrahedron has four of each, so its
        per-vertex colours were reported as per-face and gathered wrongly.
        The layout also decides which rows ``get_data`` gathers, so it is
        not a rendering preference the appearance can supply: it is a fact
        about the array that only the caller knows.
    name : str
        Human-readable label.
    id : UUID4
        Unique identifier.  Taken from the ``datastore_id`` of
        ``data_coordinate_systems[0]`` when not given; otherwise generated.
    data_coordinate_systems : list[DataCoordinateSystem]
        The store's coordinate system, as a one-entry list built by the
        caller, with one axis per ``positions`` column.  Mark an axis
        ``sampling="discrete"`` when its column holds sample indices, such
        as frame numbers.  Empty by default, in which case the store takes
        the scene's world axes when it is added to a scene.
    level_scales : list[tuple[float, ...]]
        Unused by this single-resolution store; left empty.
    level_translations : list[tuple[float, ...]]
        Unused by this single-resolution store; left empty.
    level_transforms : list[AffineTransform]
        The level-0 identity, installed from ``data_coordinate_systems``.
        Not normally passed.
    """

    store_type: Literal["mesh_memory"] = "mesh_memory"
    # Reassigning these announces a change on ``data_changed``
    # (plans/store_change_events.md): positions move the extent.
    _EXTENT_FIELDS: ClassVar[frozenset[str]] = frozenset({"positions"})
    _CONTENTS_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"indices", "colors", "colors_layout"}
    )
    DATASET_INFO_LABEL: ClassVar[str] = "in-memory mesh"
    name: str = "mesh_memory_store"
    positions: np.ndarray
    indices: np.ndarray
    colors: np.ndarray | None = None
    colors_layout: Literal["vertex", "face"] | None = None

    # Normals and bounds indexes, shared by every read; replaced on a change.
    _level_cache: LevelCache = PrivateAttr(default_factory=LevelCache)

    # validate_assignment so `store.colors = ...` re-runs the layout check.
    # Without it the invariant only holds at construction, and assigning
    # colours later leaves colors_layout None -- which `colors_mode` then
    # returns, and `get_data`'s `== "face"` test silently reads as vertex.
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("positions", mode="before")
    @classmethod
    def _coerce_positions(cls, v: Any) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @field_validator("indices", mode="before")
    @classmethod
    def _coerce_indices(cls, v: Any) -> np.ndarray:
        """Coerce to int32 — pygfx rejects int64 index buffers."""
        return np.ascontiguousarray(np.asarray(v, dtype=np.int32))

    @field_validator("colors", mode="before")
    @classmethod
    def _coerce_colors(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @model_validator(mode="after")
    def _check_colors_layout(self) -> MeshMemoryStore:
        """Require an explicit layout for colours, and sanity-check it.

        The length check catches a transposed or wrongly declared array, but
        cannot adjudicate a mesh with as many vertices as faces -- both
        counts are valid there, which is exactly the case the old inference
        got wrong.  In that case the declaration is simply authoritative,
        which is the point of having one.
        """
        if self.colors is None:
            if self.colors_layout is not None:
                raise ValueError(
                    "colors_layout was given without colors. Drop it, or "
                    "pass the colors it describes."
                )
            return self
        if self.colors_layout is None:
            raise ValueError(
                "colors requires an explicit colors_layout of 'vertex' or "
                "'face'. It is not inferred from the array length: a mesh "
                "with as many vertices as faces (a tetrahedron, say) is "
                "ambiguous, and the layout decides which rows are gathered "
                "when slicing."
            )
        expected = self.n_vertices if self.colors_layout == "vertex" else self.n_faces
        other = self.n_faces if self.colors_layout == "vertex" else self.n_vertices
        got = self.colors.shape[0]
        if got != expected and got != other:
            raise ValueError(
                f"colors_layout='{self.colors_layout}' expects {expected} "
                f"rows, but colors has {got}."
            )
        return self

    # ------------------------------------------------------------------
    # Serializers
    # ------------------------------------------------------------------

    @field_serializer("positions")
    def _ser_positions(self, v: np.ndarray, _info: Any) -> list:
        return v.tolist()

    @field_serializer("indices")
    def _ser_indices(self, v: np.ndarray, _info: Any) -> list:
        return v.tolist()

    @field_serializer("colors")
    def _ser_colors(self, v: np.ndarray | None, _info: Any) -> list | None:
        return v.tolist() if v is not None else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions per vertex.

        Present for parity with the points and lines stores, which the mesh
        store previously lacked -- every caller had to reach into
        ``positions.shape[1]`` itself.
        """
        return self.positions.shape[1]

    @property
    def n_vertices(self) -> int:
        return self.positions.shape[0]

    @property
    def axis_extents(self) -> tuple[tuple[float, float], ...] | None:
        """Per-axis ``(low, high)`` extents in level-0 data coordinates.

        The bounding box of the vertices, with no padding -- a vertex is a point,
        not a cell, so there is no half-voxel to add.  ``None`` when the
        store is empty.  See
        :attr:`~cellier.data._base_data_store.BaseDataStore.axis_extents`.
        """
        return self._cached_axis_extents(lambda: geometry_axis_extents(self.positions))

    @property
    def n_faces(self) -> int:
        return self.indices.shape[0]

    @property
    def colors_mode(self) -> str:
        """``'vertex'``, ``'face'``, or ``'none'`` -- the declared layout.

        Reads ``colors_layout`` rather than comparing array lengths.  The
        old inference reported per-vertex colours as per-face on any mesh
        with as many vertices as faces.
        """
        if self.colors is None:
            return "none"
        return self.colors_layout

    # ------------------------------------------------------------------
    # Self-description
    # ------------------------------------------------------------------

    def dataset_info(self) -> DatasetInfo:
        """Describe the mesh: vertex and face counts, extent, footprint.

        ``colors_mode`` is deliberately absent: it describes how the mesh is
        drawn, not what the store holds.

        ``Closed`` says whether the surface is watertight, which decides
        whether a 2D section of it can be filled: an open surface draws its
        outline only.  It is worked out by the first 2D section read, so it
        reads "not computed" before one.
        """
        rows = [
            *self._identity_rows(),
            ("Vertices", str(self.n_vertices)),
            ("Faces", str(self.n_faces)),
            ("Dimensions", str(self.ndim)),
            *array_extent_row(self.positions),
            ("Memory", format_bytes(self.positions.nbytes + self.indices.nbytes)),
            ("Closed", closure_text(self._level_cache.peek(("closure",)))),
        ]
        return DatasetInfo(sections=[RowSection(None, rows)])

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _invalidate_caches(self, kind) -> None:
        """Drop the slicing cache with any change to the mesh."""
        super()._invalidate_caches(kind)
        self._level_cache = LevelCache()

    def level_arrays(self, level: int = 0) -> MeshLevelArrays:
        """The arrays of *level*; this store has level 0 only."""
        if level != 0:
            raise ValueError(
                f"MeshMemoryStore has one level (0); level {level} was asked for."
            )
        return MeshLevelArrays(
            positions=self.positions,
            indices=self.indices,
            colors=self.colors,
            colors_layout=self.colors_layout,
        )

    def level_cache(self, level: int = 0) -> LevelCache:
        """What the reads of *level* share (normals, bounds indexes)."""
        return self._level_cache

    async def get_data(self, request: MeshSliceRequest) -> MeshData:
        """Return slab-filtered, reindexed, upload-ready mesh data.

        The work runs in an executor thread
        (:func:`~cellier.data.mesh._mesh_slicing.slice_mesh`), so the event
        loop stays free.  There are no cancellation checkpoints: the chunk
        scheduler, which issues these reads, never cancels one.

        Inclusion rule
        -------------
        A face survives only when **all** of its vertices are inside the
        request's region.  This is the mesh analogue of the lines store's
        "both endpoints must pass" rule and avoids projecting off-slice
        vertices onto the slice plane with the wrong colors.

        Parameters
        ----------
        request : MeshSliceRequest
            Built by ``GFXMeshVisual``.

        Returns
        -------
        MeshData
            Filtered, reindexed, projected mesh ready for GPU upload.
            ``is_empty=True`` when the slab contained no faces.
        """
        level = int(request.scale_index)
        return await run_slice(
            self.level_arrays(level), request, self.level_cache(level)
        )
