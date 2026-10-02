"""A triangle mesh held at several levels of detail, finest first.

``plans/mesh_refactor_v3.md`` M3.  The levels are supplied by the caller
(build them with quadric decimation).  A visual asks for a level by its
index and never touches the arrays, so a lazy backend can later implement
the same ``get_data``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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
    MeshSectionData,
    MeshSliceRequest,
)
from cellier.data.mesh._mesh_slicing import (
    LevelCache,
    MeshLevelArrays,
    closure_text,
    run_slice,
)


class MeshLevel(BaseModel):
    """One level of detail of a mesh.

    Parameters
    ----------
    positions : np.ndarray
        ``(n_vertices, N)`` float32 vertex positions, in the store's data
        coordinates.  Every level of a store has the same ``N``.
    indices : np.ndarray
        ``(n_faces, 3)`` int32 triangle face indices into ``positions``.
        int64 input is coerced.
    colors : np.ndarray or None
        Per-vertex ``(n_vertices, 4)`` or per-face ``(n_faces, 4)`` float32
        RGBA.  Which of the two is declared by ``colors_layout``.
    colors_layout : {"vertex", "face"} or None
        Required whenever ``colors`` is set, and rejected when it is not.

    A level is frozen: replace the store's ``levels`` to change a mesh, so
    the change is announced.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    positions: np.ndarray
    indices: np.ndarray
    colors: np.ndarray | None = None
    colors_layout: Literal["vertex", "face"] | None = None

    @field_validator("positions", mode="before")
    @classmethod
    def _coerce_positions(cls, v: Any) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @field_validator("indices", mode="before")
    @classmethod
    def _coerce_indices(cls, v: Any) -> np.ndarray:
        """Coerce to int32: pygfx rejects int64 index buffers."""
        return np.ascontiguousarray(np.asarray(v, dtype=np.int32))

    @field_validator("colors", mode="before")
    @classmethod
    def _coerce_colors(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @model_validator(mode="after")
    def _check_shapes(self) -> MeshLevel:
        if self.positions.ndim != 2 or self.positions.shape[1] < 2:
            raise ValueError(
                "positions must be (n_vertices, N) with N >= 2, got shape "
                f"{self.positions.shape}."
            )
        if self.indices.ndim != 2 or self.indices.shape[1] != 3:
            raise ValueError(
                f"indices must be (n_faces, 3), got shape {self.indices.shape}."
            )
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
                "'face'. It is not inferred from the array length."
            )
        n_vertices, n_faces = len(self.positions), len(self.indices)
        expected = n_vertices if self.colors_layout == "vertex" else n_faces
        other = n_faces if self.colors_layout == "vertex" else n_vertices
        got = self.colors.shape[0]
        if got != expected and got != other:
            raise ValueError(
                f"colors_layout='{self.colors_layout}' expects {expected} "
                f"rows, but colors has {got}."
            )
        return self

    @field_serializer("positions", "indices")
    def _ser_array(self, v: np.ndarray, _info: Any) -> list:
        return v.tolist()

    @field_serializer("colors")
    def _ser_colors(self, v: np.ndarray | None, _info: Any) -> list | None:
        return v.tolist() if v is not None else None

    @property
    def n_vertices(self) -> int:
        """Vertices in this level."""
        return self.positions.shape[0]

    @property
    def n_faces(self) -> int:
        """Faces in this level."""
        return self.indices.shape[0]


class MultiscaleMeshStore(BaseDataStore):
    """In-memory triangle mesh with levels of detail.

    Parameters
    ----------
    levels : list[MeshLevel]
        The mesh at each level, **finest first**, at least one.  Every level
        is the same surface in the same data coordinates: there is one data
        coordinate system and no per-level transform.  Every level has the
        same number of position columns.  Either every level has colours,
        all with the same ``colors_layout``, or none has.
    name : str
        Human-readable label.
    id : UUID4
        Unique identifier.  Taken from the ``datastore_id`` of
        ``data_coordinate_systems[0]`` when not given; otherwise generated.
    data_coordinate_systems : list[DataCoordinateSystem]
        The store's coordinate system, as a one-entry list, with one axis
        per position column.  Empty by default, in which case the store
        takes the scene's world axes when it is added to a scene.

    Notes
    -----
    Level numbers here and on a request (``scale_index``) are 0-based, 0 the
    finest.  A visual's ``GeometryLodConfig.coarse_level`` is 1-based, like
    the image settings.

    The store's extent is level 0's.  Build levels with quadric decimation
    rather than quadric clustering: a decimated level stays watertight, so
    its 2D section still closes and is filled.

    Reassigning ``levels`` announces a change.  A level is frozen; to change
    one, assign a new list.
    """

    store_type: Literal["mesh_multiscale"] = "mesh_multiscale"
    _EXTENT_FIELDS: ClassVar[frozenset[str]] = frozenset({"levels"})
    DATASET_INFO_LABEL: ClassVar[str] = "in-memory multiscale mesh"
    name: str = "mesh_multiscale_store"
    levels: list[MeshLevel] = Field(min_length=1)

    # Per level: normals and bounds indexes shared by every read of it.
    _level_caches: dict[int, LevelCache] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode="after")
    def _check_levels(self) -> MultiscaleMeshStore:
        columns = {level.positions.shape[1] for level in self.levels}
        if len(columns) > 1:
            raise ValueError(
                "Every level must have the same number of position columns; "
                f"got {[level.positions.shape[1] for level in self.levels]}."
            )
        layouts = [level.colors_layout for level in self.levels]
        if any(layout is None for layout in layouts) and any(
            layout is not None for layout in layouts
        ):
            raise ValueError(
                "Colours must be on every level or on none; got "
                f"colors_layout per level {layouts}."
            )
        if len(set(layouts)) > 1:
            raise ValueError(
                "Every level must have the same colors_layout: an appearance "
                f"declares one color_mode for the mesh. Got {layouts}."
            )
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    # There is deliberately no ``n_levels``: the base store reads that name
    # as "one coordinate system per level", and this store has one system.

    @property
    def level_count(self) -> int:
        """How many levels the mesh has."""
        return len(self.levels)

    @property
    def ndim(self) -> int:
        """Number of position columns."""
        return self.levels[0].positions.shape[1]

    @property
    def n_vertices(self) -> int:
        """Vertices in the finest level."""
        return self.levels[0].n_vertices

    @property
    def n_faces(self) -> int:
        """Faces in the finest level."""
        return self.levels[0].n_faces

    @property
    def axis_extents(self) -> tuple[tuple[float, float], ...] | None:
        """Per-axis ``(low, high)`` of the **finest level's** vertices.

        ``None`` when that level is empty.  See
        :attr:`~cellier.data._base_data_store.BaseDataStore.axis_extents`.
        """
        return self._cached_axis_extents(
            lambda: geometry_axis_extents(self.levels[0].positions)
        )

    @property
    def colors_mode(self) -> str:
        """``'vertex'``, ``'face'``, or ``'none'``: the declared layout."""
        layout = self.levels[0].colors_layout
        return "none" if layout is None else layout

    # ------------------------------------------------------------------
    # Self-description
    # ------------------------------------------------------------------

    def dataset_info(self) -> DatasetInfo:
        """Describe the mesh: its levels, extent and footprint.

        ``Closed`` says, per level, whether the surface is watertight, which
        decides whether a 2D section of that level can be filled.  It is
        worked out by the level's first 2D section read.
        """
        nbytes = sum(
            level.positions.nbytes + level.indices.nbytes for level in self.levels
        )
        rows = [
            *self._identity_rows(),
            ("Levels", str(self.level_count)),
            ("Dimensions", str(self.ndim)),
            *array_extent_row(self.levels[0].positions),
            ("Memory", format_bytes(nbytes)),
        ]
        for index, level in enumerate(self.levels):
            closed = closure_text(self.level_cache(index).peek(("closure",)))
            rows.append(
                (
                    f"Level {index + 1}",
                    f"{level.n_vertices} vertices, {level.n_faces} faces, "
                    f"closed: {closed}",
                )
            )
        return DatasetInfo(sections=[RowSection(None, rows)])

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _invalidate_caches(self, kind) -> None:
        """Drop every level's slicing cache with any change to the mesh."""
        super()._invalidate_caches(kind)
        self._level_caches = {}

    def _check_level(self, level: int) -> int:
        level = int(level)
        if not 0 <= level < len(self.levels):
            raise ValueError(
                f"MultiscaleMeshStore '{self.name}' has levels 0 to "
                f"{len(self.levels) - 1}; level {level} was asked for."
            )
        return level

    def level_arrays(self, level: int = 0) -> MeshLevelArrays:
        """The arrays of *level* (0 is the finest)."""
        mesh = self.levels[self._check_level(level)]
        return MeshLevelArrays(
            positions=mesh.positions,
            indices=mesh.indices,
            colors=mesh.colors,
            colors_layout=mesh.colors_layout,
        )

    def level_cache(self, level: int = 0) -> LevelCache:
        """What the reads of *level* share (normals, bounds indexes)."""
        return self._level_caches.setdefault(self._check_level(level), LevelCache())

    async def get_data(self, request: MeshSliceRequest) -> MeshData | MeshSectionData:
        """Return one level, sliced and upload-ready.

        The same read as :meth:`MeshMemoryStore.get_data`, on the level the
        request names (``scale_index``, 0 the finest).  It runs in an
        executor thread, with no cancellation checkpoints.

        Parameters
        ----------
        request : MeshSliceRequest
            Built by ``GFXMeshVisual``.

        Returns
        -------
        MeshData or MeshSectionData
            The level's faces inside the request's region, or its section.
            ``level`` is the level read.
        """
        level = self._check_level(request.scale_index)
        return await run_slice(
            self.level_arrays(level), request, self.level_cache(level)
        )
