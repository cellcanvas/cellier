# src/cellier/v2/data/lines/_lines_memory_store.py
from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator

from cellier.data._base_data_store import BaseDataStore, geometry_axis_extents
from cellier.data._dataset_info import (
    DatasetInfo,
    RowSection,
    array_extent_row,
    format_bytes,
)
from cellier.data.lines._lines_requests import LinesData, LinesSliceRequest


def _placeholder_positions(n_display: int) -> np.ndarray:
    """One invisible segment, returned when the filter produces none.

    A single degenerate segment (both vertices at the origin) avoids the
    "empty geometry is illegal" restriction in pygfx.  ``LineSegmentMaterial``
    requires an even vertex count, so the minimum placeholder is two vertices.

    Sized to the retained axes rather than fixed at three, for the reason
    given in the points store: a rank-4 store displaying data axes ``(2, 3)``
    would otherwise index a ``(2, 3)`` constant with axis 3 and raise (F8.4).
    """
    return np.zeros((2, n_display), dtype=np.float32)


class LinesMemoryStore(BaseDataStore):
    """In-memory line-segment data store backed by numpy arrays.

    Stores a collection of line segments as vertex pairs.  For segment n,
    ``positions[n * 2]`` is the start point and ``positions[n * 2 + 1]``
    is the end point.

    Positions are stored in *data-axis order*: column 0 is axis 0 (z),
    column 1 is axis 1 (y), column 2 is axis 2 (x).  The render layer
    applies the ``[:, [2, 1, 0]]`` reversal before uploading to pygfx.

    All reads are synchronous (data is in CPU RAM); ``get_data`` is
    declared ``async`` to satisfy the AsyncSlicer contract and to
    provide a single cancellation checkpoint.

    Parameters
    ----------
    positions : np.ndarray
        (n_vertices, ndim) float32 array.  Must have an even number of rows.
    colors : np.ndarray | None
        (n_vertices, 4) float32 RGBA, index-matched to positions.
        Pass None for uniform-color rendering.
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

    store_type: Literal["lines_memory"] = "lines_memory"
    # Reassigning these announces a change on ``data_changed``
    # (plans/store_change_events.md): positions move the extent.
    _EXTENT_FIELDS: ClassVar[frozenset[str]] = frozenset({"positions"})
    _CONTENTS_FIELDS: ClassVar[frozenset[str]] = frozenset({"colors"})
    DATASET_INFO_LABEL: ClassVar[str] = "in-memory lines"
    name: str = "lines_memory_store"
    positions: np.ndarray
    colors: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("positions", mode="before")
    @classmethod
    def _coerce_positions(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"positions must be a 2-D array, got shape {arr.shape}")
        if arr.shape[0] % 2 != 0:
            raise ValueError(
                f"positions must have an even number of rows (vertex pairs), "
                f"got {arr.shape[0]}"
            )
        return np.ascontiguousarray(arr)

    @field_validator("colors", mode="before")
    @classmethod
    def _coerce_colors(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float32)
        return np.ascontiguousarray(arr)

    # ------------------------------------------------------------------
    # Serializers
    # ------------------------------------------------------------------

    @field_serializer("positions")
    def _serialize_positions(self, arr: np.ndarray, _info: Any) -> list:
        return arr.tolist()

    @field_serializer("colors")
    def _serialize_colors(self, arr: np.ndarray | None, _info: Any) -> list | None:
        return arr.tolist() if arr is not None else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions per vertex."""
        return self.positions.shape[1]

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
    def n_segments(self) -> int:
        """Number of line segments (half the number of vertices)."""
        return self.positions.shape[0] // 2

    @property
    def color_mode(self) -> str:
        """``"vertex"`` when per-vertex colors are present, else ``"uniform"``.

        **Descriptive only.**  It reports what this store carries; it does
        not decide what the material does.  Where RGB comes from is declared
        on the appearance and honoured verbatim (D20) -- the render layer
        never infers it from the data and never writes it back at commit
        time.
        """
        return "vertex" if self.colors is not None else "uniform"

    # ------------------------------------------------------------------
    # Self-description
    # ------------------------------------------------------------------

    def dataset_info(self) -> DatasetInfo:
        """Describe the segments: count, dimensionality, extent, footprint.

        The vertex count is spelled out beside the segment count to make the
        two-vertices-per-segment layout visible; a reader comparing this
        block against the raw array would otherwise see twice what they
        expect.  ``color_mode`` is absent -- it is a display property.
        """
        rows = [
            *self._identity_rows(),
            ("Segments", str(self.n_segments)),
            ("Vertices", str(self.positions.shape[0])),
            ("Dimensions", str(self.ndim)),
            *array_extent_row(self.positions),
            ("Memory", format_bytes(self.positions.nbytes)),
        ]
        return DatasetInfo(sections=[RowSection(None, rows)])

    # ------------------------------------------------------------------
    # Async data access — one checkpoint for cancellability
    # ------------------------------------------------------------------

    async def get_data(self, request: LinesSliceRequest) -> LinesData:
        """Return slab-filtered segment data for *request*.

        Checkpoint
        ----------
        A  After the per-vertex slab mask is built but before gathering
           surviving vertices.  Fires if the slider is moved quickly
           enough to cancel the task before the gather.

        If CancelledError fires at the checkpoint the callback is never
        called, preventing stale geometry from reaching the GPU.

        Inclusion rule
        --------------
        A segment survives when **both** of its endpoints pass the
        proximity test on every non-displayed axis:
            ``slice_index - thickness <= coord <= slice_index + thickness``

        Parameters
        ----------
        request : LinesSliceRequest
            Built by GFXLinesMemoryVisual.build_slice_request[_2d].

        Returns
        -------
        LinesData
            Filtered, projected segment data ready for GPU upload.
            ``is_empty=True`` when the filter produced zero segments.
        """
        positions = self.positions  # (n_vertices, ndim)
        colors = self.colors  # (n_vertices, 4) or None
        # Ascending: the uploaded vertex buffer's axis order is the data's,
        # and a display permutation lives in the node matrix (design 3.14).
        # ``retained_axes`` is read off the visual's ``data -> world``
        # transform and is the right answer whenever the controller has placed
        # the visual.  ``displayed_axes`` indexes the **world**, so using it
        # here raises on a store of lower rank than the world and silently
        # uploads the wrong columns on a transform that permutes its axes; it
        # remains the fallback for a headlessly constructed visual, which has
        # no transform to read.
        displayed = list(request.retained_axes)

        # ── Phase 1: build per-vertex slab mask, then require both
        #             endpoints of each segment to pass ───────────────
        # The region is the filter (design 3.12).  A 3-D view's region is
        # unbounded, so ``contains`` is all-True and every segment survives --
        # the same outcome the "no sliced axes" branch produced, by one rule
        # instead of three.
        vertex_mask = request.region.contains(positions)
        # Reshape to (n_segments, 2) and require BOTH endpoints True.
        segment_mask = vertex_mask.reshape(-1, 2).all(axis=1)  # (n_segments,)
        surviving_edges = np.where(segment_mask)[0]
        vertex_mask = np.repeat(segment_mask, 2)  # (n_vertices,)

        # ── Checkpoint A ─────────────────────────────────────────────
        await asyncio.sleep(0)

        # ── Phase 2: gather surviving vertices ───────────────────────
        surviving_positions = positions[vertex_mask]

        if surviving_positions.shape[0] == 0:
            return LinesData(
                request_id=request.slice_request_id,
                positions=_placeholder_positions(len(displayed)),
                colors=None,
                color_mode="uniform",
                is_empty=True,
            )

        proj_positions = surviving_positions[:, displayed]
        proj_colors = colors[vertex_mask] if colors is not None else None

        return LinesData(
            request_id=request.slice_request_id,
            positions=proj_positions,
            colors=proj_colors,
            color_mode=self.color_mode,
            is_empty=False,
            original_edge_indices=surviving_edges,
        )
