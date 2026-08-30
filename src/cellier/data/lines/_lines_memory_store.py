# src/cellier/v2/data/lines/_lines_memory_store.py
from __future__ import annotations

import asyncio
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator

from cellier.data._base_data_store import BaseDataStore
from cellier.data.lines._lines_requests import LinesData, LinesSliceRequest

# Placeholder returned when the slab filter produces zero surviving segments.
# A single degenerate segment (both vertices at the origin) avoids the
# "empty geometry is illegal" restriction in pygfx.  LineSegmentMaterial
# requires an even vertex count, so the minimum placeholder is two vertices.
_PLACEHOLDER_POSITIONS = np.zeros((2, 3), dtype=np.float32)


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
    """

    store_type: Literal["lines_memory"] = "lines_memory"
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
        n_vertices = positions.shape[0]
        displayed = list(request.displayed_axes)

        # ── Phase 1: build per-vertex slab mask, then require both
        #             endpoints of each segment to pass ───────────────
        if request.slice_indices:
            vertex_mask = np.ones(n_vertices, dtype=bool)
            for axis, idx in request.slice_indices.items():
                lo = float(idx) - request.thickness
                hi = float(idx) + request.thickness
                vertex_mask &= (positions[:, axis] >= lo) & (positions[:, axis] <= hi)
            # Reshape to (n_segments, 2) and require BOTH endpoints True.
            segment_mask = vertex_mask.reshape(-1, 2).all(axis=1)  # (n_segments,)
            surviving_edges = np.where(segment_mask)[0]
            vertex_mask = np.repeat(segment_mask, 2)  # (n_vertices,)
        else:
            # 3D view — all segments survive.
            vertex_mask = np.ones(n_vertices, dtype=bool)
            surviving_edges = np.arange(n_vertices // 2)

        # ── Checkpoint A ─────────────────────────────────────────────
        await asyncio.sleep(0)

        # ── Phase 2: gather surviving vertices ───────────────────────
        surviving_positions = positions[vertex_mask]

        if surviving_positions.shape[0] == 0:
            return LinesData(
                request_id=request.slice_request_id,
                positions=_PLACEHOLDER_POSITIONS[:, displayed],
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
