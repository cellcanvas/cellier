# src/cellier/v2/data/points/_points_memory_store.py
from __future__ import annotations

import asyncio
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator

from cellier.data._base_data_store import BaseDataStore
from cellier.data.points._points_requests import PointsData, PointsSliceRequest

# Placeholder returned when the proximity filter produces zero points.
# A single invisible point avoids the "empty geometry is illegal" restriction
# in pygfx.
_PLACEHOLDER_POSITIONS = np.zeros((1, 3), dtype=np.float32)


class PointsMemoryStore(BaseDataStore):
    """In-memory point-cloud data store backed by numpy arrays.

    All reads are synchronous (data is in CPU RAM); ``get_data`` is
    declared ``async`` to satisfy the AsyncSlicer contract and to
    provide a single cancellation checkpoint.

    Positions are stored in *data-axis order*: column 0 is axis 0 (z),
    column 1 is axis 1 (y), column 2 is axis 2 (x).  The render layer
    applies the ``[:, [2, 1, 0]]`` reversal before uploading to pygfx.

    Parameters
    ----------
    positions : np.ndarray
        (n_points, ndim) float32 array.
    colors : np.ndarray | None
        (n_points, 4) float32 RGBA, index-matched to positions.
        Pass None for uniform-color rendering.
    sizes : np.ndarray | None
        (n_points,) float32 per-point sizes.
        Pass None for uniform-size rendering.
    name : str
        Human-readable label.
    """

    store_type: Literal["points_memory"] = "points_memory"
    name: str = "points_memory_store"
    positions: np.ndarray
    colors: np.ndarray | None = None
    sizes: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("positions", mode="before")
    @classmethod
    def _coerce_positions(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        return np.ascontiguousarray(arr)

    @field_validator("colors", mode="before")
    @classmethod
    def _coerce_colors(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float32)
        return np.ascontiguousarray(arr)

    @field_validator("sizes", mode="before")
    @classmethod
    def _coerce_sizes(cls, v: Any) -> np.ndarray | None:
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

    @field_serializer("sizes")
    def _serialize_sizes(self, arr: np.ndarray | None, _info: Any) -> list | None:
        return arr.tolist() if arr is not None else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions per point."""
        return self.positions.shape[1]

    @property
    def n_points(self) -> int:
        """Total number of points in the store."""
        return self.positions.shape[0]

    @property
    def color_mode(self) -> str:
        """``"vertex"`` when per-point colors are present, else ``"uniform"``.

        **Descriptive only.**  It reports what this store carries; it does
        not decide what the material does.  Where RGB comes from is declared
        on the appearance and honoured verbatim (D20) -- the render layer
        never infers it from the data and never writes it back at commit
        time.
        """
        return "vertex" if self.colors is not None else "uniform"

    @property
    def size_mode(self) -> str:
        """``"vertex"`` when per-point sizes are present, else ``"uniform"``."""
        return "vertex" if self.sizes is not None else "uniform"

    # ------------------------------------------------------------------
    # Async data access — one checkpoint for cancellability
    # ------------------------------------------------------------------

    async def get_data(self, request: PointsSliceRequest) -> PointsData:
        """Return proximity-filtered point data for *request*.

        Checkpoint
        ----------
        A  After the proximity mask is built but before gathering
           surviving points.  Fires if the slider is moved quickly
           enough to cancel the task before the gather.

        If CancelledError fires at the checkpoint, the callback is never
        called, preventing stale geometry from reaching the GPU.

        Parameters
        ----------
        request : PointsSliceRequest
            Built by GFXPointsMemoryVisual.build_slice_request[_2d].

        Returns
        -------
        PointsData
            Proximity-filtered, projected points ready for GPU upload.
            ``is_empty=True`` when the filter produced zero points.
        """
        positions = self.positions  # (n_points, ndim)
        colors = self.colors
        sizes = self.sizes
        displayed = list(request.displayed_axes)

        # ── Phase 1: build proximity mask ────────────────────────────
        # A point survives if it passes the proximity test on EVERY
        # non-displayed (sliced) axis.
        point_mask = np.ones(self.n_points, dtype=bool)

        for axis, idx in request.slice_indices.items():
            lo = float(idx) - request.thickness
            hi = float(idx) + request.thickness
            point_mask &= (positions[:, axis] >= lo) & (positions[:, axis] <= hi)

        # ── Checkpoint A ─────────────────────────────────────────────
        await asyncio.sleep(0)

        # ── Phase 2: gather surviving points ─────────────────────────
        surviving_indices = np.where(point_mask)[0]

        if surviving_indices.shape[0] == 0:
            # Empty slab — return placeholder so the node stays valid.
            return PointsData(
                request_id=request.slice_request_id,
                positions=_PLACEHOLDER_POSITIONS[:, displayed],
                colors=None,
                sizes=None,
                color_mode="uniform",
                size_mode="uniform",
                is_empty=True,
            )

        new_positions = positions[surviving_indices][:, displayed]  # (n_surv, n_disp)
        new_colors = colors[surviving_indices] if colors is not None else None
        new_sizes = sizes[surviving_indices] if sizes is not None else None

        return PointsData(
            request_id=request.slice_request_id,
            positions=new_positions,
            colors=new_colors,
            sizes=new_sizes,
            color_mode=self.color_mode,
            size_mode=self.size_mode,
            is_empty=False,
            original_indices=surviving_indices,
        )
