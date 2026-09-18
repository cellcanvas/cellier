# src/cellier/v2/data/points/_points_memory_store.py
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
from cellier.data.points._points_requests import PointsData, PointsSliceRequest


def _placeholder_positions(n_display: int) -> np.ndarray:
    """One invisible point, returned when the filter produces none.

    pygfx will not accept empty geometry, so the store returns a single
    zeroed vertex instead.  It is sized to the retained axes rather than
    fixed at three: a rank-4 store displaying data axes ``(2, 3)`` would
    otherwise index a ``(1, 3)`` constant with axis 3 and raise (F8.4).
    """
    return np.zeros((1, n_display), dtype=np.float32)


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

    store_type: Literal["points_memory"] = "points_memory"
    # Reassigning these announces a change on ``data_changed``
    # (plans/store_change_events.md): positions move the extent.
    _EXTENT_FIELDS: ClassVar[frozenset[str]] = frozenset({"positions"})
    _CONTENTS_FIELDS: ClassVar[frozenset[str]] = frozenset({"colors", "sizes"})
    DATASET_INFO_LABEL: ClassVar[str] = "in-memory points"
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
    def axis_extents(self) -> tuple[tuple[float, float], ...] | None:
        """Per-axis ``(low, high)`` extents in level-0 data coordinates.

        The bounding box of the points, with no padding -- a vertex is a point,
        not a cell, so there is no half-voxel to add.  ``None`` when the
        store is empty.  See
        :attr:`~cellier.data._base_data_store.BaseDataStore.axis_extents`.
        """
        return self._cached_axis_extents(lambda: geometry_axis_extents(self.positions))

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
    # Self-description
    # ------------------------------------------------------------------

    def dataset_info(self) -> DatasetInfo:
        """Describe the point cloud: count, dimensionality, extent, footprint.

        ``color_mode`` and ``size_mode`` are deliberately absent: they
        describe how the points are *drawn*, not what the store holds, and
        the appearance model is where a reader should look for that.
        """
        rows = [
            *self._identity_rows(),
            ("Points", str(self.n_points)),
            ("Dimensions", str(self.ndim)),
            *array_extent_row(self.positions),
            ("Memory", format_bytes(self.positions.nbytes)),
        ]
        return DatasetInfo(sections=[RowSection(None, rows)])

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
        # Ascending: the uploaded vertex buffer's axis order is the data's,
        # and a display permutation lives in the node matrix (design 3.14).
        # ``retained_axes`` is read off the visual's ``data -> world``
        # transform.  ``displayed_axes`` indexes the **world**, so using it
        # here would raise on a store of lower rank than the world and
        # silently upload the wrong columns on a transform that permutes its
        # axes; it was the fallback for a headlessly constructed visual until
        # v1 was retired (R8.3).
        displayed = list(request.retained_axes)

        # ── Phase 1: build proximity mask ────────────────────────────
        # A point survives if it passes the proximity test on EVERY
        # non-displayed (sliced) axis.
        # The region *is* the filter (design 3.12).  For images it is reduced
        # to a bounding box and rounded; for points there is nothing to round
        # -- the constraints apply to the points themselves.  A 3-D view has
        # no slabs, so the region is unbounded, ``contains`` is all-True, and
        # the "no sliced axes so the loop does not run" special case
        # disappears.
        point_mask = request.region.contains(positions)

        # ── Checkpoint A ─────────────────────────────────────────────
        await asyncio.sleep(0)

        # ── Phase 2: gather surviving points ─────────────────────────
        surviving_indices = np.where(point_mask)[0]

        if surviving_indices.shape[0] == 0:
            # Empty slab — return placeholder so the node stays valid.
            return PointsData(
                request_id=request.slice_request_id,
                positions=_placeholder_positions(len(displayed)),
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
