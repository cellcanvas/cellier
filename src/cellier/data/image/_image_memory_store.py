# src/cellier/v2/data/image/_image_memory_store.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator

from cellier.data._base_data_store import BaseDataStore, gridded_axis_extents
from cellier.data._dataset_info import (
    DatasetInfo,
    RowSection,
    format_bytes,
    format_shape,
)

if TYPE_CHECKING:
    from cellier.data._changes import StoreChangeKind
    from cellier.data.image._image_requests import ChunkRequest


class ImageMemoryStore(BaseDataStore):
    """In-memory image data store backed by a numpy array.

    Serves axis-aligned slices or full sub-volumes to the AsyncSlicer.
    All reads are synchronous (the array is in CPU RAM); the method is
    still declared ``async`` to satisfy the AsyncSlicer contract.

    Parameters
    ----------
    data : np.ndarray
        The image data. Any dtype; coerced to float32 on construction.
        Shape convention follows numpy axis order — e.g. (D, H, W) for
        3-D, (H, W) for 2-D, (T, C, D, H, W) for 5-D.
    name : str
        Human-readable label. Default ``"image_memory_store"``.
    id : UUID4
        Unique identifier.  Taken from the ``datastore_id`` of
        ``data_coordinate_systems[0]`` when not given; otherwise generated.
    data_coordinate_systems : list[DataCoordinateSystem]
        The store's coordinate system, as a one-entry list built by the
        caller, with one axis per array dimension.  A voxel grid is
        sample-indexed, so build its axes with ``sampling="discrete"``.
        Empty by default, in which case the store takes the scene's world
        axes when it is added to a scene.
    level_scales : list[tuple[float, ...]]
        Unused by this single-resolution store; left empty.
    level_translations : list[tuple[float, ...]]
        Unused by this single-resolution store; left empty.
    level_transforms : list[AffineTransform]
        The level-0 identity, installed from ``data_coordinate_systems``.
        Not normally passed.
    """

    store_type: Literal["image_memory"] = "image_memory"
    # ``data`` announces a change on ``data_changed`` when reassigned:
    # ``extent`` if its shape changed, ``contents`` otherwise
    # (plans/store_change_events.md).
    _CONTENTS_FIELDS: ClassVar[frozenset[str]] = frozenset({"data"})
    DATASET_INFO_LABEL: ClassVar[str] = "in-memory image"
    name: str = "image_memory_store"
    data: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Pydantic validators / serializers for numpy
    # ------------------------------------------------------------------

    @field_validator("data", mode="before")
    @classmethod
    def _coerce_float32(cls, v: Any) -> np.ndarray:
        """Coerce the input to a contiguous float32 C-array."""
        arr = np.asarray(v, dtype=np.float32)
        return np.ascontiguousarray(arr)

    @field_serializer("data")
    def _serialize_data(self, array: np.ndarray, _info: Any) -> list:
        """Serialise the array as a nested Python list for JSON round-trips."""
        return array.tolist()

    # ------------------------------------------------------------------
    # Read-only properties (used by CellierController.add_image)
    # ------------------------------------------------------------------

    def _change_kind(self, name: str, old: Any, new: Any) -> StoreChangeKind | None:
        """``data`` of a new shape moves the extent; the same shape does not."""
        if name == "data" and np.shape(old) != np.shape(new):
            return "extent"
        return super()._change_kind(name, old, new)

    @property
    def ndim(self) -> int:
        """Number of dimensions in the stored array."""
        return self.data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the stored array in numpy axis order."""
        return tuple(self.data.shape)

    @property
    def n_levels(self) -> int:
        """Always 1 — single-resolution, no multiscale pyramid."""
        return 1

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """List with one entry (level 0 = the full array)."""
        return [self.shape]

    @property
    def axis_extents(self) -> tuple[tuple[float, float], ...]:
        """Per-axis ``(low, high)`` extents in level-0 data coordinates.

        The edge convention: an axis of ``size`` voxels spans
        ``[-0.5, size - 0.5]``.  See
        :attr:`~cellier.data._base_data_store.BaseDataStore.axis_extents`.
        """
        return gridded_axis_extents(self.level_shapes[0])

    # ------------------------------------------------------------------
    # Self-description
    # ------------------------------------------------------------------

    def dataset_info(self) -> DatasetInfo:
        """Describe the array: shape, dtype, value range and footprint.

        The value range is a full pass over the array, affordable only
        because the data is already resident in RAM.  The zarr-backed stores
        deliberately omit it (see their ``dataset_info``).
        """
        rows = [
            *self._identity_rows(),
            ("Shape", format_shape(self.shape)),
            ("Data type", str(self.data.dtype)),
        ]
        if self.data.size:
            rows.append(
                ("Value range", f"[{self.data.min():.4g}, {self.data.max():.4g}]")
            )
        rows.append(("Memory", format_bytes(self.data.nbytes)))
        return DatasetInfo(sections=[RowSection(None, rows)])

    # ------------------------------------------------------------------
    # Async data access (called by AsyncSlicer)
    # ------------------------------------------------------------------

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Return the requested sub-region as a float32 array.

        Interprets ``request.axis_selections`` generically:

        - ``int`` entry  → sliced axis; the integer index is applied and the
          axis is dropped from the output.
        - ``(start, stop)`` tuple → displayed axis; a slice is applied and
          the axis is kept in the output.

        Out-of-bounds coordinates are clamped to array extents and
        zero-padded on the output side so the returned shape always matches
        what the caller requested.

        Parameters
        ----------
        request : ChunkRequest
            Built by ``GFXImageMemoryVisual.build_slice_request[_2d]``.
            ``request.scale_index`` is always 0 (ignored).
            ``request.axis_selections`` has one entry per data axis.

        Returns
        -------
        np.ndarray
            float32 array with one dimension per displayed (tuple) axis.
        """
        store_shape = self.data.shape

        # ── 1. Compute the output shape ─────────────────────────────────
        out_shape: list[int] = []
        for sel in request.axis_selections:
            if isinstance(sel, tuple):
                start, stop = sel
                out_shape.append(stop - start)
            # int → axis dropped from output

        out = np.zeros(out_shape, dtype=np.float32)

        # ── 2. Build clamped source indices and destination slices ───────
        src: list[int | slice] = []
        dst: list[slice] = []
        all_valid = True

        for ax, sel in enumerate(request.axis_selections):
            dim_size = store_shape[ax]
            if isinstance(sel, tuple):
                start, stop = sel
                c_start = max(0, start)
                c_stop = min(dim_size, stop)
                if c_stop <= c_start:
                    all_valid = False
                    break
                src.append(slice(c_start, c_stop))
                dst_start = c_start - start
                dst_stop = dst_start + (c_stop - c_start)
                dst.append(slice(dst_start, dst_stop))
            else:
                # Scalar — clamp to valid range, keeps axis out of output.
                idx = int(np.clip(sel, 0, dim_size - 1))
                src.append(idx)

        if all_valid:
            out[tuple(dst)] = self.data[tuple(src)]

        return out
