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

_ACCEPTED_DTYPE_TYPES = {np.int8, np.int16, np.int32}


class LabelMemoryStore(BaseDataStore):
    """In-memory label data store backed by a numpy integer array.

    Serves axis-aligned slices or full sub-volumes as int32 arrays.
    Source dtype may be int8, int16, or int32; int64/uint* are rejected.

    Parameters
    ----------
    data : np.ndarray
        Integer label array (int8, int16, or int32). Shape follows numpy
        axis order — e.g. (D, H, W) for 3-D, (H, W) for 2-D.

        Deserialisation also accepts the ``{"dtype": ..., "values": ...}``
        mapping this store serialises to; the dtype is validated, not
        coerced, so it has to survive the round trip.
    name : str
        Human-readable label. Default ``"label_memory_store"``.
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

    store_type: Literal["label_memory"] = "label_memory"
    # ``data`` announces a change on ``data_changed`` when reassigned:
    # ``extent`` if its shape changed, ``contents`` otherwise
    # (plans/store_change_events.md).
    _CONTENTS_FIELDS: ClassVar[frozenset[str]] = frozenset({"data"})
    DATASET_INFO_LABEL: ClassVar[str] = "in-memory labels"
    name: str = "label_memory_store"
    data: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    @classmethod
    def _validate_integer_dtype(cls, v: Any) -> np.ndarray:
        """Accept a label array, or the ``{"dtype", "values"}`` serialised form.

        This store *validates* the dtype where its siblings coerce, so that a
        caller handing over an int64 array is told to narrow it deliberately
        rather than having it silently truncated.  That makes the dtype
        load-bearing on the way back in too, which is why the serialised form
        carries it -- see :meth:`_serialize_data`.

        A payload's declared dtype is checked by the same rule as a live
        array's, so a hand-edited ``{"dtype": "int64", ...}`` is rejected with
        the same message as ``np.zeros(..., dtype=np.int64)``.
        """
        if isinstance(v, dict):
            v = cls._array_from_payload(v)
        arr = np.asarray(v)
        if arr.dtype.type not in _ACCEPTED_DTYPE_TYPES:
            raise ValueError(
                f"LabelMemoryStore requires int8, int16, or int32 dtype. "
                f"Got {arr.dtype}. Cast your data to np.int32 first."
            )
        return np.ascontiguousarray(arr)

    @staticmethod
    def _array_from_payload(payload: dict) -> np.ndarray:
        """Rebuild the array from a serialised ``{"dtype", "values"}`` mapping.

        Returns the array with its declared dtype and leaves the accept/reject
        decision to the caller, so there is one dtype rule rather than two.
        """
        missing = {"dtype", "values"} - set(payload)
        if missing:
            raise ValueError(
                f"A serialised LabelMemoryStore 'data' payload needs both "
                f"'dtype' and 'values'; missing {sorted(missing)}."
            )
        try:
            dtype = np.dtype(payload["dtype"])
        except TypeError as error:
            raise ValueError(
                f"Unknown dtype {payload['dtype']!r} in a serialised "
                f"LabelMemoryStore 'data' payload."
            ) from error
        return np.asarray(payload["values"], dtype=dtype)

    @field_serializer("data")
    def _serialize_data(self, array: np.ndarray, _info: Any) -> dict:
        """Serialise as ``{"dtype": ..., "values": ...}``.

        The sibling stores write a bare ``tolist()`` because they coerce on
        the way back in -- whatever dtype numpy infers from the nested list is
        overwritten.  This store validates instead, and numpy infers int64
        from a list of Python ints, so a bare list round-tripped to a dtype
        this store rejects: it could be written out but never read back.
        Naming the dtype keeps the payload self-describing and restores the
        original width rather than widening every label array to int64.
        """
        return {"dtype": array.dtype.name, "values": array.tolist()}

    def _change_kind(self, name: str, old: Any, new: Any) -> StoreChangeKind | None:
        """``data`` of a new shape moves the extent; the same shape does not."""
        if name == "data" and np.shape(old) != np.shape(new):
            return "extent"
        return super()._change_kind(name, old, new)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def n_levels(self) -> int:
        return 1

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        return [self.shape]

    @property
    def axis_extents(self) -> tuple[tuple[float, float], ...]:
        """Per-axis ``(low, high)`` extents in level-0 data coordinates.

        The edge convention: an axis of ``size`` voxels spans
        ``[-0.5, size - 0.5]``.  See
        :attr:`~cellier.data._base_data_store.BaseDataStore.axis_extents`.
        """
        return gridded_axis_extents(self.level_shapes[0])

    def dataset_info(self) -> DatasetInfo:
        """Describe the array, including how many distinct labels it holds.

        The label count is a full pass over the array (``np.unique``), which
        is affordable only because the data is already resident in RAM.
        ``OMEZarrLabelDataStore`` deliberately omits it: there the same row
        would mean reading every chunk off disk.
        """
        rows = [
            *self._identity_rows(),
            ("Shape", format_shape(self.shape)),
            ("Data type", str(self.dtype)),
        ]
        if self.data.size:
            unique = np.unique(self.data)
            rows.append(("Labels", str(len(unique))))
            rows.append(("Max label", str(int(unique[-1]))))
        rows.append(("Memory", format_bytes(self.data.nbytes)))
        return DatasetInfo(sections=[RowSection(None, rows)])

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Return the requested sub-region as an int32 array.

        Interprets ``request.axis_selections`` generically:
        - ``int`` entry  → sliced axis (dropped from output)
        - ``(start, stop)`` tuple → displayed axis (kept in output)

        Out-of-bounds coordinates are clamped and zero-padded.
        Always returns int32 regardless of source dtype.
        """
        store_shape = self.data.shape

        out_shape: list[int] = []
        for sel in request.axis_selections:
            if isinstance(sel, tuple):
                start, stop = sel
                out_shape.append(stop - start)

        out = np.zeros(out_shape, dtype=np.int32)

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
                idx = int(np.clip(sel, 0, dim_size - 1))
                src.append(idx)

        if all_valid:
            out[tuple(dst)] = self.data[tuple(src)].astype(np.int32)

        return out
