"""OMEZarrLabelDataStore — data store for OME-Zarr v0.5 label images.

Reads OME-NGFF label groups and opens per-level tensorstore handles
for async data access.  Returns int32 bricks (never float32).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    import tensorstore as ts

    from cellier.transform import DataCoordinateSystem

from cellier.data._base_data_store import BaseDataStore, gridded_axis_extents
from cellier.data._dataset_info import DatasetInfo, ome_zarr_dataset_info
from cellier.data._tensorstore_cache import (
    DEFAULT_CACHE_POOL_BYTES,
    TensorStoreCacheMixin,
)
from cellier.data.image._ome_zarr_image_store import (
    _level_geometry,
    _ngff_coordinate_systems,
    _validate_uri_scheme,
)

_ACCEPTED_LABEL_DTYPES = {np.int8, np.int16, np.int32}


class OMEZarrLabelDataStore(TensorStoreCacheMixin, BaseDataStore):
    """Multiscale OME-Zarr label store returning int32 bricks.

    Use the :meth:`from_path` class method to construct from a URI that
    points to an OME-NGFF label group (containing ``ome`` → ``multiscales``
    metadata).

    Parameters
    ----------
    store_type : Literal["ome_zarr_label"]
        Discriminator field. Always ``"ome_zarr_label"``.
    zarr_path : str
        URI to the label group.  Must point to the label sub-group
        (e.g. ``"file:///path/to/seg.ome.zarr/labels/cells"``),
        not the root OME-Zarr.
    multiscale_index : int
        Which multiscale entry to use (default 0).
    scale_names : list[str]
        Per-level relative array paths, finest to coarsest.
    level_scales : list[tuple[float, ...]]
        Full-rank per-level scale: level-k voxels in level-0 voxels.
    level_translations : list[tuple[float, ...]]
        The offset half of the same, in level-0 voxels.
    physical_scale : list[float]
        Level-0 data-to-world scale per axis, i.e. the OME global scale
        composed with the level-0 dataset scale.  ``level_scales`` is
        normalised to level-0 voxels and so has this divided out; it is kept
        here for display.  Empty when not known.
    physical_translation : list[float]
        Level-0 data-to-world translation per axis, the companion to
        ``physical_scale``.  Empty when not known.
    name : str
        Human-readable name for the store.
    id : UUID4
        Unique identifier.  Taken from the ``datastore_id`` of
        ``data_coordinate_systems[0]`` when not given; otherwise generated.
    data_coordinate_systems : list[DataCoordinateSystem]
        One system per resolution level, finest first, and the store's only
        record of its axis names, types and units.  :meth:`from_path` builds
        them from the NGFF axis metadata (every axis ``sampling="discrete"``)
        or from a caller's level-0 system.  Each system's ``datastore_id``
        must equal ``id``.
    level_transforms : list[AffineTransform]
        Level ``k`` voxels -> level ``0`` voxels, one per system, built from
        ``level_scales`` and ``level_translations``.  Not normally passed.
    """

    store_type: Literal["ome_zarr_label"] = "ome_zarr_label"
    DATASET_INFO_LABEL: ClassVar[str] = "OME-Zarr labels"
    zarr_path: str
    multiscale_index: int = 0
    scale_names: list[str]
    physical_scale: list[float] = Field(default_factory=list)
    physical_translation: list[float] = Field(default_factory=list)
    anonymous: bool = False
    name: str = "ome zarr label data store"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _ts_stores: list[ts.TensorStore] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Open all TensorStore handles (synchronous, before QtAsyncio)."""
        from cellier.data.image._ome_zarr_image_store import _open_ome_ts_stores

        self._ts_stores = _open_ome_ts_stores(
            self.zarr_path,
            self.scale_names,
            anonymous=self.anonymous,
            cache_pool_bytes=self.cache_pool_bytes,
        )
        # After the handles: the base checks the systems against the level
        # count and rank, which are read off them.
        super().model_post_init(__context)

    def _reopen_ts_stores(self) -> None:
        """Reopen every level against the store's current cache budget."""
        from cellier.data.image._ome_zarr_image_store import _open_ome_ts_stores

        self._ts_stores = _open_ome_ts_stores(
            self.zarr_path,
            self.scale_names,
            anonymous=self.anonymous,
            cache_pool_bytes=self.cache_pool_bytes,
        )

    # ── Convenience constructors ────────────────────────────────────────

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        anonymous: bool = False,
        cache_pool_bytes: int = DEFAULT_CACHE_POOL_BYTES,
        data_coordinate_system: DataCoordinateSystem | None = None,
        name: str = "ome zarr label data store",
    ) -> OMEZarrLabelDataStore:
        """Construct from a URI pointing directly at an OME-NGFF label group.

        The URI must point at a zarr group that carries ``ome.multiscales``
        metadata (i.e. the label sub-group itself, not the root OME-Zarr).

        Parameters
        ----------
        zarr_path : str
            URI with a scheme prefix: ``file://``, ``s3://``, ``gs://``,
            or ``https://``.  For local files use an absolute path, e.g.
            ``file:///home/user/data/seg.ome.zarr/labels/cells``.
        multiscale_index : int
            Which ``multiscales[]`` entry to use. Defaults to 0.
        anonymous : bool
            When True, use anonymous credentials for S3/GCS access.
        cache_pool_bytes : int
            Chunk cache cap for this store, in bytes, shared by all of its
            resolution levels.  ``0`` disables caching.
        data_coordinate_system : DataCoordinateSystem or None
            The level-0 coordinate system, one axis per array dimension.
            ``None`` builds it from the NGFF axis metadata, and an axis with
            an empty type raises.  A passed system replaces that metadata
            outright; the coarser levels copy its axes with fresh ids, and
            the store adopts its ``datastore_id``.
        name : str
            Human-readable name for the store.
        """
        import yaozarrs

        _validate_uri_scheme(zarr_path)

        group = yaozarrs.open_group(zarr_path)
        raw_attrs = group.attrs

        # OME-NGFF label groups store metadata under "ome" → "multiscales"
        # (as written by the demo writer) or at the top-level "multiscales".
        multiscales_list = cls._read_multiscales(raw_attrs, zarr_path)
        ms_raw = multiscales_list[multiscale_index]
        ms = cls._parse_multiscale(ms_raw)

        axes = ms["axes"]
        n_axes = len(axes)
        global_scale = [1.0] * n_axes
        global_translation = [0.0] * n_axes
        for ct in ms.get("coordinateTransformations") or []:
            if ct.get("type") == "scale":
                global_scale = list(ct["scale"])
            elif ct.get("type") == "translation":
                global_translation = list(ct["translation"])

        # Labels use raw dicts rather than yaozarrs typed objects, so the
        # per-dataset numbers are pulled out here; the composition itself is
        # shared with the image reader.
        datasets = ms["datasets"]
        dataset_scales: list[list[float]] = []
        dataset_translations: list[list[float]] = []
        for ds in datasets:
            ds_scale = [1.0] * n_axes
            ds_trans = [0.0] * n_axes
            for ct in ds.get("coordinateTransformations") or []:
                if ct.get("type") == "scale":
                    ds_scale = list(ct["scale"])
                elif ct.get("type") == "translation":
                    ds_trans = list(ct["translation"])
            dataset_scales.append(ds_scale)
            dataset_translations.append(ds_trans)

        level_scales, level_translations, physical_scale, physical_translation = (
            _level_geometry(
                global_scale, global_translation, dataset_scales, dataset_translations
            )
        )

        return cls(
            zarr_path=zarr_path,
            multiscale_index=multiscale_index,
            scale_names=[ds["path"] for ds in datasets],
            level_scales=level_scales,
            level_translations=level_translations,
            data_coordinate_systems=_ngff_coordinate_systems(
                [ax["name"] for ax in axes],
                [ax.get("type") or "" for ax in axes],
                [ax.get("unit") for ax in axes],
                len(datasets),
                name,
                data_coordinate_system,
            ),
            physical_scale=physical_scale,
            physical_translation=physical_translation,
            anonymous=anonymous,
            cache_pool_bytes=cache_pool_bytes,
            name=name,
        )

    @staticmethod
    def _read_multiscales(attrs: dict, zarr_path: str) -> list:
        """Extract multiscales list from raw zarr attributes dict."""
        # OME-NGFF label groups may store metadata under "ome.multiscales"
        if "ome" in attrs and isinstance(attrs["ome"], dict):
            ome_block = attrs["ome"]
            if "multiscales" in ome_block:
                return ome_block["multiscales"]
        # Or at the top-level "multiscales" (older convention)
        if "multiscales" in attrs:
            return attrs["multiscales"]
        raise ValueError(
            f"No multiscales metadata found at {zarr_path!r}. "
            f"Expected 'ome.multiscales' or 'multiscales' in zarr attributes."
        )

    @staticmethod
    def _parse_multiscale(ms_raw: dict) -> dict:
        """Normalize a multiscale metadata dict."""
        return ms_raw

    # ── Read-only properties ────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of scale levels."""
        return len(self._ts_stores)

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """Full-rank shape per level (all axes), finest first."""
        return [tuple(int(d) for d in store.domain.shape) for store in self._ts_stores]

    @property
    def axis_extents(self) -> tuple[tuple[float, float], ...]:
        """Per-axis ``(low, high)`` extents in level-0 data coordinates.

        The edge convention: an axis of ``size`` voxels spans
        ``[-0.5, size - 0.5]``.  See
        :attr:`~cellier.data._base_data_store.BaseDataStore.axis_extents`.
        """
        return gridded_axis_extents(self.level_shapes[0])

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying arrays (must be int8/int16/int32)."""
        native = self._ts_stores[0].dtype.numpy_dtype
        if native.type not in _ACCEPTED_LABEL_DTYPES:
            raise ValueError(
                f"OMEZarrLabelDataStore: unsupported dtype {native}. "
                f"Expected int8, int16, or int32."
            )
        return native

    @property
    def ndim(self) -> int:
        """Number of data dimensions, read off the level-0 handle."""
        return len(self._ts_stores[0].domain.shape)

    # ── Self-description ────────────────────────────────────────────────

    def dataset_info(self) -> DatasetInfo:
        """Describe the store from metadata it already holds.

        Like the image store, this never re-opens the group and never reads
        array data -- so no label count, which would require a full pass
        over level 0.  ``LabelMemoryStore`` reports one because its data is
        already in RAM.

        The ``Data type`` row names the on-disk dtype and the int32 the
        store hands out, which differ whenever the source is int8 or int16.
        """
        native = self.dtype
        dtype_text = str(native) if native == np.int32 else f"{native} (read as int32)"
        return ome_zarr_dataset_info(self, dtype_text)

    # ── Async data access ───────────────────────────────────────────────

    async def get_data(self, request) -> np.ndarray:
        """Read a padded brick, returning int32 (zero-padded for out-of-bounds).

        Parameters
        ----------
        request : ChunkRequest
            Padded brick specification with ``axis_selections`` and
            ``scale_index``.

        Returns
        -------
        np.ndarray
            int32 array.
        """
        store = self._ts_stores[request.scale_index]
        store_shape = tuple(int(d) for d in store.domain.shape)

        out_shape = tuple(
            stop - start
            for sel in request.axis_selections
            if isinstance(sel, tuple)
            for start, stop in [sel]
        )
        out = np.zeros(out_shape, dtype=np.int32)

        store_idx: list[int | slice] = []
        dest_starts: list[int] = []
        valid = True

        for axis_i, sel in enumerate(request.axis_selections):
            size = store_shape[axis_i]
            if isinstance(sel, int):
                store_idx.append(max(0, min(sel, size - 1)))
            else:
                start, stop = sel
                c_start = max(start, 0)
                c_stop = min(stop, size)
                if c_stop <= c_start:
                    valid = False
                    break
                store_idx.append(slice(c_start, c_stop))
                dest_starts.append(c_start - start)

        if valid:
            region = np.asarray(
                await store[tuple(store_idx)].read(),
                dtype=np.int32,
            )
            dest_idx = tuple(slice(d, d + s) for d, s in zip(dest_starts, region.shape))
            out[dest_idx] = region

        return out
