"""OMEZarrImageDataStore — data store for OME-Zarr v0.5 images.

Reads validated OME metadata via ``yaozarrs`` and opens per-level
tensorstore handles for async data access.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import tensorstore as ts
from pydantic import ConfigDict, Field, PrivateAttr

from cellier.data._axes import build_axes, level_systems
from cellier.data._axes import data_coordinate_system as build_data_coordinate_system
from cellier.data._base_data_store import BaseDataStore, gridded_axis_extents
from cellier.data._dataset_info import DatasetInfo, ome_zarr_dataset_info
from cellier.data._tensorstore_cache import (
    DEFAULT_CACHE_POOL_BYTES,
    TensorStoreCacheMixin,
    build_context,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from yaozarrs import v05

    from cellier.data.image._image_requests import ChunkRequest
    from cellier.transform import DataCoordinateSystem

# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

_SUPPORTED_SCHEMES = frozenset({"file", "s3", "gs", "gcs", "https", "http"})


def _is_windows_drive_prefix(text: str) -> bool:
    return len(text) >= 2 and text[1] == ":" and text[0].isalpha()


def _file_uri_to_local_path(uri: str) -> pathlib.Path:
    r"""Convert a ``file://`` URI to a local filesystem path.

    Accepts canonical URIs (``file:///C:/...``) and common _legacy variants
    seen in tests on Windows (e.g. ``file://C:\\...``).
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Expected file:// URI, got {uri!r}")

    netloc = parsed.netloc.replace("\\", "/")
    path = parsed.path.replace("\\", "/")

    if len(netloc) == 0 and len(path) == 0:
        raise ValueError(f"URI couldn't be parsed: {parsed}")

    # Canonical Windows file URI: file:///C:/... (drive in path with leading /)
    if (
        len(netloc) == 0
        and len(path) >= 3
        and path[0] == "/"
        and _is_windows_drive_prefix(path[1:])
    ):
        return pathlib.Path(path[1:])

    if len(netloc) > 0:
        # Handle Windows drive forms where the drive/path may appear in netloc.
        if _is_windows_drive_prefix(netloc):
            return pathlib.Path(netloc.rstrip("/") + path)
        # UNC path form: file://server/share/path
        return pathlib.Path(f"//{netloc}{path}")

    return pathlib.Path(path)


def _join_uri_path(uri: str, child_path: str) -> str:
    """Join a relative child path onto a URI root."""
    child = child_path.replace("\\", "/").lstrip("/")
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return (_file_uri_to_local_path(uri) / pathlib.PurePosixPath(child)).as_uri()
    return f"{uri.rstrip('/')}/{child}"


def _validate_uri_scheme(uri: str) -> None:
    """Raise ``ValueError`` if *uri* does not start with a supported scheme."""
    parsed = urlparse(uri)
    if parsed.scheme not in _SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported URI scheme {parsed.scheme!r} in {uri!r}. "
            f"Supported schemes: {', '.join(sorted(_SUPPORTED_SCHEMES))}."
        )


def _build_kvstore_spec(uri: str, array_path: str) -> dict:
    """Build a TensorStore kvstore spec from root URI and relative array path.

    Supports ``file://``, ``s3://``, ``gs://`` / ``gcs://``,
    ``http://`` and ``https://``.
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme

    if scheme == "file":
        root = _file_uri_to_local_path(uri)

        return {
            "driver": "file",
            "path": str(root / pathlib.PurePosixPath(array_path)),
        }
    elif scheme in ("s3",):
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        full = f"{prefix}/{array_path}" if prefix else array_path
        return {"driver": "s3", "bucket": bucket, "path": full}
    elif scheme in ("gs", "gcs"):
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        full = f"{prefix}/{array_path}" if prefix else array_path
        return {"driver": "gcs", "bucket": bucket, "path": full}
    elif scheme in ("http", "https"):
        base = uri.rstrip("/")
        return {"driver": "http", "base_url": f"{base}/{array_path}"}
    else:
        raise ValueError(f"Unsupported URI scheme: {scheme!r}")


# ---------------------------------------------------------------------------
# TensorStore opening
# ---------------------------------------------------------------------------


def _detect_zarr_driver(uri: str, array_path: str) -> str:
    """Detect zarr format (v2 or v3) for a level inside a URI.

    For ``file://`` URIs, checks sentinel files on disk.
    For remote URIs, defaults to ``zarr3``.
    """
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        level_path = _file_uri_to_local_path(uri) / pathlib.PurePosixPath(array_path)

        if (level_path / ".zarray").exists():
            return "zarr"
        if (level_path / "zarr.json").exists():
            return "zarr3"
        raise FileNotFoundError(
            f"Cannot determine zarr format for '{level_path}': "
            f"{array_path} is not a zarr file."
            f"{parsed} didn't work"
            f"neither '.zarray' (zarr v2) nor 'zarr.json' (zarr v3) found."
        )
    # Remote: default to zarr3 (OME-Zarr v0.5 implies zarr v3).
    return "zarr3"


def _open_ome_ts_stores(
    zarr_path: str,
    scale_names: list[str],
    anonymous: bool = False,
    cache_pool_bytes: int = DEFAULT_CACHE_POOL_BYTES,
) -> list[ts.TensorStore]:
    """Open one TensorStore per scale level (synchronous, read-only).

    Must be called before ``QtAsyncio.run()`` starts the event loop.

    Parameters
    ----------
    zarr_path : str
        Root URI of the OME-Zarr store.
    scale_names : list[str]
        Per-level relative array paths.
    anonymous : bool
        When True, use anonymous credentials for S3/GCS access
        (for public buckets). Default False.
    cache_pool_bytes : int
        Chunk cache cap in bytes, shared by every level opened here --
        one context serves them all, so a chunk read for one level is not
        re-decompressed for the next.  ``0`` disables caching.
    """
    context = build_context(cache_pool_bytes)
    stores: list[ts.TensorStore] = []
    scheme = urlparse(zarr_path).scheme
    for name in scale_names:
        driver = _detect_zarr_driver(zarr_path, name)
        spec: dict[str, Any] = {
            "driver": driver,
            "kvstore": _build_kvstore_spec(zarr_path, name),
        }
        # Use anonymous credentials for public cloud buckets.
        if anonymous and scheme in ("s3", "gs", "gcs"):
            if scheme == "s3":
                spec.setdefault("context", {})["aws_credentials"] = {
                    "anonymous": True,
                }
            else:
                spec.setdefault("context", {})["gcs_user_project"] = ""
        store = ts.open(spec, context=context).result()
        stores.append(store)
    return stores


# ---------------------------------------------------------------------------
# OME metadata helpers
# ---------------------------------------------------------------------------


def _extract_global_transform(
    ms: v05.Multiscale,
) -> tuple[list[float], list[float]]:
    """Extract global scale and translation from ``ms.coordinateTransformations``.

    Returns identity values when the field is absent.
    """
    from yaozarrs.v05 import ScaleTransformation, TranslationTransformation

    n = len(ms.axes)
    global_scale = [1.0] * n
    global_translation = [0.0] * n

    if ms.coordinateTransformations is not None:
        for ct in ms.coordinateTransformations:
            if isinstance(ct, ScaleTransformation):
                global_scale = list(ct.scale)
            elif isinstance(ct, TranslationTransformation):
                global_translation = list(ct.translation)

    return global_scale, global_translation


def _level_geometry(
    global_scale: Sequence[float],
    global_translation: Sequence[float],
    dataset_scales: Sequence[Sequence[float]],
    dataset_translations: Sequence[Sequence[float]],
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]], list[float], list[float]]:
    """Compose NGFF scales and translations into the store's pyramid numbers.

    Implements the math from section 3.2 of the design document over all
    axes.  Shared by both OME-Zarr readers: the image reader pulls the
    per-dataset numbers out of yaozarrs models, the label reader out of raw
    dicts.

    Returns numbers rather than transforms: a transform names the two
    coordinate systems it sits between, and those are minted once the store
    has its axes (see ``install_level_transforms``).

    Parameters
    ----------
    global_scale, global_translation : Sequence[float]
        The multiscale's own ``coordinateTransformations``, per axis.
    dataset_scales, dataset_translations : Sequence[Sequence[float]]
        Each dataset's scale and translation, finest first.

    Returns
    -------
    level_scales : list[tuple[float, ...]]
        Level-k voxels in level-0 voxels; level 0 is all ones.
    level_translations : list[tuple[float, ...]]
        The offset half of the same, in level-0 voxels.
    physical_scale : list[float]
        The level-0 data-to-world scale.  The normalisation above divides it
        out, so it is returned for the store to keep; without it the store
        cannot say where it sits in world space.
    physical_translation : list[float]
        The level-0 data-to-world translation.
    """
    per_level_scale = [
        [g * s for g, s in zip(global_scale, scale)] for scale in dataset_scales
    ]
    per_level_translation = [
        [
            gs * t + gt
            for gs, t, gt in zip(global_scale, translation, global_translation)
        ]
        for translation in dataset_translations
    ]
    s0, t0 = per_level_scale[0], per_level_translation[0]
    n_axes = len(s0)
    level_scales = [
        tuple(scale[i] / s0[i] for i in range(n_axes)) for scale in per_level_scale
    ]
    level_translations = [
        tuple((translation[i] - t0[i]) / s0[i] for i in range(n_axes))
        for translation in per_level_translation
    ]
    return level_scales, level_translations, list(s0), list(t0)


def _ngff_coordinate_systems(
    names: Sequence[str],
    types: Sequence[str],
    units: Sequence[str | None],
    n_levels: int,
    name: str,
    data_coordinate_system: DataCoordinateSystem | None,
) -> list[DataCoordinateSystem]:
    """One coordinate system per pyramid level, for both OME-Zarr readers.

    From the NGFF axes when *data_coordinate_system* is ``None``: names,
    units and types as the file states them.  An empty ``type`` raises
    rather than defaulting to ``"space"`` -- the metadata had a slot for it
    and left it blank, which is a defect in the dataset.  Every axis is
    ``sampling="discrete"``, because the data is a voxel grid.

    A caller's level-0 system replaces the metadata outright, which is also
    the way past a blank ``type``.  Either way the coarser levels are copied
    from level 0 by :func:`~cellier.data._axes.level_systems`.

    Parameters
    ----------
    names, types, units : Sequence
        The NGFF axis metadata, in data order.
    n_levels : int
        How many datasets the multiscale has.
    name : str
        The store's name, the base of each system's name.
    data_coordinate_system : DataCoordinateSystem or None
        The caller's level-0 system, or ``None`` to build one.

    Returns
    -------
    list[DataCoordinateSystem]
        One system per level, finest first.
    """
    if data_coordinate_system is None:
        axes = build_axes(names, types, units, "discrete")
        data_coordinate_system = build_data_coordinate_system(
            uuid4(), axes, f"{name}_level0"
        )
    return level_systems(data_coordinate_system, n_levels, name)


def _omero_channel_labels(metadata: Any) -> list[str] | None:
    """The channel labels an image's ``omero`` block names, or ``None``.

    ``omero`` is transitional NGFF metadata, but it is where OME-Zarr writers
    put channel names.  A channel listed without a label is called by its
    index, so the result has one entry per listed channel.
    """
    omero = getattr(metadata, "omero", None)
    channels = getattr(omero, "channels", None)
    if not channels:
        return None
    return [
        str(channel.label) if getattr(channel, "label", None) else str(index)
        for index, channel in enumerate(channels)
    ]


# ---------------------------------------------------------------------------
# OMEZarrImageDataStore
# ---------------------------------------------------------------------------


class OMEZarrImageDataStore(TensorStoreCacheMixin, BaseDataStore):
    """Data store for an OME-Zarr v0.5 image read via tensorstore.

    Use the :meth:`from_path` class method to construct from an OME-Zarr URI.

    Parameters
    ----------
    store_type : Literal["ome_zarr_image"]
        Discriminator field. Always ``"ome_zarr_image"``.
    zarr_path : str
        URI to the root OME-Zarr group. Must start with ``file://``,
        ``s3://``, ``gs://``, or ``https://``.
    multiscale_index : int
        Index into ``multiscales[]``. Defaults to 0.
    scale_names : list[str]
        Per-level relative array paths, finest to coarsest.
    level_scales : list[tuple[float, ...]]
        Full-rank (all axes) per-level scale: level-k voxels in level-0
        voxels.  Level 0 is all ones by construction.
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
    channel_labels : list[str] or None
        One name per channel, in channel-index order, from the image's
        ``omero`` metadata.  ``None`` when the image has no ``omero`` block.
        ``axis_values_from_viewer`` uses them to label a channel slider.
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

    store_type: Literal["ome_zarr_image"] = "ome_zarr_image"
    DATASET_INFO_LABEL: ClassVar[str] = "OME-Zarr image"
    zarr_path: str
    multiscale_index: int = 0
    scale_names: list[str]
    physical_scale: list[float] = Field(default_factory=list)
    physical_translation: list[float] = Field(default_factory=list)
    channel_labels: list[str] | None = None
    anonymous: bool = False
    name: str = "ome zarr image data store"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _ts_stores: list[ts.TensorStore] = PrivateAttr(default_factory=list)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def model_post_init(self, __context: Any) -> None:
        """Open all TensorStore handles (synchronous, before QtAsyncio)."""
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
        self._ts_stores = _open_ome_ts_stores(
            self.zarr_path,
            self.scale_names,
            anonymous=self.anonymous,
            cache_pool_bytes=self.cache_pool_bytes,
        )

    # ── Convenience constructor ─────────────────────────────────────────

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        series_index: int = 0,
        anonymous: bool = False,
        cache_pool_bytes: int = DEFAULT_CACHE_POOL_BYTES,
        data_coordinate_system: DataCoordinateSystem | None = None,
        name: str = "ome zarr image data store",
    ) -> OMEZarrImageDataStore:
        """Construct from an OME-Zarr v0.5 URI.

        Supports both standard Image stores and Bf2Raw (bioformats2raw)
        multi-series containers.  For Bf2Raw stores the *series_index*
        selects which child image to open.

        Parameters
        ----------
        zarr_path : str
            URI with a scheme prefix: ``file://``, ``s3://``, ``gs://``,
            or ``https://``.  For local files use an absolute path, e.g.
            ``file:///home/user/data/image.ome.zarr``.
        multiscale_index : int
            Which ``multiscales[]`` entry to use. Defaults to 0.
        series_index : int
            For Bf2Raw containers, which image series to open.
            Ignored for standard Image stores. Defaults to 0.
        anonymous : bool
            When True, use anonymous credentials for S3/GCS access
            (for public buckets). Default False.
        cache_pool_bytes : int
            Chunk cache cap for this store, in bytes, shared by all of its
            resolution levels.  ``0`` disables caching.
        data_coordinate_system : DataCoordinateSystem or None
            The level-0 coordinate system, one axis per array dimension.
            ``None`` builds it from the NGFF axis metadata.  A passed system
            replaces that metadata outright; the coarser levels copy its axes
            with fresh ids, and the store adopts its ``datastore_id``.
        name : str
            Human-readable name for the store.

        Raises
        ------
        ValueError
            If the URI scheme is not supported, the series index is out of
            range, an NGFF axis has an empty type and no
            *data_coordinate_system* is passed, or the passed system does not
            have one axis per array dimension.
        TypeError
            If the OME metadata is neither Image nor Bf2Raw (e.g. a Plate).
        """
        import yaozarrs
        from yaozarrs import v05 as ome_v05

        # 1. Validate URI scheme.
        _validate_uri_scheme(zarr_path)

        # 2. Open and validate OME metadata via yaozarrs.
        group = yaozarrs.open_group(zarr_path)
        metadata = group.ome_metadata()

        # 3. Handle Bf2Raw containers: navigate to the child image group.
        if isinstance(metadata, ome_v05.Bf2Raw):
            zarr_path, group, metadata = cls._resolve_bf2raw(
                zarr_path, group, series_index
            )

        # 4. Check it is an Image.
        if not isinstance(metadata, ome_v05.Image):
            type_name = type(metadata).__name__ if metadata is not None else "None"
            raise TypeError(
                f"Expected an OME-Zarr Image at {zarr_path!r}, "
                f"got {type_name}. Plates and other types are not supported."
            )

        # 5. Select multiscale entry.
        ms = metadata.multiscales[multiscale_index]

        # 6. Extract global coordinateTransformations.
        global_scale, global_translation = _extract_global_transform(ms)

        # 7. Derive the per-level geometry (full rank, all axes) and the
        #    level-0 physical transform it normalises away.  The level
        #    transforms themselves are built once the store has its level
        #    coordinate systems, in ``install_level_transforms``.
        n_axes = len(ms.axes)
        level_scales, level_translations, physical_scale, physical_translation = (
            _level_geometry(
                global_scale,
                global_translation,
                [list(ds.scale_transform.scale) for ds in ms.datasets],
                [
                    list(ds.translation_transform.translation)
                    if ds.translation_transform is not None
                    else [0.0] * n_axes
                    for ds in ms.datasets
                ],
            )
        )

        # 8. One coordinate system per level, from the NGFF axes unless the
        #    caller supplied the level-0 system.
        data_coordinate_systems = _ngff_coordinate_systems(
            [ax.name for ax in ms.axes],
            [ax.type or "" for ax in ms.axes],
            [getattr(ax, "unit", None) for ax in ms.axes],
            len(ms.datasets),
            name,
            data_coordinate_system,
        )

        return cls(
            zarr_path=zarr_path,
            multiscale_index=multiscale_index,
            channel_labels=_omero_channel_labels(metadata),
            scale_names=[ds.path for ds in ms.datasets],
            level_scales=level_scales,
            level_translations=level_translations,
            data_coordinate_systems=data_coordinate_systems,
            physical_scale=physical_scale,
            physical_translation=physical_translation,
            anonymous=anonymous,
            cache_pool_bytes=cache_pool_bytes,
            name=name,
        )

    @classmethod
    def _resolve_bf2raw(
        cls,
        zarr_path: str,
        group: Any,
        series_index: int,
    ) -> tuple[str, Any, Any]:
        """Navigate from a Bf2Raw root to the requested child image group.

        Parameters
        ----------
        zarr_path : str
            Root URI of the Bf2Raw container.
        group : yaozarrs.ZarrGroup
            Already-opened root group.
        series_index : int
            Which child image to open.

        Returns
        -------
        resolved_path : str
            URI pointing at the child image group.
        child_group : yaozarrs.ZarrGroup
            Opened child group.
        child_metadata : v05.Image | v05.OMEMetadata
            OME metadata read from the child group.

        Raises
        ------
        ValueError
            If *series_index* is out of range.
        """
        import yaozarrs
        from yaozarrs import v05 as ome_v05

        # Try to read the OME/zarr.json Series list for canonical paths.
        series_paths: list[str] | None = None
        if "OME" in group:
            ome_subgroup = group["OME"]
            ome_meta = ome_subgroup.ome_metadata()
            if isinstance(ome_meta, ome_v05.Series):
                series_paths = ome_meta.series

        if series_paths is not None:
            if series_index < 0 or series_index >= len(series_paths):
                raise ValueError(
                    f"series_index={series_index} out of range: "
                    f"Bf2Raw container has {len(series_paths)} series "
                    f"({series_paths})."
                )
            image_path = series_paths[series_index]
        else:
            # No Series metadata — fall back to numeric path.
            image_path = str(series_index)

        resolved_path = _join_uri_path(zarr_path, image_path)
        child_group = yaozarrs.open_group(resolved_path)
        child_metadata = child_group.ome_metadata()

        return resolved_path, child_group, child_metadata

    # ── Read-only properties ────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of scale levels."""
        return len(self._ts_stores)

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """Full-rank shape per level (all axes), finest first.

        Returns shapes over all axes, including non-spatial ones.
        The controller projects to the displayed subshape using
        ``dims.displayed_axes`` before constructing the render visual.
        """
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
    def ndim(self) -> int:
        """Number of data dimensions, read off the level-0 handle."""
        return len(self._ts_stores[0].domain.shape)

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying arrays."""
        return self._ts_stores[0].dtype.numpy_dtype

    # ── Self-description ────────────────────────────────────────────────

    def dataset_info(self) -> DatasetInfo:
        """Describe the store from metadata it already holds.

        Reads only fields parsed at construction plus the open tensorstore
        handles' shapes and dtype -- it never re-opens the group, which for
        an ``s3://`` store would mean a network round trip every time an
        appearance panel is built.

        No value range, for the same reason: it would be a full read of
        level 0.
        """
        return ome_zarr_dataset_info(self, self.dtype)

    # ── Async data access ───────────────────────────────────────────────

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Read a single padded brick, returning a zero-padded float32 array.

        Interprets ``request.axis_selections`` generically: displayed axes
        (tuple ranges) become slice dimensions in the output; sliced axes
        (int values) become point selections.

        Parameters
        ----------
        request : ChunkRequest
            Padded brick specification.

        Returns
        -------
        np.ndarray
            ``float32`` array.
        """
        store = self._ts_stores[request.scale_index]
        store_shape = tuple(int(d) for d in store.domain.shape)

        out_shape = tuple(
            stop - start
            for sel in request.axis_selections
            if isinstance(sel, tuple)
            for start, stop in [sel]
        )
        out = np.zeros(out_shape, dtype=np.float32)

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
                dtype=np.float32,
            )
            dest_idx = tuple(slice(d, d + s) for d, s in zip(dest_starts, region.shape))
            out[dest_idx] = region

        return out
