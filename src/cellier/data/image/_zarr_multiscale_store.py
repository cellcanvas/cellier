"""MultiscaleZarrDataStore — async-capable data store for multiscale zarr volumes.

Inherits from ``BaseDataStore`` (psygnal ``EventedModel`` / pydantic ``BaseModel``).
Opens all tensorstore handles synchronously in ``model_post_init`` so they are
ready before ``QtAsyncio.run()`` starts the event loop.

Placement note
--------------
Place this file adjacent to (or in the same package as) ``chunk_request.py``
and adjust the import below to match::

    from <your_package>.chunk_request import ChunkRequest

Likewise adjust the ``BaseDataStore`` import to wherever you have placed the
base class definition.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import tensorstore as ts
from pydantic import ConfigDict, PrivateAttr, model_validator

from cellier.data._axes import level_systems
from cellier.data._base_data_store import BaseDataStore, gridded_axis_extents
from cellier.data._dataset_info import (
    DatasetInfo,
    RowSection,
    format_scale,
    format_shape,
    source_label,
)
from cellier.data._tensorstore_cache import (
    DEFAULT_CACHE_POOL_BYTES,
    TensorStoreCacheMixin,
    build_context,
)

if TYPE_CHECKING:
    from cellier.data.image._image_requests import ChunkRequest
    from cellier.transform import DataCoordinateSystem

# ---------------------------------------------------------------------------
# Zarr format detection helpers (private)
# ---------------------------------------------------------------------------


def _detect_zarr_driver(level_path: pathlib.Path) -> str:
    """Return the tensorstore driver string for a zarr level directory.

    Detects format by the metadata sentinel file:

    * ``.zarray``   → zarr v2 → driver ``"zarr"``
    * ``zarr.json`` → zarr v3 → driver ``"zarr3"``

    Raises ``FileNotFoundError`` if neither file is present.
    """
    p = pathlib.Path(level_path)
    if (p / ".zarray").exists():
        return "zarr"
    if (p / "zarr.json").exists():
        return "zarr3"
    raise FileNotFoundError(
        f"Cannot determine zarr format for '{p}': "
        f"neither '.zarray' (zarr v2) nor 'zarr.json' (zarr v3) found.\n"
        f"Re-run with --make-files to regenerate the store."
    )


def _open_ts_stores(
    zarr_path: pathlib.Path,
    scale_names: list[str],
    cache_pool_bytes: int = DEFAULT_CACHE_POOL_BYTES,
) -> list[ts.TensorStore]:
    """Open one tensorstore per scale level (read-only, synchronous).

    Must be called — or triggered via ``model_post_init`` — **before**
    ``QtAsyncio.run()`` starts the event loop.

    Parameters
    ----------
    zarr_path :
        Root directory of the multiscale zarr store.
    scale_names :
        Subdirectory names in order finest → coarsest, e.g.
        ``["s0", "s1", "s2"]``.
    cache_pool_bytes :
        Chunk cache cap in bytes, shared by every level opened here --
        one context serves them all, so a chunk read for one level is not
        re-decompressed for the next.  ``0`` disables caching.

    Returns
    -------
    stores :
        One open ``ts.TensorStore`` per scale level.  Chunk data is
        not loaded until ``await store[...].read()`` is called.
    """
    context = build_context(cache_pool_bytes)
    stores: list[ts.TensorStore] = []
    for name in scale_names:
        level_path = pathlib.Path(zarr_path) / name
        driver = _detect_zarr_driver(level_path)
        spec: dict[str, Any] = {
            "driver": driver,
            "kvstore": {
                "driver": "file",
                "path": str(level_path),
            },
        }
        store = ts.open(spec, context=context).result()
        stores.append(store)
    return stores


# ---------------------------------------------------------------------------
# MultiscaleZarrDataStore
# ---------------------------------------------------------------------------


class MultiscaleZarrDataStore(TensorStoreCacheMixin, BaseDataStore):
    """Data store for a multiscale zarr volume read via tensorstore.

    Public fields are validated and serialisable (pydantic).
    Tensorstore handles are opened synchronously in ``model_post_init``
    and stored as private attributes so they are not serialised.

    Parameters
    ----------
    store_type : Literal["multiscale_zarr"]
        Discriminator field. Always ``"multiscale_zarr"``.
    zarr_path :
        Path to the root directory of the multiscale zarr store.
        Pass as a string; ``pathlib.Path`` is accepted and coerced.
    scale_names :
        Ordered list of subdirectory names, finest → coarsest,
        e.g. ``["s0", "s1", "s2"]``.
    level_scales :
        Per-level, per-axis scale of level-k voxels in level-0 voxels.
        ``level_scales[0]`` must be all ones.  Length must match
        ``scale_names``.
    level_translations :
        The offset half of the same, in level-0 voxels.
    id :
        Unique identifier.  Taken from the ``datastore_id`` of
        ``data_coordinate_systems[0]`` when not given; otherwise generated.
    data_coordinate_systems :
        One system per resolution level, finest first.  The store reads no
        axis metadata, so these come from the caller --
        :meth:`from_scale_and_translation` builds them from a level-0
        system -- or, when left empty, from the scene's world axes once the
        store is added to a scene.  Each system's ``datastore_id`` must equal
        ``id``.
    level_transforms :
        Level ``k`` voxels -> level ``0`` voxels, one per system, built from
        ``level_scales`` and ``level_translations`` once the systems exist.
        Not normally passed.
    name :
        Human-readable name for the store (inherited from
        ``BaseDataStore``; defaults to ``"multiscale zarr data store"``).

    Attributes (read-only properties)
    ------------------------------------
    n_levels :
        Number of scale levels (length of ``scale_names``).
    level_shapes :
        List of shape tuples, one per level.
    """

    # ── Public pydantic fields ──────────────────────────────────────────
    store_type: Literal["multiscale_zarr"] = "multiscale_zarr"
    DATASET_INFO_LABEL: ClassVar[str] = "multiscale zarr"
    zarr_path: str
    scale_names: list[str]
    name: str = "multiscale zarr data store"

    # ── Private tensorstore handles (not serialised) ────────────────────
    _ts_stores: list[ts.TensorStore] = PrivateAttr(default_factory=list)

    # Allow non-pydantic types in private attrs.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Validation ─────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_level_geometry(self) -> MultiscaleZarrDataStore:
        """Check that the per-level geometry matches ``scale_names``."""
        if len(self.level_scales) != len(self.scale_names):
            raise ValueError(
                f"level_scales has {len(self.level_scales)} entries "
                f"but scale_names has {len(self.scale_names)} entries; "
                f"they must match."
            )
        if len(self.level_translations) != len(self.scale_names):
            raise ValueError(
                f"level_translations has {len(self.level_translations)} "
                f"entries but scale_names has {len(self.scale_names)} "
                f"entries; they must match."
            )
        return self

    # ── Lifecycle ───────────────────────────────────────────────────────

    def model_post_init(self, __context: Any) -> None:
        """Open all tensorstore handles.

        Called automatically by pydantic after ``__init__``.
        Must run before ``QtAsyncio.run()`` starts the event loop.
        """
        self._ts_stores = _open_ts_stores(
            pathlib.Path(self.zarr_path),
            self.scale_names,
            self.cache_pool_bytes,
        )
        # After the handles: the base checks the systems against the level
        # count and rank, which are read off them.
        super().model_post_init(__context)

    def _reopen_ts_stores(self) -> None:
        """Reopen every level against the store's current cache budget."""
        self._ts_stores = _open_ts_stores(
            pathlib.Path(self.zarr_path),
            self.scale_names,
            self.cache_pool_bytes,
        )

    # ── Convenience constructor ─────────────────────────────────────────

    @classmethod
    def from_scale_and_translation(
        cls,
        *,
        zarr_path: str,
        scale_names: list[str],
        level_scales: list[tuple[float, ...]],
        level_translations: list[tuple[float, ...]],
        data_coordinate_system: DataCoordinateSystem | None = None,
        name: str = "multiscale zarr data store",
    ) -> MultiscaleZarrDataStore:
        """Construct from per-level scale and translation vectors.

        Parameters
        ----------
        zarr_path :
            Path to the root directory of the multiscale zarr store.
        scale_names :
            Ordered list of subdirectory names, finest → coarsest.
        level_scales :
            Per-level scale vectors. ``level_scales[0]`` should be all 1s.
        level_translations :
            Per-level translation vectors. ``level_translations[0]``
            should be all 0s.
        data_coordinate_system :
            The level-0 coordinate system, one axis per data dimension.  The
            coarser levels copy its axes (names, types, units and sampling)
            with fresh ids, and the store adopts its ``datastore_id`` as its
            ``id``.  ``None`` leaves the store without systems until it is
            added to a scene.
        name :
            Human-readable name for the store.
        """
        if len(level_scales) != len(level_translations):
            raise ValueError(
                f"level_scales has {len(level_scales)} entries but "
                f"level_translations has {len(level_translations)} entries; "
                f"they must match."
            )
        return cls(
            zarr_path=zarr_path,
            scale_names=scale_names,
            level_scales=[tuple(float(v) for v in sc) for sc in level_scales],
            level_translations=[
                tuple(float(v) for v in tr) for tr in level_translations
            ],
            data_coordinate_systems=(
                []
                if data_coordinate_system is None
                else level_systems(data_coordinate_system, len(scale_names), name)
            ),
            name=name,
        )

    # ── Read-only properties ────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of scale levels."""
        return len(self._ts_stores)

    @property
    def ndim(self) -> int:
        """Number of data dimensions, read off the level-0 handle.

        This store reads no axis metadata -- unlike the OME-Zarr readers it
        is constructed from bare scale and translation vectors.  Its axes
        come from a caller's ``data_coordinate_system`` or from the scene it
        is added to, and this rank is what either must match.
        """
        return len(self._ts_stores[0].domain.shape)

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """Shape for each scale level, finest first."""
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
        """Data type of the underlying arrays.

        Read off the level-0 tensorstore handle, which is already open, so
        this costs nothing.  ``get_data`` converts to float32 on the way
        out; this reports what is actually on disk.
        """
        return self._ts_stores[0].dtype.numpy_dtype

    # ── Self-description ────────────────────────────────────────────────

    def dataset_info(self) -> DatasetInfo:
        """Describe the pyramid: path, dtype, and the per-level geometry.

        No value range: unlike the in-memory stores, computing one here
        would mean reading every level-0 chunk off disk or over the network.

        The per-level scale rows are derived from ``level_scales``
        rather than asserted -- the examples used to hardcode strings like
        ``"2x isotropic"`` that no longer matched an anisotropic pyramid.
        """
        shapes = self.level_shapes
        level_rows: list[tuple[str, str]] = []
        for index, (level_name, shape) in enumerate(zip(self.scale_names, shapes)):
            scale = np.asarray(self.level_scales[index], dtype=float)
            level_rows.append(
                (
                    level_name,
                    f"{format_shape(shape)}  ({format_scale(scale)})",
                )
            )

        return DatasetInfo(
            sections=[
                RowSection(
                    None,
                    [
                        *self._identity_rows(),
                        ("Path", self.zarr_path),
                        ("Source", source_label(self.zarr_path)),
                        ("Data type", str(self.dtype)),
                        ("Scale levels", str(self.n_levels)),
                    ],
                ),
                RowSection("Scale levels", level_rows, collapsed=True),
            ]
        )

    # ── Async data access ───────────────────────────────────────────────

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Read a single padded brick, returning a zero-padded float32 array.

        Interprets ``request.axis_selections`` generically: displayed axes
        (tuple ranges) become slice dimensions in the output; sliced axes
        (int values) become point selections.

        Parameters
        ----------
        request :
            Padded brick specification.  Coordinates may be negative or
            exceed store bounds; clamping is handled internally.

        Returns
        -------
        out :
            ``float32`` array.  Shape has one dimension per displayed axis
            (those with tuple selections).  Out-of-bounds regions are
            filled with zero.
        """
        return await self._get_data_and(request)

    async def _get_data_and(self, request: ChunkRequest) -> np.ndarray:
        """Read a padded brick using generic nD axis_selections."""
        store = self._ts_stores[request.scale_index]
        store_shape = tuple(int(d) for d in store.domain.shape)

        # Output shape: one dimension per displayed (tuple) axis.
        out_shape = tuple(
            stop - start
            for sel in request.axis_selections
            if isinstance(sel, tuple)
            for start, stop in [sel]
        )
        out = np.zeros(out_shape, dtype=np.float32)

        # Build the clamped store index and track destination offsets.
        store_idx: list[int | slice] = []
        dest_starts: list[int] = []
        valid = True

        for axis_i, sel in enumerate(request.axis_selections):
            size = store_shape[axis_i]
            if isinstance(sel, int):
                # Point selection: clamp to valid range.
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
            # Compute the destination slice in out for each displayed axis.
            dest_idx = tuple(slice(d, d + s) for d, s in zip(dest_starts, region.shape))
            out[dest_idx] = region

        return out
