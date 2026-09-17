"""Chunk caching for the tensorstore-backed data stores.

Why this exists
---------------
A zarr chunk is the smallest unit tensorstore can read: any voxel inside it
costs a full read and decompress of the whole chunk.  The render layer asks
for bricks (``block_size`` voxels a side), which are usually far smaller than
a chunk, so many bricks land in the same chunk.

Tensorstore already shares one decompress between reads that are *in flight
at the same time*, so a batch of concurrent brick reads that hit one chunk
pays for it once.  But by default the decompressed chunk is dropped as soon
as the last reader lets go -- ``cache_pool.total_bytes_limit`` defaults to 0
-- so the next batch pays for it again.  The dedup window is exactly one
batch.

A cache pool widens that window to "while it still fits".  On a store whose
chunks each hold a whole 3-D volume, that is the difference between
decompressing a chunk once per batch and once per timepoint.

Each store owns its pool, sized by its ``cache_pool_bytes`` field.  The
limit is a cap, not an allocation: an unused pool costs nothing and fills
lazily under LRU.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import tensorstore as ts
from pydantic import BaseModel, PrivateAttr, field_validator

if TYPE_CHECKING:
    from collections.abc import Callable

#: Default per-store chunk cache budget, in bytes.
#:
#: Sized to hold a working set of a few timepoints for a typical multiscale
#: volume while staying a modest fraction of system memory.  ``0`` disables
#: caching, restoring the one-batch dedup window described above.
DEFAULT_CACHE_POOL_BYTES: int = 512 * 1024**2


def build_context(cache_pool_bytes: int) -> ts.Context:
    """Return a ``ts.Context`` whose chunk cache is capped at *cache_pool_bytes*.

    Pass the result to every ``ts.open`` of a single store, so all of its
    resolution levels share one budget.

    Parameters
    ----------
    cache_pool_bytes : int
        Cache cap in bytes.  ``0`` disables caching.

    Returns
    -------
    ts.Context
        A fresh context.  Contexts are immutable, so changing the budget
        means building a new one and reopening the handles against it --
        which is cheap, as tensorstore caches the array metadata too.
    """
    return ts.Context({"cache_pool": {"total_bytes_limit": int(cache_pool_bytes)}})


def cache_metrics() -> tuple[int, int]:
    """Return the process-wide ``(hits, misses)`` chunk cache counters.

    Both counts are cumulative for the life of the process and cover every
    open tensorstore, so a caller measuring one operation takes a delta
    around it rather than reading absolute values.

    Returns
    -------
    tuple[int, int]
        ``(hit_count, miss_count)``.  ``(0, 0)`` when the counters are
        unavailable -- they come from a ``tensorstore.experimental_*`` API,
        so this degrades rather than raising if it is renamed.
    """
    try:
        collected = ts.experimental_collect_matching_metrics("/tensorstore/cache")
    except Exception:  # pragma: no cover - depends on the tensorstore build
        return (0, 0)

    def _value(name: str) -> int:
        for entry in collected:
            if entry.get("name") != name:
                continue
            for value in entry.get("values", ()):
                if "value" in value:
                    return int(value["value"])
        return 0

    return (
        _value("/tensorstore/cache/hit_count"),
        _value("/tensorstore/cache/miss_count"),
    )


class TensorStoreCacheMixin(BaseModel):
    """Adds a per-store tensorstore chunk cache to a data store.

    Mixed in ahead of ``BaseDataStore``.  Supplies the ``cache_pool_bytes``
    field and makes plain assignment to it rebuild the store's cache pool::

        store.cache_pool_bytes = 2 * 1024**3

    The subclass supplies :meth:`_reopen_ts_stores`, which reopens its
    handles against the store's current budget.

    Parameters
    ----------
    cache_pool_bytes : int
        Chunk cache cap for this store, in bytes.  Shared by all of its
        resolution levels.  ``0`` disables caching.  Defaults to
        :data:`DEFAULT_CACHE_POOL_BYTES`.
    """

    cache_pool_bytes: int = DEFAULT_CACHE_POOL_BYTES

    #: Set by a paint controller while it holds this store open for writing.
    #: A weakref, so a discarded controller cannot keep the store locked.
    _paint_writer_ref: Callable[[], Any] | None = PrivateAttr(default=None)

    @field_validator("cache_pool_bytes")
    @classmethod
    def _check_cache_pool_bytes(cls, value: int) -> int:
        """Reject a negative budget at construction time."""
        if value < 0:
            raise ValueError(f"cache_pool_bytes must be >= 0, got {value}.")
        return value

    def model_post_init(self, __context: Any) -> None:
        """Hand off to the next ``model_post_init`` in the MRO.

        Declaring a ``PrivateAttr`` makes pydantic inject
        ``init_private_attributes`` as this class's ``model_post_init``, and
        that injected function does not call ``super()``.  Because the mixin
        sits ahead of ``BaseDataStore``, the injected version would end the
        chain here and the base would never install its level transforms or
        check its coordinate systems.  Defining the hook explicitly keeps the
        chain intact -- pydantic wraps it so the private attributes are still
        initialised first.
        """
        super().model_post_init(__context)

    # ── Paint interlock ─────────────────────────────────────────────────

    def register_paint_writer(self, writer: Any) -> None:
        """Record the paint write buffer currently open on this store.

        Held weakly: a controller that is dropped without tearing down does
        not leave the store permanently locked.

        Parameters
        ----------
        writer : Any
            An object with a ``transaction`` property that is ``None``
            whenever no transaction is open -- i.e. a
            ``TensorStoreWriteBuffer``.
        """
        self._paint_writer_ref = weakref.ref(writer)

    def unregister_paint_writer(self) -> None:
        """Forget the paint write buffer, undoing :meth:`register_paint_writer`."""
        self._paint_writer_ref = None

    def _has_open_paint_transaction(self) -> bool:
        """Whether a paint transaction is currently open on this store."""
        if self._paint_writer_ref is None:
            return False
        writer = self._paint_writer_ref()
        if writer is None:
            return False
        return getattr(writer, "transaction", None) is not None

    # ── Reopening ───────────────────────────────────────────────────────

    def _reopen_ts_stores(self) -> None:
        """Reopen this store's handles against ``cache_pool_bytes``.

        Implemented by the subclass, which knows its own open arguments.
        """
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any) -> None:
        """Set a field, rebuilding the cache pool when the budget changes.

        Only ``cache_pool_bytes`` is treated specially; every other field is
        set as usual.  Assigning the value it already has is a no-op, so a
        redundant write does not throw away a warm cache.

        Raises
        ------
        ValueError
            If *value* is negative.
        RuntimeError
            If a paint transaction is open on this store.  Reopening would
            leave the paint write buffer holding a handle bound to the
            discarded pool, so the change is refused until the stroke is
            committed or aborted.
        """
        if key != "cache_pool_bytes":
            super().__setattr__(key, value)
            return

        value = self._check_cache_pool_bytes(int(value))
        if value == self.__dict__.get("cache_pool_bytes"):
            # Nothing to do, and reopening would discard a warm cache.
            return
        if self._has_open_paint_transaction():
            raise RuntimeError(
                "Cannot change cache_pool_bytes while a paint transaction is "
                "open on this store: the paint write buffer holds a handle "
                "bound to the current cache pool.  Commit or abort the stroke "
                "first."
            )

        super().__setattr__(key, value)
        self._reopen_ts_stores()
