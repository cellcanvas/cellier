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

Rechecks
--------
By default tensorstore revalidates a cached chunk against the kvstore on
every read.  Locally that is a stat; remotely it is a round trip per read
(a ``304``), which makes a remote cache hit cost as much as a miss.  The
stores therefore open their handles with ``recheck_cached_data=False`` and
``recheck_cached_metadata=False``: a promise that nothing else writes the
data while it is open.

Two things turn rechecks back on:

- a registered paint writer, for as long as it is registered;
- the store's ``recheck_cached_data`` field, for data another process
  writes.

Toggling rechecks reopens the handles on the store's *existing*
``ts.Context``, so the warm pool survives: the working set is revalidated
rather than refetched.  Only a ``cache_pool_bytes`` change builds a new
context.  Cellier's own paint writes go through the reader's context, so
they are visible either way; the rechecks are for writers outside it.

A store that announces a change (``notify_changed``, or reassigning a data
field) while rechecks are off gets a **one-shot** revalidation instead: the
handles are reopened on the same context with ``recheck_cached_data="open"``
(and the same for metadata), so every chunk cached before the change is
revalidated once when next read -- a ``304`` if it is unchanged, a refetch
if it changed -- and trusted again afterwards
(``plans/progressive_loading_design_v3.md`` 5.14).  This is what makes a
live store that another process appends to show its new data without
turning rechecks on for good.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Literal

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


Recheck = bool | Literal["open"]
"""A handle's recheck policy: always, never, or once for data cached before
the handle was opened (``"open"``)."""


def recheck_spec_options(recheck: Recheck) -> dict[str, Any]:
    """Return the ``ts.open`` spec entries for the given recheck policy.

    Parameters
    ----------
    recheck : bool or "open"
        ``True`` for tensorstore's defaults: revalidate cached data on
        every read and metadata at open.  ``False`` to trust the cache
        indefinitely.  ``"open"`` to revalidate what was cached before the
        handle was opened, once, and trust it afterwards.

    Returns
    -------
    dict[str, Any]
        Entries to merge into a driver spec.  Empty when *recheck* is
        ``True``, so the handle behaves exactly as tensorstore's default.
    """
    if recheck is True:
        return {}
    return {"recheck_cached_data": recheck, "recheck_cached_metadata": recheck}


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

    It also manages the recheck policy described in the module docstring.

    The subclass supplies :meth:`_open_ts_handles`, which opens one handle
    per level against a given context and recheck policy, and calls
    :meth:`_reopen_ts_stores` from its ``model_post_init``.

    Parameters
    ----------
    cache_pool_bytes : int
        Chunk cache cap for this store, in bytes.  Shared by all of its
        resolution levels.  ``0`` disables caching.  Defaults to
        :data:`DEFAULT_CACHE_POOL_BYTES`.
    recheck_cached_data : bool
        Revalidate cached chunks on every read, permanently.  Set it when
        another process writes this data while it is open and cannot say
        when; otherwise a cached chunk is served as it was first read.  A
        writer that can say when calls ``notify_changed`` instead, which
        revalidates once (:meth:`revalidate_cache`).  Defaults to
        ``False``.  Assigning it reopens the handles on the same context,
        keeping the warm pool.
    """

    cache_pool_bytes: int = DEFAULT_CACHE_POOL_BYTES
    recheck_cached_data: bool = False

    #: Set by a paint controller while it holds this store open for writing.
    #: A weakref, so a discarded controller cannot keep the store locked.
    _paint_writer_ref: Callable[[], Any] | None = PrivateAttr(default=None)

    #: One open handle per level, finest first.
    _ts_stores: list[ts.TensorStore] = PrivateAttr(default_factory=list)

    #: The context the current handles were opened on.  Kept so a recheck
    #: toggle can reopen on it and keep the warm pool.
    _ts_context: ts.Context | None = PrivateAttr(default=None)

    #: Whether the current handles revalidate cached data.
    _ts_rechecking: bool = PrivateAttr(default=False)

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

        The first registration reopens the handles with rechecks on, on the
        same context, so a write from outside this store is not masked by
        the cache while painting.  Registering again while registered (a
        fresh buffer after an autosave) only swaps the reference.  A
        transaction already open on the old handle is unaffected: it shares
        the context, so its staged and committed writes are visible through
        the new handles.

        Parameters
        ----------
        writer : Any
            An object with a ``transaction`` property that is ``None``
            whenever no transaction is open -- i.e. a
            ``TensorStoreWriteBuffer``.
        """
        self._paint_writer_ref = weakref.ref(writer)
        self._sync_rechecks()

    def unregister_paint_writer(self) -> None:
        """Forget the paint write buffer, undoing :meth:`register_paint_writer`.

        Reopens the handles with rechecks off again, unless
        ``recheck_cached_data`` keeps them on.
        """
        self._paint_writer_ref = None
        self._sync_rechecks()

    def _has_open_paint_transaction(self) -> bool:
        """Whether a paint transaction is currently open on this store."""
        if self._paint_writer_ref is None:
            return False
        writer = self._paint_writer_ref()
        if writer is None:
            return False
        return getattr(writer, "transaction", None) is not None

    # ── Reopening ───────────────────────────────────────────────────────

    def _open_ts_handles(
        self, context: ts.Context, recheck: Recheck
    ) -> list[ts.TensorStore]:
        """Open one handle per level on *context*.

        Implemented by the subclass, which knows its own open arguments.

        Parameters
        ----------
        context : ts.Context
            The context every level is opened on.
        recheck : bool or "open"
            Whether the handles revalidate cached data
            (:func:`recheck_spec_options`).

        Returns
        -------
        list[ts.TensorStore]
            One handle per level, finest first.
        """
        raise NotImplementedError

    def _wants_rechecks(self) -> bool:
        """Whether the handles should revalidate cached data now.

        A registered writer whose buffer has been garbage collected still
        counts: rechecks stay on until :meth:`unregister_paint_writer`,
        which is slower but never stale.
        """
        return self.recheck_cached_data or self._paint_writer_ref is not None

    def _reopen_ts_stores(self) -> None:
        """Open the handles on a new context sized by ``cache_pool_bytes``.

        Used at construction and when the budget changes.  The new context
        starts with an empty pool.
        """
        context = build_context(self.cache_pool_bytes)
        recheck = self._wants_rechecks()
        self._ts_stores = self._open_ts_handles(context, recheck)
        self._ts_context = context
        self._ts_rechecking = recheck

    def _sync_rechecks(self) -> None:
        """Reopen on the same context if the recheck policy has changed.

        A no-op when the handles already follow the policy, so repeated
        registrations do not reopen.
        """
        recheck = self._wants_rechecks()
        if recheck == self._ts_rechecking:
            return
        if self._ts_context is None:
            # Not opened yet (still constructing): the first open reads the
            # policy itself.
            return
        self._ts_stores = self._open_ts_handles(self._ts_context, recheck)
        self._ts_rechecking = recheck

    def revalidate_cache(self) -> None:
        """Revalidate every cached chunk once, on its next read.

        Reopens the handles on the same context with
        ``recheck_cached_data="open"``: chunks cached before now are checked
        against the kvstore when next read (unchanged ones cost a ``304``,
        changed ones are refetched), then trusted again.  Called for every
        announced change (:meth:`_invalidate_caches`), so a store another
        process writes to shows the new data after ``notify_changed``.

        A no-op while rechecks are on, since every read revalidates then,
        and before the handles are opened.
        """
        if self._ts_context is None or self._ts_rechecking:
            return
        self._ts_stores = self._open_ts_handles(self._ts_context, "open")

    def _invalidate_caches(self, kind: Any) -> None:
        """Drop derived state, and revalidate the chunk cache once."""
        super()._invalidate_caches(kind)
        self.revalidate_cache()

    def __setattr__(self, key: str, value: Any) -> None:
        """Set a field, reopening the handles when the cache settings change.

        ``cache_pool_bytes`` rebuilds the pool on a new context.
        ``recheck_cached_data`` reopens on the same context.  Every other
        field is set as usual.  Assigning the value a field already has is
        a no-op, so a redundant write does not throw away a warm cache.

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
        if key == "recheck_cached_data":
            super().__setattr__(key, value)
            # Same context, so the paint buffer's handle stays on the live
            # pool and an open transaction is not a hazard.
            self._sync_rechecks()
            return
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
