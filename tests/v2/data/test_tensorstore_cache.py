"""Tests for the per-store tensorstore chunk cache.

The behaviour under test is that a chunk decompressed for one read is
reused by the next, rather than being dropped as soon as the reading batch
finishes.  That is asserted with tensorstore's own hit/miss counters rather
than with wall-clock timing, so the tests are exact.

The counters are process-wide and cumulative, so every assertion here takes
a delta around the reads it cares about.  These tests assume serial
execution; a parallel runner sharing the process would interleave counts.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest

from cellier.data._tensorstore_cache import (
    DEFAULT_CACHE_POOL_BYTES,
    build_context,
    cache_metrics,
)
from cellier.data.image._image_requests import ChunkRequest
from cellier.data.image._ome_zarr_image_store import OMEZarrImageDataStore
from cellier.data.label._ome_zarr_label_store import OMEZarrLabelDataStore

if TYPE_CHECKING:
    import pathlib

# ---------------------------------------------------------------------------
# Fixture: a 3-level OME-Zarr whose chunks each hold a whole level
#
# One chunk per level is the case the cache exists for: every brick read
# lands in the same chunk, so without caching each read re-decompresses the
# whole level.
# ---------------------------------------------------------------------------

_LEVEL_SHAPES = [(32, 32), (16, 16), (8, 8)]
_AXES = [
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]
_DATASETS = [
    {
        "path": "0",
        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}],
    },
    {
        "path": "1",
        "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0]}],
    },
    {
        "path": "2",
        "coordinateTransformations": [{"type": "scale", "scale": [4.0, 4.0]}],
    },
]


def _write_single_chunk_ome_zarr(root: pathlib.Path, dtype: str = "uint16") -> None:
    """Write a 3-level OME-Zarr v0.5 image, one chunk per level."""
    import zarr

    root.mkdir(parents=True, exist_ok=True)
    (root / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [
                            {
                                "name": "test",
                                "axes": _AXES,
                                "datasets": _DATASETS,
                                "version": "0.5",
                            }
                        ],
                    }
                },
            }
        )
    )
    for dataset, shape in zip(_DATASETS, _LEVEL_SHAPES):
        array = zarr.create(
            store=zarr.storage.LocalStore(str(root / dataset["path"])),
            shape=shape,
            dtype=dtype,
            chunks=shape,  # one chunk covers the whole level
            zarr_format=3,
        )
        array[...] = np.arange(int(np.prod(shape))).reshape(shape).astype(dtype)


@pytest.fixture
def single_chunk_uri(tmp_path: pathlib.Path) -> str:
    """A 3-level OME-Zarr image whose levels are one chunk each."""
    store_path = tmp_path / "image.ome.zarr"
    _write_single_chunk_ome_zarr(store_path)
    return f"file://{store_path}"


@pytest.fixture
def single_chunk_label_uri(tmp_path: pathlib.Path) -> str:
    """The same geometry, as a label group."""
    store_path = tmp_path / "labels.ome.zarr"
    _write_single_chunk_ome_zarr(store_path, dtype="int32")
    return f"file://{store_path}"


def _read_windows(store, n: int) -> None:
    """Read *n* distinct 4x4 windows that all live in the level-0 chunk."""
    for i in range(n):
        offset = (i * 4) % 16
        asyncio.run(
            store.get_data(
                ChunkRequest(
                    chunk_request_id=uuid4(),
                    slice_request_id=uuid4(),
                    scale_index=0,
                    axis_selections=((offset, offset + 4), (0, 4)),
                )
            )
        )


def _hits_and_misses(store, n: int) -> tuple[int, int]:
    """Return the ``(hits, misses)`` delta over *n* same-chunk reads."""
    hits_before, misses_before = cache_metrics()
    _read_windows(store, n)
    hits_after, misses_after = cache_metrics()
    return hits_after - hits_before, misses_after - misses_before


# ---------------------------------------------------------------------------
# The cache actually caches
# ---------------------------------------------------------------------------


def test_caching_on_reuses_the_decompressed_chunk(single_chunk_uri: str) -> None:
    """With a pool, only the first of six same-chunk reads is a miss."""
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    hits, misses = _hits_and_misses(store, 6)
    assert misses == 1
    assert hits == 5


def test_caching_off_repeats_every_read(single_chunk_uri: str) -> None:
    """With ``cache_pool_bytes=0`` every read re-reads the chunk."""
    store = OMEZarrImageDataStore.from_path(single_chunk_uri, cache_pool_bytes=0)
    hits, misses = _hits_and_misses(store, 6)
    assert hits == 0
    assert misses == 6


def test_label_store_caches_too(single_chunk_label_uri: str) -> None:
    store = OMEZarrLabelDataStore.from_path(single_chunk_label_uri)
    hits, misses = _hits_and_misses(store, 6)
    assert misses == 1
    assert hits == 5


def test_levels_share_one_pool(single_chunk_uri: str, monkeypatch) -> None:
    """All levels of a store are opened against the same context.

    Asserted at the call site: a ``Spec`` does not expose the context it was
    opened with, and a store's levels hold different chunks, so there is no
    read whose hit/miss outcome would reveal the sharing.
    """
    import tensorstore as ts

    import cellier.data.image._ome_zarr_image_store as module

    seen: list[object] = []
    original = module.ts.open

    def _spy(spec, *args, **kwargs):
        seen.append(kwargs.get("context"))
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(module.ts, "open", _spy)
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)

    assert len(seen) == len(store.scale_names) > 1
    assert all(context is seen[0] for context in seen)
    assert isinstance(seen[0], ts.Context)


def test_stores_do_not_share_a_pool(single_chunk_uri: str) -> None:
    """Each store owns its cache, so one store's budget cannot starve another."""
    cached = OMEZarrImageDataStore.from_path(single_chunk_uri)
    uncached = OMEZarrImageDataStore.from_path(single_chunk_uri, cache_pool_bytes=0)

    # Warm the cached store, then read the same data through the uncached one.
    _read_windows(cached, 2)
    hits, misses = _hits_and_misses(uncached, 3)
    assert hits == 0
    assert misses == 3

    # The cached store is still warm.
    hits, misses = _hits_and_misses(cached, 3)
    assert hits == 3
    assert misses == 0


# ---------------------------------------------------------------------------
# Setting the field
# ---------------------------------------------------------------------------


def test_default_is_applied(single_chunk_uri: str) -> None:
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    assert store.cache_pool_bytes == DEFAULT_CACHE_POOL_BYTES


def test_assignment_rebuilds_the_pool(single_chunk_uri: str) -> None:
    """Assigning the field reopens the handles against the new budget."""
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    _read_windows(store, 2)
    handle_before = store._ts_stores[0]

    store.cache_pool_bytes = 0

    assert store.cache_pool_bytes == 0
    assert store._ts_stores[0] is not handle_before
    # The warm pool is gone: nothing hits any more.
    hits, misses = _hits_and_misses(store, 4)
    assert hits == 0
    assert misses == 4


def test_assignment_back_on_restores_caching(single_chunk_uri: str) -> None:
    store = OMEZarrImageDataStore.from_path(single_chunk_uri, cache_pool_bytes=0)
    store.cache_pool_bytes = 8 * 1024**2
    hits, misses = _hits_and_misses(store, 4)
    assert hits == 3
    assert misses == 1


def test_assigning_the_same_value_keeps_the_warm_cache(single_chunk_uri: str) -> None:
    """A redundant write must not throw away the handles or the cache."""
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    _read_windows(store, 1)
    handle_before = store._ts_stores[0]

    store.cache_pool_bytes = DEFAULT_CACHE_POOL_BYTES

    assert store._ts_stores[0] is handle_before
    hits, misses = _hits_and_misses(store, 3)
    assert hits == 3
    assert misses == 0


def test_other_fields_are_unaffected(single_chunk_uri: str) -> None:
    """Assigning an unrelated field does not reopen the handles."""
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    handle_before = store._ts_stores[0]
    store.name = "renamed"
    assert store.name == "renamed"
    assert store._ts_stores[0] is handle_before


def test_field_signal_still_fires(single_chunk_uri: str) -> None:
    """The psygnal per-field signal survives the __setattr__ override."""
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    seen: list[int] = []
    store.events.cache_pool_bytes.connect(seen.append)
    store.cache_pool_bytes = 4 * 1024**2
    assert seen == [4 * 1024**2]


def test_negative_budget_rejected_at_construction(single_chunk_uri: str) -> None:
    with pytest.raises(Exception, match="cache_pool_bytes"):
        OMEZarrImageDataStore.from_path(single_chunk_uri, cache_pool_bytes=-1)


def test_negative_budget_rejected_on_assignment(single_chunk_uri: str) -> None:
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    handle_before = store._ts_stores[0]
    with pytest.raises(ValueError, match="cache_pool_bytes must be >= 0"):
        store.cache_pool_bytes = -1
    assert store.cache_pool_bytes == DEFAULT_CACHE_POOL_BYTES
    assert store._ts_stores[0] is handle_before


def test_roundtrips_through_serialization(single_chunk_uri: str) -> None:
    store = OMEZarrImageDataStore.from_path(single_chunk_uri, cache_pool_bytes=1234)
    dumped = store.model_dump()
    assert dumped["cache_pool_bytes"] == 1234
    assert OMEZarrImageDataStore.model_validate(dumped).cache_pool_bytes == 1234


# ---------------------------------------------------------------------------
# Paint interlock
# ---------------------------------------------------------------------------


class _FakeWriteBuffer:
    """Stands in for ``TensorStoreWriteBuffer``: only ``transaction`` matters."""

    def __init__(self) -> None:
        self.transaction = object()

    def commit(self) -> None:
        self.transaction = None


def test_assignment_refused_during_a_paint_transaction(single_chunk_uri: str) -> None:
    """Reopening would orphan the paint buffer's handle, so it is refused."""
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    handle_before = store._ts_stores[0]
    writer = _FakeWriteBuffer()
    store.register_paint_writer(writer)

    with pytest.raises(RuntimeError, match="paint transaction is open"):
        store.cache_pool_bytes = 0

    assert store.cache_pool_bytes == DEFAULT_CACHE_POOL_BYTES
    assert store._ts_stores[0] is handle_before


def test_assignment_allowed_after_commit(single_chunk_uri: str) -> None:
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    writer = _FakeWriteBuffer()
    store.register_paint_writer(writer)
    writer.commit()

    store.cache_pool_bytes = 0
    assert store.cache_pool_bytes == 0


def test_assignment_allowed_after_unregister(single_chunk_uri: str) -> None:
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    store.register_paint_writer(_FakeWriteBuffer())
    store.unregister_paint_writer()

    store.cache_pool_bytes = 0
    assert store.cache_pool_bytes == 0


def test_dropped_writer_does_not_lock_the_store(single_chunk_uri: str) -> None:
    """The back-reference is weak, so a discarded controller releases the lock."""
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    store.register_paint_writer(_FakeWriteBuffer())  # no strong reference kept
    import gc

    gc.collect()

    store.cache_pool_bytes = 0
    assert store.cache_pool_bytes == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_committed_write_is_visible_through_the_cached_handle(
    single_chunk_uri: str,
) -> None:
    """A cached read must not serve pre-write data after a paint commit.

    The paint module writes through the same handle the render path reads
    from, so the cache has to see a committed transaction.  Tensorstore
    guarantees this today; this pins it, because a regression here would
    surface as painted voxels silently reverting on the next reslice rather
    than as an error.
    """
    import tensorstore as ts

    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    handle = store._ts_stores[0]

    # Warm the cache for this chunk, and confirm the starting value.
    assert int(handle[5, 5].read().result()) != 4242

    transaction = ts.Transaction()
    handle.with_transaction(transaction)[5, 5] = 4242
    # Staged but not committed: the cached read still sees the old value.
    assert int(handle[5, 5].read().result()) != 4242
    transaction.commit_sync()

    assert int(handle[5, 5].read().result()) == 4242


def test_committed_write_is_visible_at_coarser_levels(single_chunk_uri: str) -> None:
    """The same, for the pyramid levels a paint commit rebuilds."""
    import tensorstore as ts

    store = OMEZarrImageDataStore.from_path(single_chunk_uri)
    for level, handle in enumerate(store._ts_stores):
        handle[1, 1].read().result()  # warm
        transaction = ts.Transaction()
        handle.with_transaction(transaction)[1, 1] = 900 + level
        transaction.commit_sync()
        assert int(handle[1, 1].read().result()) == 900 + level


def test_mixin_does_not_swallow_the_base_model_post_init(single_chunk_uri: str) -> None:
    """The mixin must not break ``BaseDataStore``'s post-init chain.

    Declaring a ``PrivateAttr`` makes pydantic inject
    ``init_private_attributes`` as a class's ``model_post_init``, and that
    injected function does not call ``super()``.  With the mixin ahead of
    ``BaseDataStore`` in the MRO, that would silently skip the base's level
    transform installation and coordinate-system checks -- stores would
    still construct, just wrong.
    """
    store = OMEZarrImageDataStore.from_path(single_chunk_uri)

    # Installed by BaseDataStore.model_post_init.
    assert len(store.level_transforms) == len(store.scale_names)

    # And its shape check still runs: one system for three levels is refused.
    with pytest.raises(ValueError, match="3 resolution level"):
        OMEZarrImageDataStore(
            zarr_path=store.zarr_path,
            scale_names=store.scale_names,
            level_scales=store.level_scales,
            level_translations=store.level_translations,
            data_coordinate_systems=[store.data_coordinate_systems[0]],
        )


def test_build_context_sets_the_limit() -> None:
    context = build_context(1234)
    assert context["cache_pool"].to_json()["total_bytes_limit"] == 1234


def test_cache_metrics_returns_two_counts() -> None:
    hits, misses = cache_metrics()
    assert isinstance(hits, int)
    assert isinstance(misses, int)
