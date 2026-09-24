# Debug Logging

Cellier includes structured debug logging to aid developers in understanding what is going on under the hood. By
default all loggers are silent (level `WARNING`). You opt in to the output you
need — by category and by level — so there is zero noise until you ask for it
and zero runtime cost in production.

## Categories

There are seven independent loggers, each covering a different part of the
pipeline:

| Category | Logger name | What it covers |
|----------|-------------------------|------------------------------------------------|
| `perf` | `cellier.render.perf` | Planning timing, fetch latency statistics |
| `gpu` | `cellier.render.gpu` | Brick/tile writes, LUT rebuilds |
| `cache` | `cellier.render.cache` | Cache hit/miss summaries, eviction, clear; `backstop_capped` (once per visual, INFO) when `backstop_max_slot_fraction` truncates a backstop |
| `slicer` | `cellier.render.slicer` | Async task lifecycle, batch progress |
| `scheduler` | `cellier.render.scheduler` | The chunk scheduler that loads multiscale visuals: per-cache passes, commit rounds (committed / evicted / discarded) and invalidations at INFO, each read issued and landed at DEBUG, failed reads at WARNING; the dims settle and rate-capped store-change reslices at INFO |
| `camera` | `cellier.render.camera` | Camera change detection, settle timer, reslice trigger |
| `source_id` | `cellier.render.source_id` | Source ID injection through the ContextVar → psygnal → bus bridge |

All seven live under the `cellier.render` parent logger, so standard Python
logging hierarchy applies.

## Log levels

Each category uses three levels:

| Level | What you see |
|-----------|--------------------------------------------------------------|
| `WARNING` | Anomalies only (e.g. cache budget exceeded). **Default.** |
| `INFO` | Per-frame / per-batch summaries — the useful overview. |
| `DEBUG` | Per-brick / per-tile detail — full firehose for deep debugging. |

### What each level looks like in practice

**`WARNING`** — only fires when something is wrong:

```
PERF  budget_exceeded  required=500  budget=256  dropped=244
```

**`INFO`** — one or two lines per frame / per batch:

```
PERF   [frame 7]  lod_select=1.2ms  dist_sort=0.3ms  frustum_cull=0.8ms  stage=0.1ms  |  required=142  culled=40  hits=130  misses=12
PERF   fetch_summary  status=complete  slice_id=d4e5...  bricks=12  batches=2  total=58ms  per_batch=28.3±5.1ms  min=23.2ms  max=33.4ms
CACHE  cache_state  frame=7  hot=130  reserve=8  free=374  hot_hits=120  reserve_hits=10  misses=12  evictions=0  pending_demote=0
SLICER batch_done  1/2  bricks=8  scales={0: 5, 1: 3}
GPU    gpu_flush  bricks_in_batch=8  resident=138
```

**`DEBUG`** — per-brick detail inside each batch:

```
PERF     fetch_batch  1/2  bricks=8  elapsed=23.2ms
SLICER     brick_received  id=abc123  scale=0  shape=(34, 34, 34)
CACHE    evict_hot  victim=BlockKey3D(level=1, gz=0, gy=2, gx=3)  slot=42
GPU      brick_written  key=BlockKey3D(level=1, gz=0, gy=1, gx=2)  slot=7  grid_pos=(0, 0, 7)
```

The `cache` examples above are from the 3D (hot/reserve) tile manager. The 2D
tile manager emits a simpler `cache_state  frame=…  occupied=…/…  free=…
hits=…  misses=…  evictions=…` line and a plain `evict` (no `evict_hot`).

## Quick start

### Enable all categories at DEBUG (full output)

```python
from cellier.logging import enable_debug_logging

enable_debug_logging()
```

### Enable specific categories

```python
enable_debug_logging(categories=("perf", "cache"))
```

### Summaries only (INFO level)

Pass `level` to set the minimum level for every enabled category. `INFO`
suppresses the per-brick / per-tile detail and keeps the per-frame and
per-batch summaries:

```python
import logging
from cellier.logging import enable_debug_logging

enable_debug_logging(level=logging.INFO)
```

### Mixed levels

`enable_debug_logging` applies a single `level` to all the categories it
enables. To run different categories at different levels, enable the most
verbose set first: the shared handler is created on the first call and keeps
that call's `level`, so the most verbose level must be installed up front or
the handler will filter the finer records out.

```python
import logging
from cellier.logging import enable_debug_logging

# Full detail for cache (installs the handler at DEBUG), summaries for perf.
enable_debug_logging(categories=("cache",), level=logging.DEBUG)
enable_debug_logging(categories=("perf",), level=logging.INFO)
```

### Disable logging

```python
from cellier.logging import disable_debug_logging

disable_debug_logging()
```

This resets all category loggers to `WARNING` and removes the handler.

### Plain output (no Rich)

```python
enable_debug_logging(use_rich=False)
```

Falls back to a plain `StreamHandler` with timestamps. Useful in CI or when
Rich is not installed.

## Rich colored output

If the `rich` package is installed (`pip install cellier[logging]`), the
default handler colors each category:

| Category | Color |
|----------|---------|
| PERF | cyan |
| GPU | green |
| CACHE | yellow |
| SLICER | magenta |
| CAMERA | blue |
| SOURCE_ID | red |

If Rich is not available, output falls back to a plain `StreamHandler`
automatically.

## Event source ID logging

The `source_id` category traces how `source_id` is injected into bus events.
It is useful when debugging echo-filtering issues or unexpected widget updates.
See [debugging_events](../How_To/debug_events.md) for a full walkthrough.

Enable it with:

```python
from cellier.logging import enable_debug_logging

enable_debug_logging(categories=("source_id",))
```

Each model mutation through a controller method produces three lines:

```
[SOURCE_ID] set    field=clim  visual=9c1b...  source=3f2a...
[SOURCE_ID] bridge handler=_on_appearance_psygnal  visual=9c1b...  field=clim  resolved_source=3f2a...  override_active=True
[SOURCE_ID] reset  field=clim  visual=9c1b...
```

- **`set`** — a controller mutation method (`update_appearance_field`,
  `update_slice_indices`, or `update_aabb_field`) was called and the
  `ContextVar` was set.
- **`bridge`** — the psygnal handler fired and read the `ContextVar`.
  `override_active=True` confirms the source came from a controller method.
  `override_active=False` means the model field was mutated directly, and
  `source_id` fell back to the controller's own ID.
- **`reset`** — the `ContextVar` was restored after the mutation completed.

## How it works under the hood

All cellier loggers start at Python's default level (`WARNING`).
`enable_debug_logging()` does two things:

1. Sets the requested category loggers to the requested `level` (`DEBUG` by
   default).
2. Attaches a handler to the `cellier.render` parent logger (once only), at
   that same level.

Every log call in the instrumented code is either:

- Emitted unconditionally (for `INFO` / `WARNING` calls, which are
  infrequent).
- Guarded by `if logger.isEnabledFor(logging.DEBUG)` (for hot loops over
  bricks/tiles), so there is zero string formatting cost when DEBUG is off.

Because the loggers are standard `logging.Logger` instances, you can
integrate them into any existing logging configuration (file handlers,
JSON formatters, external aggregation) using the logger names listed above.

## Fetch latency timing

The `perf` category includes fetch latency instrumentation for async data
loading. This is especially useful when working with remote data stores
where network I/O dominates.

At **INFO**, a single summary line is emitted when a fetch task completes
(or is cancelled):

```
PERF  fetch_summary  status=complete  slice_id=d4e5...  bricks=96  batches=12  total=340ms  per_batch=28.3±5.1ms  min=19.0ms  max=42.0ms
```

- `total` — wall-clock time from first batch to last, including callback
  overhead (GPU commit, LUT rebuild, Qt yield) between batches.
- `per_batch` — mean ± std of the I/O-only time for each `asyncio.gather()`
  call. This isolates the storage backend latency from the rendering overhead.
- `min` / `max` — fastest and slowest batch, useful for spotting tail
  latency spikes.

At **DEBUG**, each batch gets its own timing line as it completes:

```
PERF  fetch_batch  3/12  bricks=8  elapsed=31.2ms
```
