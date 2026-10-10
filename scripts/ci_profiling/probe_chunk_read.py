"""Time the brick reads the multiscale tests do, on this machine's disk.

Run on every CI platform, from the repository root.  A brick read took 63 ms
on the Windows runner in ``tests/render/test_multiscale_level_alignment.py``
(19 s over 304 calls) and next to nothing elsewhere, and the plane examples
lost 12 s to reads (``scripts/ci_profiling/ci_profiling_investigation.md``).
This probe builds the two datasets those tests read, with the tests' own
writers, and times the read the stores do, part by part.

``alignment``
    The dataset of ``test_multiscale_level_alignment.py``: an OME-Zarr 0.5
    group written with zarr-python, every level ONE compressed chunk, opened
    by ``OMEZarrImageDataStore.from_path`` through a ``file://`` URI.
``pyramid``
    The pyramid of ``tests/_plane_fixtures``: zarr v3 written with
    tensorstore, 32^3 chunks, opened by ``MultiscaleZarrDataStore``.

For each, in each directory:

``open_ms``
    Opening the store (metadata reads, one tensorstore handle per level).
``first_read_ms``
    The first brick read: nothing is cached yet.
``new_brick_ms``
    The median read of a brick not read before.  For ``alignment`` the one
    chunk is cached by then, so this should cost a copy.
``same_brick_ms``
    The median of reading one brick again and again: everything is cached.
``get_data_ms``
    The same bricks through the store's own ``get_data``, on an event loop,
    as the scheduler calls it.
``index_ms`` / ``start_ms`` / ``wait_ms``
    The read split in three: indexing the tensorstore handle, starting the
    read, and waiting for its result.

Usage
-----
    python scripts/ci_profiling/probe_chunk_read.py --out probe_chunk_read.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

# The probe reuses the tests' own dataset writers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BRICK = 32
PAD = 1


def _bricks(shape: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
    """Padded brick selections tiling *shape*, as the planner asks for them."""
    grids = [range(0, max(n, 1), BRICK) for n in shape]
    return [
        tuple((start - PAD, start + BRICK + PAD) for start in corner)
        for corner in np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
    ]


def _clamped(selection, shape) -> tuple[slice, ...]:
    return tuple(
        slice(max(start, 0), min(stop, n)) for (start, stop), n in zip(selection, shape)
    )


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 3) if values else float("nan")


def _time_reads(store, level: int = 0) -> dict:
    from cellier.data.image._image_requests import ChunkRequest

    handle = store._ts_stores[level]
    shape = tuple(int(n) for n in handle.domain.shape)
    bricks = _bricks(shape)

    def read(selection) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        view = handle[_clamped(selection, shape)]
        t1 = time.perf_counter()
        future = view.read()
        t2 = time.perf_counter()
        np.asarray(future.result(), dtype=np.float32)
        t3 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    first = read(bricks[0])
    new = [read(selection) for selection in bricks[1:]]
    same = [read(bricks[0]) for _ in range(50)]

    async def through_the_store() -> list[float]:
        times = []
        for selection in bricks:
            request = ChunkRequest(uuid4(), uuid4(), level, selection)
            start = time.perf_counter()
            await store.get_data(request)
            times.append((time.perf_counter() - start) * 1000)
        return times

    get_data = asyncio.run(through_the_store())
    return {
        "level_shape": list(shape),
        "n_bricks": len(bricks),
        "first_read_ms": round(sum(first), 3),
        "new_brick_ms": _median([sum(parts) for parts in new]),
        "same_brick_ms": _median([sum(parts) for parts in same]),
        "get_data_ms": _median(get_data),
        "get_data_max_ms": round(max(get_data), 3),
        "index_ms": _median([parts[0] for parts in new + same]),
        "start_ms": _median([parts[1] for parts in new + same]),
        "wait_ms": _median([parts[2] for parts in new + same]),
    }


def _alignment(root: Path) -> dict:
    from tests.render.test_multiscale_level_alignment import _write_dataset

    from cellier.data import OMEZarrImageDataStore

    start = time.perf_counter()
    path = _write_dataset(root)
    write_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    store = OMEZarrImageDataStore.from_path(path.as_uri())
    open_ms = (time.perf_counter() - start) * 1000
    return {
        "write_ms": round(write_ms, 1),
        "open_ms": round(open_ms, 1),
    } | _time_reads(store)


def _pyramid(root: Path) -> dict:
    from tests import _plane_fixtures as fx

    start = time.perf_counter()
    fx.write_pyramid(root, fx.ANISO)
    write_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    store, _voxel = fx.open_pyramid(root, fx.ANISO)
    open_ms = (time.perf_counter() - start) * 1000
    return {
        "write_ms": round(write_ms, 1),
        "open_ms": round(open_ms, 1),
    } | _time_reads(store)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir",
        action="append",
        help="a directory to write under (repeatable); default: temp dir and cwd",
    )
    parser.add_argument("--out", help="write the results here as JSON")
    args = parser.parse_args()

    directories = args.dir or [tempfile.gettempdir(), os.getcwd()]
    results = {"platform": platform.platform(), "directories": {}}
    print(f"platform: {results['platform']}")
    for directory in dict.fromkeys(str(Path(d).resolve()) for d in directories):
        rows = {}
        with tempfile.TemporaryDirectory(
            dir=directory, prefix="probe_chunk_read_", ignore_cleanup_errors=True
        ) as scratch:
            for name, build in (("alignment", _alignment), ("pyramid", _pyramid)):
                root = Path(scratch) / name
                root.mkdir()
                rows[name] = build(root)
        results["directories"][directory] = rows
        print(f"\n{directory}")
        for name, row in rows.items():
            print(f"  {name}: level 0 {row['level_shape']}, {row['n_bricks']} bricks")
            for key, value in row.items():
                if key.endswith("_ms"):
                    print(f"    {key:18s} {value:10.3f}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
