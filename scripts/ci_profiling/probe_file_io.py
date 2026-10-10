"""Time the small-file writes the multiscale tests do, on this machine's disk.

Run on every CI platform.  Many render tests write a fresh zarr pyramid into
``tmp_path`` (``tests/_plane_fixtures.write_pyramid``,
``tests/render/_plane_rig.write_levels``): a few hundred chunk files of a few
hundred kB each.  This probe times that write, and tells apart the three
things that could make it slow:

``tensorstore``
    The same write the tests do, with tensorstore's ``file_io_sync`` left at
    its default (every chunk is synced to disk) and switched off.
``raw``
    The same number of files of the same size written with ``open``, with and
    without ``os.fsync``.  Separates the file system from tensorstore.
``remove``
    Deleting the tree again.

Each is run in every directory given with ``--dir`` (by default the system
temp directory, where pytest's ``tmp_path`` lives, and the current directory,
which on a GitHub runner is on the workspace drive).

Usage
-----
    python scripts/ci_profiling/probe_file_io.py --out probe_file_io.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

#: The pyramid of ``tests/_plane_fixtures.ANISO``: level shapes, 32^3 chunks.
LEVEL_SHAPES = [
    (64, 256, 256),
    (64, 128, 128),
    (64, 64, 64),
    (64, 32, 32),
    (64, 16, 16),
]
CHUNK = 32


def _write_tensorstore(root: Path, sync: bool | None) -> None:
    import tensorstore as ts

    for level, shape in enumerate(LEVEL_SHAPES):
        kvstore = {"driver": "file", "path": str(root / f"s{level}")}
        if sync is not None:
            kvstore["file_io_sync"] = sync
        array = ts.open(
            {
                "driver": "zarr3",
                "kvstore": kvstore,
                "metadata": {
                    "shape": list(shape),
                    "data_type": "float32",
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {
                            "chunk_shape": [min(s, CHUNK) for s in shape]
                        },
                    },
                },
                "create": True,
                "delete_existing": True,
            }
        ).result()
        data = np.random.default_rng(level).random(shape, dtype=np.float32)
        array[...].write(data).result()


def _write_raw(root: Path, n_files: int, n_bytes: int, sync: bool) -> None:
    payload = os.urandom(n_bytes)
    root.mkdir(parents=True)
    for index in range(n_files):
        with open(root / f"chunk_{index}", "wb") as handle:
            handle.write(payload)
            if sync:
                handle.flush()
                os.fsync(handle.fileno())


def _count(root: Path) -> tuple[int, int]:
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def _timed(function) -> float:
    start = time.perf_counter()
    function()
    return (time.perf_counter() - start) * 1000


def measure_dir(base: Path, repeats: int) -> dict:
    """Run every case *repeats* times under *base*; return median ms."""
    rows: dict[str, list[float]] = {}
    n_files = n_bytes = 0
    with tempfile.TemporaryDirectory(dir=base, prefix="probe_file_io_") as scratch:
        scratch = Path(scratch)
        for repeat in range(repeats):
            for label, sync in (("default", None), ("sync_off", False)):
                root = scratch / f"ts_{label}_{repeat}"
                rows.setdefault(f"tensorstore_{label}_ms", []).append(
                    _timed(lambda r=root, s=sync: _write_tensorstore(r, s))
                )
                n_files, n_bytes = _count(root)
                rows.setdefault("remove_ms", []).append(
                    _timed(lambda r=root: shutil.rmtree(r))
                )
            per_file = n_bytes // max(n_files, 1)
            for label, sync in (("fsync", True), ("no_fsync", False)):
                root = scratch / f"raw_{label}_{repeat}"
                rows.setdefault(f"raw_{label}_ms", []).append(
                    _timed(
                        lambda r=root, s=sync, n=n_files, b=per_file: _write_raw(
                            r, n, b, s
                        )
                    )
                )
                shutil.rmtree(root)
    result = {name: round(statistics.median(times), 1) for name, times in rows.items()}
    result |= {"n_files": n_files, "n_bytes": n_bytes, "repeats": repeats}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir",
        action="append",
        help="a directory to write under (repeatable); default: temp dir and cwd",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", help="write the results here as JSON")
    args = parser.parse_args()

    directories = args.dir or [tempfile.gettempdir(), os.getcwd()]
    results = {"platform": platform.platform(), "directories": {}}
    print(f"platform: {results['platform']}")
    for directory in dict.fromkeys(str(Path(d).resolve()) for d in directories):
        row = measure_dir(Path(directory), args.repeats)
        results["directories"][directory] = row
        print(f"\n{directory}")
        print(f"  {row['n_files']} files, {row['n_bytes'] / 1e6:.1f} MB per pyramid")
        for name, value in row.items():
            if name.endswith("_ms"):
                print(f"  {name:28s} {value:9.1f}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
