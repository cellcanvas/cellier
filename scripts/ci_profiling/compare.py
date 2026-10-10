"""Compare what the ``Profile`` workflow measured on each platform.

Download a run's artifacts, then point this at the directory:

    gh run download <run-id> -D profile_out
    python scripts/ci_profiling/compare.py probes profile_out
    python scripts/ci_profiling/compare.py variants profile_out
    python scripts/ci_profiling/compare.py timings profile_out --variant baseline
    python scripts/ci_profiling/compare.py profiles profile_out test_plane_planning

The directory holds a ``probes-<platform>`` and a ``profile-<platform>``
folder per platform:

    probes-<platform>/
        machine_info.txt, probe_render.json, probe_file_io.json,
        probe_gpu_calls.json, probe_chunk_read.json
    profile-<platform>/
        junit/<variant>.xml      per-test times (setup + call + teardown)
        prof/<test file>.prof    cProfile of one test file

``probes``    the micro-probes, side by side.
``variants``  seconds per test file, for every variant of every platform.
``timings``   the test files and tests one platform is slowest at.
``profiles``  the functions a test file spends its own time in, per platform.
"""

from __future__ import annotations

import argparse
import json
import pstats
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

PREFIX = "profile-"
PROBES_PREFIX = "probes-"


def _platforms(root: Path, prefix: str = PREFIX) -> dict[str, Path]:
    found = {
        p.name.removeprefix(prefix): p
        for p in sorted(root.iterdir())
        if p.is_dir() and p.name.startswith(prefix)
    }
    if not found:
        raise SystemExit(f"no {prefix}* folders under {root}")
    return found


def _probe_folders(root: Path) -> dict[str, Path]:
    """The folders holding probe results: ``probes-*``, else ``profile-*``."""
    try:
        return _platforms(root, PROBES_PREFIX)
    except SystemExit:
        return _platforms(root)


def _junit(path: Path) -> dict[str, float]:
    """``{"tests/x/test_y.py::test_z[case]": seconds}`` from a junit file."""
    times: dict[str, float] = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        module = case.get("classname", "").replace(".", "/") + ".py"
        times[f"{module}::{case.get('name')}"] = float(case.get("time", 0.0))
    return times


def _by_file(times: dict[str, float]) -> dict[str, float]:
    files: dict[str, float] = defaultdict(float)
    for test, seconds in times.items():
        files[test.split("::")[0]] += seconds
    return dict(files)


def _table(header: list[str], rows: list[list], width: int = 62) -> None:
    def cell(value):
        return f"{value:9.2f}" if isinstance(value, float) else f"{value!s:>9}"

    print(f"{header[0]:{width}s}" + "".join(f"{h[:9]:>10}" for h in header[1:]))
    for row in rows:
        name = str(row[0])
        name = name if len(name) <= width else "..." + name[-(width - 3) :]
        print(f"{name:{width}s}" + "".join(" " + cell(v) for v in row[1:]))


def probes(root: Path) -> None:
    platforms = _probe_folders(root)
    print("## render probe (ms)")
    rows = []
    for name, folder in platforms.items():
        path = folder / "probe_render.json"
        if not path.exists():
            continue
        for result in json.loads(path.read_text()):
            for scene, row in result["scenes"].items():
                rows.append(
                    [
                        f"{name}  threads={result['LP_NUM_THREADS']}  {scene}",
                        row["first_frame_ms"],
                        row["steady_median_ms"],
                    ]
                )
            adapter = result["adapter"]
        print(f"{name}: {adapter}")
    _table(["platform / LP_NUM_THREADS / scene", "first", "steady"], rows)

    print("\n## file probe (ms per pyramid)")
    rows = []
    keys: list[str] = []
    for name, folder in platforms.items():
        path = folder / "probe_file_io.json"
        if not path.exists():
            continue
        for directory, row in json.loads(path.read_text())["directories"].items():
            keys = [k for k in row if k.endswith("_ms")]
            rows.append([f"{name}  {directory}"] + [float(row[k]) for k in keys])
    short = [k.removesuffix("_ms").replace("tensorstore", "ts") for k in keys]
    _table(["platform / directory", *short], rows)

    per = {}
    for name, folder in platforms.items():
        path = folder / "probe_gpu_calls.json"
        if path.exists():
            per[name] = json.loads(path.read_text())["calls"]
    if per:
        print("\n## single wgpu calls (median ms per call)")
        calls = list(dict.fromkeys(c for v in per.values() for c in v))
        rows = [
            [c] + [float(per[n][c]["median_ms"]) if c in per[n] else "-" for n in per]
            for c in calls
        ]
        _table(["call", *per], rows, width=44)

    rows = []
    keys = []
    for name, folder in platforms.items():
        path = folder / "probe_chunk_read.json"
        if not path.exists():
            continue
        for directory, cases in json.loads(path.read_text())["directories"].items():
            for case, row in cases.items():
                keys = [k for k in row if k.endswith("_ms")]
                rows.append(
                    [f"{name}  {case}  {directory}"] + [float(row[k]) for k in keys]
                )
    if rows:
        print("\n## brick reads (ms)")
        _table(["platform / dataset / directory", *[k[:-3] for k in keys]], rows)

    for name, folder in platforms.items():
        path = folder / "machine_info.txt"
        if path.exists():
            print(f"\n## machine: {name}\n{path.read_text().rstrip()}")


def variants(root: Path) -> None:
    for name, folder in _platforms(root).items():
        files = sorted((folder / "junit").glob("*.xml"))
        per = {f.stem: _by_file(_junit(f)) for f in files if f.stem != "full"}
        if not per:
            continue
        names = sorted({t for v in per.values() for t in v})
        print(f"\n## {name}: seconds per test file, by variant")
        rows = [[t] + [v.get(t, 0.0) for v in per.values()] for t in names]
        rows.append(["total"] + [sum(v.values()) for v in per.values()])
        _table(["file", *per], rows)


def timings(root: Path, variant: str, slow: str, top: int) -> None:
    per = {}
    for name, folder in _platforms(root).items():
        path = folder / "junit" / f"{variant}.xml"
        if path.exists():
            per[name] = _junit(path)
    if not per:
        raise SystemExit(f"no junit/{variant}.xml in any folder under {root}")
    slow_name = next((n for n in per if slow in n), None)
    if slow_name is None:
        raise SystemExit(f"no platform matching {slow!r}; have {list(per)}")
    others = [n for n in per if n != slow_name]

    def excess(row: dict[str, float]) -> float:
        rest = [row.get(n, 0.0) for n in others]
        return row.get(slow_name, 0.0) - (sum(rest) / len(rest) if rest else 0.0)

    for title, tables in (
        ("test files", {n: _by_file(t) for n, t in per.items()}),
        ("tests", per),
    ):
        names = {t for v in tables.values() for t in v}
        rows = [{n: tables[n].get(t, 0.0) for n in tables} | {"": t} for t in names]
        rows.sort(key=excess, reverse=True)
        print(f"\n## {title} where {slow_name} is slowest ({variant}), seconds")
        _table(
            [title, *per, "excess"],
            [[r[""]] + [r[n] for n in per] + [excess(r)] for r in rows[:top]],
        )
    print(
        "\ntotals: " + ", ".join(f"{n} {sum(t.values()):.0f}s" for n, t in per.items())
    )


def profiles(root: Path, test_file: str, top: int) -> None:
    per: dict[str, dict[str, float]] = {}
    totals = {}
    for name, folder in _platforms(root).items():
        path = folder / "prof" / f"{test_file}.prof"
        if not path.exists():
            continue
        stats = pstats.Stats(str(path))
        totals[name] = stats.total_tt
        per[name] = defaultdict(float)
        for (filename, _line, function), row in stats.stats.items():
            # No line number: it differs between platforms' library versions.
            # The base name by hand: a profile written on Windows has
            # backslashes, which strip_dirs leaves alone on other platforms.
            base = filename.replace("\\", "/").rsplit("/", 1)[-1]
            per[name][f"{base}:{function}"] += row[2]
    if not per:
        raise SystemExit(f"no prof/{test_file}.prof in any folder under {root}")
    names = {f for v in per.values() for f in v}
    ranked = sorted(names, key=lambda f: -max(v.get(f, 0.0) for v in per.values()))
    print(f"## {test_file}: own time per function, seconds")
    rows = [[f] + [per[n].get(f, 0.0) for n in per] for f in ranked[:top]]
    rows.append(["total"] + [totals[n] for n in per])
    _table(["function", *per], rows)
    print(
        "\n_helpers.py:proxy_func is every call into wgpu-native; see its callers"
        '\nwith: python -c "import pstats,sys; pstats.Stats(sys.argv[1])'
        ".strip_dirs().print_callers('proxy_func')\" <file.prof>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probes", "variants", "timings", "profiles"):
        sub = commands.add_parser(name)
        sub.add_argument("root", type=Path)
        if name == "timings":
            sub.add_argument("--variant", default="baseline")
            sub.add_argument("--slow", default="windows", help="the slow platform")
            sub.add_argument("--top", type=int, default=30)
        if name == "profiles":
            sub.add_argument("test_file", help="e.g. test_plane_planning")
            sub.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    if args.command == "probes":
        probes(args.root)
    elif args.command == "variants":
        variants(args.root)
    elif args.command == "timings":
        timings(args.root, args.variant, args.slow, args.top)
    else:
        profiles(args.root, args.test_file, args.top)


if __name__ == "__main__":
    main()
