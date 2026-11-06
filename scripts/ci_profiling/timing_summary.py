"""Where a test run spent its time, from pytest's ``--junitxml`` file.

    python scripts/ci_profiling/timing_summary.py test-results.xml
    python scripts/ci_profiling/timing_summary.py test-results.xml --top 40

Prints Markdown: the run's wall time against the time inside tests, the
time each pytest-xdist worker spent in tests (from the ``xdist_worker``
property ``tests/conftest.py`` records), and the slowest test files and
tests.  CI appends it to the job's step summary
(``.github/workflows/ci.yml``, "Test timings").

A test's time is its setup, call and teardown together (junit's default
``junit_duration_report = total``).  With ``--dist loadfile`` a file runs
whole on one worker, so a file's time is how long it held that worker.
The junit file itself is uploaded as an artifact; ``compare.py`` reads the
same format.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def _module(classname: str) -> str:
    """``tests/x/test_y.py`` from junit's ``tests.x.test_y[.TestClass]``."""
    parts = classname.split(".")
    while len(parts) > 1 and parts[-1].startswith("Test"):
        parts.pop()
    return "/".join(parts) + ".py"


def _minutes(seconds: float) -> str:
    return f"{seconds:.0f} s ({seconds / 60:.1f} min)"


def summarise(path: Path, top: int) -> str:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    wall = sum(float(s.get("time", 0.0)) for s in suites)

    tests: list[tuple[float, str]] = []
    files: dict[str, float] = defaultdict(float)
    file_counts: dict[str, int] = defaultdict(int)
    workers: dict[str, float] = defaultdict(float)
    worker_counts: dict[str, int] = defaultdict(int)
    for case in root.iter("testcase"):
        seconds = float(case.get("time", 0.0))
        module = _module(case.get("classname", ""))
        tests.append((seconds, f"{module}::{case.get('name')}"))
        files[module] += seconds
        file_counts[module] += 1
        worker = "main"
        for prop in case.iter("property"):
            if prop.get("name") == "xdist_worker":
                worker = prop.get("value", "main")
        workers[worker] += seconds
        worker_counts[worker] += 1

    busy = sum(seconds for seconds, _test in tests)
    lines = [
        "## Test timings",
        "",
        f"- Wall time: {_minutes(wall)}",
        f"- Time inside tests, all workers: {_minutes(busy)} over {len(tests)} tests",
        "",
    ]
    if len(workers) > 1:
        lines += ["| Worker | Tests | Time in tests |", "|---|---:|---:|"]
        for worker in sorted(workers, key=lambda w: (len(w), w)):
            lines.append(
                f"| {worker} | {worker_counts[worker]} | {workers[worker]:.0f} s |"
            )
        lines.append("")

    lines += [
        f"### Slowest {top} files",
        "",
        "| Seconds | Tests | File |",
        "|---:|---:|---|",
    ]
    for module, seconds in sorted(files.items(), key=lambda kv: -kv[1])[:top]:
        lines.append(f"| {seconds:.1f} | {file_counts[module]} | `{module}` |")
    lines += [
        "",
        f"### Slowest {top} tests",
        "",
        "| Seconds | Test |",
        "|---:|---|",
    ]
    for seconds, test in sorted(tests, reverse=True)[:top]:
        lines.append(f"| {seconds:.2f} | `{test}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("junit", type=Path, help="pytest's --junitxml file")
    parser.add_argument("--top", type=int, default=25, help="rows per table")
    args = parser.parse_args()
    if not args.junit.exists():
        raise SystemExit(f"no junit file at {args.junit}: did the tests run?")
    print(summarise(args.junit, args.top), end="")


if __name__ == "__main__":
    main()
