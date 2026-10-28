"""A/B the Windows slab hang against llvmpipe's AVX-512 on one runner.

The Windows jobs hung while lavapipe shaded the first slab projection frames
(``ci_profiling_investigation.md``, "The Windows hang in create_texture"),
and only on runners with AVX-512.  ``GALLIUM_OVERRIDE_CPU_CAPS=avx`` takes
AVX-512 away from the JIT.  Comparing jobs leaves the CPU to chance; this
runs both arms on the same machine.

Each round runs the thickness-1 cases of
``test_a_projection_reduces_the_slab_voxel_centres`` (the ones the jobs hung
in) in a fresh pytest process, alternating the arms so drift in the machine
falls on both.  The variable is read when the driver loads, hence the
processes.  A process still running after ``--timeout`` seconds is killed
and counted as a hang: the cases take a few seconds when they finish.

The outcomes go to stdout, to ``--out`` (JSON) and, on GitHub Actions, to
the job summary.  It exits 0 whatever they are: it is an experiment, not a
test.  The hang has only been seen after half the suite had run, so no hang
in either arm says the fresh process does not reproduce it, not that the
override is irrelevant.

Usage
-----
    python scripts/ci_profiling/avx512_ab.py --rounds 5 --timeout 90 --out ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from machine_info import cpu_model, has_avx512

TEST = "tests/render/test_oblique_2d_projection.py::test_a_projection_reduces_the_slab_voxel_centres"
CASES = [
    f"{TEST}[1-{op}-{family}]"
    for op in ("mip", "minip", "mean")
    for family in ("image_multiscale", "image")
]
#: The two arms: the override's value, or None to remove the variable.
ARMS = {"avx": "avx", "unset": None}


def run_once(arm: str, timeout: float, log: Path) -> dict:
    """One fresh pytest process over `CASES`, its outcome and its time."""
    env = dict(os.environ)
    env.pop("GALLIUM_OVERRIDE_CPU_CAPS", None)
    if ARMS[arm] is not None:
        env["GALLIUM_OVERRIDE_CPU_CAPS"] = ARMS[arm]
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *CASES]
    start = time.perf_counter()
    with log.open("w", encoding="utf-8", errors="replace") as out:
        try:
            code = subprocess.run(
                command,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            ).returncode
            outcome = "pass" if code == 0 else f"fail (exit {code})"
        except subprocess.TimeoutExpired:
            outcome = "hang"
    return {
        "arm": arm,
        "outcome": outcome,
        "seconds": round(time.perf_counter() - start, 1),
        "log": str(log),
    }


def summary_table(runs: list[dict]) -> str:
    lines = [
        "| Arm | Runs | Pass | Hang | Fail | Seconds (each) |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        mine = [r for r in runs if r["arm"] == arm]
        passes = sum(r["outcome"] == "pass" for r in mine)
        hangs = sum(r["outcome"] == "hang" for r in mine)
        times = ", ".join(f"{r['seconds']:.0f}" for r in mine)
        lines.append(
            f"| `{arm}` | {len(mine)} | {passes} | {hangs} | "
            f"{len(mine) - passes - hangs} | {times} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=5, help="runs per arm")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds")
    parser.add_argument("--out", type=Path, default=Path("avx512_ab.json"))
    args = parser.parse_args()

    log_dir = args.out.parent / (args.out.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    avx512 = has_avx512()
    print(f"cpu:     {cpu_model()}")
    print(f"avx-512: {'unknown' if avx512 is None else 'yes' if avx512 else 'no'}")
    print(f"cases:   {len(CASES)} per run, timeout {args.timeout:.0f} s", flush=True)

    runs = []
    for index in range(args.rounds):
        # Alternate which arm goes first, so neither always runs cold.
        order = list(ARMS) if index % 2 == 0 else list(reversed(ARMS))
        for arm in order:
            run = run_once(arm, args.timeout, log_dir / f"{index:02d}-{arm}.log")
            runs.append(run)
            print(
                f"round {index}  {arm:<5}  {run['outcome']:<14} {run['seconds']:6.1f} s",
                flush=True,
            )

    table = summary_table(runs)
    print("\n" + table)
    args.out.write_text(
        json.dumps(
            {
                "cpu": cpu_model(),
                "avx512": avx512,
                "timeout": args.timeout,
                "runs": runs,
            },
            indent=2,
        )
    )
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(
                f"### AVX-512 A/B, Python {sys.version_info.major}.{sys.version_info.minor}\n\n"
                f"{cpu_model()}, AVX-512 {'unknown' if avx512 is None else avx512}\n\n"
                f"{table}\n"
            )


if __name__ == "__main__":
    main()
