"""Print what this CI machine is and how coverage would measure on it.

Two jobs running the same tests have differed by 1.6 times on GitHub
Actions.  This prints the three things that could explain it, so a slow job
can be told from a fast one in its log:

``cpu``
    The CPU model, core count and memory of the runner.
``speed``
    A fixed piece of work, timed: a pure-Python loop and a numpy product.
    Run without coverage, so it is the machine's speed alone.
``coverage core``
    The core coverage.py starts with this project's configuration
    (``[tool.coverage.run] core`` in pyproject.toml).  ``SysMonitor`` is the
    cheap one; ``CTracer`` and ``PyTracer`` pay a call on every line run.

Run from the repository root, so coverage finds pyproject.toml.

Usage
-----
    python scripts/ci_profiling/machine_info.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time


def cpu_model() -> str:
    """The CPU's marketing name, or what the platform can tell of it."""
    try:
        if sys.platform == "linux":
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception as error:
        return f"unknown ({error!r})"
    return platform.processor() or "unknown"


def memory_gb() -> float | None:
    """Physical memory in GB, or None where it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("rest", ctypes.c_ulonglong * 6),
                ]

            status = Status()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return None


def python_loop_ms() -> float:
    """Best of three of a fixed pure-Python loop."""

    def work() -> int:
        total = 0
        for index in range(2_000_000):
            total += index % 7
        return total

    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        work()
        best = min(best, time.perf_counter() - start)
    return best * 1000


def numpy_ms() -> float:
    """Best of three of a fixed numpy sort and matrix product."""
    import numpy as np

    rng = np.random.default_rng(0)
    values = rng.random(2_000_000)
    matrix = rng.random((400, 400))
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        np.sort(values)
        matrix @ matrix
        best = min(best, time.perf_counter() - start)
    return best * 1000


def coverage_core() -> str:
    """The core coverage.py starts under this directory's configuration."""
    try:
        import coverage
    except ImportError:
        return "coverage is not installed"
    cov = coverage.Coverage(data_file=None)
    cov.start()
    try:
        info = dict(cov.sys_info())
    finally:
        cov.stop()
    return (
        f"{info.get('core')} (configured: {cov.config.core!r}, "
        f"COVERAGE_CORE={os.environ.get('COVERAGE_CORE', 'unset')}, "
        f"coverage {coverage.__version__})"
    )


def main() -> None:
    memory = memory_gb()
    print(f"platform:       {platform.platform()}")
    print(f"python:         {platform.python_version()}")
    print(f"cpu:            {cpu_model()}")
    print(f"cpu count:      {os.cpu_count()}")
    print(f"memory:         {'unknown' if memory is None else f'{memory:.1f} GB'}")
    print(f"speed, python:  {python_loop_ms():.0f} ms")
    print(f"speed, numpy:   {numpy_ms():.0f} ms")
    print(f"coverage core:  {coverage_core()}")


if __name__ == "__main__":
    main()
