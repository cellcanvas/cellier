"""Native stack dumps of a test that runs too long (a CI diagnostic).

A Windows CI job hung inside wgpu-native's ``create_texture``, right after a
frame was submitted to lavapipe, and pytest's own dumps stop at the Python
frame that made the call.  With ``CELLIER_NATIVE_DUMP_AFTER`` set (seconds
into a test, comma separated, such as ``"200,400"``) a test still running at
each of those times gets its process dumped by whichever tools are present:

- ``py-spy dump --native``: every Python thread, native frames included.
- ``cdb`` (``CELLIER_CDB``, Windows only): every thread of the process,
  lavapipe's own too, attached without stopping the process for good
  (``-pv``), plus a minidump to open in WinDbg later.  Its symbol path is
  ``CELLIER_CDB_SYMBOLS`` when set: cdb's default does not look in the folder
  of a DLL for its ``.pdb``.

Two dumps of the same hang tell a wait (the same stacks) from slow work
(stacks that moved).  The files go to ``CELLIER_NATIVE_DUMP_DIR`` (default
``native_dumps``).  Pick times below ``faulthandler_timeout`` and the
pytest-timeout limit in ``pyproject.toml``: pytest-timeout ends the process.

Unset, nothing runs.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

#: Seconds into a test at which to dump; empty when unset.
DUMP_AFTER = tuple(
    float(s) for s in os.environ.get("CELLIER_NATIVE_DUMP_AFTER", "").split(",") if s
)
DUMP_DIR = Path(os.environ.get("CELLIER_NATIVE_DUMP_DIR", "native_dumps"))
#: How long one tool may take before it is abandoned.
TOOL_TIMEOUT = 150.0

# ``lm``: the modules and whether their symbols loaded; ``~*kn``: every
# thread's stack, frame numbers on; ``qd``: detach and leave the process be.
# The path goes unquoted: ``list2cmdline`` would escape quotes as ``\"``, which
# cdb's command parser need not read the same way, and on CI it holds no space.
_CDB_COMMANDS = "lm; ~*kn 80; .dump /m /o {dmp}; qd"


@contextlib.contextmanager
def watch(nodeid: str):
    """Dump the process at each of `DUMP_AFTER` while the body runs."""
    timers = [
        threading.Timer(delay, _dump, args=(nodeid, delay)) for delay in DUMP_AFTER
    ]
    for timer in timers:
        timer.daemon = True
        timer.start()
    try:
        yield
    finally:
        for timer in timers:
            timer.cancel()


def _dump(nodeid: str, delay: float) -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', nodeid)[-120:]}-{delay:.0f}s"
    pid = os.getpid()
    header = (
        f"test:    {nodeid}\n"
        f"elapsed: {delay:.0f} s\n"
        f"pid:     {pid}\n"
        f"time:    {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
    )
    py_spy = shutil.which("py-spy")
    if py_spy is not None:
        _run(
            DUMP_DIR / f"{stem}-py-spy.txt",
            header,
            [py_spy, "dump", "--native", "--pid", str(pid)],
        )
    cdb = os.environ.get("CELLIER_CDB")
    if cdb and Path(cdb).is_file():
        dmp = (DUMP_DIR / f"{stem}.dmp").resolve()
        symbols = os.environ.get("CELLIER_CDB_SYMBOLS")
        _run(
            DUMP_DIR / f"{stem}-cdb.txt",
            header,
            [cdb, "-pv", "-p", str(pid)]
            + (["-y", symbols] if symbols else [])
            + ["-c", _CDB_COMMANDS.format(dmp=dmp)],
        )


def _run(path: Path, header: str, command: list[str]) -> None:
    """Run *command*, its output and how it ended going to *path*."""
    with path.open("w", encoding="utf-8", errors="replace") as out:
        out.write(header + f"command: {subprocess.list2cmdline(command)}\n\n")
        out.flush()
        try:
            result = subprocess.run(
                command,
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=TOOL_TIMEOUT,
                check=False,
            )
            out.write(f"\nexit code: {result.returncode}\n")
        except (OSError, subprocess.SubprocessError) as error:
            out.write(f"\nfailed: {error!r}\n")
