"""Record the golden baseline JSON from the current (unmodified) tree.

Run as ``.venv/bin/python -m tests.integration._generate_baseline``.  Writes
one file per family under ``tests/integration/baseline/``.  The replay test
(``test_golden_baseline.py``) asserts against exactly these files, so this
script is only re-run when a later phase deliberately changes a recorded
value and the readout says so.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tests.integration._harness import FAMILIES, iter_cases, run_case

BASELINE_DIR = Path(__file__).parent / "baseline"


def build() -> dict[str, dict]:
    """Return ``{family_name: <json blob>}`` for every family in the matrix."""
    per_family: dict[str, dict] = {
        f.name: {
            "family": f.name,
            "kind": f.kind,
            "drivable": {"node_matrix": None, "slice_request": None},
            "cases": {},
        }
        for f in FAMILIES
    }

    node_ok: dict[str, bool] = defaultdict(bool)
    node_seen: dict[str, bool] = defaultdict(bool)
    slice_ok: dict[str, bool] = defaultdict(bool)

    for case in iter_cases():
        result = run_case(case)
        entry = result.as_json()
        entry["transform"] = case.tspec.name
        entry["dims"] = case.dspec.name
        if case.pyramid is not None:
            entry["pyramid"] = case.pyramid.name
        if case.is_xfail:
            entry["xfail_reason"] = case.xfail_reason
        per_family[case.family.name]["cases"][case.case_id] = entry

        if result.status == "ok":
            node_seen[case.family.name] = True
            if result.node_local_matrix is not None:
                node_ok[case.family.name] = True
            if result.selections is not None:
                slice_ok[case.family.name] = True

    for name, blob in per_family.items():
        blob["drivable"]["node_matrix"] = bool(node_ok[name])
        blob["drivable"]["slice_request"] = bool(slice_ok[name])
        any_ok = any(c["status"] == "ok" for c in blob["cases"].values())
        if not any_ok:
            first = next(iter(blob["cases"].values()), {})
            blob["drivable"]["blocker"] = first.get("status", "unknown")

    return per_family


def write() -> list[Path]:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, blob in build().items():
        path = BASELINE_DIR / f"{name}.json"
        path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in write():
        print(f"wrote {path}")
