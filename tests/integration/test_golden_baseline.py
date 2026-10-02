"""Replay the golden baseline against the current tree.

This suite is the regression net that Phases 3-6 of
``plans/world_space_slicing/transform_integration_implementation.md`` gate
against.  It re-drives every ``(family, transform, dims)`` cell of the
matrix in :mod:`tests.integration._harness` and asserts the node matrix
and the datastore selections are byte-for-byte what
``baseline/<family>.json`` recorded.

**It must pass on the unmodified tree.**  The JSON stores what the code
produces *today* -- not what the design predicts.  When a later phase
deliberately changes one of the three behaviours design section 3.6
names (``_harness.XFAIL_GROUPS``), the affected cases here start failing;
that phase re-records the JSON for exactly those cases and notes the flip
in its readout.  ``test_known_behaviour_changes_are_registered`` keeps the
list of those cases visible so the flip is never silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration._harness import FAMILIES, Case, iter_cases, run_case

BASELINE_DIR = Path(__file__).parent / "baseline"

_ALL_CASES: list[Case] = list(iter_cases())


def _load(family_name: str) -> dict:
    return json.loads((BASELINE_DIR / f"{family_name}.json").read_text())


def _case_id(case: Case) -> str:
    return f"{case.family.name}::{case.case_id}"


@pytest.fixture(scope="module")
def baselines() -> dict[str, dict]:
    return {f.name: _load(f.name) for f in FAMILIES}


def test_every_family_has_a_baseline_file():
    missing = [
        f.name for f in FAMILIES if not (BASELINE_DIR / f"{f.name}.json").exists()
    ]
    assert not missing, f"no baseline recorded for: {missing}"


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
def test_replay_matches_baseline(case: Case, baselines: dict[str, dict]):
    blob = baselines[case.family.name]
    assert case.case_id in blob["cases"], (
        f"{case.case_id} is not in {case.family.name}.json -- regenerate the "
        f"baseline with `.venv/bin/python -m tests.integration._generate_baseline`"
    )
    recorded = blob["cases"][case.case_id]
    result = run_case(case).as_json()

    # Compare only the fields that are behaviour, not bookkeeping.
    for field in (
        "status",
        "node_local_matrix",
        "selections",
        "scale_indices",
        "geometry_indices",
    ):
        assert result[field] == recorded[field], (
            f"{_case_id(case)}: field {field!r} diverged from the golden "
            f"baseline.\n  baseline: {recorded[field]!r}\n  current:  "
            f"{result[field]!r}\n"
            + (
                "This case is a registered behaviour change "
                f"({recorded.get('xfail_reason')}). If the owning phase is "
                "flipping it deliberately, re-record the baseline for this "
                "family and update its readout."
                if recorded.get("xfail_reason")
                else "This case is NOT a registered behaviour change -- design "
                "3.6 marks only three sites as behaviour-changing and "
                "everything else must match bit for bit."
            )
        )


def test_known_behaviour_changes_are_registered():
    """The three design-3.6 flips are enumerated, named, and currently green.

    Each of these cases matches today (the JSON holds today's value); the
    phase that owns the flip will see it fail here, re-record, and drop the
    case from nowhere -- the registry entry stays as the audit trail.
    """
    flips = [c for c in _ALL_CASES if c.is_xfail]
    assert flips, "expected the geometry-slicing / thickness cases to be marked"

    by_family: dict[str, int] = {}
    for case in flips:
        by_family[case.family.name] = by_family.get(case.family.name, 0) + 1
        assert case.xfail_reason  # a human-readable why

    # points, lines, graph and mesh each contribute non-identity cases
    # (design 3.6) and, since the thickness floor went, identity ones too;
    # in-memory labels changed when they took the one-plane slicing rule.
    assert set(by_family) == {
        "GFXPointsMemoryVisual",
        "GFXLinesMemoryVisual",
        "GFXGraphMemoryVisual",
        "GFXMeshMemoryVisual",
        "GFXLabelMemoryVisual",
    }
