"""With default policies the scheduler core decides as it did before them.

``plans/mesh_refactor_v3.md`` 5.2, "Guard for image and labels".  The core
gained per-cache policies (a cap on target reads, a compute lane, a failure
rule).  Image and labels atlases carry the default policy, and for them
nothing may change: not which read is issued, not its lane, not what a
commit round evicts.

``trace_baseline.json`` was recorded from the core before the change.  **A
failure here is a stop condition, not a baseline to regenerate.**

The baseline holds 40 traces.  All 40 were replayed while the policies were
being added; that work is done, and replaying them was 10 s of every CI job.
`GUARDED` is the part still replayed: an empty and a full starting atlas at
both trace lengths.  The baseline keeps every trace, so setting ``GUARDED =
CASES`` replays them all again, which is the thing to do before changing the
core's issue or eviction order.
"""

from __future__ import annotations

import json

import pytest

from tests.render.scheduling._trace import BASELINE, CASES, trace_digest

#: The recorded traces replayed on every run.  Odd seeds start from a full
#: atlas; seeds below 8 are 1500 steps long, the others 400.
GUARDED_SEEDS = (0, 1, 8, 9, 10, 11)
GUARDED = tuple(case for case in CASES if case[0] in GUARDED_SEEDS)


@pytest.fixture(scope="module")
def baseline() -> dict[str, dict]:
    return json.loads(BASELINE.read_text())


def test_every_case_has_a_recorded_digest(baseline) -> None:
    assert set(baseline) == {f"{seed}:{steps}" for seed, steps in CASES}
    assert len(GUARDED) == len(GUARDED_SEEDS)


@pytest.mark.parametrize(("seed", "steps"), GUARDED)
def test_the_trace_equals_the_one_recorded_before_policies(seed, steps, baseline):
    recorded = baseline[f"{seed}:{steps}"]
    digest, events = trace_digest(seed, steps)
    assert events == recorded["events"], (
        f"seed {seed}: the core made {events} decisions, {recorded['events']} "
        f"before the change"
    )
    assert digest == recorded["digest"], (
        f"seed {seed}: the core's decisions differ from the ones recorded "
        f"before cache policies.  This changes how image and labels load; "
        f"find the cause rather than re-recording."
    )
