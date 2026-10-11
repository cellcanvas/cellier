"""Hash every decision the scheduler core makes on a random trace.

The trace guard of ``plans/mesh_refactor_v3.md`` 5.2: with default cache
policies the core must decide exactly as it did before it learned about
target caps and the compute lane.  A digest covers every ``trace`` call
(pass, issue, arrive, commit, evict, discard, invalidate), every action with
its clock, and the final read and completion counts, in order.

``trace_baseline.json`` holds the digests recorded from the core **before**
that change (``python -m tests.render.scheduling._trace`` wrote it, on the
unmodified tree).  It is not regenerated when the core changes: a difference
is a behaviour change for image and labels, and a stop condition.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from cellier.render.scheduling import ChunkClass

BASELINE = Path(__file__).parent / "trace_baseline.json"

#: ``ChunkClass.BACKSTOP`` as the baseline recorded it.  The class became a
#: priority with the backstop at 255
#: (``plans/progressive_intermediate_level_design_v2.md`` 5.1, 9.8), which
#: changes the integer in every digest and no decision.  Until the baseline
#: is regenerated (Phase 7 of that plan, on the user's go-ahead) the class is
#: translated back before it is hashed, so the guard keeps proving that
#: two-class behaviour is unchanged.
RECORDED_BACKSTOP = 1
_BACKSTOP = int(ChunkClass.BACKSTOP)


def _cls(value):
    """A class as the baseline recorded it."""
    if isinstance(value, np.ndarray):
        out = value.copy()
        out[out == _BACKSTOP] = RECORDED_BACKSTOP
        return out
    return RECORDED_BACKSTOP if int(value) == _BACKSTOP else value


def _snapshot(snap: tuple) -> tuple:
    """``(key, tier, cls, wanted_gen, rank)`` with the recorded class."""
    key, tier, cls, gen, rank = snap
    return (key, tier, _cls(cls), gen, rank)


def translate(kind: str, payload: tuple) -> tuple:
    """A trace event's payload with every class as the baseline recorded it.

    The class is in ``pass`` (the array), ``issue`` (the integer) and in
    the record snapshots of ``evict`` and ``discard``.  ``arrive``,
    ``commit`` and ``invalidate`` carry none.
    """
    if kind == "pass":
        cache_id, gen, keys, cls = payload
        return (cache_id, gen, keys, _cls(cls))
    if kind == "issue":
        cache_id, key, cls, lane = payload
        return (cache_id, key, _cls(cls), lane)
    if kind == "evict":
        cache_id, victim, candidate, recent = payload
        return (
            cache_id,
            _snapshot(victim),
            _snapshot(candidate),
            [_snapshot(s) for s in recent],
        )
    if kind == "discard":
        cache_id, row, n_free, recent = payload
        return (cache_id, _snapshot(row), n_free, [_snapshot(s) for s in recent])
    return payload


#: ``(seed, steps)`` of every recorded trace.  Odd seeds start from a full
#: atlas, as in ``test_property``.
CASES: tuple[tuple[int, int], ...] = (
    *((seed, 1500) for seed in range(8)),
    *((seed, 400) for seed in range(8, 40)),
)


def _feed(digest, obj) -> None:
    """Hash *obj* by value: arrays by their bytes, containers recursively."""
    if isinstance(obj, np.ndarray):
        digest.update(b"A")
        digest.update(str(obj.dtype).encode())
        digest.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, (tuple, list)):
        digest.update(b"(")
        for item in obj:
            _feed(digest, item)
        digest.update(b")")
    elif isinstance(obj, (np.integer, np.bool_)):
        digest.update(repr(int(obj)).encode())
    else:
        digest.update(repr(obj).encode())
    digest.update(b",")


def trace_digest(seed: int, steps: int) -> tuple[str, int]:
    """Run one random trace; return ``(digest, number of trace events)``."""
    from tests.render.scheduling import test_property as tp

    digest = hashlib.blake2b(digest_size=16)
    count = 0
    env = tp.Environment(seed)
    checks = env._on_trace

    def on_trace(kind: str, payload: tuple) -> None:
        nonlocal count
        count += 1
        _feed(digest, (kind, translate(kind, payload)))
        checks(kind, payload)

    env.core.trace = on_trace
    if seed % 2 == 1:
        env.prefill()
    # No ``env.check()`` per step: it asserts the invariants and changes
    # nothing the digest covers, and ``test_property`` runs it on the same
    # environment.  It was 80% of the guard's time.
    for step in range(steps):
        action = env.act()
        _feed(digest, ("act", action, round(env.clock, 9)))
        if step % 300 == 299:
            env.quiesce()
    env.quiesce()
    _feed(digest, sorted(env.stats.items()))
    _feed(digest, sorted(env.completions.items()))
    return digest.hexdigest(), count


def record() -> dict[str, dict]:
    """Digest every case, keyed ``"seed:steps"``."""
    logging.getLogger("cellier").setLevel(logging.CRITICAL)
    out: dict[str, dict] = {}
    for seed, steps in CASES:
        value, events = trace_digest(seed, steps)
        out[f"{seed}:{steps}"] = {"digest": value, "events": events}
    return out


if __name__ == "__main__":
    BASELINE.write_text(json.dumps(record(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {BASELINE}")
