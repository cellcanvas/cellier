"""Randomised traces against the scheduler core, checking design section 7.

A port of ``scripts/progressive/model_check_chunk_machine.py`` (V1, with the
class-first ``RECENT`` order) from the model to the real core.  The
environment owns a simulated clock and store: each read takes a jittered
latency and fails with a small probability, and each chunk has a data
version that invalidation bumps, so drawing stale data is detectable.
Actions are drawn at random -- passes (full, backstop-only drags, coalesced
bursts), reads landing, commit rounds (scene-scoped and fallback),
invalidation, retire, remove and re-register -- and every so often the trace
stops passing and runs to quiescence, which checks liveness.

Invariants checked (numbered as in design section 7):

1. slot conservation;
2. the in-flight budget and its lanes;
3. every eviction victim is ``RECENT`` and ranks strictly below the record
   taking its slot; 3b: never ``VISIBLE``; 3c: the least important
   ``RECENT`` resident goes;
4. nothing drawn is resident with pre-invalidation data;
5. liveness: quiescence completes every cache, once per generation;
6. once the backstop is resident, every cell it covers shows the current
   slice (every cell, for a full-extent backstop);
7. a key the latest pass does not want is only committed as ``RECENT``;
8. progress counts are consistent;
9. no target read starts while a ``VISIBLE`` backstop is queued anywhere;
11. the draw equals an independent painter's-order oracle (foreground on
    top; while incomplete, background slices oldest first; finer over
    coarser);
12. a retired cache issues nothing and has nothing ``QUEUED`` or
    ``VISIBLE``; a removed cache's reads are dropped;

plus: a good, non-stale read is kept as ``ARRIVED`` (12a in the model); a
kept arrival is discarded only with no free slot and no lower-ranked
``RECENT`` resident (12b); ``tier == VISIBLE`` exactly for the latest pass's
keys.

``CELLIER_SCHEDULER_STEPS`` scales the trace length (default 1500 steps per
seed, which runs in a few seconds).
"""

from __future__ import annotations

import heapq
import os
from collections import Counter

import numpy as np
import pytest

from cellier.render import SchedulerConfig
from cellier.render.scheduling import (
    ALL_SCENES,
    DEAD,
    SHARED_LANE,
    ChunkClass,
    ChunkState,
    ReadOutcome,
    ReadTicket,
    SchedulerCore,
    Tier,
)
from tests.render.scheduling._fakes import (
    BASE,
    N_LEVELS,
    FakeResidency,
    FakeStore,
    cells_of,
    desired,
    grid_dim,
    pack,
    span,
    unpack,
)

N_T = 4
N_CACHES = 3
STEPS = int(os.environ.get("CELLIER_SCHEDULER_STEPS", "1500"))

_RESIDENT = int(ChunkState.RESIDENT)
_FAILED = int(ChunkState.FAILED)
_QUEUED = int(ChunkState.QUEUED)
_FETCHING = int(ChunkState.FETCHING)
_ARRIVED = int(ChunkState.ARRIVED)
_VISIBLE = int(Tier.VISIBLE)
_RECENT = int(Tier.RECENT)
_BACKSTOP = int(ChunkClass.BACKSTOP)


def importance(snap: tuple[int, int, int, int, int]) -> tuple:
    """The eviction order from a ``(key, tier, cls, wanted_gen, rank)`` snapshot.

    Written from design 5.5, independently of the core: every ``VISIBLE``
    record outranks every ``RECENT`` one, and ``RECENT`` records compare by
    ``(cls, wanted_gen, -rank)``.
    """
    _, tier, cls, gen, rank = snap
    if tier == _VISIBLE:
        return (1, 0, 0, 0)
    return (0, cls, gen, -rank)


class Environment:
    """A simulated clock and store, random actions, and the invariant checks."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.clock = 0.0
        self.config = SchedulerConfig(
            max_in_flight=int(self.rng.integers(2, 12)),
            backstop_reserved=int(self.rng.integers(0, 4)),
            retry_max_attempts=int(self.rng.integers(1, 4)),
            retry_backoff_s=0.05,
        )
        self.core = SchedulerCore(self.config, now=lambda: self.clock)
        self.core.trace = self._on_trace
        self.core.on_complete = self._on_complete
        self.fail_p = float(self.rng.choice([0.0, 0.02, 0.1]))
        self.store = FakeStore()
        self.versions: Counter[int] = Counter()
        # (landing time, seq, ticket, version at issue)
        self.reads: list[tuple[float, int, ReadTicket, int]] = []
        self.seq = 0
        self.residency: dict[int, FakeResidency] = {}
        self.latest: dict[int, dict[int, int]] = {}  # cache -> key -> cls
        self.t_of: dict[int, int] = {}
        self.retired: set[int] = set()
        self.incarnation: Counter[int] = Counter()
        self.completions: Counter[tuple[int, int, int]] = Counter()
        self.backstop_mode = {
            c: str(self.rng.choice(("full", "view", "off"))) for c in range(N_CACHES)
        }
        self.stats: Counter[str] = Counter()
        for cache_id in range(N_CACHES):
            self._register(cache_id)

    # -- caches --------------------------------------------------------------

    def _register(self, cache_id: int) -> None:
        n_slots = int(self.rng.integers(8, 40))
        self.residency[cache_id] = FakeResidency(n_slots)
        self.core.register(cache_id, self.residency[cache_id], scene=cache_id % 2)
        self.latest[cache_id] = {}
        self.t_of.pop(cache_id, None)
        self.retired.discard(cache_id)
        self.incarnation[cache_id] += 1

    # -- the store -----------------------------------------------------------

    def issue(self) -> None:
        for ticket in self.core.next_reads():
            latency = float(self.rng.uniform(0.02, 0.15))
            self.seq += 1
            heapq.heappush(
                self.reads,
                (self.clock + latency, self.seq, ticket, self.versions[ticket.key]),
            )

    def advance(self, dt: float) -> None:
        """Move the clock and deliver every read that has landed."""
        self.clock += dt
        while self.reads and self.reads[0][0] <= self.clock:
            _, _, ticket, version = heapq.heappop(self.reads)
            failed = self.rng.random() < self.fail_p
            live = self.core.is_registered(ticket.cache_id) and (
                ticket.token is self.core._caches[ticket.cache_id].token
            )
            stale = False
            if live:
                reg = self.core.registry(ticket.cache_id)
                row = reg.find(ticket.key)
                stale = bool(reg.stale[row])
            outcome = self.core.complete_read(
                ticket,
                data=None if failed else version,
                error=OSError("injected") if failed else None,
            )
            self.stats[f"read_{outcome}"] += 1
            if not live:
                # 12: a removed cache's reads are dropped.
                assert outcome == ReadOutcome.DROPPED
            elif not failed and not stale:
                # A good read is kept, wanted or not.
                assert outcome == ReadOutcome.ARRIVED, outcome
        self.core.requeue_due()
        self.issue()

    # -- desired sets --------------------------------------------------------

    def plan(self, cache_id: int, backstop_only: bool) -> None:
        """A planner's output: backstop first, then the target nearest-first."""
        rng = self.rng
        mode = self.backstop_mode[cache_id]
        if mode == "off":
            backstop_only = False
        t = int(rng.integers(0, N_T))
        n_top = grid_dim(N_LEVELS)
        full = [(gy, gx) for gy in range(n_top) for gx in range(n_top)]
        cy, cx = rng.uniform(0, BASE, size=2)
        half = rng.uniform(1, BASE / 2 + 1)

        def cells(level: int) -> list[tuple[int, int]]:
            n, s = grid_dim(level), span(level)
            out = [
                (gy, gx)
                for gy in range(n)
                for gx in range(n)
                if gy * s < cy + half
                and (gy + 1) * s > cy - half
                and gx * s < cx + half
                and (gx + 1) * s > cx - half
            ]
            out.sort(
                key=lambda c: (
                    ((c[0] + 0.5) * s - cy) ** 2 + ((c[1] + 0.5) * s - cx) ** 2
                )
            )
            return out

        if mode == "full":
            backstop_cells = full
        elif mode == "view":
            backstop_cells = cells(N_LEVELS)
        else:
            backstop_cells = []
        backstop = [pack(N_LEVELS, t, gy, gx) for gy, gx in backstop_cells]
        target: list[int] = []
        if not backstop_only:
            level = int(rng.integers(1, N_LEVELS + 1))
            if not (level == N_LEVELS and mode != "off"):  # 5.3: dedup to backstop
                room = self.residency[cache_id].n_slots - 1 - len(backstop)
                target = [pack(level, t, gy, gx) for gy, gx in cells(level)[:room]]
        ds = desired(cache_id, backstop=backstop, target=target, store=self.store)
        self.core.set_desired(ds)
        self.latest[cache_id] = {int(k): int(c) for k, c in zip(ds.keys, ds.cls)}
        self.t_of[cache_id] = t
        self.retired.discard(cache_id)

    # -- actions -------------------------------------------------------------

    def act(self) -> str:
        r = self.rng.random()
        rng = self.rng
        if r < 0.15:
            self.plan(int(rng.integers(0, N_CACHES)), backstop_only=False)
            self._process()
            return "pass"
        if r < 0.22:
            self.plan(int(rng.integers(0, N_CACHES)), backstop_only=True)
            self._process()
            return "drag_pass"
        if r < 0.25:
            for _ in range(int(rng.integers(2, 5))):
                self.plan(
                    int(rng.integers(0, N_CACHES)), backstop_only=rng.random() < 0.5
                )
            self._process()
            return "coalesced_passes"
        if r < 0.55:
            self.advance(float(rng.uniform(0.0, 0.05)))
            return "advance"
        if r < 0.75:
            self.core.commit_round(int(rng.integers(0, 2)))  # one scene
            return "frame_round"
        if r < 0.90:
            self.core.commit_round(ALL_SCENES)
            return "fallback_round"
        if r < 0.95:
            self._invalidate()
            return "invalidate"
        if r < 0.98:
            cache_id = int(rng.integers(0, N_CACHES))
            self.core.retire(cache_id)
            self.latest[cache_id] = {}
            self.t_of.pop(cache_id, None)
            self.retired.add(cache_id)
            self._process()
            return "retire"
        cache_id = int(rng.integers(0, N_CACHES))
        self.core.remove(cache_id)
        self._register(cache_id)
        return "remove"

    def _process(self) -> None:
        self.core.process()
        self.issue()

    def _invalidate(self) -> None:
        """A region at every level for one ``t``, as paint or a live store."""
        t = int(self.rng.integers(0, N_T))
        y0, x0 = (int(v) for v in self.rng.integers(0, BASE, size=2))
        size = int(self.rng.integers(1, 6))

        def hit(key: int) -> bool:
            level, kt, gy, gx = unpack(key)
            s = span(level)
            return (
                kt == t
                and gy * s < y0 + size
                and (gy + 1) * s > y0
                and gx * s < x0 + size
                and (gx + 1) * s > x0
            )

        # The store's data changes first, then the scheduler hears of it.
        for level in range(1, N_LEVELS + 1):
            for gy in range(grid_dim(level)):
                for gx in range(grid_dim(level)):
                    key = pack(level, t, gy, gx)
                    if hit(key):
                        self.versions[key] += 1
        self.core.invalidate(self.store.id, hit)
        self.issue()

    def prefill(self, max_passes: int = 200) -> None:
        """Fill every cache's slots with residents, as after a long session."""
        for cache_id in range(N_CACHES):
            for _ in range(max_passes):
                if not self.core.free_slots(cache_id):
                    break
                self.plan(cache_id, backstop_only=False)
                self._process()
                for _ in range(20):
                    self.advance(0.2)
                    self.core.commit_round(ALL_SCENES)
                    if self.core.idle():
                        break
                self.check()
            full = not self.core.free_slots(cache_id)
            self.stats["prefill_full" if full else "prefill_not_full"] += 1

    def quiesce(self) -> None:
        """Run with no passes until idle, then check liveness (5)."""
        for _ in range(10_000):
            if self.core.idle():
                break
            self.advance(0.05)
            self.core.commit_round(ALL_SCENES)
            self.check()
        else:
            raise AssertionError("liveness: the scheduler never went idle")
        for cache_id in range(N_CACHES):
            assert self.core.is_complete(cache_id), f"cache {cache_id} incomplete"
            gen = self.core.generation(cache_id)
            if gen:
                key = (cache_id, self.incarnation[cache_id], gen)
                assert self.completions[key] == 1, (
                    f"completion fired {self.completions[key]}x for {key}"
                )
        self.check()

    # -- trace checks ----------------------------------------------------------

    def _on_complete(self, cache_id: int, gen: int) -> None:
        key = (cache_id, self.incarnation[cache_id], gen)
        self.completions[key] += 1
        assert self.completions[key] == 1, f"completion fired twice for {key}"
        assert gen == self.core.generation(cache_id)

    def _on_trace(self, kind: str, payload: tuple) -> None:
        self.stats[f"trace_{kind}"] += 1
        if kind == "issue":
            cache_id, key, cls, lane = payload
            # 12: a retired cache starts no read.
            assert cache_id not in self.retired, f"retired cache {cache_id} issued"
            if cls != _BACKSTOP:
                # 9: no target read while a VISIBLE backstop is queued.
                assert lane == SHARED_LANE
                for other in self.core.cache_ids:
                    reg = self.core.registry(other)
                    queued_backstop = (
                        (reg.state == _QUEUED)
                        & (reg.tier == _VISIBLE)
                        & (reg.cls == _BACKSTOP)
                    )
                    assert not queued_backstop.any(), (
                        f"target {key} issued ahead of a backstop in cache {other}"
                    )
        elif kind == "commit":
            cache_id, key, tier = payload
            # 7: only what the latest pass wants is committed as VISIBLE.
            assert (tier == _VISIBLE) == (key in self.latest[cache_id]), (
                f"cache {cache_id}: key {key} committed as tier {tier}"
            )
            if tier == _RECENT:
                self.stats["commit_kept"] += 1
            # 4: nothing stale is committed.
            residency = self.residency[cache_id]
            data = residency.slots[residency.writes[-1][0]][1]
            assert data == self.versions[key], f"stale commit of {key}"
        elif kind == "evict":
            cache_id, victim, candidate, recent = payload
            self.stats[f"evict_for_{Tier(candidate[1]).name}"] += 1
            # 3b: only RECENT residents are evicted.
            assert victim[1] == _RECENT, f"wanted key {victim[0]} evicted"
            # 3: only for something strictly more important.
            assert importance(victim) < importance(candidate), (victim, candidate)
            # 3c: the least important RECENT resident goes.
            assert importance(victim) == min(importance(r) for r in recent)
        elif kind == "discard":
            cache_id, arrival, n_free, recent = payload
            self.stats["discard_kept"] += 1
            # 12b: a kept arrival is discarded only with no room for it.
            assert arrival[1] == _RECENT, f"discarded wanted {arrival[0]}"
            assert n_free == 0, f"discarded {arrival[0]} with a free slot"
            lower = [r for r in recent if importance(r) < importance(arrival)]
            assert not lower, f"discarded {arrival[0]} over lower-ranked {lower[0]}"

    # -- state checks ----------------------------------------------------------

    def check(self) -> None:
        core = self.core
        # 2: the in-flight budget.
        shared, lane = core.in_flight
        assert 0 <= shared <= self.config.max_in_flight
        assert 0 <= lane <= self.config.backstop_reserved
        assert shared + lane == len(self.reads)
        for cache_id in core.cache_ids:
            self._check_cache(cache_id)

    def _check_cache(self, cache_id: int) -> None:
        core = self.core
        reg = core.registry(cache_id)
        residency = self.residency[cache_id]
        live = reg.state != DEAD
        # 1: slot conservation.
        free = core.free_slots(cache_id)
        resident = live & (reg.state == _RESIDENT)
        slots = reg.slot[resident].tolist()
        assert len(set(free)) == len(free)
        assert len(set(slots)) == len(slots)
        assert not set(free) & set(slots)
        assert len(free) + len(slots) == residency.n_slots
        assert (reg.slot[live & ~resident] == -1).all()
        # Records exist only for live work; VISIBLE exactly for the latest pass.
        wanted = self.latest[cache_id]
        keys = reg.key[live].tolist()
        assert len(set(keys)) == len(keys)
        for row in np.flatnonzero(live).tolist():
            key, state, tier = (
                int(reg.key[row]),
                int(reg.state[row]),
                int(reg.tier[row]),
            )
            assert (tier == _VISIBLE) == (key in wanted), (cache_id, key, tier)
            if state == _QUEUED:
                assert tier == _VISIBLE
            if state != _FETCHING:
                assert not reg.stale[row]
            if tier == _VISIBLE:
                assert int(reg.cls[row]) == wanted[key]
        # 12: a retired cache has nothing queued or wanted.
        if cache_id in self.retired:
            assert not (live & (reg.tier == _VISIBLE)).any()
        # 8: progress is consistent.
        p = core.progress(cache_id)
        assert p.needed_backstop + p.needed_target == len(wanted)
        assert p.resident_backstop <= p.needed_backstop
        assert p.resident_target <= p.needed_target
        assert p.in_flight == int((live & (reg.state == _FETCHING)).sum())
        # 11: the draw is exactly the painter's-order oracle.
        expected = self._expected_lut(cache_id)
        assert residency.lut == expected, f"cache {cache_id}: draw != oracle"
        # 4: everything drawn is resident and current.
        for cell, key in residency.lut.items():
            row = reg.find(key)
            assert row >= 0 and reg.state[row] == _RESIDENT, (cell, key)
            slot_key, data = residency.slots[int(reg.slot[row])]
            assert slot_key == key
            assert data == self.versions[key], f"stale draw of {key}"
        # 6: a complete backstop covers its cells with the current slice.
        t = self.t_of.get(cache_id)
        backstop = [k for k, c in wanted.items() if c == _BACKSTOP]
        if t is not None and backstop:
            rows = [reg.find(k) for k in backstop]
            if all(r >= 0 and reg.state[r] == _RESIDENT for r in rows):
                covered = {c for k in backstop for c in cells_of(k)}
                if self.backstop_mode[cache_id] == "full":
                    assert len(residency.lut) == BASE * BASE, "hole in the backstop"
                for cell in covered:
                    assert unpack(residency.lut[cell])[1] == t, (
                        f"cache {cache_id}: old slice drawn over a complete backstop"
                    )

    def _expected_lut(self, cache_id: int) -> dict[tuple[int, int], int]:
        """Design 5.8 as an oracle, from the registry columns directly."""
        reg = self.core.registry(cache_id)
        live = reg.state != DEAD
        visible = live & (reg.tier == _VISIBLE)
        done = (reg.state == _RESIDENT) | (
            (reg.state == _FAILED) & (reg.attempts >= self.config.retry_max_attempts)
        )
        background_on = not done[visible].all()
        rows = np.flatnonzero(live & (reg.state == _RESIDENT)).tolist()
        group_gen: dict[int, int] = {}
        for r in rows:
            if reg.tier[r] != _VISIBLE:
                sid = int(reg.slice_id[r])
                group_gen[sid] = max(group_gen.get(sid, -1), int(reg.wanted_gen[r]))
        drawn = [r for r in rows if reg.tier[r] == _VISIBLE or background_on]
        drawn.sort(
            key=lambda r: (
                reg.tier[r] == _VISIBLE,
                0 if reg.tier[r] == _VISIBLE else group_gen[int(reg.slice_id[r])],
                -unpack(int(reg.key[r]))[0],
            )
        )
        out: dict[tuple[int, int], int] = {}
        for r in drawn:
            for cell in cells_of(int(reg.key[r])):
                out[cell] = int(reg.key[r])
        return out


def _run(seed: int, steps: int, prefill: bool) -> Counter[str]:
    env = Environment(seed)
    if prefill:
        env.prefill()
    actions: Counter[str] = Counter()
    for step in range(steps):
        actions[env.act()] += 1
        env.check()
        if step % 300 == 299:
            env.quiesce()
    env.quiesce()
    return env.stats + actions


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_invariants_hold_on_random_traces(seed: int) -> None:
    stats = _run(seed, STEPS, prefill=False)
    # The trace exercised the machinery, not just the happy path.
    assert stats["trace_commit"] > 0
    assert stats["pass"] > 0


@pytest.mark.parametrize("seed", [4, 5, 6, 7])
def test_invariants_hold_from_a_full_atlas(seed: int) -> None:
    stats = _run(seed, STEPS, prefill=True)
    assert stats["prefill_full"] > 0
    # From a full atlas, kept arrivals must evict to be placed (V1/V2).
    assert stats["trace_evict"] > 0


def test_the_traces_reach_every_rule() -> None:
    """Across seeds, every decision the invariants guard actually happens."""
    total: Counter[str] = Counter()
    for seed in range(8):
        total += _run(seed, 500, prefill=seed % 2 == 1)
    for kind in (
        "trace_evict",
        "trace_discard",
        "commit_kept",
        "evict_for_VISIBLE",
        "evict_for_RECENT",
        f"read_{ReadOutcome.RETRY}",
        f"read_{ReadOutcome.DROPPED}",
        "invalidate",
        "retire",
        "remove",
    ):
        assert total[kind] > 0, f"no {kind} in the traces: {dict(total)}"
