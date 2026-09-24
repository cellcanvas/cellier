"""The chunk scheduler's contract with visuals and caches.

Design: ``plans/progressive_loading_design_v3.md`` section 4.  The scheduler
core knows keys, classes, ranks, tiers, states and slots, and nothing about
textures, LUTs or zarr.  A visual describes what it wants with one
:class:`DesiredSet` per cache, and each cache is a :class:`Residency` that
writes arrived data into a slot and decides what to draw.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class ChunkState(enum.IntEnum):
    """Where a record is in its life (design 5.2)."""

    QUEUED = 0
    FETCHING = 1
    ARRIVED = 2
    RESIDENT = 3
    FAILED = 4


class Tier(enum.IntEnum):
    """How much the latest pass wants a record; higher is more.

    ``PREFETCH`` is reserved for the prefetch follow-up and is never set.
    """

    RECENT = 0
    PREFETCH = 1
    VISIBLE = 2


class ChunkClass(enum.IntEnum):
    """The request class; higher is more important."""

    TARGET = 0
    BACKSTOP = 1


class PlanMode(enum.Enum):
    """What a planner should emit (design 5.10)."""

    FULL = "full"
    BACKSTOP_ONLY = "backstop_only"


@dataclass(frozen=True, slots=True, eq=False)
class DesiredSet:
    """What a planner wants for one cache, in load order.

    Parameters
    ----------
    cache_id : int
        The cache the keys belong to; registered with the scheduler.
    keys : np.ndarray
        ``int64``, unique, in load order: the backstop block first, then the
        target.  A key's position is its rank, which is its priority within
        its class.  At most ``n_slots - 1`` keys (design 5.3, truncation).
    cls : np.ndarray
        ``uint8`` :class:`ChunkClass` per key.
    slice_ids : np.ndarray
        ``int32`` interned collapsed selection per key, used to order the
        background by slice.
    build_request : Callable[[np.ndarray], Sequence[Any]]
        Keys to store requests, one per key, vectorised over a batch.
    store : Any
        Serves ``await store.get_data(request)``.  Its ``id`` names it to
        :meth:`ChunkScheduler.invalidate`.
    n_truncated_target : int
        Target keys the planner dropped to fit the cache (progress).
    n_truncated_backstop : int
        Backstop keys the planner dropped under its slot cap (progress).
    """

    cache_id: int
    keys: np.ndarray
    cls: np.ndarray
    slice_ids: np.ndarray
    build_request: Callable[[np.ndarray], Sequence[Any]]
    store: Any
    n_truncated_target: int = 0
    n_truncated_backstop: int = 0

    def __post_init__(self) -> None:
        """Coerce the arrays and check that they line up."""
        keys = np.ascontiguousarray(self.keys, dtype=np.int64)
        cls = np.ascontiguousarray(self.cls, dtype=np.uint8)
        slice_ids = np.ascontiguousarray(self.slice_ids, dtype=np.int32)
        if keys.ndim != 1 or cls.shape != keys.shape or slice_ids.shape != keys.shape:
            raise ValueError(
                "DesiredSet keys, cls and slice_ids must be 1-D and the same "
                f"length; got {keys.shape}, {cls.shape} and {slice_ids.shape}."
            )
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "cls", cls)
        object.__setattr__(self, "slice_ids", slice_ids)


@runtime_checkable
class Residency(Protocol):
    """What a cache provides to the scheduler (design 4.4).

    Slots are numbered ``0 .. n_slots - 1``; the adapter maps them onto its
    own storage (an atlas may reserve a slot of its own for "empty").
    """

    n_slots: int

    def write(self, slot: int, key: int, data: Any) -> None:
        """Upload *data* for *key* into *slot*."""
        ...

    def rebuild_draw(self, view: RegistryView) -> None:
        """Rebuild what is drawn from the cache's registry."""
        ...

    def keys_in_region(self, keys: np.ndarray, region: Any) -> np.ndarray:
        """Boolean mask: which *keys* intersect *region* (invalidation)."""
        ...


@runtime_checkable
class ChunkedVisual(Protocol):
    """A visual whose data the scheduler loads (design 4.2).

    Implementations also set the class attribute ``chunked = True``
    (:func:`is_chunked_visual`).
    """

    def plan(self, request: Any, config: Any, mode: PlanMode) -> list[DesiredSet]:
        """Return one desired set per drawn cache."""
        ...

    def residencies(self) -> Mapping[int, Residency]:
        """``cache_id -> adapter`` for every cache the visual owns."""
        ...


def is_chunked_visual(visual: object) -> bool:
    """Whether *visual* loads through the chunk scheduler (``ChunkedVisual``).

    Read off the class (``chunked = True``), so a mock or a visual that
    happens to have a ``plan`` attribute is not routed by accident.
    """
    return getattr(type(visual), "chunked", False) is True


def _readonly(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True, slots=True, eq=False)
class RegistryView:
    """A read-only view of one cache's registry, for ``rebuild_draw``.

    Arrays are parallel, one entry per live record, sorted by key.  They are
    views of the registry, not copies: read them during the call and do not
    keep them.

    Attributes
    ----------
    key, state, tier, cls, rank, slot, slice_id, wanted_gen : np.ndarray
        The registry columns (design 5.1).  ``slot`` is -1 when the record
        holds none.
    generation : int
        The cache's current pass generation.
    complete : bool
        Every ``VISIBLE`` record is resident or given up.
    """

    key: np.ndarray
    state: np.ndarray
    tier: np.ndarray
    cls: np.ndarray
    rank: np.ndarray
    slot: np.ndarray
    slice_id: np.ndarray
    wanted_gen: np.ndarray
    generation: int
    complete: bool

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, np.ndarray], generation: int, complete: bool
    ) -> RegistryView:
        """Wrap *arrays* in read-only views."""
        names = ("key", "state", "tier", "cls", "rank", "slot", "slice_id")
        columns = {name: _readonly(arrays[name]) for name in (*names, "wanted_gen")}
        return cls(**columns, generation=generation, complete=complete)

    def paint_groups(self) -> list[np.ndarray]:
        """Resident record indices, grouped in painter's order (design 5.8).

        Background first, only while the ``VISIBLE`` set is incomplete, then
        the foreground: every ``VISIBLE`` resident.

        The background is grouped by *slice generation*: each record takes
        the newest ``wanted_gen`` among the residents sharing its slice id,
        and records with the same value form one group, oldest first.  A
        pass has one slice position, so a group is one earlier view; its
        slice ids differ only where a planner interns a per-level selection,
        and grouping those together keeps their levels in one group.  Within
        a group the order is by key; a mosaic adapter sorts each group
        coarsest to finest itself.

        Returns
        -------
        list[np.ndarray]
            Index arrays into this view.  The last is the foreground (it may
            be empty).
        """
        resident = self.state == ChunkState.RESIDENT
        foreground = np.flatnonzero(resident & (self.tier == Tier.VISIBLE))
        if self.complete:
            return [foreground]
        background = np.flatnonzero(resident & (self.tier != Tier.VISIBLE))
        if not len(background):
            return [foreground]
        _, inverse = np.unique(self.slice_id[background], return_inverse=True)
        newest = np.full(int(inverse.max()) + 1, np.iinfo(np.int64).min, np.int64)
        np.maximum.at(newest, inverse, self.wanted_gen[background])
        group_gen = newest[inverse]
        groups = [background[group_gen == g] for g in np.unique(group_gen).tolist()]
        groups.append(foreground)
        return groups


@dataclass(slots=True)
class CacheProgress:
    """Loading progress for one cache (design 5.13).

    Attributes
    ----------
    needed_backstop, resident_backstop : int
        ``VISIBLE`` backstop records, and how many are resident.
    needed_target, resident_target : int
        The same for the target class.
    in_flight : int
        Reads outstanding for this cache, any tier.
    failed : int
        ``VISIBLE`` records given up after ``retry_max_attempts``.
    truncated_target, truncated_backstop : int
        Keys the planner dropped to fit, from the latest desired set.
    """

    needed_backstop: int = 0
    resident_backstop: int = 0
    needed_target: int = 0
    resident_target: int = 0
    in_flight: int = 0
    failed: int = 0
    truncated_target: int = 0
    truncated_backstop: int = 0

    def __add__(self, other: CacheProgress) -> CacheProgress:
        """Sum two caches' counts, as a visual does over its caches."""
        return CacheProgress(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__slots__
            }
        )


@dataclass(slots=True, eq=False)
class ReadTicket:
    """One issued read: what to fetch and where the answer goes.

    Attributes
    ----------
    cache_id : int
        The cache the read is for.
    key : int
        The record's key.
    request : Any
        The store request built for the key.
    store : Any
        The store to read from.
    lane : int
        The capacity the read holds: ``0`` shared, ``1`` the backstop lane.
    token : object
        The cache incarnation that issued it; a read for a removed (or
        removed and re-registered) cache is dropped when it lands.
    """

    cache_id: int
    key: int
    request: Any
    store: Any
    lane: int
    token: object = field(repr=False)
