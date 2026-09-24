"""One cache's registry: a struct of numpy arrays sorted by key.

Design: ``plans/progressive_loading_design_v3.md`` section 5.1, and D1 in
``progressive_loading_findings.md``: a dict of per-chunk objects costs about
1.2 us a key, so records are rows of parallel arrays and a pass is a handful
of vectorised operations.

Deleting a row marks it with the ``DEAD`` state (a tombstone) rather than
copying every array.  Tombstones are compacted away at the next pass or
insertion, so a row index is stable between those, and nothing outside the
registry holds one across them: side tables key by the packed key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellier.render.scheduling._types import ChunkState, RegistryView, Tier

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The tombstone state of a deleted row.
DEAD: int = 255

#: Column name -> (dtype, value of a new row).  ``key`` is separate.
_COLUMNS: dict[str, tuple[type, object]] = {
    "state": (np.uint8, int(ChunkState.QUEUED)),
    "tier": (np.uint8, int(Tier.VISIBLE)),
    "cls": (np.uint8, 0),
    "rank": (np.int32, 0),
    "slot": (np.int32, -1),
    "slice_id": (np.int32, 0),
    "wanted_gen": (np.int32, 0),
    "attempts": (np.int16, 0),
    "retry_at": (np.float64, 0.0),
    "stale": (np.bool_, False),
}


class CacheRegistry:
    """The records of one cache, as parallel arrays sorted by ``key``.

    Attributes
    ----------
    key : np.ndarray
        ``int64``, sorted ascending, unique among live rows.
    state, tier, cls, rank, slot, slice_id, wanted_gen, attempts, retry_at,
    stale : np.ndarray
        One entry per row; see design 5.1.  ``state == DEAD`` marks a
        deleted row.
    """

    def __init__(self) -> None:
        self.key = np.empty(0, dtype=np.int64)
        for name, (dtype, _) in _COLUMNS.items():
            setattr(self, name, np.empty(0, dtype=dtype))
        self._n_dead = 0

    # -- size ----------------------------------------------------------------

    def __len__(self) -> int:
        """Live rows."""
        return len(self.key) - self._n_dead

    @property
    def live(self) -> np.ndarray:
        """Boolean mask of live rows."""
        return self.state != DEAD

    # -- lookup --------------------------------------------------------------

    def find(self, key: int) -> int:
        """Row of live *key*, or ``-1``."""
        i = int(np.searchsorted(self.key, key))
        if i < len(self.key) and self.key[i] == key and self.state[i] != DEAD:
            return i
        return -1

    def find_many(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows of *keys*, and which of them are live.

        Returns
        -------
        rows : np.ndarray
            ``int64`` row per key; meaningful only where *found*.
        found : np.ndarray
            Boolean mask: the key has a live row.
        """
        n = len(self.key)
        if n == 0:
            return np.zeros(len(keys), dtype=np.int64), np.zeros(len(keys), bool)
        rows = np.minimum(np.searchsorted(self.key, keys), n - 1)
        found = (self.key[rows] == keys) & (self.state[rows] != DEAD)
        return rows, found

    # -- mutation ------------------------------------------------------------

    def kill(self, rows: np.ndarray | int) -> None:
        """Delete *rows* (tombstones; compacted later)."""
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        if not len(rows):
            return
        rows = rows[self.state[rows] != DEAD]
        self.state[rows] = DEAD
        self.slot[rows] = -1
        self.stale[rows] = False
        self._n_dead += len(rows)

    def compact(self) -> None:
        """Drop tombstones.  Invalidates row indices."""
        if not self._n_dead:
            return
        keep = self.state != DEAD
        self.key = self.key[keep]
        for name in _COLUMNS:
            setattr(self, name, getattr(self, name)[keep])
        self._n_dead = 0

    def insert(self, keys: np.ndarray, values: Mapping[str, np.ndarray]) -> None:
        """Insert new rows for *keys*, which must not be live already.

        Compacts first, so the tombstone of a deleted key cannot duplicate it.

        Parameters
        ----------
        keys : np.ndarray
            ``int64`` keys, in any order.
        values : Mapping[str, np.ndarray]
            Per-key values for some columns; the rest take their defaults.
        """
        if not len(keys):
            return
        self.compact()
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        at = np.searchsorted(self.key, keys)
        self.key = np.insert(self.key, at, keys)
        for name, (dtype, default) in _COLUMNS.items():
            value = values.get(name)
            fill = default if value is None else np.asarray(value, dtype=dtype)[order]
            setattr(self, name, np.insert(getattr(self, name), at, fill))

    # -- views ---------------------------------------------------------------

    def view(self, generation: int, complete: bool) -> RegistryView:
        """A read-only view of the live rows (compacts first)."""
        self.compact()
        arrays = {name: getattr(self, name) for name in ("key", *_COLUMNS)}
        return RegistryView.from_arrays(arrays, generation, complete)
