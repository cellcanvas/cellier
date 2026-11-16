"""Command history primitives for paint controllers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from uuid import UUID


@dataclass(frozen=True)
class PaintStrokeCommand:
    """Immutable record of one completed brush stroke.

    Attributes
    ----------
    visual_id : UUID
    data_store_id : UUID
    voxel_indices : np.ndarray
        Shape ``(N, ndim)``, int64.  Each row is one voxel index in
        data-array order (e.g. ``(z, y, x)`` for a 3-D store).
    old_values : np.ndarray
        Shape ``(N,)``, in the store's dtype.  Value at each voxel before
        the stroke.
    new_values : np.ndarray
        Shape ``(N,)``, in the store's dtype.  Value written by the stroke.
    """

    visual_id: UUID
    data_store_id: UUID
    voxel_indices: np.ndarray
    old_values: np.ndarray
    new_values: np.ndarray


class ActiveStroke:
    """Accumulator that builds a `PaintStrokeCommand` during a drag.

    ``record`` is called once per brush application.  Successive calls
    concatenate their voxel arrays.  ``finalise`` constructs the
    immutable `PaintStrokeCommand`.  After ``finalise``, no further
    ``record`` calls are permitted.

    Parameters
    ----------
    visual_id : UUID
        The painted visual.
    data_store_id : UUID
        The store the stroke writes to.
    gesture_id : UUID or None
        The gesture this stroke belongs to, if known.
    dtype : numpy dtype or None
        The dtype the values are kept in; pass the store's, so undo writes
        back exactly what was there (label ids above 2**24 do not survive
        a float32 round trip).  None keeps each ``record`` call's own
        dtype, and float32 for a stroke with no voxels.
    """

    def __init__(
        self,
        visual_id: UUID,
        data_store_id: UUID,
        gesture_id: UUID | None = None,
        dtype: np.typing.DTypeLike | None = None,
    ) -> None:
        self._visual_id = visual_id
        self._data_store_id = data_store_id
        self._gesture_id = gesture_id
        self._dtype = None if dtype is None else np.dtype(dtype)
        self._voxel_chunks: list[np.ndarray] = []
        self._old_chunks: list[np.ndarray] = []
        self._new_chunks: list[np.ndarray] = []
        self._finalised = False

    @property
    def gesture_id(self) -> UUID | None:
        """The gesture this stroke belongs to, if known."""
        return self._gesture_id

    def record(
        self,
        voxel_indices: np.ndarray,
        old_values: np.ndarray,
        new_values: np.ndarray,
    ) -> None:
        """Append the voxels painted in one brush application."""
        if self._finalised:
            raise RuntimeError("Cannot record on a finalised ActiveStroke.")
        self._voxel_chunks.append(np.asarray(voxel_indices, dtype=np.int64))
        self._old_chunks.append(np.asarray(old_values, dtype=self._dtype))
        self._new_chunks.append(np.asarray(new_values, dtype=self._dtype))

    def finalise(self) -> PaintStrokeCommand:
        """Return the completed command. Do not call ``record`` after this.

        Successive ``record`` calls during a drag often overlap.  For each
        unique voxel we keep the **first** ``old_values`` seen — that is
        the true pre-stroke value.  Without this dedup, an undo replays
        stale "old" values captured after earlier in-stroke writes,
        leaving residue.  ``new_values`` are also taken from the first
        occurrence (within one stroke they are constant anyway).
        """
        self._finalised = True
        if self._voxel_chunks:
            voxel_indices_all = np.concatenate(self._voxel_chunks, axis=0)
            old_values_all = np.concatenate(self._old_chunks, axis=0)
            new_values_all = np.concatenate(self._new_chunks, axis=0)
            voxel_indices, first_idx = np.unique(
                voxel_indices_all, axis=0, return_index=True
            )
            old_values = old_values_all[first_idx]
            new_values = new_values_all[first_idx]
        else:
            voxel_indices = np.zeros((0, 0), dtype=np.int64)
            empty_dtype = np.float32 if self._dtype is None else self._dtype
            old_values = np.zeros((0,), dtype=empty_dtype)
            new_values = np.zeros((0,), dtype=empty_dtype)
        return PaintStrokeCommand(
            visual_id=self._visual_id,
            data_store_id=self._data_store_id,
            voxel_indices=voxel_indices,
            old_values=old_values,
            new_values=new_values,
        )


class CommandHistory:
    """Bounded deque of `PaintStrokeCommand` with undo/redo pointers.

    Only the undo stack is bounded by ``max_depth``; the redo stack is
    unbounded because it is cleared by ``push`` before it can grow
    indefinitely.
    """

    def __init__(self, max_depth: int = 100) -> None:
        self._undo_stack: deque[PaintStrokeCommand] = deque(maxlen=max_depth)
        self._redo_stack: deque[PaintStrokeCommand] = deque()

    @property
    def can_undo(self) -> bool:
        """True if there is at least one command on the undo stack."""
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """True if there is at least one command on the redo stack."""
        return bool(self._redo_stack)

    def push(self, command: PaintStrokeCommand) -> None:
        """Push a new command; clears the redo stack."""
        self._undo_stack.append(command)
        self._redo_stack.clear()

    def undo(self) -> PaintStrokeCommand | None:
        """Pop from the undo stack and push to the redo stack."""
        if not self._undo_stack:
            return None
        command = self._undo_stack.pop()
        self._redo_stack.append(command)
        return command

    def redo(self) -> PaintStrokeCommand | None:
        """Pop from the redo stack and push to the undo stack."""
        if not self._redo_stack:
            return None
        command = self._redo_stack.pop()
        self._undo_stack.append(command)
        return command
