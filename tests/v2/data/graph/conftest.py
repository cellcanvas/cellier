"""Shared builders for the graph store tests.

``_request`` is the single place a ``GraphSliceRequest`` is assembled, so the
tests read as "slice this graph on axis 0 at index 5 with a +/-3 window"
rather than as tuple plumbing.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.data.graph import GraphSliceRequest


@pytest.fixture
def make_request():
    """Return ``make(displayed, sliced, extents=None, fades=None)``."""

    def _make(
        displayed: tuple[int, ...],
        sliced: dict[int, int],
        extents: dict[int, tuple[float, float]] | None = None,
        fades: dict[int, tuple[float, float, float]] | None = None,
    ) -> GraphSliceRequest:
        shared = uuid4()
        return GraphSliceRequest(
            slice_request_id=shared,
            chunk_request_id=shared,
            scale_index=0,
            displayed_axes=displayed,
            slice_indices=dict(sliced),
            extents=dict(extents or {}),
            fades=dict(fades or {}),
        )

    return _make


@pytest.fixture
def tracking_lineage():
    """Return ``(positions, edges)`` for a small 4-D ``tzyx`` lineage.

    Five tracks x ten timepoints; every edge spans ``t`` -> ``t + 1``, which
    is the topology that renders nothing at all under a both-endpoints edge
    rule and is why D5 says either.
    """
    n_tracks, n_time = 5, 10
    positions = np.zeros((n_tracks * n_time, 4), dtype=np.float32)
    edges = []
    for track in range(n_tracks):
        for t in range(n_time):
            row = track * n_time + t
            positions[row] = (t, 10.0 + track, 20.0 + t, 30.0 + track)
            if t > 0:
                edges.append((row - 1, row))
    return positions, np.asarray(edges, dtype=np.int32)
