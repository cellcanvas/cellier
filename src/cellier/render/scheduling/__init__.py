"""The chunk scheduler: progressive loading for multiscale visuals.

Design: ``plans/progressive_loading_design_v3.md``.  The core is visual-kind
agnostic: a visual plans a :class:`DesiredSet` per cache, and each cache is
a :class:`Residency` that stores arrived data and decides what to draw.
"""

from cellier.render.scheduling._core import (
    ALL_SCENES,
    BACKSTOP_LANE,
    SHARED_LANE,
    ReadOutcome,
    SchedulerCore,
    recent_importance,
    store_key,
)
from cellier.render.scheduling._registry import DEAD, CacheRegistry
from cellier.render.scheduling._scheduler import ChunkScheduler
from cellier.render.scheduling._types import (
    CacheProgress,
    ChunkClass,
    ChunkedVisual,
    ChunkState,
    DesiredSet,
    PlanMode,
    ReadTicket,
    RegistryView,
    Residency,
    Tier,
    is_chunked_visual,
)

__all__ = [
    "ALL_SCENES",
    "BACKSTOP_LANE",
    "DEAD",
    "SHARED_LANE",
    "CacheProgress",
    "CacheRegistry",
    "ChunkClass",
    "ChunkScheduler",
    "ChunkState",
    "ChunkedVisual",
    "DesiredSet",
    "PlanMode",
    "ReadOutcome",
    "ReadTicket",
    "RegistryView",
    "Residency",
    "SchedulerCore",
    "Tier",
    "is_chunked_visual",
    "recent_importance",
    "store_key",
]
