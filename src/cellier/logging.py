"""Structured debug logging for the Cellier v2 rendering pipeline.

Provides seven named loggers (perf, gpu, cache, slicer, scheduler, camera,
source_id)
and convenience functions to enable/disable them.  All log calls in
instrumented modules are guarded so they have zero cost when disabled.

Usage
-----
>>> from cellier.logging import enable_debug_logging
>>> enable_debug_logging()  # all categories, Rich colors
>>> enable_debug_logging(categories=("perf",))  # perf only
>>> enable_debug_logging(use_rich=False)  # plain StreamHandler
"""

from __future__ import annotations

import logging
import sys

_PERF_LOGGER = logging.getLogger("cellier.render.perf")
_GPU_LOGGER = logging.getLogger("cellier.render.gpu")
_CACHE_LOGGER = logging.getLogger("cellier.render.cache")
_SLICER_LOGGER = logging.getLogger("cellier.render.slicer")
# The chunk scheduler that loads multiscale visuals: passes, commit rounds,
# invalidation, failed reads, and the deferred reslices (dims settle,
# rate-capped store changes).
_SCHEDULER_LOGGER = logging.getLogger("cellier.render.scheduler")
_CAMERA_LOGGER = logging.getLogger("cellier.render.camera")
# Traces source_id injection through the ContextVar→psygnal→bus bridge.
# Enable this logger to see who triggered each model mutation, which
# bridge handler resolved the ContextVar, and whether it fell back to
# the controller's own ID.
_SOURCE_ID_LOGGER = logging.getLogger("cellier.render.source_id")

_CATEGORY_MAP = {
    "perf": _PERF_LOGGER,
    "gpu": _GPU_LOGGER,
    "cache": _CACHE_LOGGER,
    "slicer": _SLICER_LOGGER,
    "scheduler": _SCHEDULER_LOGGER,
    "camera": _CAMERA_LOGGER,
    "source_id": _SOURCE_ID_LOGGER,
}

_ALL_CATEGORIES = tuple(_CATEGORY_MAP.keys())

# Tag used to identify handlers added by enable_debug_logging().
_HANDLER_TAG = "_cellier_debug"


def enable_debug_logging(
    categories: tuple[str, ...] = _ALL_CATEGORIES,
    use_rich: bool = True,
    level: int = logging.DEBUG,
) -> None:
    """Activate logging for the requested categories.

    Parameters
    ----------
    categories :
        Which loggers to enable.  Defaults to all seven.
    use_rich :
        If ``True`` (default), attempt to use ``rich.logging.RichHandler``
        for colored output.  Falls back to a plain ``StreamHandler`` if
        Rich is not installed.
    level :
        Minimum log level to emit.  Defaults to ``logging.DEBUG``.
        Pass ``logging.INFO`` to suppress per-frame debug details and
        see only coarser events (reslice summaries, LUT rebuilds, etc.).
    """
    parent_logger = logging.getLogger("cellier.render")

    # Set the requested level on each category logger.
    for cat in categories:
        logger = _CATEGORY_MAP.get(cat)
        if logger is not None:
            logger.setLevel(level)

    # Attach a handler to the parent logger (once only).
    if any(getattr(h, _HANDLER_TAG, False) for h in parent_logger.handlers):
        print(
            f"[cellier] debug logging enabled: categories={set(categories)}",
            file=sys.stderr,
        )
        return

    handler: logging.Handler | None = None
    if use_rich:
        try:
            handler = _make_rich_handler()
        except ImportError:
            handler = None

    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(name)s  %(message)s", datefmt="%H:%M:%S")
        )

    handler.setLevel(level)
    setattr(handler, _HANDLER_TAG, True)
    parent_logger.addHandler(handler)
    parent_logger.setLevel(level)

    print(
        f"[cellier] debug logging enabled: categories={set(categories)}",
        file=sys.stderr,
    )


def disable_debug_logging() -> None:
    """Deactivate debug logging: reset levels to WARNING and remove handlers."""
    parent_logger = logging.getLogger("cellier.render")
    for logger in _CATEGORY_MAP.values():
        logger.setLevel(logging.WARNING)
    parent_logger.handlers = [
        h for h in parent_logger.handlers if not getattr(h, _HANDLER_TAG, False)
    ]


# ---------------------------------------------------------------------------
# Rich handler
# ---------------------------------------------------------------------------

_COLOR_MAP = {
    "cellier.render.perf": ("cyan", "PERF"),
    "cellier.render.gpu": ("green", "GPU"),
    "cellier.render.cache": ("yellow", "CACHE"),
    "cellier.render.slicer": ("magenta", "SLICER"),
    "cellier.render.camera": ("blue", "CAMERA"),
    "cellier.render.source_id": ("red", "SOURCE_ID"),
}


def _make_rich_handler() -> logging.Handler:
    """Create a ``CellierRichHandler`` (raises ``ImportError`` if Rich is absent)."""
    from rich.logging import RichHandler

    class CellierRichHandler(RichHandler):
        """RichHandler subclass that colors log records by category."""

        def emit(self, record: logging.LogRecord) -> None:
            color, label = _COLOR_MAP.get(record.name, ("white", "???"))
            record.msg = f"[{color}]\\[{label}][/{color}] {record.msg}"
            super().emit(record)

    return CellierRichHandler(
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
