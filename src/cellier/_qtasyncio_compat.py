"""PySide6 ``QtAsyncio`` compatibility patches.

PySide6's ``QAsyncioTask`` and ``QAsyncioEventLoop`` break when a read task is
cancelled inside ``asyncio.gather``: the task lacks a method ``gather`` calls,
and the loop's exception handler can crash on the report.  The two patches
below fix both.

Importing this module applies them; it is a no-op without PySide6.
``cellier.render.render_manager`` imports it, so the patches are in place
before any read task can be cancelled under a ``QAsyncioEventLoop``.
"""

from __future__ import annotations

import asyncio

# ---------------------------------------------------------------------------
# PySide6 compatibility patch
#
# QAsyncioTask is a from-scratch reimplementation of asyncio.Task that does
# not inherit from CPython's asyncio.Task.  As a result it is missing several
# CPython-internal methods.  One of these, _make_cancelled_error(), is called
# unconditionally by asyncio.gather's internal _GatheringFuture._done_callback
# when a child task is cancelled.  Without the method the cancellation raises:
#
#   AttributeError: 'QAsyncioTask' object has no attribute '_make_cancelled_error'
#
# The fix is to add the missing method.  The implementation matches CPython's
# exactly (asyncio/tasks.py): construct and return a CancelledError, optionally
# carrying the cancellation message stored on the task.
#
# The hasattr guard makes this a no-op if PySide6 ever adds the method itself.
# ---------------------------------------------------------------------------
try:
    from PySide6.QtAsyncio.events import QAsyncioEventLoop as _QAsyncioEventLoop
    from PySide6.QtAsyncio.tasks import QAsyncioTask as _QAsyncioTask

    # Patch 1: QAsyncioTask._make_cancelled_error
    # Called by asyncio.gather's _GatheringFuture._done_callback when a child
    # task is cancelled.  QAsyncioTask omits this CPython-internal method.
    if not hasattr(_QAsyncioTask, "_make_cancelled_error"):

        def _make_cancelled_error(self):  # type: ignore[misc]
            return asyncio.CancelledError(getattr(self, "_cancel_message", None))

        _QAsyncioTask._make_cancelled_error = _make_cancelled_error  # type: ignore[attr-defined]

    # Patch 2: QAsyncioEventLoop.default_exception_handler
    # When a cancelled task's CancelledError flows through QAsyncioTask._step,
    # the loop's default_exception_handler is called with a context dict that
    # may not contain a 'task' key.  Some PySide6 versions format the message
    # as f"... from task {context['task']._name}" — crashing with KeyError.
    # Additionally, asyncio.gather logs a spurious
    # "_GatheringFuture exception was never retrieved (CancelledError)"
    # message whenever a gather is cancelled — this is expected and harmless.
    # Both cases are handled here.
    _original_exc_handler = _QAsyncioEventLoop.default_exception_handler

    def _safe_default_exception_handler(  # type: ignore[misc]
        self, context: dict
    ) -> None:
        # Silently suppress CancelledError reports — these are always
        # expected when a read task is cancelled mid-gather and
        # produce no actionable information.
        exc = context.get("exception")
        if isinstance(exc, asyncio.CancelledError):
            return

        task = context.get("task")
        if task is None:
            # Fallback: call the original handler; if it crashes on 'task',
            # print a minimal message instead so real errors are visible.
            try:
                _original_exc_handler(self, context)
            except (KeyError, AttributeError):
                msg = context.get("message", "<no message>")
                print(
                    f"cellier: unhandled exception in event loop: {msg}"
                    + (f" ({exc!r})" if exc else "")
                )
        else:
            _original_exc_handler(self, context)

    _QAsyncioEventLoop.default_exception_handler = _safe_default_exception_handler  # type: ignore[attr-defined]

except ImportError:
    pass  # Not running under PySide6 / QtAsyncio — patches are not needed.
