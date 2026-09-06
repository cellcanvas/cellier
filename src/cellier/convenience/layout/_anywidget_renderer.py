"""Anywidget renderer -- the view layer for Layout specs on anywidget hosts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.convenience.layout._walk import render_layout

if TYPE_CHECKING:
    from cellier.convenience._hosts import LayoutHost
    from cellier.convenience.layout._spec import Layout


class _RenderView:
    """Rendered view with teardown tracking.

    Holds the composed root widget and every closeable object built during
    rendering.  ``close()`` tears them all down idempotently.
    """

    def __init__(self, root: object, closeables: list) -> None:
        self.root = root
        self._closeables = closeables
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # The leaves first, so each control emits ``closed`` and the controller
        # drops its subscriptions, then the root -- which closes the container
        # widgets the host built to compose them.  Those are widgets too, and
        # nothing else holds them, so skipping the root leaves the whole
        # scaffolding registered with ``ipywidgets``.
        for obj in [*self._closeables, self.root]:
            close = getattr(obj, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:
                pass


def render_anywidget(layout: Layout, viewer: object, host: LayoutHost) -> _RenderView:
    """Render a Layout spec to an anywidget host.

    A wrapper: the walk is shared with every other backend
    (``convenience.layout._walk.render_layout``).  The ``_RenderView`` is what
    differs -- on this toolkit teardown is the caller's, through the handle
    ``display()`` returns, rather than the root window's.
    """
    view = render_layout(layout, viewer, host)
    return _RenderView(view.root, view.closeables)
