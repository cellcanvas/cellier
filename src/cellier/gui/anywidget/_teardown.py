"""Releasing anywidget controls when a view goes away.

``ipywidgets`` registers every ``Widget`` in a **process-global** table at
construction and only ``Widget.close()`` removes it, so a widget that is merely
dropped stays alive -- holding its traits, its synced ``_esm``/``_css`` source
text, and every object it subscribed to -- for the lifetime of the kernel.  In
a notebook that means building a viewer, closing it, and building another
leaves the first one's controls behind, every time.

``Widget.close()`` alone is not enough, because a ``DOMWidget`` constructs
auxiliary widgets of its own -- ``layout``, and ``style`` where it has one --
which are registered the same way and are **not** closed with their owner.
Closing a viewer with a dozen controls therefore left a dozen ``Layout``
widgets behind even when every control was closed correctly.

Every cellier anywidget's ``close()`` goes through `close_aux_widgets`
for that reason.  ``Widget.close()`` is idempotent (it no-ops once the comm is
gone), so a widget reachable by more than one teardown path -- a control that
is both a tracked leaf and a child of a container, say -- is safe to close
twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipywidgets import Widget

_AUX_TRAITS = ("layout", "style")


def close_aux_widgets(widget: Widget) -> None:
    """Close the auxiliary widgets *widget* owns, before closing *widget*.

    Call this from a ``close()`` override, immediately before
    ``super().close()``.  It deliberately does **not** close *widget* itself,
    so the override keeps control of that ordering.
    """
    from ipywidgets import Widget as _Widget

    for name in _AUX_TRAITS:
        aux = getattr(widget, name, None)
        if isinstance(aux, _Widget):
            aux.close()
