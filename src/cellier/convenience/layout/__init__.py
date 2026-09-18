"""Declarative layout specification for cellier viewers.

The model layer of the layout system: host-agnostic dataclasses that describe
viewer layout without any rendering logic.  Pass a :class:`Layout` instance to
:func:`~cellier.convenience.display`, :func:`~cellier.convenience.launch`, or
:func:`~cellier.convenience.show` to render it.

Quick start::

    from cellier.convenience import display
    from cellier.convenience.layout import AppearanceControls, Layout

    layout = Layout(
        center=canvas_view,
        left_dock=AppearanceControls(),
    )
    display(viewer, layout)
"""

from cellier.convenience.layout._spec import (
    AppearanceControls,
    Grid,
    HStack,
    Layout,
    OverlayControls,
    RenderControls,
    VStack,
)

__all__ = [
    "AppearanceControls",
    "Grid",
    "HStack",
    "Layout",
    "OverlayControls",
    "RenderControls",
    "VStack",
]
