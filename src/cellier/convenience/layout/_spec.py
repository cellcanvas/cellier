"""Host-agnostic layout specification (the model layer).

These dataclasses describe the structure of a viewer layout without any
host-specific rendering logic.  The renderer for each host (anywidget or Qt)
reads the spec and produces the appropriate widget tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

AppearancePresentation = Literal["selector", "collapsible_sections"]
"""How an :class:`AppearanceControls` dock presents several visuals."""


@dataclass
class HStack:
    """Horizontal stack of center items or control specs."""

    items: list


@dataclass
class VStack:
    """Vertical stack of center items or control specs."""

    items: list


@dataclass
class Grid:
    """2-D grid of center items.

    Parameters
    ----------
    cells : list[list]
        Row-major grid; ``None`` leaves a cell empty.
    """

    cells: list[list]


@dataclass
class AppearanceControls:
    """Dock spec: appearance controls for the viewer's configured visuals.

    The dock follows the viewer as visuals configured through ``controls=``
    are added and removed.

    Parameters
    ----------
    presentation : "selector" or "collapsible_sections"
        How the dock presents several visuals.  ``"selector"`` (default) shows
        one visual's controls at a time, with a selector above them once two
        or more visuals are configured.  ``"collapsible_sections"`` shows
        every configured visual at once, each in a collapsible section titled
        with its name.  The first section the dock builds starts expanded and
        every later one collapsed, so adding a visual never pushes open
        controls down the dock.
    """

    presentation: AppearancePresentation = "selector"

    def __post_init__(self) -> None:
        """Reject an unknown presentation here rather than at render time."""
        valid = get_args(AppearancePresentation)
        if self.presentation not in valid:
            raise ValueError(
                f"{self.presentation!r} is not a valid AppearanceControls "
                f"presentation. Valid presentations: {list(valid)}."
            )


@dataclass
class RenderControls:
    """Dock spec: one panel per renderer post-processing feature.

    Renders the Qt (``cellier.gui.qt.render``) or anywidget
    (``cellier.gui.anywidget.render``) panel for each named section, wired
    to the viewer's controller.  Unlike :class:`AppearanceControls` this
    needs no configured visual: render settings belong to the renderer, so
    the dock is available on any viewer.

    Parameters
    ----------
    sections : tuple[str, ...]
        Which panels to show, in order.  Any of ``"outline"``, ``"ambient_occlusion"``
        and ``"temporal"``.  Defaults to all three.
    """

    sections: tuple[str, ...] = ("ambient_occlusion", "outline", "temporal")


@dataclass
class Layout:
    """Full layout specification (the model).

    Describes what goes in the center region and each optional dock.  Pass to
    :func:`~cellier.convenience.display` (anywidget) or
    :func:`~cellier.convenience.launch` / :func:`~cellier.convenience.show`
    (Qt) to render.

    Parameters
    ----------
    center : canvas view or HStack or VStack or Grid
        Main content.  Typically a single ``AnywidgetCanvasView`` /
        ``QtCanvasWidget``, or a composed layout of multiple canvas views.
        The 2D/3D toggle is part of the dims control embedded in the canvas
        view, so it does not need a dock of its own.
    left_dock, right_dock, top_dock, bottom_dock :
        Content for each dock region.  Accepts :class:`AppearanceControls`,
        :class:`RenderControls`, or a stack of those.  ``None`` hides the dock.
    """

    center: object
    left_dock: object = None
    right_dock: object = None
    top_dock: object = None
    bottom_dock: object = None

    @classmethod
    def single(
        cls,
        canvas,
        *,
        appearance: Literal["left", "right", "top", "bottom"] | bool = False,
        render: Literal["left", "right", "top", "bottom"] | bool = False,
    ) -> Layout:
        """Single-canvas preset.

        Parameters
        ----------
        canvas :
            Canvas view returned by ``build_canvas_widget``.
        appearance : dock name or False
            Where to place appearance controls.  ``False`` (default) omits them.
        render : dock name or False
            Where to place the renderer settings panels (outlines, ambient
            occlusion, temporal accumulation).  ``False`` (default) omits
            them.

        Controls placed in the same dock stack top to bottom in the order
        appearance, render.
        """
        docks: dict[str, list] = {}
        for where, spec in (
            (appearance, AppearanceControls),
            (render, RenderControls),
        ):
            if where:
                docks.setdefault(f"{where}_dock", []).append(spec())
        return cls(
            center=canvas,
            **{
                name: specs[0] if len(specs) == 1 else VStack(items=specs)
                for name, specs in docks.items()
            },
        )
