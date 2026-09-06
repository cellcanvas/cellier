"""Four-panel grid widget builder for the cellier OrthoViewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.convenience._hosts import LayoutHost
    from cellier.convenience._ortho_viewer import OrthoViewer
    from cellier.gui._constants import GuiName

#: ``(row, column, panel key, header)`` for the 2x2 ortho layout.
#:
#: Read by every toolkit.  It used to be Qt-private, which is why Qt labelled
#: its panels ``XY`` / ``XZ`` / ``YZ`` / ``3D`` and the notebook labelled none
#: of them -- the same four-panel viewer, annotated in a script and bare in a
#: notebook (``plans/gui_backend_unification.md`` D15).
PANEL_LAYOUT: tuple[tuple[int, int, str, str], ...] = (
    (0, 0, "xy", "XY"),
    (0, 1, "xz", "XZ"),
    (1, 0, "yz", "YZ"),
    (1, 1, "vol", "3D"),
)


class OrthoCanvasGrid:
    """The four ortho panel canvas leaves, arranged 2x2 on any toolkit.

    ``compose(host)`` builds the grid through the host, so the arrangement --
    which panel sits where, and what each is called -- is written once and
    both front ends draw it the same way.

    Attributes
    ----------
    canvases : dict[str, object]
        Per-panel canvas leaves keyed ``"xy"``, ``"xz"``, ``"yz"``, ``"vol"``.
    """

    def __init__(self, canvases: dict) -> None:
        self.canvases = canvases
        self._widget = None

    def compose(self, host: LayoutHost) -> object:
        """Arrange the four panels as a labelled 2x2 grid via *host*."""
        rows: list[list[object]] = [[None, None], [None, None]]
        for row, column, key, header in PANEL_LAYOUT:
            rows[row][column] = host.stack(
                [self.canvases[key].compose(host)], direction="v", title=header
            )
        return host.grid(rows)

    @property
    def widget(self):
        """The composed Qt grid, for embedding in a hand-built Qt layout.

        Qt only, and a convenience: the layout system reaches the grid through
        :meth:`compose` like any other center leaf.
        """
        if self._widget is None:
            from cellier.convenience._hosts import QtLayoutHost

            self._widget = self.compose(QtLayoutHost())
        return self._widget

    def close(self) -> None:
        """Unsubscribe every panel's dims control from the bus."""
        for view in self.canvases.values():
            view.close()


#: Kept as the toolkit-specific names the two builders used to return.  They
#: are one class now, so an ``isinstance`` check against either still holds.
OrthoCanvasWidgets = OrthoCanvasGrid
OrthoAnywidgetCanvases = OrthoCanvasGrid


def build_ortho_grid_widget(
    ortho: OrthoViewer,
    axis_ranges: dict[int, tuple[float, float]],
    *,
    gui: GuiName | None = None,
    fov: float = 70.0,
    depth_range_3d: tuple[float, float] = (1.0, 8000.0),
    depth_range_2d: tuple[float, float] = (-500.0, 500.0),
    canvas_size: tuple[int, int] | None = None,
) -> OrthoCanvasGrid:
    """Build the 2x2 canvas grid for an :class:`OrthoViewer`.

    Creates (or reuses) a canvas per panel with wired dims sliders, and returns
    a leaf whose ``compose(host)`` lays them out as a labelled grid -- ``XY``
    and ``XZ`` on the top row, ``YZ`` and the ``3D`` volume on the bottom.

    Parameters
    ----------
    ortho : OrthoViewer
        The orthoviewer whose four scenes are attached.
    axis_ranges : dict[int, tuple[float, float]]
        Axis index to ``(world_min, world_max)`` for the slider ranges,
        typically from :func:`cellier.convenience.axis_ranges_from_ortho`.
    gui : "qt", "anywidget", or None
        Defaults to ``ortho.gui``; raises if it conflicts with it.
    fov : float
        Vertical field of view in degrees for the 3D camera.
    depth_range_3d, depth_range_2d : tuple[float, float]
        ``(near, far)`` clip distances for the 3D / 2D cameras.
    canvas_size : tuple[int, int] or None
        Initial CSS pixel size per panel.  Meaningful to anywidget only.

    Returns
    -------
    OrthoCanvasGrid

    Raises
    ------
    ValueError
        If *gui* conflicts with ``ortho.gui``, or names a front end with no
        widgets (``"offscreen"``).
    """
    from cellier.convenience._backend import backend_for
    from cellier.convenience.gui._canvas import (
        _ensure_qapplication,
        _resolve_gui,
        build_canvas_view,
    )

    gui = _resolve_gui(ortho, gui)
    backend = backend_for(
        gui, lacks="no embeddable widget, so no ortho grid can be built for it"
    )
    _ensure_qapplication(gui)

    return OrthoCanvasGrid(
        {
            key: build_canvas_view(
                ortho.controller,
                scene,
                axis_ranges,
                backend=backend,
                fov=fov,
                depth_range_3d=depth_range_3d,
                depth_range_2d=depth_range_2d,
                canvas_size=canvas_size,
            )
            for key, scene in ortho.scenes.items()
        }
    )
