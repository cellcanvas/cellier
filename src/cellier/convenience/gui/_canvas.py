"""Canvas widget builder for the cellier Viewer convenience layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.controller import CellierController
    from cellier.convenience._hosts import LayoutHost
    from cellier.convenience._viewer import Viewer
    from cellier.gui._constants import GuiName
    from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel
    from cellier.gui.qt import QtCanvasWidget
    from cellier.scene.scene import Scene


def _resolve_gui(viewer: Viewer, gui: str | None) -> str:
    """Resolve the requested *gui*, defaulting to and validating against viewer.gui."""
    if gui is None:
        return viewer.gui
    if gui != viewer.gui:
        raise ValueError(
            f"gui={gui!r} conflicts with viewer.gui={viewer.gui!r}. "
            "Build the canvas with the same gui the viewer was created with."
        )
    return gui


@dataclass
class AnywidgetCanvasView:
    """An anywidget canvas leaf plus its dims panel leaf.

    Returned by :func:`build_canvas_widget` for ``gui="anywidget"``.  Composes
    to a single column: canvas above the dims sliders.

    Appearance controls are **not** part of this view.  An earlier design put
    them in a left column here as well as in a dock; nothing ever reached that
    path -- ``build_canvas_widget`` never forwarded the parameters it needed --
    and ``Layout(left_dock=AppearanceControls())`` already produces the same
    ``[controls | canvas]`` arrangement through the supported path, on both
    toolkits (design section 7.2).

    Attributes
    ----------
    canvas : object
        The ``rendercanvas`` anywidget canvas.
    dims : AnywidgetDimsPanel
        Axis slice sliders plus the 2D/3D toggle button, below the canvas.
    canvas_size : tuple[int, int]
        The CSS pixel size the canvas was constructed with.  Reused by
        :meth:`compose` as the column's responsive min-width floor.
    """

    canvas: object
    dims: AnywidgetDimsPanel
    canvas_size: tuple[int, int] = (600, 600)

    @property
    def dims_control(self) -> AnywidgetDimsPanel:
        """The bus-facing dims widget, under the name every leaf uses.

        ``QtCanvasWidget`` spells it ``dims_control``; this spells it ``dims``.
        One name lets the shared canvas builder wire either without asking
        which toolkit it is holding.
        """
        return self.dims

    def compose(self, host: LayoutHost) -> object:
        """Arrange canvas above dims, floored at ``canvas_size[0]``."""
        return host.stack(
            [host.leaf(self.canvas), host.leaf(self.dims)],
            min_width=self.canvas_size[0],
        )

    def close(self) -> None:
        """Unsubscribe the dims panel from the bus."""
        self.dims.close()


def build_canvas_view(
    controller: CellierController,
    scene: Scene,
    axis_ranges: dict[int, tuple[float, float]],
    *,
    backend,
    render_modes: set[str] | None = None,
    initial_dim: str | None = None,
    fov: float = 70.0,
    depth_range_3d: tuple[float, float] = (1.0, 8000.0),
    depth_range_2d: tuple[float, float] = (-500.0, 500.0),
    canvas_size: tuple[int, int] | None = None,
    non_displayed: tuple[int, ...] = (),
):
    """Build a wired canvas leaf for *scene*, reusing any existing canvas.

    One builder for every toolkit.  If *scene* already has a canvas -- one
    restored by ``from_file``, say -- that canvas and its camera state are
    reused; otherwise a new one is created.  *backend* decides which widget
    wraps it, and the dims control is wired to the bus here so the wiring is
    stated once rather than once per toolkit.

    Parameters
    ----------
    controller : CellierController
        The controller owning *scene*.
    scene : Scene
        The scene whose canvas this leaf controls.
    axis_ranges : dict[int, tuple[float, float]]
        Axis index to ``(world_min, world_max)``, for the slider ranges.
    backend : GuiBackend
        Supplies the toolkit's canvas widget.
    render_modes : set[str] or None
        Camera modes to prepare on a new canvas.  Defaults to the scene's own.
        Ignored when reusing an existing canvas, as are the arguments below.
    initial_dim : str or None
        Active mode for a new canvas.  Inferred from the scene when ``None``.
    fov : float
        Vertical field of view in degrees for a new 3D camera.
    depth_range_3d, depth_range_2d : tuple[float, float]
        ``(near, far)`` clip distances for a new 3D / 2D camera.
    canvas_size : tuple[int, int] or None
        Initial CSS pixel size.  Meaningful to the anywidget backend only.
    non_displayed : tuple[int, ...]
        Axes to exclude from the sliders regardless of dims state.
    """
    canvas_ids = controller.get_canvas_ids(scene.id)
    if not canvas_ids:
        controller.add_canvas(
            scene.id,
            render_modes=set(scene.render_modes)
            if render_modes is None
            else render_modes,
            initial_dim=initial_dim,
            fov=fov,
            depth_range_3d=depth_range_3d,
            depth_range_2d=depth_range_2d,
            canvas_size=canvas_size,
        )
        canvas_ids = controller.get_canvas_ids(scene.id)

    view = backend.canvas_view(
        scene,
        controller.get_canvas_view(canvas_ids[-1]),
        axis_ranges,
        canvas_size=canvas_size,
        non_displayed=non_displayed,
    )
    controller.connect_widget(
        view.dims_control,
        subscription_specs=view.dims_control.subscription_specs(),
    )
    return view


def build_canvas_widget(
    viewer: Viewer,
    axis_ranges: dict[int, tuple[float, float]],
    *,
    gui: GuiName | None = None,
    render_modes: set[str] | None = None,
    initial_dim: str | None = None,
    fov: float = 70.0,
    depth_range_3d: tuple[float, float] = (1.0, 8000.0),
    depth_range_2d: tuple[float, float] = (-500.0, 500.0),
    canvas_size: tuple[int, int] | None = None,
) -> QtCanvasWidget | AnywidgetCanvasView:
    """Build a canvas widget with wired dims sliders for the viewer.

    Creates a canvas attached to the viewer's scene and connects its dims
    sliders to the controller event bus.  For ``gui="qt"`` this returns a
    ``QtCanvasWidget`` (canvas above a dims slider panel); for
    ``gui="anywidget"`` it returns an :class:`AnywidgetCanvasView` holding the
    canvas and control-panel leaves plus a ``compose(host)`` method.

    Parameters
    ----------
    viewer : Viewer
        The viewer to attach the canvas to.
    axis_ranges : dict[int, tuple[float, float]]
        Mapping of axis index to ``(world_min, world_max)`` used to set the
        slider ranges.  Typically obtained from
        :func:`cellier.convenience.axis_ranges_from_viewer`.
    gui : "qt", "anywidget", or None
        GUI toolkit.  Defaults to ``viewer.gui`` when ``None``; raises if it
        conflicts with ``viewer.gui``.
    render_modes : set[str] or None
        Which camera modes to prepare on the canvas.  Defaults to the scene's
        own ``render_modes``.
    initial_dim : str or None
        Which mode is active first.  Inferred from the scene's current
        ``displayed_axes`` when ``None``.
    fov : float
        Vertical field-of-view in degrees for the 3D perspective camera.
        Default ``70.0``.
    depth_range_3d : tuple[float, float]
        ``(near, far)`` clip distances for the 3D camera.
        Default ``(1.0, 8000.0)``.
    depth_range_2d : tuple[float, float]
        ``(near, far)`` clip distances for the 2D camera.
        Default ``(-500.0, 500.0)``.
    canvas_size : tuple[int, int] or None
        Initial CSS pixel size for the anywidget canvas.  Ignored for the Qt
        gui.  Defaults to ``(600, 600)`` for the anywidget gui.

    Returns
    -------
    QtCanvasWidget or AnywidgetCanvasView
        For ``gui="qt"``, a widget composing the render surface and dims
        slider panel (embed ``canvas_widget.widget``).  For
        ``gui="anywidget"``, an :class:`AnywidgetCanvasView`.

    Raises
    ------
    ValueError
        If *gui* conflicts with ``viewer.gui`` or is not recognised.
    """
    from cellier.convenience._backend import backend_for

    gui = _resolve_gui(viewer, gui)
    # Resolving the backend is also what refuses a gui with no widgets.
    backend = backend_for(
        gui, lacks="no embeddable widget, so no canvas widget can be built for it"
    )
    _ensure_qapplication(gui)
    return build_canvas_view(
        viewer.controller,
        viewer.scene,
        axis_ranges,
        backend=backend,
        render_modes=render_modes,
        initial_dim=initial_dim,
        fov=fov,
        depth_range_3d=depth_range_3d,
        depth_range_2d=depth_range_2d,
        canvas_size=canvas_size,
    )


def _ensure_qapplication(gui: str) -> None:
    """Create a ``QApplication`` if the Qt backend needs one and has none.

    Building a Qt widget without one crashes, and the convenience API is meant
    to be callable from a bare script, so this is the one place that quietly
    makes one.
    """
    if gui != "qt":
        return
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([sys.argv[0]])
