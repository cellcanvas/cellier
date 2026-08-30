"""Render-layer canvas overlays for cellier v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

if TYPE_CHECKING:
    from cellier.visuals._canvas_overlay import CenteredAxes2D


class GFXCanvasOverlay(ABC):
    """Base class for render-layer canvas overlays.

    Each overlay owns its own ``gfx.Scene`` and ``gfx.Camera``.
    ``CanvasView._draw_frame`` calls :meth:`on_frame` every frame so the
    overlay can react to camera or canvas changes.

    Subclasses are responsible for building ``overlay_scene`` and
    ``overlay_camera`` at construction time.

    **Overlays must not write depth.**  They are folded into the main frame
    with ``flush=False``, which keeps the colour *and depth* buffers open, so
    an overlay draws into the same targets as the scene and the effect chain
    then runs once over both.  Any depth an overlay leaves behind was
    produced by the overlay's own camera -- a ``ScreenCoordsCamera``, for the
    one overlay that exists today -- and the screen-space effect passes
    interpret the whole buffer with the *scene* camera's matrices.  For the
    ambient occlusion pass that means a patch of arbitrary occlusion under
    the widget, which is far more visible than the same error would be
    anywhere else.  Set ``depth_test=False, depth_write=False`` on every
    material an overlay adds; pygfx's default is auto, which resolves to
    writing depth for most materials.
    """

    @property
    @abstractmethod
    def overlay_scene(self) -> gfx.Scene:
        """The pygfx Scene containing overlay geometry."""
        ...

    @property
    @abstractmethod
    def overlay_camera(self) -> gfx.Camera:
        """The camera used to render :attr:`overlay_scene`."""
        ...

    @abstractmethod
    def on_frame(self, canvas_width: float, canvas_height: float) -> None:
        """Called once per frame before the overlay is rendered.

        Implementations should rebuild geometry only when the canvas size or
        camera orientation actually changes.

        Parameters
        ----------
        canvas_width : float
            Canvas width in logical pixels.
        canvas_height : float
            Canvas height in logical pixels.
        """
        ...

    @abstractmethod
    def set_visible(self, visible: bool) -> None:
        """Show or hide the overlay.

        Parameters
        ----------
        visible : bool
            ``True`` to show, ``False`` to hide.
        """
        ...


def _compute_screen_dir(
    direction_world_xyz: tuple[float, float, float],
    camera: gfx.Camera,
) -> np.ndarray:
    """Map a world-space direction vector to a normalised 2-D screen direction.

    Applies the linear (rotation+scale) component of the camera matrix to
    ``direction_world_xyz``, then projects the result to the screen XY plane
    and normalises.  Ignores the translation column, so this is correct for
    direction vectors (w = 0).

    Parameters
    ----------
    direction_world_xyz : tuple[float, float, float]
        Unit vector in pygfx world-space XYZ.
    camera : gfx.Camera
        The active camera.  Its ``camera_matrix`` (world->NDC) is read fresh
        each call so the result is always up to date.

    Returns
    -------
    np.ndarray
        Normalised ``(2,)`` float32 direction in screen space.
        Falls back to ``(1.0, 0.0)`` if the projected length is near zero.
    """
    direction = np.asarray(direction_world_xyz, dtype=np.float64)
    camera_matrix = np.array(camera.camera_matrix, dtype=np.float64)  # 4x4, world->NDC
    # Apply the linear part only (w=0 for direction vectors).
    direction_ndc = camera_matrix[:3, :3] @ direction  # (3,)
    direction_screen = direction_ndc[:2]  # take X and Y; Z is depth
    screen_length = float(np.linalg.norm(direction_screen))
    if screen_length < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (direction_screen / screen_length).astype(np.float32)


class GFXCenteredAxes2D(GFXCanvasOverlay):
    """Render-layer implementation of :class:`CenteredAxes2D`.

    Uses ``gfx.ScreenCoordsCamera(invert_y=True)`` so pixel coordinates
    match the Qt convention: origin top-left, y increasing downward.
    Geometry is rebuilt only when the canvas size or camera orientation
    changes, not on every pan or zoom.

    Parameters
    ----------
    model : CenteredAxes2D
        The model-layer overlay description.
    camera : gfx.Camera
        The main scene camera.  A live reference is kept so that
        :meth:`on_frame` always reads the current camera matrix.
    """

    def __init__(
        self,
        model: CenteredAxes2D,
        camera: gfx.Camera,
    ) -> None:
        self._model = model
        self._camera = camera

        self._last_size: tuple[float, float] = (-1.0, -1.0)
        # Store last camera rotation as a tuple for cheap equality comparison.
        self._last_camera_rotation: tuple[float, ...] = ()

        # ── Overlay scene and camera ──────────────────────────────────────
        self._overlay_scene = gfx.Scene()
        # invert_y=True: origin top-left, y increases downward — matches Qt.
        self._overlay_camera = gfx.ScreenCoordsCamera(invert_y=True)

        # ── Line geometry (4 vertices = 2 disconnected segments) ──────────
        # Positions are placeholder zeros; _rebuild_geometry fills them.
        # gfx.LineSegmentMaterial draws pairs (0-1, 2-3) as independent
        # segments — not a connected polyline.
        appearance = model.appearance
        positions = np.zeros((4, 3), dtype=np.float32)
        colors = np.array(
            [
                appearance.axis_a_color,
                appearance.axis_a_color,
                appearance.axis_b_color,
                appearance.axis_b_color,
            ],
            dtype=np.float32,
        )
        line_geometry = gfx.Geometry(
            positions=gfx.Buffer(positions),
            colors=gfx.Buffer(colors),
        )
        line_material = gfx.LineSegmentMaterial(
            thickness=appearance.line_thickness_px,
            thickness_space="screen",
            color_mode="vertex",
            depth_test=False,
            depth_write=False,
        )
        self._line = gfx.Line(line_geometry, line_material)
        self._overlay_scene.add(self._line)

        # ── Optional text labels ──────────────────────────────────────────
        if appearance.show_labels:
            # depth_write=False is required, not cosmetic.  Overlays are
            # folded into the main frame with ``flush=False``, so they share
            # the scene's colour *and depth* buffers, and the effect chain
            # runs once at the end over both.  A label that wrote depth would
            # leave values produced by ScreenCoordsCamera's projection in a
            # buffer the ambient occlusion pass unprojects with the *scene*
            # camera, giving a blob of arbitrary occlusion under the widget.
            # ``gfx.TextMaterial`` leaves ``depth_write`` on auto, which
            # resolves to True for its default alpha mode, so it has to be
            # said explicitly.  The axis lines already do (see above).
            label_material = gfx.TextMaterial(
                color=appearance.label_color, depth_write=False
            )
            self._label_a: gfx.Text | None = gfx.Text(
                text=model.axis_a_label,
                screen_space=True,
                font_size=appearance.font_size_px,
                anchor="middleleft",
                material=label_material,
            )
            self._label_b: gfx.Text | None = gfx.Text(
                text=model.axis_b_label,
                screen_space=True,
                font_size=appearance.font_size_px,
                anchor="topcenter",
                material=label_material,
            )
            self._overlay_scene.add(self._label_a)
            self._overlay_scene.add(self._label_b)
        else:
            self._label_a = None
            self._label_b = None

        self._line.visible = model.visible
        if self._label_a is not None:
            self._label_a.visible = model.visible
        if self._label_b is not None:
            self._label_b.visible = model.visible

    # ------------------------------------------------------------------
    # GFXCanvasOverlay interface
    # ------------------------------------------------------------------

    @property
    def overlay_scene(self) -> gfx.Scene:
        """The pygfx Scene containing the axis-line geometry."""
        return self._overlay_scene

    @property
    def overlay_camera(self) -> gfx.Camera:
        """The ScreenCoordsCamera used to render the overlay."""
        return self._overlay_camera

    def on_frame(self, canvas_width: float, canvas_height: float) -> None:
        """Rebuild geometry if canvas size or camera orientation changed.

        Camera rotation is captured as a plain tuple so the comparison is
        allocation-free on the hot path.

        Parameters
        ----------
        canvas_width : float
            Canvas width in logical pixels.
        canvas_height : float
            Canvas height in logical pixels.
        """
        camera_rotation = tuple(float(v) for v in self._camera.world.rotation)
        size_changed = (canvas_width, canvas_height) != self._last_size
        rotation_changed = camera_rotation != self._last_camera_rotation

        if not size_changed and not rotation_changed:
            return

        self._last_size = (canvas_width, canvas_height)
        self._last_camera_rotation = camera_rotation
        self._rebuild_geometry(canvas_width, canvas_height)

    def set_visible(self, visible: bool) -> None:
        """Show or hide the overlay geometry.

        Parameters
        ----------
        visible : bool
            ``True`` to show, ``False`` to hide.
        """
        self._line.visible = visible
        if self._label_a is not None:
            self._label_a.visible = visible
        if self._label_b is not None:
            self._label_b.visible = visible

    # ------------------------------------------------------------------
    # Geometry rebuild
    # ------------------------------------------------------------------

    def _rebuild_geometry(self, canvas_width: float, canvas_height: float) -> None:
        """Recompute pixel-space positions and upload to the GPU buffer.

        Computes screen-space direction vectors from the live camera matrix,
        then places two line segments of ``length_px`` from the anchor point.
        Text label positions are updated to the segment tips.

        In ``ScreenCoordsCamera(invert_y=True)`` coordinates, origin is
        top-left and y increases downward.

        Parameters
        ----------
        canvas_width : float
            Canvas width in logical pixels.
        canvas_height : float
            Canvas height in logical pixels.
        """
        appearance = self._model.appearance
        offset_x, offset_y = appearance.corner_offset_px
        length = appearance.length_px

        # Compute anchor origin in top-left pixel coordinates.
        if appearance.corner == "center":
            anchor_x = canvas_width / 2.0
            anchor_y = canvas_height / 2.0
        else:
            is_right = "right" in appearance.corner
            is_bottom = "bottom" in appearance.corner
            anchor_x = canvas_width - offset_x if is_right else offset_x
            anchor_y = canvas_height - offset_y if is_bottom else offset_y

        # Map world-space directions to normalised screen-space directions.
        # In ScreenCoordsCamera(invert_y=True), screen y increases downward,
        # but _compute_screen_dir returns NDC y (up=positive).  Flip y so
        # the arrow points in the correct screen direction.
        screen_dir_a = _compute_screen_dir(self._model.axis_a_direction, self._camera)
        screen_dir_b = _compute_screen_dir(self._model.axis_b_direction, self._camera)
        # NDC y -> screen y (invert_y=True convention).
        screen_dir_a = np.array([screen_dir_a[0], -screen_dir_a[1]], dtype=np.float32)
        screen_dir_b = np.array([screen_dir_b[0], -screen_dir_b[1]], dtype=np.float32)

        tip_a = np.array(
            [anchor_x + screen_dir_a[0] * length, anchor_y + screen_dir_a[1] * length],
            dtype=np.float32,
        )
        tip_b = np.array(
            [anchor_x + screen_dir_b[0] * length, anchor_y + screen_dir_b[1] * length],
            dtype=np.float32,
        )

        positions = np.array(
            [
                [anchor_x, anchor_y, 0.0],
                [tip_a[0], tip_a[1], 0.0],
                [anchor_x, anchor_y, 0.0],
                [tip_b[0], tip_b[1], 0.0],
            ],
            dtype=np.float32,
        )

        self._line.geometry.positions.data[:] = positions
        self._line.geometry.positions.update_full()

        if self._label_a is not None:
            self._label_a.local.position = (
                float(tip_a[0]),
                float(tip_a[1]),
                0.0,
            )
        if self._label_b is not None:
            self._label_b.local.position = (
                float(tip_b[0]),
                float(tip_b[1]),
                0.0,
            )
