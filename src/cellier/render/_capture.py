"""Offscreen frame capture -- reproducible screenshots of a scene.

Every frame this module produces comes from a **dedicated offscreen canvas**,
never from the canvas a user is looking at.  That is the whole design, and it
is what buys the contract :func:`capture_scene` states: two captures of the
same viewer state produce byte-identical arrays.

A live canvas cannot deliver that.  It is scheduler-driven (nothing can make
it draw synchronously), it is sized by a window rather than by the caller, and
its framebuffer carries whatever the accumulation history happened to hold
when the shutter fell.  ``rendercanvas.offscreen`` has none of those
properties: ``draw()`` runs the callback and returns the presented frame,
right now, at exactly the size asked for.

The capture canvas is a real ``CanvasView`` on the **same scene**, so the frame
travels the real pipeline -- cellier's blender, the ambient occlusion, outline
and accumulation passes, the canvas overlays, the per-frame visual tick.  It is
not a second renderer with its own idea of how a scene looks, which is what
makes a capture worth trusting.
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from cellier.render.canvas_view import CanvasView
    from cellier.render.render_manager import RenderManager

#: Frame ceiling for ``frames="converged"``.  A request that would need more
#: raises rather than silently returning an unsettled frame.
DEFAULT_MAX_FRAMES: int = 64

#: How much of the temporal accumulator's error may remain when
#: ``frames="converged"`` stops: 0.01 means "within 1% of the settled image".
DEFAULT_RESIDUAL: float = 0.01


def frames_to_settle(blend_weight: float, residual: float = DEFAULT_RESIDUAL) -> int:
    """Frames the temporal accumulator needs to come within *residual* of settled.

    ``frames="converged"`` computes its frame count from the accumulator's own
    dynamics instead of watching the pixels, because **measurement showed that
    watching the pixels does not work**.  Three variants were built and
    discarded on an isosurface scene with ambient occlusion enabled:

    * *stop when consecutive frames match.*  With AO on, the kernel rotation
      varies every frame, so the EMA reaches a steady state rather than a fixed
      point: the frame-to-frame maximum difference plateaus at 3-17 code values
      **forever** and the test can never pass.
    * *stop when a high percentile of the difference is small.*  It passes
      immediately -- the dither touches under 0.1% of channel values (measured
      p99.9 = 1, mean = 0.003) -- so it hides exactly the noise the
      accumulation is there to average away.
    * *compare against the frame from half way back* instead of the previous
      one.  Same outcome, because the problem is the statistic and not the
      baseline: at blend weights 0.5, 0.2, 0.1 and 0.02 every variant stopped
      after 2-4 draws, including the 0.02 case that needs hundreds.

    The accumulator, though, converges geometrically: each frame leaves
    ``(1 - blend_weight)`` of the error, so reaching *residual* takes
    ``log(residual) / log(1 - blend_weight)`` frames.  That is exact, needs no
    tolerance, and -- being a count rather than a measurement -- is identical
    on every machine.

    Parameters
    ----------
    blend_weight : float
        The accumulator's steady-state weight, in ``(0, 1]``.
    residual : float
        Fraction of the initial error still allowed when stopping.

    Returns
    -------
    int
        Frames to draw, at least 1.
    """
    if blend_weight >= 1.0:
        # Every frame replaces the history outright; there is nothing to settle.
        return 1
    return max(1, math.ceil(math.log(residual) / math.log(1.0 - blend_weight)))


def capture_scene(
    render_manager: RenderManager,
    scene_id: UUID,
    *,
    seed_canvas_id: UUID | None = None,
    size: tuple[int, int] | None = None,
    scale: float = 1.0,
    frames: int | Literal["converged"] = 1,
    dim: str | None = None,
    fov: float = 70.0,
    depth_range: tuple[float, float] | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    residual: float = DEFAULT_RESIDUAL,
) -> np.ndarray:
    """Render *scene_id* to an offscreen canvas and return the frame.

    Parameters
    ----------
    render_manager : RenderManager
        Owner of the scene and its canvases.
    scene_id : UUID
        Scene to render.
    seed_canvas_id : UUID or None
        Canvas whose camera, dimensionality and depth range the capture
        should copy.  This selects a **viewpoint, not a surface**: the pixels
        always come from a fresh offscreen canvas.  When ``None`` the capture
        canvas fits the camera to the scene itself.
    size : tuple[int, int] or None
        ``(width, height)`` before *scale*.  Defaults to the seed canvas's
        physical size, so an unqualified capture reproduces the on-screen
        framing; ``(600, 600)`` when there is no seed canvas.
    scale : float
        Multiplier on *size*.  ``scale=2`` doubles the output resolution.
    frames : int or "converged"
        ``1`` (default) disables temporal accumulation and draws one frame:
        crisp and instant.  ``"converged"`` enables accumulation and draws
        until successive frames stop changing -- what you want whenever
        ambient occlusion is on, since a single-sample AO frame is visibly
        noisy.  ``N`` enables accumulation and draws exactly N frames.
    dim : str or None
        ``"2d"`` or ``"3d"``.  Defaults to the seed canvas's dimensionality,
        or the scene's when there is no seed canvas.
    fov : float
        Vertical field of view for the 3D camera.
    depth_range : tuple[float, float] or None
        Near/far clip distances.  Defaults to the seed canvas's.
    max_frames : int
        Frame ceiling for ``frames="converged"``.  A scene whose accumulator
        would need more raises rather than returning an unsettled frame.
    residual : float
        How much of the accumulator's error may remain when
        ``frames="converged"`` stops.  See :func:`frames_to_settle`.

    Returns
    -------
    np.ndarray
        RGBA ``uint8`` array of shape ``(height, width, 4)``, where *height*
        and *width* are the scaled request.

    Raises
    ------
    ValueError
        If *frames* is neither a positive integer nor ``"converged"``.
    RuntimeError
        If ``frames="converged"`` would need more than *max_frames* frames.
    """
    if frames != "converged" and (not isinstance(frames, int) or frames < 1):
        raise ValueError(
            f"frames must be a positive int or 'converged', got {frames!r}"
        )

    seed = (
        render_manager._canvases.get(seed_canvas_id)
        if seed_canvas_id is not None
        else None
    )
    if seed_canvas_id is not None and seed is None:
        raise KeyError(f"No canvas registered with id {seed_canvas_id}")

    width, height = _resolve_size(seed, size, scale)
    capture_dim = dim or (seed._dim if seed is not None else "3d")

    capture_id = uuid4()
    canvas_view = render_manager.add_canvas(
        capture_id,
        scene_id,
        parent=None,
        dim=capture_dim,
        fov=fov if seed is None else seed._fov,
        depth_range=depth_range or (seed._depth_range if seed else (1.0, 8000.0)),
        gui="offscreen",
        size=(width, height),
    )
    # Deliberately no event bus: a capture is an observer, and a canvas that
    # emits would push CameraChangedEvent onto the bus, which the controller
    # answers with a settle reslice.  Taking a screenshot must not reload the
    # scene under the window the user is looking at.
    try:
        _seed_camera(canvas_view, seed, render_manager, scene_id)
        _configure_determinism(canvas_view, frames)
        render_manager.reset_frame_counters(scene_id)

        canvas = canvas_view.widget
        canvas.request_draw(canvas_view._draw_frame)

        if frames == "converged":
            frames = _settling_frames(canvas_view, max_frames, residual)
        frame = None
        for _ in range(frames):
            frame = np.asarray(canvas.draw())
        return frame
    finally:
        # A CanvasView owns a WgpuRenderer and a registered event handler, and
        # is kept alive by the backend rather than by refcounting, so a
        # capture that skipped this would leak a renderer per screenshot.
        render_manager.remove_canvas(capture_id)


def _resolve_size(
    seed: CanvasView | None,
    size: tuple[int, int] | None,
    scale: float,
) -> tuple[int, int]:
    """Return the ``(width, height)`` the capture canvas should be built at."""
    if size is None:
        if seed is not None:
            size = tuple(int(v) for v in seed.widget.get_physical_size())
        else:
            size = (600, 600)
    width = max(1, round(size[0] * scale))
    height = max(1, round(size[1] * scale))
    return width, height


def _seed_camera(
    canvas_view: CanvasView,
    seed: CanvasView | None,
    render_manager: RenderManager,
    scene_id: UUID,
) -> None:
    """Point the capture canvas at what *seed* is looking at, or fit the scene."""
    if seed is None:
        # No canvas to copy -- fit the scene, which is what makes
        # ``screenshot()`` work on a viewer that never built a canvas at all.
        canvas_view.show_object(render_manager.get_scene(scene_id))
        return

    # pygfx's own ``get_state``/``set_state`` rather than cellier's
    # ``CameraState``: the latter is a *logical* snapshot for the model layer
    # and drops what a faithful copy needs (the orthographic extent, the
    # reference up, ``maintain_aspect``).  Camera to camera, the pygfx pair is
    # the lossless one.
    #
    # The guard keeps the setters from reading as a user camera move, which
    # ``_draw_frame`` would otherwise report -- on a canvas that has no event
    # bus, but the flag also suppresses the accumulation reset, and the state
    # cache below is what makes the first frame see no diff at all.
    canvas_view._applying_model_state = True
    try:
        canvas_view._camera.set_state(seed._camera.get_state())
    finally:
        canvas_view._applying_model_state = False
    canvas_view._last_camera_state = canvas_view.capture_camera_state()


def _configure_determinism(
    canvas_view: CanvasView,
    frames: int | Literal["converged"],
) -> None:
    """Set the accumulation pass according to *frames*.

    ``frames=1`` bypasses accumulation entirely: with one frame there is no
    history to blend, and leaving the pass on would return ``alpha`` of a
    picture rather than the picture.
    """
    canvas_view._accum_pass.enabled = frames != 1


def _settling_frames(
    canvas_view: CanvasView,
    max_frames: int,
    residual: float,
) -> int:
    """Frames ``"converged"`` should draw on *canvas_view*, or raise.

    Read from the capture canvas's own accumulation pass rather than from the
    config, so a canvas whose blend weight was changed after construction gets
    the count its accumulator actually needs.
    """
    needed = frames_to_settle(canvas_view._accum_pass.blend_weight, residual)
    if needed > max_frames:
        raise RuntimeError(
            f"frames='converged' needs {needed} frames to bring the temporal "
            f"accumulator within {residual:g} of settled at blend_weight="
            f"{canvas_view._accum_pass.blend_weight:g}, but max_frames is "
            f"{max_frames}. Raise max_frames, raise temporal_blend_weight, or "
            f"pass frames={max_frames} to accept an unsettled image."
        )
    return needed


def write_png(path: str | Path, image: np.ndarray) -> None:
    """Write an RGBA uint8 array to *path* as a PNG.

    Hand-rolled on ``zlib`` rather than delegating to Pillow or imageio,
    because cellier declares neither.  A capture that can be taken but not
    saved is a poor deal, and a save that raises ``ImportError`` at the end of
    a long headless render is a worse one -- so this depends only on the
    standard library and cannot fail for want of a package.

    Parameters
    ----------
    path : str or Path
        Destination file.
    image : np.ndarray
        ``(height, width, 4)`` uint8 array, as returned by
        :func:`capture_scene`.

    Raises
    ------
    ValueError
        If *image* is not an ``(h, w, 4)`` uint8 array.
    """
    array = np.ascontiguousarray(image)
    if array.ndim != 3 or array.shape[2] != 4 or array.dtype != np.uint8:
        raise ValueError(
            "write_png expects an (height, width, 4) uint8 array, got "
            f"shape {array.shape} dtype {array.dtype}"
        )
    height, width = array.shape[:2]

    # Each scanline is prefixed with its filter type; 0 means "no filter",
    # which costs a little size and saves picking a heuristic.
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(png)
