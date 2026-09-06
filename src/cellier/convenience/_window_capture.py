"""Composite screenshots of a whole Qt viewer window.

A window is chrome plus canvases, and the two come from different renderers.
Qt paints the chrome -- docks, labels, sliders, the grid's panel titles -- and
``QWidget.grab()`` captures all of it faithfully, **except** the render
canvases: a wgpu surface is not part of Qt's paint pipeline, so the canvas
rectangles come back as flat fill.  Measured on a window holding one label and
one cellier canvas: the window grabbed 159 distinct colours, the canvas region
exactly one.

So the picture is assembled rather than taken.  Qt supplies the chrome, and
each canvas rectangle is filled in by an offscreen capture at that widget's
physical size (see :mod:`cellier.render._capture`), which is also what makes
the canvas half reproducible.  The chrome half is only ever as reproducible as
the platform's font rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from cellier.render._capture import write_png

if TYPE_CHECKING:
    from pathlib import Path

    from cellier.controller import CellierController


def screenshot_window(
    window: object,
    controller: CellierController,
    *,
    frames: int | Literal["converged"] = 1,
    save: str | Path | None = None,
    **capture_kwargs,
) -> np.ndarray:
    """Capture a Qt window, canvases included, as an RGBA uint8 array.

    Works headless (``QT_QPA_PLATFORM=offscreen``), which is the setting it
    exists for: on a real display the OS screenshot tool already does this.

    Parameters
    ----------
    window : QWidget
        The top-level window to capture, e.g. the ``QMainWindow`` returned by
        :func:`~cellier.convenience.launch`.
    controller : CellierController
        The controller owning the canvases inside *window*.  Every canvas
        whose widget belongs to this window and is visible is composited in.
    frames : int or "converged"
        Passed to each canvas capture; see
        :meth:`~cellier.controller.CellierController.screenshot`.
    save : str, Path, or None
        When given, also write the composite to this path as a PNG.
    **capture_kwargs
        Forwarded to each canvas capture.

    Returns
    -------
    np.ndarray
        RGBA uint8 array of shape ``(height, width, 4)`` in **physical**
        pixels, so a window on a 2x display comes back at twice its logical
        size -- matching what ``grab()`` produces.

    Notes
    -----
    Only the canvas regions are reproducible.  Chrome depends on the
    platform's widget style and font rendering, so this is a picture *of* a
    window rather than a value to assert on byte-for-byte.
    """
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage

    pixmap = window.grab()
    if pixmap.isNull():
        raise RuntimeError(
            "QWidget.grab() returned a null pixmap; the window has no "
            "renderable surface (is it shown?)."
        )
    device_pixel_ratio = pixmap.devicePixelRatio()
    composite = _qimage_to_rgba(
        pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    )

    for canvas_id in controller.canvas_ids:
        widget = controller.get_canvas_view(canvas_id).widget
        if not _belongs_to(widget, window):
            continue

        top_left = widget.mapTo(window, QPoint(0, 0))
        x = round(top_left.x() * device_pixel_ratio)
        y = round(top_left.y() * device_pixel_ratio)
        width = round(widget.width() * device_pixel_ratio)
        height = round(widget.height() * device_pixel_ratio)
        if width <= 0 or height <= 0:
            continue

        frame = controller.screenshot(
            canvas_id, size=(width, height), frames=frames, **capture_kwargs
        )
        # A canvas can sit partly outside the grab (a scrolled dock, a widget
        # mid-layout), so paste the overlapping region rather than assuming
        # the whole rectangle lands.
        y_end = min(y + height, composite.shape[0])
        x_end = min(x + width, composite.shape[1])
        if y >= y_end or x >= x_end:
            continue
        composite[y:y_end, x:x_end] = frame[: y_end - y, : x_end - x]

    if save is not None:
        write_png(save, composite)
    return composite


def _belongs_to(widget: object, window: object) -> bool:
    """Whether *widget* is a visible descendant of *window*."""
    try:
        return widget.window() is window and widget.isVisible()
    except RuntimeError:
        # Qt already deleted the underlying C++ widget.
        return False


def _qimage_to_rgba(image: object) -> np.ndarray:
    """Copy a ``Format_RGBA8888`` QImage into an owned ``(h, w, 4)`` array.

    ``constBits()`` exposes padded scanlines, so the buffer is reshaped by
    ``bytesPerLine`` and then sliced to the real width -- reading it as
    ``height * width * 4`` gives a sheared image on any width whose stride Qt
    rounded up.  The copy matters too: the buffer belongs to the QImage.
    """
    height = image.height()
    width = image.width()
    stride = image.bytesPerLine() // 4
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8)
    return buffer.reshape(height, stride, 4)[:, :width].copy()
