r"""Render a cellier viewer to a PNG, headlessly.

The missing third leg of this repo's verification story.  ``_qt_acceptance``
checks Qt panel structure, the marimo harness checks anywidget JS, and neither
looks at a rendered frame -- so a change to a shader, a colormap, a transform
or a render pass could only be judged by a human running a demo.  This script
turns "run it and look" into one command that writes a file.

Usage::

    .venv/bin/python -m cellier.convenience.capture <target> --out picture.png

where ``<target>`` is either

* a **Python file exposing** ``build()`` returning a ``Viewer`` or
  ``OrthoViewer`` (already populated; it must not call ``launch``/``show``), or
* a **viewer file** written by ``Viewer.to_file`` / ``OrthoViewer.to_file``.

Examples::

    .venv/bin/python -m cellier.convenience.capture demos/volume.py \
        --size 900x700 --frames converged --out /tmp/volume.png

    .venv/bin/python -m cellier.convenience.capture saved_viewer.json \
        --panel xy --scale 2 --out /tmp/xy.png

The capture is reproducible: run it twice, get identical bytes.  ``Qt`` is
forced onto its offscreen platform plugin by :func:`main` before the target is
imported, so a target that builds a Qt viewer works with no display attached.
Importing this module has no such side effect.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path


def _parse_size(text: str) -> tuple[int, int]:
    """Parse ``WxH`` into ``(width, height)``."""
    try:
        width, _, height = text.lower().partition("x")
        return int(width), int(height)
    except ValueError:  # pragma: no cover - argparse reports it
        raise argparse.ArgumentTypeError(
            f"expected a size like 900x700, got {text!r}"
        ) from None


def _parse_frames(text: str) -> int | str:
    """Parse the ``--frames`` argument: a positive int or ``converged``."""
    if text == "converged":
        return text
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer or 'converged', got {text!r}"
        ) from None
    if value < 1:
        raise argparse.ArgumentTypeError("--frames must be >= 1")
    return value


def _load_viewer(target: Path) -> object:
    """Build the viewer named by *target*, from a script or a saved file."""
    if target.suffix == ".py":
        return _viewer_from_script(target)
    return _viewer_from_file(target)


def _viewer_from_script(target: Path) -> object:
    """Import *target* and call its ``build()``."""
    spec = importlib.util.spec_from_file_location(target.stem, target)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {target}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so a target using dataclasses, pickle, or
    # anything else that looks itself up by module name still works.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    build = getattr(module, "build", None)
    if build is None:
        raise SystemExit(
            f"{target} defines no build(). Add a build() returning a populated "
            "Viewer or OrthoViewer -- and do not call launch()/show() in it, "
            "which would block waiting for a window."
        )
    return build()


def _viewer_from_file(target: Path) -> object:
    """Restore a viewer written by ``to_file``, trying both viewer classes."""
    from cellier.convenience import OrthoViewer, Viewer

    try:
        return OrthoViewer.from_file(target)
    except Exception:
        # Not a four-panel orthoviewer; the single-scene reader is the other
        # possibility, and its error is the one worth showing.
        return Viewer.from_file(target)


def _ensure_canvases(viewer: object, size: tuple[int, int]) -> None:
    """Give every scene a canvas, because **reslicing is canvas-driven**.

    A slice request is planned per canvas -- from its camera, its size and its
    frustum -- so a scene with no canvas requests nothing, and ``reslice_all``
    on a freshly built viewer emits zero ``ResliceStartedEvent``.  Capturing
    then yields a correctly-rendered picture of an empty scene, which reads as
    a broken renderer rather than as missing data.

    A target built for this script has usually never called ``add_canvas``
    (there is no window to put one in), so this supplies the missing canvas
    before the load rather than making every target remember to.
    """
    controller = viewer.controller
    for scene_id in controller._model.scenes:
        if not controller.get_canvas_ids(scene_id):
            controller.add_canvas(scene_id=scene_id, canvas_size=size)


def _fit_cameras(controller: object) -> None:
    """Fit every scene's canvases to that scene's bounding box."""
    for scene_id in controller._model.scenes:
        controller.fit_camera(scene_id)


def _camera_states(controller: object) -> dict:
    """Snapshot every canvas's camera, keyed by canvas id."""
    return {
        canvas_id: canvas.capture_camera_state()
        for canvas_id, canvas in controller._render_manager._canvases.items()
    }


async def _wait_for_slicer(controller: object) -> None:
    """Await in-flight loading until none is left.

    Two loaders: the slicer's tasks (in-memory visuals; at most 50 rounds)
    and the chunk scheduler (multiscale visuals), drained with a commit
    round per poll, as a drawing canvas would run them.  Reslices still
    waiting on a timer (a dims settle, a rate-capped store change) are
    awaited first, since they start new loading.
    """
    render_manager = controller._render_manager
    slicer = render_manager._slicer
    deferred = getattr(controller, "_deferred_reslice_tasks", list)
    for _ in range(50):
        pending = deferred()
        if not pending:
            break
        await asyncio.gather(*pending, return_exceptions=True)
    for _ in range(50):
        tasks = list(slicer._tasks.values())
        if not tasks:
            break
        await asyncio.gather(*tasks)
    await render_manager.scheduler.drain(commit=True)


async def _load_data(viewer: object) -> None:
    """Drive loading to quiescence so the capture sees loaded data.

    A capture renders what is resident on the GPU, so a picture taken before
    the first reads land is an honest picture of an empty scene -- which reads
    as "the renderer is broken" rather than "the data had not arrived".

    **The cameras are fitted before the first reslice.**  Slice requests are
    planned per canvas from its camera: a multiscale visual picks its 2D tiles
    and 3D bricks from the view and frustum, so reslicing an unfitted camera
    loads only what that camera happens to see -- one corner tile in 2D,
    nothing in 3D -- and fitting afterwards moves the camera onto data that
    was never requested.  In-memory images load a whole texture and cannot
    show this.

    A fit before the load can do nothing: ``fit_camera`` skips a scene that
    has no bounds yet, and some visuals only get bounds once data lands.  So
    the cameras are fitted again after the load, and if that moved any of
    them, everything is resliced for the new view and loaded again.
    """
    controller = viewer.controller
    _fit_cameras(controller)
    controller.reslice_all()
    await _wait_for_slicer(controller)

    before = _camera_states(controller)
    _fit_cameras(controller)
    if _camera_states(controller) != before:
        controller.reslice_all()
        await _wait_for_slicer(controller)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build the viewer, capture, and write the PNG."""
    parser = argparse.ArgumentParser(
        description="Render a cellier viewer to a PNG, headlessly.",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="a .py file exposing build(), or a viewer file from to_file()",
    )
    parser.add_argument("--out", type=Path, required=True, help="PNG to write")
    parser.add_argument(
        "--size",
        type=_parse_size,
        default=None,
        help="WxH in pixels (per panel for an orthoviewer grid)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="resolution multiplier"
    )
    parser.add_argument(
        "--frames",
        type=_parse_frames,
        default=1,
        help=(
            "1 (default) for a single crisp frame, N to accumulate N frames, "
            "or 'converged' to accumulate until the image settles -- use that "
            "whenever ambient occlusion is on"
        ),
    )
    parser.add_argument(
        "--panel",
        default=None,
        help="orthoviewer only: xy, xz, yz or vol (default: all four, 2x2)",
    )
    args = parser.parse_args(argv)

    # Before the target is imported: a target is allowed to build a gui="qt"
    # viewer, and without this it would try to open a display that a CI box or
    # an agent session does not have.  Capture itself never needs one -- it
    # renders offscreen whatever the viewer's gui is.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    if not args.target.exists():
        parser.error(f"no such target: {args.target}")

    viewer = _load_viewer(args.target)
    _ensure_canvases(viewer, args.size or (600, 600))
    asyncio.run(_load_data(viewer))

    kwargs = {"size": args.size, "scale": args.scale, "frames": args.frames}
    if args.panel is not None:
        kwargs["panel"] = args.panel
    frame = viewer.screenshot(save=args.out, **kwargs)

    print(f"wrote {args.out}  {frame.shape[1]}x{frame.shape[0]}")
    viewer.controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
