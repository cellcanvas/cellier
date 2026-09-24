"""Offscreen render harness and shared data fixtures for ``cellier.render``.

Phase 2 of the render-coverage plan.  The suite historically tests visual
*construction* and *request planning* but never renders a frame or asserts a
pixel; the whole slice-response -> texture-upload -> ``build_material`` -> draw
half of every visual was uncovered.  This module supplies the shared pieces the
per-visual render tests (Phase 3) build on:

- ``offscreen_renderer`` -- probes for a working ``wgpu`` adapter once per
  session and skips cleanly when none is available (matching how the existing
  ``qtbot``-driven tests already require a real adapter).
- ``render_scene`` -- fits the canvas camera to the scene, then renders the
  controller's live pygfx scene into an offscreen ``rendercanvas`` and returns
  an RGBA ``uint8`` array for pixel assertions.
- ``drive_reslice`` -- drives the async slicer to quiescence so brick/tile
  reads commit to the GPU before a frame is drawn.
- Small reusable data fixtures: an in-memory image volume, a labels volume, and
  a 2-level multiscale zarr store (``small_zarr_store`` is copied here because
  the render tests moved out of ``tests/v2/`` and no longer see that conftest).

Every helper that touches the GPU funnels through ``offscreen_renderer`` so a
machine without an adapter skips the render tests instead of erroring.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx
import pytest
import tensorstore as ts

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.data.label._label_memory_store import LabelMemoryStore

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from uuid import UUID

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@pytest.fixture
def controller(qtbot) -> Iterator[CellierController]:
    """A ``CellierController`` with camera-driven reslicing disabled.

    The render tests drive reslicing explicitly and read pixels deterministically.
    The camera-settle debounce (``_on_camera_changed`` -> ``create_task``) would
    otherwise schedule ``_settle_after`` coroutines that never get awaited once
    the test's event loop closes, surfacing as ``PytestUnraisableExceptionWarning``
    under the suite's ``filterwarnings = error``.  Disabling it keeps frames
    deterministic and the loop clean.  ``qtbot`` ensures a ``QApplication`` exists
    for the offscreen canvas the controller builds in ``add_canvas``.
    """
    ctrl = CellierController()
    ctrl.camera_reslice_enabled = False
    yield ctrl
    # The autouse fixture in tests/conftest.py would catch this too, but the
    # shared fixture that hands out the controller is the honest place to give
    # it back: close() cancels pending slices, closes the canvases, disconnects
    # the psygnal bridges and clears the buses.
    ctrl.close()


# ---------------------------------------------------------------------------
# GPU adapter probe + offscreen render helper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def offscreen_renderer() -> Callable[
    [gfx.Scene, gfx.Camera, tuple[int, int]], np.ndarray
]:
    """Return an ``(scene, camera, size) -> RGBA array`` renderer.

    Creating the offscreen canvas + ``WgpuRenderer`` is what actually touches
    the GPU adapter, so the probe happens here: if no adapter is available the
    whole render suite skips with a clear message rather than erroring per test.
    """
    from rendercanvas.offscreen import RenderCanvas as OffscreenRenderCanvas

    try:
        probe_canvas = OffscreenRenderCanvas(size=(16, 16), pixel_ratio=1)
        gfx.WgpuRenderer(probe_canvas)
    except Exception as exc:  # pragma: no cover - env-dependent skip path
        pytest.skip(f"no usable wgpu offscreen adapter: {exc}")

    def _render(
        scene: gfx.Scene,
        camera: gfx.Camera,
        size: tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        canvas = OffscreenRenderCanvas(size=size, pixel_ratio=1)
        renderer = gfx.WgpuRenderer(canvas)

        # ``rendercanvas`` swallows anything the draw callback raises -- silently,
        # it does not even log it -- and leaves ``_last_image`` as ``None``, so
        # ``draw()`` returns ``None``.  Capture the error here: otherwise a failed
        # render reaches the test as ``np.asarray(None)``, and the first pixel
        # read reports ``IndexError: too many indices for array``, naming neither
        # the real exception nor the render.
        draw_errors: list[BaseException] = []

        def _draw() -> None:
            try:
                renderer.render(scene, camera)
            except BaseException as exc:
                draw_errors.append(exc)
                raise

        canvas.request_draw(_draw)
        image = canvas.draw()

        if draw_errors:
            # Lead with the underlying error rather than leaving it to the
            # ``__cause__`` chain: pytest's ``short test summary info`` prints only
            # this final line (truncated to terminal width), and that summary is
            # often all a CI log gets read for.
            cause = draw_errors[0]
            raise RuntimeError(
                f"offscreen draw failed -- {type(cause).__name__}: {cause}"
            ) from cause
        if image is None:
            raise RuntimeError(
                "the offscreen canvas produced no frame, and the draw callback "
                "did not raise -- rendercanvas never presented an image."
            )
        return np.asarray(image)

    return _render


@pytest.fixture
def render_scene(
    offscreen_renderer: Callable[..., np.ndarray],
) -> Callable[..., np.ndarray]:
    """Return ``render(controller, scene_id, size=..., background=False) -> RGBA``.

    Fits the scene's camera (``fit_camera`` uses the same ``show_object`` call
    the app does) and draws the controller's live pygfx scene through the
    offscreen renderer.  A square ``size`` keeps the camera aspect (fit on the
    default square canvas) consistent with the offscreen framebuffer.

    The scene background is **hidden by default**, and that is what makes this
    harness's central contract hold: *a pixel with alpha > 0 is a pixel a
    visual drew*.  cellier ships an opaque gray gradient behind every scene
    (see ``BackgroundAppearance``), and with it drawn the whole framebuffer
    comes back at alpha 255, so the mask means nothing -- and content blended
    over gray stops reading as a pure hue, which breaks colour assertions just
    as thoroughly.

    Pass ``background=True`` to draw the scene's own background instead.  Only
    a test *about* the background should need it; see
    ``test_render_harness.py`` for the ones that do.
    """

    def _render(
        controller: CellierController,
        scene_id: UUID,
        size: tuple[int, int] = (128, 128),
        background: bool = False,
    ) -> np.ndarray:
        # Through the public write API rather than reaching into the render
        # layer, so the harness travels the same path the app does.
        controller.update_background_field(scene_id, "visible", background)
        controller.fit_camera(scene_id)
        canvas_id = controller.get_canvas_ids(scene_id)[0]
        canvas_view = controller._render_manager._canvases[canvas_id]
        gfx_scene = canvas_view._get_scene_fn(scene_id)
        return offscreen_renderer(gfx_scene, canvas_view.camera, size)

    return _render


async def drain_loading(controller: CellierController) -> None:
    """Drive the slicer and the chunk scheduler until nothing is loading.

    The slicer runs one task per visual (multiscale 2D, in-memory, geometry);
    the chunk scheduler loads multiscale 3D and commits on drawn frames, which
    a test does not produce, so commit rounds run here as reads land.
    """
    render_manager = controller._render_manager
    slicer = render_manager._slicer
    for _ in range(20):
        # Let a pass scheduled with call_soon run before looking.
        await asyncio.sleep(0)
        # A reslice waiting on a timer (a rate-capped store change, a dims
        # settle) starts loading of its own.
        deferred = controller._deferred_reslice_tasks()
        if deferred:
            await asyncio.gather(*deferred, return_exceptions=True)
            await asyncio.sleep(0)
        tasks = list(slicer._tasks.values())
        if tasks:
            await asyncio.gather(*tasks)
        await render_manager.scheduler.drain(commit=True)
        if not slicer._tasks or all(t.done() for t in slicer._tasks.values()):
            return


@pytest.fixture
def drive_reslice() -> Callable[[CellierController], Awaitable[None]]:
    """Return an async helper that drives the slicer to quiescence.

    ``reslice_all`` / ``reslice_scene`` schedule one async task per visual; a
    single visual may spawn follow-up tasks (multiscale LOD rounds), so we loop
    until no tasks remain before the caller renders a frame.
    """

    async def _drive(controller: CellierController) -> None:
        await drain_loading(controller)

    return _drive


@pytest.fixture
def reslice(
    drive_reslice: Callable[[CellierController], Awaitable[None]],
) -> Callable[..., Awaitable[None]]:
    """Return ``reslice(controller, scene_id) -> awaitable`` that loads a scene.

    Fits the camera *before* reslicing: the 3D multiscale brick selection is
    frustum-driven, so it commits nothing until the camera actually frames the
    volume.  Fitting first (the node matrices are set at construction, so no
    data need be loaded) makes that path select and commit bricks.  It is a
    harmless no-op for the 2D / in-memory paths, which don't depend on the
    frustum.
    """

    async def _reslice(controller: CellierController, scene_id: UUID) -> None:
        controller.fit_camera(scene_id)
        controller.reslice_all()
        await drive_reslice(controller)

    return _reslice


# ---------------------------------------------------------------------------
# Reusable data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_volume() -> ImageMemoryStore:
    """A (16, 24, 32) float32 volume with a bright cuboid in the centre."""
    data = np.zeros((16, 24, 32), dtype=np.float32)
    data[4:12, 6:18, 8:24] = 1.0
    return ImageMemoryStore(data=data, name="image_volume")


@pytest.fixture
def gradient_image() -> ImageMemoryStore:
    """A (8, 16, 16) volume whose values ramp 0->1 along the last (X) axis."""
    data = np.zeros((8, 16, 16), dtype=np.float32)
    data[...] = np.linspace(0.0, 1.0, 16, dtype=np.float32)[None, None, :]
    return ImageMemoryStore(data=data, name="gradient_image")


@pytest.fixture
def labels_volume() -> LabelMemoryStore:
    """A (8, 24, 32) int32 label volume with two labelled blocks."""
    data = np.zeros((8, 24, 32), dtype=np.int32)
    data[:, 4:12, 4:12] = 3
    data[:, 14:20, 18:28] = 7
    return LabelMemoryStore(data=data, name="labels_volume")


@pytest.fixture
def small_zarr_store(tmp_path):
    """A minimal 2-level multiscale zarr v3 store on disk (zeros).

    Copied from ``tests/v2/conftest.py`` — the render tests moved out of
    ``tests/v2/`` and no longer inherit that fixture.
    """
    for name, shape in [("s0", (8, 8, 8)), ("s1", (4, 4, 4))]:
        level_path = tmp_path / name
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(level_path)},
            "metadata": {
                "shape": list(shape),
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [4, 4, 4]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        store[...].write(np.zeros(shape, dtype=np.float32)).result()

    return tmp_path


def _write_multiscale_zarr(
    root,
    levels: list[tuple[str, tuple[int, ...]]],
    fill: Callable[[np.ndarray], None],
    dtype: str = "float32",
) -> None:
    """Write per-level zarr v3 arrays under *root*, filled via *fill*."""
    for name, shape in levels:
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / name)},
            "metadata": {
                "shape": list(shape),
                "data_type": dtype,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [max(1, s // 2) for s in shape]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        arr = np.zeros(shape, dtype=dtype)
        fill(arr)
        store[...].write(arr).result()


@pytest.fixture
def multiscale_image_store(tmp_path) -> MultiscaleZarrDataStore:
    """A 2-level (s0=16^3, s1=8^3) multiscale image with a bright interior."""

    def _fill(arr: np.ndarray) -> None:
        lo = [s // 4 for s in arr.shape]
        hi = [s - s // 4 for s in arr.shape]
        arr[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = 1.0

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (16, 16, 16)), ("s1", (8, 8, 8))],
        fill=_fill,
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        name="multiscale_image_store",
    )


@pytest.fixture
def multiscale_labels_store(tmp_path) -> MultiscaleZarrDataStore:
    """A 2-level (s0=16^3, s1=8^3) multiscale label store with two blocks."""

    def _fill(arr: np.ndarray) -> None:
        _d, h, w = arr.shape
        arr[:, h // 4 : h // 2, w // 4 : w // 2] = 3
        arr[:, h // 2 : 3 * h // 4, w // 2 : 3 * w // 4] = 7

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (16, 16, 16)), ("s1", (8, 8, 8))],
        fill=_fill,
        dtype="int32",
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        name="multiscale_labels_store",
    )
