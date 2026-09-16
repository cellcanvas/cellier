"""Tests for offscreen capture (``cellier.render._capture``).

The contract under test is the one the capture API states: **two captures of
the same viewer state produce byte-identical arrays**, at exactly the size
asked for, without a window and whichever GUI toolkit the controller targets.

Everything that draws goes through the ``offscreen_renderer`` fixture's adapter
probe first, so a machine with no usable wgpu adapter skips rather than errors
-- the same rule the rest of ``tests/render`` follows.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render._capture import frames_to_settle, write_png
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


@pytest.fixture
def loaded_volume_scene(controller, image_volume, offscreen_renderer):
    """A 3D scene with one canvas and a committed in-memory volume.

    A canvas is required, not incidental: slice requests are planned per
    canvas, so a scene without one loads nothing and every capture of it is a
    correct picture of an empty scene.
    """
    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_image(
        data=image_volume,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="iso", iso_threshold=0.5
        ),
    )
    controller.add_canvas(scene_id=scene.id, canvas_size=(128, 96))
    return scene


# ---------------------------------------------------------------------------
# The reproducibility contract
# ---------------------------------------------------------------------------


async def test_two_captures_are_byte_identical(
    controller, loaded_volume_scene, reslice
):
    """The headline contract: the same state captures to the same bytes."""
    await reslice(controller, loaded_volume_scene.id)
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]

    first = controller.screenshot(canvas_id, size=(120, 90))
    second = controller.screenshot(canvas_id, size=(120, 90))

    assert np.array_equal(first, second)


async def test_capture_before_any_frame_is_drawn_succeeds(
    controller, loaded_volume_scene, reslice
):
    """No frame need ever have been presented on the canvas being captured.

    The previous implementation read back the canvas's own framebuffer, which
    does not exist until the backend has drawn -- so this raised
    ``AttributeError`` from inside pygfx.  Capturing on its own canvas removes
    the precondition entirely.
    """
    await reslice(controller, loaded_volume_scene.id)
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]

    frame = controller.screenshot(canvas_id, size=(64, 64))

    assert frame.shape == (64, 64, 4)
    assert frame.dtype == np.uint8


async def test_requested_size_and_scale_are_honoured(
    controller, loaded_volume_scene, reslice
):
    """``size`` and ``scale`` decide the output shape, not the canvas or display.

    Both arguments existed before and both were silently ignored: the returned
    array was whatever the last real draw had produced.
    """
    await reslice(controller, loaded_volume_scene.id)
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]

    assert controller.screenshot(canvas_id, size=(200, 150)).shape == (150, 200, 4)
    assert controller.screenshot(canvas_id, size=(200, 150), scale=2).shape == (
        300,
        400,
        4,
    )
    assert controller.screenshot(canvas_id, size=(37, 91)).shape == (91, 37, 4)


async def test_default_size_reproduces_the_canvas_framing(
    controller, loaded_volume_scene, reslice
):
    """An unqualified capture comes back at the canvas's own physical size."""
    await reslice(controller, loaded_volume_scene.id)
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]
    width, height = controller.get_canvas_view(canvas_id).widget.get_physical_size()

    frame = controller.screenshot(canvas_id)

    assert frame.shape == (int(height), int(width), 4)


async def test_capture_leaves_no_canvas_behind(
    controller, loaded_volume_scene, reslice
):
    """The temporary capture canvas is removed, renderer and all.

    A ``CanvasView`` is kept alive by the backend rather than by refcounting,
    so a capture that skipped teardown would leak a ``WgpuRenderer`` per
    screenshot -- cumulative, and invisible until a suite slows to a crawl.
    """
    await reslice(controller, loaded_volume_scene.id)
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]
    before = set(controller.canvas_ids)

    for _ in range(3):
        controller.screenshot(canvas_id, size=(32, 32))

    assert set(controller.canvas_ids) == before


async def test_capture_does_not_disturb_the_live_canvas(
    controller, loaded_volume_scene, reslice
):
    """Taking a screenshot must not move the camera of the canvas on screen."""
    await reslice(controller, loaded_volume_scene.id)
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]
    before = controller.get_camera_state(canvas_id)

    controller.screenshot(canvas_id, size=(200, 150), scale=2)

    assert controller.get_camera_state(canvas_id) == before


# ---------------------------------------------------------------------------
# frames=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frames", [2, 4, "converged"])
async def test_accumulated_captures_are_reproducible(
    controller, loaded_volume_scene, reslice, frames
):
    """Accumulation is reproducible because the frame counters are rewound.

    Without ``RenderManager.reset_frame_counters`` the second capture would
    start part-way through the accumulation history and the AO kernel
    rotation, and land somewhere else.
    """
    await reslice(controller, loaded_volume_scene.id)
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]

    first = controller.screenshot(canvas_id, size=(64, 48), frames=frames)
    second = controller.screenshot(canvas_id, size=(64, 48), frames=frames)

    assert np.array_equal(first, second)


@pytest.mark.parametrize("frames", [0, -1, 1.5, "settled", None])
async def test_invalid_frames_is_rejected(
    controller, loaded_volume_scene, reslice, frames
):
    """A bad ``frames`` raises rather than being coerced into something odd."""
    await reslice(controller, loaded_volume_scene.id)
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]

    with pytest.raises(ValueError, match="positive int or 'converged'"):
        controller.screenshot(canvas_id, size=(32, 32), frames=frames)


# ---------------------------------------------------------------------------
# frames="converged" -- the settling count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("blend_weight", "expected"),
    [(1.0, 1), (0.5, 7), (0.2, 21), (0.1, 44), (0.02, 228)],
)
def test_settling_count_follows_the_accumulator(blend_weight, expected):
    """The frame count comes from the EMA's own dynamics, not from the pixels.

    Each frame leaves ``1 - blend_weight`` of the error, so reaching a 1%
    residual takes ``log(0.01) / log(1 - blend_weight)`` frames.  A weight of
    1 replaces the history outright and needs a single frame.
    """
    assert frames_to_settle(blend_weight) == expected


def test_a_tighter_residual_costs_more_frames():
    """``residual`` is the knob: ask to be closer to settled, draw more."""
    assert frames_to_settle(0.1, residual=0.1) < frames_to_settle(0.1)
    assert frames_to_settle(0.1) < frames_to_settle(0.1, residual=0.001)


async def test_converged_differs_from_a_single_frame(
    controller, loaded_volume_scene, reslice
):
    """Accumulating actually changes the image; it is not a slow no-op.

    With ambient occlusion enabled the kernel rotates each frame, so the
    settled image is the average of many rotations rather than one sample.
    """
    await reslice(controller, loaded_volume_scene.id)
    controller.ambient_occlusion_enabled = True
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]

    single = controller.screenshot(canvas_id, size=(96, 72), frames=1)
    settled = controller.screenshot(canvas_id, size=(96, 72), frames="converged")

    assert not np.array_equal(single, settled)


async def test_a_scene_needing_too_many_frames_raises(
    controller, loaded_volume_scene, reslice
):
    """Exhaustion raises, and the message says exactly how to proceed.

    Silently returning an unsettled frame would break the contract the whole
    method exists to make, and would do it invisibly.
    """
    await reslice(controller, loaded_volume_scene.id)
    controller.temporal_blend_weight = 0.02
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]

    with pytest.raises(RuntimeError, match="needs 228 frames"):
        controller.screenshot(canvas_id, size=(32, 32), frames="converged")


async def test_raising_max_frames_lets_a_slow_accumulator_settle(
    controller, loaded_volume_scene, reslice
):
    """The way out the error names actually works."""
    await reslice(controller, loaded_volume_scene.id)
    controller.temporal_blend_weight = 0.02
    canvas_id = controller.get_canvas_ids(loaded_volume_scene.id)[0]

    frame = controller.screenshot(
        canvas_id, size=(32, 32), frames="converged", max_frames=300
    )

    assert frame.shape == (32, 32, 4)


# ---------------------------------------------------------------------------
# PNG output
# ---------------------------------------------------------------------------


def test_write_png_round_trips(tmp_path):
    """The stdlib PNG writer produces a file that reads back unchanged."""
    imageio = pytest.importorskip("imageio.v3")
    image = np.zeros((7, 13, 4), dtype=np.uint8)
    image[..., 0] = 200
    image[..., 3] = 255
    image[2:5, 3:9, 1] = 128
    path = tmp_path / "frame.png"

    write_png(path, image)

    assert np.array_equal(imageio.imread(path), image)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.float32),
    ],
)
def test_write_png_rejects_the_wrong_array(tmp_path, bad):
    """Anything but an ``(h, w, 4)`` uint8 array is refused, not written badly."""
    with pytest.raises(ValueError, match="expects an"):
        write_png(tmp_path / "bad.png", bad)
