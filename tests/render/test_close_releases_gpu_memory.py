"""``close()`` must free the large GPU-side buffers without the cycle collector.

A multiscale visual's brick cache is a CPU array plus a texture of up to the
whole ``gpu_budget_bytes`` (1 GiB by default), and an in-memory visual uploads
its data as a texture.  Before ``close()`` released them explicitly they were
freed only once the closed controller itself became unreachable -- and the
controller sits in reference cycles, so that waited on a full collection.
Python schedules those by object count, not bytes, so gigabytes piled up
between collections; on Windows, where ``np.zeros`` commits its memory up
front, CI ran out of memory and the software Vulkan device was lost.

These tests keep the closed controller referenced (as a cycle, a pending task
or a pytest traceback would) and switch the collector off, so the only way a
texture can die is by ``close()`` dropping the references to it.
"""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx
import pytest

from cellier.controller import CellierController
from cellier.visuals import (
    MultiscaleImageRenderConfig,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests.conftest import _track_instances

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from uuid import UUID

# Small budgets: the claim is about release, not size, and a 1 GiB cache per
# slot is what exhausted CI in the first place.
_BUDGET_3D = 4 * 1024**2
_BUDGET_2D = 1024**2


@contextmanager
def _collector_off() -> Iterator[None]:
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _add_image(controller: CellierController, request) -> UUID:
    scene = controller.add_scene(dim="3d")
    controller.add_image(request.getfixturevalue("image_volume"), scene.id)
    return scene.id


def _add_labels(controller: CellierController, request) -> UUID:
    scene = controller.add_scene(dim="3d")
    controller.add_labels(request.getfixturevalue("labels_volume"), scene.id)
    return scene.id


def _add_image_multiscale(controller: CellierController, request) -> UUID:
    scene = controller.add_scene(dim="3d")
    controller.add_image_multiscale(
        request.getfixturevalue("multiscale_image_store"),
        scene.id,
        render_config=MultiscaleImageRenderConfig(
            block_size=8, gpu_budget_bytes=_BUDGET_3D, gpu_budget_bytes_2d=_BUDGET_2D
        ),
    )
    return scene.id


def _add_labels_multiscale(controller: CellierController, request) -> UUID:
    scene = controller.add_scene(dim="3d")
    controller.add_labels_multiscale(
        request.getfixturevalue("multiscale_labels_store"),
        scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(
            block_size=8, gpu_budget_bytes=_BUDGET_3D, gpu_budget_bytes_2d=_BUDGET_2D
        ),
    )
    return scene.id


_BUILDERS: dict[str, Callable[[CellierController, pytest.FixtureRequest], UUID]] = {
    "image": _add_image,
    "labels": _add_labels,
    "image_multiscale": _add_image_multiscale,
    "labels_multiscale": _add_labels_multiscale,
}


def _build(request, kind: str) -> tuple[CellierController, UUID]:
    """A controller with one *kind* visual on a canvas, its first slice queued."""
    controller = CellierController(gui="offscreen")
    controller.camera_reslice_enabled = False
    scene_id = _BUILDERS[kind](controller, request)
    controller.add_canvas(scene_id)
    controller.fit_camera(scene_id)
    controller.reslice_all()
    return controller, scene_id


async def _build_and_draw(
    request, drive_reslice, kind: str
) -> tuple[CellierController, UUID]:
    """A controller with one *kind* visual, sliced and drawn once."""
    controller, scene_id = _build(request, kind)
    await drive_reslice(controller)
    # Draw through the canvas's own renderer, so every texture has its GPU
    # object and the renderer has seen the scene.
    for canvas_view in controller._render_manager._canvases.values():
        canvas_view._canvas.draw()
    return controller, scene_id


def _describe(texture: gfx.Texture) -> str:
    holders = sorted(
        {type(ref).__name__ for ref in gc.get_referrers(texture)} - {"list", "frame"}
    )
    return f"{texture.size} {texture.format} held by {holders}"


def _process_wide(texture: gfx.Texture) -> bool:
    """True for a pygfx colormap: built on first use, then kept by the module."""
    return any(
        isinstance(cmap, gfx.TextureMap) and cmap.texture is texture
        for cmap in vars(gfx.cm).values()
    )


def _live(textures: list[weakref.ref]) -> list[gfx.Texture]:
    return [
        texture
        for ref in textures
        if (texture := ref()) is not None and not _process_wide(texture)
    ]


def _survivors(textures: list[weakref.ref]) -> list[str]:
    return [_describe(texture) for texture in _live(textures)]


def _array_refs(textures: list[weakref.ref]) -> list[weakref.ref]:
    """Weakrefs to the CPU arrays behind every live texture."""
    return [
        weakref.ref(texture.data)
        for texture in _live(textures)
        if isinstance(texture.data, np.ndarray)
    ]


@pytest.fixture(params=sorted(_BUILDERS))
def kind(request) -> str:
    return request.param


async def test_close_frees_every_texture_and_its_array(
    kind, request, monkeypatch, drive_reslice
):
    textures = _track_instances(monkeypatch, gfx.Texture)
    with _collector_off():
        controller, _scene_id = await _build_and_draw(request, drive_reslice, kind)
        assert textures, "the visual built no textures -- the test proves nothing"
        arrays = _array_refs(textures)

        controller.close()

        assert _survivors(textures) == []
        assert [ref for ref in arrays if ref() is not None] == []
    # The closed controller is still referenced here, deliberately.
    assert controller is not None


async def test_close_mid_slice_frees_every_texture(kind, request, monkeypatch):
    """Close while slice tasks are still in flight.

    ``close()`` cancels them, but a cancelled task only runs its cancellation
    once the loop turns, and until then its coroutine frame -- and whatever
    visual or slot it was loading into -- is still referenced.
    """
    textures = _track_instances(monkeypatch, gfx.Texture)
    with _collector_off():
        controller, _scene_id = _build(request, kind)
        in_flight = list(controller._render_manager._slicer._tasks.values())
        assert in_flight, "no slice was in flight -- the test proves nothing"

        controller.close()

        assert _survivors(textures) == []
        del in_flight


async def test_remove_visual_frees_its_textures(
    kind, request, monkeypatch, drive_reslice
):
    textures = _track_instances(monkeypatch, gfx.Texture)
    with _collector_off():
        controller, scene_id = await _build_and_draw(request, drive_reslice, kind)
        (visual_id,) = controller._render_manager._scenes[scene_id].visual_ids

        controller.remove_visual(visual_id)

        assert _survivors(textures) == []
    controller.close()


async def test_remove_scene_frees_its_textures(
    kind, request, monkeypatch, drive_reslice
):
    textures = _track_instances(monkeypatch, gfx.Texture)
    with _collector_off():
        controller, scene_id = await _build_and_draw(request, drive_reslice, kind)

        controller.remove_scene(scene_id)

        assert _survivors(textures) == []
    controller.close()
