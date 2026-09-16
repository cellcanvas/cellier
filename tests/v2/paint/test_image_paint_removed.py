"""Image visuals cannot be painted; only labels can (design D5, section 3.12)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.image._ome_zarr_image_store import OMEZarrImageDataStore
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from cellier.visuals._image_memory import InMemoryImageAppearance

if TYPE_CHECKING:
    from pathlib import Path


def _scene(controller: CellierController):
    return controller.add_scene(
        dim="2d",
        coordinate_system=world_coordinate_system(spatial_axes("y", "x")),
        name="paint_scene",
        render_modes={"2d"},
    )


def _make_zarr(path: Path, shape: tuple[int, int] = (64, 64)) -> Path:
    """Create a tiny single-level OME-Zarr v0.5 float32 image."""
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_array("s0", shape=shape, chunks=(32, 32), dtype=np.float32)
    arr[:] = np.zeros(shape, dtype=np.float32)
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0]}
                        ],
                    }
                ],
                "name": "test",
            }
        ],
    }
    return path


async def test_in_memory_image_paint_raises(qtbot):
    controller = CellierController()
    scene = _scene(controller)
    visual = controller.add_image(
        data=ImageMemoryStore(data=np.zeros((32, 32), dtype=np.float32)),
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]

    with pytest.raises(TypeError, match="No PaintController"):
        controller.add_paint_controller(visual_id=visual.id, canvas_id=canvas_id)


async def test_multiscale_image_paint_raises(qtbot, tmp_path):
    zarr_path = _make_zarr(tmp_path / "image.ome.zarr")
    data_store = OMEZarrImageDataStore.from_path(
        f"file://{zarr_path.resolve()}", name="t"
    )
    controller = CellierController()
    scene = _scene(controller)
    visual = controller.add_image_multiscale(
        data=data_store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(),
        render_config=MultiscaleImageRenderConfig(block_size=16),
        single=MultiscaleImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]

    with pytest.raises(TypeError, match="No PaintController"):
        controller.add_paint_controller(visual_id=visual.id, canvas_id=canvas_id)

    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    assert not hasattr(gfx_visual, "patch_paint_texture")
    assert not hasattr(gfx_visual._inner_node_2d.material, "paint_cache_texture")
