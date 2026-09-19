"""Tests for ``cellier.convenience.capture``, the headless capture command.

What these cover is its load order.  Slice requests are planned per canvas from its
camera, so a multiscale visual has to be resliced for the *fitted* camera;
resliced before the fit, it loads only the tiles the unfitted camera happened
to see.  In-memory images load a whole texture and cannot show the bug, so
the fixtures here are multiscale.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import tensorstore as ts

from cellier.convenience import Viewer, capture
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import MultiscaleImageAppearance, MultiscaleImageSingleAppearance

#: Canvas and capture size.  Large enough that the unfitted camera sees only
#: part of the data (a small canvas happens to see all of it and hides the
#: bug).  The fitted image spans roughly pixels 88..512.
_SIZE = (600, 600)


@pytest.fixture
def capture_module():
    """The ``cellier.convenience.capture`` module."""
    return capture


@pytest.fixture
def bright_multiscale_store(tmp_path) -> MultiscaleZarrDataStore:
    """A 2-level ``(8, 128, 128)`` multiscale image ramping from 0 to 1 along x.

    128 voxels across is four 32-voxel tiles per axis at level 0.  The ramp is
    what makes a missing tile visible: a tile that never loaded samples a
    cache slot holding some *other* tile's data, so uniform data would look
    fully loaded even on the bug.  With a ramp, a stray tile breaks the
    left-to-right increase.  Level 1 is the level-0 ramp averaged 2x, so it
    is a ramp too and a legitimately coarser tile still reads as increasing.
    """
    level0 = np.broadcast_to(
        np.linspace(0.0, 1.0, 128, dtype=np.float32), (8, 128, 128)
    ).copy()
    level1 = level0.reshape(4, 2, 64, 2, 64, 2).mean(axis=(1, 3, 5))
    for name, data in (("s0", level0), ("s1", level1)):
        shape = data.shape
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(tmp_path / name)},
            "metadata": {
                "shape": list(shape),
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [4, 32, 32]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        ts.open(spec).result()[...].write(data).result()
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        name="bright_multiscale",
    )


@pytest.fixture
def multiscale_viewer(bright_multiscale_store, offscreen_gpu) -> Viewer:
    """A headless 2D ``Viewer`` holding the bright multiscale image, no canvas."""
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="2d", gui="offscreen")
    viewer.add_image_multiscale(
        bright_multiscale_store,
        appearance=MultiscaleImageAppearance(),
        single=MultiscaleImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    return viewer


def _load(capture_module, viewer: Viewer) -> None:
    """Run the script's canvas + load steps, as ``main`` does."""
    capture_module._ensure_canvases(viewer, _SIZE)
    asyncio.run(capture_module._load_data(viewer))


def test_cameras_are_fitted_before_the_first_reslice(
    capture_module, multiscale_viewer, monkeypatch
):
    """The first reslice plans against a camera that has already been fitted."""
    controller = multiscale_viewer.controller
    calls: list[str] = []
    real_fit = controller.fit_camera
    real_reslice = controller.reslice_all

    def fit_camera(*args, **kwargs):
        calls.append("fit")
        return real_fit(*args, **kwargs)

    def reslice_all(*args, **kwargs):
        calls.append("reslice")
        return real_reslice(*args, **kwargs)

    monkeypatch.setattr(controller, "fit_camera", fit_camera)
    monkeypatch.setattr(controller, "reslice_all", reslice_all)

    _load(capture_module, multiscale_viewer)

    assert "reslice" in calls
    assert "fit" in calls
    assert calls.index("fit") < calls.index("reslice")


def test_a_2d_multiscale_capture_loads_the_whole_view(
    capture_module, multiscale_viewer
):
    """Every tile of the fitted image holds its own data, not just one corner.

    Two rows across the image, seven samples each 60 px apart -- about 18
    voxels, so the samples span every 32-voxel tile.  The data ramps along x,
    so a fully loaded row is strictly monotonic; a tile showing another
    tile's data breaks that (on the bug the row reads flat).  The direction is
    not asserted: it depends on the colormap and the camera's x orientation,
    neither of which this test is about.
    """
    _load(capture_module, multiscale_viewer)

    frame = multiscale_viewer.screenshot(size=_SIZE)

    columns = list(range(120, 481, 60))
    for y in (200, 400):
        row = [int(frame[y, x, 0]) for x in columns]
        steps = [b - a for a, b in zip(row, row[1:])]
        assert all(s > 0 for s in steps) or all(s < 0 for s in steps), (
            f"row y={y} is not strictly monotonic at x={columns}: {row}"
        )
