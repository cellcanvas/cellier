"""Shared scenes for the MaskedImage render tests (masked image design v2).

Both store kinds go through the same helpers, so a parametrized test over
``"memory"`` and ``"lazy"`` is the check that one model serves both (MI-D3).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from cellier.convenience import Viewer
from cellier.data import ImageMemoryStore, MultiscaleZarrDataStore
from cellier.scene import BackgroundAppearance
from cellier.scene.dims import spatial_axes
from cellier.visuals import MaskedImageRenderConfig
from tests.render.conftest import _write_multiscale_zarr

if TYPE_CHECKING:
    from pathlib import Path

#: Small budgets: each side allocates its whole budget (G3).
SMALL_MASKED = {
    "gpu_budget_bytes": 16 * 1024**2,
    "value_gpu_budget_bytes": 16 * 1024**2,
}
SHAPE = (24, 32, 40)


def masked_config(**fields) -> MaskedImageRenderConfig:
    """A render config with small budgets and 16-voxel bricks."""
    return MaskedImageRenderConfig(**{**SMALL_MASKED, "block_size": 16, **fields})


def new_viewer(dim: str, time: bool = False) -> Viewer:
    """A black offscreen viewer of world ``zyx`` (``tzyx`` with *time*)."""
    axes = (*((("t", "time"),) if time else ()), *spatial_axes("z", "y", "x"))
    displayed = ("y", "x") if dim == "2d" else ("z", "y", "x")
    viewer = Viewer(axes, displayed_axes=displayed, gui="offscreen")
    viewer.background = BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 0.0, 1.0))
    return viewer


def lazy_store(root: Path, name: str, data: np.ndarray) -> MultiscaleZarrDataStore:
    """A two-level zarr store of *data*; level 1 halves the last three axes."""
    root = root / name
    root.mkdir(parents=True, exist_ok=True)
    lead = data.ndim - 3
    coarse = data[(slice(None),) * lead + (slice(None, None, 2),) * 3]

    def fill(arr):
        arr[...] = data if arr.shape == data.shape else coarse

    _write_multiscale_zarr(
        root,
        levels=[("s0", data.shape), ("s1", coarse.shape)],
        fill=fill,
        dtype=str(data.dtype),
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0", "s1"],
        level_scales=[(1.0,) * data.ndim, (1.0,) * lead + (2.0, 2.0, 2.0)],
        level_translations=[(0.0,) * data.ndim, (0.0,) * lead + (0.5, 0.5, 0.5)],
        name=name,
    )


def store(kind: str, root: Path, name: str, data: np.ndarray):
    """*data* as an in-memory (``"memory"``) or a two-level zarr store."""
    if kind == "memory":
        return ImageMemoryStore(data=data, name=name)
    return lazy_store(root, name, data)


def random_pair(seed: int = 0, shape=SHAPE) -> tuple[np.ndarray, np.ndarray]:
    """Non-uniform float32 mask and value arrays (unloaded tiles show others)."""
    rng = np.random.default_rng(seed)
    return (
        rng.random(shape).astype(np.float32),
        rng.random(shape).astype(np.float32),
    )


def gfx_visual(viewer: Viewer, visual_id):
    """The render visual of *visual_id*."""
    scenes = viewer.controller._render_manager._scenes
    return scenes[viewer.scene.id].get_visual(visual_id)


def load(viewer: Viewer, *, slice_z: float | None = None, size=(256, 256)) -> None:
    """Give the scene a canvas, set the 2D slice, and load to quiescence."""
    from cellier.convenience.capture import _ensure_canvases, _load_data

    _ensure_canvases(viewer, size)

    async def _run() -> None:
        if slice_z is not None:
            viewer.set_slice_positions(
                {viewer.scene.dims.world_coordinate_system.ndim - 3: float(slice_z)}
            )
        await _load_data(viewer)

    asyncio.run(_run())


def delay(data_store, seconds: float) -> None:
    """Make every read of *data_store* wait *seconds* first."""
    original = data_store.get_data

    async def slow(request):
        await asyncio.sleep(seconds)
        return await original(request)

    object.__setattr__(data_store, "get_data", slow)


def undelay(data_store) -> None:
    """Undo :func:`delay`."""
    object.__delattr__(data_store, "get_data")


# -- the synthetic example's pair (masked image design v2, 11) -----------------

#: The ellipsoid's semi-axes, ``(z, y, x)``, as fractions of the mask's side.
SEMI_AXES = (110 / 256, 90 / 256, 70 / 256)
#: Width of one value band, in mask voxels.
BAND_WIDTH = 8.0


def _ellipsoid_inside(side: int, scale: int = 1) -> np.ndarray:
    n = side * scale
    centre = (side - 1) / 2.0
    coords = (np.arange(n, dtype=np.float32) + 0.5) / scale - 0.5 - centre
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij", sparse=True)
    a, b, c = (s * side for s in SEMI_AXES)
    return (z / a) ** 2 + (y / b) ** 2 + (x / c) ** 2 <= 1.0


def band_pair(side: int, value_scale: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The example's ``(mask, value)``: an ellipsoid's distance map and its bands.

    The mask is the distance to the ellipsoid's boundary in mask voxels, zero
    outside; the value is ``floor(distance / BAND_WIDTH) + 1`` inside, zero
    outside, on a grid of ``value_scale`` voxels a mask voxel.  A surface at
    threshold ``iso`` lies in band ``floor(iso / BAND_WIDTH) + 1``.
    """
    from scipy import ndimage as ndi

    inside = _ellipsoid_inside(side)
    mask = ndi.distance_transform_edt(inside).astype(np.float32)
    if value_scale == 1:
        distance, value_inside = mask, inside
    else:
        value_inside = _ellipsoid_inside(side, value_scale)
        distance = (
            ndi.distance_transform_edt(value_inside).astype(np.float32) / value_scale
        )
    value = np.where(value_inside, np.floor(distance / BAND_WIDTH) + 1.0, 0.0)
    return mask, value.astype(np.float32)


def value_to_mask_matrix(value_scale: int) -> np.ndarray | None:
    """Value level-0 voxels to mask level-0 voxels, or ``None`` on one grid."""
    if value_scale == 1:
        return None
    matrix = np.eye(4)
    factor = 1.0 / value_scale
    matrix[:3, :3] *= factor
    matrix[:3, 3] = 0.5 * factor - 0.5
    return matrix


#: One primary a band: band 1 red, 2 green, 3 blue (clim 0.5 to 3.5).
BAND_COLORS = ("red", "lime", "blue")


def band_colormap():
    """A nearest colormap of :data:`BAND_COLORS`."""
    import cmap

    return cmap.Colormap(list(BAND_COLORS), interpolation="nearest")


def band_counts(frame: np.ndarray) -> np.ndarray:
    """Drawn pixels per band, read as each pixel's brightest channel.

    Phong shading scales a primary and adds a grey highlight, so the
    brightest channel is the band whatever the lighting.
    """
    rgb = frame[..., :3].astype(np.int64)
    drawn = rgb.max(axis=-1) > 8
    return np.bincount(np.argmax(rgb, axis=-1)[drawn], minlength=3)


# -- pyramids whose coarse levels are shorter (masked image design 6.4) --------


def halve(data: np.ndarray, round_up: bool = False) -> np.ndarray:
    """A 2x block mean of *data*.

    ``round_up=False`` drops an odd last voxel (floor halving, as
    ``block_mean`` does), so the level covers less than the one before;
    ``round_up=True`` pads it first with the array's minimum, the outside
    value of a signed distance.
    """
    if round_up:
        pad = [(0, n % 2) for n in data.shape]
        data = np.pad(data, pad, mode="constant", constant_values=data.min())
    even = tuple(n // 2 * 2 for n in data.shape)
    data = data[: even[0], : even[1], : even[2]]
    shape = (even[0] // 2, 2, even[1] // 2, 2, even[2] // 2, 2)
    return data.reshape(shape).mean(axis=(1, 3, 5)).astype(data.dtype)


def pyramid_store(
    root: Path, name: str, data: np.ndarray, n_levels: int, round_up: bool = False
) -> MultiscaleZarrDataStore:
    """A zarr store of *data* halved ``n_levels - 1`` times (see :func:`halve`)."""
    levels = [data]
    for _ in range(n_levels - 1):
        levels.append(halve(levels[-1], round_up))
    pending = iter(levels)

    def fill(arr):
        arr[...] = next(pending)

    _write_multiscale_zarr(
        root / name,
        levels=[(f"s{k}", level.shape) for k, level in enumerate(levels)],
        fill=fill,
        dtype=str(data.dtype),
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root / name),
        scale_names=[f"s{k}" for k in range(n_levels)],
        level_scales=[(2.0**k,) * 3 for k in range(n_levels)],
        level_translations=[(0.5 * (2.0**k - 1),) * 3 for k in range(n_levels)],
        name=name,
    )


# -- target_bricks="surface" (masked image design 8.3) ---------------------------

#: The spike of :func:`spiked_pair`: radius and length past the ellipsoid, in
#: mask voxels, along +x from the centre.
SPIKE_RADIUS = 1.5
SPIKE_LENGTH = 20


def spiked_pair(side: int, spike: bool) -> tuple[np.ndarray, np.ndarray]:
    """:func:`band_pair`'s ellipsoid, optionally with a thin spike on +x.

    The spike is the case design 3.1 documents: its coarse mean falls below
    the threshold, so the coarse pass never sees it.  The value is the bands
    of the mask's own distance.
    """
    from scipy import ndimage as ndi

    inside = _ellipsoid_inside(side)
    if spike:
        centre = (side - 1) / 2.0
        z, y, x = np.ogrid[:side, :side, :side]
        x_end = centre + SEMI_AXES[2] * side + SPIKE_LENGTH
        tube = (y - centre) ** 2 + (z - centre) ** 2 <= SPIKE_RADIUS**2
        inside = inside | (tube & (x >= centre) & (x <= x_end))
    mask = ndi.distance_transform_edt(inside).astype(np.float32)
    value = np.where(inside, np.floor(mask / BAND_WIDTH) + 1.0, 0.0)
    return mask, value.astype(np.float32)


def record_targets(gfx) -> dict[str, list[np.ndarray]]:
    """Record every 3D target plan each side of *gfx* keeps, after its filter.

    Returns a dict ``{"mask": [...], "value": [...]}`` that each full plan
    appends its kept ``(N, 4)`` rows to (``None`` for a held-back target).
    """
    out: dict[str, list] = {"mask": [], "value": []}
    mask_filter = gfx._filter_target_3d
    value = gfx.value_side
    value_filter = value.target_filter

    def on_mask(planner, arr, config, fill):
        kept = mask_filter(planner, arr, config, fill)
        out["mask"].append(None if kept is None else np.array(kept))
        return kept

    def on_value(planner, arr):
        kept = value_filter(planner, arr)
        out["value"].append(None if kept is None else np.array(kept))
        return kept

    gfx._filter_target_3d = on_mask
    value.target_filter = on_value
    return out
