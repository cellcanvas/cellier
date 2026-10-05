"""Shared helpers for the clipping planes render tests.

A clipped visual is compared with the same visual drawn from a store whose
clipped voxels were zeroed, under one camera.  A voxel-centre mask and an
exact plane may differ within a voxel of the cut, so pixels are counted off
a band a few pixels wide round every edge of the reference.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from cellier.data import ImageMemoryStore, LabelMemoryStore, MultiscaleZarrDataStore
from cellier.transform import AffineTransform
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageSingleAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import _write_multiscale_zarr

N = 32
Z_SCALE = 2.0
X_SHIFT = 5.0
BAND = 3
#: A colour channel differing by more than this is a different pixel.
COLOUR_TOL = 24
VIRIDIS_ZERO = np.array([68, 1, 84], dtype=np.int16)
#: The data's centre in the scene, pygfx (x, y, z).
CENTRE = np.array([15.5 + X_SHIFT, 15.5, 15.5 * Z_SCALE])

#: ``(point, normal)`` pairs over data ``(z, y, x)``; the normal's side is kept.
#: The axis-aligned ones sit on voxel boundaries (a voxel is centred on its
#: index), where a voxel mask and the plane cut in the same place.
PLANES = {
    "axis": [((0, 0, 15.5), (0, 0, 1))],
    "oblique": [((16, 16, 16), (-0.6, 0.5, 1.0))],
    "along_z": [((15.5, 0, 0), (1, 0, 0))],
    "slab": [((0, 0, 11.5), (0, 0, 1)), ((0, 0, 21.5), (0, 0, -1))],
}

KINDS = [
    "image_memory_mip",
    "image_memory_iso",
    "image_multiscale_mip",
    "image_multiscale_iso",
    "labels_memory",
    "labels_multiscale",
]


def image_data(n: int = N) -> np.ndarray:
    """An ellipsoid whose brightness ramps along x: not uniform anywhere."""
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    c = (n - 1) / 2
    r = np.sqrt(((z - c) / 0.9) ** 2 + (y - c) ** 2 + ((x - c) / 0.8) ** 2)
    return np.where(r < 0.42 * n, 0.55 + 0.45 * x / n, 0.0).astype(np.float32)


def label_data(n: int = N) -> np.ndarray:
    data = np.zeros((n, n, n), dtype=np.int32)
    q = n // 8
    data[q : 4 * q, q : 7 * q, q : 7 * q] = 3
    data[4 * q : 7 * q, q : 7 * q, q : 7 * q] = 7
    return data


def data_for(kind: str) -> np.ndarray:
    return label_data() if kind.startswith("labels") else image_data()


def kept_mask(planes, n: int = N) -> np.ndarray:
    """Per voxel centre, in data ``(z, y, x)``: is it kept by every plane?"""
    index = np.stack(np.mgrid[:n, :n, :n], axis=-1).astype(np.float64)
    kept = np.ones((n, n, n), dtype=bool)
    for point, normal in planes:
        kept &= index @ np.asarray(normal, float) >= float(np.dot(normal, point))
    return kept


def clipping_planes(store, planes) -> tuple[ClippingPlane, ...]:
    system = store.data_coordinate_systems[0]
    return tuple(
        ClippingPlane.from_point_normal(system, point, normal, axes=("z", "y", "x"))
        for point, normal in planes
    )


def _multiscale(tmp_path, name, data) -> MultiscaleZarrDataStore:
    root = tmp_path / name
    root.mkdir()
    coarse = data[::2, ::2, ::2]

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
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        name=name,
    )


def _transform(controller, scene_id, store):
    controller._ensure_data_coordinate_systems(scene_id, store)
    data = store.data_coordinate_systems[0]
    world = controller._model.scenes[scene_id].dims.world_coordinate_system
    return AffineTransform.from_axis_map(
        data,
        world,
        {data.axis_by_name(n).id: world.axis_by_name(n).id for n in "zyx"},
        scale={data.axis_by_name("z").id: Z_SCALE},
        translation={data.axis_by_name("x").id: X_SHIFT},
    )


def add_visual(kind, controller, scene_id, data, tmp_path, tag):
    """Add one visual of *kind*; returns ``(visual model, store)``.

    Anisotropic and translated: world ``z = 2 * data z``, ``x = data x + 5``.
    """
    if kind == "labels_multiscale_smooth_iso":
        store = _multiscale(tmp_path, tag, data)
        visual = controller.add_labels_multiscale(
            data=store,
            scene_id=scene_id,
            appearance=MultiscaleLabelsAppearance(
                force_level=0, render_mode="smooth_iso"
            ),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=_transform(controller, scene_id, store),
        )
        return visual, store
    if kind.startswith("image_memory"):
        mode = kind.rsplit("_", 1)[1]
        store = ImageMemoryStore(data=data, name=tag)
        visual = controller.add_image(
            data=store,
            scene_id=scene_id,
            single=InMemoryImageSingleAppearance(
                color_map="viridis",
                clim=(0.0, 1.0),
                render_mode=mode,
                iso_threshold=0.3,
            ),
            transform=_transform(controller, scene_id, store),
        )
    elif kind.startswith("image_multiscale"):
        mode = kind.removeprefix("image_multiscale_")
        store = _multiscale(tmp_path, tag, data)
        visual = controller.add_image_multiscale(
            data=store,
            scene_id=scene_id,
            appearance=MultiscaleImageAppearance(force_level=0),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                color_map="viridis",
                clim=(0.0, 1.0),
                render_mode=mode,
                iso_threshold=0.3,
            ),
            transform=_transform(controller, scene_id, store),
        )
    elif kind == "labels_memory":
        store = LabelMemoryStore(data=data, name=tag)
        visual = controller.add_labels(
            data=store,
            scene_id=scene_id,
            transform=_transform(controller, scene_id, store),
        )
    elif kind == "labels_multiscale":
        store = _multiscale(tmp_path, tag, data)
        visual = controller.add_labels_multiscale(
            data=store,
            scene_id=scene_id,
            appearance=MultiscaleLabelsAppearance(force_level=0),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=_transform(controller, scene_id, store),
        )
    else:
        raise ValueError(kind)
    return visual, store


def gfx_scene(controller, scene_id):
    """``(pygfx scene, camera)`` of a scene's first canvas, background hidden."""
    controller.update_background_field(scene_id, "visible", False)
    canvas_id = controller.get_canvas_ids(scene_id)[0]
    view = controller._render_manager._canvases[canvas_id]
    return view._get_scene_fn(scene_id), view.camera


def camera_views(camera):
    """``(name, place)`` pairs: camera placements to compare under."""
    fitted = np.array(camera.local.position, dtype=float)
    distance = float(np.linalg.norm(fitted - CENTRE))

    def fit():
        camera.local.position = tuple(fitted)
        camera.look_at(tuple(CENTRE))

    def side():
        direction = np.array([-0.7, 0.45, 0.55])
        camera.local.position = tuple(
            CENTRE + direction / np.linalg.norm(direction) * distance
        )
        camera.look_at(tuple(CENTRE))

    def inside():
        camera.local.position = tuple(CENTRE + np.array([-9.0, 2.0, 3.0]))
        camera.look_at(tuple(CENTRE + np.array([10.0, 3.0, -4.0])))

    return [("fit", fit), ("side", side), ("inside", inside)]


def hide_colormap_zero(frame: np.ndarray) -> np.ndarray:
    """Count a pixel at the colormap's zero as not drawn.

    An image draws its zeros, so a masked voxel is still a drawn pixel: an
    in-memory MIP fills its whole proxy box.  The box's antialiased edge is
    neither the zero colour nor data, and a clipped box has its edge in a
    different place, so partly covered pixels are dropped too.
    """
    zero = np.abs(frame[..., :3].astype(np.int16) - VIRIDIS_ZERO).max(-1) < 8
    frame[zero | (frame[..., 3] < 255)] = 0
    return frame


def compare(clipped: np.ndarray, reference: np.ndarray) -> dict:
    """Pixels where two frames disagree, counted off the edge band.

    ``silhouette``: where something is drawn.  ``colour``: any channel
    further apart than ``COLOUR_TOL``.
    """
    drawn, wanted = clipped[..., 3] > 0, reference[..., 3] > 0
    ref = reference.astype(np.int16)
    step = np.zeros(wanted.shape, dtype=bool)
    for axis in (0, 1):
        jump = np.abs(np.diff(ref, axis=axis)).max(axis=-1) > COLOUR_TOL
        pad = [(0, 0), (0, 0)]
        pad[axis] = (0, 1)
        step |= np.pad(jump, pad)
    edge = binary_dilation(step, iterations=BAND) | (
        binary_dilation(wanted, iterations=BAND)
        & ~binary_erosion(wanted, iterations=BAND)
    )
    silhouette = drawn != wanted
    colour = np.abs(clipped.astype(np.int16) - ref).max(axis=-1) > COLOUR_TOL
    return {
        "reference_px": int(wanted.sum()),
        "clipped_px": int(drawn.sum()),
        "silhouette_off_edge": int((silhouette & ~edge).sum()),
        "colour_off_edge": int((colour & ~edge).sum()),
    }
