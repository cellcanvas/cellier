"""Analytic oracles for transformed rendering (full affine design v2, D10).

Every expectation here is numpy on the transform and the camera, or a
property of the data -- nothing is rendered by another library and no
silhouette is stored (decision D10, thresholds from the Phase 0 readout,
section 4):

* **landmark** -- a small block / point / cross at a known data position
  ``P0``; its rendered centroid must sit where the camera's own
  view-projection puts ``L p + t``, within ``LANDMARK_PX`` at ``SIZE`` px.
  Under perspective a box's silhouette centroid is not the projection of its
  centre, so a box's expectation is the area centroid of the convex hull of
  its projected corners.
* **picks** -- picks inside a silhouette decode to the known voxel.

The data is world ``zyx``; a transform is ``world = L (data - c) + c`` with
``c`` the data centre, so every transform keeps the landmark near where the
identity puts it.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas

from cellier.convenience import Viewer
from cellier.convenience.capture import _ensure_canvases, _load_data
from cellier.data import (
    ImageMemoryStore,
    LabelMemoryStore,
    MultiscaleZarrDataStore,
)
from cellier.data.graph._graph_memory_store import GraphMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.mesh._mesh_multiscale_store import MeshLevel, MultiscaleMeshStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.scene import BackgroundAppearance
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    LinesMemoryAppearance,
    MeshFlatAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    PointsMarkerAppearance,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import _write_multiscale_zarr

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SHAPE = (24, 32, 40)
CENTRE = (np.asarray(SHAPE, dtype=np.float64) - 1.0) / 2.0
#: The landmark, data ``zyx``.
P0 = np.array([7.0, 20.0, 28.0])
SIZE = 960
LANDMARK_PX = 0.5
#: Half width of the landmark block: 3 voxels, edges at +/- 1.5.
BLOCK_HALF = 1.5

RASTER = ("image", "image_multiscale", "labels", "labels_multiscale")
GEOMETRY = ("mesh", "mesh_multiscale", "points", "lines", "graph")
FAMILIES = RASTER + GEOMETRY


def rot(axis_a: int, axis_b: int, degrees: float) -> np.ndarray:
    """A rotation in the plane of data axes ``a`` and ``b`` (``zyx`` indices)."""
    theta = np.deg2rad(degrees)
    m = np.eye(3)
    m[axis_a, axis_a] = np.cos(theta)
    m[axis_a, axis_b] = -np.sin(theta)
    m[axis_b, axis_a] = np.sin(theta)
    m[axis_b, axis_b] = np.cos(theta)
    return m


def shear(row: int, column: int, amount: float) -> np.ndarray:
    """``world[row] += amount * data[column]``."""
    m = np.eye(3)
    m[row, column] = amount
    return m


#: The investigation's transforms (``zyx``: 0 = z, 1 = y, 2 = x).  A 2D view
#: displays ``(y, x)`` and slices ``z``.
LINEARS: dict[str, np.ndarray] = {
    "scale": np.diag([2.0, 1.0, 1.5]),
    "rot_yx": rot(1, 2, 30.0),
    "shear_yx": shear(2, 1, 0.5),
    "rot_zx": rot(0, 2, 30.0),
    "shear_x_by_z": shear(2, 0, 0.5),
    "shear_z_by_x": shear(0, 2, 0.5),
    "rot90_yx": rot(1, 2, 90.0).round(12),
    "flip_x": np.diag([1.0, 1.0, -1.0]),
    "general": (
        rot(0, 1, 20.0) @ rot(1, 2, 25.0) @ rot(0, 2, 15.0) @ np.diag([2.0, 1, 1])
    ),
}


def world_matrix(linear: np.ndarray) -> np.ndarray:
    """The 4x4 ``data -> world`` matrix applying *linear* about the centre."""
    m = np.eye(4)
    m[:3, :3] = linear
    m[:3, 3] = CENTRE - linear @ CENTRE
    return m


def to_world(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ np.asarray(point, dtype=np.float64) + matrix[:3, 3]


# ---------------------------------------------------------------------------
# Viewers and visuals
# ---------------------------------------------------------------------------


def new_viewer(dim: str) -> Viewer:
    """A black-background offscreen viewer of world ``zyx``."""
    displayed = ("y", "x") if dim == "2d" else ("z", "y", "x")
    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=displayed, gui="offscreen"
    )
    viewer.background = BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 0.0, 1.0))
    return viewer


def bind(viewer: Viewer, store, matrix: np.ndarray) -> AffineTransform:
    """A transform from *store*'s data system to the viewer's world."""
    controller = viewer.controller
    controller._ensure_data_coordinate_systems(viewer.scene.id, store)
    data = store.data_coordinate_systems[0]
    world = viewer.scene.dims.world_coordinate_system
    return AffineTransform.from_matrix(matrix, data, world, name="data_to_world")


def block_volume(dtype, centre: np.ndarray = P0) -> np.ndarray:
    """A 3-voxel cube of ones at *centre*."""
    data = np.zeros(SHAPE, dtype=dtype)
    z, y, x = np.asarray(centre).astype(int)
    data[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 1
    return data


def multiscale_store(
    root: Path, name: str, data: np.ndarray
) -> MultiscaleZarrDataStore:
    """A two-level zarr store of *data* (level 1 is every other voxel)."""
    root = root / name
    root.mkdir(parents=True, exist_ok=True)
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


_BITS = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], float)
_CUBE_FACES = np.array(
    [
        [0, 1, 3], [0, 3, 2],
        [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4],
        [1, 5, 7], [1, 7, 3],
    ],
    dtype=np.int32,
)  # fmt: skip


def cube_corners(centre: np.ndarray = P0, half: float = BLOCK_HALF) -> np.ndarray:
    return (np.asarray(centre) - half + 2 * half * _BITS).astype(np.float32)


def cross_vertices(centre: np.ndarray = P0, half: float = 1.0) -> np.ndarray:
    """Three segments through *centre*, one per axis: a symmetric 3D cross."""
    pairs = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = half
        pairs.extend([centre - step, centre + step])
    return np.asarray(pairs, dtype=np.float32)


def add_bounds(viewer: Viewer, matrix: np.ndarray) -> None:
    """A hidden, empty image of ``SHAPE``: every family's camera fits the same box."""
    store = ImageMemoryStore(data=np.zeros(SHAPE, np.float32), name="bounds")
    visual = viewer.add_image(
        store,
        single=InMemoryImageSingleAppearance(clim=(0.0, 1.0), render_mode="mip"),
        transform=bind(viewer, store, matrix),
    )
    viewer.controller.set_visual_visible(visual.id, False)


def add_landmark(viewer: Viewer, family: str, matrix: np.ndarray, tmp: Path):
    """Add *family*'s landmark at ``P0`` under *matrix*; returns the visual."""
    if family in GEOMETRY:
        add_bounds(viewer, matrix)
    if family == "image":
        store = ImageMemoryStore(data=block_volume(np.float32), name="landmark")
        return viewer.add_image(
            store,
            # Nearest: a linearly interpolated block has rounded corners at the
            # threshold, which moves its centroid off the box hull's.
            appearance=InMemoryImageAppearance(interpolation="nearest"),
            single=InMemoryImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="mip"
            ),
            transform=bind(viewer, store, matrix),
        )
    if family == "labels":
        store = LabelMemoryStore(data=block_volume(np.int32), name="landmark")
        return viewer.add_labels(
            store,
            appearance=InMemoryLabelsAppearance(render_mode="iso_categorical"),
            transform=bind(viewer, store, matrix),
        )
    if family == "image_multiscale":
        store = multiscale_store(tmp, "landmark", block_volume(np.float32))
        return viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(force_level=0),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="mip"
            ),
            transform=bind(viewer, store, matrix),
        )
    if family == "labels_multiscale":
        store = multiscale_store(tmp, "landmark", block_volume(np.int32))
        return viewer.add_labels_multiscale(
            store,
            # Four steps per voxel: at the default one, a ray that only grazes
            # a voxel corner can step over it, which shrinks a 3-voxel
            # silhouette by up to 1.6 px under the identity alone.
            appearance=MultiscaleLabelsAppearance(
                force_level=0, ray_steps_per_voxel=4.0
            ),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=bind(viewer, store, matrix),
        )
    if family in ("mesh", "mesh_multiscale"):
        corners = cube_corners()
        appearance = MeshFlatAppearance(color=(1.0, 1.0, 1.0, 1.0))
        if family == "mesh":
            store = MeshMemoryStore(positions=corners, indices=_CUBE_FACES, name="m")
            return viewer.add_mesh(
                store, appearance=appearance, transform=bind(viewer, store, matrix)
            )
        level = MeshLevel(positions=corners, indices=_CUBE_FACES)
        store = MultiscaleMeshStore(levels=[level, level], name="m")
        return viewer.add_multiscale_mesh(
            store, appearance=appearance, transform=bind(viewer, store, matrix)
        )
    if family == "points":
        store = PointsMemoryStore(positions=P0[None].astype(np.float32), name="p")
        return viewer.add_points(
            store,
            appearance=PointsMarkerAppearance(size=9.0, color=(1, 1, 1, 1)),
            transform=bind(viewer, store, matrix),
        )
    if family == "lines":
        store = LinesMemoryStore(positions=cross_vertices(), name="l")
        return viewer.add_lines(
            store,
            appearance=LinesMemoryAppearance(color=(1, 1, 1, 1), thickness=3.0),
            transform=bind(viewer, store, matrix),
        )
    if family == "graph":
        vertices = cross_vertices()
        edges = np.arange(6, dtype=np.int64).reshape(3, 2)
        store = GraphMemoryStore.from_arrays(vertices, edges, name="g")
        return viewer.add_graph(store, transform=bind(viewer, store, matrix))
    raise ValueError(family)


# ---------------------------------------------------------------------------
# Rendering and the analytic expectation
# ---------------------------------------------------------------------------


def canvas_view(viewer: Viewer):
    controller = viewer.controller
    return controller._render_manager._canvases[
        controller.get_canvas_ids(viewer.scene.id)[0]
    ]


def load(viewer: Viewer, *, slice_z: float | None = None, half_thickness=None):
    """Give the scene a canvas, set the 2D slice, and load to quiescence."""
    _ensure_canvases(viewer, (SIZE, SIZE))

    async def _run() -> None:
        if slice_z is not None:
            viewer.set_slice_positions({0: float(slice_z)})
        if half_thickness is not None:
            viewer.controller.update_thickness(
                viewer.scene.id, {0: float(half_thickness)}
            )
        await _load_data(viewer)

    asyncio.run(_run())


def render(viewer: Viewer, size: int = SIZE):
    """Draw the scene with its own camera; ``(frame, renderer, camera)``."""
    viewer.controller.update_background_field(viewer.scene.id, "visible", False)
    view = canvas_view(viewer)
    scene = view._get_scene_fn(viewer.scene.id)
    camera = view.camera
    canvas = RenderCanvas(size=(size, size), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    canvas.request_draw(lambda: renderer.render(scene, camera))
    return np.asarray(canvas.draw()), renderer, camera


def nudge_2d_camera(camera, size: int = SIZE) -> None:
    """Move a 2D camera by a fraction of a pixel.

    A fitted 2D camera puts voxel edges exactly on pixel centres, where the
    rasteriser's tie rule decides inclusion and biases a centroid by up to
    half a pixel depending on the edge's direction.  The analytic
    expectation follows the camera, so moving it is free.
    """
    pixel = float(camera.width) / size
    x, y, z = camera.local.position
    camera.local.position = (x + 0.31 * pixel, y + 0.17 * pixel, z)


def project(camera, world_zyx: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Pixel ``(row, col)`` of a world point, pixel centres at integers."""
    h = np.r_[np.asarray(world_zyx, dtype=np.float64)[::-1], 1.0]
    clip = np.asarray(camera.camera_matrix, dtype=np.float64) @ h
    ndc = clip[:3] / clip[3]
    col = (ndc[0] + 1.0) / 2.0 * size - 0.5
    row = (1.0 - ndc[1]) / 2.0 * size - 0.5
    return np.array([row, col])


def drawn(frame: np.ndarray) -> np.ndarray:
    return frame[..., :3].astype(np.float64).max(axis=-1) > 20


def centroid(frame: np.ndarray) -> np.ndarray | None:
    """The drawn pixels' centroid ``(row, col)``; ``None`` when nothing drew."""
    rows, cols = np.nonzero(drawn(frame))
    if rows.size == 0:
        return None
    return np.array([rows.mean(), cols.mean()])


def hull_centroid(points: np.ndarray) -> np.ndarray:
    """Area centroid of the convex hull of 2D *points* (monotone chain)."""
    pts = sorted(map(tuple, np.asarray(points, dtype=np.float64)))

    def half(sequence):
        out: list = []
        for p in sequence:
            while (
                len(out) >= 2
                and (
                    (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                    - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])
                )
                <= 0
            ):
                out.pop()
            out.append(p)
        return out[:-1]

    hull = np.array(half(pts) + half(pts[::-1]))
    x, y = hull[:, 0], hull[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    return np.array(
        [((x + xn) * cross).sum() / (6 * area), ((y + yn) * cross).sum() / (6 * area)]
    )


def expected_centroid(
    family: str, camera, matrix: np.ndarray, *, dim: str = "3d", slice_z=None
) -> np.ndarray:
    """Where *family*'s landmark centroid must be under *matrix*.

    A block (raster, mesh) in 3D: the hull of its projected corners.  In 2D
    the block's cross-section at the slice is drawn, so the hull of its four
    in-plane corners at ``slice_z`` (data ``z`` equals world ``z`` for every
    view these tests draw in 2D).  A point, cross or graph is symmetric about
    ``P0``, so its centroid is the projection of ``P0``.
    """
    if family in ("points", "lines", "graph"):
        return project(camera, to_world(matrix, P0))
    corners = cube_corners().astype(np.float64)
    if dim == "2d":
        corners = corners[corners[:, 0] == corners[:, 0].min()]
        corners[:, 0] = P0[0]
    projected = np.array([project(camera, to_world(matrix, c)) for c in corners])
    return hull_centroid(projected)


def landmark_error(
    family: str,
    name_or_matrix,
    dim: str,
    tmp: Path,
    *,
    camera_setup: Callable | None = None,
) -> float:
    """Pixel distance between the drawn and the expected landmark centroid.

    ``inf`` when nothing drew.
    """
    matrix = (
        world_matrix(LINEARS[name_or_matrix])
        if isinstance(name_or_matrix, str)
        else name_or_matrix
    )
    viewer = new_viewer(dim)
    try:
        add_landmark(viewer, family, matrix, tmp)
        if dim == "2d":
            load(
                viewer,
                slice_z=to_world(matrix, P0)[0],
                half_thickness=0.6 if family in GEOMETRY else None,
            )
            nudge_2d_camera(canvas_view(viewer).camera)
        else:
            load(viewer)
        if camera_setup is not None:
            camera_setup(canvas_view(viewer).camera, matrix)
        frame, _, camera = render(viewer)
        got = centroid(frame)
        if got is None:
            return float("inf")
        expected = expected_centroid(family, camera, matrix, dim=dim)
        return float(np.linalg.norm(got - expected))
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Time series: a drift ``x += v t`` (full affine design v2, SLICE_SHIFT)
# ---------------------------------------------------------------------------

#: Frames in the drifting series, and the drift in world x per frame.
N_FRAMES = 3
DRIFT = 3.0
#: The families a ``tzyx`` store exists for (a mesh store has no time axis).
TIME_FAMILIES = (
    "image",
    "image_multiscale",
    "labels",
    "labels_multiscale",
    "points",
    "lines",
    "graph",
)


def new_tzyx_viewer(dim: str) -> Viewer:
    """A black-background offscreen viewer of world ``tzyx``."""
    displayed = ("y", "x") if dim == "2d" else ("z", "y", "x")
    viewer = Viewer(
        [("t", "time"), *spatial_axes("z", "y", "x")],
        displayed_axes=displayed,
        gui="offscreen",
    )
    viewer.background = BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 0.0, 1.0))
    return viewer


def drift_matrix(velocity: float = DRIFT) -> np.ndarray:
    """The 5x5 ``tzyx -> TZYX`` matrix: world ``x = x + v t``."""
    matrix = np.eye(5)
    matrix[3, 0] = velocity
    return matrix


def frame_matrix(t: float, velocity: float = DRIFT) -> np.ndarray:
    """The ``zyx`` 4x4 a drifting frame *t* is placed with."""
    matrix = np.eye(4)
    matrix[2, 3] = velocity * t
    return matrix


def _series(volume: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.broadcast_to(volume, (N_FRAMES, *volume.shape)))


def _multiscale_series(root: Path, name: str, data: np.ndarray):
    root = root / name
    root.mkdir(parents=True, exist_ok=True)
    coarse = data[:, ::2, ::2, ::2]

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
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
        name=name,
    )


def _per_frame(rows: np.ndarray) -> np.ndarray:
    """``(n, 3)`` zyx rows repeated for every frame, as ``(n * T, 4)`` tzyx."""
    frames = [
        np.concatenate([np.full((len(rows), 1), float(t)), rows], axis=1)
        for t in range(N_FRAMES)
    ]
    return np.concatenate(frames).astype(np.float32)


def add_drift_landmark(viewer: Viewer, family: str, matrix: np.ndarray, tmp: Path):
    """Add *family*'s landmark at ``P0`` in every frame of a ``tzyx`` series."""
    transform_of = lambda store: bind(viewer, store, matrix)  # noqa: E731
    if family in ("points", "lines", "graph"):
        bounds = ImageMemoryStore(data=_series(np.zeros(SHAPE, np.float32)), name="b")
        hidden = viewer.add_image(
            bounds,
            single=InMemoryImageSingleAppearance(clim=(0.0, 1.0), render_mode="mip"),
            transform=transform_of(bounds),
        )
        viewer.controller.set_visual_visible(hidden.id, False)
    if family == "image":
        store = ImageMemoryStore(data=_series(block_volume(np.float32)), name="s")
        return viewer.add_image(
            store,
            appearance=InMemoryImageAppearance(interpolation="nearest"),
            single=InMemoryImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="mip"
            ),
            transform=transform_of(store),
        )
    if family == "labels":
        store = LabelMemoryStore(data=_series(block_volume(np.int32)), name="s")
        return viewer.add_labels(
            store,
            appearance=InMemoryLabelsAppearance(render_mode="iso_categorical"),
            transform=transform_of(store),
        )
    if family == "image_multiscale":
        store = _multiscale_series(tmp, "s", _series(block_volume(np.float32)))
        return viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(force_level=0),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="mip"
            ),
            transform=transform_of(store),
        )
    if family == "labels_multiscale":
        store = _multiscale_series(tmp, "s", _series(block_volume(np.int32)))
        return viewer.add_labels_multiscale(
            store,
            appearance=MultiscaleLabelsAppearance(
                force_level=0, ray_steps_per_voxel=4.0
            ),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=transform_of(store),
        )
    if family == "points":
        store = PointsMemoryStore(positions=_per_frame(P0[None]), name="p")
        return viewer.add_points(
            store,
            appearance=PointsMarkerAppearance(size=9.0, color=(1, 1, 1, 1)),
            transform=transform_of(store),
        )
    if family == "lines":
        store = LinesMemoryStore(positions=_per_frame(cross_vertices()), name="l")
        return viewer.add_lines(
            store,
            appearance=LinesMemoryAppearance(color=(1, 1, 1, 1), thickness=3.0),
            transform=transform_of(store),
        )
    if family == "graph":
        vertices = _per_frame(cross_vertices())
        edges = np.arange(len(vertices), dtype=np.int64).reshape(-1, 2)
        store = GraphMemoryStore.from_arrays(vertices, edges, name="g")
        return viewer.add_graph(store, transform=transform_of(store))
    raise ValueError(family)
