"""Every pyramid level is drawn where its level -> data transform puts it.

``plans/multiscale_level_transform_v2.md``.  A small OME-Zarr v0.5 pyramid
(image and labels) is written in one of four **kinds** (``KINDS``), read back
through the real OME-Zarr readers and drawn at every forced level, in 2D and
3D, for the image and the labels.  Along an axis with cumulative factor
``F``, level ``k`` has metadata ``scale = F * s0`` and ``translation =
t * s0 + T0``, with ``t`` in level-0 voxels:

- ``offset``: offset striding, level ``k`` keeps level-0 voxels
  ``F // 2 + F * i``; ``t = F // 2``.
- ``plain``: plain striding, voxels ``F * i``; ``t = 0``.  The most common
  real case (scale-only metadata).
- ``offset_t0``: ``offset`` with a non-zero level-0 translation ``T0`` on
  every level, which catches ``t_0`` applied twice or not at all.
- ``block_average``: each coarse voxel is the mean of its ``F`` block;
  ``t = (F - 1) / 2``.  Image only: averaging labels is meaningless.

Measurement.  Each sphere centre sits on a point every level's sampling
lattice is symmetric about, so a correctly drawn level shows each sphere
centred on its true centre however blocky it gets.  The test finds each
sphere's blob on screen, takes its centroid, and compares it with the true
centre projected through the camera that drew the frame.  The pixel error is
converted to level-0 voxels per data axis with the local projection Jacobian,
so the tolerance does not depend on zoom.

Views.  3D uses a narrow field of view (``FOV_3D``, close to orthographic, so
a symmetric blob projects to a symmetric silhouette), once looking down z
(measures y and x) and once down y (measures z and x).  The spheres lie on a
diagonal so their silhouettes stay separate in both views at every level.
2D slices z through the middle sphere's centre and measures y and x.

Before the fix each path draws coarse voxel ``i`` at its own implied position
(``expected_before``); the cases that model predicts to be off by more than
``TOL_VOXELS`` are ``xfail(strict=True)``, so a case that starts passing, or a
model prediction that turns out wrong, fails the suite.  Tolerance reasoning
is next to ``TOL_VOXELS``.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from scipy import ndimage

from tests._gpu_budget import SMALL_BUDGETS

#: Level-0 shape and voxel size, (z, y, x).  Every axis differs, so a swapped
#: or dropped axis shows up.
SHAPE = (32, 96, 128)
VOXEL = (2.0, 0.6, 0.4)

#: Cumulative downscale factor per level, (z, y, x).
FACTORS = ((1, 1, 1), (1, 2, 2), (2, 4, 2), (2, 8, 4))

#: Sphere centres (level-0 voxels) and radii (micrometers).  Every
#: coordinate is a multiple of its axis's largest factor (z 2, y 8, x 4), so
#: every level's sampling lattice is symmetric about each centre in every
#: kind; ``block_average`` moves them half a voxel (``_centres``).
#: ``test_centres_are_symmetric_under_every_lattice`` checks this.
SPHERES = (((8, 24, 32), 5.0), ((16, 48, 64), 6.0), ((24, 72, 96), 5.0))

KINDS = ("offset", "plain", "offset_t0", "block_average")

#: Level-0 OME translation (micrometers, z y x) of the ``offset_t0`` kind.
T0_OFFSET = (3.0, -1.2, 0.8)


def level_translation(kind: str, factor: int) -> float:
    """``t``: the level's translation along an axis, level-0 voxels."""
    if kind == "plain":
        return 0.0
    if kind == "block_average":
        return (factor - 1) / 2
    return float(factor // 2)


def _t0(kind: str) -> tuple[float, float, float]:
    return T0_OFFSET if kind == "offset_t0" else (0.0, 0.0, 0.0)


def _centres(kind: str) -> list[tuple[float, float, float]]:
    """Sphere centres, level-0 voxels.

    Block averaging's lattice is symmetric about block boundaries
    (``M - 0.5``), striding's about voxel centres.
    """
    shift = 0.5 if kind == "block_average" else 0.0
    return [tuple(c - shift for c in centre) for centre, _ in SPHERES]


def _paths(kind: str) -> tuple[str, ...]:
    return tuple(p for p in PATHS if kind != "block_average" or "labels" not in p)


#: The 2D slice: z through the middle sphere.
SLICE_Z = 16
SLICE_SPHERES = (1,)

SIZE = (960, 960)

#: 3D field of view, degrees.  Near-orthographic, so a symmetric blob projects
#: to a symmetric silhouette wherever it sits in the frame.  (``fov = 0``,
#: true orthographic, breaks the volume shaders' ray setup.)
FOV_3D = 1.0
VALUE = 200
THRESHOLD = 100.0

#: Pass threshold, in level-0 voxels per axis.  Every convention error this
#: test exists to catch is at least half a level-0 voxel.  Measured noise
#: (Phase 0, 960 px): 2D <= 0.06; 3D <= 0.10 once ray steps are adequate
#: (with today's step starvation, S1, up to 0.27 at level 2).  0.2 is twice
#: the 3D noise and 2.5x below the smallest error.
TOL_VOXELS = 0.2

PATHS = ("3d_image", "3d_labels", "2d_image", "2d_labels")

#: Paths whose shaders apply the level transform (plan v2): 2D in Phase 5,
#: 3D in Phase 6.  ``expected_before`` documents what they drew before.
FIXED_PATHS = frozenset(PATHS)


def expected_before(path: str, factor: int, t: float) -> float:
    """Drawn minus true centre, level-0 voxels, before the fix.

    Coarse voxel ``i`` belongs at ``F*i + t``.  The shaders draw it at
    ``F*i`` (3D image), ``F*i - 0.5`` (both labels paths) and
    ``F*i + (F-1)/2`` (2D image), whatever ``t`` is.
    """
    if path == "3d_image":
        return -t
    if path in ("3d_labels", "2d_labels"):
        return -t - 0.5
    return (factor - 1) / 2 - t


def _measured_axes(path: str, view: str) -> tuple[int, ...]:
    """Data axes a view measures: down z -> (y, x); down y -> (z, x)."""
    if view == "3d_y":
        return (0, 2)
    return (1, 2)


def _views(path: str) -> tuple[str, ...]:
    return ("3d_z", "3d_y") if path.startswith("3d") else ("2d",)


def _fails_before(kind: str, path: str, level: int) -> bool:
    for view in _views(path):
        for axis in _measured_axes(path, view):
            factor = FACTORS[level][axis]
            error = expected_before(path, factor, level_translation(kind, factor))
            if abs(error) > TOL_VOXELS:
                return True
    return False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _level_transforms(factors, kind):
    return [
        {"type": "scale", "scale": [f * s for f, s in zip(factors, VOXEL)]},
        {
            "type": "translation",
            "translation": [
                level_translation(kind, f) * s + t0
                for f, s, t0 in zip(factors, VOXEL, _t0(kind))
            ],
        },
    ]


def _multiscales(name, kind):
    return [
        {
            "name": name,
            "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
            "datasets": [
                {
                    "path": str(k),
                    "coordinateTransformations": _level_transforms(f, kind),
                }
                for k, f in enumerate(FACTORS)
            ],
        }
    ]


def _downsample(level0, factors, kind):
    if kind == "block_average":
        shape = []
        for n, f in zip(level0.shape, factors):
            shape += [n // f, f]
        blocks = level0.astype(np.float64).reshape(shape)
        mean = blocks.mean(axis=(1, 3, 5))
        return np.round(mean).astype(level0.dtype)
    start = 0 if kind == "plain" else None
    return level0[
        tuple(slice(f // 2 if start is None else start, None, f) for f in factors)
    ]


def _write_pyramid(group, level0, kind):
    for k, factors in enumerate(FACTORS):
        data = _downsample(level0, factors, kind)
        arr = group.create_array(
            str(k), shape=data.shape, dtype=data.dtype, chunks=data.shape
        )
        arr[...] = data


def _write_dataset(root, kind="offset"):
    import zarr
    from skimage.measure import label

    z, y, x = np.indices(SHAPE, dtype=np.float64)
    image = np.zeros(SHAPE, dtype=np.uint8)
    for (cz, cy, cx), (_, radius) in zip(_centres(kind), SPHERES):
        d2 = (
            ((z - cz) * VOXEL[0]) ** 2
            + ((y - cy) * VOXEL[1]) ** 2
            + ((x - cx) * VOXEL[2]) ** 2
        )
        image[d2 <= radius**2] = VALUE
    labels = label(image > THRESHOLD).astype(np.int32)

    path = root / f"spheres_{kind}.ome.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=3)
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales("spheres", kind),
    }
    _write_pyramid(group, image, kind)
    labels_root = group.create_group("labels")
    labels_root.attrs["ome"] = {"version": "0.5", "labels": ["spheres"]}
    labels_group = labels_root.create_group("spheres")
    labels_group.attrs["ome"] = {
        "version": "0.5",
        "image-label": {},
        "multiscales": _multiscales(
            "spheres", "plain" if kind == "block_average" else kind
        ),
    }
    # Averaging labels is meaningless: the block-average kind carries strided
    # labels (with matching metadata) that no test measures.
    _write_pyramid(labels_group, labels, "plain" if kind == "block_average" else kind)
    return path


def _build_viewer(path):
    from cellier.convenience import Viewer, spatial_axes
    from cellier.data import OMEZarrImageDataStore, OMEZarrLabelDataStore
    from cellier.scene import BackgroundAppearance
    from cellier.transform import AffineTransform
    from cellier.visuals import (
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageSingleAppearance,
        MultiscaleLabelRenderConfig,
        MultiscaleLabelsAppearance,
    )

    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        dim="3d",
        render_modes={"2d", "3d"},
        gui="offscreen",
    )
    viewer.background = BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 0.0, 1.0))
    world = viewer.scene.dims.world_coordinate_system

    def to_world(store):
        cs = store.data_coordinate_systems[0]
        return AffineTransform.from_axis_map(
            cs,
            world,
            axis_map={a: a for a in "zyx"},
            scale=dict(zip("zyx", store.physical_scale)),
            translation=dict(zip("zyx", store.physical_translation)),
        )

    image_store = OMEZarrImageDataStore.from_path(path.as_uri())
    labels_store = OMEZarrLabelDataStore.from_path(
        (path / "labels" / "spheres").as_uri()
    )
    image = viewer.add_image_multiscale(
        image_store,
        appearance=MultiscaleImageAppearance(force_level=1),
        transform=to_world(image_store),
        single=MultiscaleImageSingleAppearance(
            color_map="gray",
            clim=(0.0, THRESHOLD),
            render_mode="iso",
            iso_threshold=THRESHOLD,
        ),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS),
    )
    labels = viewer.add_labels_multiscale(
        labels_store,
        appearance=MultiscaleLabelsAppearance(force_level=1),
        transform=to_world(labels_store),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    return viewer, image, labels


# ---------------------------------------------------------------------------
# Rendering and measurement
# ---------------------------------------------------------------------------


def _world_centres(kind="offset"):
    """True sphere centres in world (pygfx x, y, z) order."""
    t0 = _t0(kind)
    return np.array(
        [
            [c[2] * VOXEL[2] + t0[2], c[1] * VOXEL[1] + t0[1], c[0] * VOXEL[0] + t0[0]]
            for c in _centres(kind)
        ]
    )


def _project(matrix, world_xyz, shape):
    """World points to continuous pixel coordinates ``(row, col)``."""
    h = np.c_[world_xyz, np.ones(len(world_xyz))] @ np.asarray(matrix).T
    ndc = h[:, :3] / h[:, 3:4]
    rows = (1.0 - ndc[:, 1]) / 2.0 * shape[0]
    cols = (ndc[:, 0] + 1.0) / 2.0 * shape[1]
    return np.stack([rows, cols], axis=1)


def _blob_centroids(frame):
    """Continuous ``(row, col)`` centroids of foreground blobs."""
    mask = frame[..., :3].max(axis=-1) > 20
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.empty((0, 2))
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = [i + 1 for i in range(n) if sizes[i] >= 20]
    cents = np.array(ndimage.center_of_mass(mask, lab, keep))
    return cents + 0.5  # pixel (r, c) covers [r, r + 1]


#: World (pygfx) axis index for each data axis: z -> 2, y -> 1, x -> 0.
_WORLD_AXIS = {0: 2, 1: 1, 2: 0}


def _errors_voxels(frame, matrix, spheres, data_axes, kind="offset"):
    """Per sphere, drawn minus true centre in level-0 voxels on *data_axes*."""
    shape = frame.shape[:2]
    world = _world_centres(kind)[list(spheres)]
    true_px = _project(matrix, world, shape)
    blobs = _blob_centroids(frame)
    out = []
    for w, t in zip(world, true_px):
        if len(blobs) == 0:
            out.append([np.nan] * len(data_axes))
            continue
        blob = blobs[np.argmin(((blobs - t) ** 2).sum(axis=1))]
        # Jacobian: pixels per micrometer along each measured world axis.
        jac = []
        for axis in data_axes:
            step = np.zeros(3)
            step[_WORLD_AXIS[axis]] = 1.0
            plus = _project(matrix, (w + step)[None], shape)[0]
            minus = _project(matrix, (w - step)[None], shape)[0]
            jac.append((plus - minus) / 2.0)
        jac = np.array(jac).T  # (2 px, n axes)
        err_um = np.linalg.lstsq(jac, blob - t, rcond=None)[0]
        out.append([e / VOXEL[a] for e, a in zip(err_um, data_axes)])
    return np.array(out)


async def _render_all(viewer, image, labels, kind="offset"):
    from cellier.convenience.capture import _load_data, _wait_for_slicer

    controller = viewer.controller
    scene_id = viewer.scene.id
    await _load_data(viewer)
    canvas = next(iter(controller._render_manager._canvases.values()))
    gfx_scene = canvas._get_scene_fn(canvas._scene_id)
    frames = {}

    async def shoot(key_prefix, spheres):
        for level in range(len(FACTORS)):
            image.appearance.force_level = labels.appearance.force_level = level + 1
            for which, vis in (("image", (True, False)), ("labels", (False, True))):
                if which == "labels" and kind == "block_average":
                    continue
                # A hidden visual skips slicing, so reslice after showing it.
                image.appearance.visible, labels.appearance.visible = vis
                controller.reslice_all()
                await _wait_for_slicer(controller)
                frame = viewer.screenshot(size=SIZE)
                matrix = np.array(canvas._camera.camera_matrix)
                frames[(key_prefix, which, level)] = (frame, matrix, spheres)
                if (key_prefix, which, level) == ("3d_z", "image", 0):
                    frame = viewer.screenshot(size=SIZE)
                    frames[("repeat", which, level)] = (frame, matrix, spheres)
        image.appearance.visible = labels.appearance.visible = True

    camera = canvas._camera_3d
    for view, view_dir, up in (
        ("3d_z", (0, 0, -1), (0, 1, 0)),
        ("3d_y", (0, -1, 0), (0, 0, 1)),
    ):
        camera.fov = FOV_3D
        camera.show_object(gfx_scene, view_dir=view_dir, up=up)
        await shoot(view, tuple(range(len(SPHERES))))

    controller.set_displayed_axes(scene_id, (1, 2))
    controller.update_slice_indices(scene_id, {0: SLICE_Z * VOXEL[0] + _t0(kind)[0]})
    await _load_data(viewer)
    await shoot("2d", SLICE_SPHERES)
    return frames


def render_kind(kind, root):
    """Every frame of one kind: ``{(view, which, level): (frame, M, spheres)}``."""
    from cellier.convenience.capture import _ensure_canvases

    path = _write_dataset(root, kind)
    viewer, image, labels = _build_viewer(path)
    _ensure_canvases(viewer, SIZE)
    try:
        return asyncio.run(_render_all(viewer, image, labels, kind))
    finally:
        viewer.controller.close()


@pytest.fixture(scope="module")
def renders(offscreen_gpu, tmp_path_factory):
    """``renders(kind)``: that kind's frames, rendered once per module."""
    cache = {}

    def _get(kind):
        if kind not in cache:
            cache[kind] = render_kind(kind, tmp_path_factory.mktemp(kind))
        return cache[kind]

    return _get


def measure(
    frames, path: str, level: int, kind: str = "offset"
) -> dict[tuple[str, int], np.ndarray]:
    """``{(view, data_axis): per-sphere errors}`` for one path and level."""
    which = path.split("_", 1)[1]
    out = {}
    for view in _views(path):
        frame, matrix, spheres = frames[(view, which, level)]
        axes = _measured_axes(path, view)
        errs = _errors_voxels(frame, matrix, spheres, axes, kind)
        for i, axis in enumerate(axes):
            out[(view, axis)] = errs[:, i]
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _cases():
    for kind in KINDS:
        for path in _paths(kind):
            for level in range(len(FACTORS)):
                marks = []
                if path not in FIXED_PATHS and _fails_before(kind, path, level):
                    marks.append(
                        pytest.mark.xfail(
                            strict=True,
                            reason=(
                                "level translation ignored (plan v2 Phase "
                                f"{5 if path.startswith('2d') else 6})"
                            ),
                        )
                    )
                yield pytest.param(
                    kind, path, level, marks=marks, id=f"{kind}-{path}-L{level}"
                )


@pytest.mark.parametrize(("kind", "path", "level"), list(_cases()))
def test_level_drawn_at_its_transform(renders, kind, path, level):
    errors = measure(renders(kind), path, level, kind)
    worst = max(float(np.nanmax(np.abs(e))) for e in errors.values())
    assert not any(np.isnan(e).any() for e in errors.values()), errors
    assert worst <= TOL_VOXELS, {k: np.round(v, 3).tolist() for k, v in errors.items()}


def _sphere_areas(frame, matrix, spheres, kind="offset"):
    """Pixel area of the blob nearest each sphere's projected centre."""
    mask = frame[..., :3].max(axis=-1) > 20
    lab, _ = ndimage.label(mask)
    true_px = _project(matrix, _world_centres(kind)[list(spheres)], frame.shape[:2])
    areas = []
    for r, c in true_px:
        r, c = (
            int(np.clip(r, 0, mask.shape[0] - 1)),
            int(np.clip(c, 0, mask.shape[1] - 1)),
        )
        # The blob under the projected centre; 0 when the centre is empty.
        blob = lab[r, c]
        areas.append(float((lab == blob).sum()) if blob else 0.0)
    return np.array(areas)


def _frame_cases():
    for kind in KINDS:
        for view in ("3d_z", "3d_y", "2d"):
            for which in ("image", "labels"):
                if which == "labels" and kind == "block_average":
                    continue
                for level in range(len(FACTORS)):
                    key = (view, which, level)
                    yield pytest.param(kind, key, id=f"{kind}-{view}-{which}-L{level}")


@pytest.mark.parametrize(("kind", "key"), list(_frame_cases()))
def test_every_sphere_drawn(renders, kind, key):
    """Each sphere draws one blob, at least a third of its level-0 area.

    Area, not just a blob count: a starved ray march (too few steps along the
    ray, plan finding S1) leaves slivers and speckle, which count as blobs,
    and erodes the spheres.  Blocky coarse spheres vary in area, never by 3x.
    """
    frames = renders(kind)
    frame, matrix, spheres = frames[key]
    view, which, _ = key
    ref_frame, ref_matrix, _ = frames[(view, which, 0)]
    areas = _sphere_areas(frame, matrix, spheres, kind)
    ref = _sphere_areas(ref_frame, ref_matrix, spheres, kind)
    assert len(_blob_centroids(frame)) == len(spheres)
    assert (areas >= ref / 3.0).all(), (areas.tolist(), ref.tolist())


def test_render_is_deterministic(renders):
    """The harness itself: re-rendering the same state gives the same frame."""
    frames = renders("offset")
    first = frames[("3d_z", "image", 0)][0]
    again = frames[("repeat", "image", 0)][0]
    np.testing.assert_array_equal(again, first)


@pytest.mark.parametrize("kind", KINDS)
def test_centres_are_symmetric_under_every_lattice(kind):
    """The measurement's premise: each centre is a symmetry point of each level.

    Level voxel centres sit at ``t + F * i`` (level-0 voxels); a point ``c`` is
    a symmetry centre of that lattice when ``(c - t) / F`` is a multiple of
    one half.  The level-0 raster is symmetric about ``c`` when ``2 c`` is an
    integer.
    """
    for centre in _centres(kind):
        for axis, c in enumerate(centre):
            assert (2 * c) % 1 == 0
            for factors in FACTORS:
                f = factors[axis]
                phase = (c - level_translation(kind, f)) / f
                assert (2 * phase) % 1 == 0, (kind, centre, axis, f)
