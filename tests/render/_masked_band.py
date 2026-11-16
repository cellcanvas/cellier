"""The 2D band's scenes.

One ellipsoid as four kinds of mask, sliced at ``z = SLICE`` and drawn as a
MaskedImage band.  The data are placed by a rotation about ``y`` (0 or 30
degrees), so the slice is aligned with the mask's grid or tilted against
it, and the mask is drawn at level 1 or forced to level 2.

The tests compare a frame with the exact in-plane truth in
``masked_band_truth.json``, which ``scripts/masked_image/freeze_band_truth.py``
writes from the same scenes (the camera's pixels and a CPU distance transform
of the mask as the drawn level reads it).  The truth is stored, not computed
in the tests.
"""

from __future__ import annotations

import base64
import itertools
import json
import zlib
from pathlib import Path

import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas

from cellier.transform import AffineTransform
from cellier.visuals import MaskedImageAppearance, MaskedImageSingleAppearance
from tests.render._masked import (
    _ellipsoid_inside,
    load,
    masked_config,
    new_viewer,
    pyramid_store,
)

SIDE = 64
SLICE = 32.0
SIZE = 160
#: World units (level-0 voxels) a pixel.
ZOOM = 0.5
#: Pixels within this of a band edge are not counted.
EDGE_TOL_PX = 0.5
TRUTH_PATH = Path(__file__).parent / "masked_band_truth.json"

#: ``name -> (start, end)`` of the band along inward, world units.
BANDS = {
    "inward_t8": (0.0, 8.0),
    "centred_t8": (-4.0, 4.0),
    "outward_t8": (-8.0, 0.0),
    "hairline": (0.0, 0.0),
}
#: ``name -> (level, tilt degrees)``; level 0 is the finest.
VIEWS = {
    "L0_aligned": (0, 0.0),
    "L0_tilted30": (0, 30.0),
    "L1_aligned": (1, 0.0),
    "L1_tilted30": (1, 30.0),
}
MASK_KINDS = ("binary", "probability", "edt_clamped", "signed_distance")


def masks(side: int = SIDE) -> dict[str, tuple[np.ndarray, float]]:
    """``name -> (mask, iso)``: the ellipsoid as four kinds of mask."""
    from scipy import ndimage as ndi

    inside = _ellipsoid_inside(side)
    edt = ndi.distance_transform_edt(inside).astype(np.float32)
    sdf = edt - ndi.distance_transform_edt(~inside).astype(np.float32)
    return {
        "binary": (inside.astype(np.float32), 0.5),
        "probability": ((1.0 / (1.0 + np.exp(-sdf))).astype(np.float32), 0.5),
        "edt_clamped": (edt, 2.0),
        "signed_distance": (sdf, 2.0),
    }


def rotation_y(degrees: float) -> np.ndarray:
    """A rotation about ``y`` in ``(z, y, x)`` order."""
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def data_matrix(tilt: float, side: int = SIDE) -> np.ndarray:
    """Voxel -> world, 4x4 ``(z, y, x)``: a rotation about the volume's centre."""
    centre = np.full(3, (side - 1) / 2.0)
    linear = rotation_y(tilt)
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = centre - linear @ centre
    return matrix


def build(
    root: Path,
    mask: np.ndarray,
    iso: float,
    view: str,
    band: tuple[float, float] = (0.0, 0.0),
    **single,
):
    """A loaded 2D viewer of the band of *mask*; ``(viewer, model)``.

    The value is 3.0 everywhere, so a drawn pixel is the band's own and
    every drawn pixel has one colour.
    """
    level, tilt = VIEWS[view]
    viewer = new_viewer("2d")
    controller = viewer.controller
    mask_store = pyramid_store(root, "mask", mask, 2)
    value_store = pyramid_store(root, "value", np.full(mask.shape, 3.0, np.float32), 2)
    controller._ensure_data_coordinate_systems(viewer.scene.id, mask_store)
    transform = AffineTransform.from_matrix(
        data_matrix(tilt),
        mask_store.data_coordinate_systems[0],
        viewer.scene.dims.world_coordinate_system,
    )
    model = controller.add_masked_image(
        mask_store,
        value_store,
        viewer.scene.id,
        transform=transform,
        appearance=MaskedImageAppearance(force_level=level + 1, frustum_cull=False),
        single=MaskedImageSingleAppearance(
            color_map="gray",
            clim=(0.0, 3.0),
            iso_threshold=iso,
            surface_thickness=band[1] - band[0],
            surface_placement=placement(band),
            **single,
        ),
        render_config=masked_config(),
    )
    load(viewer, slice_z=SLICE, size=(SIZE, SIZE))
    camera = canvas_view(viewer).camera
    camera.width = camera.height = SIZE * ZOOM
    # The truth was frozen with the camera centred on the cut the slice makes
    # through the data.  A fit centres on the data's whole box, which for a
    # tilted volume is wider than the cut, so the camera is placed here.
    x, y = cut_centre(tilt)
    camera.local.position = (x, y, 0.0)
    return viewer, model


def cut_centre(
    tilt: float, side: int = SIDE, slice_z: float = SLICE
) -> tuple[float, float]:
    """Centre of the box of the slice ``z = slice_z`` through the tilted data.

    The data's voxel box is mapped by `data_matrix`; each of its 12 edges
    that crosses the slice plane gives a point of the cut.  Returns the
    centre of the cut's box in pygfx ``(x, y)``.
    """
    matrix = data_matrix(tilt, side)
    corners = np.array(list(itertools.product((-0.5, side - 0.5), repeat=3)))
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    points = []
    for i, j in itertools.combinations(range(len(corners)), 2):
        if np.count_nonzero(corners[i] != corners[j]) != 1:
            continue  # not an edge
        a, b = world[i], world[j]
        if a[0] == b[0]:
            if a[0] == slice_z:
                points += [a, b]
            continue
        t = (slice_z - a[0]) / (b[0] - a[0])
        if 0.0 <= t <= 1.0:
            points.append(a + t * (b - a))
    zyx = np.array(points)
    centre = (zyx[:, 1:].min(axis=0) + zyx[:, 1:].max(axis=0)) / 2.0
    return float(centre[1]), float(centre[0])


def placement(band: tuple[float, float]) -> str:
    """The placement whose interval is *band* for its thickness."""
    start, end = band
    if start == 0.0:
        return "inward"
    if end == 0.0:
        return "outward"
    return "centred"


def canvas_view(viewer):
    controller = viewer.controller
    return controller._render_manager._canvases[
        controller.get_canvas_ids(viewer.scene.id)[0]
    ]


class Frames:
    """One offscreen canvas drawing a viewer's scene with its own camera."""

    def __init__(self, viewer) -> None:
        viewer.controller.update_background_field(viewer.scene.id, "visible", False)
        self.view = canvas_view(viewer)
        self.scene = self.view._get_scene_fn(viewer.scene.id)
        self.canvas = RenderCanvas(size=(SIZE, SIZE), pixel_ratio=1)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.renderer.pixel_scale = 1
        self.renderer.ppaa = "none"
        self.canvas.request_draw(
            lambda: self.renderer.render(self.scene, self.view.camera)
        )

    def frame(self) -> np.ndarray:
        self.canvas.request_draw()
        return np.asarray(self.canvas.draw()).copy()


def drawn(frame: np.ndarray) -> np.ndarray:
    """The band's pixels: the value is 3.0, white at clim (0, 3)."""
    return frame[..., :3].max(axis=-1) > 128


def pixel_world(camera) -> tuple[np.ndarray, float]:
    """``(world (z, y, x) of each pixel centre on the slice, world per pixel)``.

    The 2D slice route renders the displayed ``(y, x)`` as pygfx
    ``(y, x)``; the slice is the world's ``z = SLICE``.
    """
    inverse = np.linalg.inv(np.asarray(camera.camera_matrix, np.float64))
    cols, rows = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    ndc = np.stack(
        [
            (cols + 0.5) / SIZE * 2 - 1,
            1 - (rows + 0.5) / SIZE * 2,
            np.zeros(cols.shape),
            np.ones(cols.shape),
        ],
        axis=-1,
    )
    point = ndc @ inverse.T
    point = point[..., :3] / point[..., 3:]
    world = np.stack([np.full(cols.shape, SLICE), point[..., 1], point[..., 0]], -1)
    px = float(np.linalg.norm(world[0, 1] - world[0, 0]))
    return world, px


# -- the stored truth ----------------------------------------------------------


#: Distances saturate here, past every band's reach (8 world units, 16 px).
SATURATE_PX = 24.0


def encode_distance(d_px: np.ndarray) -> str:
    """Signed distances in pixels, as base64 zlib int16 of twentieths.

    Saturated at ``SATURATE_PX``, which no band reaches.
    """
    d_px = np.clip(d_px, -SATURATE_PX, SATURATE_PX)
    q = np.round(d_px * 20.0).astype("<i2")
    return base64.b64encode(zlib.compress(q.tobytes(), 9)).decode("ascii")


def decode_distance(text: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(text))
    return np.frombuffer(raw, dtype="<i2").reshape(SIZE, SIZE).astype(np.float64) / 20.0


def load_truth() -> dict:
    return json.loads(TRUTH_PATH.read_text())


def placed(band: tuple[float, float], px: float, min_px: float = 1.0):
    """The band widened to *min_px* pixels about its centre."""
    a, b = band
    if b - a < min_px * px:
        c = 0.5 * (a + b)
        a, b = c - 0.5 * min_px * px, c + 0.5 * min_px * px
    return a, b


def score(is_drawn: np.ndarray, d_px: np.ndarray, band, px: float) -> dict:
    """A frame against the truth: pixels missing, extra, and the edges' reach.

    Pixels within ``EDGE_TOL_PX`` of a band edge count neither way.
    """
    a, b = placed(band, px)
    d = d_px * px
    lo, hi = a - 0.5 * px, b + 0.5 * px
    expected = (d > lo) & (d < hi)
    tol = EDGE_TOL_PX * px
    decided = (np.abs(d - lo) > tol) & (np.abs(d - hi) > tol)
    out = {
        "drawn": int(is_drawn.sum()),
        "expected": int(expected.sum()),
        "missing": int((expected & ~is_drawn & decided).sum()),
        "extra": int((is_drawn & ~expected & decided).sum()),
    }
    covered = d[is_drawn]
    if covered.size:
        out["reach_past_outer_px"] = float((covered.max() - hi) / px)
        out["reach_past_inner_px"] = float((lo - covered.min()) / px)
    return out
