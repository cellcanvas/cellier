"""The cut face of a clipped volume (clipping planes design 4.3, D21).

A clipped ray starts inside the object, so an ISO or label volume draws a
solid face on the plane.  That face sits on the plane and is lit, and
written to the ambient-occlusion normal target, with the plane's normal.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest
from pygfx.renderers.wgpu import get_shared
from rendercanvas.offscreen import RenderCanvas as OffscreenRenderCanvas

from cellier.render._cellier_blender import NORMAL_TARGET, install_cellier_blender
from cellier.render._clipping import reduce_clipping_planes
from tests.render import _clipping as h
from tests.render._pick import PickFrame

SIZE = 192
KINDS = [
    "image_memory_iso",
    "image_multiscale_iso",
    "image_multiscale_smooth_iso",
    "labels_memory",
    "labels_multiscale",
    "labels_multiscale_smooth_iso",
]
#: One plane, two meeting at an edge, three at a corner.
CASES = {
    "axis": h.PLANES["axis"],
    "oblique": h.PLANES["oblique"],
    "edge": [((0, 0, 14.5), (0, 0, 1)), ((0, 15.5, 0), (0, 1, 0))],
    "corner": [
        ((0, 0, 14.5), (0, 0, 1)),
        ((0, 15.5, 0), (0, 1, 0)),
        ((13.5, 0, 0), (1, 0, 0)),
    ],
}


def _read_normal_target(renderer, size: int) -> np.ndarray:
    raw = get_shared().device.queue.read_texture(
        {
            "texture": renderer._blender.get_texture(NORMAL_TARGET),
            "mip_level": 0,
            "origin": (0, 0, 0),
        },
        {"offset": 0, "bytes_per_row": 8 * size, "rows_per_image": size},
        (size, size, 1),
    )
    return np.frombuffer(raw, np.float16).reshape(size, size, 4).astype(np.float32)


def _near_a_label_edge(point_zyx: np.ndarray) -> bool:
    """Whether a data point is within two voxels of a face of a label block.

    ``label_data`` fills ``[4, 28)`` on y and x, split at z = 16 into two
    labels spanning ``[4, 16)`` and ``[16, 28)``.
    """
    z, y, x = point_zyx
    faces = [(z, (3.5, 15.5, 27.5)), (y, (3.5, 27.5)), (x, (3.5, 27.5))]
    return any(abs(value - face) < 2.0 for value, at in faces for face in at)


def _look_at_the_cut(camera, world_planes) -> None:
    """Put the camera on the clipped side, looking at the cut obliquely."""
    normal = np.sum(
        [np.array(p[:3]) / np.linalg.norm(p[:3]) for p in world_planes], axis=0
    )
    normal /= np.linalg.norm(normal)
    fitted = np.array(camera.local.position, dtype=float)
    distance = float(np.linalg.norm(fitted - h.CENTRE))
    direction = -normal + np.array([0.15, 0.35, 0.25])
    camera.local.position = tuple(
        h.CENTRE + direction / np.linalg.norm(direction) * distance
    )
    camera.look_at(tuple(h.CENTRE))


#: Kinds every arrangement is drawn for: one image and one labels kind, both
#: multiscale.
ALL_ARRANGEMENTS = ("image_multiscale_iso", "labels_multiscale")


def _cases() -> list:
    """Every kind at the corner; two kinds with every arrangement.

    Three planes meeting at a corner is the arrangement with the most to get
    wrong (which plane a face belongs to), so every kind has it.  The plane
    arithmetic is shared by the volume shaders, so one plane on an axis, one
    oblique and two at an edge are drawn for the kinds of `ALL_ARRANGEMENTS`
    only.  The full cross was 24 cases.
    """
    cases = [(kind, "corner") for kind in KINDS]
    cases += [
        (kind, case) for kind in ALL_ARRANGEMENTS for case in CASES if case != "corner"
    ]
    return [pytest.param(*case, id="-".join(case)) for case in cases]


@pytest.mark.parametrize(("kind", "case"), _cases())
async def test_the_cut_face_is_on_the_plane_with_its_normal(
    kind, case, controller, reslice, tmp_path
):
    planes = CASES[case]
    scene = controller.add_scene(dim="3d", name=f"{kind}_{case}")
    visual, store = h.add_visual(
        kind, controller, scene.id, h.data_for(kind), tmp_path, f"{kind}_{case}"
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    controller.fit_camera(scene.id)
    visual.clipping_planes = h.clipping_planes(store, planes)
    await reslice(controller, scene.id)

    gfx_scene, camera = h.gfx_scene(controller, scene.id)
    world_planes = reduce_clipping_planes(
        controller.render_spaces(visual.id),
        visual.transform,
        {},
        visual.clipping_planes,
    )
    _look_at_the_cut(camera, world_planes)

    canvas = OffscreenRenderCanvas(size=(SIZE, SIZE), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    install_cellier_blender(renderer, [NORMAL_TARGET])
    canvas.request_draw(lambda: renderer.render(gfx_scene, camera))
    frame = np.asarray(canvas.draw())
    normals = _read_normal_target(renderer, SIZE)
    # One read-back for every pixel picked below, not one each.
    picks = PickFrame(renderer)

    manager = controller._render_manager._scenes[scene.id]
    render_visual = manager.get_visual(visual.id)
    view = np.asarray(camera.view_matrix)[:3, :3]
    expected = []
    for a, b, c, _d in world_planes:
        direction = view @ (np.array([a, b, c]) / np.linalg.norm([a, b, c]))
        expected.append(direction / np.linalg.norm(direction))

    data_planes = [
        (np.asarray(normal, float), float(np.dot(normal, point)))
        for point, normal in planes
    ]
    rng = np.random.default_rng(0)
    rows, cols = np.nonzero(frame[..., 3] > 0)
    take = rng.choice(len(rows), size=min(500, len(rows)), replace=False)
    angles, behind = [], []
    for row, col in zip(rows[take], cols[take]):
        info = picks.info((col + 0.5, row + 0.5))
        hit = info.get("world_object")
        if hit is None:
            continue
        coordinate = render_visual.pick_data_coordinate(hit, info)
        if coordinate is None:
            continue
        # Pygfx order, voxel i spanning [i, i + 1): to data (z, y, x) centres.
        point = np.array([float(v) - 0.5 for v in coordinate])[::-1]
        distances = [
            (point @ normal - offset) / np.linalg.norm(normal)
            for normal, offset in data_planes
        ]
        # Nothing is drawn on the clipped side of any plane.
        behind.append(min(distances))
        nearest = int(np.argmin(np.abs(distances)))
        if abs(distances[nearest]) > 0.75 or sum(abs(d) < 1.5 for d in distances) > 1:
            continue  # the object's own surface, or too near an edge to tell
        if kind == "labels_multiscale_smooth_iso" and _near_a_label_edge(point):
            # The soft field rounds a label's edges, so within a voxel or two
            # of one the hit is the label's own surface, a little inside the
            # plane, with its own normal.  That is intended (design 4.3).
            continue
        written = normals[row, col, :3]
        if float(np.linalg.norm(written)) < 0.5:
            continue  # unwritten: the occlusion pass reconstructs from depth
        cosine = abs(float(written @ expected[nearest])) / float(
            np.linalg.norm(written)
        )
        angles.append(float(np.degrees(np.arccos(min(1.0, cosine)))))

    assert len(behind) > 100
    # A pick reports a voxel, so allow a voxel of slack.
    assert min(behind) > -1.0
    assert len(angles) > 40, len(angles)
    assert np.percentile(angles, 95) < 2.0, (np.median(angles), np.max(angles))
