"""Guards on the pick payload's bit layout.

Nothing else in the suite exercises the packing end to end: the helpers in
``test_pick_data_coordinate.py`` take hand-made floats, and
``test_canvas_pick_info.py`` only checks NamedTuple fields.  So a change to
the ``pick_pack`` convention, or to how many fields a fragment shader
writes, would go unnoticed.

That matters because ``CellierBlender`` adds a fourth output
(``outline_id``) to the generated ``FragmentOutput`` for every shader.
The whole argument for the ``outline_id`` target over repacking pick is
that picking is left untouched -- these tests are what makes that claim
checkable rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest

from cellier.render._cellier_blender import (
    OUTLINE_ID_TARGET,
    ensure_extra_targets,
)

# The 3D packing cellier's label and multiscale volume brick shaders write:
#   pick_pack(global_id, 20) + pick_pack(qx, wx) + pick_pack(qy, wy)
#                            + pick_pack(qz, wz)
# with the widths of ``pick_bit_layout(dataset_size)``; see
# cellier.pick_encoding.wgsl.


def _pack_3d(global_id: int, q, widths) -> int:
    """Assemble a pick value exactly as the brick shaders lay it out."""
    value = global_id & 0xFFFFF
    shift = 20
    for field, width in zip(q, widths, strict=True):
        value |= (int(field) & ((1 << width) - 1)) << shift
        shift += width
    return value


@pytest.mark.parametrize(
    "dataset_size", [(16, 32, 64), (2048, 10000, 40000), (32640, 32640, 32640)]
)
def test_3d_pick_layout_round_trips(offscreen_renderer, dataset_size):
    """id + three layout-sized fields decode back to the position encoded.

    Locks the layout the 3D label and volume brick shaders share.  If this
    breaks, ``LabelsPickInfo.data_coordinate`` silently starts reporting
    the wrong voxel.
    """
    from cellier.render._pick_layout import pick_bit_layout
    from cellier.render.visuals._image import NormSizedVolume

    norm_size = np.array([2.0, 4.0, 8.0])
    node = NormSizedVolume(
        gfx.Geometry(grid=gfx.Texture(np.zeros((2, 2, 2), np.float32), dim=3)),
        gfx.VolumeMipMaterial(),
        norm_size=norm_size,
        dataset_size=np.asarray(dataset_size, dtype=np.float64),
    )
    layout = pick_bit_layout(dataset_size)
    assert node.pick_layout == layout
    assert sum(layout.widths) <= 44

    size = np.asarray(dataset_size, dtype=np.float64)
    for frac in [(0.0, 0.0, 0.0), (0.5, 0.25, 0.75), (0.999, 0.999, 0.999)]:
        voxel = np.asarray(frac) * size
        q = np.floor(voxel * 2.0 ** np.asarray(layout.frac_bits)).astype(int)
        info = node._wgpu_get_pick_info(_pack_3d(12345, q, layout.widths))
        decoded = layout.decode(q)
        expected = (decoded / size - 0.5) * norm_size
        assert info["norm_pos"] == pytest.approx(expected, abs=1e-12)
        assert info["voxel_pos"] == pytest.approx(decoded, abs=0.0)
        assert (np.abs(decoded - voxel) <= 0.5 * np.asarray(layout.step)).all()
        # "index" is deliberately absent so callers cannot take the wrong
        # decode path; that contract is part of the layout.
        assert "index" not in info


def test_3d_pick_layout_keeps_global_id_in_the_low_20_bits(offscreen_renderer):
    """The outline shader reads ``global_id`` from bits 0..19 of every visual.

    Every pygfx and cellier shader starts its payload with
    ``pick_pack(global_id, 20)``.  The outline composite pass depends on
    that being uniform across visual types, for any field widths.
    """
    from cellier.render._pick_layout import pick_bit_layout

    for dataset_size in [(16, 32, 64), (100000, 100000, 2000)]:
        widths = pick_bit_layout(dataset_size).widths
        top = [(1 << w) - 1 for w in widths]
        for global_id in (1, 0xFFFFF, 818773):
            value = _pack_3d(global_id, top, widths)
            assert value & 0xFFFFF == global_id
            assert value < 2**64


def test_outline_target_does_not_disturb_the_pick_payload(offscreen_renderer):
    """Adding a fourth fragment output leaves pick byte-identical.

    Renders the same scene through the stock blender and through
    ``CellierBlender`` and compares what ``get_pick_info`` reports at a
    grid of positions.  This is the direct check that the extra target
    leaves picking untouched.
    """
    from rendercanvas.offscreen import RenderCanvas

    geometry = gfx.box_geometry(1.5, 1.5, 1.5)
    material = gfx.MeshBasicMaterial(color="#ff8800", pick_write=True)
    mesh = gfx.Mesh(geometry, material)
    scene = gfx.Scene()
    scene.add(mesh)
    camera = gfx.OrthographicCamera()
    camera.show_object(scene)

    def _pick_all(with_target: bool):
        canvas = RenderCanvas(size=(64, 64), pixel_ratio=1)
        renderer = gfx.WgpuRenderer(canvas)
        renderer.pixel_scale = 1
        renderer.ppaa = "none"
        if with_target:
            assert ensure_extra_targets(renderer, [OUTLINE_ID_TARGET]) is True
        canvas.request_draw(lambda: renderer.render(scene, camera))
        canvas.draw()
        results = []
        for y in range(8, 64, 8):
            for x in range(8, 64, 8):
                info = renderer.get_pick_info((x, y))
                results.append(
                    (
                        info.get("world_object") is mesh,
                        info.get("face_index"),
                        tuple(info.get("face_coord") or ()),
                    )
                )
        return results

    stock = _pick_all(with_target=False)
    with_target = _pick_all(with_target=True)

    assert any(hit for hit, _f, _c in stock), "the probe grid never hit the mesh"
    assert stock == with_target
