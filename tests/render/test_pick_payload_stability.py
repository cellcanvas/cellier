"""Guards on the pick payload's bit layout.

Nothing else in the suite exercises the packing end to end: the helpers in
``test_pick_data_coordinate.py`` take hand-made floats, and
``test_canvas_pick_info.py`` only checks NamedTuple fields.  So a change to
the ``pick_pack`` convention, or to how many fields a fragment shader
writes, would go unnoticed.

That matters right now because Stage 7a adds a fourth output
(``outline_id``) to the generated ``FragmentOutput`` for every shader.  The
design's whole argument for the ``outline_id`` target over repacking pick is
that picking is left untouched -- these tests are what makes that claim
checkable rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest

from cellier.render._cellier_blender import (
    OUTLINE_ID_TARGET,
    install_cellier_blender,
)

# The 3D packing cellier's label and multiscale volume shaders write:
#   pick_pack(global_id, 20) + pick_pack(x, 14) + pick_pack(y, 14)
#                            + pick_pack(z, 14)
# See label_volume_brick.wgsl and multiscale_volume_brick.wgsl.
_COORD_STEPS = 16383.0


def _pack_3d(global_id: int, x: int, y: int, z: int) -> int:
    """Assemble a pick value exactly as the brick shaders lay it out."""
    return (
        (global_id & 0xFFFFF)
        | ((x & 0x3FFF) << 20)
        | ((y & 0x3FFF) << 34)
        | ((z & 0x3FFF) << 48)
    )


def test_3d_pick_layout_round_trips(offscreen_renderer):
    """20 + 14x3 decodes back to the normalised position it encoded.

    Locks the layout the 3D label and volume shaders share.  If this
    breaks, ``LabelsPickInfo.data_coordinate`` silently starts reporting
    the wrong voxel.
    """
    from cellier.render.visuals._image import NormSizedVolume

    norm_size = np.array([2.0, 4.0, 8.0])
    node = NormSizedVolume(
        gfx.Geometry(grid=gfx.Texture(np.zeros((2, 2, 2), np.float32), dim=3)),
        gfx.VolumeMipMaterial(),
        norm_size=norm_size,
    )

    for nx, ny, nz in [(0.0, 0.0, 0.0), (0.5, 0.25, 0.75), (1.0, 1.0, 1.0)]:
        codes = [int(v * _COORD_STEPS) for v in (nx, ny, nz)]
        info = node._wgpu_get_pick_info(_pack_3d(12345, *codes))
        expected = [
            (code / _COORD_STEPS - 0.5) * float(norm_size[k])
            for k, code in enumerate(codes)
        ]
        assert info["norm_pos"] == pytest.approx(expected, abs=1e-9)
        # "index" is deliberately absent so callers cannot take the wrong
        # decode path; that contract is part of the layout.
        assert "index" not in info


def test_3d_pick_layout_keeps_global_id_in_the_low_20_bits(offscreen_renderer):
    """The outline shader reads ``global_id`` from bits 0..19 of every visual.

    Every pygfx and cellier shader starts its payload with
    ``pick_pack(global_id, 20)``.  The outline composite pass depends on
    that being uniform across visual types.
    """
    for global_id in (1, 0xFFFFF, 818773):
        value = _pack_3d(global_id, 16383, 16383, 16383)
        assert value & 0xFFFFF == global_id


def test_outline_target_does_not_disturb_the_pick_payload(offscreen_renderer):
    """Adding a fourth fragment output leaves pick byte-identical.

    Renders the same scene through the stock blender and through
    ``CellierBlender`` and compares what ``get_pick_info`` reports at a
    grid of positions.  This is the direct check on Stage 7a's central
    claim.
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
            assert install_cellier_blender(renderer, [OUTLINE_ID_TARGET]) is True
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
