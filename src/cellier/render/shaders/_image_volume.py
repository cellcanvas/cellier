"""In-memory image volume materials and shader.

cellier renders in-memory 3D images through its own copy of pygfx's volume
raycaster rather than through ``gfx.VolumeMipMaterial`` and friends.  The
materials here are thin subclasses of the pygfx ones -- every property
(``clim``, ``map``, ``interpolation``, ``threshold``, ``step_size``,
``opacity``, ``pick_write``, the depth and alpha settings) behaves exactly
as it did -- and the shader differs from upstream only in the three ways
``image_volume.wgsl`` documents:

* the iso branch writes the ``normal`` render target, so ambient occlusion
  gets a 1.3-degree surface normal instead of a 34-degree one reconstructed
  from depth;
* the iso branch writes depth through the full 4x4 world transform, fixing
  a translated iso volume winning depth tests it should lose
  (``scripts/pygfx_iso_depth_bug.py``);
* the normal is transformed by the inverse transpose, which matters for the
  anisotropic voxel spacing cellier renders routinely.

Registration is per material subclass rather than against
``gfx.VolumeRayMaterial``: pygfx's registry resolves by MRO, so registering
against the base would take over rendering for every pygfx volume material
in the process, including ones cellier did not create.
"""

from __future__ import annotations

from pathlib import Path

import pygfx as gfx
from pygfx.objects import Volume
from pygfx.renderers.wgpu import Binding, register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.volumeshader import VolumeRayShader

from cellier.render._render_planes import (
    RENDER_PLANES_STRUCT,
    make_render_planes_buffer,
)
from cellier.visuals._render_plane import PLANE_RENDER_MODE

_WGSL_DIR = Path(__file__).parent / "wgsl"

IMAGE_VOLUME_WGSL: str = (_WGSL_DIR / "image_volume.wgsl").read_text()


class ImageVolumeMipMaterial(gfx.VolumeMipMaterial):
    """Maximum intensity projection, rendered by cellier's shader."""


class ImageVolumeMinipMaterial(gfx.VolumeMinipMaterial):
    """Minimum intensity projection, rendered by cellier's shader."""


class ImageVolumeIsoMaterial(gfx.VolumeIsoMaterial):
    """Isosurface, rendered by cellier's shader.

    This is the one that actually differs from upstream: it writes a real
    surface normal to the ``normal`` render target and its depth accounts
    for the volume's translation.
    """


class ImageVolumePlaneMaterial(gfx.VolumeRayMaterial):
    """The ``"plane"`` render mode: the volume drawn on its render planes.

    One sample per fragment where the view ray meets the nearest plane
    (plane rendering design v3, 5.1).  The planes are not state of the
    material: they are read from ``render_planes_buffer``, which the render
    visual owns and shares between its materials.

    Parameters
    ----------
    render_planes_buffer : gfx.Buffer or None
        The visual's ``u_render_planes`` buffer.  ``None`` gives the
        material a buffer of its own with no plane, which draws nothing.
    **kwargs :
        As for ``gfx.VolumeRayMaterial``.
    """

    render_mode = PLANE_RENDER_MODE

    def __init__(self, render_planes_buffer: gfx.Buffer | None = None, **kwargs):
        super().__init__(**kwargs)
        self.render_planes_buffer = (
            render_planes_buffer
            if render_planes_buffer is not None
            else make_render_planes_buffer()
        )


#: ``InMemoryImageAppearance.render_mode`` -> material class.
IMAGE_VOLUME_MATERIALS: dict[str, type] = {
    "mip": ImageVolumeMipMaterial,
    "iso": ImageVolumeIsoMaterial,
    "minip": ImageVolumeMinipMaterial,
    PLANE_RENDER_MODE: ImageVolumePlaneMaterial,
}


@register_wgpu_render_function(Volume, ImageVolumePlaneMaterial)
@register_wgpu_render_function(Volume, ImageVolumeMipMaterial)
@register_wgpu_render_function(Volume, ImageVolumeMinipMaterial)
@register_wgpu_render_function(Volume, ImageVolumeIsoMaterial)
class ImageVolumeShader(VolumeRayShader):
    """Volume raycaster with cellier's iso fixes and the normal target.

    Everything except the WGSL is inherited: bindings, the pipeline info
    (front-face culling, so the back planes are the reference), the render
    info, and the ``mode`` template var that selects the raycast body.
    """

    def __init__(self, wobject) -> None:
        super().__init__(wobject)
        # Default for the ``normal`` render target.  ``write_normal`` is
        # overridden by CellierBlender.get_shader_kwargs when the target
        # exists; without it the write compiles away, so the same shader
        # stays valid on a canvas using the stock blender.
        self["write_normal"] = False

    def get_bindings(self, wobject, shared, scene):
        """The inherited bindings, plus the render planes in plane mode."""
        bindings = super().get_bindings(wobject, shared, scene)
        material = wobject.material
        if material.render_mode == PLANE_RENDER_MODE:
            # Bound for this mode only, so the other modes' shaders are
            # what they were before the mode existed.
            group = bindings[0]
            index = len(group)
            group[index] = Binding(
                "u_render_planes",
                "buffer/uniform",
                material.render_planes_buffer,
                "FRAGMENT",
                structname=RENDER_PLANES_STRUCT,
            )
            self.define_binding(0, index, group[index])
        return bindings

    def get_code(self) -> str:
        """Return cellier's volume raycasting WGSL."""
        return IMAGE_VOLUME_WGSL
