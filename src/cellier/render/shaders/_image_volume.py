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
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.volumeshader import VolumeRayShader

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


#: ``InMemoryImageAppearance.render_mode`` -> material class.
IMAGE_VOLUME_MATERIALS: dict[str, type] = {
    "mip": ImageVolumeMipMaterial,
    "iso": ImageVolumeIsoMaterial,
    "minip": ImageVolumeMinipMaterial,
}


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

    def get_code(self) -> str:
        """Return cellier's volume raycasting WGSL."""
        return IMAGE_VOLUME_WGSL
