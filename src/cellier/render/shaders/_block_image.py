"""Image material using tile-cache + LUT indirection rendering.

Inherits from ``ImageBasicMaterial`` to get ``clim``, ``gamma``,
``map``, and ``interpolation`` properties.  Carries extra texture and
buffer references for the tile cache system.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pygfx as gfx
import wgpu
from pygfx.objects import Image
from pygfx.renderers.wgpu import (
    Binding,
    GfxSampler,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.imageshader import ImageShader

if TYPE_CHECKING:
    from pygfx.resources import Buffer

_WGSL_DIR = Path(__file__).parent / "wgsl"
IMAGE_BLOCK_WGSL = (_WGSL_DIR / "image_block.wgsl").read_text()


class ImageBlockMaterial(gfx.ImageBasicMaterial):
    """Image material using tile-cache + LUT indirection rendering.

    Parameters
    ----------
    cache_texture : gfx.Texture
        2D float32 texture -- the fixed-size tile cache.
    lut_texture : gfx.Texture
        RGBA float32 2D texture -- the per-tile address lookup table.
        Shape ``(gH, gW, 4)``.
    lut_params_buffer : Buffer
        Uniform buffer containing LUT spatial parameters.
    block_scales_buffer : Buffer
        Uniform buffer with per-level scale factors (10 x vec4).
    clim : tuple[float, float]
        Contrast limits.
    map : gfx.TextureMap, optional
        1D colourmap texture.
    interpolation : str
        Sampler filter for the cache texture.
    """

    def __init__(
        self,
        cache_texture: gfx.Texture,
        lut_texture: gfx.Texture,
        lut_params_buffer: Buffer,
        block_scales_buffer: Buffer,
        clim: tuple[float, float] = (0.0, 1.0),
        map: gfx.TextureMap | None = None,
        interpolation: str = "nearest",
        **kwargs,
    ) -> None:
        super().__init__(
            clim=clim,
            map=map,
            interpolation=interpolation,
            **kwargs,
        )
        self.cache_texture = cache_texture
        self.lut_texture = lut_texture
        self.lut_params_buffer = lut_params_buffer
        self.block_scales_buffer = block_scales_buffer


_vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


@register_wgpu_render_function(Image, ImageBlockMaterial)
class ImageBlockShader(ImageShader):
    """Shader for LUT-based tile-cache 2D image rendering."""

    type = "render"

    def get_bindings(self, wobject, shared, scene):
        """Return all GPU resource bindings for the image block shader.

        Binds:
        - t_img / s_img: proxy texture (grid-dim), drives geometry
        - t_cache / s_cache: tile cache texture
        - t_lut: float32 LUT texture (textureLoad, no sampler)
        - u_lut_params: LutParams uniform
        - u_block_scales: BlockScales uniform
        """
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Proxy texture -- grid-dim, drives get_im_geometry() for quad size.
        proxy_view = GfxTextureView(geometry.grid)
        proxy_sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", proxy_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_img", "texture/auto", proxy_view, _vertex_and_fragment)
        )

        # Cache texture + sampler -- actual tile data.
        cache_view = GfxTextureView(material.cache_texture)
        cache_sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(
            Binding("s_cache", "sampler/filtering", cache_sampler, "FRAGMENT")
        )
        bindings.append(Binding("t_cache", "texture/auto", cache_view, "FRAGMENT"))

        # Colourmap.
        if material.map is not None:
            bindings.extend(self.define_img_colormap(material.map))

        # LUT texture (textureLoad, no sampler needed).
        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(Binding("t_lut", "texture/auto", lut_view, "FRAGMENT"))

        # Uniform buffers.
        # structname= tells pygfx to auto-generate the WGSL struct
        # from the numpy dtype. Do NOT write the struct manually in WGSL.
        bindings.append(
            Binding(
                "u_lut_params",
                "buffer/uniform",
                material.lut_params_buffer,
                "FRAGMENT",
                structname="LutParams",
            )
        )
        bindings.append(
            Binding(
                "u_block_scales",
                "buffer/uniform",
                material.block_scales_buffer,
                "FRAGMENT",
                structname="BlockScales",
            )
        )

        bindings = dict(enumerate(bindings))
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        """Return pipeline configuration."""
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        """Return draw-call parameters."""
        return {"indices": (6, 1)}

    def get_code(self):
        """Return the WGSL source for this shader."""
        return IMAGE_BLOCK_WGSL
