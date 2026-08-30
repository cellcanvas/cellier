"""Label multiscale shaders.

LabelVolumeBrickMaterial + LabelVolumeBrickShader — 3D categorical iso rendering.
LabelBlockMaterial + LabelBlockShader — 2D tile-cache label rendering.

Importing this module registers both shaders with pygfx via
@register_wgpu_render_function.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx
import wgpu
from pygfx.objects import Image, Volume
from pygfx.renderers.wgpu import (
    Binding,
    GfxSampler,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.imageshader import ImageShader
from pygfx.renderers.wgpu.shaders.volumeshader import BaseVolumeShader

from cellier.render.shaders._label_colormap import (
    build_outline_selection_texture,
)

if TYPE_CHECKING:
    from pygfx.resources import Buffer

_WGSL_DIR = Path(__file__).parent / "wgsl"

_LABEL_VOLUME_BRICK_WGSL = (_WGSL_DIR / "label_volume_brick.wgsl").read_text()
_LABEL_BLOCK_WGSL = (_WGSL_DIR / "label_block.wgsl").read_text()

_vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


# ---------------------------------------------------------------------------
# 3D material
# ---------------------------------------------------------------------------


class LabelVolumeBrickMaterial(gfx.VolumeBasicMaterial):
    """Volume material for 3D multiscale label rendering (brick cache).

    Parameters
    ----------
    cache_texture : gfx.Texture
        3D int32 texture — the fixed-size brick cache.
    lut_texture : gfx.Texture
        RGBA8UI 3D texture — the per-brick address lookup table.
    brick_max_texture : gfx.Texture
        R32Float 3D texture — per-brick contains_label flag (0.0 / 1.0).
    vol_params_buffer : Buffer
        VolParams uniform buffer.
    block_scales_buffer : Buffer
        BlockScales uniform buffer.
    label_params_buffer : Buffer
        LabelParams uniform buffer (background_label, salt, n_entries).
    label_keys_texture : gfx.Texture | None
        Direct-mode sorted label keys (r32sint).
    label_colors_texture : gfx.Texture | None
        Direct-mode RGBA per entry (rgba32float).
    background_label : int
        Label ID treated as transparent.
    colormap_mode : "random" | "direct"
        Colormap mode (baked into shader; frozen after construction).
    salt : int
        Hash seed for random mode.
    render_mode : "iso_categorical" | "flat_categorical" | "gradient_debug" \
        | "smooth_iso"
        3D render mode.  ``gradient_debug`` renders abs(normalize(gradient))
        as RGB (R=|X|, G=|Y|, B=|Z|); flat regions with no gradient show as
        solid blue.  Use to diagnose brick-boundary normal discontinuities.
        ``smooth_iso`` uses a soft trilinear density field for bisection and
        a 3x3x3 Sobel normal kernel for smooth curved shading.
    """

    def __init__(
        self,
        cache_texture: gfx.Texture,
        lut_texture: gfx.Texture,
        brick_max_texture: gfx.Texture,
        vol_params_buffer: Buffer,
        block_scales_buffer: Buffer,
        label_params_buffer: Buffer,
        label_keys_texture: gfx.Texture | None = None,
        label_colors_texture: gfx.Texture | None = None,
        background_label: int = 0,
        colormap_mode: str = "random",
        salt: int = 0,
        render_mode: str = "iso_categorical",
        n_entries: int = 0,
        outline_selection_texture: gfx.Texture | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cache_texture = cache_texture
        self.lut_texture = lut_texture
        self.brick_max_texture = brick_max_texture
        self.vol_params_buffer = vol_params_buffer
        self.block_scales_buffer = block_scales_buffer
        self.label_params_buffer = label_params_buffer
        self.label_keys_texture = label_keys_texture
        self.label_colors_texture = label_colors_texture
        self.background_label = background_label
        self.colormap_mode = colormap_mode
        self.salt = salt
        self.render_mode = render_mode
        self.n_entries = n_entries
        # Sorted (label, slot) pairs the outline key binary-searches.  Always
        # present so ``has_outline_selection`` never flips: the texture is
        # bound at shader-build time, so swapping it in later would force a
        # pipeline rebuild.  ``n_outline_entries`` bounds the search instead.
        self.outline_selection_texture = (
            outline_selection_texture
            if outline_selection_texture is not None
            else build_outline_selection_texture()
        )


# ---------------------------------------------------------------------------
# 3D shader
# ---------------------------------------------------------------------------


@register_wgpu_render_function(Volume, LabelVolumeBrickMaterial)
class LabelVolumeBrickShader(BaseVolumeShader):
    """Shader for 3D multiscale label brick-cache rendering."""

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        m = wobject.material
        self["colormap_mode"] = m.colormap_mode
        # Defaults for the outline_id target.  ``write_outline_id`` is
        # overridden by CellierBlender.get_shader_kwargs when the target
        # exists; without it the write compiles away, so the same shader
        # stays valid on a canvas using the stock blender.
        self["write_outline_id"] = False
        self["has_outline_selection"] = m.outline_selection_texture is not None
        self["render_mode"] = m.render_mode
        # Default for the ``normal`` render target.  ``write_normal`` is
        # overridden by CellierBlender.get_shader_kwargs when the target
        # exists; without it the write compiles away, so the same shader
        # stays valid on a canvas using the stock blender.
        self["write_normal"] = False

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Proxy texture for Volume geometry.
        proxy_view = GfxTextureView(geometry.grid)
        proxy_sampler = GfxSampler("nearest", "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", proxy_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_img", "texture/auto", proxy_view, _vertex_and_fragment)
        )

        # Label brick cache (int32, no sampler needed).
        cache_view = GfxTextureView(material.cache_texture)
        bindings.append(Binding("t_cache", "texture/auto", cache_view, "FRAGMENT"))

        # LUT texture (per-brick slot indices).
        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(Binding("t_lut", "texture/auto", lut_view, "FRAGMENT"))

        # Per-brick contains_label flag (R32Float, 0.0 or 1.0).
        brick_max_view = GfxTextureView(material.brick_max_texture)
        bindings.append(
            Binding("t_brick_max", "texture/auto", brick_max_view, "FRAGMENT")
        )

        # Label params uniform.
        bindings.append(
            Binding(
                "u_label_params",
                "buffer/uniform",
                material.label_params_buffer,
                _vertex_and_fragment,
                structname="LabelParams",
            )
        )

        # Vol params uniform.
        bindings.append(
            Binding(
                "u_vol_params",
                "buffer/uniform",
                material.vol_params_buffer,
                _vertex_and_fragment,
                structname="VolParams",
            )
        )

        # Block scales uniform.
        bindings.append(
            Binding(
                "u_block_scales",
                "buffer/uniform",
                material.block_scales_buffer,
                "FRAGMENT",
                structname="BlockScales",
            )
        )

        # Direct-mode LUT textures (only when colormap_mode == "direct").
        if material.label_keys_texture is not None:
            bindings.append(
                Binding(
                    "t_label_keys",
                    "texture/auto",
                    GfxTextureView(material.label_keys_texture),
                    "FRAGMENT",
                )
            )
            bindings.append(
                Binding(
                    "t_label_colors",
                    "texture/auto",
                    GfxTextureView(material.label_colors_texture),
                    "FRAGMENT",
                )
            )

        # Outline selection lookup (omitted when nothing is selected, to
        # avoid a binding mismatch -- same pattern as the direct-mode LUT).
        if material.outline_selection_texture is not None:
            bindings.append(
                Binding(
                    "t_outline_selection",
                    "texture/auto",
                    GfxTextureView(material.outline_selection_texture),
                    "FRAGMENT",
                )
            )
        bindings = dict(enumerate(bindings))
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (36, 1)}

    def get_code(self):
        return _LABEL_VOLUME_BRICK_WGSL


# ---------------------------------------------------------------------------
# 2D material
# ---------------------------------------------------------------------------


class LabelBlockMaterial(gfx.ImageBasicMaterial):
    """Image material for 2D multiscale label tile-cache rendering.

    Parameters
    ----------
    cache_texture : gfx.Texture
        2D int32 texture — the fixed-size tile cache.
    lut_texture : gfx.Texture
        Float32 2D LUT texture (slot indices + level per tile).
    lut_params_buffer : Buffer
        LutParams uniform buffer.
    block_scales_buffer : Buffer
        BlockScales uniform buffer.
    label_params_buffer : Buffer
        LabelParams uniform buffer.
    label_keys_texture : gfx.Texture | None
        Direct-mode sorted label keys.
    label_colors_texture : gfx.Texture | None
        Direct-mode RGBA per entry.
    paint_cache_texture : gfx.Texture | None
        2D float32 paint cache (stripes-of-tiles layout).
    paint_lut_texture : gfx.Texture | None
        2D float32 paint LUT (slot_index, presence) per tile.
    colormap_mode : "random" | "direct"
        Colormap mode (baked into shader).
    background_label : int
        Label ID treated as transparent.
    """

    def __init__(
        self,
        cache_texture: gfx.Texture,
        lut_texture: gfx.Texture,
        lut_params_buffer: Buffer,
        block_scales_buffer: Buffer,
        label_params_buffer: Buffer,
        label_keys_texture: gfx.Texture | None = None,
        label_colors_texture: gfx.Texture | None = None,
        paint_cache_texture: gfx.Texture | None = None,
        paint_lut_texture: gfx.Texture | None = None,
        colormap_mode: str = "random",
        background_label: int = 0,
        n_entries: int = 0,
        outline_selection_texture: gfx.Texture | None = None,
        **kwargs,
    ) -> None:
        super().__init__(interpolation="nearest", **kwargs)
        self.cache_texture = cache_texture
        self.lut_texture = lut_texture
        self.lut_params_buffer = lut_params_buffer
        self.block_scales_buffer = block_scales_buffer
        self.label_params_buffer = label_params_buffer
        self.label_keys_texture = label_keys_texture
        self.label_colors_texture = label_colors_texture
        self.paint_cache_texture = paint_cache_texture
        self.paint_lut_texture = paint_lut_texture
        self.colormap_mode = colormap_mode
        self.background_label = background_label
        self.n_entries = n_entries
        # Sorted (label, slot) pairs the outline key binary-searches.  Always
        # present so ``has_outline_selection`` never flips: the texture is
        # bound at shader-build time, so swapping it in later would force a
        # pipeline rebuild.  ``n_outline_entries`` bounds the search instead.
        self.outline_selection_texture = (
            outline_selection_texture
            if outline_selection_texture is not None
            else build_outline_selection_texture()
        )


# ---------------------------------------------------------------------------
# 2D shader
# ---------------------------------------------------------------------------


@register_wgpu_render_function(Image, LabelBlockMaterial)
class LabelBlockShader(ImageShader):
    """Shader for 2D multiscale label tile-cache rendering."""

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        m = wobject.material
        self["colormap_mode"] = m.colormap_mode
        # Defaults for the outline_id target.  ``write_outline_id`` is
        # overridden by CellierBlender.get_shader_kwargs when the target
        # exists; without it the write compiles away, so the same shader
        # stays valid on a canvas using the stock blender.
        self["write_outline_id"] = False
        self["has_outline_selection"] = m.outline_selection_texture is not None
        self["use_colormap"] = False

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Proxy texture for Image geometry (drives get_im_geometry quad size).
        proxy_view = GfxTextureView(geometry.grid)
        proxy_sampler = GfxSampler("nearest", "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", proxy_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_img", "texture/auto", proxy_view, _vertex_and_fragment)
        )

        # Label tile cache (int32, textureLoad, no sampler).
        cache_view = GfxTextureView(material.cache_texture)
        bindings.append(Binding("t_cache", "texture/auto", cache_view, "FRAGMENT"))

        # LUT texture.
        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(Binding("t_lut", "texture/auto", lut_view, "FRAGMENT"))

        # Label params uniform.
        bindings.append(
            Binding(
                "u_label_params",
                "buffer/uniform",
                material.label_params_buffer,
                "FRAGMENT",
                structname="LabelParams",
            )
        )

        # LUT params uniform.
        bindings.append(
            Binding(
                "u_lut_params",
                "buffer/uniform",
                material.lut_params_buffer,
                "FRAGMENT",
                structname="LutParams",
            )
        )

        # Block scales uniform.
        bindings.append(
            Binding(
                "u_block_scales",
                "buffer/uniform",
                material.block_scales_buffer,
                "FRAGMENT",
                structname="BlockScales",
            )
        )

        # Direct-mode LUT textures.
        if material.label_keys_texture is not None:
            bindings.append(
                Binding(
                    "t_label_keys",
                    "texture/auto",
                    GfxTextureView(material.label_keys_texture),
                    "FRAGMENT",
                )
            )
            bindings.append(
                Binding(
                    "t_label_colors",
                    "texture/auto",
                    GfxTextureView(material.label_colors_texture),
                    "FRAGMENT",
                )
            )

        # Paint cache + LUT.  The WGSL always samples both, so fall back to
        # 1x1 zero textures when the material was built without paint resources.
        if material.paint_cache_texture is None:
            paint_cache_view = GfxTextureView(
                gfx.Texture(np.zeros((1, 1, 2), dtype=np.float32), dim=2, format="2xf4")
            )
        else:
            paint_cache_view = GfxTextureView(material.paint_cache_texture)
        bindings.append(
            Binding("t_paint_cache", "texture/auto", paint_cache_view, "FRAGMENT")
        )

        if material.paint_lut_texture is None:
            paint_lut_view = GfxTextureView(
                gfx.Texture(np.zeros((1, 1, 2), dtype=np.float32), dim=2, format="2xf4")
            )
        else:
            paint_lut_view = GfxTextureView(material.paint_lut_texture)
        bindings.append(
            Binding("t_paint_lut", "texture/auto", paint_lut_view, "FRAGMENT")
        )

        # Outline selection lookup (omitted when nothing is selected, to
        # avoid a binding mismatch -- same pattern as the direct-mode LUT).
        if material.outline_selection_texture is not None:
            bindings.append(
                Binding(
                    "t_outline_selection",
                    "texture/auto",
                    GfxTextureView(material.outline_selection_texture),
                    "FRAGMENT",
                )
            )
        bindings = dict(enumerate(bindings))
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (6, 1)}

    def get_code(self):
        return _LABEL_BLOCK_WGSL
