"""3D label volume material and shader (r32sint texture, categorical iso)."""

from __future__ import annotations

from pathlib import Path

import pygfx as gfx
import wgpu
from pygfx.objects import Volume
from pygfx.renderers.wgpu import (
    Binding,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.volumeshader import BaseVolumeShader

from cellier.render.shaders._label_colormap import (
    build_outline_selection_texture,
)

_WGSL_DIR = Path(__file__).parent / "wgsl"

_LABEL_VOLUME_WGSL = (_WGSL_DIR / "label_volume.wgsl").read_text()

_VERTEX_AND_FRAGMENT = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


class LabelVolumeMaterial(gfx.VolumeBasicMaterial):
    """3D in-memory label volume material for categorical ISO rendering.

    Parameters
    ----------
    background_label : int
        Label ID to skip during ray marching. Default 0.
    colormap_mode : "random" or "direct"
        Frozen after construction.
    salt : int
        Hash seed for random mode.
    render_mode : "iso_categorical" or "flat_categorical"
        Whether to apply Lambertian shading.
    label_keys_texture : gfx.Texture or None
        Sorted int32 label-ID texture for direct mode.
    label_colors_texture : gfx.Texture or None
        RGBA float32 color texture for direct mode.
    n_entries : int
        Number of entries in the direct-mode LUT.
    label_params_buffer : gfx.Buffer
        Uniform buffer containing background_label, salt, n_entries, _pad.
    """

    def __init__(
        self,
        background_label: int = 0,
        colormap_mode: str = "random",
        salt: int = 0,
        render_mode: str = "iso_categorical",
        label_keys_texture=None,
        label_colors_texture=None,
        n_entries: int = 0,
        label_params_buffer=None,
        outline_selection_texture=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.background_label = background_label
        self.colormap_mode = colormap_mode
        self.salt = salt
        self.render_mode = render_mode
        self.label_keys_texture = label_keys_texture
        self.label_colors_texture = label_colors_texture
        self.n_entries = n_entries
        self.label_params_buffer = label_params_buffer
        # Sorted (label, slot) pairs the outline key binary-searches.  Always
        # present so ``has_outline_selection`` never flips: the texture is
        # bound at shader-build time, so swapping it in later would force a
        # pipeline rebuild.  ``n_outline_entries`` bounds the search instead.
        self.outline_selection_texture = (
            outline_selection_texture
            if outline_selection_texture is not None
            else build_outline_selection_texture()
        )


@register_wgpu_render_function(Volume, LabelVolumeMaterial)
class LabelVolumeShader(BaseVolumeShader):
    """Shader for 3D in-memory label volume rendering."""

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        self["colormap_mode"] = material.colormap_mode
        # Defaults for the outline_id target.  ``write_outline_id`` is
        # overridden by CellierBlender.get_shader_kwargs when the target
        # exists; without it the write compiles away, so the same shader
        # stays valid on a canvas using the stock blender.
        self["write_outline_id"] = False
        self["has_outline_selection"] = material.outline_selection_texture is not None
        self["render_mode"] = material.render_mode
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
            Binding(
                "u_label_params",
                "buffer/uniform",
                material.label_params_buffer,
                structname="LabelParams",
            ),
        ]

        # 3D label texture bound to both stages for textureDimensions in vertex shader.
        tex_view = GfxTextureView(geometry.grid)
        bindings.append(
            Binding("t_img", "texture/auto", tex_view, _VERTEX_AND_FRAGMENT)
        )

        # Direct-mode LUT textures.
        if (
            material.colormap_mode == "direct"
            and material.label_keys_texture is not None
        ):
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
        return _LABEL_VOLUME_WGSL
