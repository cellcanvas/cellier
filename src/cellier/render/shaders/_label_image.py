"""2D label image material and shader (r32sint slice, categorical colormap)."""

from __future__ import annotations

from pathlib import Path

import pygfx as gfx
import wgpu
from pygfx.objects import Image
from pygfx.renderers.wgpu import (
    Binding,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.imageshader import ImageShader

from cellier.render.shaders._label_colormap import (
    build_outline_selection_texture,
)

_WGSL_DIR = Path(__file__).parent / "wgsl"

_LABEL_IMAGE_WGSL = (_WGSL_DIR / "label_image.wgsl").read_text()

_VERTEX_AND_FRAGMENT = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


class LabelImageMaterial(gfx.ImageBasicMaterial):
    """2D label image material for r32sint slices with categorical coloring.

    Parameters
    ----------
    background_label : int
        Label ID to discard (transparent). Default 0.
    colormap_mode : "random" or "direct"
        Frozen after construction.
    salt : int
        Hash seed for random mode.
    label_keys_texture : gfx.Texture or None
        Sorted int32 label-ID texture for direct mode.
    label_colors_texture : gfx.Texture or None
        RGBA float32 color texture for direct mode.
    n_entries : int
        Number of entries in the direct-mode LUT.
    label_params_buffer : gfx.Buffer
        Uniform buffer containing background_label, salt, n_entries, _pad.
    opacity : float
    pick_write : bool
    """

    def __init__(
        self,
        background_label: int = 0,
        colormap_mode: str = "random",
        salt: int = 0,
        label_keys_texture=None,
        label_colors_texture=None,
        n_entries: int = 0,
        label_params_buffer=None,
        outline_selection_texture=None,
        opacity: float = 1.0,
        pick_write: bool = True,
        **kwargs,
    ) -> None:
        # Pass only opacity and pick_write; clim/map/interpolation irrelevant.
        super().__init__(opacity=opacity, pick_write=pick_write, **kwargs)
        self.background_label = background_label
        self.colormap_mode = colormap_mode
        self.salt = salt
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


@register_wgpu_render_function(Image, LabelImageMaterial)
class LabelImageShader(ImageShader):
    """Shader for 2D in-memory label rendering."""

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
        self["use_colormap"] = False

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

        # Label data texture (r32sint) bound to both stages for textureDimensions.
        tex_view = GfxTextureView(geometry.grid)
        bindings.append(
            Binding("t_img", "texture/auto", tex_view, _VERTEX_AND_FRAGMENT)
        )

        # Direct-mode LUT textures (omitted in random mode to avoid binding mismatch).
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
        return {"indices": (6, 1)}

    def get_code(self):
        return _LABEL_IMAGE_WGSL
