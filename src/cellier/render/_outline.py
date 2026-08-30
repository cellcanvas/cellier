"""Screen-space outline effect pass.

A single ``EffectPass`` that reads the renderer's pick buffer, derives a
per-pixel outline key, tests four neighbouring taps at the configured
thickness, and composites a coloured outline over the rendered image.

It follows the structure of
:class:`~cellier.render._temporal_accumulation.TemporalAccumulationPass` --
a custom ``render()`` driving an inner full-quad pass -- but owns no
private textures: it composites straight into the effect-chain target.

``_OutlineQuadPass`` is deliberately **not** a ``FullQuadPass`` subclass.
That class builds its binding layout from a naming convention knowing only
``float`` and ``depth`` sample types, so an ``rgba16uint`` pick texture
cannot be declared through it.  Instead the layout is built here and handed
to the module-level ``create_full_quad_pipeline``, which is the documented
low-level entry point ``FullQuadPass`` itself is built on -- so the vertex
shader, the ``Varyings`` struct and the pipeline cache are all shared with
every other effect pass.

Thickness is measured in **internal pixels**.  Effect passes run before the
output pass's SSAA downsample, so at ``pixel_ratio > 1`` the on-screen band
is correspondingly thinner than the configured number.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import wgpu
from pygfx.renderers.wgpu.engine.binding import Binding
from pygfx.renderers.wgpu.engine.effectpasses import (
    EffectPass,
    create_full_quad_pipeline,
)
from pygfx.renderers.wgpu.engine.shared import get_shared
from pygfx.renderers.wgpu.shader.bindings import BindingDefinitions
from pygfx.renderers.wgpu.shader.templating import apply_templating
from pygfx.utils import array_from_shadertype

from cellier.render._cellier_blender import OUTLINE_ID_TARGET, get_extra_target_view
from cellier.render._pick_buffer import get_pick_view
from cellier.render._visual_lut import MAX_SLOT, get_shared_visual_lut

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    import pygfx as gfx

    from cellier.render._config import OutlineConfig
    from cellier.render._visual_lut import VisualLut

_WGSL_DIR = Path(__file__).parent / "shaders" / "wgsl"
OUTLINE_WGSL: str = (_WGSL_DIR / "outline.wgsl").read_text()

RGBA = tuple[float, float, float, float]


class _OutlineQuadPass:
    """Full-quad pass binding colour, pick, and the outline LUT.

    Parameters
    ----------
    palette_size : int
        Number of palette entries the uniform carries.  Fixed at
        ``MAX_SLOT`` so a palette change never recompiles the shader.
    """

    uniform_type: ClassVar[dict] = {
        "palette": f"{MAX_SLOT}*4xf4",
        "boundary_color": "4xf4",
        "inner_color": "4xf4",
        "boundaries_enabled": "i4",
        "selection_enabled": "i4",
    }

    def __init__(self) -> None:
        self._device = get_shared().device

        self._uniform_data = array_from_shadertype(self.uniform_type)
        self._buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        definitions = BindingDefinitions()
        definitions.define_binding(
            0, 0, Binding("u_outline", "buffer/uniform", self._uniform_data.dtype)
        )
        self._uniform_definition = definitions.get_code()
        # The shader only ever uses textureLoad (integer textures cannot be
        # filtered), but the layout keeps a sampler at binding 1 so the
        # binding indices line up with every other pygfx effect pass.
        self._sampler = self._device.create_sampler(
            min_filter="linear", mag_filter="linear"
        )

        self._template_vars: dict[str, object] = {
            "b_t_in": 1,
            "b_t_out": 0,
            "s_t_in": 2,
            "s_t_out": 2,
            "inner_t": 0,
            "has_inward": True,
            "has_outward": False,
            "has_outline_id": False,
        }
        self._pipeline = None
        self._pipeline_hash: object = None

        self._uniform_data["boundaries_enabled"] = 0
        self._uniform_data["selection_enabled"] = 1

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_template_vars(self, **kwargs: object) -> None:
        """Update shader template vars, forcing a recompile if they changed."""
        for name, value in kwargs.items():
            if self._template_vars.get(name) != value:
                self._template_vars[name] = value
                self._pipeline = None

    def set_uniform(self, name: str, value: object) -> None:
        """Write one uniform field.  Never triggers a recompile."""
        self._uniform_data[name] = value

    def set_palette(self, palette: Sequence[RGBA]) -> None:
        """Fill the palette uniform, padding unused slots with transparent."""
        for index in range(MAX_SLOT):
            if index < len(palette):
                self._uniform_data["palette"][index] = palette[index]
            else:
                self._uniform_data["palette"][index] = (0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        command_encoder: wgpu.GPUCommandEncoder,
        color_tex: wgpu.GPUTextureView,
        pick_tex: wgpu.GPUTextureView,
        lut_tex: wgpu.GPUTextureView,
        target_tex: wgpu.GPUTextureView,
        outline_id_tex: wgpu.GPUTextureView | None = None,
    ) -> None:
        """Composite the outline over *color_tex* into *target_tex*."""
        # Whether the label-key target is bound changes the binding layout,
        # so it is part of the pipeline identity as well as a template var.
        self.set_template_vars(has_outline_id=outline_id_tex is not None)
        pipeline_hash = (
            (target_format := target_tex.texture.format),
            (outline_id_tex is not None),
        )
        if self._pipeline is None or self._pipeline_hash != pipeline_hash:
            self._pipeline = self._create_pipeline(
                target_format, with_outline_id=outline_id_tex is not None
            )
            self._pipeline_hash = pipeline_hash

        self._device.queue.write_buffer(
            self._buffer, 0, self._uniform_data, 0, self._uniform_data.nbytes
        )

        bind_group = self._device.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0),
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._buffer,
                        "offset": 0,
                        "size": self._buffer.size,
                    },
                },
                {"binding": 1, "resource": self._sampler},
                {"binding": 2, "resource": color_tex},
                {"binding": 3, "resource": pick_tex},
                {"binding": 4, "resource": lut_tex},
            ]
            + (
                [{"binding": 5, "resource": outline_id_tex}]
                if outline_id_tex is not None
                else []
            ),
        )

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=target_tex,
                    resolve_target=None,
                    clear_value=(0, 0, 0, 0),
                    load_op=wgpu.LoadOp.clear,
                    store_op="store",
                )
            ],
            depth_stencil_attachment=None,
        )
        render_pass.set_pipeline(self._pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(4, 1)
        render_pass.end()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_pipeline(self, target_format: str, *, with_outline_id: bool):
        binding_layout = [
            wgpu.BindGroupLayoutEntry(
                binding=0,
                visibility=wgpu.ShaderStage.FRAGMENT,
                buffer=wgpu.BufferBindingLayout(type="uniform"),
            ),
            wgpu.BindGroupLayoutEntry(
                binding=1,
                visibility=wgpu.ShaderStage.FRAGMENT,
                sampler=wgpu.SamplerBindingLayout(type="filtering"),
            ),
            wgpu.BindGroupLayoutEntry(
                binding=2,
                visibility=wgpu.ShaderStage.FRAGMENT,
                texture=wgpu.TextureBindingLayout(
                    sample_type=wgpu.TextureSampleType.float,
                    view_dimension="2d",
                    multisampled=False,
                ),
            ),
            wgpu.BindGroupLayoutEntry(
                binding=3,
                visibility=wgpu.ShaderStage.FRAGMENT,
                texture=wgpu.TextureBindingLayout(
                    sample_type=wgpu.TextureSampleType.uint,
                    view_dimension="2d",
                    multisampled=False,
                ),
            ),
            wgpu.BindGroupLayoutEntry(
                binding=4,
                visibility=wgpu.ShaderStage.FRAGMENT,
                texture=wgpu.TextureBindingLayout(
                    sample_type=wgpu.TextureSampleType.uint,
                    view_dimension="2d",
                    multisampled=False,
                ),
            ),
        ]
        if with_outline_id:
            binding_layout.append(
                wgpu.BindGroupLayoutEntry(
                    binding=5,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.uint,
                        view_dimension="2d",
                        multisampled=False,
                    ),
                )
            )
        definitions = (
            self._uniform_definition
            + """
            @group(0) @binding(1)
            var texSampler: sampler;
            @group(0) @binding(2)
            var colorTex: texture_2d<f32>;
            @group(0) @binding(3)
            var pickTex: texture_2d<u32>;
            @group(0) @binding(4)
            var lutTex: texture_2d<u32>;
        """
        )
        if with_outline_id:
            definitions += """
            @group(0) @binding(5)
            var outlineIdTex: texture_2d<u32>;
        """
        targets = [
            wgpu.ColorTargetState(
                format=target_format,
                blend={
                    "alpha": {
                        "operation": wgpu.BlendOperation.add,
                        "src_factor": wgpu.BlendFactor.one,
                        "dst_factor": wgpu.BlendFactor.zero,
                    },
                    "color": {
                        "operation": wgpu.BlendOperation.add,
                        "src_factor": wgpu.BlendFactor.one,
                        "dst_factor": wgpu.BlendFactor.zero,
                    },
                },
            )
        ]
        code = definitions + apply_templating(OUTLINE_WGSL, **self._template_vars)
        return create_full_quad_pipeline(targets, binding_layout, code)


class OutlinePass(EffectPass):
    """Composite screen-space outlines from the pick buffer.

    Insert **ahead of** ``TemporalAccumulationPass``: the volume raymarcher
    jitters per frame, so silhouette pixels shift sub-pixel between frames
    and compositing the outline before accumulation lets the EMA average
    that into a free antialiased edge rather than a flicker.

    The pass is inert whenever ``enabled`` is ``False`` -- pygfx's
    ``flush()`` skips disabled passes entirely -- so it costs nothing on a
    canvas that never turns outlines on.

    Parameters
    ----------
    renderer : gfx.WgpuRenderer
        The renderer whose pick buffer supplies the ids.  ``flush()`` hands
        an effect pass only colour, depth and target views, so the pick
        view is fetched from here each frame.
    lut : VisualLut or None
        The shared ``global_id`` -> entry table.  ``None`` (the default)
        resolves it on first draw, so a canvas that never enables outlines
        never allocates the 1 MB texture.
    """

    USES_DEPTH = False

    # Fallback shader, used only if the base machinery ever compiles a
    # pipeline for this object.  The real work happens in _OutlineQuadPass.
    wgsl = """
        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            return textureSample(colorTex, texSampler, varyings.texCoord);
        }
    """

    def __init__(
        self, renderer: gfx.WgpuRenderer, lut: VisualLut | None = None
    ) -> None:
        super().__init__()
        self._renderer = renderer
        self._lut = lut
        self._quad_pass = _OutlineQuadPass()
        self.enabled = False

    @property
    def lut(self) -> VisualLut:
        """The outline table, allocated on first access."""
        if self._lut is None:
            self._lut = get_shared_visual_lut()
        return self._lut

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def apply_config(self, config: OutlineConfig) -> None:
        """Push an ``OutlineConfig`` onto the pass.

        Thicknesses and the contrast band width are template vars, so
        changing them recompiles the shader.  Enables, colours and the
        palette are uniforms and do not.
        """
        quad = self._quad_pass
        quad.set_template_vars(
            b_t_in=int(config.boundaries.inward_thickness),
            b_t_out=int(config.boundaries.outward_thickness),
            s_t_in=int(config.selection.inward_thickness),
            s_t_out=int(config.selection.outward_thickness),
            inner_t=int(config.inner_thickness),
        )
        quad.set_uniform("boundaries_enabled", int(bool(config.boundaries.enabled)))
        quad.set_uniform("selection_enabled", int(bool(config.selection.enabled)))
        quad.set_uniform("boundary_color", tuple(config.boundaries.color))
        quad.set_uniform("inner_color", tuple(config.inner_color))
        quad.set_palette([tuple(c) for c in config.palette])
        self.enabled = bool(config.enabled)

    def set_placements(self, *, has_inward: bool, has_outward: bool) -> None:
        """Declare which placements are present in the current selection.

        These are template vars: a scene with no outward-placed visual
        compiles the outward branch away entirely, collapsing the shader
        back to four taps per band.  Recompiles only when the *set* of
        placements changes.
        """
        self._quad_pass.set_template_vars(
            has_inward=bool(has_inward), has_outward=bool(has_outward)
        )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        command_encoder: wgpu.GPUCommandEncoder,
        color_tex: wgpu.GPUTextureView,
        depth_tex: wgpu.GPUTextureView | None,
        target_tex: wgpu.GPUTextureView,
    ) -> None:
        """Composite outlines, or pass the frame through if pick is absent."""
        pick_tex = get_pick_view(self._renderer)
        if pick_tex is None:
            # No bindable pick target on this renderer: nothing to outline.
            super().render(command_encoder, color_tex, depth_tex, target_tex)
            return
        self._quad_pass.render(
            command_encoder,
            color_tex,
            pick_tex,
            self.lut.view,
            target_tex,
            get_extra_target_view(self._renderer, OUTLINE_ID_TARGET),
        )
