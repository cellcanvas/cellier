"""Screen-space ambient occlusion effect pass.

One ``EffectPass`` driving three inner full-quad passes over two private
single-channel textures::

    _AOComputePass    depthTex (+ normalTex, pickTex, lutTex)  -> _ao_tex
    _AOBlurPass       _ao_tex                                  -> _ao_blur_tex
    _AOCompositePass  colorTex + _ao_blur_tex                  -> targetTex

Structurally this follows
:class:`~cellier.render._temporal_accumulation.TemporalAccumulationPass`
-- private textures, lazily reallocated on size change -- rather than
:class:`~cellier.render._outline.OutlinePass`, which owns none.  The
private textures exist because the occlusion field has to be blurred
before it is applied, and a blur cannot read the target it writes.

Unlike the pick buffer, the depth target ships with ``TEXTURE_BINDING``
already granted, and ``flush()`` hands it to any pass declaring
``USES_DEPTH = True``.  So this pass needs no usage grant and no blender
subclass; the ``normal`` target it *prefers* for volumes does, and that is
:mod:`cellier.render._cellier_blender`'s job.

``r8unorm`` for the intermediates gives 256 levels of occlusion -- ample
for a quantity that is blurred and then multiplied -- at one byte per
pixel each, and is both filterable and renderable so no format gymnastics
are needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import wgpu
from pygfx.renderers.wgpu.engine.binding import Binding
from pygfx.renderers.wgpu.engine.effectpasses import (
    EffectPass,
    FullQuadPass,
    create_full_quad_pipeline,
)
from pygfx.renderers.wgpu.engine.shared import get_shared
from pygfx.renderers.wgpu.shader.bindings import BindingDefinitions
from pygfx.renderers.wgpu.shader.templating import apply_templating
from pygfx.utils import array_from_shadertype

from cellier.render._cellier_blender import NORMAL_TARGET, get_extra_target_view
from cellier.render._pick_buffer import get_pick_view
from cellier.render._visual_lut import get_shared_visual_lut

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    import pygfx as gfx

    from cellier.render._config import SSAOConfig

_WGSL_DIR = Path(__file__).parent / "shaders" / "wgsl"
SSAO_WGSL: str = (_WGSL_DIR / "ssao.wgsl").read_text()

#: Entries the kernel uniform always carries, whatever ``n_samples`` is.
#: Fixing the array size keeps a sample-count change from altering the
#: uniform layout, the same reason ``_outline.py`` sizes its palette at
#: ``MAX_SLOT``.  64 entries is 1 KB against a 64 KB uniform limit.
MAX_KERNEL_SAMPLES: int = 64


def make_kernel(n_samples: int, seed: int = 0) -> np.ndarray:
    """Build the tangent-space hemisphere kernel.

    Points are drawn in the ``z >= 0`` hemisphere, normalised, then pulled
    toward the origin by an accelerating ramp,
    ``scale = lerp(0.1, 1.0, s * s)``, taken from both reference articles.
    That is a weighting decision rather than an optimisation: close-range
    occlusion carries most of the perceptual signal, so the sample budget
    is spent near the fragment.

    Parameters
    ----------
    n_samples : int
        How many entries actually take part in the ramp.  The returned
        array is always ``MAX_KERNEL_SAMPLES`` long; the remainder is
        zero, and the shader never reads past ``n_samples``.
    seed : int
        Seed for the deterministic generator, so a frame is reproducible.

    Returns
    -------
    numpy.ndarray
        ``(MAX_KERNEL_SAMPLES, 4)`` float32, ``w`` unused.

    Notes
    -----
    The ramp runs over *n_samples*, not over ``MAX_KERNEL_SAMPLES``.
    Spreading it over the fixed array instead would leave a 16-sample
    kernel with every sample inside the innermost quarter of the
    hemisphere, which reads as almost no occlusion at all.
    """
    n_samples = max(1, int(n_samples))
    rng = np.random.default_rng(seed)
    kernel = np.zeros((MAX_KERNEL_SAMPLES, 4), dtype=np.float32)
    for index in range(min(n_samples, MAX_KERNEL_SAMPLES)):
        vector = np.array(
            [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(0.0, 1.0),
            ]
        )
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            vector = np.array([0.0, 0.0, 1.0])
            norm = 1.0
        vector /= norm
        # Distribute within the hemisphere rather than only on its shell.
        vector *= rng.uniform(0.0, 1.0)
        ramp = index / n_samples
        vector *= 0.1 + 0.9 * ramp * ramp
        kernel[index, :3] = vector
    return kernel


class _AOComputePass:
    """Inner pass computing the raw occlusion field from depth.

    Deliberately **not** a ``FullQuadPass`` subclass, for the same reason
    ``_OutlineQuadPass`` is not: that class builds its binding layout from
    a naming convention knowing only ``float`` and ``depth`` sample types,
    so the ``rgba16uint`` pick texture and the ``r8uint`` visual LUT cannot
    be declared through it.  The layout is built here and handed to the
    module-level ``create_full_quad_pipeline``, which is the documented
    low-level entry point ``FullQuadPass`` itself is built on -- so the
    vertex shader, the ``Varyings`` struct and the pipeline cache are all
    shared with every other effect pass.

    Binding indices are fixed rather than packed, so ``normalTex`` keeps
    index 3 whether or not ``pickTex`` is bound.  WebGPU allows gaps in a
    bind group as long as the layout agrees, and a stable numbering is
    much easier to read against the WGSL.
    """

    uniform_type: ClassVar[dict] = {
        "projection_transform": "4x4xf4",
        "projection_transform_inv": "4x4xf4",
        "kernel": f"{MAX_KERNEL_SAMPLES}*4xf4",
        "radius": "f4",
        "bias": "f4",
        "width": "i4",
        "height": "i4",
        "frame_index": "i4",
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
            0, 0, Binding("u_effect", "buffer/uniform", self._uniform_data.dtype)
        )
        self._uniform_definition = definitions.get_code()
        # The shader only ever uses textureLoad, but the layout keeps a
        # sampler at binding 1 so the indices line up with every other
        # pygfx effect pass.
        self._sampler = self._device.create_sampler(
            min_filter="linear", mag_filter="linear"
        )

        self._template_vars: dict[str, object] = {
            "stage": "compute",
            "n_samples": 16,
            "has_normal_target": False,
            "has_exclusions": False,
        }
        self._pipeline = None
        self._pipeline_hash: object = None

    def set_template_var(self, **kwargs: object) -> None:
        """Update shader template vars, forcing a recompile if they changed."""
        for name, value in kwargs.items():
            if self._template_vars.get(name) != value:
                self._template_vars[name] = value
                self._pipeline = None

    def render(
        self,
        command_encoder: wgpu.GPUCommandEncoder,
        depth_tex: wgpu.GPUTextureView,
        target_tex: wgpu.GPUTextureView,
        normal_tex: wgpu.GPUTextureView | None = None,
        pick_tex: wgpu.GPUTextureView | None = None,
        lut_tex: wgpu.GPUTextureView | None = None,
    ) -> None:
        """Write the raw occlusion field into *target_tex*."""
        with_normal = normal_tex is not None
        with_exclusions = pick_tex is not None and lut_tex is not None
        self.set_template_var(
            has_normal_target=with_normal, has_exclusions=with_exclusions
        )
        pipeline_hash = (
            (target_format := target_tex.texture.format),
            with_normal,
            with_exclusions,
        )
        if self._pipeline is None or self._pipeline_hash != pipeline_hash:
            self._pipeline = self._create_pipeline(
                target_format,
                with_normal=with_normal,
                with_exclusions=with_exclusions,
            )
            self._pipeline_hash = pipeline_hash

        self._device.queue.write_buffer(
            self._buffer, 0, self._uniform_data, 0, self._uniform_data.nbytes
        )

        entries = [
            {
                "binding": 0,
                "resource": {
                    "buffer": self._buffer,
                    "offset": 0,
                    "size": self._buffer.size,
                },
            },
            {"binding": 1, "resource": self._sampler},
            {"binding": 2, "resource": depth_tex},
        ]
        if with_normal:
            entries.append({"binding": 3, "resource": normal_tex})
        if with_exclusions:
            entries.append({"binding": 4, "resource": pick_tex})
            entries.append({"binding": 5, "resource": lut_tex})

        bind_group = self._device.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0), entries=entries
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

    def _create_pipeline(
        self, target_format: str, *, with_normal: bool, with_exclusions: bool
    ):
        def _texture_entry(binding: int, sample_type: str):
            return wgpu.BindGroupLayoutEntry(
                binding=binding,
                visibility=wgpu.ShaderStage.FRAGMENT,
                texture=wgpu.TextureBindingLayout(
                    sample_type=sample_type,
                    view_dimension="2d",
                    multisampled=False,
                ),
            )

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
            _texture_entry(2, wgpu.TextureSampleType.depth),
        ]
        definitions = (
            self._uniform_definition
            + """
            @group(0) @binding(1)
            var texSampler: sampler;
            @group(0) @binding(2)
            var depthTex: texture_depth_2d;
        """
        )
        if with_normal:
            binding_layout.append(_texture_entry(3, wgpu.TextureSampleType.float))
            definitions += """
            @group(0) @binding(3)
            var normalTex: texture_2d<f32>;
        """
        if with_exclusions:
            binding_layout.append(_texture_entry(4, wgpu.TextureSampleType.uint))
            binding_layout.append(_texture_entry(5, wgpu.TextureSampleType.uint))
            definitions += """
            @group(0) @binding(4)
            var pickTex: texture_2d<u32>;
            @group(0) @binding(5)
            var lutTex: texture_2d<u32>;
        """
        targets = [wgpu.ColorTargetState(format=target_format, blend=None)]
        code = definitions + apply_templating(SSAO_WGSL, **self._template_vars)
        return create_full_quad_pipeline(targets, binding_layout, code)


class _AOBlurPass(FullQuadPass):
    """Inner pass box-blurring the occlusion field."""

    # The blur is fully described by its template var, but a uniform
    # buffer of size zero is not a thing, so one unused field stands in.
    uniform_type: ClassVar[dict] = {"unused": "f4"}

    wgsl = SSAO_WGSL

    def __init__(self) -> None:
        super().__init__()
        self._set_template_var(stage="blur", blur_radius=2)


class _AOCompositePass(FullQuadPass):
    """Inner pass multiplying the occlusion into the colour image."""

    uniform_type: ClassVar[dict] = {"strength": "f4", "power": "f4"}

    wgsl = SSAO_WGSL

    def __init__(self) -> None:
        super().__init__()
        self._set_template_var(stage="composite")


class SSAOPass(EffectPass):
    """Darken creases by sampling occlusion out of the depth buffer.

    Insert **ahead of** ``OutlinePass`` and ``TemporalAccumulationPass``.
    Ahead of the outline because an outline is a UI annotation rather than
    a lit surface and its palette colours must stay exact; ahead of
    accumulation because the EMA there denoises the per-frame kernel
    rotation for free, which is what lets the sample count sit at 16
    instead of 64.

    The pass is inert whenever ``enabled`` is ``False`` -- pygfx's
    ``flush()`` skips disabled passes entirely -- so it costs nothing on a
    canvas that never turns it on.

    Parameters
    ----------
    renderer : gfx.WgpuRenderer
        The renderer whose depth and ``normal`` targets supply the
        geometry.  ``flush()`` hands an effect pass only colour, depth and
        target views, so anything else is fetched from here each frame.
    get_camera : Callable[[], gfx.Camera]
        Returns the camera the frame was rendered with.  A callable rather
        than a reference because a canvas swaps cameras on a 2D/3D toggle.
        ``render()`` runs during ``flush()``, after the camera is fully
        evaluated, so the pass can update its own uniforms with no
        cooperation from the draw loop.
    """

    USES_DEPTH = True

    # Fallback shader, used only if the base machinery ever compiles a
    # pipeline for this object.  The real work happens in the inner passes.
    wgsl = """
        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            return textureSample(colorTex, texSampler, varyings.texCoord);
        }
    """

    def __init__(
        self,
        renderer: gfx.WgpuRenderer,
        get_camera: Callable[[], gfx.Camera],
    ) -> None:
        super().__init__()
        self._renderer = renderer
        self._get_camera = get_camera

        self._compute_pass = _AOComputePass()
        self._blur_pass = _AOBlurPass()
        self._composite_pass = _AOCompositePass()

        self._ao_texture: wgpu.GPUTexture | None = None
        self._ao_view: wgpu.GPUTextureView | None = None
        self._ao_blur_texture: wgpu.GPUTexture | None = None
        self._ao_blur_view: wgpu.GPUTextureView | None = None
        self._current_size: tuple[int, int] | None = None

        self._n_samples: int = 16
        self._blur_radius: int = 2
        self._explicit_radius: float | None = None
        self._auto_radius: float = 1.0
        self._auto_radius_fraction: float = 0.02
        self._bias_fraction: float = 0.05
        self._frame_index: int = 0
        self._has_exclusions: bool = False

        self._compute_pass._uniform_data["kernel"] = make_kernel(self._n_samples)
        self._push_radius()
        self._composite_pass._uniform_data["strength"] = 1.0
        self._composite_pass._uniform_data["power"] = 1.0

        self.enabled = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def apply_config(self, config: SSAOConfig) -> None:
        """Push an ``SSAOConfig`` onto the pass.

        ``n_samples`` and ``blur_radius`` are template vars, so changing
        them recompiles the inner shaders.  Everything else is a uniform
        and does not.
        """
        self.n_samples = int(config.n_samples)
        self.blur_radius = int(config.blur_radius)
        self.radius = config.radius
        self.auto_radius_fraction = float(config.auto_radius_fraction)
        self.bias = float(config.bias)
        self.strength = float(config.strength)
        self.power = float(config.power)
        self.enabled = bool(config.enabled)

    @property
    def n_samples(self) -> int:
        """Hemisphere samples per pixel.  A template var: recompiles."""
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value: int) -> None:
        value = max(1, min(int(value), MAX_KERNEL_SAMPLES))
        if value == self._n_samples:
            return
        self._n_samples = value
        # The ramp is spread over the used samples, so the kernel has to
        # be rebuilt -- but its uniform layout is fixed, so this is a
        # buffer write, not a pipeline change.
        self._compute_pass._uniform_data["kernel"] = make_kernel(value)
        self._compute_pass.set_template_var(n_samples=value)

    @property
    def blur_radius(self) -> int:
        """Box-blur half-width in internal pixels.  A template var."""
        return self._blur_radius

    @blur_radius.setter
    def blur_radius(self, value: int) -> None:
        value = max(0, int(value))
        if value == self._blur_radius:
            return
        self._blur_radius = value
        self._blur_pass._set_template_var(blur_radius=value)

    @property
    def radius(self) -> float | None:
        """Explicit hemisphere radius in scene units, or ``None`` for auto."""
        return self._explicit_radius

    @radius.setter
    def radius(self, value: float | None) -> None:
        self._explicit_radius = None if value is None else float(value)
        self._push_radius()

    @property
    def auto_radius_fraction(self) -> float:
        """Fraction of the scene bbox diagonal used by the auto radius."""
        return self._auto_radius_fraction

    @auto_radius_fraction.setter
    def auto_radius_fraction(self, value: float) -> None:
        self._auto_radius_fraction = float(value)
        self._push_radius()

    @property
    def effective_radius(self) -> float:
        """The radius actually in use, explicit or auto-derived."""
        if self._explicit_radius is not None:
            return self._explicit_radius
        return self._auto_radius

    def set_scene_extent(self, diagonal: float) -> None:
        """Record the scene bounding box diagonal for the auto radius.

        Called when the answer can change -- a camera fit, or an
        ``AABBChangedEvent`` -- rather than per frame, because
        ``scene.get_world_bounding_box()`` walks the whole scene graph and
        a multiscale visual is a ``gfx.Group`` with many brick children.

        Parameters
        ----------
        diagonal : float
            Length of the scene bounding box diagonal, in scene units.
            Non-finite or non-positive values are ignored, so a degenerate
            box (an empty scene, or one fitted before the first reslice
            completed) leaves the previous radius in place.
        """
        diagonal = float(diagonal)
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            return
        self._auto_radius = diagonal * self._auto_radius_fraction
        self._push_radius()

    @property
    def bias(self) -> float:
        """Depth-comparison bias, as a fraction of the effective radius.

        Dimensionless for the same reason the radius is auto-derived: the
        radius spans cellier's coordinate systems, and an absolute bias
        that suits one of them suits none of the others.  Measured on a
        box resting on a plane at radius 6, an absolute 0.025 leaves the
        flat plane self-occluding by 7 percent; the same number read as a
        fraction leaves it at 0.4 percent while the contact crease still
        reaches 0.59.
        """
        return self._bias_fraction

    @bias.setter
    def bias(self, value: float) -> None:
        self._bias_fraction = float(value)
        self._push_radius()

    @property
    def strength(self) -> float:
        """How far the multiply is applied, 0 (off) to 1 (full)."""
        return float(self._composite_pass._uniform_data["strength"])

    @strength.setter
    def strength(self, value: float) -> None:
        self._composite_pass._uniform_data["strength"] = float(value)

    @property
    def power(self) -> float:
        """Contrast exponent applied to the occlusion before the multiply."""
        return float(self._composite_pass._uniform_data["power"])

    @power.setter
    def power(self, value: float) -> None:
        self._composite_pass._uniform_data["power"] = float(value)

    def set_has_exclusions(self, value: bool) -> None:
        """Declare whether any visual is excluded from receiving occlusion.

        Binds the pick buffer and the shared visual LUT while true.  A
        scene with nothing excluded pays nothing: the lookup is compiled
        away and neither texture is bound.

        Parameters
        ----------
        value : bool
            Whether at least one visual carries the exclusion flag.
        """
        self._has_exclusions = bool(value)

    def _push_radius(self) -> None:
        radius = self.effective_radius
        self._compute_pass._uniform_data["radius"] = radius
        self._compute_pass._uniform_data["bias"] = radius * self._bias_fraction

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
        """Compute, blur and apply the occlusion; pass through without depth."""
        if depth_tex is None:
            # No bindable depth target on this renderer: nothing to occlude.
            super().render(command_encoder, color_tex, depth_tex, target_tex)
            return

        width, height = color_tex.texture.size[:2]
        if (width, height) != self._current_size:
            self._reallocate(width, height)
            self._current_size = (width, height)

        camera = self._get_camera()
        data = self._compute_pass._uniform_data
        # pygfx stores 4x4 uniforms transposed (renderer.py writes
        # camera.projection_matrix.T into stdinfo).  Dropping the .T here
        # yields an occlusion field that is plausible-looking and entirely
        # wrong.
        data["projection_transform"] = np.asarray(
            camera.projection_matrix, dtype=np.float32
        ).T
        data["projection_transform_inv"] = np.asarray(
            camera.projection_matrix_inverse, dtype=np.float32
        ).T
        data["width"] = width
        data["height"] = height
        data["frame_index"] = self._frame_index
        self._frame_index = (self._frame_index + 1) % 4096

        self._render_compute(command_encoder, depth_tex)
        self._blur_pass.render(
            command_encoder, aoTex=self._ao_view, targetTex=self._ao_blur_view
        )
        self._composite_pass.render(
            command_encoder,
            colorTex=color_tex,
            aoTex=self._ao_blur_view,
            targetTex=target_tex,
        )

    def _render_compute(
        self,
        command_encoder: wgpu.GPUCommandEncoder,
        depth_tex: wgpu.GPUTextureView,
    ) -> None:
        """Run the occlusion compute pass with whatever targets exist.

        Two optional inputs are resolved per frame rather than cached: the
        ``normal`` target, present only on a canvas that opted in at
        construction, and the pick buffer plus shared visual LUT, bound
        only while something is actually excluded.  Both are part of the
        pipeline identity, so a change recompiles rather than binding a
        layout that no longer matches, and a scene with no exclusions
        compiles the lookup away entirely.
        """
        normal_tex = get_extra_target_view(self._renderer, NORMAL_TARGET)
        pick_tex: wgpu.GPUTextureView | None = None
        lut_tex: wgpu.GPUTextureView | None = None
        if self._has_exclusions:
            pick_tex = get_pick_view(self._renderer)
            if pick_tex is not None:
                lut_tex = get_shared_visual_lut().view
        self._compute_pass.render(
            command_encoder,
            depth_tex=depth_tex,
            target_tex=self._ao_view,
            normal_tex=normal_tex,
            pick_tex=pick_tex,
            lut_tex=lut_tex,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reallocate(self, width: int, height: int) -> None:
        """Create (or recreate) the raw and blurred occlusion textures."""
        device = get_shared().device
        usage = (
            wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_SRC
        )
        self._ao_texture = device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.r8unorm,
            usage=usage,
            dimension="2d",
        )
        self._ao_view = self._ao_texture.create_view()
        self._ao_blur_texture = device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.r8unorm,
            usage=usage,
            dimension="2d",
        )
        self._ao_blur_view = self._ao_blur_texture.create_view()
