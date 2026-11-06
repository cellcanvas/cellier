"""The MaskedImage's mask material and shader (masked image design v2, 6.1).

One shader: the multiscale brick shader with its ``masked`` template variable
set.  The value image's lookup (``get_lod_scale``, the brick rule, the level
mapping and ``sample_atlas``) is the mask's own text, copied with every name
and binding renamed when this module is imported (:func:`instantiate_by_text`),
so there is one definition of the brick rule.  The copy is registered as the
``cellier_masked.value_access.wgsl`` include.

Importing this module registers :class:`MaskedVolumeBrickShader` for
``(Volume, MaskedVolumeBrickMaterial)``.  It subclasses the material's
registered shader, as the image volume shaders do.
"""

from __future__ import annotations

import re
from typing import ClassVar

import numpy as np
import pygfx as gfx
from pygfx.objects import Volume
from pygfx.renderers.wgpu import (
    Binding,
    GfxSampler,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shader.templating import (
    register_wgsl_loader,
    root_loader,
)

from cellier.render.shaders._multiscale_volume_brick import (
    _WGSL_PATH,
    _WGSL_SOURCE,
    MultiscaleVolumeBrickMaterial,
    MultiscaleVolumeBrickShader,
)

__all__ = [
    "N_LEVEL_BOUNDS",
    "MaskedVolumeBrickMaterial",
    "MaskedVolumeBrickShader",
    "instantiate_by_text",
    "value_access_source",
]

#: Levels the per-level bound uniforms cover, as ``u_block_scales`` does.
N_LEVEL_BOUNDS = 9
#: Function names the value side needs its own copy of.
ACCESSORS = (
    "get_lod_scale",
    "get_brick_span",
    "get_brick_count",
    "brick_corner_from_cell",
    "get_level_offset",
    "placed_corner_k",
    "sample_atlas",
)
#: Mask binding -> value binding.
RESOURCES = {
    "u_block_scales": "u_value_block_scales",
    "u_vol_params": "u_value_vol_params",
    "t_cache": "t_value_cache",
    "s_cache": "s_value_cache",
}
#: ``sample_atlas``'s masked read past a level's end (6.4), and the line it
#: stands for: the value side keeps the plain read (G2 clamps it already).
_EXTENT_CLAMP = re.compile(
    r"    \$\$ if masked and extent_clamp and render_mode != 'plane'\n.*?\n"
    r"    \$\$ else\n"
    r"(?P<plain>.*?\n)    \$\$ endif\n",
    re.DOTALL,
)


def _function_text(source: str, name: str) -> str:
    """The text of WGSL function *name*, from ``fn`` to its closing brace."""
    start = source.index(f"fn {name}(")
    end = source.index("\n}\n", start) + 3
    return source[start:end]


def value_access_source() -> str:
    """The mask's lookup, in dependency order, as the value side copies it.

    ``sample_atlas`` is taken with its masked branch resolved to the plain
    read: the copy is for the value, whose reads are clamped to its level's
    voxel centres (G2) before they reach it.
    """
    wgsl_dir = _WGSL_PATH.parent
    sample = _function_text(_WGSL_SOURCE, "sample_atlas")
    sample, count = _EXTENT_CLAMP.subn(lambda match: match["plain"], sample)
    if count != 1:
        raise ValueError("sample_atlas changed: re-derive the value side's copy")
    if "$$" in sample:
        raise ValueError("sample_atlas holds a template branch the copy cannot take")
    return "\n".join(
        [
            _function_text(_WGSL_SOURCE, "get_lod_scale"),
            (wgsl_dir / "brick_rule.wgsl").read_text(encoding="utf-8"),
            (wgsl_dir / "level_mapping.wgsl").read_text(encoding="utf-8"),
            sample,
        ]
    )


def instantiate_by_text(source: str) -> str:
    """*source* with every accessor and binding renamed for the value side.

    Raises
    ------
    ValueError
        If a name is not found, or a mask binding survives: the failure the
        copy must guard against is the value side silently reading the
        mask's atlas.
    """
    for name in ACCESSORS:
        source, count = re.subn(rf"\b{name}\b", f"value_{name}", source)
        if count == 0:
            raise ValueError(f"accessor {name!r} not found")
    for mask_name, value_name in RESOURCES.items():
        source, count = re.subn(rf"\b{mask_name}\b", value_name, source)
        if count == 0:
            raise ValueError(f"binding {mask_name!r} not found")
    code = re.sub(r"//.*", "", source)
    for mask_name in RESOURCES:
        if re.search(rf"\b{mask_name}\b", code):
            raise ValueError(f"binding {mask_name!r} survived")
    for name in ACCESSORS:
        if re.search(rf"(?<!value_)\b{name}\b", code):
            raise ValueError(f"accessor {name!r} survived")
    return source


_VALUE_ACCESS = instantiate_by_text(value_access_source())
if "cellier_masked" not in root_loader.mapping:
    register_wgsl_loader("cellier_masked", {"value_access.wgsl": _VALUE_ACCESS})


class MaskedVolumeBrickMaterial(MultiscaleVolumeBrickMaterial):
    """The mask's brick material, reading a value atlas beside its own.

    It draws as the plain material until :meth:`set_value_source` binds a
    value atlas; from then on the shader is compiled with ``masked``.

    The extra uniforms: ``mask_to_value`` (the mask node's normalized space
    to value level-0 voxels), each level's last voxel centre per side
    (``mask_hi_k``, ``value_hi_k``), the surface interval along the normal in
    world units (``surface_start``, ``surface_end``), the projection's step
    and cap, and the 2D band's fields.
    """

    uniform_type: ClassVar[dict] = dict(
        MultiscaleVolumeBrickMaterial.uniform_type,
        mask_to_value="4x4xf4",
        mask_snap="4xf4",
        value_snap="4xf4",
        **{f"mask_hi_{k}": "4xf4" for k in range(1, N_LEVEL_BOUNDS + 1)},
        **{f"value_hi_{k}": "4xf4" for k in range(1, N_LEVEL_BOUNDS + 1)},
        surface_start="f4",
        surface_end="f4",
        surface_min_width_px="f4",
        mask_edge_fade="f4",
        projection_steps_per_voxel="f4",
        projection_cap="f4",
        band_rays="f4",
        band_cap="f4",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._store.masked = False
        self._store.surface_op = "max"
        self._store.value_source = None
        self._store.extent_clamp = True
        # Bound in every masked pipeline; the mask side replaces it with its
        # atlas's table (_cell_ranges.py), an empty one never skips.
        self._store.mask_ranges = gfx.Texture(np.zeros((1, 1, 1, 4), np.float32), dim=3)
        data = self.uniform_buffer.data
        data["mask_to_value"] = np.eye(4, dtype=np.float32)
        data["projection_steps_per_voxel"] = 1.0
        data["projection_cap"] = 64.0
        data["surface_min_width_px"] = 1.0
        data["band_rays"] = 16.0
        data["band_cap"] = 32.0
        self.uniform_buffer.update_full()

    @property
    def masked(self) -> bool:
        """Whether a value atlas is bound (the ``masked`` template variable)."""
        return self._store.masked

    @property
    def value_source(self) -> tuple | None:
        """``(atlas, LUT, vol params, block scales)`` of the value side."""
        return self._store.value_source

    def set_value_source(self, cache_tex, lut_tex, vol_params, block_scales) -> None:
        """Bind the value side's atlas, LUT and buffers.

        Held in ``_store``, so a new atlas re-runs ``get_bindings``.
        """
        source = (cache_tex, lut_tex, vol_params, block_scales)
        current = self._store.value_source
        if current is None or any(
            a is not b for a, b in zip(source, current, strict=True)
        ):
            self._store.value_source = source
        self._store.masked = True

    @property
    def mask_ranges(self) -> gfx.Texture:
        """The per-cell range table of the mask's atlas (design 7.4)."""
        return self._store.mask_ranges

    def set_mask_ranges(self, texture: gfx.Texture) -> None:
        """Bind the mask atlas's range table (``CellRanges.texture``)."""
        if texture is not self._store.mask_ranges:
            self._store.mask_ranges = texture

    @property
    def extent_clamp(self) -> bool:
        """Whether the mask's 3D read repeats a level's last voxel (6.4, MI-P16).

        Always on in use; a switch for the tests' control only.
        """
        return self._store.extent_clamp

    @extent_clamp.setter
    def extent_clamp(self, value: bool) -> None:
        self._store.extent_clamp = bool(value)

    @property
    def surface_op(self) -> str:
        """``"max"``, ``"mean"`` or ``"min"`` along the normal (a template variable)."""
        return self._store.surface_op

    @surface_op.setter
    def surface_op(self, value: str) -> None:
        if value not in ("max", "mean", "min"):
            raise ValueError(f"surface_op must be max, mean or min, got {value!r}")
        if self._store.surface_op != value:
            self._store.surface_op = value

    def set_mask_to_value(self, matrix: np.ndarray) -> None:
        """The mask node's normalized space to value level-0 voxels, shader order."""
        # pygfx uniforms hold matrices column first.
        matrix = np.asarray(matrix, dtype=np.float32).T
        if not np.array_equal(self.uniform_buffer.data["mask_to_value"], matrix):
            self.uniform_buffer.data["mask_to_value"] = matrix
            self.uniform_buffer.update_full()

    def set_level_bounds(self, side: str, level_shapes) -> None:
        """Write each level's last voxel centre for *side* (``"mask"``, ``"value"``).

        Parameters
        ----------
        side : str
            Which side's uniforms.
        level_shapes : sequence of tuple[int, int, int]
            Each level's spatial shape in data order ``(z, y, x)``, finest
            first.  Levels past the last repeat it.
        """
        data = self.uniform_buffer.data
        changed = False
        for k in range(1, N_LEVEL_BOUNDS + 1):
            shape = level_shapes[min(k, len(level_shapes)) - 1]
            bound = np.array(
                [shape[2] - 1, shape[1] - 1, shape[0] - 1, 0.0], dtype=np.float32
            )
            name = f"{side}_hi_{k}"
            if not np.array_equal(data[name], bound):
                data[name] = bound
                changed = True
        if changed:
            self.uniform_buffer.update_full()

    def set_surface(
        self,
        start: float,
        end: float,
        steps_per_voxel: float = 1.0,
        cap: int = 64,
    ) -> None:
        """The placed interval along ``inward``, world units, and its sampling."""
        self._set_floats(
            surface_start=start,
            surface_end=end,
            projection_steps_per_voxel=steps_per_voxel,
            projection_cap=cap,
        )

    def set_band(self, min_width_px: float, rays: int, cap: int) -> None:
        """The 2D band's minimum width and its contour search (MI-P15)."""
        self._set_floats(
            surface_min_width_px=min_width_px, band_rays=rays, band_cap=cap
        )

    def set_read(self, mask_axis: int | None, value_axis: int | None) -> None:
        """The sliced axis of each side in a 2D view, shader order, or ``None``.

        ``None`` is a tilted plane, read at its true depth; an axis is
        snapped as a 2D tile reads it (MI-D14).  The mask then also takes
        the tile's fade at the data's edge.
        """
        data = self.uniform_buffer.data
        for name, axis in (("mask_snap", mask_axis), ("value_snap", value_axis)):
            snap = np.zeros(4, dtype=np.float32)
            if axis is not None:
                snap[axis] = 1.0
            data[name] = snap
        data["mask_edge_fade"] = 0.0 if mask_axis is None else 1.0
        self.uniform_buffer.update_full()

    def _set_floats(self, **values: float) -> None:
        data = self.uniform_buffer.data
        changed = False
        for name, value in values.items():
            if float(data[name]) != float(value):
                data[name] = float(value)
                changed = True
        if changed:
            self.uniform_buffer.update_full()


@register_wgpu_render_function(Volume, MaskedVolumeBrickMaterial)
class MaskedVolumeBrickShader(MultiscaleVolumeBrickShader):
    """The brick shader with the value bindings, once the material has them."""

    def get_bindings(self, wobject, shared, scene):
        material = wobject.material
        result = super().get_bindings(wobject, shared, scene)
        if not material.masked:
            return result
        self["masked"] = True
        self["extent_clamp"] = bool(material.extent_clamp)
        self["surface_op"] = str(material.surface_op)
        cache, lut, vol_params, block_scales = material.value_source
        bindings = result[0]
        extra = [
            Binding(
                "s_value_cache",
                "sampler/filtering",
                GfxSampler(material.interpolation, "clamp"),
                "FRAGMENT",
            ),
            Binding("t_value_cache", "texture/auto", GfxTextureView(cache), "FRAGMENT"),
            Binding("t_value_lut", "texture/auto", GfxTextureView(lut), "FRAGMENT"),
            # The mask's dtypes: pygfx keeps one struct per dtype and asserts
            # its name.
            Binding(
                "u_value_vol_params",
                "buffer/uniform",
                vol_params,
                "FRAGMENT",
                structname="VolParams",
            ),
            Binding(
                "u_value_block_scales",
                "buffer/uniform",
                block_scales,
                "FRAGMENT",
                structname="BlockScales",
            ),
            Binding(
                "t_mask_range",
                "texture/auto",
                GfxTextureView(material.mask_ranges),
                "FRAGMENT",
            ),
        ]
        start = len(bindings)
        for offset, binding in enumerate(extra):
            bindings[start + offset] = binding
        self.define_bindings(0, bindings)
        return {0: bindings}
