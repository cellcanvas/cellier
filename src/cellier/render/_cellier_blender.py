"""A pygfx ``Blender`` carrying cellier's extra render targets.

Two cellier features need per-pixel data the stock blender has nowhere to
put, and both are solved the same way: an extra colour attachment written
by cellier's own shaders in the same pass as colour and pick.

``outline_id`` (``r32uint``)
    A per-pixel *label* key for the screen-space outline pass.  The pick
    buffer has no room -- 64 bits, with 20 spent on ``global_id`` and 42 on
    the surface coordinate -- and buying space by cutting the pick
    coordinate's precision would cost 3D picking accuracy and force a
    hashed, collision-prone key.  32 bits is enough for the key to be exact.

``normal`` (``rgba16float``)
    A per-pixel **view-space** surface normal for the ambient occlusion
    pass.  Reconstructing a normal from depth works on meshes (0.66 deg
    median error) and fails on raymarched isosurfaces (34 deg, worsening
    as the user zooms in), because a trilinearly interpolated isosurface
    carries about a quarter voxel of fixed world-space roughness.  The
    gradient cellier's volume shaders already compute for Phong measures
    1.31 deg, so they write it here and the occlusion pass prefers it
    wherever it is present.

**No upstream pygfx changes are needed.**  ``custom_targets`` is stubbed
at ``blender.py:108``, but the three methods the pipeline actually asks the
blender for are public, and a subclass assigned to ``renderer._blender`` is
picked up end to end: ``renderer.py`` passes the blender into
``get_renderstate``, and ``pipeline.py`` calls ``get_shader_kwargs`` and
``get_color_descriptors`` on that instance.

Four things make this safe:

* ``Blender.hash`` derives from ``_texture_info``, so adding a target
  changes the hash and pipelines are never reused across blenders with
  mismatched targets.  ``BlendRenderState`` keys on that hash.
* WGSL zero-initialises function-scope ``var``, and every pygfx shader
  builds its output with ``var out: FragmentOutput;`` (none uses a struct
  literal), so shaders that never assign a new field emit 0 -- which is
  exactly "no outline key", and exactly "no written normal, reconstruct
  one instead".
* ``ensure_target_size`` and ``clear`` walk ``_texture_info`` generically,
  so resize and per-frame clearing need no extra handling.
* Appending the targets last, in one fixed order, in all three methods
  keeps target-state order, attachment order and ``@location(N)`` aligned.

The one brittle part is ``get_shader_kwargs``, which returns the
``FragmentOutput`` struct as a **WGSL source string** branched four ways by
alpha method.  A pygfx rewording surfaces as a shader compile error inside
the draw callback, which ``rendercanvas`` swallows without logging.
``tests/render/test_cellier_blender.py`` renders a frame and reads the
targets back so that becomes a CI failure instead of a black canvas.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

import wgpu
from pygfx.renderers.wgpu.engine.blender import Blender

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    import pygfx as gfx

#: Name of the per-pixel label key target.
OUTLINE_ID_TARGET: str = "outline_id"

#: Name of the per-pixel view-space normal target.
NORMAL_TARGET: str = "normal"


class ExtraTarget(NamedTuple):
    """One optional render target cellier can append to the blender.

    Parameters
    ----------
    name : str
        Target name, used as the ``texture_info`` key, as the generated
        ``FragmentOutput`` field name, and by ``get_extra_target_view``.
    format : str
        wgpu texture format.
    wgsl_type : str
        Type of the generated struct field.
    template_var : str
        Shader template var set ``True`` when this target is present, so a
        cellier shader compiles its write away on a canvas using the stock
        blender.
    gated_on_pick : bool
        When ``True`` the field and its write mask follow
        ``material_pick_write``, the way pick itself does -- appropriate
        for a target that only exists to identify things.  When ``False``
        the target is always written, which is what a geometric quantity
        like a normal needs: a visual with picking turned off still has a
        surface.
    """

    name: str
    format: str
    wgsl_type: str
    template_var: str
    gated_on_pick: bool


#: The targets cellier knows how to append, in the order they are appended.
#: The order is load-bearing: target-state order, attachment order and
#: ``@location(N)`` all derive from it and must agree.
EXTRA_TARGETS: dict[str, ExtraTarget] = {
    OUTLINE_ID_TARGET: ExtraTarget(
        name=OUTLINE_ID_TARGET,
        format=wgpu.TextureFormat.r32uint,
        wgsl_type="u32",
        template_var="write_outline_id",
        gated_on_pick=True,
    ),
    NORMAL_TARGET: ExtraTarget(
        name=NORMAL_TARGET,
        format=wgpu.TextureFormat.rgba16float,
        wgsl_type="vec4<f32>",
        template_var="write_normal",
        gated_on_pick=False,
    ),
}

_STRUCT_RE = re.compile(r"struct\s+FragmentOutput\s*\{(.*?)\n(\s*)\};", re.DOTALL)
_LOCATION_RE = re.compile(r"@location\((\d+)\)")


class CellierBlender(Blender):
    """``Blender`` with cellier's optional extra render targets appended.

    Behaves exactly like the stock blender for every object that does not
    write the new fields; the colour output of a scene is unchanged.

    Parameters
    ----------
    extra_targets : Iterable[str] or None
        Names from :data:`EXTRA_TARGETS`.  ``None`` means all of them,
        which is only useful in tests -- a real canvas asks for the
        targets its enabled features need and pays for nothing else.
    **kwargs
        Forwarded to ``Blender`` (``enable_pick``, ``enable_depth``).

    Raises
    ------
    ValueError
        If a name is not a known extra target.
    """

    def __init__(self, extra_targets: Iterable[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if extra_targets is None:
            names = list(EXTRA_TARGETS)
        else:
            names = [name for name in EXTRA_TARGETS if name in set(extra_targets)]
            unknown = set(extra_targets) - set(EXTRA_TARGETS)
            if unknown:
                raise ValueError(
                    f"unknown extra render target(s): {sorted(unknown)}; "
                    f"known targets are {sorted(EXTRA_TARGETS)}"
                )
        self._extra_targets: tuple[ExtraTarget, ...] = tuple(
            EXTRA_TARGETS[name] for name in names
        )
        # texture_info is a public property returning the live dict.
        for target in self._extra_targets:
            self.texture_info[target.name] = {
                "name": target.name,
                "format": target.format,
                "usage": (
                    wgpu.TextureUsage.RENDER_ATTACHMENT
                    | wgpu.TextureUsage.TEXTURE_BINDING
                    | wgpu.TextureUsage.COPY_SRC
                ),
                "is_used": False,
                "clear": True,
            }

    @property
    def extra_targets(self) -> tuple[ExtraTarget, ...]:
        """The extra targets this blender carries, in append order."""
        return self._extra_targets

    # ------------------------------------------------------------------
    # Blender overrides -- each is super() plus the appended entries
    # ------------------------------------------------------------------

    def get_color_descriptors(self, material_pick_write, alpha_config):
        """Append the extra target states to the pipeline targets.

        Follows the same convention pygfx uses for pick: the target state
        is always declared so all pipelines match, and whether an object
        actually writes it is decided by the write mask.
        """
        target_states = super().get_color_descriptors(material_pick_write, alpha_config)
        for target in self._extra_targets:
            texinfo = self.texture_info[target.name]
            texinfo["is_used"] = True
            enabled = bool(material_pick_write) or not target.gated_on_pick
            target_states.append(
                {
                    "format": texinfo["format"],
                    "blend": None,
                    "write_mask": wgpu.ColorWrite.ALL if enabled else 0,
                }
            )
        return target_states

    def get_color_attachments(self, pass_type):
        """Append the extra attachments, honouring each one's clear flag."""
        attachments = super().get_color_attachments(pass_type)
        for target in self._extra_targets:
            texinfo = self.texture_info[target.name]

            load_op = wgpu.LoadOp.load
            if texinfo["clear"]:
                texinfo["clear"] = False
                load_op = wgpu.LoadOp.clear

            attachments.append(
                wgpu.RenderPassColorAttachment(
                    view=self.get_texture_view(
                        target.name,
                        wgpu.TextureUsage.RENDER_ATTACHMENT,
                        create_if_not_exist=True,
                    ),
                    resolve_target=None,
                    clear_value=(0, 0, 0, 0),
                    load_op=load_op,
                    store_op="store",
                )
            )
        return attachments

    def get_shader_kwargs(self, material_pick_write, alpha_config):
        """Add cellier's fields to the generated ``FragmentOutput``.

        Each target also contributes its template var, so a cellier shader
        compiles the corresponding write away on a canvas using the stock
        blender and the same shader source stays valid on both.
        """
        kwargs = super().get_shader_kwargs(material_pick_write, alpha_config)
        code = kwargs["fragment_output_code"]
        for target in self._extra_targets:
            enabled = bool(material_pick_write) or not target.gated_on_pick
            code = self._add_field(code, target, enabled=enabled)
            kwargs[target.template_var] = enabled
        kwargs["fragment_output_code"] = code
        return kwargs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _add_field(code: str, target: ExtraTarget, *, enabled: bool) -> str:
        """Insert one extra field into the ``FragmentOutput`` struct.

        The location index is derived from the highest ``@location``
        already in the struct rather than hardcoded, because it differs per
        alpha method -- 2 for opaque/blended/stochastic, 3 for weighted,
        where accum and reveal take 0 and 1 -- and because a second call
        has to land after whatever the first one added.  Commented-out
        fields still count: pygfx disables pick by commenting the line out,
        but the pipeline keeps the target, so the location number stays
        reserved, and the same is true of a disabled field added here.

        Raises
        ------
        RuntimeError
            If the struct cannot be found.  That means pygfx changed the
            shape of the generated code, and failing loudly here is much
            better than a shader compile error inside the draw callback.
        """
        match = _STRUCT_RE.search(code)
        if match is None:
            raise RuntimeError(
                "could not find the FragmentOutput struct in the code pygfx "
                "generated. cellier.render._cellier_blender needs updating "
                "for this pygfx version."
            )
        body, indent = match.group(1), match.group(2)
        locations = [int(v) for v in _LOCATION_RE.findall(body)]
        if not locations:
            raise RuntimeError(
                "the FragmentOutput struct pygfx generated declares no "
                "@location fields; cellier.render._cellier_blender cannot "
                f"place the {target.name} field."
            )
        next_location = max(locations) + 1

        prefix = "" if enabled else "// "
        field = (
            f"\n{indent}    {prefix}@location({next_location}) "
            f"{target.name}: {target.wgsl_type},"
        )
        insert_at = match.start(2) - 1  # just before the newline before "};"
        return code[:insert_at] + field + code[insert_at:]


# ---------------------------------------------------------------------------
# Installation and access
# ---------------------------------------------------------------------------


def install_cellier_blender(
    renderer: gfx.WgpuRenderer, extra_targets: Iterable[str]
) -> bool:
    """Swap in a blender carrying *extra_targets*.

    Must be called before the renderer's first draw, and only for the
    targets a feature actually needs: the target list feeds
    ``Blender.hash``, which keys the pipeline cache, so adding or removing
    one mid-session would invalidate every pipeline in the process.

    Calling this twice on the same renderer is a no-op only if the second
    call asks for a subset of what is already installed; asking for more
    returns ``False`` rather than silently replacing a blender whose
    pipelines are already in the cache.

    Parameters
    ----------
    renderer : gfx.WgpuRenderer
        The renderer whose blender should be replaced.
    extra_targets : Iterable[str]
        Names from :data:`EXTRA_TARGETS`.  An empty iterable installs
        nothing and returns ``False``.

    Returns
    -------
    bool
        ``True`` if a blender carrying every requested target is in place.
        ``False`` if nothing was requested, or the pygfx internals are not
        the expected shape, in which case the stock blender is left alone
        and the dependent features degrade rather than break.
    """
    wanted = set(extra_targets)
    if not wanted:
        return False
    existing = getattr(renderer, "_blender", None)
    if existing is None:
        return False
    if isinstance(existing, CellierBlender):
        have = {target.name for target in existing.extra_targets}
        return wanted <= have
    try:
        existing_info = existing.texture_info
        replacement = CellierBlender(
            extra_targets=wanted,
            enable_pick="pick" in existing_info,
            enable_depth="depth" in existing_info,
        )
        # Carry over any usage bits already granted on the blender being
        # replaced -- notably TEXTURE_BINDING on pick, which
        # ``enable_pick_texture_binding`` may have added.  Without this the
        # grant is silently discarded and the dependent feature degrades to
        # a passthrough, which is the sort of failure that shows up as "it
        # just does nothing".  Copying makes the two calls order-independent.
        for name, info in existing_info.items():
            if name in replacement.texture_info:
                replacement.texture_info[name]["usage"] |= info["usage"]
        renderer._blender = replacement
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return True


def get_extra_target_view(
    renderer: gfx.WgpuRenderer, name: str
) -> wgpu.GPUTextureView | None:
    """Return a bindable view on one extra target, or ``None``.

    ``None`` means this renderer is on the stock blender, or was not given
    this particular target, so the caller should fall back: outlines to
    whole-object silhouettes, ambient occlusion to depth reconstruction.

    Parameters
    ----------
    renderer : gfx.WgpuRenderer
        The renderer to fetch the view from.
    name : str
        A key of :data:`EXTRA_TARGETS`.

    Returns
    -------
    wgpu.GPUTextureView or None
        The bindable view, or ``None``.

    Notes
    -----
    ``wgpu.GPUError`` is load-bearing in the except clause: a view that
    cannot be created raises ``GPUValidationError`` rather than returning
    ``None``, and because this runs inside the draw callback,
    ``rendercanvas`` swallows that without so much as a log line.
    """
    blender = getattr(renderer, "_blender", None)
    if blender is None:
        return None
    try:
        return blender.get_texture_view(
            name, wgpu.TextureUsage.TEXTURE_BINDING, create_if_not_exist=False
        )
    except (AttributeError, KeyError, TypeError, wgpu.GPUError):
        return None
