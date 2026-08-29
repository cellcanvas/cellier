"""A pygfx ``Blender`` carrying an extra ``outline_id`` render target.

Phase 2 of the outline design needs a per-pixel *label* key, not just a
per-pixel object id.  The pick buffer has no room: 64 bits, with 20 spent
on ``global_id`` and 42 on the surface coordinate.  Rather than buy space
by cutting the pick coordinate's precision -- which costs 3D picking
accuracy and forces a hashed, collision-prone key -- this module adds a
fourth render target, ``r32uint``, written by cellier's label shaders in
the same pass as colour and pick.  32 bits is enough for the key to be
exact.

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
  literal), so shaders that never assign the new field emit 0 -- which is
  exactly "no outline key".
* ``ensure_target_size`` and ``clear`` walk ``_texture_info`` generically,
  so resize and per-frame clearing need no extra handling.
* Appending the target last in all three methods keeps target-state order,
  attachment order and ``@location(N)`` aligned.

The one brittle part is ``get_shader_kwargs``, which returns the
``FragmentOutput`` struct as a **WGSL source string** branched four ways by
alpha method.  A pygfx rewording surfaces as a shader compile error inside
the draw callback, which ``rendercanvas`` swallows without logging.
``tests/render/test_outline_blender.py`` renders a frame and reads the
target back so that becomes a CI failure instead of a black canvas.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import wgpu
from pygfx.renderers.wgpu.engine.blender import Blender

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pygfx as gfx

#: Name of the extra render target, as seen by ``Blender.texture_info``.
OUTLINE_ID_TARGET: str = "outline_id"

#: Name of the field added to the generated ``FragmentOutput`` struct.
OUTLINE_ID_FIELD: str = "outline_id"

_STRUCT_RE = re.compile(r"struct\s+FragmentOutput\s*\{(.*?)\n(\s*)\};", re.DOTALL)
_LOCATION_RE = re.compile(r"@location\((\d+)\)")


class OutlineBlender(Blender):
    """``Blender`` with an extra ``r32uint`` target for label outline keys.

    Behaves exactly like the stock blender for every object that does not
    write the new field; the colour output of a scene is unchanged.

    Parameters
    ----------
    **kwargs
        Forwarded to ``Blender`` (``enable_pick``, ``enable_depth``).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # texture_info is a public property returning the live dict.
        self.texture_info[OUTLINE_ID_TARGET] = {
            "name": OUTLINE_ID_TARGET,
            "format": wgpu.TextureFormat.r32uint,
            "usage": (
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_SRC
            ),
            "is_used": False,
            "clear": True,
        }

    # ------------------------------------------------------------------
    # Blender overrides -- each is super() plus one appended entry
    # ------------------------------------------------------------------

    def get_color_descriptors(self, material_pick_write, alpha_config):
        """Append the ``outline_id`` target state to the pipeline targets.

        Follows the same convention pygfx uses for pick: the target state
        is always declared so all pipelines match, and whether an object
        actually writes it is decided by the write mask.
        """
        target_states = super().get_color_descriptors(material_pick_write, alpha_config)
        texinfo = self.texture_info[OUTLINE_ID_TARGET]
        texinfo["is_used"] = True
        target_states.append(
            {
                "format": texinfo["format"],
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL if material_pick_write else 0,
            }
        )
        return target_states

    def get_color_attachments(self, pass_type):
        """Append the ``outline_id`` attachment, honouring the clear flag."""
        attachments = super().get_color_attachments(pass_type)
        texinfo = self.texture_info[OUTLINE_ID_TARGET]

        load_op = wgpu.LoadOp.load
        if texinfo["clear"]:
            texinfo["clear"] = False
            load_op = wgpu.LoadOp.clear

        attachments.append(
            wgpu.RenderPassColorAttachment(
                view=self.get_texture_view(
                    OUTLINE_ID_TARGET,
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
        """Add the ``outline_id`` field to the generated ``FragmentOutput``.

        Gated on ``material_pick_write`` for the same reason pick is: an
        object that writes no pick has no outline either, and leaving both
        out keeps the declared locations contiguous.  Adds a
        ``write_outline_id`` template var so cellier's label shaders can
        compile the write away on a canvas using the stock blender.
        """
        kwargs = super().get_shader_kwargs(material_pick_write, alpha_config)
        do_outline_id = bool(material_pick_write)
        kwargs["fragment_output_code"] = self._add_outline_id_field(
            kwargs["fragment_output_code"], enabled=do_outline_id
        )
        kwargs["write_outline_id"] = do_outline_id
        return kwargs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _add_outline_id_field(code: str, *, enabled: bool) -> str:
        """Insert the extra field into the ``FragmentOutput`` struct.

        The location index is derived from the highest ``@location`` already
        in the struct rather than hardcoded, because it differs per alpha
        method -- 2 for opaque/blended/stochastic, 3 for weighted, where
        accum and reveal take 0 and 1.  Commented-out fields still count:
        pygfx disables pick by commenting the line out, but the pipeline
        keeps the target, so the location number stays reserved.

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
                "generated. cellier.render._outline_blender needs updating "
                "for this pygfx version."
            )
        body, indent = match.group(1), match.group(2)
        locations = [int(v) for v in _LOCATION_RE.findall(body)]
        if not locations:
            raise RuntimeError(
                "the FragmentOutput struct pygfx generated declares no "
                "@location fields; cellier.render._outline_blender cannot "
                "place the outline_id field."
            )
        next_location = max(locations) + 1

        prefix = "" if enabled else "// "
        field = (
            f"\n{indent}    {prefix}@location({next_location}) {OUTLINE_ID_FIELD}: u32,"
        )
        insert_at = match.start(2) - 1  # just before the newline before "};"
        return code[:insert_at] + field + code[insert_at:]


# ---------------------------------------------------------------------------
# Installation and access
# ---------------------------------------------------------------------------


def install_outline_blender(renderer: gfx.WgpuRenderer) -> bool:
    """Swap in a blender carrying the ``outline_id`` target.

    Must be called before the renderer's first draw, and only when
    outlines are enabled: the target list feeds ``Blender.hash``, which
    keys the pipeline cache, so adding or removing it mid-session would
    invalidate every pipeline in the process.

    Parameters
    ----------
    renderer : gfx.WgpuRenderer
        The renderer whose blender should be replaced.

    Returns
    -------
    bool
        ``True`` if the blender was installed.  ``False`` if the pygfx
        internals are not the expected shape, in which case the stock
        blender is left alone and label outlines degrade to whole-object
        silhouettes.
    """
    existing = getattr(renderer, "_blender", None)
    if existing is None:
        return False
    if isinstance(existing, OutlineBlender):
        return True
    try:
        existing_info = existing.texture_info
        replacement = OutlineBlender(
            enable_pick="pick" in existing_info,
            enable_depth="depth" in existing_info,
        )
        # Carry over any usage bits already granted on the blender being
        # replaced -- notably TEXTURE_BINDING on pick, which
        # ``enable_pick_texture_binding`` may have added.  Without this the
        # grant is silently discarded and outlining degrades to a
        # passthrough, which is the sort of failure that shows up as "the
        # feature just does nothing".  Copying makes the two calls
        # order-independent.
        for name, info in existing_info.items():
            if name in replacement.texture_info:
                replacement.texture_info[name]["usage"] |= info["usage"]
        renderer._blender = replacement
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return True


def get_outline_id_view(renderer: gfx.WgpuRenderer) -> wgpu.GPUTextureView | None:
    """Return a bindable view on the ``outline_id`` target, or ``None``.

    ``None`` means this renderer is on the stock blender, so there are no
    label keys and the composite pass should fall back to ``global_id``.
    """
    blender = getattr(renderer, "_blender", None)
    if blender is None:
        return None
    try:
        return blender.get_texture_view(
            OUTLINE_ID_TARGET,
            wgpu.TextureUsage.TEXTURE_BINDING,
            create_if_not_exist=False,
        )
    except (AttributeError, KeyError, TypeError, wgpu.GPUError):
        return None
