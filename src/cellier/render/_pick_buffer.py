"""Access to the pygfx renderer's pick texture.

This is the **only** module in cellier that touches ``renderer._blender``.
The screen-space outline pass reads per-pixel object ids out of the pick
buffer, which pygfx allocates without the ``TEXTURE_BINDING`` usage, so it
cannot be bound to a shader as it ships.

Two facts make the grant below safe:

* ``Blender`` creates its textures lazily from ``texture_info[name]["usage"]``,
  so raising the usage bits before the first draw is honoured.
* ``ensure_target_size`` clears the textures but *not* ``_texture_info``, so
  the grant survives every resize.

``texture_info``, ``get_texture`` and ``get_texture_view`` are all public on
``Blender``; the private access is the ``_blender`` attribute itself.  Both
functions degrade rather than raise: a pygfx rename disables outlining
instead of breaking the canvas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import wgpu

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pygfx as gfx


def enable_pick_texture_binding(renderer: gfx.WgpuRenderer) -> bool:
    """Grant ``TEXTURE_BINDING`` to the renderer's pick texture.

    Must be called before the renderer's first draw: once the pick texture
    exists its usage flags are fixed.

    Parameters
    ----------
    renderer : gfx.WgpuRenderer
        The renderer whose blender should be modified.

    Returns
    -------
    bool
        ``True`` if the usage was granted.  ``False`` if the pygfx
        internals are not the expected shape, or the pick texture has
        already been created (too late to change its usage), in which case
        outlining stays unavailable on this renderer.
    """
    blender = getattr(renderer, "_blender", None)
    if blender is None:
        return False
    try:
        if blender.get_texture("pick") is not None:
            # Already created; its usage flags can no longer be changed.
            return False
        blender.texture_info["pick"]["usage"] |= wgpu.TextureUsage.TEXTURE_BINDING
    except (AttributeError, KeyError, TypeError):
        return False
    return True


def get_pick_view(renderer: gfx.WgpuRenderer) -> wgpu.GPUTextureView | None:
    """Return a bindable view on the renderer's pick texture, or ``None``.

    Parameters
    ----------
    renderer : gfx.WgpuRenderer
        The renderer to fetch the pick view from.

    Returns
    -------
    wgpu.GPUTextureView or None
        ``None`` when this renderer has no pick target, or when the
        ``TEXTURE_BINDING`` usage was never granted.  Callers should skip
        the frame's effect rather than treat this as fatal.

    Notes
    -----
    ``wgpu.GPUError`` is load-bearing in the except clause.  Without the
    usage grant ``create_view`` does not return ``None``, it raises
    ``GPUValidationError`` -- and because this runs inside the draw
    callback, ``rendercanvas`` swallows that without so much as a log line
    and presents nothing.  The observed failure is a black canvas and an
    empty console.  Catching it degrades to a passthrough frame instead.
    """
    blender = getattr(renderer, "_blender", None)
    if blender is None:
        return None
    try:
        return blender.get_texture_view(
            "pick", wgpu.TextureUsage.TEXTURE_BINDING, create_if_not_exist=False
        )
    except (AttributeError, KeyError, TypeError, wgpu.GPUError):
        return None
