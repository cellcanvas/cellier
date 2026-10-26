"""Release a visual's GPU memory when it is closed or dropped.

Two things kept a multiscale visual's brick caches (up to ``gpu_budget_bytes``
each) alive past their use:

* **Callback cycles.**  A visual hands its own bound methods to objects it
  owns (the residency's ``on_write``, a slot's data-ready listener).  Held
  strongly, each is a reference cycle, so a visual dropped without ``close()``
  lives until the cyclic garbage collector runs -- which a 1 GiB array does
  little to trigger, as it is a single allocation.  :func:`weak_callback`
  holds the instance weakly instead.
* **pygfx's bind-group cache.**  pygfx keeps bind groups in a process-global
  LRU (128 entries) that holds them strongly, and a bind group keeps its
  textures alive in wgpu.  A closed visual's textures therefore kept their GPU
  memory -- host RAM on lavapipe and Apple silicon -- until enough later bind
  groups evicted them.  :func:`destroy_textures` frees that memory now.
"""

from __future__ import annotations

import inspect
import weakref
from typing import TYPE_CHECKING, Any, Callable

import wgpu
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object
from pygfx.resources._base import resource_update_registry

if TYPE_CHECKING:
    import pygfx as gfx


def weak_callback(fn: Callable | None, dead_result: Any = None) -> Callable | None:
    """*fn*, holding its instance weakly when it is a bound method.

    Parameters
    ----------
    fn : Callable or None
        The callback.  Anything but a bound method is returned unchanged.
    dead_result : Any
        What the callback returns once its instance has been collected.

    Returns
    -------
    callback : Callable or None
        Calls *fn* while its instance lives, and returns *dead_result* after.
    """
    if fn is None or not inspect.ismethod(fn):
        return fn
    ref = weakref.WeakMethod(fn)

    def call(*args: Any) -> Any:
        method = ref()
        return dead_result if method is None else method(*args)

    return call


def destroy_textures(*textures: gfx.Texture | None) -> None:
    """Free the GPU memory of *textures* now.  They are unusable afterwards.

    A texture that never reached the GPU (or ``None``) is skipped.  Pending
    uploads are dropped first: pygfx flushes every pending resource in the
    process on any renderer's next draw, and one aimed at a destroyed texture
    would be a validation error.  The bind groups still naming a destroyed
    texture are never used again: pygfx keys them by the objects' unique ids.
    """
    for texture in textures:
        wgpu_texture = getattr(texture, "_wgpu_object", None)
        if wgpu_texture is None:
            continue
        resource_update_registry._syncable.discard(texture)
        wgpu_texture.destroy()


def allocate_zeroed_texture(texture: gfx.Texture) -> None:
    """Create *texture* on the GPU now, without uploading its data.

    For a texture whose CPU data is still all zeros (a brick atlas nothing
    has been written into): WebGPU zero-initialises a new texture, so the
    upload pygfx would make on the first draw -- the whole atlas, up to
    ``gpu_budget_bytes`` -- is skipped.  A 1 GiB atlas took about 0.45 s to
    upload on its first draw (oblique 2D design v3, U12).  Bricks written
    afterwards upload as usual.  A texture already on the GPU is left alone.

    The one pygfx internal used here is ``ensure_wgpu_object``, with the
    usage and chunk bookkeeping it is paired with; a canary test pins them.
    """
    if getattr(texture, "_wgpu_object", None) is not None:
        return
    # pygfx adds a texture's usage flags when it binds it, which has not
    # happened yet: the atlas is sampled by the brick shaders.
    texture._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING
    ensure_wgpu_object(texture)
    # Drop the pending full upload: the GPU copy already holds the zeros.
    texture._gfx_get_chunk_descriptions()
    resource_update_registry._syncable.discard(texture)
