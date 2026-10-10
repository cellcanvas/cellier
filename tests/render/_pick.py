"""Pick many pixels of one drawn frame with a single GPU read-back.

``renderer.get_pick_info`` copies one pixel to a buffer, submits and waits
for the buffer to map: a round trip to the GPU per pixel.  On a software
rasteriser that is a memory copy, but on a real (or virtualised) GPU it is
milliseconds, and a test that picks hundreds of pixels spends all its time
waiting (``scripts/ci_profiling/ci_profiling_investigation.md``).

`PickFrame` reads the renderer's whole pick target once and decodes a pixel
the way ``get_pick_info`` does.  It reaches into pygfx's private parts to do
so (the blender's ``"pick"`` texture, the object id provider and
``_wgpu_get_pick_info``), so ``tests/render/test_pick_frame.py`` checks it
against ``get_pick_info`` itself.
"""

from __future__ import annotations

import numpy as np
from pygfx.objects._base import id_provider
from pygfx.renderers.wgpu import get_shared

#: The low bits of a pick value that hold the world object's id.
_OBJECT_ID_MASK = 2**20 - 1


class PickFrame:
    """The pick target of the frame *renderer* last drew.

    Take it after the draw and before the next one: it is a copy, and does
    not follow later frames.

    Parameters
    ----------
    renderer : pygfx.WgpuRenderer
        The renderer, after a draw.
    """

    def __init__(self, renderer) -> None:
        texture = renderer._blender.get_texture("pick")
        width, height, _depth = texture.size
        # rgba16uint: 8 bytes a pixel, read here as one 64-bit value.
        raw = get_shared().device.queue.read_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": 8 * width, "rows_per_image": height},
            (width, height, 1),
        )
        self.values = np.frombuffer(raw, np.uint64).reshape(height, width)
        self._logical_size = tuple(renderer.logical_size)

    def info(self, pos: tuple[float, float]) -> dict:
        """What ``renderer.get_pick_info(pos)`` returns, without ``"rgba"``.

        Parameters
        ----------
        pos : tuple[float, float]
            ``(x, y)`` in logical pixels, origin at the top left.

        Returns
        -------
        info : dict
            ``"world_object"`` (``None`` where nothing pickable was drawn)
            and the object's own pick fields.
        """
        height, width = self.values.shape
        # The mapping of pygfx's ``_copy_pixel``.
        x = max(0, min(width - 1, int(pos[0] / self._logical_size[0] * width)))
        y = max(0, min(height - 1, int(pos[1] / self._logical_size[1] * height)))
        pick_value = int(self.values[y, x])
        wobject = id_provider.get_object_from_id(pick_value & _OBJECT_ID_MASK)
        info = {"world_object": wobject}
        if wobject:
            info.update(wobject._wgpu_get_pick_info(pick_value))
        return info
