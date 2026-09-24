"""Generic GPU brick-cache for 3-D volume rendering.

Manages a fixed-size 3-D texture that acts as a flat pool of brick
slots, and the TileManager that tracks which logical brick occupies
which physical slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.logging import _GPU_LOGGER
from cellier.render.block_cache._cache_parameters_3d import (
    BlockCacheParameters3D,
    build_cache_texture_3d,
    commit_block_3d,
)
from cellier.render.block_cache._tile_manager_3d import (
    BlockKey3D,
    TileManager3D,
    TileSlot,
)

if TYPE_CHECKING:
    import numpy as np


class BlockCache3D:
    """Fixed-size GPU slot pool: a 3-D texture of brick slots.

    Slot allocation and eviction belong to the chunk scheduler; this owns the
    texture and writes bricks into it.

    Parameters
    ----------
    cache_parameters : CacheInfo
        Cache sizing metadata produced by ``compute_cache_info()``.
    dtype : np.dtype or None
        Data type for the cache texture. Defaults to float32.
        Pass ``np.int32`` for label caches.

    Attributes
    ----------
    info : CacheInfo
        Cache sizing metadata (grid dims, slot count, padded brick size).
    tile_manager : TileManager3D
        Slot geometry, and the view of drawn bricks.
    cache_data : np.ndarray
        CPU-side backing array, shape ``(cD, cH, cW)``.
    cache_tex : gfx.Texture
        GPU 3-D texture wrapping ``cache_data``.
    """

    def __init__(self, cache_parameters: BlockCacheParameters3D, dtype=None) -> None:
        self.info = cache_parameters
        self.tile_manager = TileManager3D(cache_parameters)
        self.cache_data, self.cache_tex = build_cache_texture_3d(
            cache_parameters, dtype=dtype
        )

    def write_brick(
        self,
        slot: TileSlot,
        data: np.ndarray,
        key: BlockKey3D | None = None,
        background_label: int = 0,
    ) -> bool:
        """Write a padded brick into the CPU array and mark dirty for GPU upload.

        The actual GPU transfer is deferred until the next
        ``renderer.render()`` call (uses pygfx ``update_range``).

        Parameters
        ----------
        slot : TileSlot
            Target slot — ``grid_pos`` determines the write offset.
        data : np.ndarray
            Array of shape ``(pbs, pbs, pbs)`` where
            ``pbs = block_size + 2 * overlap``.
        key : BlockKey3D or None
            Brick identity for logging.
        background_label : int
            For integer (label) caches: value treated as background.
            Returns True if the brick contains any non-background value.
            Ignored for float caches (always returns False).

        Returns
        -------
        bool
            True if brick contains any non-background value (useful for
            label caches to set ``slot.brick_max``).  For float caches
            this is always False.
        """
        import numpy as np

        commit_block_3d(
            cache_data=self.cache_data,
            cache_tex=self.cache_tex,
            grid_pos=slot.grid_pos,
            padded_block_size=self.info.padded_block_size,
            data=data,
        )
        _GPU_LOGGER.debug(
            "brick_written  key=%s  slot=%d  grid_pos=%s",
            key,
            slot.index,
            slot.grid_pos,
        )
        if np.issubdtype(self.cache_data.dtype, np.integer):
            return bool(np.any(data != background_label))
        return False

    @property
    def n_resident(self) -> int:
        """Number of bricks the LUT currently draws."""
        return len(self.tile_manager.tilemap)

    def clear(self) -> None:
        """Forget the drawn view.  The slots' data is simply overwritten later."""
        self.tile_manager.clear()
