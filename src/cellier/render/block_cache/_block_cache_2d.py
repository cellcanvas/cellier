import numpy as np

from cellier.logging import _GPU_LOGGER
from cellier.render.block_cache._cache_parameters_2d import (
    BlockCacheParameters2D,
    build_cache_texture_2d,
)
from cellier.render.block_cache._tile_manager_2d import (
    BlockKey2D,
    TileManager2D,
    TileSlot,
)


class BlockCache2D:
    """Fixed-size GPU slot pool for 2D tiles.

    Which tile lives in which slot is the chunk scheduler's decision; the
    image residency adapter writes the tiles.

    Parameters
    ----------
    cache_parameters : BlockCacheParameters2D
        Cache sizing metadata produced by ``compute_block_cache_parameters_2d()``.

    Attributes
    ----------
    info : BlockCacheParameters2D
        Cache sizing metadata (grid dims, slot count, padded block size).
    tile_manager : TileManager2D
        Slot geometry and the drawn-tile view.
    cache_data : np.ndarray
        CPU-side backing array, shape ``(cH, cW)``, dtype float32.
    cache_tex : gfx.Texture
        GPU 2-D float32 texture wrapping ``cache_data``.
    """

    def __init__(self, cache_parameters: BlockCacheParameters2D, dtype=None) -> None:
        self.info = cache_parameters
        self.tile_manager = TileManager2D(cache_parameters)
        self.cache_data, self.cache_tex = build_cache_texture_2d(
            cache_parameters, dtype=dtype
        )

    def write_tile(
        self, slot: TileSlot, data: np.ndarray, key: BlockKey2D | None = None
    ) -> None:
        """Write a padded 2D tile into the CPU array and mark dirty for GPU upload.

        Parameters
        ----------
        slot : TileSlot
            Target slot -- ``grid_pos`` determines the write offset.
        data : np.ndarray
            Float32 array of shape ``(pbs, pbs)`` where
            ``pbs = block_size + 2 * overlap``.
        key : BlockKey2D or None
            Tile identity for logging.
        """
        sy, sx = slot.grid_pos
        pbs = self.info.padded_block_size
        y0 = sy * pbs
        x0 = sx * pbs
        self.cache_data[y0 : y0 + pbs, x0 : x0 + pbs] = data
        self.cache_tex.update_range(
            offset=(x0, y0, 0),
            size=(pbs, pbs, 1),
        )
        _GPU_LOGGER.debug(
            "brick_written  key=%s  slot=%d  grid_pos=%s",
            key,
            slot.index,
            slot.grid_pos,
        )

    @property
    def n_resident(self) -> int:
        """Number of tiles the LUT currently draws."""
        return len(self.tile_manager.tilemap)

    def clear(self) -> None:
        """Forget the drawn view.  The slots' data is simply overwritten later."""
        self.tile_manager.clear()
