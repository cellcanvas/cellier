"""Block layout parameters for bricked volume rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockLayout3D:
    """Brick grid geometry for one level of detail level of a 3-D volume.

    Describes how a 3-D volume is divided into a regular grid of
    fixed-size bricks.  The layout is computed once per level of detail level at
    startup and then read by the level of detail selector, the LUT builder, and
    the shader uniform construction.  It has no mutable state and no
    knowledge of GPU resources.

    In the rendering pipeline, ``BlockLayout3D`` serves as the shared
    coordinate reference between the planner (which selects which bricks
    to level of detail) and the indirection system (which builds the LUT texture
    mapping virtual grid positions to physical cache slots).

    Parameters
    ----------
    volume_shape : tuple[int, int, int]
        Array dimensions of this level of detail level in numpy axis order.
        For a multiscale pyramid this is the shape of
        one specific level.
    block_size : int
        Side length of one rendering brick in voxels, applied equally
        to all three axes. This is the unit of data read from the
        source store and placed into one GPU cache slot. The
        corresponding GPU slot side length is
        block_size + 2 * overlap (see CacheParameters), where
        the overlap border prevents linear interpolation from bleeding
        across brick boundaries. Does not need to match the chunk size
        of the underlying zarr store. We should investigate allowing
        anisotropic block sizes in the future. Defaults to 32.

    Attributes
    ----------
    grid_dims : tuple[int, int, int]
        Number of bricks needed to tile this LOAD level along each axis
        of the volume, in numpy axis order.
        Computed as ceil(volume_shape[i] / block_size) per axis.
        This is also the shape of the LUT texture for this level, so it is
        bounded by the device's 3D texture limit (2048 per axis under the
        default WebGPU limits); ``build_lut_texture`` checks that bound.
    padded_shape : tuple[int, int, int]
        Volume extent rounded up to a whole number of bricks, in numpy
        axis order ``(D, H, W)``.  Bricks at the volume boundary may
        extend past ``volume_shape`` and are zero-padded on load.
    n_bricks : int
        Total number of bricks across all three axes.
    """

    volume_shape: tuple[int, int, int]
    block_size: int = 32

    @property
    def grid_dims(self) -> tuple[int, int, int]:
        """Number of blocks along each axis of the volume at this LOAD level."""
        n_depth, n_height, n_width = self.volume_shape
        return (
            math.ceil(n_depth / self.block_size),
            math.ceil(n_height / self.block_size),
            math.ceil(n_width / self.block_size),
        )

    @property
    def padded_shape(self) -> tuple[int, int, int]:
        """Volume extent rounded up to whole blocks, in numpy order ``(D, H, W)``."""
        grid_depth, grid_height, grid_width = self.grid_dims
        return (
            grid_depth * self.block_size,
            grid_height * self.block_size,
            grid_width * self.block_size,
        )

    @property
    def n_bricks(self) -> int:
        """Total number of blocks across all three axes."""
        grid_depth, grid_height, grid_width = self.grid_dims
        return grid_depth * grid_height * grid_width
