# Multiscale brick lookup

This document explains how multiscale image and label visuals find the texel
for a sample, a bug that made them draw the wrong bricks on pyramids whose
level ratio is not an integer, and the rule that fixes it. It also covers a
related cache-key problem found while diagnosing it.

If you touch the LUT writers, the brick/tile shaders, or the block cache keys,
read [the checklist](#checklist-for-changes) at the end.

## How the lookup works

Multiscale visuals load data in fixed-size **bricks** (3D) or **tiles** (2D),
`block_size` voxels on a side (32 by default). Each brick is stored in one slot
of a texture **atlas**, padded on every side with `overlap` voxels of the
neighbouring data so that samples near the brick edge still read real values:

| Path | Padding (`overlap`) |
|---|---|
| 3D image bricks | 3 |
| 3D label bricks | 2 |
| 2D image and label tiles | 1 |

A **LUT** (lookup texture) says which atlas slot to read for each part of the
volume. It has one **base cell** per finest-level brick, so a 389 x 610 voxel
image with 32-voxel blocks has a 13 x 20 cell LUT. Each cell stores an atlas
slot and a level.

Two sides use the LUT:

1. **The writer** (`rebuild_lut` / `rebuild_lut_2d`) writes every resident
   brick into the base cells it covers, coarsest level first, so finer data
   overwrites coarser placeholders. A level-k brick covers several base cells.
2. **The shader** turns a sample position into a base cell, reads the cell's
   slot and level, works out the corner of that brick in level-k voxels, and
   samples the atlas at `position / scale - corner`.

Both sides therefore answer the same question: *which level-k brick owns this
base cell?* The writer answers it when it decides which cells to fill; the
shader answers it when it computes the corner. If the answers differ, the
shader reads the right slot with the wrong offset, i.e. another part of the
data, the padding, or the next slot in the atlas.

## The bug

### Symptoms

On the light-sheet organoid dataset
(`scripts/v2/transforms_v2/lightsheet_viewer.py`), in 3D:

- At startup a strip of the volume was missing. Lowering `lod_bias` (finer
  levels) filled it; raising it brought it back.
- While scrubbing time, and after zooming, blocks of the volume showed content
  from elsewhere in the image, and the blocks changed as bricks re-fetched.

### Cause

The dataset's pyramid halves y and x with odd sizes and never downsamples z.
The OME-Zarr metadata records extent-preserving scales, so the level ratios are
not integers:

| Level | y shape | y ratio | x shape | x ratio |
|---|---|---|---|---|
| 1 | 389 | 1 | 610 | 1 |
| 2 | 194 | 2.005 | 305 | 2.0 |
| 3 | 96 | 4.052 | 152 | 4.013 |
| 4 | 47 | 8.277 | 76 | 8.026 |
| 5 | 23 | 16.913 | 38 | 16.053 |

A level-k brick spans `ratio` base cells, which is not a whole number, so the
writer and the shaders had to round somewhere, and they rounded differently:

- **The writer** gave each brick `round(ratio)` cells: brick `g` wrote cells
  `[g * round(ratio), (g + 1) * round(ratio))`.
- **The MIP shader** (`lookup_brick_mip`) computed the brick as
  `floor(cell / ratio)` with the unrounded ratio. At level 4 y, cell 8 belongs
  to brick 1 in the LUT but `floor(8 / 8.277) = 0` in the shader, so the whole
  cell was sampled against brick 0's corner.
- **The other 3D paths** (`setup_brick`, the label shader, the iso gradient)
  computed the brick from the unrounded *position*, so only slivers at brick
  boundaries were wrong.
- **The 2D tile shaders** wrapped the position with
  `(pos / ratio) mod block_size`, which is wrong in the same slivers.

On top of that, at level 3 y the writer's 3 bricks x 4 cells cover 12 cells of a
13-row grid, so **the last row was never written** at that level.

The cells the MIP shader got wrong, predicted from the metadata alone:

| Level | y cells | x cells |
|---|---|---|
| 2 | 2, 4, 6, 8, 10, 12 | none |
| 3 | 4, 8 (and row 12 unwritten) | 4, 8, 12, 16 |
| 4 | 8 | 8, 16 |
| 5 | none | 16 |

Why it looked time dependent: a wrong-brick read lands in the brick's padding
or in the neighbouring atlas slot. What sits there depends on which bricks are
loaded, and scrubbing or zooming loads and evicts bricks.

Any pyramid with non-integer ratios is affected, and those are common: halving
an odd size and recording the extent-preserving scale is the default for many
OME-Zarr writers. Exact power-of-two pyramids were never affected, which is why
the existing tests did not catch it.

### How it was diagnosed

`scripts/v2/transforms_v2/lightsheet_diagnostics.py` wraps the viewer and
instruments the render visuals in place. The decisive observations:

- The wrong-brick cells predicted from the metadata (table above) matched the
  live LUT exactly.
- Overwriting the shader's level scales with the LUT's rounded scales removed
  both the startup strip and the blocks seen while scrubbing time.

## Why not round everywhere

Giving the shader the rounded ratios makes it agree with the LUT, and was the
counterfactual above, but it draws every coarse level stretched by
`round(ratio) / ratio`. On this dataset that is up to 3.3% (level 4 y: 8 instead
of 8.277), about 3.4 um at the far edge of the volume, so levels misregister
against each other, visible as a jump when the LOD changes, and picks at coarse
levels report the wrong position.

## The fix: one integer rule, float positions

`cellier/render/lut_indirection/_cell_brick_rule.py` defines the single answer
both sides use:

```text
span   = max(1, round_half_up(ratio))       # base cells per level-k brick
bricks = ceil(level_shape / block_size)     # level-k brick count
brick  = min(cell // span, bricks - 1)      # the last brick owns the tail
```

- The **writers** fill `cell_range(g, span, bricks, grid)` for each brick, so
  every base cell is written by exactly the brick the rule names, including
  the tail row.
- Python uploads `span` and `bricks` per level in the existing `BlockScales`
  uniform (fields `span_k` and `bricks_k`, next to `scale_k`), so the shaders
  never round anything themselves.
- The **shaders** take the brick corner from the LUT cell with the same rule:
  `brick_corner_from_cell` in `cellier.brick_rule.wgsl` (both 3D brick
  shaders) and `tile_corner_from_cell` in `cellier.tile_rule.wgsl` (both 2D
  tile shaders). The iso gradient reuses the corner of the brick its LUT entry
  came from instead of recomputing one.
- **Positions inside the brick** still divide by the unrounded ratio, so
  geometry is exact.

### What it costs

The cells a brick owns and the brick's own extent no longer coincide exactly,
so a sample can land slightly outside the brick that owns its cell and read
that brick's padding, which holds the correct neighbouring data. The distance
is `max_out_of_brick` in `_cell_brick_rule`. For halving pyramids it stays
around one level-k voxel; on the light-sheet dataset the worst case is 1.07
voxels (level 4 y).

To keep this safe:

- Every sampler clamps its in-brick coordinate to the padded tile, so anything
  beyond the padding repeats the edge texel instead of reading another slot.
- `LutIndirectionManager3D` / `LutIndirectionManager2D` log a
  `brick_rule_padding` warning on the `cellier.render.gpu` logger when a
  level needs more than `overlap - 0.5` voxels, or when a brick owns no cell at
  all. On the light-sheet dataset 3D fits; 2D tiles (overlap 1) warn at levels
  3 and 4 in y, where a sliver of under a pixel repeats edge texels.

### Alternatives considered

- **Unrounded rule everywhere** (`brick = floor(cell / ratio)` for both sides).
  Also consistent, but a sample can then land up to a whole cell (`32 / ratio`
  voxels, 16 at level 2) outside its brick, far beyond any padding.
- **Per-level page tables**, i.e. one LUT per level indexed by that level's own
  brick coordinates. Exact for any ratio, including 1.5x or 3x pyramids, but a
  much larger change to the textures and the shader fallback logic. Worth
  revisiting if the padding warning fires on real data.

## Slice keys

Every cached brick carries a slice key (`BlockKey3D.slice_coord`,
`BlockKey2D.slice_coord`) recording where the collapsed axes were when it was
fetched, so bricks from different time points or channels do not collide.

The key used to be the continuous position of the selection pulled back into
data space. The fetch itself rounds that position to a plane, but the key did
not. On the light-sheet time axis, a `NonUniformAxisTransform` spanning about
400,000 seconds over 667 frames, almost every slider move produced a new key
for the same frame. The diagnostics log showed it directly: moving within frame
460 gave `hits=0 misses=33` with continuous keys and `hits=33 misses=0` with
rounded ones.

This never drew wrong pixels (a brick's data always matched its key), but it
re-fetched every visible brick on every sub-frame slider move and filled the
cache with duplicate copies of the same frame.

The key is now the level-0 selection the fetch actually uses on each collapsed
axis, an integer plane or a `(start, stop)` window for a slab
(`MultiscaleRegionPlanner._block_key_slice_coord`, built from
`axis_selections_from_box`). The 2D tile cache is still cleared on every slice
change (`SliceCoordinator._on_dims_changed`), so for now the benefit is in 3D.

## Tests that guard this

- `tests/render/lut_indirection/test_cell_brick_rule.py`: the rule itself,
  including the light-sheet pyramid's spans, counts, tail ownership and
  out-of-brick distances.
- `tests/render/lut_indirection/test_non_integer_pyramid_lut.py`: both LUT
  writers write every cell with the brick the rule names, and both scale
  buffers upload the rule in shader axis order.
- `tests/render/test_multiscale_non_integer_pyramid.py`: renders an odd-shaped
  pyramid at the finest and coarsest level (image and labels, 2D and 3D) and
  requires the pictures to agree.
- `tests/render/test_image_multiscale_render.py::test_slider_positions_on_the_same_frame_share_brick_keys`:
  two slider positions on the same frame share a key and the second plan is
  all cache hits.

## Checklist for changes

- A shader path that turns a position into a brick or tile corner must use
  `brick_corner_from_cell` / `tile_corner_from_cell` with the LUT cell it read,
  never the position and the float scale.
- A LUT writer must fill `cell_range(...)`, never
  `g * round(scale)` spans of its own.
- Construct the LUT managers and scale buffers with `level_shapes` (and the
  managers with `border`); without shapes the tail row is not owned.
- A new block cache key must describe what is fetched, not a continuous
  position.
