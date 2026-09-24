# Multiscale brick lookup

This document explains how multiscale image and label visuals find the texel
for a sample, a bug that made them draw the wrong bricks on pyramids whose
level ratio is not an integer, and the rule that fixes it. It also covers
where each level is placed ([Level placement](#level-placement)), how densely
3D rays sample ([Ray step density](#ray-step-density)), and a related
cache-key problem found while diagnosing the brick bug.

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
| 2D image and label tiles | 2 (`TILE_BORDER_2D`) |

A **LUT** (lookup texture) says which atlas slot to read for each part of the
volume. It has one **base cell** per finest-level brick, so a 389 x 610 voxel
image with 32-voxel blocks has a 13 x 20 cell LUT. Each cell stores an atlas
slot and a level.

Two sides use the LUT:

1. **The writer** (`paint_lut`, through each LUT manager's `paint`) writes every resident
   brick into the base cells it covers, coarsest level first, so finer data
   overwrites coarser placeholders. A level-k brick covers several base cells.
2. **The shader** turns a sample position into a base cell, reads the cell's
   slot and level, works out the corner of that brick in level-k voxels, and
   samples the atlas at the level coordinate `(p - t) / scale - corner`
   (see [Level placement](#level-placement) for `p` and `t`).

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
  level needs more than `overlap - sampling_margin` voxels, or when a brick
  owns no cell at all. The distance includes the level's translation, and
  the margin is how far past a sample the path reads (see
  [Padding budgets](#padding-budgets)). On the light-sheet dataset, with its
  published translations, nothing warns; with 1-voxel 2D tiles (before the
  tiles got 2) the 2D image warned at levels 3 and 4 in y.

### Alternatives considered

- **Unrounded rule everywhere** (`brick = floor(cell / ratio)` for both sides).
  Also consistent, but a sample can then land up to a whole cell (`32 / ratio`
  voxels, 16 at level 2) outside its brick, far beyond any padding.
- **Per-level page tables**, i.e. one LUT per level indexed by that level's own
  brick coordinates. Exact for any ratio, including 1.5x or 3x pyramids, but a
  much larger change to the textures and the shader fallback logic. Worth
  revisiting if the padding warning fires on real data.

## Level placement

The rule above says *which brick* to read. Where a level sits in the volume
comes from its **level-to-data transform**: per axis a scale `s` and a
translation `t` in level-0 voxels (the OME-Zarr per-dataset
`coordinateTransformations`, relative to level 0). The convention is voxel
centres, `p = s * u + t` (see
[Coordinate systems](coordinate_systems.md#multiscale-level-placement)), and
`cellier.render._level_mapping` is the one definition of it.

### What was wrong

Until plan v2 (`plans/multiscale_level_transform_v2.md`) the translations
were parsed but never reached the GPU, and each shader assumed its own
placement. Where each drew the centre of coarse voxel `i` at cumulative
factor `F` (correct: `F * i + t`):

| Path | Drawn centre | Implied convention |
|---|---|---|
| 3D image | `F * i` | plain striding |
| 3D labels | `F * i - 0.5` | none |
| 2D image | `F * i + (F - 1) / 2` | block averaging |
| 2D labels | `F * i - 0.5` | none |

So stepping the level moved things by up to `F / 2` voxels, labels sat half a
voxel off the image (and off picking and the paint overlay) even at level 0,
and the same dataset drew differently in 2D and 3D.
`examples/multiscale_transform_validation.py` shows it.

### How it works now

- Python uploads `t` per level as `offset_k` in the `BlockScales` uniform
  (next to `scale_k`, `span_k`, `bricks_k`), shader order, level-0 voxels.
  `cellier.level_mapping.wgsl` reads it (`get_level_offset`).
- **2D** (`image_block.wgsl`, `label_block.wgsl`): per fragment,
  `p = pos - 0.5` (the proxy's edge position to the centred one),
  `u = (p - t) / s`; the image samples texel `u - corner + 0.5`, labels read
  voxel `floor(u - corner + 0.5)`.
- **3D** (`multiscale_volume_brick.wgsl`, `label_volume_brick.wgsl`): the
  offset is folded into the brick corner **once per brick**
  (`placed_corner_k`: `corner + (t + shift) / s`, computed in `setup_brick`,
  `lookup_brick_mip` and `lookup_brick_context`), so a per-sample lookup is
  just `p / s - placed_corner`. Looking `t` up per sample (a `switch` over
  the levels) cost 17-30% of the frame; per brick it is about 3%. The labels
  shader works in index space (`p + 0.5`) and folds that half voxel into the
  corner too (`shift = 0.5`).
- **Label rays sample at the midpoint of each step** (`t + (i + 0.5) *
  step`). They carry no jitter and each brick segment starts on a brick
  face, so at one sample per voxel face-on, samples starting on the face land
  exactly on voxel faces, where `floor(u + 0.5)` is a coin toss.
- Brick selection, the DDA and the LUT stay in index space on the base cell
  grid: a translation moves a level by less than one of its voxels
  (the contract below), which the padding absorbs.
- CPU planning (level-of-detail sorting, frustum and viewport culling,
  `keys_in_region`) uses the same boxes: `brick_box_data` /
  `brick_centre_data`.

### The contract

The anchor-brick LUT above can only place levels that stay over the level-0
block they summarise. `cellier.data._level_contract.validate_level_transforms`
enforces, per coarse level and axis: a diagonal transform; `s >= 1` and not
decreasing with level; `-0.5 <= t <= s - 0.5`; and the level covers level 0's
extent to within one of its voxels. Stores that break it raise when they get
their transforms or are added to a scene. Block averaging (`t = (s - 1) / 2`),
plain striding (`t = 0`) and offset striding (`t = s // 2`) all pass, with
integer or non-integer ratios.

Arbitrary translations (cropped or shifted coarse levels) would need
per-level page tables and a per-sample brick lookup, and a rendering domain
that is the union of the levels; that is future work, and the helpers above
carry over.

### Padding budgets

What the ghost border has to absorb is the out-of-brick distance
(`max_out_of_brick`: the cell rule plus the translation term) plus how far
past a sample position the path reads, in level-k voxels:

| Path | Border | Read past the sample | Allowance |
|---|---|---|---|
| 2D image | 2 | 0.5 (linear) | 1.5 |
| 2D labels | 2 | 0 (nearest) | 2.0 |
| 3D image | 3 | 1.5 (gradient probe; bisection is `1 / ray_steps_per_voxel`) | 1.5 |
| 3D labels | 2 | 1 (one bisection step at the default density) | 1.0 |

The visuals pass these margins to the LUT managers at the default ray
density; the check only warns, because exceeding the border repeats an edge
texel. Worst cases in the repo: 0.48 for integer pyramids with their real
translations, 1.34 for the light-sheet pyramid with block-average placement
(0.63 with its published `t = 0`).

## Ray step density

The 3D brick shaders take `ray_steps_per_voxel` samples per voxel of the
drawn level, counted along the ray (`ray_step_count`: the L2 length of the
ray segment in level-k voxels), set on `MultiscaleImageAppearance` and
`MultiscaleLabelsAppearance` (default 1.0, allowed 0.5-8, a live uniform).
The old count, `24 / max(lod_scale)` steps per brick in normalised physical
space, depended on the block size and starved thin axes: on an anisotropic
level (z scale 1, y/x 16) a view down z took about one step per brick and
dropped whole labels. Labels need about one step per voxel (nearest sampling
must visit each voxel); a MIP at 0.5 can lose up to all of a one-voxel spot
(the worst-case loss is `1 / (2 * ray_steps_per_voxel)` of the peak).

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
- `tests/render/test_multiscale_level_alignment.py`: every level of four
  pyramid kinds (offset, plain, offset with a level-0 translation, block
  average), 2D and 3D, image and labels, drawn at its transform to within
  0.2 level-0 voxels.
- `tests/render/test_multiscale_labels_2d_reference.py`: 2D labels on a
  translated non-integer pyramid match the reference sampler
  (`tests/render/_level_reference.py`) pixel for pixel, and a 2D pick just
  inside a label edge returns the drawn label;
  `tests/render/test_multiscale_labels_3d_pick.py` does the pick in 3D.
- `tests/data/test_level_contract.py`, `tests/render/test_level_mapping.py`,
  `tests/render/test_level_offsets_uploaded.py`: the contract, the mapping
  helpers, and the offsets reaching the uniform.
- `tests/render/test_ray_steps_per_voxel.py`: the step density is live and
  sets what a ray can miss.

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
- A sampler must place the level with `cellier.render._level_mapping` (CPU)
  or `level_mapping.wgsl` (GPU): `u = (p - t) / s` from the centred position
  `p`, never `position / scale`.
- Anything that depends on the level goes in the per-brick setup
  (`BrickInfo`, `BrickContext`), never in the per-sample path.
- Pass the store's translations (`_translation_vecs_data`) wherever the
  scales go: the scale buffers and the LUT managers.
