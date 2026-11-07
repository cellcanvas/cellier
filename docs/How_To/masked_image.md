# Colour a surface with another image

A **MaskedImage** draws an isosurface found in one image (the **mask**) and
colours it from a second image (the **value**). Use it when the shape comes
from one image and the colour should come from another. For example: a cell
boundary from a segmentation, coloured by a fluorescence channel.

- **In 3D** the surface is found and lit in the mask. Each pixel's colour is
  the max, mean or min of the value image along the surface normal, over a
  thickness you choose.
- **In 2D** the same thickness is the width of a contour band round the mask's
  threshold, and each pixel in the band shows the value image at that pixel.

The two images can be on different grids, at different resolutions, in memory
or multiscale, in any mix.

## The short version

```python
from cellier.convenience import Viewer
from cellier.convenience.gui import ImageControl
from cellier.scene.dims import spatial_axes
from cellier.visuals import MaskedImageSingleAppearance

viewer = Viewer(spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"))
viewer.add_masked_image(
    mask_store,
    value_store,
    single=MaskedImageSingleAppearance(
        iso_threshold=0.0,       # in the mask's units
        surface_thickness=4.0,   # world units
        color_map="viridis",     # the value's colour map ...
        clim=(100.0, 2000.0),    # ... and contrast, in the value's units
    ),
    controls=[ImageControl(fields=["visible", "color_map", "clim", "iso_threshold",
                                   "surface_thickness", "surface_placement",
                                   "surface_projection"])],
)
```

The runnable version is `examples/masked_image/masked_image_synthetic.py`: an
ellipsoid's distance map as the mask and concentric bands as the value, so
each threshold lights a surface of one band.

## Whose units each field is in

| Field | Applies to | Units |
|---|---|---|
| `iso_threshold` | the mask | the mask array's own numbers |
| `color_map`, `clim`, `opacity` | the value | `clim` in the value array's own numbers |
| `surface_thickness` | both | world units |
| `surface_placement` | both | `"inward"` (into the mask, default), `"centred"`, `"outward"` |
| `surface_projection` | 3D | `"max"` (default), `"mean"` or `"min"` along the normal |
| `surface_min_width_px`, `band_rays`, `band_cap` | 2D | see [the 2D band](#the-2d-band) |

The numbers are never normalised. A `uint8` binary mask of 0 and 255 takes a
threshold of 127.5; one of 0 and 1 takes 0.5. Any dtype works, but both images
are held on the GPU as float32 whatever the store holds, so a `uint8` image
costs four times its size there. Integers above 2^24 lose precision in
float32, which no mask or band value comes near.

The image control labels the two sliders "Contrast (value)" and "Threshold
(mask)". Its threshold slider does not read the data to find its range. Set
`ImageControl(threshold_range=..., thickness_range=...)` to suit your mask.

## Make a good mask

The 3D surface is lit from the mask's gradient, so the mask's profile decides
how the surface looks.

- **A binary mask** (0 and 1) works, but lights as a staircase: its gradient
  only changes at voxel faces.
- **A signed distance field** works best: positive inside, negative outside,
  in world units, and smoothed by about a voxel (a Gaussian of sigma one
  voxel). The normals are then smooth, and the threshold and the thickness are
  both distances: `iso_threshold=0` is the boundary, `+6` a surface 6 units
  inside it, `-6` one 6 units outside it.
  `scripts/masked_image/prepare_mask.py` builds one from a labels image. It
  also clamps the field far from the boundary, which compresses well.

In 2D any profile works: binary, a probability map or a distance field. The
band is found by a search for the contour, not from the gradient.

### Build the pyramid rounded up

A multiscale mask's coarse levels should **cover the finest level's whole
extent**. When you halve a level with an odd number of voxels, round up and pad
the odd edge with the mask's *outside* value. Do not drop the last voxel.

A pyramid that drops it is still drawn correctly, but less accurately. Each
coarse level is then a few finest voxels short along its far edges. In that
strip the mask's read repeats the level's last voxel rather than reading past
it, so nothing spurious is drawn there. But the strip shows the edge voxel's
value, not the data, until a finer level loads. A rounded-up pyramid has no
such strip. `prepare_mask.py` still floor-halves; the synthetic example rounds
up.

## Two grids: `value_to_mask`

When the images share a grid, leave `value_to_mask` as `None`. Otherwise pass
the map from the **value**'s level-0 data coordinates to the **mask**'s, as an
`AffineTransform` between the two stores' level-0 systems or as an augmented
matrix in data axis order:

```python
import numpy as np

# The value at twice the mask's resolution on every axis, voxel centres aligned.
value_to_mask = np.eye(4)
value_to_mask[:3, :3] *= 0.5
value_to_mask[:3, 3] = -0.25
```

`transform` places the mask in the world. The value image follows it: its
world transform is `transform` after `value_to_mask`, so moving the visual
moves both.

## Time, channels and other axes

- **The value needs the same non-spatial axes as the mask**: the same time
  axis, the same `c` axis, following the same world axes. A static value under
  a time-varying mask is refused, with a message saying which axis differs.
- **A `c` axis is a slider, not a channel.** A MaskedImage has one value
  channel, so `channel_axis`, `composite` and `channels` are refused. To see
  another channel, move the `c` slider.
- **Only the current time point is drawn.** When the time slider moves, both
  images show nothing of the new time point until it loads, and never a mix of
  two time points. Stepping back to a time point whose bricks are still on the
  GPU shows it at once.
- **A large time series should be a multiscale store.** A multiscale mask
  shows a whole coarse surface one frame after a step. An in-memory one has a
  single level, so a step loads it at full resolution: one blank frame up to
  about 256 cubed, but at 512 cubed about 0.1 s blank and then 0.2 to 0.3 s of
  the surface filling in brick by brick.

## GPU memory

Each image gets its own 3D atlas, `gpu_budget_bytes` for the mask and
`value_gpu_budget_bytes` for the value, **1,100 MiB each by default**. Each
atlas is allocated at its full budget whatever the image's size, so the default
reserves about 2 GiB even for a small pair. Lower both to reserve less:

```python
from cellier.visuals import MaskedImageRenderConfig

render_config = MaskedImageRenderConfig(
    gpu_budget_bytes=256 * 1024**2,
    value_gpu_budget_bytes=256 * 1024**2,
)
```

An in-memory image is loaded whole, so its one level must fit its budget; one
that does not is refused when it is added. At the default that is about 543
cubed.

## The 2D band

A 2D view draws the band of `surface_thickness` world units round the mask's
contour at `iso_threshold`, placed inward, centred or outward. Each pixel in
the band shows the value image at that pixel.

- **`surface_min_width_px`** (default 1): the band is drawn at least this many
  screen pixels wide, so it neither shimmers when zoomed out nor vanishes at
  thickness 0. Thickness 0 is a hairline of this width. 0 draws the strict
  physical width.
- **`band_rays`** (default 16) and **`band_cap`** (default 32) set the
  contour search: the rays in the fan, and the steps a ray takes before its
  step widens. Fewer rays are faster, and on a steep mask lose up to a few
  percent of the band at its far edge (4 rays look faceted there).
- A region of the mask that has not loaded draws no band yet. It never draws
  a band at the edge of the loaded region.

A tilted 2D view (the slice cuts the data at an angle) reads each image at its
true depth, so the band follows the cut.

## Loading only the bricks near the surface

By default a 3D view loads every brick it shows, as any multiscale image does.
For a large volume where the surface is a small part of it,
`MaskedImageAppearance(target_bricks="surface")` loads only the bricks near the
surface:

```python
from cellier.visuals import MaskedImageAppearance

viewer.add_masked_image(
    mask_store, value_store,
    appearance=MaskedImageAppearance(target_bricks="surface"),
)
```

It finds the surface in the mask's coarsest loaded level (the backstop), then
keeps the mask's bricks near a coarse cell the threshold crosses, and the
value's bricks within `surface_thickness` of one. On a 512 cubed test
ellipsoid, at a surface 1 voxel deep and 10 thick, it reads 28% of the
bricks. The costs:

- **A feature too thin for the coarse level is never loaded.** A thin spike's
  coarse average falls below the threshold, so the coarse pass cannot see it:
  it stays absent, not just blurry, and waiting does not help. Use
  `"visible"` when thin features matter.
- Each new time point waits for its coarse level before the fine bricks are
  asked for.
- A threshold or thickness change loads the bricks near the new surface.
  While you drag, the plan is redone at most 10 times a second.
- It has no effect in 2D, and none on an in-memory image (one level, all of it
  loaded).

The image control shows it as "Load bricks", in 3D only.

## Picking

A MaskedImage emits `MaskedImagePickEvent` rather than `ImagePickEvent`. Its
`pick_info` has the hit in both images' level-0 coordinates
(`data_coordinate`, `value_coordinate`), the mask's value there
(`mask_value`), the value image's (`surface_value`), and in 3D the max, mean
or min along the normal that colours the surface (`projected_value`). Values
are read at level 0 at the nearest voxel, so a pick can differ a little from
the drawn colour while a coarse level is on screen.

## What else works

- **Clipping planes**, in the mask's level-0 data coordinates. A cut face is
  coloured from the value image along the plane's normal.
- **Outlines** and **ambient occlusion**, as on any image.
- **Level of detail**: `force_level` forces the mask's level and
  `value_force_level` the value's, since the pyramids can differ. Both are
  1-based (1 is the finest).

## Not supported

- More than one value channel, and composite mode.
- Render planes, and slab projection: `slab_projection` must stay `"none"`.
- A value image without the mask's non-spatial axes.

## In-memory images in 2D

An in-memory side is read in 2D by the same rule as a multiscale one. It can
differ from a plain in-memory 2D image visual only in the outer half voxel at
the data's edge, where this rule fades out and the plain image clamps.
