# Rotate, shear and drift data

A visual's `transform` maps its data into the scene's world. It can be any
invertible affine transform: a scale and a shift, a rotation, a shear, a
mirror, or any combination -- and space may be coupled to time, which is
how a drift correction is written. This page says what draws in which view,
how cellier tells you when a view cannot draw something, and what to watch
for.

## The short version

```python
import numpy as np
from cellier.transform import AffineTransform

# data (z, y, x) -> world (z, y, x): rotate the (y, x) plane by 30 degrees.
theta = np.deg2rad(30)
matrix = np.eye(4)
matrix[1:3, 1:3] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

controller._ensure_data_coordinate_systems(scene.id, store)
transform = AffineTransform.from_matrix(
    matrix, store.data_coordinate_systems[0], scene.dims.world_coordinate_system
)
viewer.add_image(store, transform=transform)
```

The matrix is homogeneous, `(n_world + 1, n_data + 1)`, rows world axes and
columns data axes, in the order of the coordinate systems (`z, y, x`, not
pygfx's `x, y, z`). Assigning a new one later works the same way:
`visual.transform = transform`.

`examples/convenience/affine_transforms.py` is a complete example: a
rotated, anisotropic, drift-corrected time series with points tracking a
blob in it.

## What draws in which view

In a 3D view every visual draws under any affine. In a 2D view what matters
is how the transform meets the **slice**: the world axes the view does not
display are cut at the slider position, and that cut can be a plane of the
data or a plane tilted through it. cellier classifies each visual in each
view (`cellier.transform.classify_view`):

| The view is | Meaning | Example, 2D view of `(y, x)` |
|---|---|---|
| axis aligned | what worked before rotations | a scale; a 90 degree turn in `(y, x)`; a mirror |
| in plane | rotated or sheared within the displayed axes | a 30 degree turn in `(y, x)`; `x += 0.5 y` |
| slice shift | moving the slice moves the data across the view | `x += 0.5 z`; a drift `x += v t` |
| oblique | the slice is a tilted plane through the data | a turn in `(z, x)`; `z += 0.5 x` |

And what each kind of visual draws:

| Visual | axis aligned | in plane | slice shift | oblique |
|---|---|---|---|---|
| image, labels (in memory and multiscale) | yes | yes | yes | yes (from 3D bricks, below) |
| mesh, multiscale mesh | yes | yes | yes | yes (the section is cut by the tilted plane) |
| points, lines, graph | yes | yes | yes | yes (the slab is tilted through them) |

## Images and labels on a tilted slice

A 2D view whose slice is tilted through an image or labels volume draws it
from the volume's 3D bricks, sampled on the slice (the *brick route*). There
is no setting: cellier picks the path per view, and aligned views keep the
2D tiles they always used. Everything you set on the visual applies -- the
colormap and contrast, composite channels, label colours and outlines,
clipping planes, picks -- and an orthoviewer of a rotated volume draws it in
every panel. An in-memory array is served as a two-level pyramid: the array
itself and a coarse level, built on a worker when the visual is added (1/64
of the array's memory). Until that build lands a tilted view draws the full
level only.

What to know:

- **Turn on `coarsest_while_moving_2d` for interactive use.** A tilted view
  at native zoom re-reads bricks on every slider tick unless the visual's
  appearance has `coarsest_while_moving_2d=True`, which loads only the
  coarse level while the slider, a transform drag or a thickness drag moves
  and the full level once it stops. It is off by default, as for every 2D
  view; with it on a tilted view keeps 30 fps.
- **Drag a transform inside `transform_interaction`.** A rotation slider,
  or a script that animates a transform, should wrap the drag:

  ```python
  with viewer.transform_interaction(visual):
      for angle in angles:
          visual.transform = rotated(angle)
  ```

  Inside it each change is a tick of one drag (coarse while moving, as
  above), and a visual that starts the drag aligned and tilts stays on the
  brick route until the drag ends, so it never flickers between paths.
  `OrthoViewer.transform_interaction(visuals)` opens it on every panel's
  copy.
- **The first switch.** When a view first moves a visual onto the brick
  route, the visual is blank for about a frame while its brick atlas is
  created. Opening `transform_interaction` creates it ahead of time.
- **Memory.** The atlas is sized by `render_config.gpu_budget_bytes` (1 GiB
  by default; `gpu_budget_bytes_2d` sizes only the aligned 2D tiles). An
  in-memory visual allocates nothing until a view first draws it tilted.
  Each orthoviewer panel's copy has its own atlas, so a volume tilted in
  all three 2D panels can use three budgets.
- **Overlapping visuals.** Give overlapping 2D visuals distinct
  `render_order`; the lower one is drawn on top, as on aligned views.

A tilted slice cannot always be drawn. It draws nothing, with a status
(below), when the slice meets the data along a line (a 2D image seen
edge-on or tilted), or when it runs through more than three data axes (a
4D series whose drift and tilt mix time into the slice).

Points, lines and graph elements in a slice-shift or oblique view are each
placed at their own position, so a thick slab of points under a shear fans
out as it should. On a tilted slab, geometry is selected by exact
containment: give the view a thickness (`controller.update_thickness`), or
a zero-thickness tilted plane selects next to nothing.

## When a view does not draw a visual

A visual a view cannot draw draws nothing rather than something wrong, and:

- `controller.draw_status(visual_id)` returns a `DrawStatus` with the
  `alignment`, `drawable` and a `reason` that says what to do;
- `controller.on_draw_status_changed(scene_id, callback, owner_id=...)` calls
  back with a `DrawStatusChangedEvent` whenever a status changes (a view
  change, a transform change);
- each change into "not drawn" logs one warning on the
  `cellier.draw_status` logger;
- the convenience viewers (`Viewer`, `OrthoViewer`) write
  `Not shown: <name> (<ALIGNMENT>)` in the top-left corner of the canvas
  (and `<name>: <reason>` for a visual drawn with a caveat, such as a slab
  projection that draws its plane instead). Turn
  it off with `Viewer(..., draw_status_overlay=False)`, or hide or restyle
  it through `viewer.draw_status_overlay` (its fields are live).

```python
status = viewer.controller.draw_status(visual.id)
if not status.drawable:
    print(status.alignment, status.reason)
```

A few things are refused when the transform is assigned, because no view
could draw them: a channel axis coupled to any other axis (space and time
may be coupled; channels may not), and a singular transform. `add_*` raises
a `ValueError`; `visual.transform = ...` puts the previous transform back
and raises it too (wrapped in psygnal's `EmitLoopError`, since it is raised
from the model's change event).

## Small numbers are zero

A 90 degree rotation built with `np.cos` and `np.sin` holds `6e-17` where
the exact rotation holds zero, and a float32 pipeline leaves `1e-7`. cellier
treats an entry as zero when it is below `1e-6` relative to both its row and
its column (`cellier.transform.COUPLING_TOLERANCE`), so such a rotation is
the axis-aligned view it is meant to be. The snapped matrix
(`cellier.transform.effective_transform`) is what drawing and slicing use;
your transform is never changed. The worst a snapped entry can move data
is `1e-6` times the axis length: 0.1 voxel across 100,000 voxels.

## Drift correction

A drift is space coupled to time. For a `tzyx` series whose stage drifted
`v` voxels in `x` per frame, the correction is `x - v t`:

```python
matrix = np.eye(5)          # (t, z, y, x) -> (T, Z, Y, X)
matrix[3, 0] = -v           # world x = x - v t
```

Scrub `t` and the data stays put in the world, in 2D and in 3D. The `t`
slider stays discrete (one step per frame) when the store's time axis is
declared `sampling="discrete"`. A `(t, x)` kymograph view of the same
series is "in plane" and draws; a `(t, y)` view slices `x`, which reads
`t`, and is oblique: an image draws it on the brick route.

## Lighting

Volume surfaces (`iso` images, labels in their shaded modes) are lit with
their **world** normal, so anisotropic voxels and sheared data are lit as
the shape they are in the world, not as the voxel grid. Labels on
anisotropic voxels look smoother than they did before this was so.

## Painting

Labels can be painted in a 2D view that is axis aligned, or in plane with a
rotation and a per-axis scale (no shear): the brush is a ball of data voxels
round the cursor. Under a shear, or in a slice-shift or oblique view,
`controller.add_paint_controller` raises, naming the alignment, and a
paint controller made earlier stops painting with a warning.

## Picking and clipping

Picks decode to the data coordinate under any affine, tilted 2D views
included. Clipping planes, render planes (the `"plane"` render mode) and
their gizmos work on rotated and sheared data too; a render plane needs its
three axes in a view the visual draws. In a tilted 2D view the model's
render planes do not apply: the view draws its own slice.

## Projecting a slab

Any 2D view of an image can show the maximum, minimum or mean through the
scene's slice thickness instead of one plane, aligned or tilted; see
[Project a slab](slab_projection.md).
