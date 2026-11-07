# Project a slab

A 2D view of an image normally draws one plane: the slice at the slider
position. With a slab projection it draws, at every pixel, the maximum,
minimum or mean of the image through a slab of the scene -- a maximum
intensity projection over the slice thickness. It works in every 2D view,
aligned with the data or tilted through it, for in-memory and multiscale
images.

## The short version

```python
from cellier.visuals import MultiscaleImageAppearance

visual = viewer.add_image_multiscale(
    store,
    appearance=MultiscaleImageAppearance(
        slab_projection="mip",      # "mip", "minip" or "mean"; "none" draws the plane
        projection_axes=("z",),     # the world axis to project along
        coarsest_while_moving_2d=True,
    ),
)
# The slab is the scene's thickness on that axis: slice +- 8 world units.
viewer.controller.update_thickness(viewer.scene.id, {0: 8.0})
```

The same three fields are on `InMemoryImageAppearance`. They are live: set
`visual.appearance.slab_projection = "mean"` and the view redraws.

## The fields

| Field | Meaning |
|---|---|
| `slab_projection` | `"mip"` (maximum), `"minip"` (minimum), `"mean"`, or `"none"` (the default: one plane) |
| `projection_axes` | the world axis to project along, by name, e.g. `("z",)`; empty (the default) projects nothing |
| `projection_steps_per_voxel` | samples per voxel along a ray, 0.25 to 8; default 1 |

The slab is **the scene's thickness**, not a field of the image:
`dims.selection.thickness` holds a half-thickness per world axis, and the
projection reduces `[slice - h, slice + h)` on its axis. Every visual of the
scene sees the same slab -- points and meshes are selected by it too -- but
each image decides for itself whether it projects. A labels visual never
projects, so labels painted over a projecting image stay one plane (and
paintable, in an aligned view).

A thickness of zero draws the plane, as without a projection.

## What is reduced

- In a view aligned with the data, a ray runs along one data axis: the
  projection reduces the voxels whose centres lie in the slab, each once.
  This is exact.
- In a tilted view (the data is rotated or sheared out of the slice), a ray
  crosses several data axes. It is sampled one voxel apart along the ray,
  starting at the slice; with nearest interpolation a sample counts when
  the centre of the voxel it reads is in the slab, with linear when the
  sample is. Samples outside the data are skipped; a ray with none draws
  nothing.
- The level of detail is the one the plane would draw at this zoom, so a
  coarse level samples a thin slab coarsely, and values may change as the
  view refines. A slab thinner than one voxel of the drawn level -- a thin
  slab while a drag shows the coarse level -- reads the one voxel its
  centre is in.
- `mean` of integer data is a float; the contrast limits and colormap apply
  to the projected value. In a composite each channel projects on its own,
  then the channels blend.
- A pick on a `mip` or `minip` reports the data coordinate of the winning
  sample; on a `mean`, the point on the slice.

## The thickness box

The dims panel's `+/-` box next to a slider sets that axis's half-thickness.
Stepping it -- its arrows, the arrow keys, the mouse wheel -- is a drag: with
`coarsest_while_moving_2d` on, each step loads only the coarse level and the
full level loads once the box has been still for a moment. Typing a value
is a jump and loads in full at once. From a script,
`controller.update_thickness(scene_id, {axis: h}, interactive=True)` is a
step, and every change inside `controller.dims_interaction(scene_id)` too.

## When the plane is drawn instead

The image draws its plane, and its draw status
(`controller.draw_status(visual.id)`) and the canvas overlay say why, when:

| Setting | Status |
|---|---|
| `projection_axes` names more than one axis | `projects over one axis only; drawing the slice` |
| it names a displayed axis | `cannot project along a displayed axis` |
| it names an axis the world does not have | `no world axis named <axis>` |

A slab over a 2D image that is tilted or seen edge-on draws nothing, with
`this view cuts the image along a line; nothing to draw`. A projection along
a world axis the image does not reach (a 2D image in a 3D world) draws the
plane: every sample on a ray would read the same value.

## Performance

A projection draws from the image's 3D bricks, like a tilted view (see
[Rotate, shear and drift data](affine_transforms.md)): the brick atlas is
sized by `render_config.gpu_budget_bytes`, the first projection of a visual
may show it blank for about a frame, and interactive use wants
`coarsest_while_moving_2d=True`. A thick slab reads every brick it crosses,
so its first load takes longer than a plane's.
