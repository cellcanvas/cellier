# Set the orientation of the 3D camera

When cellier fits the 3D camera to a scene it looks along minus all three
displayed axes, with the first displayed axis up. Data stored the other way
up, a scan whose top is at low `z`, say, then appears upside down. A canvas
takes the view direction and the up vector of its fits as a setting, so the
data opens the right way up and stays that way.

## The short version

```python
canvas_widget = build_canvas_widget(
    viewer,
    axis_values_from_viewer(viewer),
    fit_view_direction_3d={"z": 1.0, "y": -1.0, "x": -1.0},
    fit_up_3d={"z": -1.0},
)
```

- **`fit_up_3d`** is the direction that points up on screen.
- **`fit_view_direction_3d`** is the direction the camera looks along, from
  the camera towards the scene.

Either can be given without the other. The one left out keeps its default.

## Vectors are given on named axes

Each vector is a mapping of world axis to component. The key is an axis name
or an axis id, and an axis left out has component 0:

```python
fit_up_3d={"z": -1.0}                               # up is minus z
fit_view_direction_3d={"z": 1.0, "y": -1.0, "x": -1.0}
```

There is no tuple form. A bare `(0, 0, -1)` does not say which axis each
entry is on, and the renderer's own order is the reverse of the world's, so a
tuple raises a `TypeError` instead of being guessed at.

The length of a vector does not matter, only its direction.

## Where the setting goes

The same two keywords are on every function that makes a canvas:

| You call | With |
|---|---|
| `build_canvas_widget(viewer, ...)` | `fit_view_direction_3d=`, `fit_up_3d=` |
| `viewer.add_canvas(...)` | the same |
| `controller.add_canvas(scene_id, ...)` | the same |
| `build_ortho_grid_widget(ortho, ...)` | the same; only the 3D panel uses them |

The setting is fixed when the canvas is made. It is stored on the `Canvas`
model, by axis id, so a saved viewer keeps it.

## When it is used

Every fit of the 3D camera uses it:

- `viewer.fit_camera()` and `controller.fit_camera(scene_id)`;
- the fit `run(viewer, layout, fit="ready")` makes once the first view has
  loaded;
- the first time a canvas that started in 2D switches to 3D.

Orbiting afterwards keeps the up vector: the orbit controller turns the
camera about it.

The 2D camera is not affected. It always looks down on its plane.

## When the displayed axes do not match

The vectors are on world axes, but a 3D view shows three *displayed* axes,
and those can change. A vector is used only while **every axis it has a
component on is displayed**. Otherwise that vector takes its default, with no
warning:

```python
# World (t, z, y, x), and the canvas was made with fit_up_3d={"z": -1.0}.
viewer.set_displayed_dimensions(("z", "y", "x"))
viewer.fit_camera()                                # up is minus z

viewer.set_displayed_dimensions(("t", "y", "x"))
viewer.fit_camera()                                # z is not shown: up is t
```

Changing which three axes are displayed does not move the camera by itself.
The orientation for the new axes is used by the next fit.

A vector is never projected onto the axes that are shown. Dropping the `z`
part of `{"z": 1.0, "y": -1.0, "x": -1.0}` would leave a view along the `yx`
diagonal, which is not what was asked for.

Each vector falls back on its own. One more case takes the default for both:
the view direction and the up vector in use are parallel, which leaves the
camera's roll undefined. That can happen when only one is given. A view
direction of `{"z": -1.0}` alone is parallel to the default up when `z` is
the first displayed axis, so give an up vector with it:

```python
fit_view_direction_3d={"z": -1.0}, fit_up_3d={"y": 1.0}
```

## What is refused

These raise when the canvas is made:

| Given | Raises |
|---|---|
| A tuple or list instead of a mapping | `TypeError` |
| An axis the world does not have | `KeyError` |
| A vector with no non-zero component, or one that is not finite | `ValueError` |
| A view direction and an up vector that are parallel | `ValueError` |

## One fit with another orientation

`look_at_visual` on the controller fits the camera once and takes the same
two vectors:

```python
viewer.controller.look_at_visual(
    visual.id,
    canvas_id,
    view_direction={"x": 1.0},
    up={"z": -1.0},
)
```

A vector left out, or one that cannot be used on the displayed axes, is the
canvas's own setting, and the default after that. In a 2D view both are
ignored.

This does not change the canvas's setting: the next `fit_camera` goes back
to it.

## Why not a transform

Turning the data over with each visual's transform also works, but it
changes more than the view:

- Negating one axis is a mirror image. A left lung becomes a right lung.
- A half turn avoids the mirror, but it turns the 2D slices over as well.
- World coordinates stop being the data's own. Slice positions, planes and
  anything already in world coordinates need the same change.

The fit orientation moves only the camera.
