# Capture screenshots

Every screenshot cellier takes is rendered on a **dedicated offscreen canvas**,
never read back from the canvas on screen. That one decision is what makes the
result reproducible: two captures of the same viewer state produce
byte-identical arrays, at exactly the size you asked for, with or without a
window open, whichever GUI toolkit the viewer targets.

## The short version

```python
frame = viewer.screenshot(size=(1600, 1200))          # RGBA uint8, (1200, 1600, 4)
viewer.screenshot(size=(1600, 1200), save="figure.png")
```

`frame` is an `(height, width, 4)` `uint8` array. `save=` writes a PNG using
only the standard library, so it never fails for want of an image package.

Not using the convenience API? Everything here works on
`CellierController` directly -- see [Without the convenience
API](#without-the-convenience-api).

## Capture what is loaded, not what is loading

A capture shows the data **currently resident on the GPU**. It does not
reslice, so capture after the load has finished:

```python
viewer.on_ready(lambda: viewer.screenshot(save="startup.png"))
launch(viewer, layout)
```

Two consequences worth knowing up front:

- **Slice requests are planned per canvas.** A viewer that has never called
  `add_canvas` has never requested any data, so capturing it returns a
  correctly-rendered picture of an empty scene. Add a canvas first.
- **A larger `scale` enlarges the level of detail already loaded**; it does not
  fetch a finer one. A 4x capture of a multiscale volume shows the on-screen
  level, bigger.

## Sharp images: the `frames` argument

| `frames=` | What you get |
| --- | --- |
| `1` (default) | temporal accumulation off, one draw. Crisp and instant. |
| `"converged"` | accumulate for as long as the accumulator needs to settle. |
| `N` | accumulate exactly N frames. |

Use `frames="converged"` whenever ambient occlusion is on -- a single-sample AO
frame is visibly noisy, and the accumulation pass is what averages the
per-frame kernel rotation away:

```python
viewer.ambient_occlusion_enabled = True
viewer.screenshot(size=(1600, 1200), frames="converged", save="ao.png")
```

`"converged"` does **not** watch the pixels and wait for them to stop moving;
it computes how many frames the accumulator needs. Each frame leaves
`1 - temporal_blend_weight` of the error, so reaching a 1% residual takes
`log(0.01) / log(1 - blend_weight)` frames -- 44 at the default weight of 0.1,
7 at 0.5, 228 at 0.02. (Watching the pixels does not work: with ambient
occlusion on, a few hundred pixels dither forever and the image never becomes
literally still. `cellier.render._capture.frames_to_settle` records the
measurements.)

Two knobs follow from that:

- `residual=` -- how close to settled is close enough. Default `0.01`.
- `max_frames=` -- the ceiling, 64 by default. A blend weight below ~0.07
  needs more, and asking for `"converged"` there raises with the exact number
  rather than returning an unsettled image.

```python
viewer.temporal_blend_weight = 0.02
viewer.screenshot(frames="converged", max_frames=300)   # 228 frames
```

## Several canvases, and orthoviewers

`Viewer` and `OrthoViewer` address canvases differently, because they select
different things. On a `Viewer` the argument picks a **viewpoint onto one
scene**; on an `OrthoViewer` it picks **which of four scenes** to render.

```python
viewer.canvases                                  # tuple[UUID, ...]
viewer.screenshot()                              # one canvas: uses it
                                                 # several: raises, naming them
viewer.screenshot(canvas=viewer.canvases[1])     # pick a viewpoint

ortho.screenshot(panel="xy", size=(800, 800))    # one panel -> (800, 800, 4)
ortho.screenshot(size=(800, 800))                # 2x2 grid  -> (1600, 1600, 4)
```

The orthoviewer grid is laid out the way the on-screen grid is -- XY and XZ on
the top row, YZ and the 3D volume below -- and `size` is **per panel**. It is
canvases only: the `XY` / `XZ` / `YZ` / `3D` titles are chrome and do not
appear. `frames="converged"` matters more here than elsewhere, because the
`vol` panel accumulates while the three slice panels do not.

## Without the convenience API

`CellierController.screenshot` is the primary entry point, not a fallback:
`Viewer.screenshot`, `OrthoViewer.screenshot` and `screenshot_window` are all
thin wrappers over it. Everything above is available at the controller level;
what the convenience layer adds is addressing and a `save=` argument, not
capability.

The body below is a coroutine -- `on_scene_ready` is a callback, and awaiting
it needs an event loop:

```python
import asyncio

from cellier.controller import CellierController
from cellier.render._capture import write_png

controller = CellierController(gui="offscreen")
scene = controller.add_scene(dim="3d", name="scene")
controller.add_image(data=store, scene_id=scene.id, appearance=appearance)

# Required: slice requests are planned per canvas, so a scene without one
# never loads anything and every capture of it is a picture of emptiness.
controller.add_canvas(scene_id=scene.id, canvas_size=(400, 300))

# on_scene_ready is the supported wait; the convenience on_ready wraps it.
ready = asyncio.Event()
controller.on_scene_ready(scene.id, ready.set)
await ready.wait()
controller.fit_camera(scene.id)

canvas_id = controller.get_canvas_ids(scene.id)[0]
frame = controller.screenshot(canvas_id, size=(400, 300))   # (300, 400, 4) uint8
write_png("shot.png", frame)
```

### The two methods

```python
controller.screenshot(canvas_id, size=None, scale=1.0, *, frames=1, **kwargs)
controller.screenshot_scene(scene_id, size=None, scale=1.0, *, frames=1, dim=None, **kwargs)
```

`screenshot` copies *canvas_id*'s camera, dimensionality and depth range, then
renders its own frame -- the argument names a **viewpoint, not a surface**.
`size` defaults to that canvas's physical size, so an unqualified call
reproduces the on-screen framing.

`screenshot_scene` fits the camera to the scene instead of copying one, for
when a scene has no canvas or when no existing canvas has the viewpoint you
want. `dim` (`"2d"` / `"3d"`) is inferred from the scene's displayed axes.

Both take the same `frames`, `max_frames` and `residual` arguments described
above, and both return an `(height, width, 4)` `uint8` array.

### Finding canvases

```python
controller.get_canvas_ids(scene_id)   # canvases on one scene, creation order
controller.canvas_ids                 # every canvas, across all scenes
controller.get_canvas_view(canvas_id) # the render-layer CanvasView
```

`get_canvas_ids` is the one to reach for. `canvas_ids` exists for code that
must find canvases without knowing their scene -- it is what
`screenshot_window` walks to discover which canvases live inside a window.

There is no raise-on-ambiguity here: a controller-level call names its canvas
explicitly, so there is nothing to disambiguate. That guard belongs to
`Viewer.screenshot`, which is allowed to omit the argument.

### Writing the PNG

`save=` is a convenience-layer argument. At the controller level, call the
writer directly:

```python
from cellier.render._capture import write_png

write_png("shot.png", controller.screenshot(canvas_id))
```

It takes an `(h, w, 4)` `uint8` array, refuses anything else, and is built on
`zlib` alone -- so it cannot fail for want of Pillow or imageio, neither of
which cellier depends on. It currently lives in a private module; treat the
import as provisional.

## A whole window, chrome included

Qt's `QWidget.grab()` captures docks, labels and sliders faithfully but cannot
see a render canvas -- a wgpu surface is not part of Qt's paint pipeline, so the
canvas rectangles come back as flat fill. `screenshot_window` assembles the
picture instead: Qt supplies the chrome, and each canvas rectangle is filled in
by an offscreen capture.

```python
from cellier.convenience import screenshot_window

window = launch(viewer, layout)
screenshot_window(window, viewer.controller, save="window.png")
```

Only the canvas regions are reproducible; chrome depends on the platform's
widget style and font rendering. On a real display the OS screenshot tool does
this job too -- `screenshot_window` exists for the headless case
(`QT_QPA_PLATFORM=offscreen`), where there is no screen to shoot.

There is no window capture for notebook front ends: nothing on the Python side
can reach the browser's compositor. Canvas capture works there exactly as it
does under Qt.

## Headless, from the command line

`cellier.convenience.capture` turns a viewer into a PNG with no display attached --
useful in CI, and the fastest way for an agent to actually look at a render
change:

```bash
.venv/bin/python -m cellier.convenience.capture my_demo.py --size 900x700 --frames converged --out /tmp/shot.png
```

The target is either a Python file exposing `build()` that returns a populated
`Viewer` or `OrthoViewer` (it must not call `launch`/`show`), or a file written
by `Viewer.to_file`. The script gives every scene a canvas, drives the slicer to
quiescence, fits the cameras, and captures -- so the picture shows loaded data
rather than an empty scene.

```python
# my_demo.py
def build():
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="offscreen")
    viewer.add_image(data=store, appearance=InMemoryImageAppearance())
    viewer.set_displayed_dimensions(("z", "y", "x"))
    return viewer
```

## The `offscreen` gui

`gui="offscreen"` builds a viewer with no window at all -- the right choice for
scripts and tests that only ever capture:

```python
viewer = Viewer(spatial_axes("z", "y", "x"), gui="offscreen")
```

It has no embeddable widget, so `build_canvas_widget`, `launch`, `show` and
`display` reject it and say so. Capture does not require it: `screenshot()`
renders offscreen whatever gui the viewer targets, so a Qt or notebook viewer
captures identically.

## What the contract does not promise

Identical bytes across different GPUs or driver versions. Assertions in tests
should stay structural -- an alpha mask, a hue, a mean -- as the render suite's
own tests do.
