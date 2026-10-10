"""Time offscreen frames of two fixed pygfx scenes on this machine's adapter.

Run on every CI platform to compare the cost of a frame where nothing but
the GPU driver differs.  It uses pygfx alone (no cellier), so a difference
between platforms here is the driver's, not ours.

Two scenes are drawn at the size the render tests use:

``mesh``
    A lit sphere.  Cheap fragments; mostly per-frame overhead.
``volume``
    A 96^3 volume, MIP ray cast.  Expensive fragments; mostly rasteriser time.

For each scene the probe reports the first frame (which builds the shader
and the pipeline) and the median of the frames after it.

With ``--thread-sweep`` the probe runs itself once per ``LP_NUM_THREADS``
value, which sets how many threads llvmpipe rasterises with.  The variable
is read when the driver loads, hence the subprocesses.  If the frame time
does not change with the thread count, the driver is not using its threads.

Usage
-----
    python scripts/ci_profiling/probe_render.py --out probe_render.json
    python scripts/ci_profiling/probe_render.py --thread-sweep default,1,2,4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

SIZE = (320, 320)


def _scenes():
    import numpy as np
    import pygfx as gfx

    mesh_scene = gfx.Scene()
    mesh_scene.add(gfx.AmbientLight(), gfx.DirectionalLight())
    mesh_scene.add(
        gfx.Mesh(
            gfx.sphere_geometry(radius=40, width_segments=64, height_segments=32),
            gfx.MeshPhongMaterial(color="#3a7"),
        )
    )

    z, y, x = np.indices((96, 96, 96), dtype=np.float32)
    data = (np.sin(x / 7) * np.sin(y / 9) * np.sin(z / 11) * 0.5 + 0.5).astype(
        np.float32
    )
    volume_scene = gfx.Scene()
    volume_scene.add(
        gfx.Volume(
            gfx.Geometry(grid=gfx.Texture(data, dim=3)),
            gfx.VolumeMipMaterial(clim=(0.0, 1.0), map=gfx.cm.viridis),
        )
    )
    return {"mesh": mesh_scene, "volume": volume_scene}


def measure(n_frames: int) -> dict:
    """Draw each scene *n_frames* times and return the timings in ms."""
    import numpy as np
    import pygfx as gfx
    from pygfx.renderers.wgpu import get_shared
    from rendercanvas.offscreen import RenderCanvas

    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "LP_NUM_THREADS": os.environ.get("LP_NUM_THREADS", "default"),
        "adapter": get_shared().device.adapter.summary,
        "size": list(SIZE),
        "scenes": {},
    }
    for name, scene in _scenes().items():
        canvas = RenderCanvas(size=SIZE, pixel_ratio=1)
        renderer = gfx.WgpuRenderer(canvas)
        camera = gfx.PerspectiveCamera(50)
        camera.show_object(scene, view_dir=(-0.5, -0.4, -1.0))
        canvas.request_draw(lambda r=renderer, s=scene, c=camera: r.render(s, c))

        times = []
        for index in range(n_frames):
            # Turn the camera a little so no frame can be reused.
            camera.local.x += 0.01 * index
            start = time.perf_counter()
            frame = np.asarray(canvas.draw())
            times.append((time.perf_counter() - start) * 1000)
        if not frame[..., 3].any():
            raise RuntimeError(f"the {name} scene drew nothing")
        steady = times[1:]
        result["scenes"][name] = {
            "first_frame_ms": round(times[0], 2),
            "steady_median_ms": round(statistics.median(steady), 2),
            "steady_min_ms": round(min(steady), 2),
            "steady_max_ms": round(max(steady), 2),
            "n_frames": n_frames,
        }
        canvas.close()
    return result


def _print(result: dict) -> None:
    print(f"adapter: {result['adapter']}")
    print(
        f"cpu_count: {result['cpu_count']}  LP_NUM_THREADS: {result['LP_NUM_THREADS']}"
    )
    for name, row in result["scenes"].items():
        print(
            f"  {name:7s} first {row['first_frame_ms']:9.1f} ms   "
            f"steady median {row['steady_median_ms']:8.2f} ms   "
            f"(min {row['steady_min_ms']:.2f}, max {row['steady_max_ms']:.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", help="write the results here as JSON")
    parser.add_argument(
        "--thread-sweep",
        help="comma-separated LP_NUM_THREADS values; 'default' leaves it unset",
    )
    parser.add_argument("--json-stdout", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.json_stdout:
        print(json.dumps(measure(args.frames)))
        return

    if args.thread_sweep:
        results = []
        for threads in args.thread_sweep.split(","):
            env = dict(os.environ)
            env.pop("LP_NUM_THREADS", None)
            if threads != "default":
                env["LP_NUM_THREADS"] = threads
            done = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--json-stdout",
                    "--frames",
                    str(args.frames),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0:
                print(f"LP_NUM_THREADS={threads} failed:\n{done.stderr}")
                continue
            result = json.loads(done.stdout.strip().splitlines()[-1])
            _print(result)
            results.append(result)
    else:
        results = [measure(args.frames)]
        _print(results[0])

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
