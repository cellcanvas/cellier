"""Time single wgpu calls on this machine's adapter, with no rendering.

Run on every CI platform.  In the slow render tests on Windows the time is
inside three wgpu calls, not in drawing as such: ``queue.submit`` (3.5 times
Ubuntu's), ``create_texture`` (5 times) and ``create_view`` (8 times), while
a plain frame is only 1.2 to 1.5 times slower
(``scripts/ci_profiling/ci_profiling_investigation.md``).  This probe times
those calls alone, with wgpu directly (no pygfx scene, no cellier), to find
which of them the driver is slow at and whether it grows with texture size.

``create_texture``
    Creating (and destroying) a texture, over a range of sizes up to the
    brick caches' (a 3D float texture of tens of MB).
``first_use``
    Writing one 32^3 brick into a fresh 3D texture and reading a texel
    back, against doing the same a second time.  wgpu clears a texture the
    first time part of it is used, so the difference is the cost of that
    clear: it lands in ``write_texture`` or ``submit``, not in
    ``create_texture``.
``render_target``
    What a new renderer costs: creating the colour, pick and depth targets
    of a 1920 x 1920 frame (the size ``test_multiscale_level_alignment.py``
    draws at, 88 times over), clearing them in a first pass, clearing them
    again, and destroying them.  Unlike the textures above these have the
    ``RENDER_ATTACHMENT`` usage, and a software rasteriser keeps them in
    main memory: 59 MB a set.
``create_view``
    Creating a view of a texture.
``submit``
    Submitting an empty command buffer, a render pass that only clears a
    320 x 320 colour and depth target, and the same followed by reading a
    pixel back (which is what a pick does).
``write_texture`` / ``write_buffer``
    Uploading one 32^3 float brick; writing a 256 byte uniform buffer.

Every figure is the median over the repeats, in milliseconds per call.

Usage
-----
    python scripts/ci_profiling/probe_gpu_calls.py --out probe_gpu_calls.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

BRICK = 32

#: ``label -> (size, dimension, format, bytes per texel)``.
TEXTURES = {
    "2d 320x320 rgba8": ((320, 320, 1), "2d", "rgba8unorm", 4),
    "2d 2048x2048 rgba8": ((2048, 2048, 1), "2d", "rgba8unorm", 4),
    "3d 64^3 r32float": ((64, 64, 64), "3d", "r32float", 4),
    "3d 128^3 r32float": ((128, 128, 128), "3d", "r32float", 4),
    "3d 256^3 r32float": ((256, 256, 256), "3d", "r32float", 4),
    "3d 256^3 r8": ((256, 256, 256), "3d", "r8unorm", 1),
}


def _ms(function, repeats: int) -> dict:
    """Call *function* *repeats* times; median, min and max in ms."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": round(statistics.median(times), 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "repeats": repeats,
    }


def measure(repeats: int) -> dict:
    import wgpu
    from pygfx.renderers.wgpu import get_shared

    # The device the render tests draw with.
    device = get_shared().device
    queue = device.queue
    usage = wgpu.TextureUsage
    result = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "adapter": device.adapter.summary,
        "calls": {},
    }
    calls = result["calls"]

    def read_texel(texture) -> None:
        """Block until everything submitted so far has run."""
        queue.read_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": 256, "rows_per_image": 1},
            (1, 1, 1),
        )

    # --- create_texture, by size ------------------------------------------
    for label, (size, dimension, texture_format, n_bytes) in TEXTURES.items():
        megabytes = size[0] * size[1] * size[2] * n_bytes / 1e6
        made = []

        def create(size=size, dimension=dimension, fmt=texture_format, made=made):
            made.append(
                device.create_texture(
                    size=size,
                    dimension=dimension,
                    format=fmt,
                    usage=usage.COPY_DST | usage.COPY_SRC | usage.TEXTURE_BINDING,
                )
            )

        def destroy(made=made):
            made.pop().destroy()

        # One at a time, so the largest never holds more than one copy.
        create_times, destroy_times = [], []
        for _ in range(min(repeats, 20)):
            create_times.append(_ms(create, 1)["median_ms"])
            destroy_times.append(_ms(destroy, 1)["median_ms"])
        calls[f"create_texture {label}"] = {
            "median_ms": round(statistics.median(create_times), 4),
            "max_ms": round(max(create_times), 4),
            "megabytes": round(megabytes, 1),
        }
        calls[f"destroy_texture {label}"] = {
            "median_ms": round(statistics.median(destroy_times), 4),
            "max_ms": round(max(destroy_times), 4),
            "megabytes": round(megabytes, 1),
        }

    # --- first use of a 3D texture: the clear wgpu owes it ----------------
    brick = np.random.default_rng(0).random((BRICK,) * 3, dtype=np.float32)

    def write_brick(texture) -> None:
        queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            brick,
            {"offset": 0, "bytes_per_row": 4 * BRICK, "rows_per_image": BRICK},
            (BRICK, BRICK, BRICK),
        )

    for edge in (64, 128, 256):
        first, second = [], []
        for _ in range(5):
            texture = device.create_texture(
                size=(edge, edge, edge),
                dimension="3d",
                format="r32float",
                usage=usage.COPY_DST | usage.COPY_SRC | usage.TEXTURE_BINDING,
            )

            def use(texture=texture):
                write_brick(texture)
                read_texel(texture)

            first.append(_ms(use, 1)["median_ms"])
            second.append(_ms(use, 1)["median_ms"])
            texture.destroy()
        megabytes = round(edge**3 * 4 / 1e6, 1)
        calls[f"first_use 3d {edge}^3 r32float"] = {
            "median_ms": round(statistics.median(first), 4),
            "max_ms": round(max(first), 4),
            "megabytes": megabytes,
        }
        calls[f"second_use 3d {edge}^3 r32float"] = {
            "median_ms": round(statistics.median(second), 4),
            "max_ms": round(max(second), 4),
            "megabytes": megabytes,
        }

    # --- a renderer's targets: create, first clear, second clear, destroy --
    target_usage = usage.RENDER_ATTACHMENT | usage.COPY_SRC | usage.TEXTURE_BINDING
    for edge in (320, 1920):
        timings = {"create": [], "first_clear": [], "second_clear": [], "destroy": []}
        for _ in range(10):
            start = time.perf_counter()
            targets = [
                device.create_texture(
                    size=(edge, edge, 1),
                    dimension="2d",
                    format=texture_format,
                    usage=target_usage,
                )
                for texture_format in ("rgba8unorm-srgb", "rgba16uint", "depth32float")
            ]
            timings["create"].append((time.perf_counter() - start) * 1000)

            def clear(targets=targets):
                encoder = device.create_command_encoder()
                encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": target.create_view(),
                            "resolve_target": None,
                            "clear_value": (0.0, 0.0, 0.0, 0.0),
                            "load_op": "clear",
                            "store_op": "store",
                        }
                        for target in targets[:2]
                    ],
                    depth_stencil_attachment={
                        "view": targets[2].create_view(),
                        "depth_clear_value": 1.0,
                        "depth_load_op": "clear",
                        "depth_store_op": "store",
                    },
                ).end()
                queue.submit([encoder.finish()])
                read_texel(targets[0])

            timings["first_clear"].append(_ms(clear, 1)["median_ms"])
            timings["second_clear"].append(_ms(clear, 1)["median_ms"])
            start = time.perf_counter()
            for target in targets:
                target.destroy()
            timings["destroy"].append((time.perf_counter() - start) * 1000)
        megabytes = round(edge * edge * 16 / 1e6, 1)
        for step, times in timings.items():
            calls[f"render_target {edge}x{edge} {step}"] = {
                "median_ms": round(statistics.median(times), 4),
                "max_ms": round(max(times), 4),
                "megabytes": megabytes,
            }

    # --- create_view ------------------------------------------------------
    colour = device.create_texture(
        size=(320, 320, 1),
        dimension="2d",
        format="rgba8unorm",
        usage=usage.RENDER_ATTACHMENT | usage.COPY_SRC | usage.TEXTURE_BINDING,
    )
    depth = device.create_texture(
        size=(320, 320, 1),
        dimension="2d",
        format="depth32float",
        usage=usage.RENDER_ATTACHMENT,
    )
    calls["create_view 2d 320x320"] = _ms(colour.create_view, repeats * 5)

    # --- submit -----------------------------------------------------------
    colour_view = colour.create_view()
    depth_view = depth.create_view()

    def submit_empty():
        queue.submit([device.create_command_encoder().finish()])

    def submit_clear():
        encoder = device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": colour_view,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
            depth_stencil_attachment={
                "view": depth_view,
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )
        render_pass.end()
        queue.submit([encoder.finish()])

    def submit_clear_and_read():
        submit_clear()
        read_texel(colour)

    calls["submit empty"] = _ms(submit_empty, repeats)
    read_texel(colour)
    calls["submit clear pass 320x320"] = _ms(submit_clear, repeats)
    read_texel(colour)
    calls["submit clear pass + read a pixel"] = _ms(submit_clear_and_read, repeats)
    calls["read a pixel"] = _ms(lambda: read_texel(colour), repeats)

    # --- uploads ----------------------------------------------------------
    cache = device.create_texture(
        size=(128, 128, 128),
        dimension="3d",
        format="r32float",
        usage=usage.COPY_DST | usage.COPY_SRC | usage.TEXTURE_BINDING,
    )
    write_brick(cache)
    read_texel(cache)
    calls["write_texture 32^3 r32float brick"] = _ms(
        lambda: write_brick(cache), repeats
    )
    read_texel(cache)

    uniform = device.create_buffer(
        size=256, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )
    payload = bytes(256)
    calls["write_buffer 256 bytes"] = _ms(
        lambda: queue.write_buffer(uniform, 0, payload), repeats * 5
    )
    read_texel(colour)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--out", help="write the results here as JSON")
    args = parser.parse_args()

    result = measure(args.repeats)
    print(f"adapter: {result['adapter']}")
    print(f"{'call':42s} {'median ms':>10} {'max ms':>10} {'MB':>7}")
    for name, row in result["calls"].items():
        megabytes = f"{row['megabytes']:7.1f}" if "megabytes" in row else ""
        print(f"{name:42s} {row['median_ms']:10.3f} {row['max_ms']:10.3f} {megabytes}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
