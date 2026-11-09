"""The multiscale brick shader's generated WGSL, for the text-identity test.

The ``masked`` template variable and the shared slab include must leave the
plain image shader's generated text unchanged.  :func:`capture` renders a
fixed set of plain multiscale image scenes and records, per template-variable
set the shader was generated with, the SHA-256 of the generated code, with
every template variable it was generated with, so the test can make the code
again from the source alone.  ``brick_shader_text_baseline.json`` holds that
record made at commit ``7e1b683f``, before any MaskedImage change; it must
never be regenerated to make the test pass.

The hash is of the code without its comments (:func:`strip_comments`), so a
comment can be reworded without touching the baseline.  The record first
hashed the full text; its digests were converted once, while the full text
still matched ``7e1b683f`` byte for byte, so they are the comment-free hashes
of that commit's code.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

BASELINE = Path(__file__).parent / "brick_shader_text_baseline.json"

#: ``(name, displayed, render_mode, interpolation, slab_op, tilted)``.
SCENES = [
    ("iso_nearest", "3d", "iso", "nearest", "none", False),
    ("iso_linear", "3d", "iso", "linear", "none", False),
    ("mip_linear", "3d", "mip", "linear", "none", False),
    ("smooth_iso_linear", "3d", "smooth_iso", "linear", "none", False),
    ("attenuated_mip_linear", "3d", "attenuated_mip", "linear", "none", False),
    ("plane_linear", "3d", "plane", "linear", "none", False),
    ("slice_linear", "2d", "iso", "linear", "none", True),
    ("slice_nearest", "2d", "iso", "nearest", "none", True),
    ("slab_mip", "2d", "iso", "linear", "mip", True),
    ("slab_mean", "2d", "iso", "linear", "mean", True),
    ("slab_minip", "2d", "iso", "linear", "minip", True),
]


def strip_comments(code: str) -> str:
    """WGSL *code* without comments, trailing whitespace or blank lines.

    WGSL has no string literals, so ``//`` always starts a line comment.
    Block comments nest, as WGSL allows.  What is left is the code alone: a
    comment that is reworded, re-wrapped, added or deleted leaves it as it is.
    """
    out = []
    depth = 0
    i = 0
    while i < len(code):
        pair = code[i : i + 2]
        if pair == "/*":
            depth += 1
            i += 2
        elif depth and pair == "*/":
            depth -= 1
            i += 2
        elif depth:
            i += 1
        elif pair == "//":
            end = code.find("\n", i)
            i = len(code) if end == -1 else end
        else:
            out.append(code[i])
            i += 1
    lines = (line.rstrip() for line in "".join(out).split("\n"))
    return "\n".join(line for line in lines if line)


def code_hash(code: str) -> str:
    """The SHA-256 of generated WGSL with its comments stripped."""
    return hashlib.sha256(strip_comments(code).encode()).hexdigest()


def generate(source: str, template_vars: dict) -> str:
    """WGSL from *source* as pygfx's ``BaseShader.generate_wgsl`` makes it."""
    from pygfx.renderers.wgpu.shader.resolve import resolve_shadercode
    from pygfx.renderers.wgpu.shader.templating import apply_templating

    return resolve_shadercode(apply_templating(source, **template_vars))


def _json_safe(value):
    """*value* with numpy arrays and scalars made plain Python numbers."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@contextmanager
def recording(shader_class):
    """Record ``[template vars, sha256(code)]`` of each code *shader_class* makes.

    The template variables are every one the code was generated with (the
    shader's own, its bindings' and the renderer's), so the code can be made
    again from the source alone.
    """
    record: list = []
    original = shader_class.generate_wgsl

    def generate_wgsl(self, **more):
        code = original(self, **more)
        if type(self) is shader_class:
            template_vars = {}
            template_vars.update(self._template_vars_init)
            template_vars.update(self._template_vars_bindings)
            template_vars.update(more)
            template_vars["bindings_code"] = self._binding_definitions.get_code()
            template_vars = _json_safe(template_vars)
            json.dumps(template_vars)  # every value must survive the file
            # The stored variables must make the same code again.
            again = generate(self.get_code(), template_vars)
            assert code_hash(again) == code_hash(code), "variables do not round trip"
            record.append([template_vars, code_hash(code)])
        return code

    shader_class.generate_wgsl = generate_wgsl
    try:
        yield record
    finally:
        shader_class.generate_wgsl = original


def _render_scene(tmp_path: Path, scene: tuple) -> None:
    from cellier.transform import AffineTransform
    from cellier.visuals import (
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageSingleAppearance,
        RenderPlane,
    )
    from tests._gpu_budget import SMALL_BUDGETS
    from tests.render._affine import (
        load,
        multiscale_store,
        new_viewer,
        render,
        rot,
        world_matrix,
    )

    name, displayed, render_mode, interpolation, slab_op, tilted = scene
    viewer = new_viewer(displayed)
    try:
        rng = np.random.default_rng(0)
        data = rng.random((24, 32, 40)).astype(np.float32)
        store = multiscale_store(tmp_path, name, data)
        viewer.controller._ensure_data_coordinate_systems(viewer.scene.id, store)
        matrix = world_matrix(rot(0, 2, 30.0) if tilted else np.eye(3))
        world = viewer.scene.dims.world_coordinate_system
        transform = AffineTransform.from_matrix(
            matrix, store.data_coordinate_systems[0], world
        )
        planes = ()
        if render_mode == "plane":
            planes = (
                RenderPlane.from_point_normal(world, (12.0, 16.0, 20.0), (1, 0, 0)),
            )
        viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(
                interpolation=interpolation,
                slab_projection=slab_op,
                projection_axes=("z",) if slab_op != "none" else (),
            ),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                render_mode=render_mode, clim=(0.0, 1.0), iso_threshold=0.5
            ),
            transform=transform,
            render_planes=planes,
        )
        if slab_op != "none":
            viewer.controller.update_thickness(viewer.scene.id, {0: 3.0})
        load(viewer, slice_z=11.0 if displayed == "2d" else None)
        render(viewer, size=128)
    finally:
        viewer.controller.close()


def capture(tmp_path: Path) -> dict[str, list]:
    """``scene name -> [[template vars, sha256], ...]`` for every scene.

    Run each scene in a fresh process: pygfx caches pipelines by template
    variables, so a scene whose variables were seen before generates nothing.
    """
    from cellier.render.shaders._multiscale_volume_brick import (
        MultiscaleVolumeBrickShader,
    )

    out: dict[str, list] = {}
    for scene in SCENES:
        with recording(MultiscaleVolumeBrickShader) as record:
            _render_scene(tmp_path, scene)
        out[scene[0]] = list(record)
    return out


if __name__ == "__main__":
    import sys
    import tempfile

    if BASELINE.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{BASELINE} exists: it must never be regenerated")
    with tempfile.TemporaryDirectory() as tmp:
        result = capture(Path(tmp))
    for name, record in result.items():
        print(name, len(record))
    BASELINE.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
