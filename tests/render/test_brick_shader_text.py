"""The plain multiscale brick shader's text is unchanged.

``brick_shader_text_baseline.json`` records, for a fixed set of plain image
scenes made at commit ``7e1b683f``, every template variable the shader was
generated with and the SHA-256 of the code.  Templating the current source
with those variables (``masked`` false) must make the same code: the masked
branch and the shared includes may not change a plain image's shader.
"""

from __future__ import annotations

import json

import pytest

from tests.render._brick_shader_text import BASELINE, code_hash, generate


def _entries():
    record = json.loads(BASELINE.read_text())
    return [
        pytest.param(template_vars, digest, id=f"{scene}-{i}")
        for scene, entries in sorted(record.items())
        for i, (template_vars, digest) in enumerate(entries)
    ]


@pytest.mark.parametrize(("template_vars", "digest"), _entries())
def test_plain_brick_shader_text_is_unchanged(template_vars, digest):
    from cellier.render.shaders._multiscale_volume_brick import _WGSL_SOURCE

    template_vars = {"masked": False, **template_vars}
    assert code_hash(generate(_WGSL_SOURCE, template_vars)) == digest
