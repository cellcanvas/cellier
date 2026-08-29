"""WGSL shaders and pygfx materials for cellier render visuals.

Importing this package registers a ``cellier`` WGSL include loader with
pygfx, so cellier shaders can pull in shared snippets the same way they
already pull in pygfx's::

    {$ include 'cellier.label_outline_key.wgsl' $}

That keeps the outline-key logic in one file rather than duplicated across
the four label shaders, which matters because the key must stay in step
with ``random_label_color``'s hash for collisions to be self-concealing.
"""

from pathlib import Path

from pygfx.renderers.wgpu.shader.templating import (
    register_wgsl_loader,
    root_loader,
)

_WGSL_DIR = Path(__file__).parent / "wgsl"


def _load_cellier_wgsl(name: str) -> str:
    """Return the source of a shared cellier WGSL snippet."""
    path = _WGSL_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"no cellier wgsl snippet named {name!r}")
    return path.read_text()


# Guard against a double registration: pygfx raises rather than replacing,
# and a re-import (or a test that reloads the package) should be harmless.
if "cellier" not in root_loader.mapping:
    register_wgsl_loader("cellier", _load_cellier_wgsl)
