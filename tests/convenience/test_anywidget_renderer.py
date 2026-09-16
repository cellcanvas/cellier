"""Tests for ``cellier.convenience.layout._anywidget_renderer``.

The real-host happy paths are covered by ``tests/v2/test_anywidget.py``; this
fills the parity/guard gaps: the appearance-controls None-guards, the center
recursion (via a fake host), and ``_RenderView`` teardown.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import AppearanceControls
from cellier.convenience.layout._walk import render_dock

pytest.importorskip("anywidget")

from cellier.convenience import Viewer
from cellier.convenience._hosts import JupyterHost
from cellier.convenience.layout._anywidget_renderer import _RenderView
from cellier.convenience.layout._spec import Grid, HStack, VStack
from cellier.convenience.layout._walk import render_center
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


class _FakeHost:
    """Records composition calls for the center-recursion tests."""

    def leaf(self, widget):
        return ("leaf", widget)

    def stack(self, items, *, direction="v", align=None, min_width=None, gap=None):
        return ("stack", direction, list(items))

    def grid(self, rows):
        return ("grid", [list(r) for r in rows])

    def present(self, root):
        return root


class _FakeLeaf:
    def __init__(self):
        self.closed = False

    def compose(self, host):
        return ("composed", self)

    def close(self):
        self.closed = True


def _assert_placeholder(spec, viewer, placeholder):
    """The dock renders with nothing in it but *placeholder*.

    It is not ``None``: a dock follows its viewer, so it has to exist before
    anything is configured for a later add to fill it.
    """
    closeables: list = []
    root = render_dock(spec, viewer, JupyterHost(), closeables)
    (dock,) = closeables
    assert dock.targets == []
    assert list(root.children) == []
    assert root.title == placeholder


# ---------------------------------------------------------------------------
# _render_appearance_controls guards
# ---------------------------------------------------------------------------


def test_appearance_controls_placeholder_without_any_config():
    from cellier.convenience.layout._controls_dock import APPEARANCE_PLACEHOLDER

    store = ImageMemoryStore(data=np.zeros((8, 16, 16), dtype=np.float32))
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="anywidget")
    viewer.add_image(
        store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )  # no controls=
    _assert_placeholder(AppearanceControls(), viewer, APPEARANCE_PLACEHOLDER)


# ---------------------------------------------------------------------------
# _render_center recursion (fake host)
# ---------------------------------------------------------------------------


def test_render_center_hstack():
    result = render_center(HStack(items=[_FakeLeaf()]), _FakeHost(), [])
    assert result[0] == "stack" and result[1] == "h"


def test_render_center_vstack():
    result = render_center(VStack(items=[_FakeLeaf()]), _FakeHost(), [])
    assert result[0] == "stack" and result[1] == "v"


def test_render_center_grid():
    result = render_center(Grid(cells=[[_FakeLeaf(), None]]), _FakeHost(), [])
    assert result[0] == "grid"


def test_render_center_leaf_is_tracked_as_closeable():
    closeables: list = []
    leaf = _FakeLeaf()
    result = render_center(leaf, _FakeHost(), closeables)
    assert result == ("composed", leaf)
    assert closeables == [leaf]


# ---------------------------------------------------------------------------
# _RenderView teardown
# ---------------------------------------------------------------------------


def test_render_view_close_swallows_errors_and_is_idempotent():
    class _Boom:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1
            raise RuntimeError("boom")

    boom = _Boom()
    view = _RenderView(root=object(), closeables=[boom])

    view.close()  # error swallowed
    assert boom.calls == 1

    view.close()  # idempotent: already closed, no second call
    assert boom.calls == 1
