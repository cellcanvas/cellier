"""Tests for ``cellier.convenience.layout._anywidget_renderer``.

The real-host happy paths are covered by ``tests/v2/test_anywidget.py``; this
fills the parity/guard gaps: the channel-controls build (mirrors the Qt
``_render_channel_controls_qt`` test), the appearance-controls None-guards, the
center recursion (via a fake host), and ``_RenderView`` teardown.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import AppearanceControls
from cellier.convenience.layout._walk import render_dock

pytest.importorskip("anywidget")

from cellier.convenience import ChannelControls, Viewer
from cellier.convenience._hosts import JupyterHost
from cellier.convenience.gui import (
    ChannelControlsConfig,
)
from cellier.convenience.layout._anywidget_renderer import _RenderView
from cellier.convenience.layout._spec import Grid, HStack, VStack
from cellier.convenience.layout._walk import render_center
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.visuals._channel_appearance import ChannelAppearance
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


def _multichannel_viewer():
    data = np.random.default_rng(0).random((3, 2, 16, 16)).astype(np.float32)
    store = ImageMemoryStore(data=data)
    viewer = Viewer(("z", "c", "y", "x"), dim="2d", gui="anywidget")
    channels = {
        0: ChannelAppearance(color_map="red", clim=(0.0, 1.0)),
        1: ChannelAppearance(color_map="green", clim=(0.0, 1.0)),
    }
    viewer.add_multichannel_image(
        store, channel_axis=1, channels=channels, controls=ChannelControlsConfig()
    )
    return viewer


# ---------------------------------------------------------------------------
# _render_channel_controls (parity with the Qt path)
# ---------------------------------------------------------------------------


def test_render_channel_controls_builds_and_registers_widget():
    viewer = _multichannel_viewer()
    closeables: list = []

    leaf = render_dock(ChannelControls(), viewer, JupyterHost(), closeables)

    assert leaf is not None
    assert len(closeables) == 1  # the AnywidgetChannelList is tracked for teardown


def test_render_channel_controls_none_without_config():
    data = np.random.default_rng(0).random((3, 2, 16, 16)).astype(np.float32)
    store = ImageMemoryStore(data=data)
    viewer = Viewer(("z", "c", "y", "x"), dim="2d", gui="anywidget")
    viewer.add_multichannel_image(
        store,
        channel_axis=1,
        channels={0: ChannelAppearance(color_map="red", clim=(0.0, 1.0))},
    )  # no controls=
    assert render_dock(ChannelControls(), viewer, JupyterHost(), []) is None


# ---------------------------------------------------------------------------
# _render_appearance_controls guards
# ---------------------------------------------------------------------------


def test_appearance_controls_none_without_any_config():
    store = ImageMemoryStore(data=np.zeros((8, 16, 16), dtype=np.float32))
    viewer = Viewer(("z", "y", "x"), gui="anywidget")
    viewer.add_image(
        store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
    )  # no controls=
    assert render_dock(AppearanceControls(), viewer, JupyterHost(), []) is None


def test_appearance_controls_none_when_only_channel_config():
    # A multichannel visual records a ChannelControlsConfig, which the
    # appearance builder must skip -> no appearance panel.
    viewer = _multichannel_viewer()
    assert render_dock(AppearanceControls(), viewer, JupyterHost(), []) is None


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
