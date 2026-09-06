"""Convenience-layer tests for the ChannelControls dock (Qt default path)."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import ChannelControls, OrthoViewer, Viewer
from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.gui._controls_config import ChannelControlsConfig
from cellier.convenience.layout._shared import _resolve_channel_visual_ids
from cellier.convenience.layout._walk import render_dock
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.visuals._channel_appearance import ChannelAppearance


def _make_channel_appearance(**kwargs) -> ChannelAppearance:
    defaults = {"color_map": "viridis", "clim": (0.0, 1.0)}
    defaults.update(kwargs)
    return ChannelAppearance(**defaults)


def _make_store(n_channels=2) -> ImageMemoryStore:
    data = np.random.default_rng(0).random((3, n_channels, 16, 16)).astype(np.float32)
    return ImageMemoryStore(data=data)


def _channels(n=2) -> dict[int, ChannelAppearance]:
    return {i: _make_channel_appearance() for i in range(n)}


def _find_control(widget, channel_index: int, field: str):
    applier = widget._appliers[(channel_index, field)]
    return applier.__closure__[0].cell_contents


# ---------------------------------------------------------------------------
# Resolver + cap validation (design section 7.3 / 11.4)
# ---------------------------------------------------------------------------


def test_resolver_single_viewer_returns_config_and_visual():
    viewer = Viewer(("z", "c", "y", "x"), dim="2d")
    visual = viewer.add_multichannel_image(
        _make_store(),
        channel_axis=1,
        channels=_channels(2),
        controls=ChannelControlsConfig(),
    )

    resolved = _resolve_channel_visual_ids(viewer)
    assert resolved is not None
    config, visual_ids, channels = resolved
    assert isinstance(config, ChannelControlsConfig)
    assert visual_ids == [visual.id]
    assert set(channels) == {0, 1}


def test_resolver_returns_none_without_controls():
    viewer = Viewer(("z", "c", "y", "x"), dim="2d")
    viewer.add_multichannel_image(_make_store(), channel_axis=1, channels=_channels(2))
    assert _resolve_channel_visual_ids(viewer) is None


def test_resolver_raises_over_cap():
    viewer = Viewer(("z", "c", "y", "x"), dim="2d")
    viewer.add_multichannel_image(
        _make_store(3),
        channel_axis=1,
        channels=_channels(3),
        max_channels_3d=2,  # min(8, 2) = 2 < 3 channels
        controls=ChannelControlsConfig(),
    )
    with pytest.raises(ValueError, match="min\\(max_channels_2d"):
        _resolve_channel_visual_ids(viewer)


def test_resolver_succeeds_at_exactly_cap():
    viewer = Viewer(("z", "c", "y", "x"), dim="2d")
    viewer.add_multichannel_image(
        _make_store(3),
        channel_axis=1,
        channels=_channels(3),
        max_channels_3d=3,  # min(8, 3) = 3 == 3 channels
        controls=ChannelControlsConfig(),
    )
    resolved = _resolve_channel_visual_ids(viewer)
    assert resolved is not None
    assert len(resolved.channels) == 3


def test_bumping_cap_makes_failing_case_pass():
    # Same 3-channel set that failed at max_channels_3d=2 passes at 4.
    viewer = Viewer(("z", "c", "y", "x"), dim="2d")
    viewer.add_multichannel_image(
        _make_store(3),
        channel_axis=1,
        channels=_channels(3),
        max_channels_3d=4,
        controls=ChannelControlsConfig(),
    )
    resolved = _resolve_channel_visual_ids(viewer)
    assert resolved is not None
    assert len(resolved.channels) == 3


# ---------------------------------------------------------------------------
# OrthoViewer: one widget drives every panel
# ---------------------------------------------------------------------------


def test_ortho_resolver_gathers_all_panel_ids():
    ortho = OrthoViewer(("z", "c", "y", "x"), spatial_axes=("z", "y", "x"))
    visuals = ortho.add_multichannel_image(
        _make_store(2),
        channel_axis=1,
        channels=_channels(2),
        controls=ChannelControlsConfig(),
    )

    resolved = _resolve_channel_visual_ids(ortho)
    assert resolved is not None
    _config, visual_ids, _channels_out = resolved
    assert set(visual_ids) == {v.id for v in visuals.values()}
    assert len(visual_ids) == 4


def test_ortho_edit_reaches_all_panels(qtbot):
    from cellier.gui.qt.visuals import QtChannelList

    ortho = OrthoViewer(("z", "c", "y", "x"), spatial_axes=("z", "y", "x"))
    visuals = ortho.add_multichannel_image(
        _make_store(2),
        channel_axis=1,
        channels=_channels(2),
        controls=ChannelControlsConfig(),
    )

    _config, visual_ids, channels = _resolve_channel_visual_ids(ortho)
    widget = QtChannelList(visual_ids, channels)
    ortho.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )

    checkbox = _find_control(widget, 1, "visible")
    checkbox.setChecked(False)

    for visual in visuals.values():
        assert visual.channels[1].visible is False


# ---------------------------------------------------------------------------
# Qt renderer branch
# ---------------------------------------------------------------------------


def test_render_channel_controls_qt_builds_widget(qtbot):

    viewer = Viewer(("z", "c", "y", "x"), dim="2d")
    viewer.add_multichannel_image(
        _make_store(2),
        channel_axis=1,
        channels=_channels(2),
        controls=ChannelControlsConfig(),
    )

    rendered = render_dock(ChannelControls(), viewer, QtLayoutHost(), [])
    assert rendered is not None


def test_render_dock_qt_dispatches_channel_controls(qtbot):
    from cellier.convenience.layout._spec import ChannelControls

    viewer = Viewer(("z", "c", "y", "x"), dim="2d")
    viewer.add_multichannel_image(
        _make_store(2),
        channel_axis=1,
        channels=_channels(2),
        controls=ChannelControlsConfig(),
    )

    rendered = render_dock(ChannelControls(), viewer, QtLayoutHost(), [])
    assert rendered is not None


def test_render_channel_controls_qt_none_without_config(qtbot):

    viewer = Viewer(("z", "c", "y", "x"), dim="2d")
    viewer.add_multichannel_image(_make_store(2), channel_axis=1, channels=_channels(2))
    assert render_dock(ChannelControls(), viewer, QtLayoutHost(), []) is None
