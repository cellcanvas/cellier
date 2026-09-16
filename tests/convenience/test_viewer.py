"""Tests for cellier.convenience.Viewer."""

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.render._config import RenderManagerConfig, SlicingConfig
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageChannelAppearance,
    InMemoryImageSingleAppearance,
    MultiscaleImageChannelAppearance,
)
from cellier.visuals._image_memory import InMemoryImageAppearance


@pytest.fixture
def image_store() -> ImageMemoryStore:
    data = np.zeros((8, 16, 16), dtype=np.float32)
    return ImageMemoryStore(data=data, name="test_image")


def test_serialization_roundtrip(tmp_path, image_store):
    """Viewer serializes and deserializes with equivalent ViewerModels."""
    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        dim="2d",
        render_config=RenderManagerConfig(slicing=SlicingConfig(batch_size=16)),
    )
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        name="test_image",
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    path = tmp_path / "viewer.json"
    viewer.to_file(path)

    loaded = Viewer.from_file(path)

    assert viewer.controller._model == loaded.controller._model


_CZYX = [("c", "channel"), *spatial_axes("z", "y", "x")]


def _channels(cls, count: int) -> dict:
    """*count* channels; channel 0 hidden, so a round trip must keep visibility."""
    return {
        index: cls(color_map="magenta" if index else "green", visible=bool(index))
        for index in range(count)
    }


@pytest.mark.parametrize(
    ("composite", "n_channels"),
    [(False, 2), (True, 2), (True, 0)],
    ids=["single", "composite", "empty_composite"],
)
def test_serialization_roundtrip_unified_image(tmp_path, composite, n_channels):
    """Unified image design 3.11: mode, channels and the single page survive."""
    store = ImageMemoryStore(data=np.zeros((2, 4, 8, 8), dtype=np.float32), name="c")
    viewer = Viewer(_CZYX, dim="2d")
    viewer.add_image(
        store,
        channel_axis=0,
        composite=composite,
        channels=_channels(InMemoryImageChannelAppearance, n_channels),
        single=InMemoryImageSingleAppearance(color_map="viridis", clim=(0.0, 2.0)),
        name="czyx",
    )

    path = tmp_path / "viewer.json"
    viewer.to_file(path)
    loaded = Viewer.from_file(path)

    assert viewer.controller._model == loaded.controller._model
    (visual,) = loaded.scene.visuals
    assert (visual.channel_axis, visual.composite) == (0, composite)
    assert sorted(visual.channels) == list(range(n_channels))
    assert [c.visible for _, c in sorted(visual.channels.items())] == [
        bool(i) for i in range(n_channels)
    ]
    assert visual.single.clim == (0.0, 2.0)
    # The derived slider axes follow the restored mode.
    assert (0 in loaded.scene.slider_axes) is (not composite)


@pytest.mark.parametrize("composite", [False, True], ids=["single", "composite"])
def test_serialization_roundtrip_unified_multiscale_image(
    tmp_path, multichannel_multiscale_store, composite
):
    viewer = Viewer(_CZYX, dim="2d")
    viewer.add_image_multiscale(
        multichannel_multiscale_store,
        channel_axis=0,
        composite=composite,
        channels=_channels(MultiscaleImageChannelAppearance, 2),
        name="czyx",
    )

    path = tmp_path / "viewer.json"
    viewer.to_file(path)
    loaded = Viewer.from_file(path)

    assert viewer.controller._model == loaded.controller._model
    (visual,) = loaded.scene.visuals
    assert (visual.channel_axis, visual.composite) == (0, composite)
    assert sorted(visual.channels) == [0, 1]
