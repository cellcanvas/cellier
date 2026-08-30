"""Tests that each TypedDict companion covers all fields of its model class."""

import pytest

from cellier.convenience._kwarg_dicts import (
    ChannelAppearanceKwargs,
    GraphAppearanceKwargs,
    InMemoryImageAppearanceKwargs,
    InMemoryLabelsAppearanceKwargs,
    LinesMemoryAppearanceKwargs,
    MeshFlatAppearanceKwargs,
    MeshPhongAppearanceKwargs,
    MultiscaleImageAppearanceKwargs,
    MultiscaleImageRenderConfigKwargs,
    MultiscaleLabelRenderConfigKwargs,
    MultiscaleLabelsAppearanceKwargs,
    PointsMarkerAppearanceKwargs,
)
from cellier.visuals._channel_appearance import ChannelAppearance
from cellier.visuals._graph_memory import GraphAppearance
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from cellier.visuals._label_memory import InMemoryLabelsAppearance
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from cellier.visuals._lines_memory import LinesMemoryAppearance
from cellier.visuals._mesh_memory import MeshFlatAppearance, MeshPhongAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance


@pytest.mark.parametrize(
    "typeddict_cls, model_cls",
    [
        (InMemoryImageAppearanceKwargs, InMemoryImageAppearance),
        (InMemoryLabelsAppearanceKwargs, InMemoryLabelsAppearance),
        (MeshFlatAppearanceKwargs, MeshFlatAppearance),
        (MeshPhongAppearanceKwargs, MeshPhongAppearance),
        (GraphAppearanceKwargs, GraphAppearance),
        (PointsMarkerAppearanceKwargs, PointsMarkerAppearance),
        (LinesMemoryAppearanceKwargs, LinesMemoryAppearance),
        (MultiscaleImageAppearanceKwargs, MultiscaleImageAppearance),
        (MultiscaleLabelsAppearanceKwargs, MultiscaleLabelsAppearance),
        (ChannelAppearanceKwargs, ChannelAppearance),
        (MultiscaleImageRenderConfigKwargs, MultiscaleImageRenderConfig),
        (MultiscaleLabelRenderConfigKwargs, MultiscaleLabelRenderConfig),
    ],
)
def test_typeddict_covers_model_fields(typeddict_cls, model_cls):
    """Every field on the model must appear as a key in its TypedDict companion."""
    # Use __required_keys__ | __optional_keys__ rather than get_type_hints() so
    # that third-party annotation types (e.g. cmap.Colormap) don't need to be
    # importable at test time.
    typeddict_keys = typeddict_cls.__required_keys__ | typeddict_cls.__optional_keys__
    model_fields = set(model_cls.model_fields)
    missing = model_fields - typeddict_keys
    assert not missing, (
        f"{typeddict_cls.__name__} is missing fields present on "
        f"{model_cls.__name__}: {missing}"
    )
