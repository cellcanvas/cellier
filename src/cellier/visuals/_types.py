"""Discriminated union of all visual model types for cellier v2."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.visuals._graph_memory import GraphVisual
from cellier.visuals._image import (
    MultichannelMultiscaleImageVisual,
    MultiscaleImageVisual,
)
from cellier.visuals._image_memory import ImageVisual, MultichannelImageVisual
from cellier.visuals._label_memory import LabelMemoryVisual
from cellier.visuals._labels import MultiscaleLabelVisual
from cellier.visuals._lines_memory import LinesVisual
from cellier.visuals._mesh_memory import MeshVisual
from cellier.visuals._points_memory import PointsVisual

VisualType = Annotated[
    Union[
        MultiscaleImageVisual,
        ImageVisual,
        MultichannelImageVisual,
        MultichannelMultiscaleImageVisual,
        LabelMemoryVisual,
        MultiscaleLabelVisual,
        PointsVisual,
        LinesVisual,
        MeshVisual,
        GraphVisual,
    ],
    Field(discriminator="visual_type"),
]
