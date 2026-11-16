"""Discriminated union of all visual model types."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.visuals._graph_memory import GraphVisual
from cellier.visuals._image import MultiscaleImageVisual
from cellier.visuals._image_memory import ImageVisual
from cellier.visuals._label_memory import LabelMemoryVisual
from cellier.visuals._labels import MultiscaleLabelVisual
from cellier.visuals._lines_memory import LinesVisual
from cellier.visuals._masked_image import MaskedImageVisual
from cellier.visuals._mesh_memory import MeshVisual, MultiscaleMeshVisual
from cellier.visuals._points_memory import PointsVisual

VisualType = Annotated[
    Union[
        MultiscaleImageVisual,
        MaskedImageVisual,
        ImageVisual,
        LabelMemoryVisual,
        MultiscaleLabelVisual,
        PointsVisual,
        LinesVisual,
        MeshVisual,
        MultiscaleMeshVisual,
        GraphVisual,
    ],
    Field(discriminator="visual_type"),
]
"""Any visual model, told apart by its ``visual_type`` field.

The type of ``Scene.visuals`` entries; a saved viewer file uses it to
rebuild each visual as the right class.
"""
