"""Discriminated union of all data store types for cellier v2."""

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.data.graph._graph_memory_store import GraphMemoryStore
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.image._ome_zarr_image_store import OMEZarrImageDataStore
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore

DataStoreType = Annotated[
    Union[
        MultiscaleZarrDataStore,
        ImageMemoryStore,
        OMEZarrImageDataStore,
        LabelMemoryStore,
        PointsMemoryStore,
        LinesMemoryStore,
        MeshMemoryStore,
        GraphMemoryStore,
    ],
    Field(discriminator="store_type"),
]
