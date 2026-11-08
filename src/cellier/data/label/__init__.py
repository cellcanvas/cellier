"""Label data stores."""

from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.data.label._ome_zarr_label_store import OMEZarrLabelDataStore

__all__ = ["LabelMemoryStore", "OMEZarrLabelDataStore"]
