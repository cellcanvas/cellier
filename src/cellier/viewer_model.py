"""Top-level viewer model."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from pydantic import UUID4, AfterValidator, Field

from cellier._model import CellierEventedModel
from cellier.data._types import DataStoreType
from cellier.render._config import RenderManagerConfig
from cellier.scene.scene import Scene


class DataManager(CellierEventedModel):
    """Thin evented wrapper around a dict of data stores.

    Parameters
    ----------
    stores : dict[UUID4, DataStoreType]
        Mapping of store id to store model. Keyed by ``store.id``.
    """

    stores: dict[
        UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))],
        DataStoreType,
    ] = Field(default_factory=dict)


class ViewerModel(CellierEventedModel):
    """Top-level model tree for a cellier viewer session.

    Parameters
    ----------
    data : DataManager
        Manages all data stores.
    scenes : dict[UUID4, Scene]
        Mapping of scene id to scene model. Keyed by ``scene.id``.
    """

    data: DataManager
    scenes: dict[
        UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))],
        Scene,
    ] = Field(default_factory=dict)
    render_config: RenderManagerConfig = Field(default_factory=RenderManagerConfig)

    def to_file(self, path: str | Path) -> None:
        """Serialize the viewer model to a JSON file.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """
        Path(path).write_text(self.model_dump_json())

    @classmethod
    def from_file(cls, path: str | Path) -> ViewerModel:
        """Deserialize a viewer model from a JSON file.

        Parameters
        ----------
        path : str | Path
            Source file path.

        Returns
        -------
        ViewerModel
        """
        return cls.model_validate_json(Path(path).read_text())
