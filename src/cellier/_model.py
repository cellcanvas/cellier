"""Base classes for cellier's pydantic models.

Every model refuses a field it does not declare (``extra="forbid"``), so a
misspelled or removed key raises instead of being dropped without a word.
A key that old saved files carry and that is still accepted is dropped by
the model that owns it, with a validator that names it.
"""

from __future__ import annotations

from psygnal import EventedModel
from pydantic import BaseModel, ConfigDict


class CellierModel(BaseModel):
    """A pydantic model that refuses undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class CellierEventedModel(EventedModel):
    """A psygnal evented model that refuses undeclared fields."""

    model_config = ConfigDict(extra="forbid")
