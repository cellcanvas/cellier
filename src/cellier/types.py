"""Shared annotated types for models."""

from typing import Annotated

import numpy as np
from pydantic import GetPydanticSchema
from pydantic_core import core_schema


def _numpy_float32_schema():
    return core_schema.no_info_plain_validator_function(
        lambda v: np.asarray(v, dtype=np.float32),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda arr: arr.tolist(),
            info_arg=False,
        ),
    )


NumpyFloat32Array = Annotated[
    np.ndarray,
    GetPydanticSchema(lambda tp, handler: _numpy_float32_schema()),
]
