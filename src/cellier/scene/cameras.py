"""Camera and camera controller models."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union
from uuid import uuid4

import numpy as np
from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, ConfigDict, Field

from cellier.types import NumpyFloat32Array

if TYPE_CHECKING:
    from psygnal import EmissionInfo


class BaseCameraController(EventedModel):
    """Base model for camera controllers.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    enabled : bool
        Whether the controller responds to input. Default True.
    """

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    enabled: bool = True


class OrbitCameraController(BaseCameraController):
    """Config for a 3D orbit camera controller.

    Parameters
    ----------
    controller_type : Literal["orbit"]
        Discriminator field. Always ``"orbit"``.
    """

    controller_type: Literal["orbit"] = "orbit"


class PanZoomCameraController(BaseCameraController):
    """Config for a 2D pan/zoom camera controller.

    Parameters
    ----------
    controller_type : Literal["pan_zoom"]
        Discriminator field. Always ``"pan_zoom"``.
    """

    controller_type: Literal["pan_zoom"] = "pan_zoom"


CameraControllerType = Annotated[
    Union[OrbitCameraController, PanZoomCameraController],
    Field(discriminator="controller_type"),
]


class BaseCamera(EventedModel):
    """Base model for cameras.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    controller : CameraControllerType
        Discriminated union of camera controllers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    controller: CameraControllerType

    def model_post_init(self, __context: Any) -> None:
        """Wire controller event relay after model initialization."""
        self.controller.events.all.connect(self._on_controller_updated)

    def _on_controller_updated(self, info: EmissionInfo) -> None:
        self.events.controller.emit(self.controller)


class PerspectiveCamera(BaseCamera):
    """3D perspective camera state.

    Parameters
    ----------
    camera_type : Literal["perspective"]
        Discriminator field. Always ``"perspective"``.
    fov : float
        Field of view in degrees. Default 70.0.
    zoom : float
        Zoom level. Default 1.0.
    near_clipping_plane : float
        Near clipping distance. Default 1.0.
    far_clipping_plane : float
        Far clipping distance. Default 8000.0.
    position : NumpyFloat32Array
        World-space camera position, shape (3,).
    rotation : NumpyFloat32Array
        Quaternion [x, y, z, w], shape (4,).
    up_direction : NumpyFloat32Array
        World-space up vector, shape (3,).
    frustum : NumpyFloat32Array
        Near and far plane corners, shape (2, 4, 3).
    """

    camera_type: Literal["perspective"] = "perspective"
    fov: float = 70.0
    zoom: float = 1.0
    near_clipping_plane: float = 1.0
    far_clipping_plane: float = 8000.0
    position: NumpyFloat32Array = Field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    rotation: NumpyFloat32Array = Field(
        default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32)
    )
    up_direction: NumpyFloat32Array = Field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    frustum: NumpyFloat32Array = Field(
        default_factory=lambda: np.zeros((2, 4, 3), dtype=np.float32)
    )


class OrthographicCamera(BaseCamera):
    """2D orthographic camera state.

    Parameters
    ----------
    camera_type : Literal["orthographic"]
        Discriminator field. Always ``"orthographic"``.
    width : float
        Width of the view volume. Default 10.0.
    height : float
        Height of the view volume. Default 10.0.
    zoom : float
        Zoom level. Default 1.0.
    near_clipping_plane : float
        Near clipping distance. Default -500.0.
    far_clipping_plane : float
        Far clipping distance. Default 500.0.
    position : NumpyFloat32Array
        World-space camera position, shape (3,).
    rotation : NumpyFloat32Array
        Quaternion [x, y, z, w], shape (4,).
    """

    camera_type: Literal["orthographic"] = "orthographic"
    width: float = 10.0
    height: float = 10.0
    zoom: float = 1.0
    near_clipping_plane: float = -500.0
    far_clipping_plane: float = 500.0
    position: NumpyFloat32Array = Field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    rotation: NumpyFloat32Array = Field(
        default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32)
    )


CameraType = Annotated[
    Union[PerspectiveCamera, OrthographicCamera],
    Field(discriminator="camera_type"),
]
