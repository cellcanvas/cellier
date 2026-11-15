"""Camera and camera controller models."""

from __future__ import annotations

import math
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Union
from uuid import uuid4

import numpy as np
from pydantic import (
    UUID4,
    AfterValidator,
    ConfigDict,
    Field,
    PrivateAttr,
)

from cellier._model import CellierEventedModel
from cellier.types import NumpyFloat32Array

if TYPE_CHECKING:
    from collections.abc import Iterator

    from psygnal import EmissionInfo


def orbit_distance(fov: float, width: float, height: float) -> float | None:
    """The distance ahead of a 3D camera of the point its orbit controller turns about.

    pygfx's orbit controller turns the camera about the point where the
    view is ``width`` by ``height``: ``0.5 / tan(fov / 2)`` times the mean of
    the two.  The field of view is clamped as pygfx clamps it: below 0.5
    degrees it is 0, below 1 it is 1, above 179 it is 179.

    Parameters
    ----------
    fov : float
        Vertical field of view in degrees.
    width, height : float
        The size of the view at that point, in world units (the camera's
        ``width`` and ``height``, or a ``CameraState.extent``).

    Returns
    -------
    float or None
        The distance in world units; ``None`` at a field of view of 0,
        where the camera turns about itself.
    """
    if fov < 0.5:
        return None
    fov = min(max(fov, 1.0), 179.0)
    return 0.5 / math.tan(math.radians(fov) / 2.0) * 0.5 * (width + height)


class BaseCameraController(CellierEventedModel):
    """Base model for camera controllers.

    The model of the mouse/keyboard controller that drives a canvas camera.
    The controller type follows the camera: a perspective (3D) camera is
    driven by an orbit controller and an orthographic (2D) camera by a
    pan/zoom controller, and each camera model only accepts its own kind.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    enabled : bool
        Whether mouse and keyboard input moves the camera.  Assigning it
        on a canvas's camera model turns the live controller on or off;
        ``CellierController.set_camera_controller_enabled`` sets it on all
        of a canvas's cameras.  Read when a canvas is created from the
        model.  Default True.
    """

    model_config = ConfigDict(validate_assignment=True)

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    enabled: bool = True


class OrbitCameraController(BaseCameraController):
    """Orbit controller of a 3D perspective camera.

    See `BaseCameraController`.

    Parameters
    ----------
    controller_type : Literal["orbit"]
        Discriminator field. Always ``"orbit"``.

    Notes
    -----
    The controller turns the camera about a point straight ahead of it, at
    the distance `orbit_distance` gives for the camera's field of view,
    width and height: the camera's ``width`` and ``height`` are the size of
    the view at that point.  The point is not stored anywhere else.
    """

    controller_type: Literal["orbit"] = "orbit"


class PanZoomCameraController(BaseCameraController):
    """Pan/zoom controller of a 2D orthographic camera.

    See `BaseCameraController`.

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
"""A camera controller (orbit or pan/zoom), told apart by ``controller_type``."""


class BaseCamera(CellierEventedModel):
    """Base model for cameras.

    The model of one of a canvas's cameras (``Canvas.cameras``), kept in
    step with the live camera both ways:

    * After every camera move (mouse input, ``set_camera_state``, a fit)
      the controller writes the pose fields of the camera that moved:
      ``position``, ``rotation``, the clipping planes, and
      ``fov`` / ``up_direction`` / ``frustum`` / ``width`` / ``height``
      or ``width`` / ``height``.
    * Assigning a field moves the live camera.  For the camera the canvas
      is showing this is a camera **jump**, as with
      ``CellierController.set_camera_state``: camera-sensitive visuals
      reslice at once.  Each assignment is its own jump; to set several
      fields in one move use ``set_camera_state``.  Assigning to the
      camera of the other dimensionality (the 3D camera of a canvas
      showing 2D) places it for when the canvas switches to it, and the
      switch then does not fit it (``Viewer.set_displayed_dimensions``
      fits the camera after every switch all the same).

    When a canvas is created from a model (``add_canvas_model``,
    ``from_model``, ``from_file``), each camera starts at the model's
    pose, unless the pose is the default one (``position`` all zeros and
    ``rotation`` the identity), which a camera that was never placed has;
    such a camera is fitted to the data as usual.

    The pose fields are in the renderer's ``(x, y, z)`` frame, in which
    ``x`` is the **last** displayed world axis, not in the order of the
    scene's world coordinate system.

    Only field assignments are followed: replacing a camera model in
    ``Canvas.cameras`` after the canvas was created does not reach the
    live camera.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    controller : CameraControllerType
        The camera's controller model: an `OrbitCameraController`
        for a perspective camera, a `PanZoomCameraController` for an
        orthographic one.  Replacing it applies the new one's ``enabled``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    controller: CameraControllerType

    # True while the controller writes the live camera's state into the
    # model, so the writes are not taken for assignments to apply.
    _mirroring: bool = PrivateAttr(default=False)

    #: Fields only the live camera writes; assigning them raises.
    _DERIVED_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def model_post_init(self, __context: Any) -> None:
        """Wire controller event relay after model initialization."""
        self.controller.events.all.connect(self._on_controller_updated)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse derived fields; keep the controller relay on a new controller."""
        if name in self._DERIVED_FIELDS and not self._mirroring:
            raise AttributeError(
                f"{type(self).__name__}.{name} is derived from the live "
                "camera and cannot be assigned."
            )
        if name != "controller":
            super().__setattr__(name, value)
            return
        old = self.controller
        old.events.all.disconnect(self._on_controller_updated)
        try:
            super().__setattr__(name, value)
        finally:
            self.controller.events.all.connect(self._on_controller_updated)

    @contextmanager
    def _mirror_writes(self) -> Iterator[None]:
        """Mark field writes as coming from the live camera (controller use)."""
        previous = self._mirroring
        self._mirroring = True
        try:
            yield
        finally:
            self._mirroring = previous

    @property
    def has_pose(self) -> bool:
        """Whether the pose differs from the default one of an unplaced camera.

        ``False`` when ``position`` is all zeros and ``rotation`` the
        identity quaternion.  A canvas created from the model starts such a
        camera unplaced (it is fitted to the data); any other pose is
        applied.
        """
        return bool(
            np.any(np.asarray(self.position) != 0)
            or not np.array_equal(np.asarray(self.rotation), [0.0, 0.0, 0.0, 1.0])
        )

    def _on_controller_updated(self, info: EmissionInfo) -> None:
        self.events.controller.emit(self.controller)


class PerspectiveCamera(BaseCamera):
    """A canvas's 3D perspective camera; see `BaseCamera`.

    Parameters
    ----------
    camera_type : Literal["perspective"]
        Discriminator field. Always ``"perspective"``.
    fov : float
        Field of view in degrees. Default 70.0.  0 is an orthographic
        projection.
    width : float or None
        Width in world units of the view at the point the orbit controller
        turns the camera about (`orbit_distance` ahead of the camera).  At
        a ``fov`` of 0 the camera turns about itself and every plane is
        this size, so it is the width of the view.  A wheel zoom changes
        it (and moves the camera); the live camera writes it, as
        ``CameraState.extent``.  Assigning it moves the orbit point, or at
        a ``fov`` of 0 resizes the view.  ``None`` (the default) until the
        live camera first writes it, and in files saved before the field
        existed; a camera placed from such a model keeps its own size.
    height : float or None
        Height of the view at the orbit point, as ``width``.  Default
        ``None``.
    near_clipping_plane : float
        Near clipping distance. Default 1.0.
    far_clipping_plane : float
        Far clipping distance. Default 8000.0.  A camera fit raises it when the
        scene's data needs more (never lowers it), and the controller writes
        the raised value here.
    position : NumpyFloat32Array
        Camera position in the renderer's ``(x, y, z)`` frame, shape (3,).
    rotation : NumpyFloat32Array
        Quaternion [x, y, z, w], shape (4,).
    up_direction : NumpyFloat32Array
        The camera's up vector in the renderer's ``(x, y, z)`` frame, shape
        (3,); the live camera writes the one its rotation gives.  Assigning
        it turns the camera about its view direction to bring its up as
        close to the vector as it goes, and makes it the up the orbit
        controller keeps.  All zeros until the camera is first written.
    frustum : NumpyFloat32Array
        Corners of the view frustum in the renderer's frame, shape
        (2, 4, 3): the near plane's four corners, then the far plane's.
        Written by the live camera; assigning it raises ``AttributeError``.
        All zeros until the camera is first written.
    controller : OrbitCameraController
        Must be an orbit controller.
    """

    _DERIVED_FIELDS: ClassVar[frozenset[str]] = frozenset({"frustum"})

    controller: OrbitCameraController

    camera_type: Literal["perspective"] = "perspective"
    fov: float = 70.0
    # The view's size at the orbit point; ``None`` until first written.
    width: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    height: float | None = Field(default=None, gt=0, allow_inf_nan=False)
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

    @property
    def orbit_distance(self) -> float | None:
        """How far ahead of the camera the orbit controller turns it.

        Worked out from ``fov``, ``width`` and ``height`` by
        `orbit_distance`; ``None`` at a ``fov`` of 0 (the camera turns about
        itself) and while ``width`` or ``height`` is ``None``.
        """
        if self.width is None or self.height is None:
            return None
        return orbit_distance(self.fov, self.width, self.height)


class OrthographicCamera(BaseCamera):
    """A canvas's 2D orthographic camera; see `BaseCamera`.

    Parameters
    ----------
    camera_type : Literal["orthographic"]
        Discriminator field. Always ``"orthographic"``.
    width : float
        Width of the view volume. Default 10.0.
    height : float
        Height of the view volume. Default 10.0.
    near_clipping_plane : float
        Near clipping distance. Default -500.0.
    far_clipping_plane : float
        Far clipping distance. Default 500.0.
    position : NumpyFloat32Array
        Camera position in the renderer's ``(x, y, z)`` frame, shape (3,).
    rotation : NumpyFloat32Array
        Quaternion [x, y, z, w], shape (4,).
    controller : PanZoomCameraController
        Must be a pan/zoom controller.
    """

    camera_type: Literal["orthographic"] = "orthographic"
    controller: PanZoomCameraController
    width: float = 10.0
    height: float = 10.0

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
"""A camera model (perspective or orthographic), told apart by ``camera_type``."""
