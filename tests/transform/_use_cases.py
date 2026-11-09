"""The four data -> world use cases the transform tests run over.

- ``scale_only``: ``zyx -> ZYX``, a diagonal scale.
- ``channel_passthrough``: ``czyx -> cZYX``, the channel axis maps to
  itself; scale and translation on the spatial axes only.
- ``channel_broadcast``: ``tzyx -> TCZYX``, the world has a channel axis
  the data lacks, so the data is broadcast over ``C``.
- ``reversed_axes``: ``zyx -> XYZ``, the same call as ``scale_only`` with
  the world axes in the opposite order (an anti-diagonal matrix).

Each case is here twice: as raw ``(linear, translation, broadcast)``
arrays in ``USE_CASES`` (for test_geometry_ops.py, which works on arrays
rather than models) and as model builders in ``MODEL_USE_CASES``.
"""

import numpy as np

# zyx -> ZYX, scale only
SCALE_ONLY = (np.diag([0.5, 0.1, 0.1]), np.zeros(3), ())

# czyx -> cZYX, scale + translation on ZYX only
CHANNEL_PASSTHROUGH = (
    np.diag([1.0, 0.5, 0.1, 0.1]),
    np.array([0.0, 10.0, -2.0, 3.0]),
    (),
)

# tzyx -> TCZYX, broadcast over C (output index 1)
CHANNEL_BROADCAST = (
    np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0.5, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1],
        ],
        float,
    ),
    np.array([0.0, 0.0, 10.0, -2.0, 3.0]),
    (1,),
)

# zyx -> XYZ  (rows X, Y, Z; columns z, y, x)
REVERSED_AXES = (
    np.array([[0, 0, 0.1], [0, 0.1, 0], [0.5, 0, 0]], float),
    np.zeros(3),
    (),
)

USE_CASES = {
    "scale_only": SCALE_ONLY,
    "channel_passthrough": CHANNEL_PASSTHROUGH,
    "channel_broadcast": CHANNEL_BROADCAST,
    "reversed_axes": REVERSED_AXES,
}


# ---------------------------------------------------------------------
# The same four cases as models, built through from_axis_map.  Each
# builder returns (data_system, world_system,
# transform) so a test can assert against the systems it was built from.
# ---------------------------------------------------------------------


def _space(name):
    from cellier.transform import Axis

    return Axis(name=name, axis_type="space", unit="micrometer")


def scale_only():
    """zyx -> ZYX, scale only."""
    from uuid import uuid4

    from cellier.transform import (
        AffineTransform,
        DataCoordinateSystem,
        WorldCoordinateSystem,
    )

    data = DataCoordinateSystem(
        name="cells",
        datastore_id=uuid4(),
        axes=(_space("z"), _space("y"), _space("x")),
    )
    world = WorldCoordinateSystem(
        axes=(_space("Z"), _space("Y"), _space("X")),
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.5, "y": 0.1, "x": 0.1},
    )
    return data, world, transform


def channel_passthrough():
    """czyx -> cZYX, scale + translation on ZYX only."""
    from uuid import uuid4

    from cellier.transform import (
        AffineTransform,
        Axis,
        DataCoordinateSystem,
        WorldCoordinateSystem,
    )

    data = DataCoordinateSystem(
        name="cells",
        datastore_id=uuid4(),
        axes=(
            Axis(name="c", axis_type="channel"),
            _space("z"),
            _space("y"),
            _space("x"),
        ),
    )
    world = WorldCoordinateSystem(
        axes=(
            Axis(name="c", axis_type="channel"),
            _space("Z"),
            _space("Y"),
            _space("X"),
        )
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"c": "c", "z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.5, "y": 0.1, "x": 0.1},
        translation={"z": 10.0, "y": -2.0, "x": 3.0},
    )
    return data, world, transform


def channel_broadcast():
    """tzyx -> TCZYX, broadcast over C."""
    from uuid import uuid4

    from cellier.transform import (
        AffineTransform,
        Axis,
        DataCoordinateSystem,
        WorldCoordinateSystem,
    )

    data = DataCoordinateSystem(
        name="timelapse",
        datastore_id=uuid4(),
        axes=(
            Axis(name="t", axis_type="time", unit="second"),
            _space("z"),
            _space("y"),
            _space("x"),
        ),
    )
    world = WorldCoordinateSystem(
        axes=(
            Axis(name="T", axis_type="time", unit="second"),
            Axis(name="C", axis_type="channel"),
            _space("Z"),
            _space("Y"),
            _space("X"),
        )
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"t": "T", "z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.5, "y": 0.1, "x": 0.1},
        translation={"z": 10.0, "y": -2.0, "x": 3.0},
        broadcast_output_axes=["C"],
    )
    return data, world, transform


def reversed_axes():
    """zyx -> XYZ; the identical call to scale_only, with a reordered world."""
    from uuid import uuid4

    from cellier.transform import (
        AffineTransform,
        DataCoordinateSystem,
        WorldCoordinateSystem,
    )

    data = DataCoordinateSystem(
        name="cells",
        datastore_id=uuid4(),
        axes=(_space("z"), _space("y"), _space("x")),
    )
    world = WorldCoordinateSystem(
        axes=(_space("X"), _space("Y"), _space("Z")),
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.5, "y": 0.1, "x": 0.1},
    )
    return data, world, transform


MODEL_USE_CASES = {
    "scale_only": scale_only,
    "channel_passthrough": channel_passthrough,
    "channel_broadcast": channel_broadcast,
    "reversed_axes": reversed_axes,
}
