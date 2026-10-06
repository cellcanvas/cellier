"""The "Level of detail" control: toolkit-neutral logic.

One control per multiscale image or labels visual (plane rendering design
v3, 9.3): a slider for ``appearance.settled_lod_bias`` and a checkbox for
"coarsest while moving".  The checkbox edits the setting of the view the
control is shown in: ``coarsest_while_moving_3d`` in a 3D view and
``coarsest_while_moving_2d`` in a 2D one.

The Qt and anywidget controls import everything that is not toolkit from
here, so the two cannot drift.
"""

from __future__ import annotations

from typing import Any

LEVEL_OF_DETAIL_TITLE = "Level of detail"

SETTLED_BIAS_LABEL = "Settled bias"
COARSEST_LABEL = "Coarsest while moving"

#: Slider bounds for ``settled_lod_bias``.  Higher is coarser.
SETTLED_BIAS_RANGE: tuple[float, float] = (0.001, 5.0)

#: The appearance fields the control drives.
LEVEL_OF_DETAIL_FIELDS: tuple[str, ...] = (
    "settled_lod_bias",
    "coarsest_while_moving_3d",
    "coarsest_while_moving_2d",
)


def moving_field(n_displayed_dimensions: int) -> str:
    """The "coarsest while moving" field of a view.

    Parameters
    ----------
    n_displayed_dimensions : int
        2 or 3: what the view the control is shown in displays.

    Returns
    -------
    str
        ``"coarsest_while_moving_3d"`` for 3, ``"coarsest_while_moving_2d"``
        otherwise.
    """
    if n_displayed_dimensions == 3:
        return "coarsest_while_moving_3d"
    return "coarsest_while_moving_2d"


def level_of_detail_values(appearance: Any) -> dict[str, Any]:
    """Read the control's construction values off a multiscale appearance.

    Parameters
    ----------
    appearance : MultiscaleImageAppearance or MultiscaleLabelsAppearance
        The appearance model the control drives.

    Returns
    -------
    dict
        ``settled_lod_bias`` (float) and the two ``coarsest_while_moving``
        booleans, keyed by field name.
    """
    return {
        "settled_lod_bias": float(appearance.settled_lod_bias),
        "coarsest_while_moving_3d": bool(appearance.coarsest_while_moving_3d),
        "coarsest_while_moving_2d": bool(appearance.coarsest_while_moving_2d),
    }


def from_control_value(field: str, value: Any) -> Any:
    """Coerce a control's value to the model's type for *field*."""
    if field == "settled_lod_bias":
        return float(value)
    return bool(value)
