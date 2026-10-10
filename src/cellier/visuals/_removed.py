"""Field names removed by a hard break, and their refusal.

A removed name is not aliased and not mapped at load time (plane rendering
design v3 7.6): passing one raises, naming the replacement, so it cannot
pass silently as an ignored extra field.
"""

from __future__ import annotations

from typing import Any

#: On ``MultiscaleImageAppearance`` and ``MultiscaleLabelsAppearance``.
REMOVED_APPEARANCE_FIELDS: dict[str, str] = {
    "lod_bias": "use settled_lod_bias",
}

#: On ``ProgressiveLoadingConfig``.
REMOVED_LOADING_FIELDS: dict[str, str] = {
    "dims_drag": (
        "use appearance.coarsest_while_moving_3d and "
        "appearance.coarsest_while_moving_2d (True for 'backstop', False for "
        "'eager')"
    ),
    "backstop": "the backstop always loads",
}


#: Appearance keys of the multiscale controls configs
#: (``cellier.convenience.gui``): removed key to the key that replaces it.
REMOVED_CONTROL_KEYS: dict[str, str] = {
    "lod_bias": "level_of_detail",
    "settled_lod_bias": "level_of_detail",
}


def refuse_removed_fields(data: Any, removed: dict[str, str], model: str) -> Any:
    """Raise if *data*, a model's input, names a removed field.

    Parameters
    ----------
    data : Any
        The input of a ``model_validator(mode="before")``.
    removed : dict[str, str]
        Removed field name to what replaces it.
    model : str
        The model's name, for the message.

    Raises
    ------
    ValueError
        Naming the first removed field found and its replacement.
    """
    if isinstance(data, dict):
        for name, replacement in removed.items():
            if name in data:
                raise ValueError(f"{model}.{name} was removed: {replacement}.")
    return data
