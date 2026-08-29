"""Label colormap GPU resource builders."""

from __future__ import annotations

import warnings

import numpy as np
import pygfx as gfx
import wgpu

_MAX_DIRECT_ENTRIES = 65536

LABEL_PARAMS_DTYPE = np.dtype(
    [
        ("background_label", np.int32),
        ("salt", np.uint32),
        ("n_entries", np.uint32),
        ("n_outline_entries", np.uint32),
    ]
)


def build_direct_lut_textures(
    color_dict: dict[int, tuple[float, float, float, float]],
) -> tuple[gfx.Texture, gfx.Texture, int]:
    """Build GPU textures for direct-mode colormap binary search.

    Returns (keys_tex, colors_tex, n_entries).
    keys_tex  : r32sint 2D texture, shape (H=1, W=n, C=1) — sorted int32 IDs
    colors_tex: rgba32float 2D texture, shape (H=1, W=n, C=4) — RGBA per entry
    n_entries : int — number of entries (0 for empty dict)
    """
    n_raw = len(color_dict)
    if n_raw > _MAX_DIRECT_ENTRIES:
        warnings.warn(
            f"color_dict has {n_raw} entries, which exceeds the maximum of "
            f"{_MAX_DIRECT_ENTRIES}. Only the {_MAX_DIRECT_ENTRIES} entries "
            f"with the smallest label IDs will be used.",
            stacklevel=3,
        )
    keys_sorted = sorted(color_dict.keys())[:_MAX_DIRECT_ENTRIES]
    n = len(keys_sorted)

    if n == 0:
        key_data = np.array([[0]], dtype=np.int32).reshape(1, 1, 1)
        color_data = np.zeros((1, 1, 4), dtype=np.float32)
        return (
            gfx.Texture(key_data, dim=2, format="1xi4"),
            gfx.Texture(color_data, dim=2, format="4xf4"),
            0,
        )

    # pygfx 2D texture shape: (H, W, C)
    key_data = np.array(keys_sorted, dtype=np.int32).reshape(1, n, 1)
    color_data = np.array(
        [color_dict[k] for k in keys_sorted], dtype=np.float32
    ).reshape(1, n, 4)

    keys_tex = gfx.Texture(key_data, dim=2, format="1xi4")
    colors_tex = gfx.Texture(color_data, dim=2, format="4xf4")
    return keys_tex, colors_tex, n


def build_label_params_buffer(
    background_label: int,
    salt: int,
    n_entries: int,
    n_outline_entries: int = 0,
) -> gfx.Buffer:
    """Build the u_label_params uniform buffer (16 bytes, aligned).

    ``n_outline_entries`` is the number of selected-label entries in the
    outline selection texture; it took over the slot that used to be
    padding, so the buffer size is unchanged.
    """
    params = np.zeros((), dtype=LABEL_PARAMS_DTYPE)
    params["background_label"] = np.int32(background_label)
    params["salt"] = np.uint32(salt & 0xFFFFFFFF)
    params["n_entries"] = np.uint32(n_entries)
    params["n_outline_entries"] = np.uint32(n_outline_entries)
    return gfx.Buffer(
        params, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )


#: Fixed number of slots in the outline selection texture.
#: Fixed on purpose: the texture is bound at shader-build time, so resizing
#: it would mean a new texture object, a new binding and a pipeline rebuild
#: every time the selection changes.  A fixed capacity updated in place
#: makes ``set_label_selection`` a data upload.
OUTLINE_SELECTION_CAPACITY = 256


def build_outline_selection_texture() -> gfx.Texture:
    """Build the empty (label_id, slot) texture the outline key searches.

    Returns
    -------
    gfx.Texture
        An ``rg32sint`` texture of shape ``(1, OUTLINE_SELECTION_CAPACITY, 2)``.
        Contents are meaningless until :func:`update_outline_selection` fills
        them; ``u_label_params.n_outline_entries`` is what bounds the search.
    """
    data = np.zeros((1, OUTLINE_SELECTION_CAPACITY, 2), dtype=np.int32)
    return gfx.Texture(data, dim=2, format="2xi4")


def update_outline_selection(texture: gfx.Texture, selection: dict[int, int]) -> int:
    """Write *selection* into *texture* in place; return the entry count.

    Parameters
    ----------
    texture : gfx.Texture
        A texture from :func:`build_outline_selection_texture`.
    selection : dict[int, int]
        ``{label value: palette slot}``.  Slots are clamped into ``1..15``:
        that is the range the outline key's partition reserves for selected
        labels, and the palette cap the ``global_id`` LUT already imposes.
        Entries beyond ``OUTLINE_SELECTION_CAPACITY`` are dropped with a
        warning.

    Returns
    -------
    int
        Number of entries written, for ``n_outline_entries``.
    """
    keys_sorted = sorted(selection)
    if len(keys_sorted) > OUTLINE_SELECTION_CAPACITY:
        warnings.warn(
            f"{len(keys_sorted)} labels selected for outlining, which exceeds "
            f"the capacity of {OUTLINE_SELECTION_CAPACITY}. Only the entries "
            f"with the smallest label IDs will be outlined.",
            stacklevel=2,
        )
        keys_sorted = keys_sorted[:OUTLINE_SELECTION_CAPACITY]

    data = texture.data
    data[...] = 0
    for index, label in enumerate(keys_sorted):
        data[0, index, 0] = np.int32(label)
        data[0, index, 1] = np.int32(min(max(int(selection[label]), 1), 15))
    texture.update_range((0, 0, 0), texture.size)
    return len(keys_sorted)
