"""The shared per-visual lookup table.

Maps a pygfx ``global_id`` (what the pick buffer stores, and what
``wobject.id`` returns on the Python side) to a one-byte entry carrying
every per-visual flag the effect passes need.

Two features read it: the screen-space outline pass (bits 0-6) and the
ambient occlusion pass (bit 7).  **One table, one authoritative map.**
``apply`` is a whole-state sync -- ids present in the table but absent from
the supplied mapping are cleared -- so if the two features kept separate
maps, enabling outlines would silently wipe every occlusion exclusion.
``RenderManager`` therefore holds a single ``{visual_id: VisualFlags}`` map
and derives the whole table from it.

The table is a 1024x1024 ``r8uint`` texture -- 1 MB -- indexed as
``(id & 1023, id >> 10)``.  It is deliberately *not* smaller: pygfx does
not allocate ids with a counter, it draws them randomly over the full
2^20 range::

    id = 0
    while id in self._ids_in_use:
        id = random.randint(1, 1_048_575)

so a scene with ten objects has ten ids scattered anywhere in a
million-wide space and any smaller table would miss nearly all of them.
The upside of a full-range table is that it is a sparse scatter: changing
what is outlined is a handful of single-texel writes, never a full
re-upload.

Ids are not stable across sessions, so the table is always built at
runtime from cellier's visual-to-world-object mapping and never persisted.

Entry layout, one byte:

===== =========================================================
bits  field
===== =========================================================
0-3   selection slot; 0 = not selected, else palette index v - 1
4-5   kind; 0 = not outlined, 1 = whole object, 2 = label
6     placement; 0 = inward, 1 = outward
7     excluded from ambient occlusion
===== =========================================================

Bit 7 is the last free bit.  If a third per-object flag is ever needed the
format goes ``r8uint`` -> ``r16uint`` (2 MB) and the index arithmetic is
unchanged.

Entry 0 is permanently inert: the pick target clears to zero and pygfx's
``IdProvider`` seeds ``_ids_in_use = {0}``, so no object is ever given id
0 and background needs no special case in the shader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import wgpu

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

LUT_WIDTH: int = 1024
LUT_HEIGHT: int = 1024

#: Highest usable selection slot.  Slot 0 means "not selected".
MAX_SLOT: int = 15

KIND_NONE: int = 0
KIND_WHOLE_OBJECT: int = 1
KIND_LABEL: int = 2

PLACEMENT_INWARD: int = 0
PLACEMENT_OUTWARD: int = 1

PLACEMENT_NAMES: dict[str, int] = {
    "inward": PLACEMENT_INWARD,
    "outward": PLACEMENT_OUTWARD,
}

#: Bit 7: the visual does not receive ambient occlusion.
AO_EXCLUDED_BIT: int = 0x80


def lut_index(object_id: int) -> tuple[int, int]:
    """Return the ``(x, y)`` texel holding *object_id*'s entry.

    Parameters
    ----------
    object_id : int
        A pygfx ``global_id`` (``wobject.id``), in ``[0, 2**20)``.

    Returns
    -------
    tuple[int, int]
        Column and row into the 1024x1024 table.
    """
    object_id = int(object_id)
    return object_id & (LUT_WIDTH - 1), (object_id >> 10) & (LUT_HEIGHT - 1)


def encode_entry(
    slot: int,
    kind: int = KIND_WHOLE_OBJECT,
    placement: int = PLACEMENT_INWARD,
    ao_excluded: bool = False,
) -> int:
    """Pack a per-visual entry into its single byte.

    Parameters
    ----------
    slot : int
        Selection slot in ``[0, 15]``.  0 means "outlined by the
        boundaries layer only", not "not outlined" -- ``kind`` decides
        that.
    kind : int
        ``KIND_NONE``, ``KIND_WHOLE_OBJECT`` or ``KIND_LABEL``.
    placement : int
        ``PLACEMENT_INWARD`` or ``PLACEMENT_OUTWARD``.
    ao_excluded : bool
        Whether the visual is excluded from *receiving* ambient occlusion.
        Independent of every other field: a visual can be excluded from
        occlusion and not outlined at all, which is what a MIP volume in a
        scene with no selection looks like.

    Returns
    -------
    int
        The packed byte.

    Raises
    ------
    ValueError
        If any field is out of range.
    """
    if not 0 <= slot <= MAX_SLOT:
        raise ValueError(f"slot must be in [0, {MAX_SLOT}], got {slot}")
    if kind not in (KIND_NONE, KIND_WHOLE_OBJECT, KIND_LABEL):
        raise ValueError(f"unknown outline kind: {kind}")
    if placement not in (PLACEMENT_INWARD, PLACEMENT_OUTWARD):
        raise ValueError(f"unknown outline placement: {placement}")
    return (
        (slot & 0xF)
        | ((kind & 0x3) << 4)
        | ((placement & 0x1) << 6)
        | (AO_EXCLUDED_BIT if ao_excluded else 0)
    )


def decode_entry(value: int) -> tuple[int, int, int, bool]:
    """Unpack an entry byte into ``(slot, kind, placement, ao_excluded)``."""
    value = int(value)
    return (
        value & 0xF,
        (value >> 4) & 0x3,
        (value >> 6) & 0x1,
        bool(value & AO_EXCLUDED_BIT),
    )


class VisualLut:
    """GPU-resident ``global_id`` -> per-visual flags table.

    One table serves every canvas: pygfx ids are unique per process, so
    the mapping is global and the texture can be shared by all outline
    passes on the same device.

    Parameters
    ----------
    device : wgpu.GPUDevice or None
        Device to allocate on.  Defaults to pygfx's shared device.
    """

    def __init__(self, device: wgpu.GPUDevice | None = None) -> None:
        if device is None:
            from pygfx.renderers.wgpu.engine.shared import get_shared

            device = get_shared().device
        self._device = device
        self._data = np.zeros((LUT_HEIGHT, LUT_WIDTH), dtype=np.uint8)
        self._written: dict[int, int] = {}

        self._texture = device.create_texture(
            size=(LUT_WIDTH, LUT_HEIGHT, 1),
            format=wgpu.TextureFormat.r8uint,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d",
        )
        self._view = self._texture.create_view()
        # Upload the cleared table once so the texture is never read
        # uninitialised.
        device.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            self._data,
            {"offset": 0, "bytes_per_row": LUT_WIDTH, "rows_per_image": LUT_HEIGHT},
            (LUT_WIDTH, LUT_HEIGHT, 1),
        )

    @property
    def view(self) -> wgpu.GPUTextureView:
        """Bindable view on the table."""
        return self._view

    @property
    def texture(self) -> wgpu.GPUTexture:
        """The underlying table texture."""
        return self._texture

    @property
    def entries(self) -> dict[int, int]:
        """Copy of the currently written ``{object_id: entry byte}`` map."""
        return dict(self._written)

    def get_entry(self, object_id: int) -> int:
        """Return the entry byte currently stored for *object_id*."""
        x, y = lut_index(object_id)
        return int(self._data[y, x])

    def set_entry(self, object_id: int, value: int) -> bool:
        """Write one entry, uploading only if it changed.

        Parameters
        ----------
        object_id : int
            The pygfx ``global_id`` to write.
        value : int
            Packed entry byte, from :func:`encode_entry`.

        Returns
        -------
        bool
            ``True`` if the table changed and was uploaded.
        """
        object_id = int(object_id)
        if object_id == 0:
            # Entry 0 is background and must stay inert.
            return False
        value = int(value) & 0xFF
        x, y = lut_index(object_id)
        if int(self._data[y, x]) == value and self._written.get(object_id) == value:
            return False
        self._data[y, x] = value
        if value:
            self._written[object_id] = value
        else:
            self._written.pop(object_id, None)
        self._upload_texel(x, y, value)
        return True

    def clear_entry(self, object_id: int) -> bool:
        """Reset every flag for *object_id*."""
        return self.set_entry(object_id, 0)

    def apply(self, entries: Mapping[int, int]) -> bool:
        """Make the table match *entries* exactly.

        Ids present in the table but absent from *entries* are cleared, so
        this is the whole-state sync used once per frame.  Only texels that
        actually differ are uploaded.

        Parameters
        ----------
        entries : Mapping[int, int]
            ``{object_id: entry byte}`` for every flag that should be
            set right now, across *all* features that share the table.
            Anything absent is cleared, so callers must supply the whole
            state rather than their own feature's slice of it.

        Returns
        -------
        bool
            ``True`` if any texel changed.
        """
        changed = False
        for stale_id in set(self._written) - set(entries):
            changed |= self.clear_entry(stale_id)
        for object_id, value in entries.items():
            changed |= self.set_entry(object_id, value)
        return changed

    def clear(self) -> bool:
        """Reset every written entry."""
        return self.apply({})

    def _upload_texel(self, x: int, y: int, value: int) -> None:
        self._device.queue.write_texture(
            {
                "texture": self._texture,
                "mip_level": 0,
                "origin": (x, y, 0),
            },
            np.array([[value]], dtype=np.uint8),
            {"offset": 0, "bytes_per_row": 1, "rows_per_image": 1},
            (1, 1, 1),
        )


_SHARED_LUT: VisualLut | None = None


def get_shared_visual_lut() -> VisualLut:
    """Return the process-wide visual LUT, creating it on first use.

    pygfx allocates ``global_id`` from a single process-wide
    ``IdProvider`` and every canvas shares one wgpu device, so the
    id-to-entry mapping is global and one 1 MB table serves every canvas.

    Returns
    -------
    VisualLut
        The shared table.
    """
    global _SHARED_LUT
    if _SHARED_LUT is None:
        _SHARED_LUT = VisualLut()
    return _SHARED_LUT


def peek_shared_visual_lut() -> VisualLut | None:
    """Return the shared table if it exists, without creating it.

    Lets callers skip work on a process where neither feature that reads
    the table has ever needed it, which is the default.
    """
    return _SHARED_LUT
