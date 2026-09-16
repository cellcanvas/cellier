from __future__ import annotations

from typing import Literal

from pydantic import Field

from cellier.visuals._base_visual import BaseAppearance, BaseVisual

OutlineMode = Literal["per_label", "whole_object", "all_boundaries"]
"""How a labels visual is outlined; see ``BaseLabelsVisual``."""


class BaseLabelsAppearance(BaseAppearance):
    """Base appearance parameters shared by all label visuals.

    Parameters
    ----------
    colormap_mode : "random" or "direct"
        Frozen — raises ``ValidationError`` on mutation.
    background_label : int
        Label ID treated as transparent (discarded). Default 0.
    salt : int
        Hash seed for random colormap mode. Default 0.
    color_dict : dict
        Explicit label-ID → RGBA mapping for direct mode.
    render_mode : "iso_categorical" or "flat_categorical"
        3D rendering mode.
    """

    colormap_mode: Literal["random", "direct"] = Field(default="random", frozen=True)
    background_label: int = 0
    salt: int = 0
    color_dict: dict[int, tuple[float, float, float, float]] = Field(
        default_factory=dict
    )
    render_mode: Literal["iso_categorical", "flat_categorical"] = "iso_categorical"


class InMemoryLabelsAppearance(BaseLabelsAppearance):
    """Appearance parameters for an in-memory label visual."""


class BaseLabelsVisual(BaseVisual):
    """Shared base for the in-memory and multiscale labels visuals.

    Exists to declare :attr:`outline_selected_labels` once.  A labels visual
    is outlined *per label* rather than as one silhouette, so it needs a
    second piece of state the other visual types do not have.

    Parameters
    ----------
    outline_mode : str
        How the outline pass treats this volume.

        ``"per_label"`` (the default) makes each label its own region: a
        band appears between touching labels, and the selection colour
        comes from :attr:`outline_selected_labels` rather than from the
        visual's own slot.

        ``"whole_object"`` treats the volume as one region, exactly as a
        mesh or an intensity volume is treated -- one band around the union
        of every label, coloured by ``outline.slot``, and no lines between
        touching labels.  Also the only mode available on a canvas built
        without the ``outline_id`` render target, since per-label keys live
        there.

        ``"all_boundaries"`` bands every label, boundaries between touching
        labels included, all in the single colour of ``outline.slot``.  It
        is whole-volume's colour behaviour with per-label's boundaries, and
        :attr:`outline_selected_labels` has no effect in it -- not merely no
        colour effect: the selection is suppressed on the GPU, because a
        selected label's outline key *is* its slot number and two touching
        labels sharing a slot would then share a key and lose the boundary
        between them.  The model field is left untouched, so switching back
        to ``"per_label"`` restores the selection.

        With ``outline.placement = "outward"`` a boundary between two
        touching labels comes out double-width, since each side casts a halo
        onto the other.  ``"inward"`` is the default for labels visuals.
    outline_selected_labels : dict[int, int]
        Maps a label value to the palette slot the selection layer draws it
        in.  Empty (the default) means no label is selected, so an outlined
        labels visual shows boundaries only.  The visual's own
        ``outline.slot`` makes it eligible for the boundaries layer; this is
        what puts a selection colour on individual labels.  Only meaningful
        in ``"per_label"`` mode; see :attr:`outline_mode`.

    Notes
    -----
    The mode is carried as ``kind`` in the LUT entry, which decides two
    independent things -- what the outline key is, and where the colour
    comes from:

    ==================  =============  ==========================
    mode                key            colour
    ==================  =============  ==========================
    ``whole_object``    pick id        ``outline.slot``
    ``per_label``       label field    ``outline_selected_labels``
    ``all_boundaries``  label field    ``outline.slot``
    ==================  =============  ==========================

    ``outline_selected_labels`` is kept on the visual rather than pushed
    straight to the GPU because a labels material can be rebuilt underneath
    it -- the multiscale visual rebuilds its materials whenever the
    displayed level shapes change -- and a selection that lives only in a
    texture is lost when that happens.
    """

    outline_mode: OutlineMode = "per_label"
    outline_selected_labels: dict[int, int] = Field(default_factory=dict)


class LabelMemoryVisual(BaseLabelsVisual):
    """Model-layer visual for in-memory label arrays.

    Parameters
    ----------
    appearance : InMemoryLabelsAppearance
        Colormap and rendering appearance.
    """

    visual_type: Literal["label_memory"] = "label_memory"
    appearance: InMemoryLabelsAppearance
    requires_camera_reslice: bool = Field(default=False, frozen=True)
