from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from cellier.visuals._base_visual import BaseAppearance, BaseVisual
from cellier.visuals._render_plane import PLANE_RENDER_MODE, RenderPlane

OutlineMode = Literal["per_label", "whole_object", "all_boundaries", "depth_threshold"]
"""How a labels visual is outlined; see ``BaseLabelsVisual``."""

DEFAULT_OUTLINE_DEPTH_RANGE: tuple[float, float] = (0.01, 0.06)
"""The default ``outline_depth_range`` of a labels visual."""


def validate_outline_depth_range(value) -> tuple[float, float]:
    """Return *value* as an ``outline_depth_range``, or raise.

    Parameters
    ----------
    value :
        ``(start, solid)``: two numbers with ``0 < start <= 1`` and
        ``start <= solid``.

    Returns
    -------
    tuple[float, float]
        The pair as floats.

    Raises
    ------
    ValueError
        If *value* is not such a pair.
    """
    try:
        start, solid = (float(part) for part in value)
    except (TypeError, ValueError):
        raise ValueError(
            f"outline_depth_range must be a pair (start, solid); got {value!r}."
        ) from None
    if not 0.0 < start <= 1.0:
        raise ValueError(
            f"the start of outline_depth_range must be in (0, 1]; got {start}."
        )
    if not solid >= start:
        raise ValueError(
            "the solid value of outline_depth_range must not be below its "
            f"start; got {(start, solid)}."
        )
    return start, solid


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
    render_mode : str
        3D rendering mode.

        ``"iso_categorical"`` (the default) draws the voxel faces of every
        label, lit by the face normals.  ``"flat_categorical"`` draws the
        same faces unlit.

        ``"smooth_iso"`` draws the same voxel faces, lit by a normal that
        varies smoothly across them, so a label looks rounded while its
        outline stays on the voxel grid.  It costs about the same as
        ``"iso_categorical"``.

        ``"smooth_geometry"`` draws a smooth surface per label instead of
        voxel faces: each label's voxels are blended trilinearly and the
        surface sits where a label carries more than
        ``smooth_geometry_level`` of the weight.  At the default level, flat
        sides stay on the voxel faces, convex corners round off and concave
        ones fill in, so outlines and depth are smooth too.  Where two
        labels touch, each is rounded on its own, which leaves a small
        groove along their shared edge.  A single voxel still draws, as a
        small blob.  It is several times as expensive as the other modes.

        ``"plane"`` draws the labels on the visual's ``render_planes``
        instead of as a volume.
    smooth_geometry_level : float
        Where the ``"smooth_geometry"`` surface sits: the share of the
        blended weight a label must carry, in [0.05, 0.95].  The default,
        0.5, puts flat sides exactly on the voxel faces.  Lower grows every
        label -- voxels that touch only along an edge or at a corner join up,
        thin structures look solid, and the grooves between touching labels
        narrow -- and higher shrinks it.  No effect in the other modes.
    """

    colormap_mode: Literal["random", "direct"] = Field(default="random", frozen=True)
    background_label: int = 0
    salt: int = 0
    color_dict: dict[int, tuple[float, float, float, float]] = Field(
        default_factory=dict
    )
    render_mode: Literal[
        "iso_categorical", "flat_categorical", "smooth_iso", "smooth_geometry", "plane"
    ] = "iso_categorical"
    smooth_geometry_level: float = Field(default=0.5, ge=0.05, le=0.95)


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

        ``"depth_threshold"`` is ``"all_boundaries"`` with one thing more:
        a band where one part of a label passes in front of another part of
        the *same* label, as where one branch of a tree crosses another.
        How large a jump in depth is drawn, and how strongly, is
        :attr:`outline_depth_range`.  With ``outline.placement = "inward"``
        that band lies on the nearer surface, along its edge; with
        ``"outward"`` on the farther one, just outside the edge of the
        nearer.  It draws depth bands in a 3D view of a surface render mode
        only: in 2D and in ``"plane"`` mode there is no jump in depth, and
        it is ``"all_boundaries"``.  :attr:`outline_selected_labels` has no
        effect in it either.

        With ``outline.placement = "outward"`` a boundary between two
        touching labels comes out double-width, since each side casts a halo
        onto the other.  ``"inward"`` is the default for labels visuals.
    outline_depth_range : tuple[float, float]
        ``(start, solid)`` of the depth bands of ``"depth_threshold"``
        mode, both as fractions of the distance from the camera to the
        nearer of the two surfaces.  A jump in depth of ``start`` is where
        a line begins, at no alpha; from ``solid`` up the line is drawn in
        full; between the two its alpha rises smoothly.  ``start ==
        solid`` is a hard edge.  Needs ``0 < start <= 1`` and ``start <=
        solid``.  Kept, and without effect, in the other modes.

        The surface of a labels volume is made of voxel faces, and the step
        from one voxel to the next is a real jump in depth.  Zoomed in far
        enough that a voxel covers several pixels, or while a coarse level
        of a multiscale volume is on screen, those steps are outlined too.
        A larger ``start`` removes them.

        The pair is checked when the visual is built and when it is set
        through the controller (``update_visual_render_field``); assign it
        there rather than on the model.
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
    ``depth_threshold`` label field    ``outline.slot``
    ==================  =============  ==========================

    ``depth_threshold`` has the kind of ``all_boundaries``; what sets it
    apart is that its entry also carries :attr:`outline_depth_range`.

    ``render_planes`` are the planes the visual draws its labels on in a 3D
    view while ``appearance.render_mode`` is ``"plane"``; ignored otherwise.
    World space, at most four; see :class:`~cellier.visuals.RenderPlane`.

    ``outline_selected_labels`` is kept on the visual rather than pushed
    straight to the GPU because a labels material can be rebuilt underneath
    it -- the multiscale visual rebuilds its materials whenever the
    displayed level shapes change -- and a selection that lives only in a
    texture is lost when that happens.
    """

    outline_mode: OutlineMode = "per_label"
    outline_selected_labels: dict[int, int] = Field(default_factory=dict)
    outline_depth_range: tuple[float, float] = DEFAULT_OUTLINE_DEPTH_RANGE
    render_planes: tuple[RenderPlane, ...] = ()

    @field_validator("outline_depth_range")
    @classmethod
    def _check_outline_depth_range(cls, value):
        return validate_outline_depth_range(value)

    def plane_mode(self) -> bool:
        """Whether a 3D view draws this visual on its ``render_planes``."""
        return self.appearance.render_mode == PLANE_RENDER_MODE


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
