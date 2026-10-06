"""Image visual models shared by both storage types, and the in-memory family.

One image visual per storage type (unified image design D1).  Single-channel
and composite viewing are two modes of the same visual: ``composite`` switches
between them, ``single`` holds the one appearance single mode draws with, and
``channels`` holds the per-channel appearances composite mode draws with
(sections 3.1 and 3.3).
"""

from __future__ import annotations

from typing import Any, Literal

from cmap import Colormap
from psygnal import EventedModel
from pydantic import ConfigDict, Field, model_validator

from cellier.visuals._base_visual import BaseDrawAppearance, BaseVisual
from cellier.visuals._render_plane import PLANE_RENDER_MODE, RenderPlane

__all__ = [
    "BaseImageAppearance",
    "BaseImageSingleAppearance",
    "BaseImageVisual",
    "ImageVisual",
    "InMemoryImageAppearance",
    "InMemoryImageChannelAppearance",
    "InMemoryImageSingleAppearance",
    "check_channels_share_render_mode",
    "effective_transparency_mode",
]

TransparencyMode = Literal[
    "blend", "add", "multiply", "weighted_blend", "weighted_solid"
]


def _default_colormap() -> Colormap:
    return Colormap("gray")


class BaseImageAppearance(BaseDrawAppearance):
    """Appearance shared by both modes of an image visual.

    Parameters
    ----------
    visible : bool
        The master switch.  In single mode it is the only visibility; in
        composite mode each channel's ``visible`` is gated by it (D12).
    transparency_mode : str or None
        The pygfx alpha mode for every drawn channel.  ``None`` (the default)
        is automatic: ``"blend"`` in single mode and ``"add"`` in composite
        mode, following the mode as it switches.  An explicit value always
        wins.
    interpolation : str
        Texture sampler filter.  ``"nearest"`` (default) or ``"linear"``.
    """

    model_config = ConfigDict(validate_assignment=True)

    transparency_mode: TransparencyMode | None = None
    interpolation: Literal["linear", "nearest"] = "nearest"


class BaseImageSingleAppearance(EventedModel):
    """The mode-dependent fields both image families share (D11).

    Parameters
    ----------
    color_map : cmap.Colormap
        Colormap applied after contrast normalisation.  Accepts any
        cmap-registered name.  Default ``"gray"``.
    clim : tuple[float, float]
        Contrast limits ``(min, max)``.  Default ``(0.0, 1.0)``.
    opacity : float
        Opacity in ``[0, 1]``.  Default ``1.0``.
    iso_threshold : float
        Isosurface threshold for the iso render modes.
    """

    model_config = ConfigDict(validate_assignment=True)

    color_map: Colormap = Field(default_factory=_default_colormap)
    clim: tuple[float, float] = (0.0, 1.0)
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    iso_threshold: float = 0.5


class InMemoryImageAppearance(BaseImageAppearance):
    """Shared appearance of an in-memory image visual.

    Every field has a default, so a plain image needs no ``appearance=``.
    """


class InMemoryImageSingleAppearance(BaseImageSingleAppearance):
    """Single-mode appearance of an in-memory image visual.

    Parameters
    ----------
    render_mode : str
        Rendering mode for the 3D view: ``"mip"`` (default), ``"iso"``,
        ``"minip"``, or ``"plane"``, which draws the data on the visual's
        ``render_planes`` instead of as a volume.  Ignored by the 2D view.
    """

    render_mode: Literal["mip", "iso", "minip", "plane"] = "mip"


class InMemoryImageChannelAppearance(InMemoryImageSingleAppearance):
    """One channel's appearance on an in-memory image, in composite mode.

    Parameters
    ----------
    visible : bool
        Whether composite mode draws this channel.  Gated by the shared
        ``appearance.visible``.  Default ``True``.
    """

    visible: bool = True


class BaseImageVisual(BaseVisual):
    """The fields and rules both image families share (design 3.1).

    Parameters
    ----------
    channel_axis : int or None
        The data axis a composite draws channels along.  Fixed at
        construction (D20).  ``None`` means the image has no channels:
        composite is unavailable and the visual keeps one slot (D34).  The
        axis must map to a world axis through ``transform``; the controller
        checks that, since the model has no transform to check against when it
        is built.
    composite : bool
        ``False`` (single mode) draws the one sample the world sliders select,
        with ``single``.  ``True`` (composite mode) draws every visible entry
        of ``channels``, each with its own appearance, and ignores the slice
        position along the channel axis.

        **Assign it through ``CellierController.set_image_composite``.**  That
        refuses to composite an axis the scene displays (design 3.4).  A
        direct assignment is not checked, because the check needs the scene's
        ``displayed_axes``; the reslice and the events still happen.
    channels : dict[int, channel appearance]
        Per-channel appearances, keyed by index along ``channel_axis``.  Empty
        is valid and draws nothing in composite mode (D16).

        **Every channel has the same ``render_mode``**, hidden ones
        included: the dict is refused otherwise, here and on assignment.
        Change it with ``CellierController.set_image_render_mode``, which
        sets every channel together; assigning it to one channel of several
        is refused.  ``single.render_mode`` is separate.
    max_channels : int
        The most channels ``channels`` may hold, in 2D and 3D alike (D15).
        Default 4.
    render_planes : tuple[RenderPlane, ...]
        The planes the visual draws its data on in a 3D view while its
        render mode is ``"plane"``; ignored in every other mode and in a 2D
        view.  World space; see :class:`~cellier.visuals.RenderPlane`.  At
        most four.  Assign a new tuple to change them: one assignment is one
        event.  Default empty, which in ``"plane"`` mode draws nothing.
    """

    model_config = ConfigDict(validate_assignment=True)

    channel_axis: int | None = Field(default=None, frozen=True)
    composite: bool = Field(
        default=False,
        description=(
            "Draw the visible channels together instead of the one the "
            "sliders select.  Set it with CellierController.set_image_composite, "
            "which refuses a composited axis the scene displays; a direct "
            "assignment skips that check."
        ),
    )
    max_channels: int = Field(default=4, ge=1)
    render_planes: tuple[RenderPlane, ...] = ()

    @model_validator(mode="after")
    def _validate_modes(self) -> BaseImageVisual:
        if self.composite and self.channel_axis is None:
            raise ValueError(
                "composite=True requires a channel_axis: there is no axis to "
                "draw channels along."
            )
        channels = getattr(self, "channels", {})
        if len(channels) > self.max_channels:
            raise ValueError(
                f"channels has {len(channels)} entries but max_channels="
                f"{self.max_channels}.  Raise max_channels or remove a channel."
            )
        check_channels_share_render_mode(channels, self.name)
        return self

    def __setattr__(self, name: str, value: object) -> None:
        """Keep the old ``channels`` when a new dict is refused.

        The rules on ``channels`` (one render mode, ``max_channels``) are
        checked after pydantic has assigned the field, and a failed check
        does not undo the assignment.
        """
        if name != "channels":
            super().__setattr__(name, value)
            return
        previous = self.channels
        try:
            super().__setattr__(name, value)
        except ValueError:
            self.__dict__["channels"] = previous
            raise

    def drawn_channels(self, size: int | None = None) -> tuple[int, ...]:
        """The channel indices composite mode draws, ascending.

        ``D = {k in channels : appearance.visible and channels[k].visible and
        0 <= k < size}`` (design 3.3).  Empty when the visual is hidden.

        Parameters
        ----------
        size : int or None
            The length of the channel axis.  ``None`` skips the range check.

        Returns
        -------
        tuple[int, ...]
            The drawn indices.
        """
        if not self.appearance.visible:
            return ()
        return tuple(
            sorted(
                index
                for index, channel in self.channels.items()
                if channel.visible and (size is None or 0 <= index < size)
            )
        )

    def plane_mode(self) -> bool:
        """Whether a 3D view draws this visual on its ``render_planes``.

        Single mode reads ``single.render_mode``.  Composite mode reads its
        drawn channels, which share one render mode like all the channels.
        """
        if not self.composite:
            return self.single.render_mode == PLANE_RENDER_MODE
        drawn = self.drawn_channels()
        return bool(drawn) and all(
            self.channels[index].render_mode == PLANE_RENDER_MODE for index in drawn
        )

    def draws_nothing(self, size: int | None = None) -> bool:
        """Whether the visual draws nothing and so needs no slicing (3.3).

        Hidden, or in composite mode with no drawn channel.
        """
        if not self.appearance.visible:
            return True
        return self.composite and not self.drawn_channels(size)


def check_channels_share_render_mode(channels: Any, name: str = "image") -> None:
    """Raise unless every channel appearance has the same ``render_mode``.

    Parameters
    ----------
    channels : dict[int, channel appearance]
        An image visual's ``channels``.
    name : str
        The visual's name, for the message.

    Raises
    ------
    ValueError
        Naming each channel's mode.
    """
    modes = {index: channel.render_mode for index, channel in channels.items()}
    if len(set(modes.values())) > 1:
        raise ValueError(
            f"The channels of image {name!r} have different render modes "
            f"{dict(sorted(modes.items()))}.  Every channel of an image has "
            "the same render mode; set it with "
            "CellierController.set_image_render_mode."
        )


def effective_transparency_mode(visual: Any) -> str:
    """The alpha mode an image visual draws its channels with.

    ``appearance.transparency_mode`` when set; otherwise ``"add"`` in
    composite mode and ``"blend"`` in single mode.

    Parameters
    ----------
    visual : BaseImageVisual
        The image visual.

    Returns
    -------
    str
        A pygfx alpha mode.
    """
    explicit = visual.appearance.transparency_mode
    if explicit is not None:
        return explicit
    return "add" if visual.composite else "blend"


class ImageVisual(BaseImageVisual):
    """Model-layer visual for a single-resolution in-memory image.

    Backed by an ``ImageMemoryStore``.  Camera movement does **not** trigger a
    reslice because the data is not view-dependent -- the whole slice is
    always loaded.

    Parameters
    ----------
    visual_type : Literal["image_memory"]
        Discriminator field; always ``"image_memory"``.
    appearance : InMemoryImageAppearance
        Shared by both modes.
    single : InMemoryImageSingleAppearance
        Single mode's appearance.
    channels : dict[int, InMemoryImageChannelAppearance]
        Composite mode's per-channel appearances.
    requires_camera_reslice : bool
        Always ``False``; frozen.
    """

    visual_type: Literal["image_memory"] = "image_memory"
    appearance: InMemoryImageAppearance = Field(default_factory=InMemoryImageAppearance)
    single: InMemoryImageSingleAppearance = Field(
        default_factory=InMemoryImageSingleAppearance
    )
    channels: dict[int, InMemoryImageChannelAppearance] = Field(default_factory=dict)
    requires_camera_reslice: bool = Field(default=False, frozen=True)
