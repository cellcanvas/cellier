"""Qt panel for the screen-space outline settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from cellier.gui._render_controls import RENDER_SECTION_TITLES
from cellier.gui.qt.render._base import QtRenderConfigPanel

if TYPE_CHECKING:
    from cellier.render._config import OutlineConfig


class QtOutlineControls(QtRenderConfigPanel):
    """Screen-space outline settings on the cellier bus.

    Two layers run in the same fragment invocation and can both be active:
    *boundaries* draws every outlined region, *selection* draws only the
    regions carrying a palette slot.

    Every control here comes from one spec shared with the notebook panel
    (``AnywidgetOutlineControls``) -- including the absence of a colour
    control on the selection
    layer, whose colour comes from the palette slot rather than from its
    own field.

    Which visuals are outlined, and in which slot, is a per-visual setting
    rather than a render-config one; see
    ``CellierController.set_visual_outline`` and ``set_label_selection``.

    Parameters
    ----------
    config :
        The live ``OutlineConfig`` to read initial values from -- typically
        ``controller.render_config.outline``.
    parent :
        Optional Qt parent widget.
    slot_usage :
        Zero-argument callable returning ``{slot: how many visuals use it}``,
        annotating the palette swatches.  Derived state -- a visual moving
        between slots changes it with no config field changing -- so it is a
        callable rather than a value.  ``None`` draws no annotation.
    """

    section: ClassVar[str] = "outline"
    title: ClassVar[str] = RENDER_SECTION_TITLES["outline"]

    def __init__(self, config: OutlineConfig, *, parent=None, slot_usage=None) -> None:
        # Forwarded before ``_build_from_spec``, which is what draws the
        # palette and reads it.  The anywidget twin takes ``**kwargs`` and so
        # always forwarded it; this one dropped the keyword on the floor,
        # which is half of why the annotation never appeared on either side.
        super().__init__(parent, slot_usage=slot_usage)
        self._build_from_spec(config)
