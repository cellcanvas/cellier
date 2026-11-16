"""The three notebook render-settings panels.

Each is the ``AnywidgetRenderConfigPanel`` base plus whatever its feature
needs that a control spec cannot express.  The prose explaining what each
feature *is* lives on the Qt twins in ``cellier.gui.qt.render``; repeating
it here would be two copies of one explanation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from cellier.gui._render_controls import (
    RADIUS_SLIDER_HEADROOM,
    RENDER_SECTION_TITLES,
)
from cellier.gui.anywidget.render._base import AnywidgetRenderConfigPanel

if TYPE_CHECKING:
    from collections.abc import Callable

    from cellier.render._config import (
        AmbientOcclusionConfig,
        OutlineConfig,
        TemporalAccumulationConfig,
    )

#: Frames after which the exponential average has essentially converged at
#: the default alpha of 0.1.  Only used to word the readout.
_SETTLED_FRAMES = 10


class AnywidgetOutlineControls(AnywidgetRenderConfigPanel):
    """Screen-space outline settings for the notebook GUI.

    See `cellier.gui.qt.QtOutlineControls` for what the two
    layers are and why the selection layer has no colour control.
    """

    section: ClassVar[str] = "outline"

    def __init__(self, config: OutlineConfig, **kwargs) -> None:
        super().__init__(config, title=RENDER_SECTION_TITLES["outline"], **kwargs)


class AnywidgetAmbientOcclusionControls(AnywidgetRenderConfigPanel):
    """Ambient occlusion settings for the notebook GUI.

    See `cellier.gui.qt.QtAmbientOcclusionControls` for what the radius
    means and why it is derived from the scene by default.

    Parameters
    ----------
    config :
        The live ``AmbientOcclusionConfig`` -- typically ``controller.render_config.ambient_occlusion``.
    effective_radius :
        Called to read the radius actually in use, in scene units.  ``None``
        omits the readout.
    """

    section: ClassVar[str] = "ambient_occlusion"

    def __init__(
        self,
        config: AmbientOcclusionConfig,
        *,
        effective_radius: Callable[[], float | None] | None = None,
        **kwargs,
    ) -> None:
        # Read before super().__init__: _describe runs during it.
        self._effective_radius = effective_radius
        super().__init__(
            config, title=RENDER_SECTION_TITLES["ambient_occlusion"], **kwargs
        )
        if effective_radius is not None:
            self._add_readout("Radius in use", self._describe_radius)

    def _describe(self, control):
        """Give the radius slider a range, which its spec cannot carry.

        A radius is in scene units, so no absolute maximum means anything
        across cellier's coordinate systems -- a bounding box may be 96
        units or 0.0003.  The range comes from the radius actually in use.
        """
        described = super()._describe(control)
        if control.field == "radius" and described["max"] is None:
            effective = self._read_effective_radius()
            described["max"] = max(effective * RADIUS_SLIDER_HEADROOM, 1.0)
            described["step"] = described["max"] / 1000.0
        return described

    def _read_effective_radius(self) -> float:
        """The radius in use, or 1.0 when there is no canvas to ask."""
        if self._effective_radius is not None:
            value = self._effective_radius()
            if value:
                return float(value)
        return 1.0

    def _describe_radius(self) -> str:
        value = self._effective_radius()
        if not value:
            return "no canvas yet"
        return f"{value:.4g} scene units"


class AnywidgetTemporalControls(AnywidgetRenderConfigPanel):
    """Temporal accumulation settings for the notebook GUI.

    See `cellier.gui.qt.QtTemporalControls` for what the pass
    does and why the convergence readout is worth its space.

    Parameters
    ----------
    config :
        The live ``TemporalAccumulationConfig``.
    frame_count :
        Called to read how many frames have accumulated on the
        least-converged canvas.  ``None`` omits the readout.
    on_reset :
        Called when the user asks to discard the accumulated history.  An
        action rather than a setting, so it does not travel on the bus.
        ``None`` omits the button.
    """

    section: ClassVar[str] = "temporal"

    def __init__(
        self,
        config: TemporalAccumulationConfig,
        *,
        frame_count: Callable[[], int | None] | None = None,
        on_reset: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, title=RENDER_SECTION_TITLES["temporal"], **kwargs)
        self._frame_count = frame_count
        self._on_reset = on_reset
        if frame_count is not None:
            self._add_readout("State", self._describe_convergence)
        if on_reset is not None:
            self.action_label = "Reset history"
            self.observe(self._on_action, names=["_action_clicks"])

    def _describe_convergence(self) -> str:
        count = self._frame_count() if self._frame_count is not None else None
        if count is None:
            return "no canvas yet"
        if count == 0:
            return "restarting"
        if count < _SETTLED_FRAMES:
            return f"settling ({count} frames)"
        return f"settled ({count} frames)"

    def _on_action(self, _change) -> None:
        if self._on_reset is not None:
            self._on_reset()
            self.refresh_readouts()
