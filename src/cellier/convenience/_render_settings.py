"""Renderer post-processing settings on the convenience viewers.

``Viewer`` and ``OrthoViewer`` both expose these, and they are pure
delegation to the controller, so they live in one mixin rather than in two
copies.  Every setter goes through
``CellierController.update_render_config_field``, which means a
notebook-cell assignment reaches any connected widget rather than leaving
the panel showing a stale value.

The three features:

* **Outlines** draw a screen-space contour around chosen visuals.  Which
  visuals, and in which palette slot, is a *per-visual* setting reached
  through ``viewer.controller.set_visual_outline``; what is here is the
  configuration shared by all of them.
* **Ambient occlusion** darkens creases by sampling the depth buffer.  3D
  only.
* **Temporal accumulation** averages successive jittered frames, which is
  what lets the raymarcher and the occlusion kernel use few samples each
  and still settle to a clean image.  3D only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.controller import CellierController
    from cellier.render._config import RenderManagerConfig


class RenderSettingsMixin:
    """Renderer post-processing settings, delegated to the controller."""

    controller: CellierController

    # -- Everything at once --------------------------------------------

    @property
    def render_config(self) -> RenderManagerConfig:
        """Live rendering configuration.

        Mutating a field here changes the model but not the GPU state, and
        notifies no widget.  The named properties below do all three; reach
        for this to read the whole configuration or to serialise it.
        """
        return self.controller.render_config

    # -- Outlines -------------------------------------------------------

    @property
    def outline_enabled(self) -> bool:
        """Whether the screen-space outline pass is active."""
        return self.controller.outline_enabled

    @outline_enabled.setter
    def outline_enabled(self, value: bool) -> None:
        self.controller.outline_enabled = value

    @property
    def outline_boundaries_enabled(self) -> bool:
        """Whether the boundaries layer (every outlined region) draws."""
        return self.controller.outline_boundaries_enabled

    @outline_boundaries_enabled.setter
    def outline_boundaries_enabled(self, value: bool) -> None:
        self.controller.outline_boundaries_enabled = value

    @property
    def outline_selection_enabled(self) -> bool:
        """Whether the selection layer (regions with a palette slot) draws."""
        return self.controller.outline_selection_enabled

    @outline_selection_enabled.setter
    def outline_selection_enabled(self, value: bool) -> None:
        self.controller.outline_selection_enabled = value

    # -- Ambient occlusion ----------------------------------------------

    @property
    def ambient_occlusion_enabled(self) -> bool:
        """Whether the ambient occlusion pass is active.  3D only."""
        return self.controller.ambient_occlusion_enabled

    @ambient_occlusion_enabled.setter
    def ambient_occlusion_enabled(self, value: bool) -> None:
        self.controller.ambient_occlusion_enabled = value

    @property
    def ambient_occlusion_radius(self) -> float | None:
        """Occlusion hemisphere radius in scene units, or ``None`` for auto.

        ``None`` derives it from the scene bounding box diagonal, which is
        the only default that means anything across cellier's coordinate
        systems.  `ambient_occlusion_effective_radius` reports what that came to.
        """
        return self.controller.ambient_occlusion_radius

    @ambient_occlusion_radius.setter
    def ambient_occlusion_radius(self, value: float | None) -> None:
        self.controller.ambient_occlusion_radius = value

    @property
    def ambient_occlusion_auto_radius_fraction(self) -> float:
        """Fraction of the scene bounding box diagonal used when radius is auto."""
        return self.controller.ambient_occlusion_auto_radius_fraction

    @ambient_occlusion_auto_radius_fraction.setter
    def ambient_occlusion_auto_radius_fraction(self, value: float) -> None:
        self.controller.ambient_occlusion_auto_radius_fraction = value

    @property
    def ambient_occlusion_effective_radius(self) -> float | None:
        """The occlusion radius actually in use, in scene units.  Read-only."""
        return self.controller.ambient_occlusion_effective_radius

    @property
    def ambient_occlusion_strength(self) -> float:
        """How far the occlusion is applied, 0 (off) to 1 (full)."""
        return self.controller.ambient_occlusion_strength

    @ambient_occlusion_strength.setter
    def ambient_occlusion_strength(self, value: float) -> None:
        self.controller.ambient_occlusion_strength = value

    @property
    def ambient_occlusion_power(self) -> float:
        """Contrast exponent applied to the occlusion before the multiply."""
        return self.controller.ambient_occlusion_power

    @ambient_occlusion_power.setter
    def ambient_occlusion_power(self, value: float) -> None:
        self.controller.ambient_occlusion_power = value

    @property
    def ambient_occlusion_bias(self) -> float:
        """Depth-comparison bias, as a fraction of the effective radius."""
        return self.controller.ambient_occlusion_bias

    @ambient_occlusion_bias.setter
    def ambient_occlusion_bias(self, value: float) -> None:
        self.controller.ambient_occlusion_bias = value

    @property
    def ambient_occlusion_n_samples(self) -> int:
        """Hemisphere samples per pixel.  Changing this recompiles the shader."""
        return self.controller.ambient_occlusion_n_samples

    @ambient_occlusion_n_samples.setter
    def ambient_occlusion_n_samples(self, value: int) -> None:
        self.controller.ambient_occlusion_n_samples = value

    @property
    def ambient_occlusion_blur_radius(self) -> int:
        """Occlusion box-blur half-width in internal pixels.  Recompiles."""
        return self.controller.ambient_occlusion_blur_radius

    @ambient_occlusion_blur_radius.setter
    def ambient_occlusion_blur_radius(self, value: int) -> None:
        self.controller.ambient_occlusion_blur_radius = value

    # -- Temporal accumulation ------------------------------------------

    @property
    def temporal_enabled(self) -> bool:
        """Whether the temporal accumulation pass is active.  3D only."""
        return self.controller.temporal_enabled

    @temporal_enabled.setter
    def temporal_enabled(self, value: bool) -> None:
        self.controller.temporal_enabled = value

    @property
    def temporal_blend_weight(self) -> float:
        """Minimum EMA blend weight for the current frame, in ``(0, 1]``.

        Lower values give a smoother settled image and take longer to get
        there after a camera move.
        """
        return self.controller.temporal_blend_weight

    @temporal_blend_weight.setter
    def temporal_blend_weight(self, value: float) -> None:
        self.controller.temporal_blend_weight = value

    def reset_temporal_accumulation(self) -> None:
        """Discard the accumulated history on every canvas."""
        self.controller.reset_temporal_accumulation()
