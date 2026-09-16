import uuid
from typing import Annotated, Literal
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, ConfigDict, Field

from cellier.render._config import MAX_OUTLINE_SLOT
from cellier.transform import TransformType


class AABBParams(EventedModel):
    """Parameters for an axis-aligned bounding box wireframe overlay.

    Parameters
    ----------
    enabled : bool
        If True, display the AABB wireframe. Default False.
    color : str
        Line color as a CSS color string. Default ``"#ffffff"``.
    line_width : float
        Line thickness in screen pixels. Default ``2.0``.
    """

    enabled: bool = False
    color: str = "#ffffff"
    line_width: float = 2.0


class VisualOutline(EventedModel):
    """Screen-space outline assignment for one visual.

    The *global* outline configuration -- palette, thicknesses, layer
    enables, the contrast band -- lives in
    ``RenderManagerConfig.outline``, because it is shared by every outlined
    visual.  What is here is only the part that is genuinely about this one
    visual: which palette slot it uses, and which side of its own edge the
    band sits on.

    Parameters
    ----------
    slot : int
        ``0`` (the default) means the visual is not outlined.  ``1`` upward
        selects palette entry ``slot - 1`` from
        ``RenderManagerConfig.outline.palette`` for the selection layer; any
        nonzero slot also makes the visual visible to the boundaries layer.
    placement : str or None
        ``"inward"`` puts the band inside the region's own footprint, so
        the region never appears to grow, but one thinner than twice the
        band is consumed entirely.  ``"outward"`` puts it outside, reading
        as a halo and leaving the region intact.  ``None`` (the default)
        defers to the per-type rule -- outward for lines, points and
        graphs, whose default footprints are a few screen pixels wide, and
        inward for everything else.  Left as ``None`` rather than resolved
        at construction so a visual that never asked for a placement
        follows the rule if the rule changes.

    Notes
    -----
    Outlining requires ``pick_write``: the pass is a screen-space
    post-process and the pick buffer is the only per-pixel identity channel
    it has.  Setting a nonzero slot on a visual with ``pick_write = False``
    turns picking back on and warns.
    """

    # Validate on assignment: these bounds exist to be enforced against a
    # GUI slider and a user's literal, not only against a constructor.
    model_config = ConfigDict(validate_assignment=True)

    slot: int = Field(default=0, ge=0, le=MAX_OUTLINE_SLOT)
    placement: Literal["inward", "outward"] | None = None


class BaseDrawAppearance(EventedModel):
    """How any visual is drawn, apart from its opacity.

    Split out of :class:`BaseAppearance` so an image, whose opacity is per
    channel (unified image design 3.1), does not inherit a visual-level
    ``opacity`` that would mean nothing.

    Parameters
    ----------
    visible : bool
        If True, the visual is visible.  Default True.
    render_order : int
        Pygfx node render order.  Objects with higher values are drawn later
        and therefore appear on top when depth values are equal.  Default 0.
    depth_test : bool
        Whether the material tests against the depth buffer.
    depth_write : bool
        Whether the material writes to the depth buffer.
    depth_compare : str
        The depth comparison.
    transparency_mode : str
        The pygfx alpha mode.  Default ``"blend"``.
    """

    visible: bool = True
    render_order: int = 0
    depth_test: bool = True
    depth_write: bool = True
    depth_compare: Literal["<", "<=", "==", "!=", ">=", ">"] = "<"
    transparency_mode: Literal[
        "blend", "add", "multiply", "weighted_blend", "weighted_solid"
    ] = "blend"


class BaseAppearance(BaseDrawAppearance):
    """Base model for the materials of every non-image visual.

    Parameters
    ----------
    opacity : float
        Master opacity multiplier in [0, 1].  Default 1.0.
    """

    opacity: float = Field(default=1.0, ge=0.0, le=1.0)


class BaseVisual(EventedModel):
    """The base class for all Visuals.

    Parameters
    ----------
    name : str
        Human-readable label for the visual.
    data_store_id : str
        UUID string of the data store this visual reads from.
    pick_write : bool
        If True, the visual can be picked in the canvas via the picking
        buffer. Default True.
    transform : BaseTransform or None
        The ``data -> world`` affine transform, from the store's level-0
        coordinate system to the scene's world.  ``None`` (the default) means
        "not decided yet": ``CellierController.add_visual`` builds the
        identity between those two systems and assigns it.

        There is no coordinate-system-less identity to default to (D18).  A
        transform names its endpoints by id, and a visual constructed before
        it is added to a scene does not yet know which world it is going into.
    aabb : AABBParams
        Axis-aligned bounding box wireframe parameters. Default disabled.
    outline : VisualOutline
        Screen-space outline assignment. Default is not outlined.
    ambient_occlusion : bool or None
        Whether this visual receives ambient occlusion.  ``None`` (the
        default) is automatic: excluded while it renders in a MIP-family
        mode, which writes the depth of an extremum sample rather than of a
        surface, and included otherwise.  ``True`` and ``False`` are
        explicit and survive a render-mode change.
    id : UUID4
        Unique identifier for the visual. Auto-generated; do not set manually.

    Notes
    -----
    Each concrete visual subclass declares its own typed ``appearance`` field.
    Image visuals also carry a ``single`` appearance and per-channel
    ``channels`` appearances (unified image design 3.1).
    """

    name: str
    data_store_id: str
    pick_write: bool = True
    transform: TransformType | None = None
    requires_camera_reslice: bool = Field(default=False, frozen=True)
    aabb: AABBParams = Field(default_factory=AABBParams)
    outline: VisualOutline = Field(default_factory=VisualOutline)
    ambient_occlusion: bool | None = None

    # store a UUID to identify this specific visual
    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
