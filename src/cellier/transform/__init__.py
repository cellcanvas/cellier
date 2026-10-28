"""Coordinate systems, and the transforms between them.

A transform here names the two coordinate systems it maps between, rather
than carrying only a matrix.  That is what lets the viewer refuse one built
against a look-alike pair of spaces, what makes a pyramid's per-level maps
compose without a downsampling assumption, and what a region needs in order
to be pulled back through anything at all.

Built alongside an earlier, unnamed transform layer under this same path --
the two were deliberately independent, neither importing from the other (D1)
-- and took the name over when that layer was retired in Phase 8 of the
integration.
"""

from cellier.transform._affine import AffineTransform
from cellier.transform._alignment import (
    COUPLING_TOLERANCE,
    SliceAlignment,
    ViewClassification,
    classify_view,
    effective_transform,
)
from cellier.transform._axis import Axis, AxisRef, AxisSampling, AxisType
from cellier.transform._base import BaseTransform
from cellier.transform._by_dimension import ByDimensionTransform, TransformBlock
from cellier.transform._coordinate_system import (
    CoordinateSystem,
    CoordinateSystemType,
    DataCoordinateSystem,
    RenderedCoordinateSystem,
    VisualCoordinateSystem,
    WorldCoordinateSystem,
)
from cellier.transform._geometry import AxisAlignedBoundingBox, Plane
from cellier.transform._geometry_ops import (
    DegenerateNormalError,
    NonAffineTransformError,
    NonInvertibleTransformError,
)
from cellier.transform._nonuniform import (
    AxisCoordinates,
    NonUniformAxisTransform,
)
from cellier.transform._region import ConvexRegion, HalfSpace
from cellier.transform._selection import RegionSelection
from cellier.transform._slice_plane import (
    BRICK_ROUTE_AXES,
    DepthAlignment,
    ProjectionGeometry,
    ProjectionRefusal,
    ProjectionRefusalReason,
    RouteDecision,
    RouteRefusalReason,
    SlicePlaneFrame,
    SliceRoute,
    choose_slice_route,
    classify_projection,
    projection_axis,
    ray_step,
    slice_footprint,
    slice_plane_frame,
)
from cellier.transform._types import TransformType

__all__ = [
    "BRICK_ROUTE_AXES",
    "COUPLING_TOLERANCE",
    "AffineTransform",
    "Axis",
    "AxisAlignedBoundingBox",
    "AxisCoordinates",
    "AxisRef",
    "AxisSampling",
    "AxisType",
    "BaseTransform",
    "ByDimensionTransform",
    "ConvexRegion",
    "CoordinateSystem",
    "CoordinateSystemType",
    "DataCoordinateSystem",
    "DegenerateNormalError",
    "DepthAlignment",
    "HalfSpace",
    "NonAffineTransformError",
    "NonInvertibleTransformError",
    "NonUniformAxisTransform",
    "Plane",
    "ProjectionGeometry",
    "ProjectionRefusal",
    "ProjectionRefusalReason",
    "RegionSelection",
    "RenderedCoordinateSystem",
    "RouteDecision",
    "RouteRefusalReason",
    "SliceAlignment",
    "SlicePlaneFrame",
    "SliceRoute",
    "TransformBlock",
    "TransformType",
    "ViewClassification",
    "VisualCoordinateSystem",
    "WorldCoordinateSystem",
    "choose_slice_route",
    "classify_projection",
    "classify_view",
    "effective_transform",
    "projection_axis",
    "ray_step",
    "slice_footprint",
    "slice_plane_frame",
]
