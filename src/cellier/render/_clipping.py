"""Clipping planes in the render layer (clipping planes design v2, 4.1, 4.6).

A visual's planes are authored in its level-0 data coordinates and may
have a component on any data axis.  ``reduce_clipping_planes`` turns them
into what one view tests against: ``(a, b, c, d)`` in the scene's rendered
space, in pygfx order, kept where ``a*x + b*y + c*z >= d``.  In a 2D view
the depth component is zero and the result is the line where the plane
meets the slice.

``ClippingPlanesMixin`` gives a render visual the one seam through which
planes reach its materials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.data._plane_clip import PlaneTuple, plane_tuples
from cellier.render._spaces import affine_for_node

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from cellier.render._spaces import RenderSpaces
    from cellier.transform import BaseTransform

#: What a disabled plane is uploaded as.  It keeps everything
#: (``0 >= -1``), so toggling ``enabled`` never changes the plane count and
#: never recompiles a shader.
KEEP_EVERYTHING: tuple[float, float, float, float] = (0.0, 0.0, 0.0, -1.0)


def reduce_clipping_planes(
    spaces: RenderSpaces,
    data_to_world: BaseTransform,
    constants: Mapping[int, float],
    planes: Sequence[Any],
) -> list[tuple[float, float, float, float]]:
    """Reduce data-space clipping planes to one view's rendered space.

    Takes the same ``constants`` as :func:`~cellier.render._spaces.node_matrix`
    and must be given the same values: the plane is cut by the slice the
    node is drawn at.

    For a plane ``n . p >= d``: the collapsed axes are substituted
    (``d_eff = d - sum(n[h] * constants[h])``), and the rest is carried
    through ``r = L x + tau``, the retained data axes to rendered space:
    ``n' = L^-T n_retained``, ``d' = d_eff + n' . tau``.  It never calls
    ``transform.map_plane``, so a non-affine collapsed axis (a non-uniform
    time axis) is fine.

    Parameters
    ----------
    spaces : RenderSpaces
        The systems the visual is placed with.
    data_to_world : BaseTransform
        The visual's own transform.
    constants : Mapping[int, float]
        ``{collapsed data axis: data position}``.
    planes : Sequence[ClippingPlane]
        The visual's planes, in level-0 data coordinates.

    Returns
    -------
    list[tuple[float, float, float, float]]
        One ``(a, b, c, d)`` per plane, in pygfx ``(x, y, z)`` order.  A
        disabled plane gives :data:`KEEP_EVERYTHING`.  A plane parallel to
        the slice gives ``(0, 0, 0, d')``: everything or nothing.
    """
    if not planes:
        return []
    rendered = affine_for_node(data_to_world, constants).then(
        spaces.world_to_rendered, spaces.world, spaces.rendered
    )
    ndim = rendered.linear.shape[1]
    retained = [axis for axis in range(ndim) if axis not in constants]
    linear = np.asarray(rendered.linear, dtype=np.float64)[:, retained]
    tau = np.asarray(rendered.translation, dtype=np.float64)
    out: list[tuple[float, float, float, float]] = []
    for item in planes:
        if not item.enabled:
            out.append(KEEP_EVERYTHING)
            continue
        normal = item.plane.normal
        offset = item.plane.offset - sum(
            float(normal[axis]) * float(value) for axis, value in constants.items()
        )
        reduced = np.linalg.solve(linear.T, normal[retained])
        abc = np.zeros(3, dtype=np.float64)
        abc[: len(reduced)] = reduced[::-1]
        out.append(
            (float(abc[0]), float(abc[1]), float(abc[2]), offset + float(reduced @ tau))
        )
    return out


def expand_rendered_plane(
    spaces: RenderSpaces,
    data_to_world: BaseTransform,
    constants: Mapping[int, float],
    point: Sequence[float],
    normal: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Turn a plane of a 3D view's rendered space into a data-space plane.

    The inverse of :func:`reduce_clipping_planes` for one plane, for a pose
    a gizmo reports: the plane through *point* that keeps the side *normal*
    points to.  With ``r = L x + tau``: ``n_retained = L^T n'`` and
    ``d = n' . point - n' . tau``.  The components on the collapsed axes
    are zero, so the plane holds at every position of those axes.

    The data-space normal is not rescaled: on anisotropic data a unit
    rendered normal gives a data normal of another length, and the offset
    is in the same scale.

    Parameters
    ----------
    spaces : RenderSpaces
        The systems the visual is placed with.
    data_to_world : BaseTransform
        The visual's own transform.
    constants : Mapping[int, float]
        ``{collapsed data axis: data position}``, as for
        :func:`reduce_clipping_planes`.
    point : Sequence[float]
        A point on the plane, in rendered space, pygfx ``(x, y, z)`` order.
    normal : Sequence[float]
        The normal there, in the same order.  It points into the kept
        half-space.

    Returns
    -------
    tuple[np.ndarray, float]
        The normal, one entry per data axis, and the offset: the plane
        ``normal . p >= offset`` in level-0 data coordinates.

    Raises
    ------
    ValueError
        If the view does not keep three data axes, or *normal* is zero.
    """
    rendered = affine_for_node(data_to_world, constants).then(
        spaces.world_to_rendered, spaces.world, spaces.rendered
    )
    ndim = rendered.linear.shape[1]
    retained = [axis for axis in range(ndim) if axis not in constants]
    if len(retained) != 3:
        raise ValueError(
            "A rendered plane can be expanded for a 3D view only; this view "
            f"keeps {len(retained)} data axes."
        )
    linear = np.asarray(rendered.linear, dtype=np.float64)[:, retained]
    tau = np.asarray(rendered.translation, dtype=np.float64)
    xyz = np.asarray(normal, dtype=np.float64)
    if not np.any(xyz):
        raise ValueError("A plane's normal must not be all zero.")
    # Rendered order is pygfx order reversed, as in the reduction.
    reduced = xyz[::-1]
    expanded = np.zeros(ndim, dtype=np.float64)
    expanded[retained] = linear.T @ reduced
    offset = float(xyz @ np.asarray(point, dtype=np.float64)) - float(reduced @ tau)
    return expanded, offset


def data_half_space_rows(
    planes: Sequence[Any],
    retained_axes: Sequence[int],
    constants: Mapping[int, float],
) -> np.ndarray:
    """The enabled planes as culling rows over the retained data axes.

    Rows are ``(n_x, n_y, n_z, w)`` (2D: the depth entry is zero) in level-0
    data coordinates of the retained axes in pygfx order, kept where
    ``dot(row[:3], p) + row[3] >= 0``: the form the frustum rows of brick
    and tile culling take, so the two concatenate.

    Parameters
    ----------
    planes : Sequence[ClippingPlane]
        The visual's planes.
    retained_axes : Sequence[int]
        The data axes the view keeps, ascending.
    constants : Mapping[int, float]
        ``{collapsed data axis: data position}``.

    Returns
    -------
    np.ndarray
        ``(n_enabled, 4)`` float64.
    """
    rows = []
    retained = list(retained_axes)
    for item in planes:
        if not item.enabled:
            continue
        normal = item.plane.normal
        offset = item.plane.offset - sum(
            float(normal[axis]) * float(value) for axis, value in constants.items()
        )
        row = np.zeros(4, dtype=np.float64)
        kept = normal[retained][::-1]
        row[: len(kept)] = kept
        row[3] = -offset
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def drawn_materials(*nodes: Any) -> list[Any]:
    """The materials of every image and volume under *nodes*.

    Lines are left out: the bounding-box wireframe sits under the same
    nodes and is not clipped.
    """
    import pygfx as gfx

    found = []
    for node in nodes:
        if node is None:
            continue
        for obj in node.iter(lambda o: isinstance(o, (gfx.Volume, gfx.Image))):
            material = getattr(obj, "material", None)
            if material is not None:
                found.append(material)
    return found


def notify_planes_placed(visual: Any) -> None:
    """Tell whoever draws *visual*'s plane outlines that they may be stale.

    Called by the render planes and clipping planes mixins whenever a
    visual has reduced its planes again: new planes, a new outline, a new
    view, a node placed at another slice.  The scene manager sets
    ``_planes_placed_listener`` on the visuals it holds.
    """
    listener = getattr(visual, "_planes_placed_listener", None)
    if listener is not None:
        listener()


class ClippingPlanesMixin:
    """The one way clipping planes reach a render visual's materials.

    The planes are state of the render visual, not of a material: a
    material that is created or swapped in later must be given them.  A
    host class:

    - implements :meth:`_clip_targets`;
    - calls :meth:`_apply_clipping_planes` wherever it places its node (the
      slice the node is drawn at may have moved) and wherever it creates or
      assigns a material.

    It reads ``self._spaces`` and ``self._transform``.
    """

    _clipping_planes: tuple = ()

    #: Whether a change of planes changes what this visual reads, so the
    #: controller reslices it.  ``True`` for multiscale image and labels
    #: (clipped bricks and tiles are not fetched).
    clipping_planes_affect_request: bool = False

    @property
    def clipping_planes(self) -> tuple:
        """The planes last given to this visual."""
        return self._clipping_planes

    def set_clipping_planes(self, planes: Iterable[Any]) -> None:
        """Adopt *planes* and write them to every material."""
        self._clipping_planes = tuple(planes)
        self._apply_clipping_planes()

    def on_clipping_planes_changed(self, event: Any) -> None:
        """Bus handler for ``ClippingPlanesChangedEvent``."""
        self.set_clipping_planes(event.clipping_planes)

    def clip_frame(self) -> tuple[Any, Any, dict[int, float]] | None:
        """What a plane of this visual is reduced with, for the view drawn.

        Returns
        -------
        tuple or None
            ``(spaces, data_to_world, constants)`` as
            :func:`reduce_clipping_planes` takes them, with the constants of
            the first group :meth:`_clip_targets` yields.  ``None`` for a
            visual that has not been placed.
        """
        spaces = getattr(self, "_spaces", None)
        transform = getattr(self, "_transform", None)
        if spaces is None or transform is None:
            return None
        for _materials, constants in self._clip_targets():
            return spaces, transform, dict(constants)
        return None

    def outline_box(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The visual's data box in level-0 data coordinates, if it has one.

        ``(low, high)`` over the retained data axes, as
        :func:`~cellier.render._plane_outline.box_frame` takes them.  The
        volume visuals give the voxel-edge box their shaders cut with.
        """
        return None

    def outline_frame(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The box this visual's plane outlines are cut by, in rendered space.

        Plane outline design 5.1 and 5.3.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] or None
            ``(rows, corners)``: six half-space rows and eight corners, in
            pygfx ``(x, y, z)``.  ``None`` when the visual is not placed in
            a 3D view or its box is not known.
        """
        from cellier.render._plane_outline import box_frame

        frame = self.clip_frame()
        box = self.outline_box()
        if frame is None or box is None:
            return None
        spaces, transform, constants = frame
        if len(spaces.retained_axes) != 3:
            return None
        return box_frame(spaces, transform, constants, *box)

    def _clip_targets(self) -> Iterable[tuple[Iterable[Any], Mapping[int, float]]]:
        """Yield ``(materials, constants)`` groups.

        ``constants`` are the collapsed data positions the group's node is
        drawn at.  Materials held aside (an empty placeholder, the real
        material while the placeholder is shown) are included, so one that
        comes back does not bring stale planes.
        """
        raise NotImplementedError

    def _apply_clipping_planes(self) -> None:
        """Reduce the stored planes and write them to every material."""
        spaces = getattr(self, "_spaces", None)
        transform = getattr(self, "_transform", None)
        planes = self._clipping_planes
        if getattr(self, "_clip_on_cpu", False):
            # The read clips; a reduced plane would cut the flattened
            # geometry a second time, at the wrong place.
            planes = ()
        placed = spaces is not None and transform is not None
        for materials, constants in self._clip_targets():
            if not planes:
                reduced: list = []
            elif not placed:
                continue
            else:
                reduced = reduce_clipping_planes(spaces, transform, constants, planes)
            for material in materials:
                if material is None:
                    continue
                if not reduced and not material.clipping_planes:
                    continue
                material.clipping_planes = reduced
        # A render plane's outline is cut by these planes (plane outline
        # design 3.1), at the slice they were just reduced for.
        notify_planes_placed(self)


class GeometryClippingMixin(ClippingPlanesMixin):
    """Clipping for geometry that a view may flatten (design 4.2, 5.2).

    The shader is exact whenever the rendered position determines the
    position on every axis a plane has a component on.  Geometry flattened
    along such an axis (a 2D view of a slab, a trail along time) has lost
    that coordinate, so the read clips it instead, by its true position,
    and the shader does not.

    A host calls :meth:`_begin_request_clipping` when it builds a request.
    """

    #: Whether the last request was clipped in the read.
    _clip_on_cpu: bool = False

    def outline_frame(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The box of the geometry drawn, in rendered space.

        A geometry visual has no shape: its box is the bounding box of the
        vertices it draws, in its node's frame, as its bounding-box
        wireframe is.  ``None`` in a 2D view, before there is geometry, and
        while the read clips: geometry flattened along an axis a plane
        acts on has no one cut face.
        """
        from cellier.render._plane_outline import node_box_frame

        if self._clip_on_cpu:
            return None
        frame = self.clip_frame()
        if frame is None or len(frame[0].retained_axes) != 3:
            return None
        found = self._outline_node_box()
        if found is None:
            return None
        node, low, high = found
        return node_box_frame(np.asarray(node.world.matrix), low, high)

    def _outline_node_box(self) -> tuple[Any, Any, Any] | None:
        """``(node, low, high)``: the drawn geometry's box in a node's frame."""
        raise NotImplementedError

    def _wants_cpu_clip(self) -> bool:
        """Whether an enabled plane has a component on a collapsed axis."""
        spaces = getattr(self, "_spaces", None)
        if spaces is None or not self._clipping_planes:
            return False
        collapsed = list(spaces.collapsed_axes)
        if not collapsed:
            return False
        return any(
            item.enabled and bool(np.any(item.plane.normal[collapsed] != 0.0))
            for item in self._clipping_planes
        )

    @property
    def clipping_planes_affect_request(self) -> bool:
        """A change of planes needs a read when the read clips, or did."""
        return self._clip_on_cpu or self._wants_cpu_clip()

    def _begin_request_clipping(self) -> tuple[PlaneTuple, ...]:
        """Decide where this request is clipped, and tell the materials.

        Returns
        -------
        tuple[PlaneTuple, ...]
            The planes the read applies, as ``(normal, offset)`` pairs in
            data coordinates; empty when the shader clips.
        """
        self._clip_on_cpu = self._wants_cpu_clip()
        self._apply_clipping_planes()
        return plane_tuples(self._clipping_planes) if self._clip_on_cpu else ()
