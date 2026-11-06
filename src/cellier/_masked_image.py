"""Controller-side checks of a MaskedImage's two stores (masked design 4.3, 5.2).

A MaskedImage reads a **mask** store and a **value** store.  These pure
functions resolve the value's placement and refuse a pair that cannot be drawn
together, each with the reason in the message, before any render object is
built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from cellier.transform import AffineTransform

if TYPE_CHECKING:
    from cellier.events import MaskedImagePickInfo
    from cellier.transform import CoordinateSystem
    from cellier.visuals._masked_image import (
        MaskedImageRenderConfig,
        MaskedImageVisual,
    )

__all__ = [
    "MaskedPick",
    "check_in_memory_fits",
    "check_masked_pair",
    "resolve_value_to_mask",
    "value_world_transform",
    "visual_store_ids",
]


def visual_store_ids(visual: Any) -> tuple[UUID, ...]:
    """Every store *visual* reads: its own, and a MaskedImage's value store."""
    ids = [UUID(str(visual.data_store_id))]
    value_store_id = getattr(visual, "value_store_id", None)
    if value_store_id is not None:
        ids.append(UUID(str(value_store_id)))
    return tuple(ids)


def resolve_value_to_mask(
    value_to_mask: AffineTransform | np.ndarray | None,
    value_system: CoordinateSystem,
    mask_system: CoordinateSystem,
) -> AffineTransform | None:
    """*value_to_mask* as a transform between the two stores' level-0 systems.

    Parameters
    ----------
    value_to_mask : AffineTransform, array-like or None
        A transform, an augmented matrix (value level-0 data coordinates to
        mask level-0 data coordinates, in data axis order), or ``None`` (the
        two share a grid).
    value_system, mask_system : CoordinateSystem
        The value's and the mask's level-0 data coordinate systems.

    Returns
    -------
    AffineTransform or None
        ``None`` passes through.

    Raises
    ------
    ValueError
        If a transform's endpoints are not the two systems.
    """
    if value_to_mask is None:
        return None
    if not isinstance(value_to_mask, AffineTransform):
        return AffineTransform.from_matrix(
            np.asarray(value_to_mask, dtype=np.float64),
            value_system,
            mask_system,
            name="value_to_mask",
        )
    if (
        value_to_mask.input_coordinate_system != value_system.id
        or value_to_mask.output_coordinate_system != mask_system.id
    ):
        raise ValueError(
            "value_to_mask must map the value store's level-0 system "
            f"({value_system.id}, '{value_system.name}') into the mask store's "
            f"({mask_system.id}, '{mask_system.name}'); it maps "
            f"{value_to_mask.input_coordinate_system} into "
            f"{value_to_mask.output_coordinate_system}."
        )
    return value_to_mask


def value_world_transform(
    transform: AffineTransform,
    value_to_mask: AffineTransform | None,
    value_system: CoordinateSystem,
    world: CoordinateSystem,
) -> AffineTransform:
    """The value image's world transform: ``transform @ value_to_mask`` (MI-P5).

    Parameters
    ----------
    transform : AffineTransform
        The visual's (the mask's) data-to-world transform.
    value_to_mask : AffineTransform or None
        Value level-0 to mask level-0; ``None`` is the identity.
    value_system : CoordinateSystem
        The value store's level-0 system, the result's input.
    world : CoordinateSystem
        The scene's world system, the result's output.
    """
    matrix = np.asarray(transform.matrix, dtype=np.float64)
    if value_to_mask is not None:
        matrix = matrix @ np.asarray(value_to_mask.matrix, dtype=np.float64)
    return AffineTransform.from_matrix(
        matrix, value_system, world, name="value_to_world"
    )


def _space_axes(system: CoordinateSystem) -> list[int]:
    return [i for i, axis in enumerate(system.axes) if axis.axis_type == "space"]


def _other_axes(system: CoordinateSystem) -> list[int]:
    return [i for i, axis in enumerate(system.axes) if axis.axis_type != "space"]


def _non_spatial_world_axes(
    system: CoordinateSystem, transform: AffineTransform, side: str
) -> dict[int, int]:
    """``{data axis: world axis}`` for *system*'s non-spatial axes.

    Raises
    ------
    ValueError
        If a non-spatial data axis maps to no single world axis.
    """
    correspondence = transform.exclusive_axis_correspondence()
    out: dict[int, int] = {}
    for axis in _other_axes(system):
        if axis not in correspondence:
            name = system.axes[axis].name
            raise ValueError(
                f"The {side} store's axis {axis} ('{name}', "
                f"{system.axes[axis].axis_type}) maps to no single world axis.  "
                "A MaskedImage's non-spatial axes are sliders, and each must "
                "follow one world axis."
            )
        out[axis] = correspondence[axis]
    return out


def check_masked_pair(
    mask_system: CoordinateSystem,
    value_system: CoordinateSystem,
    mask_transform: AffineTransform,
    value_to_mask: AffineTransform | None,
    world: CoordinateSystem,
) -> None:
    """Refuse a mask and value pair that cannot be drawn together (design 4.3).

    Refused: a store with fewer than three spatial axes (MI-P4); stores whose
    non-spatial axes do not correspond, one to one, to the same world axes (no
    broadcast); and a ``value_to_mask`` that couples a slider axis to a
    spatial axis.

    Parameters
    ----------
    mask_system, value_system : CoordinateSystem
        The two stores' level-0 data coordinate systems.
    mask_transform : AffineTransform
        The mask's data-to-world transform (the visual's).
    value_to_mask : AffineTransform or None
        Value level-0 to mask level-0; ``None`` is the identity.
    world : CoordinateSystem
        The scene's world system.

    Raises
    ------
    ValueError
        On any of the above, naming the axes.
    """
    for side, system in (("mask", mask_system), ("value", value_system)):
        n_space = len(_space_axes(system))
        if n_space < 3:
            names = tuple(axis.name for axis in system.axes)
            raise ValueError(
                f"The {side} store has {n_space} spatial axes {names}; a "
                "MaskedImage draws a surface and needs three."
            )
    if value_to_mask is None:
        if mask_system.ndim != value_system.ndim or [
            axis.axis_type for axis in mask_system.axes
        ] != [axis.axis_type for axis in value_system.axes]:
            raise ValueError(
                "With no value_to_mask the two stores share a grid, so their "
                "axes must match: the mask has "
                f"{tuple((a.name, a.axis_type) for a in mask_system.axes)}, the "
                f"value {tuple((a.name, a.axis_type) for a in value_system.axes)}."
                "  Give value_to_mask to place the value."
            )
    else:
        coupling = value_to_mask.axis_coupling()
        mask_space = set(_space_axes(mask_system))
        value_space = set(_space_axes(value_system))
        for row in range(coupling.shape[0]):
            for column in np.flatnonzero(coupling[row]):
                if (row in mask_space) != (int(column) in value_space):
                    raise ValueError(
                        f"value_to_mask couples the value's axis {int(column)} "
                        f"('{value_system.axes[int(column)].name}') to the "
                        f"mask's axis {row} ('{mask_system.axes[row].name}').  "
                        "A slider axis may only map to a slider axis, and a "
                        "spatial axis to spatial axes."
                    )
    value_transform = value_world_transform(
        mask_transform, value_to_mask, value_system, world
    )
    mask_axes = _non_spatial_world_axes(mask_system, mask_transform, "mask")
    value_axes = _non_spatial_world_axes(value_system, value_transform, "value")
    mask_world = sorted(mask_axes.values())
    value_world = sorted(value_axes.values())
    if mask_world != value_world:

        def names(indices: list[int]) -> tuple[str, ...]:
            return tuple(world.axes[i].name for i in indices)

        raise ValueError(
            f"The mask's non-spatial axes follow world axes {names(mask_world)} "
            f"and the value's {names(value_world)}.  They must be the same, one "
            "to one: a value is not broadcast over a slider it does not have."
        )


def check_in_memory_fits(
    store: Any,
    system: CoordinateSystem,
    side: str,
    field: str,
    render_config: MaskedImageRenderConfig,
) -> None:
    """Refuse an in-memory side whose one level does not fit its atlas (MI-D15).

    An in-memory side is single scale and loads every brick of the current
    slice, since its whole image is the fallback level; the atlas must hold
    them all, or part of it would never be drawn.  Only the three spatial axes
    count: the others are sliced.  The slot count is the one the slots compute
    from the budget (overlap 3), less the reserved slot 0.

    Parameters
    ----------
    store : Any
        The side's store; only an ``ImageMemoryStore`` is checked.
    system : CoordinateSystem
        Its level-0 system, to find the spatial axes.
    side : str
        ``"mask"`` or ``"value"``, for the message.
    field : str
        The render-config field that is the side's budget.
    render_config : MaskedImageRenderConfig
        Block size and budgets.

    Raises
    ------
    ValueError
        Naming the side, the bricks needed, the slots the budget gives, and
        the two ways out.
    """
    from cellier.data import ImageMemoryStore
    from cellier.render.block_cache._cache_parameters_3d import (
        compute_block_cache_parameters_3d,
    )

    if not isinstance(store, ImageMemoryStore):
        return
    shape = tuple(int(store.level_shapes[0][i]) for i in _space_axes(system))
    block_size = int(render_config.block_size)
    budget = int(getattr(render_config, field))
    needed = int(np.prod([-(-n // block_size) for n in shape]))
    params = compute_block_cache_parameters_3d(block_size, budget, overlap=3)
    available = int(params.n_slots) - 1
    if needed > available:
        raise ValueError(
            f"The in-memory {side} image, spatial shape {shape}, needs {needed} "
            f"bricks of {block_size} voxels at once, and its GPU budget of "
            f"{budget / 1024**2:.0f} MiB ({field}) holds {available}.  An "
            "in-memory side of a MaskedImage is single scale and loads all of "
            f"it: raise {field}, or use a multiscale store."
        )


# -- picks (masked image design 9.3) -------------------------------------------

#: Samples along the normal before the step widens: the shader's
#: ``PROJECTION_CAP`` (``render/visuals/_masked_image.py``).
_PROJECTION_CAP = 64
#: A hit this close to an enabled clipping plane, in mask voxels, with the
#: mask inside there, is on the plane's cut face (the shader's
#: ``on_cut_face``: the ray's clipped start was already inside).
_CUT_FACE_VOXELS = 0.5


def _surface_interval(thickness: float, placement: str) -> tuple[float, float]:
    """The placed interval along the inward normal (design 6.2)."""
    if placement == "inward":
        return 0.0, thickness
    if placement == "outward":
        return -thickness, 0.0
    return -0.5 * thickness, 0.5 * thickness


@dataclass(frozen=True)
class MaskedPick:
    """The level-0 reads behind one MaskedImage pick, and their answer.

    A pick is answered in two passes (design 9.3): the mask at the hit and,
    in 3D, its 7-voxel stencil, which gives the normal; then the value at
    the hit and along that normal.  Every read is the nearest voxel at level
    0.  Positions are **centred** level-0 voxels (voxel ``i`` at ``i``) and
    full rank; the non-spatial components stay at the hit's.

    Use :meth:`for_visual`, then :meth:`mask_reads`, :meth:`value_reads`
    and :meth:`info` in that order.

    Parameters
    ----------
    hit : np.ndarray
        The hit in the mask's centred level-0 voxels, full rank.
    mask_shape, value_shape : tuple[int, ...]
        The two stores' level-0 shapes.
    mask_space : tuple[int, ...]
        The mask's spatial data axes.
    mask_to_world : np.ndarray
        The linear part of the mask's data-to-world transform, ``(world
        ndim, data ndim)``.
    mask_to_value : np.ndarray
        Mask level-0 voxels to value level-0 voxels, augmented, full rank.
    three_d : bool
        A 3D canvas: the value is projected along the normal.  In 2D the
        projected value is the value at the hit (MI-D8).
    iso_threshold, surface_thickness : float
        The single appearance's.
    surface_placement, surface_projection : str
        The single appearance's.
    steps_per_voxel : float
        Samples a value voxel along the normal
        (``appearance.projection_steps_per_voxel``).
    planes : tuple
        The enabled clipping planes' :class:`Plane`, in the mask's level-0
        data coordinates.
    """

    hit: np.ndarray
    mask_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    mask_space: tuple[int, ...]
    mask_to_world: np.ndarray
    mask_to_value: np.ndarray
    three_d: bool
    iso_threshold: float
    surface_thickness: float
    surface_placement: str
    surface_projection: str
    steps_per_voxel: float
    planes: tuple = ()

    @classmethod
    def for_visual(
        cls,
        visual: MaskedImageVisual,
        data_coordinate: tuple[float, ...],
        mask_store: Any,
        value_store: Any,
        three_d: bool,
    ) -> MaskedPick:
        """The pick of *visual* at *data_coordinate* (floor convention).

        Parameters
        ----------
        visual : MaskedImageVisual
            The MaskedImage hit.
        data_coordinate : tuple[float, ...]
            The promoted hit, mask level-0 data coordinates, ``floor`` the
            voxel (``ImagePickInfo``'s convention).
        mask_store, value_store : Any
            The two stores, for their shapes and level-0 systems.
        three_d : bool
            Whether the canvas is 3D.
        """
        mask_system = mask_store.data_coordinate_systems[0]
        n = mask_system.ndim
        if visual.value_to_mask is None:
            mask_to_value = np.eye(n + 1)
        else:
            mask_to_value = np.linalg.inv(
                np.asarray(visual.value_to_mask.matrix, dtype=np.float64)
            )
        matrix = np.asarray(visual.transform.matrix, dtype=np.float64)
        single = visual.single
        return cls(
            hit=np.asarray(data_coordinate, dtype=np.float64) - 0.5,
            mask_shape=tuple(int(v) for v in mask_store.level_shapes[0]),
            value_shape=tuple(int(v) for v in value_store.level_shapes[0]),
            mask_space=tuple(_space_axes(mask_system)),
            mask_to_world=matrix[:-1, :-1],
            mask_to_value=mask_to_value,
            three_d=bool(three_d),
            iso_threshold=float(single.iso_threshold),
            surface_thickness=float(single.surface_thickness),
            surface_placement=single.surface_placement,
            surface_projection=single.surface_projection,
            steps_per_voxel=float(visual.appearance.projection_steps_per_voxel),
            planes=tuple(item.plane for item in visual.clipping_planes if item.enabled),
        )

    # -- the first pass: the mask ----------------------------------------------

    def _mask_index(self, offset: tuple[int, int] | None = None):
        index = np.floor(self.hit + 0.5).astype(np.int64)
        if np.any(index < 0) or np.any(index >= np.asarray(self.mask_shape)):
            return None
        if offset is not None:
            axis, step = offset
            index[axis] = min(max(index[axis] + step, 0), self.mask_shape[axis] - 1)
        return tuple(int(i) for i in index)

    @property
    def _projects(self) -> bool:
        return self.three_d and self.surface_thickness > 0.0

    def mask_reads(self) -> dict[Any, tuple[int, ...]]:
        """``key -> voxel`` to read from the mask: the hit, and its stencil in 3D.

        Empty when the hit lies outside the mask's data.
        """
        hit = self._mask_index()
        if hit is None:
            return {}
        reads: dict[Any, tuple[int, ...]] = {"hit": hit}
        if self._projects:
            for axis in self.mask_space:
                for step in (-1, 1):
                    reads[(axis, step)] = self._mask_index((axis, step))
        return reads

    def _inward(self, mask_values: dict) -> np.ndarray | None:
        """The inward normal as a mask-voxel step one world unit long.

        The mask's gradient by central differences over the stencil (one
        sided at the data's edge), or on a cut face the plane's normal
        (MI-P10).  A normal is a covector ``g``; the step of unit world
        length along it is ``G^-1 g / sqrt(g G^-1 g)``, ``G = L^T L`` over
        the spatial axes.  ``None`` where the gradient vanishes.
        """
        space = list(self.mask_space)
        covector = None
        if float(mask_values["hit"]) >= self.iso_threshold:
            for plane in self.planes:
                if abs(float(plane.signed_distance(self.hit))) <= _CUT_FACE_VOXELS:
                    covector = np.asarray(plane.normal, dtype=np.float64)[space]
                    break
        if covector is None:
            covector = np.zeros(len(space))
            for k, axis in enumerate(space):
                lo, hi = self._mask_index((axis, -1)), self._mask_index((axis, 1))
                span = hi[axis] - lo[axis]
                if span:
                    covector[k] = (
                        float(mask_values[(axis, 1)]) - float(mask_values[(axis, -1)])
                    ) / span
        linear = self.mask_to_world[:, space]
        metric = linear.T @ linear
        step = np.linalg.solve(metric, covector)
        length = float(covector @ step)
        if not np.isfinite(length) or length <= 0.0:
            return None
        out = np.zeros(self.hit.size)
        out[space] = step / np.sqrt(length)
        return out

    # -- the second pass: the value ---------------------------------------------

    def _to_value(self, position: np.ndarray) -> np.ndarray:
        return (self.mask_to_value @ np.append(position, 1.0))[:-1]

    def _value_index(self, value_position: np.ndarray) -> tuple[int, ...] | None:
        index = np.floor(value_position + 0.5).astype(np.int64)
        if np.any(index < 0) or np.any(index >= np.asarray(self.value_shape)):
            return None
        return tuple(int(i) for i in index)

    def value_reads(self, mask_values: dict) -> dict[Any, tuple[int, ...]]:
        """``key -> voxel`` to read from the value image.

        ``"hit"`` is the value at the hit; ``("sample", i)`` the samples
        along the inward normal, placed and counted as the shader places
        them (``masked_project``: the midpoint of each of ``n`` steps, ``n``
        the thickness in value voxels times ``steps_per_voxel``, at most
        64).  A sample outside the value's data is not read.
        """
        reads: dict[Any, tuple[int, ...]] = {}
        hit = self._value_index(self._to_value(self.hit))
        if hit is not None:
            reads["hit"] = hit
        if not self._projects or "hit" not in mask_values:
            return reads
        inward = self._inward(mask_values)
        if inward is None:
            return reads
        start, end = _surface_interval(self.surface_thickness, self.surface_placement)
        along = self.mask_to_value[:-1, :-1] @ inward
        voxels = float(np.linalg.norm(along)) * (end - start)
        n = int(min(max(np.ceil(self.steps_per_voxel * voxels), 1), _PROJECTION_CAP))
        ds = (end - start) / n
        for i in range(n):
            position = self.hit + inward * (start + (i + 0.5) * ds)
            index = self._value_index(self._to_value(position))
            if index is not None:
                reads[("sample", i)] = index
        return reads

    def info(self, mask_values: dict, value_values: dict) -> MaskedImagePickInfo:
        """The pick's answer from both passes' values."""
        from cellier.events import MaskedImagePickInfo

        def number(values: dict, key) -> float | None:
            return float(values[key]) if key in values else None

        surface = number(value_values, "hit")
        samples = [
            float(value)
            for key, value in value_values.items()
            if isinstance(key, tuple) and key[0] == "sample"
        ]
        if not self._projects:
            projected = surface
        elif not samples:
            projected = None
        elif self.surface_projection == "max":
            projected = max(samples)
        elif self.surface_projection == "min":
            projected = min(samples)
        else:
            projected = sum(samples) / len(samples)
        value_position = self._to_value(self.hit)
        return MaskedImagePickInfo(
            data_coordinate=tuple(float(v) for v in self.hit + 0.5),
            value_coordinate=tuple(float(v) for v in value_position + 0.5),
            mask_value=number(mask_values, "hit"),
            surface_value=surface,
            projected_value=projected,
        )
