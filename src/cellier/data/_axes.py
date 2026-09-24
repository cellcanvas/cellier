"""Building a datastore's coordinate systems from what the store can say.

``Axis.axis_type`` has no default on purpose: a channel or time axis typed
``"space"`` is not caught where it is written, but much later, by
``from_axis_map`` refusing to map axes whose types disagree.  So a store never
guesses a type out of nothing.  There are exactly three ways it gets one, in
descending order of authority:

1. **The dataset says.**  The OME-Zarr readers take axis types straight
   from the NGFF metadata, and an empty ``type`` there raises rather than
   defaulting -- the metadata had a slot for it and left it blank, which is a
   defect in the dataset.  A geff file's axes are read the same way, except
   that geff makes ``type`` optional, so an unset one takes the name rule of
   :func:`axis_types_from_names`.
2. **The caller says.**  A ``DataCoordinateSystem`` the caller built, passed
   as ``data_coordinate_systems=``.  Its axes carry their own types.
3. **The scene says.**  A store added to a scene with no systems of its own
   takes the world's trailing axes, name, type and unit alike.  That is not a
   guess: the world was declared explicitly by the caller, and inheriting from
   it is what makes the ``data -> world`` transform typecheck by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellier.data._level_contract import validate_store_levels
from cellier.transform import (
    AffineTransform,
    Axis,
    AxisSampling,
    AxisType,
    DataCoordinateSystem,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.transform import CoordinateSystem, WorldCoordinateSystem


#: Axis names that carry a non-spatial type when only names are supplied.
_TYPE_BY_NAME: dict[str, AxisType] = {
    "t": "time",
    "time": "time",
    "c": "channel",
    "channel": "channel",
}


def axis_types_from_names(names: Sequence[str]) -> tuple[AxisType, ...]:
    """Apply the name-based type rule to a sequence of axis names.

    ``t`` / ``time`` are time, ``c`` / ``channel`` are channel, everything
    else is space.  Case-insensitive.  This is used **only** where a type
    slot is optional and left unset -- :func:`build_axes` called with
    ``types=None``, and a geff axis with no ``type``; it is a convenience for
    the conventional names, not a fallback for the general case.

    Parameters
    ----------
    names : Sequence[str]
        The axis names, in data order.

    Returns
    -------
    tuple[AxisType, ...]
        One type per name.
    """
    return tuple(_TYPE_BY_NAME.get(name.lower(), "space") for name in names)


def build_axes(
    names: Sequence[str],
    types: Sequence[AxisType] | None = None,
    units: Sequence[str | None] | None = None,
    sampling: AxisSampling | Sequence[AxisSampling] = "continuous",
) -> tuple[Axis, ...]:
    """Build axes from names, with types and units where they are known.

    Parameters
    ----------
    names : Sequence[str]
        Axis names, in data order.
    types : Sequence[AxisType] or None
        Axis types, one per name.  ``None`` applies
        :func:`axis_types_from_names`.
    units : Sequence[str | None] or None
        Physical units, one per name.  ``None`` leaves every unit unset.
    sampling : AxisSampling or Sequence[AxisSampling]
        Whether each axis's coordinates are sample indices.  A single value
        applies to every axis, which is what a voxel grid wants; a sequence
        gives one per axis, which is what a geometry store wants when its
        ``t`` column holds frame numbers while ``zyx`` hold measured
        positions.

    Returns
    -------
    tuple[Axis, ...]
        The axes, each with a fresh id.

    Raises
    ------
    ValueError
        If *types* or *units* has a different length from *names*, or if a
        type is empty.  An empty type is a dataset defect, not something to
        default: it is named here with the axis it came from.
    """
    resolved_types = axis_types_from_names(names) if types is None else tuple(types)
    if len(resolved_types) != len(names):
        raise ValueError(
            f"axis_types must have one entry per axis name; got "
            f"{len(resolved_types)} types for {len(names)} names {list(names)}."
        )
    resolved_units: tuple[str | None, ...] = (
        (None,) * len(names) if units is None else tuple(units)
    )
    if len(resolved_units) != len(names):
        raise ValueError(
            f"axis_units must have one entry per axis name; got "
            f"{len(resolved_units)} units for {len(names)} names {list(names)}."
        )
    resolved_sampling: tuple[AxisSampling, ...] = (
        (sampling,) * len(names) if isinstance(sampling, str) else tuple(sampling)
    )
    if len(resolved_sampling) != len(names):
        raise ValueError(
            f"axis_sampling must have one entry per axis name; got "
            f"{len(resolved_sampling)} entries for {len(names)} names "
            f"{list(names)}."
        )
    axes: list[Axis] = []
    for name, axis_type, unit, axis_sampling in zip(
        names, resolved_types, resolved_units, resolved_sampling
    ):
        if not axis_type:
            raise ValueError(
                f"Axis '{name}' has an empty axis_type.  OME-NGFF has a slot "
                f"for it and this dataset left it blank; there is no honest "
                f"default among {AxisType.__args__}.  Fix the metadata, or "
                f"build the DataCoordinateSystem yourself and pass it as "
                f"data_coordinate_system= to the store's from_* constructor."
            )
        axes.append(
            Axis(
                name=name,
                axis_type=axis_type,
                unit=unit or None,
                sampling=axis_sampling,
            )
        )
    return tuple(axes)


def data_coordinate_system(
    datastore_id: UUID,
    axes: Sequence[Axis],
    name: str,
) -> DataCoordinateSystem:
    """Wrap axes into a datastore's coordinate system.

    Parameters
    ----------
    datastore_id : UUID
        The owning store's id.
    axes : Sequence[Axis]
        The axes, in data order.
    name : str
        Human-readable name for the system.

    Returns
    -------
    DataCoordinateSystem
        The system.
    """
    return DataCoordinateSystem(name=name, axes=tuple(axes), datastore_id=datastore_id)


def level_coordinate_systems(
    datastore_id: UUID,
    axes: Sequence[Axis],
    n_levels: int,
    name: str,
) -> list[DataCoordinateSystem]:
    """Build one coordinate system per resolution level.

    The level-0 system is built from *axes*; the coarser levels follow from
    it by :func:`level_systems`.

    Parameters
    ----------
    datastore_id : UUID
        The owning store's id.
    axes : Sequence[Axis]
        The level-0 axes.  Levels 1 upward copy their descriptions.
    n_levels : int
        How many levels the pyramid has.
    name : str
        Base name; each level gets ``f"{name}_level{k}"``.

    Returns
    -------
    list[DataCoordinateSystem]
        One system per level, finest first.
    """
    level_zero = data_coordinate_system(datastore_id, axes, f"{name}_level0")
    return level_systems(level_zero, n_levels, name)


def level_systems(
    level_zero: DataCoordinateSystem,
    n_levels: int,
    name: str,
) -> list[DataCoordinateSystem]:
    """Extend a level-0 coordinate system to one system per resolution level.

    Every coarser level copies the level-0 axes' *names*, *types*, *units*
    and *sampling* -- they describe the same physical quantities, sampled the
    same way -- but is a distinct system with fresh axis ids, because a
    level-2 voxel is not a level-0 voxel and a transform between them is
    exactly what says so.  ``datastore_id`` carries over, so every level
    names the same store.

    Parameters
    ----------
    level_zero : DataCoordinateSystem
        The finest level's system.  Returned unchanged as the first entry.
    n_levels : int
        How many levels the pyramid has.
    name : str
        Base name; level ``k >= 1`` is named ``f"{name}_level{k}"``.

    Returns
    -------
    list[DataCoordinateSystem]
        One system per level, finest first.
    """
    systems = [level_zero]
    for level in range(1, n_levels):
        systems.append(
            data_coordinate_system(
                level_zero.datastore_id,
                tuple(
                    Axis(
                        name=axis.name,
                        axis_type=axis.axis_type,
                        unit=axis.unit,
                        sampling=axis.sampling,
                    )
                    for axis in level_zero.axes
                ),
                f"{name}_level{level}",
            )
        )
    return systems


def identity_transform(
    input_coordinate_system: CoordinateSystem,
    output_coordinate_system: CoordinateSystem,
    name: str | None = None,
) -> AffineTransform:
    """Build the positional identity between two equal-rank systems.

    Axis *i* maps to axis *i*, scale 1, translation 0.  The correspondence is
    still stated rather than assumed -- ``from_axis_map`` requires it -- and
    the two systems' axis types must agree, which is the check that catches a
    world and a dataset that disagree about what their axes are.

    Parameters
    ----------
    input_coordinate_system : CoordinateSystem
        The system to map from.
    output_coordinate_system : CoordinateSystem
        The system to map to.  Must have the same rank.
    name : str or None
        Optional name for the transform.

    Returns
    -------
    AffineTransform
        The identity between the two systems.

    Raises
    ------
    ValueError
        If the ranks differ.
    """
    if input_coordinate_system.ndim != output_coordinate_system.ndim:
        raise ValueError(
            f"An identity transform needs equal ranks; "
            f"'{input_coordinate_system.name}' has "
            f"{input_coordinate_system.ndim} axes and "
            f"'{output_coordinate_system.name}' has "
            f"{output_coordinate_system.ndim}."
        )
    return AffineTransform.from_axis_map(
        input_coordinate_system,
        output_coordinate_system,
        axis_map={
            axis.id: output_coordinate_system.axes[index].id
            for index, axis in enumerate(input_coordinate_system.axes)
        },
        name=name,
    )


def data_axes_from_world(
    world_coordinate_system: WorldCoordinateSystem,
    ndim: int,
    declared: Mapping[int, tuple[str, AxisType]] | None = None,
) -> tuple[Axis, ...]:
    """Take a store's axes from the trailing axes of the world.

    The fallback of last resort, for a store that carries no axis metadata --
    a bare ``ImageMemoryStore(data=arr)``.  Trailing rather than leading
    because that is the alignment the rest of cellier already assumes: a
    scene's initial ``displayed_axes`` is the last two or three world axes,
    and a dataset that is a sub-rank of the world is a spatial volume missing
    the leading batch-like axes, not the reverse.

    Names, types and units are inherited from the world axes, so the
    ``data -> world`` transform this store gets typechecks by construction.
    The ids are fresh: these are the axes of a different system.

    *declared* covers the axes a store has that the world does not -- a
    multichannel visual's channel axis is the shipping case, and the visual
    knows which one it is.  Those are taken from *declared* and the rest fall
    into the trailing world window around them.

    Parameters
    ----------
    world_coordinate_system : WorldCoordinateSystem
        The scene's world.
    ndim : int
        The store's rank.
    declared : Mapping[int, tuple[str, AxisType]] or None
        ``{data axis index: (name, axis_type)}`` for axes that have no world
        counterpart.

    Returns
    -------
    tuple[Axis, ...]
        ``ndim`` axes.

    Raises
    ------
    ValueError
        If, after removing the declared axes, the store still has more axes
        than the world.
    """
    declared = dict(declared or {})
    inherited = ndim - len(declared)
    if inherited > world_coordinate_system.ndim:
        raise ValueError(
            f"A data store with {ndim} axes ({len(declared)} of them declared) "
            f"cannot take the rest from a world with only "
            f"{world_coordinate_system.ndim} axes "
            f"({world_coordinate_system.axis_names()}).  Give the scene more "
            f"axes, or construct the store with its own data_coordinate_systems."
        )
    window = list(
        world_coordinate_system.axes[world_coordinate_system.ndim - inherited :]
    )
    axes: list[Axis] = []
    for index in range(ndim):
        if index in declared:
            name, axis_type = declared[index]
            axes.append(Axis(name=name, axis_type=axis_type))
        else:
            source = window.pop(0)
            axes.append(
                Axis(name=source.name, axis_type=source.axis_type, unit=source.unit)
            )
    return tuple(axes)


def install_level_transforms(store: object) -> None:
    """Give a multi-level store its ``level k -> level 0`` transforms, as v2.

    The per-level numbers are model state: the readers derive them from
    OME-NGFF metadata and keep them as ``level_scales`` and
    ``level_translations``.  What makes them a *transform* is the pair of
    coordinate systems they sit between, which do not exist until the store
    has them -- from its own axis metadata, or from the scene's world through
    ``CellierController._ensure_data_coordinate_systems``.  So this runs after
    the systems are installed, from both places.

    A no-op when the store already has one transform per system: a store
    restored from JSON keeps the ones it was serialized with, ids included.

    Until Phase 8 the readers built these as v1 matrices, which name no
    endpoints, and the region pull-back derived a named form alongside them
    (D5.1).  There is one list now.

    Parameters
    ----------
    store : object
        The datastore.  Must expose ``data_coordinate_systems``,
        ``level_scales`` and ``level_translations``.
    """
    systems = list(getattr(store, "data_coordinate_systems", []))
    if not systems:
        return
    existing = list(getattr(store, "level_transforms", []))
    if len(existing) == len(systems):
        return
    scales = list(getattr(store, "level_scales", []))
    translations = list(getattr(store, "level_translations", []))
    if not scales:
        # A single-level store's only transform is the identity on itself.
        store.level_transforms = [identity_transform(systems[0], systems[0])]
        return
    if len(scales) != len(systems):
        raise ValueError(
            f"Data store '{getattr(store, 'name', '?')}' has {len(scales)} "
            f"level scales but {len(systems)} coordinate systems; they must "
            f"match, one per resolution level."
        )
    ndim = systems[0].ndim
    transforms: list[AffineTransform] = []
    for level, system in enumerate(systems):
        scale = tuple(float(value) for value in scales[level])
        offset = (
            tuple(float(value) for value in translations[level])
            if level < len(translations)
            else (0.0,) * ndim
        )
        transforms.append(
            AffineTransform.from_axis_map(
                system,
                systems[0],
                axis_map={
                    system.axes[index].id: systems[0].axes[index].id
                    for index in range(ndim)
                },
                scale={system.axes[index].id: scale[index] for index in range(ndim)},
                translation={
                    system.axes[index].id: offset[index] for index in range(ndim)
                },
                name=f"level{level}_to_level0",
            )
        )
    store.level_transforms = transforms
    validate_store_levels(store)


def default_data_to_world(
    data_coordinate_system: DataCoordinateSystem,
    world_coordinate_system: WorldCoordinateSystem,
) -> AffineTransform:
    """Build the ``data -> world`` transform a visual gets when none is given.

    D18 forbids a coordinate-system-less identity, so there is no default on
    the model field: the controller builds this one at ``add_visual``, when
    both systems are finally known.

    The correspondence is by **trailing position**, the same alignment the
    rest of cellier already assumes -- a scene's initial ``displayed_axes`` is
    the last two or three world axes.  The two ranks are usually equal, in
    which case this is the identity on indices, and the two ways they can
    differ have one honest reading each:

    * *fewer data axes than world axes.*  The leading world axes are
      **broadcast**: a 3-D volume in a ``TZYX`` world exists at every ``T``,
      which is not the same claim as sitting at ``T = 0`` (D25).
    * *more data axes than world axes.*  The leading data axes are projected
      away -- a multichannel store's channel axis is composited by the visual
      and never reaches the world.  ``from_axis_map`` cannot express a dropped
      input axis, so the matrix is built directly.

    Parameters
    ----------
    data_coordinate_system : DataCoordinateSystem
        The store's level-0 system.
    world_coordinate_system : WorldCoordinateSystem
        The scene's world.

    Returns
    -------
    AffineTransform
        A scale-1, translation-0 transform between the two systems.
    """
    n_data = data_coordinate_system.ndim
    n_world = world_coordinate_system.ndim
    if n_data <= n_world:
        offset = n_world - n_data
        return AffineTransform.from_axis_map(
            data_coordinate_system,
            world_coordinate_system,
            axis_map={
                data_coordinate_system.axes[index].id: world_coordinate_system.axes[
                    index + offset
                ].id
                for index in range(n_data)
            },
            broadcast_output_axes=[
                world_coordinate_system.axes[index].id for index in range(offset)
            ],
            name="data_to_world",
        )
    dropped = n_data - n_world
    matrix = np.zeros((n_world + 1, n_data + 1))
    matrix[n_world, n_data] = 1.0
    for world_index in range(n_world):
        matrix[world_index, world_index + dropped] = 1.0
    return AffineTransform.from_matrix(
        matrix,
        data_coordinate_system,
        world_coordinate_system,
        name="data_to_world",
    )


def scale_and_translation_transform(
    data_coordinate_system: DataCoordinateSystem,
    world_coordinate_system: WorldCoordinateSystem,
    scale: Sequence[float],
    translation: Sequence[float] | None = None,
) -> AffineTransform:
    """Name the endpoints of a per-axis scale and offset.

    For a store that states its own geometry as raw numbers -- a geff file's
    per-axis ``scale`` and ``offset`` (D23) -- and so has said *what* the map
    is without saying *between which spaces*.  The controller knows both, so
    this is where they get attached.

    This replaced ``transform_from_v1``, which did the same job for a bare v1
    matrix and went with v1 in Phase 8.  The difference is that a caller now
    supplies numbers rather than an object that looked like a transform while
    naming nothing.

    Parameters
    ----------
    data_coordinate_system : DataCoordinateSystem
        The store's level-0 system.
    world_coordinate_system : WorldCoordinateSystem
        The scene's world.
    scale : Sequence[float]
        Per-data-axis scale.
    translation : Sequence[float] or None
        Per-data-axis offset.  ``None`` is all zeros.

    Returns
    -------
    AffineTransform
        The map, with both endpoints named.

    Raises
    ------
    ValueError
        If *scale* does not have one entry per axis of both systems.  The
        correspondence is positional, so there is nothing to align against
        when the two systems disagree about how many axes there are.
    """
    ndim = data_coordinate_system.ndim
    if len(scale) != ndim or ndim != world_coordinate_system.ndim:
        raise ValueError(
            f"A {len(scale)}-axis scale cannot be placed between "
            f"'{data_coordinate_system.name}' ({ndim} axes) and "
            f"'{world_coordinate_system.name}' "
            f"({world_coordinate_system.ndim} axes): the correspondence is "
            f"positional, so there is nothing to infer.  Build the transform "
            f"with AffineTransform.from_axis_map instead."
        )
    offsets = tuple(translation) if translation is not None else (0.0,) * ndim
    data_axes = data_coordinate_system.axes
    world_axes = world_coordinate_system.axes
    return AffineTransform.from_axis_map(
        data_coordinate_system,
        world_coordinate_system,
        axis_map={data_axes[i].id: world_axes[i].id for i in range(ndim)},
        scale={data_axes[i].id: float(scale[i]) for i in range(ndim)},
        translation={data_axes[i].id: float(offsets[i]) for i in range(ndim)},
        name="data_to_world",
    )


def store_level_transforms(store: object) -> list[AffineTransform]:
    """The level ``k`` -> level ``0`` transforms of a store.

    A thin read since Phase 8.  The store holds one v2 transform per
    coordinate system, built by :func:`install_level_transforms` from its own
    ``level_scales`` and ``level_translations`` as soon as the systems exist.
    Until then the readers held v1 matrices, the region pull-back needed a v2
    form, and this derived one alongside them (D5.1).

    Parameters
    ----------
    store : object
        A datastore.

    Returns
    -------
    list[AffineTransform]
        One transform per level, finest first, index 0 the identity.  Empty
        when the store has no coordinate systems.
    """
    if not getattr(store, "data_coordinate_systems", None):
        return []
    return list(getattr(store, "level_transforms", []))
