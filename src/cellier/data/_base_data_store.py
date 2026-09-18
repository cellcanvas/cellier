"""Base class for data stores."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated, Any, ClassVar
from uuid import uuid4

import numpy as np
from psygnal import EventedModel, Signal
from pydantic import UUID4, AfterValidator, Field, PrivateAttr, model_validator

from cellier.data._axes import identity_transform, install_level_transforms
from cellier.data._changes import STORE_CHANGE_KINDS, StoreChange
from cellier.data._dataset_info import DatasetInfo, RowSection
from cellier.transform import (  # noqa: TC001
    AffineTransform,
    DataCoordinateSystem,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cellier.data._changes import DataRegion, StoreChangeKind

#: The extents of an axis that carries no data at all.
#:
#: ``axis_extents`` returns this for a geometry store holding zero
#: vertices.  It is distinct from a zero-width extent: an empty store does
#: not occupy the point ``{0}``, it occupies nothing, and a consumer taking
#: a union across visuals must skip it rather than pull the union to zero.
NO_EXTENT = None


def gridded_axis_extents(shape: Sequence[int]) -> tuple[tuple[float, float], ...]:
    """Per-axis extents of a regular grid, on the edge convention.

    A voxel is centred on its integer index and spans
    ``[i - 0.5, i + 0.5)``, so an axis of ``size`` voxels spans
    ``[-0.5, size - 0.5]`` -- half a voxel beyond the first and last
    centres.  This is the convention
    :func:`~cellier.render.visuals._slicing.round_world_to_voxel` documents
    and the whole render layer already uses; returning the centres
    ``(0, size - 1)`` instead would understate every axis by half a voxel at
    each end.

    Parameters
    ----------
    shape : Sequence[int]
        The level-0 shape, one entry per data axis.

    Returns
    -------
    tuple[tuple[float, float], ...]
        ``(low, high)`` per axis, in level-0 data coordinates.
    """
    return tuple((-0.5, float(size) - 0.5) for size in shape)


def geometry_axis_extents(
    positions: np.ndarray,
) -> tuple[tuple[float, float], ...] | None:
    """Per-axis extents of a point cloud, as its bounding box.

    Geometry has no grid to take a shape from, so its extent is simply the
    range its coordinates occupy.  Unlike the gridded case there is no
    half-voxel padding: a vertex is a point, not a cell.

    Parameters
    ----------
    positions : np.ndarray
        ``(n_vertices, ndim)`` array in level-0 data coordinates.

    Returns
    -------
    tuple[tuple[float, float], ...] or None
        ``(low, high)`` per axis, or ``NO_EXTENT`` when the store holds no
        vertices -- an empty store occupies nothing, which is not the same
        as occupying zero width at the origin.
    """
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be a 2D (n_vertices, ndim) array to have "
            f"per-axis extents; got shape {positions.shape}."
        )
    if positions.shape[0] == 0:
        return NO_EXTENT
    lows = np.min(positions, axis=0)
    highs = np.max(positions, axis=0)
    return tuple(
        (float(low), float(high)) for low, high in zip(lows, highs, strict=True)
    )


class BaseDataStore(EventedModel):
    """The base class for all DataStores.

    Parameters
    ----------
    id : UUID4
        The unique identifier for the data store.  When not given it is
        taken from the first entry of ``data_coordinate_systems`` (its
        ``datastore_id``), or generated when that is empty too.
    name : str
        The name of the data store.
    data_coordinate_systems : list[DataCoordinateSystem]
        The store's intrinsic (voxel) coordinate systems, one per resolution
        level, index ``0`` being the finest.  A single-resolution store has
        one entry.

        **Constructed by the caller.**  There is no shorthand: pass built
        ``DataCoordinateSystem`` objects.  Every system's ``datastore_id``
        must equal the store's ``id``, and a mismatch raises.  A caller
        building the system before the store exists gives it
        ``datastore_id=uuid4()`` and omits ``id``; the store adopts it.

        **Stored, not derived.**  Transforms serialize their endpoints as
        UUIDs, so a system rebuilt on load would mint fresh ids and every
        transform naming it would point at nothing.

        The default is an **empty list**: a store with no axis metadata does
        not invent one, because ``Axis.axis_type`` has no honest default.
        Concrete stores that can say -- the OME-Zarr readers -- populate it at
        construction.  A store that reaches ``CellierController.add_visual``
        still empty has one derived from the scene's world axes; see
        ``CellierController._ensure_data_coordinate_systems``.
    level_scales : list[tuple[float, ...]]
        Per-level, per-axis scale of level ``k`` voxels in level ``0`` voxels.
        ``level_scales[0]`` is all ones.  Empty for a single-level store.

        **This is where a pyramid's geometry is stated.**  The readers derive
        it from OME-NGFF metadata; it is raw numbers, which is all a store can
        know on its own.
    level_translations : list[tuple[float, ...]]
        The offset half of the same, in level-0 voxels.
        ``level_translations[0]`` is all zeros.
    level_transforms : list[AffineTransform]
        Level ``k`` voxel space -> level ``0`` voxel space, one per entry in
        ``data_coordinate_systems``.  Index ``0`` is the identity.  Empty
        alongside an empty ``data_coordinate_systems``.

        **Derived, not authored.**  Built from ``level_scales`` /
        ``level_translations`` by ``install_level_transforms`` as soon as the
        store has its coordinate systems -- at construction when they are
        passed, or from the scene's world.  It is stored rather than
        recomputed on read
        because a transform serializes its endpoints as ids, and a rebuilt
        one would name systems that no longer exist.

    Attributes
    ----------
    id : str
        The unique identifier for the data store.
    data_changed : Signal
        Emits a :class:`~cellier.data._changes.StoreChange` whenever the
        store's data changes: automatically when a data field is reassigned,
        and on :meth:`notify_changed` after an in-place write.  A controller
        that registers the store relays it to the event bus and reslices the
        visuals reading the store (``plans/store_change_events.md``).
        Changes must happen on the main thread.
    """

    data_changed: ClassVar[Signal] = Signal(object)

    #: Data fields whose reassignment can move the store's extent.
    _EXTENT_FIELDS: ClassVar[frozenset[str]] = frozenset()
    #: Data fields whose reassignment changes values but not the extent.
    _CONTENTS_FIELDS: ClassVar[frozenset[str]] = frozenset()

    # ``(axis_extents,)`` once computed, ``None`` when stale.  A 1-tuple
    # rather than a sentinel because ``NO_EXTENT`` is itself ``None`` and a
    # sentinel object would not survive a deep copy's identity check.
    _axis_extents_cache: tuple | None = PrivateAttr(default=None)

    # store a UUID to identify this specific scene.
    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    name: str = "data store"
    data_coordinate_systems: list[DataCoordinateSystem] = Field(default_factory=list)
    level_scales: list[tuple[float, ...]] = Field(default_factory=list)
    level_translations: list[tuple[float, ...]] = Field(default_factory=list)
    level_transforms: list[AffineTransform] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _adopt_datastore_id(cls, data: Any) -> Any:
        """Take the store's ``id`` from its coordinate systems when none is given.

        A ``DataCoordinateSystem`` has to name its store's id, but a caller
        builds the system before the store exists.  Adopting the system's
        ``datastore_id`` lets the system be written first without also
        threading the same UUID through ``id=``.
        """
        if not isinstance(data, dict) or data.get("id") is not None:
            return data
        systems = data.get("data_coordinate_systems")
        if not systems:
            return data
        first = systems[0]
        datastore_id = (
            first.get("datastore_id")
            if isinstance(first, dict)
            else getattr(first, "datastore_id", None)
        )
        if datastore_id is None:
            return data
        return {**data, "id": datastore_id}

    @model_validator(mode="after")
    def _check_datastore_ids(self) -> BaseDataStore:
        """Require every coordinate system to name this store."""
        for system in self.data_coordinate_systems:
            if system.datastore_id != self.id:
                raise ValueError(
                    f"Coordinate system '{system.name}' belongs to datastore "
                    f"{system.datastore_id}, but this store's id is {self.id}.  "
                    f"Build the system with datastore_id equal to the store's "
                    f"id, or omit id= and the store adopts it."
                )
        return self

    # -- change announcements -------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign a field, announcing the change when it is a data field.

        A class-level hook rather than a connection to ``self.events`` made
        in ``model_post_init``: ``model_copy(deep=True)`` -- which
        ``to_model`` and ``from_model`` use -- runs no ``model_post_init``, so
        a connection would silently go missing on every copy.  Construction
        assigns through ``__dict__`` and announces nothing.
        """
        tracked = name in self._EXTENT_FIELDS or name in self._CONTENTS_FIELDS
        old = self.__dict__.get(name) if tracked else None
        super().__setattr__(name, value)
        if tracked:
            kind = self._change_kind(name, old, getattr(self, name))
            if kind is not None:
                self.notify_changed(kind)

    def _change_kind(self, name: str, old: Any, new: Any) -> StoreChangeKind | None:
        """Which kind of change reassigning data field *name* is.

        The default reads the class tables.  A store whose answer depends on
        the values -- an image whose ``data`` may or may not change shape --
        overrides this.  ``None`` announces nothing.
        """
        if name in self._EXTENT_FIELDS:
            return "extent"
        if name in self._CONTENTS_FIELDS:
            return "contents"
        return None

    def notify_changed(
        self,
        kind: StoreChangeKind = "contents",
        regions: Sequence[DataRegion] | None = None,
    ) -> None:
        """Announce that the store's data changed.

        Reassigning a data field announces itself; call this after an
        **in-place** write, which nothing can see -- ``store.data[...] = v``
        or ``store.positions[:] = p``.

        Parameters
        ----------
        kind : "extent" or "contents"
            ``"extent"`` when the data may now occupy a different region of
            data space (vertices moved, the shape changed, the store grew);
            ``"contents"`` when only values changed within the same region.
            When unsure, pass ``"extent"``.
        regions : Sequence[DataRegion] or None
            Where the data changed, as per-axis ``(start, stop)`` in level-0
            data coordinates.  ``None`` (the default) means anywhere.

        Raises
        ------
        ValueError
            If *kind* is not one of the two kinds.
        """
        if kind not in STORE_CHANGE_KINDS:
            raise ValueError(
                f"Unknown store change kind {kind!r}; expected one of "
                f"{list(STORE_CHANGE_KINDS)}."
            )
        normalized = (
            None
            if regions is None
            else tuple(
                tuple((float(start), float(stop)) for start, stop in region)
                for region in regions
            )
        )
        self._invalidate_caches(kind)
        self.data_changed.emit(StoreChange(kind, normalized))

    def _invalidate_caches(self, kind: StoreChangeKind) -> None:
        """Drop state derived from the data before the change is announced.

        Runs before ``data_changed`` fires, so a listener that reads the store
        in response sees fresh values.  Subclasses with caches of their own
        extend this.
        """
        if kind == "extent":
            self._axis_extents_cache = None

    def _cached_axis_extents(
        self, compute: Callable[[], tuple[tuple[float, float], ...] | None]
    ) -> tuple[tuple[float, float], ...] | None:
        """Return the cached extent, computing it with *compute* when stale.

        For stores whose extent costs a pass over the data (the geometry
        stores' min/max).  Cleared by an ``"extent"`` change.
        """
        if self._axis_extents_cache is None:
            self._axis_extents_cache = (compute(),)
        return self._axis_extents_cache[0]

    def model_post_init(self, __context: Any) -> None:
        """Check the systems passed at construction and install their transforms.

        Stores that open array handles in their own ``model_post_init`` call
        this after opening them: the level count and rank it checks against
        are read off the handles.
        """
        super().model_post_init(__context)
        if self.data_coordinate_systems:
            self._check_coordinate_system_shapes(self.data_coordinate_systems)
        install_level_transforms(self)

    def _check_coordinate_system_shapes(
        self, systems: Sequence[DataCoordinateSystem]
    ) -> None:
        """Require one system per resolution level and one axis per dimension.

        Reads ``n_levels`` and ``ndim`` where the concrete store defines them;
        a check whose quantity the store does not define is skipped.

        Raises
        ------
        ValueError
            If the number of systems differs from ``n_levels``, or a system's
            rank differs from ``ndim``.
        """
        n_levels = getattr(self, "n_levels", None)
        if n_levels is not None and len(systems) != n_levels:
            raise ValueError(
                f"Data store '{self.name}' has {n_levels} resolution level(s) "
                f"but {len(systems)} coordinate system(s); pass one per level, "
                f"finest first."
            )
        ndim = getattr(self, "ndim", None)
        if ndim is None:
            return
        for system in systems:
            if system.ndim != ndim:
                raise ValueError(
                    f"Coordinate system '{system.name}' has {system.ndim} axes "
                    f"{system.axis_names()}, but data store '{self.name}' has "
                    f"{ndim} dimensions; build one axis per dimension."
                )

    @property
    def data_coordinate_system(self) -> DataCoordinateSystem:
        """The level-0 (intrinsic) coordinate system.

        Returns
        -------
        DataCoordinateSystem
            The finest-resolution system.

        Raises
        ------
        ValueError
            If the store has no coordinate systems.  The message names the
            two ways to supply them, because this is the error a caller who
            built a bare in-memory store outside a viewer will hit.
        """
        if not self.data_coordinate_systems:
            raise ValueError(
                f"Data store '{self.name}' has no data_coordinate_systems.  "
                f"Pass data_coordinate_systems=[DataCoordinateSystem(...)] "
                f"when constructing it, or add it to a scene, which derives "
                f"them from the scene's world axes."
            )
        return self.data_coordinate_systems[0]

    def set_data_coordinate_systems(
        self,
        systems: list[DataCoordinateSystem],
        level_transforms: list[AffineTransform] | None = None,
    ) -> None:
        """Install the store's coordinate systems and level transforms.

        Assignment rather than construction because the systems are sometimes
        derived from the scene the store is added to, which the store cannot
        see at construction time.

        Parameters
        ----------
        systems : list[DataCoordinateSystem]
            One per resolution level, finest first.
        level_transforms : list[AffineTransform] or None
            Level ``k`` -> level ``0``.  ``None`` is allowed only for a
            single-level store, where the transform is the identity.

        Raises
        ------
        ValueError
            If *systems* is empty, if *level_transforms* has a different
            length, or if a multi-level store is given no level transforms --
            there is no downsampling factor to infer them from here.  Also if
            the systems are not one per resolution level with one axis per
            data dimension.
        """
        if not systems:
            raise ValueError(
                "set_data_coordinate_systems needs at least the level-0 system."
            )
        self._check_coordinate_system_shapes(systems)
        if level_transforms is None:
            if len(systems) > 1:
                raise ValueError(
                    f"A {len(systems)}-level store needs explicit "
                    f"level_transforms; only a single-level store's transform "
                    f"is the identity."
                )
            level_transforms = [identity_transform(systems[0], systems[0])]
        if len(level_transforms) != len(systems):
            raise ValueError(
                f"level_transforms must have one entry per coordinate system; "
                f"got {len(level_transforms)} for {len(systems)} systems."
            )
        self.data_coordinate_systems = list(systems)
        self.level_transforms = list(level_transforms)

    # -- extents ---------------------------------------------------------

    @property
    def axis_extents(self) -> tuple[tuple[float, float], ...] | None:
        """Per-axis ``(low, high)`` extents in **level-0** data coordinates.

        The one number the world-extent of a visual is built from.  Every
        consumer -- the axis ranges a slider is sized from, and the
        out-of-domain check that decides a visual has no data at a given
        slice position -- reads it through the visual's own transform:

        .. code-block:: text

            world extent on an axis = transform.map(store.axis_extents)

        Stating it here, once, is what lets that rule have no special case:
        a gridded store and a point cloud answer the same question by
        different rules, and a transform maps either without knowing which
        it was given.

        **Level 0 always.**  Other resolution levels derive from it through
        ``level_transforms``, and a per-level extent would be several ways
        to say one thing.

        There are exactly two implementations.  Gridded stores use
        :func:`gridded_axis_extents` -- the **edge** convention,
        ``(-0.5, size - 0.5)``, because a voxel is centred on its index.
        Geometry stores use :func:`geometry_axis_extents` -- the bounding
        box of their vertices, with no padding, because a vertex is a point.

        **Cached by geometry stores.**  Their extent is a pass over every
        vertex, so they compute it once and keep it until an ``"extent"``
        change (``notify_changed``, or reassigning ``positions``).  An
        in-place write to ``positions`` without ``notify_changed("extent")``
        leaves the cached extent stale, by contract.

        Returns
        -------
        tuple[tuple[float, float], ...] or None
            One ``(low, high)`` pair per data axis, or ``NO_EXTENT`` when
            the store holds no data at all.

        Raises
        ------
        NotImplementedError
            If a concrete store has not implemented it.  There is no
            default: a store that cannot say what it spans must say so,
            rather than inventing an extent that consumers would size a
            slider from.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement axis_extents.  A "
            f"gridded store returns gridded_axis_extents(self.level_shapes[0]); "
            f"a geometry store returns geometry_axis_extents(self.positions)."
        )

    # ── Self-description ────────────────────────────────────────────────

    DATASET_INFO_LABEL: ClassVar[str] = "data store"
    """What this kind of store calls itself in the ``Store type`` row.

    A human-readable name rather than the ``store_type`` discriminator: the
    row is read by a person, and ``"in-memory points"`` says more than
    ``"points_memory"``.
    """

    def dataset_info(self) -> DatasetInfo:
        """Describe what this store holds, for display in a dataset-info widget.

        Returns the store's identity only.  Each concrete store overrides
        this to append what is specific to it -- shapes and scale levels for
        an image, node and edge counts for a graph -- so a new store type
        gains a populated widget by implementing one method, with nothing to
        register in the GUI layer.

        Every implementation is **cheap**: it reads metadata the store
        already holds and never triggers a read of the underlying array.
        Statistics that would require one (an image's value range, a label
        image's unique labels) are reported only where the data is already
        resident in memory.

        Returns
        -------
        DatasetInfo
            The sections to draw, in display order.
        """
        return DatasetInfo(sections=[RowSection(None, self._identity_rows())])

    def _identity_rows(self) -> list[tuple[str, str]]:
        """The ``Name``/``Store type`` rows every store's block opens with."""
        return [
            ("Name", self.name),
            ("Store type", self.DATASET_INFO_LABEL),
        ]
