"""CI-sized multiscale stores for the plane rendering tests.

``plans/plane_rendering_design_v3.md`` 14.0.2.  The measured store
(``ExpA_VIP_ASLM_on.zarr``, 2 GB) is not in CI, so these small pyramids
reproduce what matters about it: a level pyramid that does not downsample
every axis (``aniso_pyramid``), an isotropic one, one with non-integer level
ratios, one with a time axis, an integer label one, and a channel stack.

Every voxel of every level holds a value of its own where a pixel test
compares voxels, so a pixel can name the voxel (and level) it was drawn from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import tensorstore as ts

from cellier.data import MultiscaleZarrDataStore
from cellier.transform import Axis, DataCoordinateSystem

if TYPE_CHECKING:
    from pathlib import Path

AXES_ZYX = ("z", "y", "x")


@dataclass(frozen=True)
class PyramidSpec:
    """How to build a pyramid: level-0 shape, per-level factors, voxel size."""

    shape0: tuple[int, ...]
    #: ``factors[k][axis]``: level k's voxel size in level-0 voxels.
    factors: tuple[tuple[float, ...], ...]
    #: Physical voxel size of level 0 per axis (the visual's transform).
    voxel_size: tuple[float, ...]
    dtype: str = "uint16"
    axes: tuple[str, ...] = AXES_ZYX
    #: ``translations[k][axis]``: where level k's voxel 0 is centred, in
    #: level-0 voxels.  ``None`` is block averaging, ``(factor - 1) / 2``.
    translations: tuple[tuple[float, ...], ...] | None = None

    @property
    def n_levels(self) -> int:
        return len(self.factors)

    def level_translation(self, level: int) -> tuple[float, ...]:
        if self.translations is not None:
            return tuple(self.translations[level])
        return tuple((f - 1.0) / 2.0 for f in self.factors[level])

    def level_shape(self, level: int) -> tuple[int, ...]:
        return tuple(
            max(1, math.ceil(n / f))
            for n, f in zip(self.shape0, self.factors[level], strict=True)
        )


def halving(n_levels: int, down: tuple[int, ...]) -> tuple[tuple[float, ...], ...]:
    """Factors halving the axes in *down* (1 = halve) each level."""
    return tuple(
        tuple(float(2**k) if d else 1.0 for d in down) for k in range(n_levels)
    )


#: z is never downsampled, as in the measured store.
ANISO = PyramidSpec(
    shape0=(64, 256, 256),
    factors=halving(5, (0, 1, 1)),
    voxel_size=(5.0, 6.55, 6.55),
)
ISO = PyramidSpec(
    shape0=(128, 128, 128),
    factors=halving(3, (1, 1, 1)),
    voxel_size=(1.0, 1.0, 1.0),
)
#: Ratios near 2 that are not exactly 2 (round up each time).
NONPOW2 = PyramidSpec(
    shape0=(97, 151, 131),
    factors=(
        (1.0, 1.0, 1.0),
        (97 / 49, 151 / 76, 131 / 66),
        (97 / 25, 151 / 38, 131 / 33),
        (97 / 13, 151 / 19, 131 / 17),
    ),
    voxel_size=(1.0, 1.0, 1.0),
)
TZYX = PyramidSpec(
    shape0=(6, 32, 64, 64),
    factors=tuple((1.0, 2.0**k, 2.0**k, 2.0**k) for k in range(3)),
    voxel_size=(1.0, 2.0, 1.0, 1.0),
    axes=("t", "z", "y", "x"),
)


def _fill_unique(shape: tuple[int, ...], level: int, dtype: str) -> np.ndarray:
    """A value per voxel that names the voxel (wrapping at the dtype's range)."""
    index = np.indices(shape, dtype=np.int64)
    weights = (1, 31, 977, 13007)[: len(shape)][::-1]
    total = sum(w * i for w, i in zip(weights, index, strict=True))
    total = total + 101 * level
    info = np.iinfo(np.dtype(dtype))
    return (total % (info.max - 1) + 1).astype(dtype)


def write_pyramid(root: Path, spec: PyramidSpec, *, labels: bool = False) -> None:
    """Write the pyramid as zarr v3 arrays ``s0`` .. ``sN`` under *root*."""
    dtype = "int32" if labels else spec.dtype
    for level in range(spec.n_levels):
        shape = spec.level_shape(level)
        zspec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / f"s{level}")},
            "metadata": {
                "shape": list(shape),
                "data_type": dtype,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [min(s, 32) for s in shape]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        array = ts.open(zspec).result()
        if labels:
            data = (_fill_unique(shape, level, "uint16").astype(np.int32) % 40) + 1
            data[..., ::7] = 0  # a background stripe
        else:
            data = _fill_unique(shape, level, dtype)
        array[...].write(data).result()


def open_pyramid(root: Path, spec: PyramidSpec, *, name: str = "pyramid"):
    """Open the pyramid at *root* as a store; returns ``(store, voxel_size)``.

    Level-k voxel centres sit at ``(factor - 1) / 2`` level-0 voxels, as the
    measured store's loader does, unless the spec gives ``translations``.
    """
    system = DataCoordinateSystem(
        name="data",
        datastore_id=__import__("uuid").uuid4(),
        axes=tuple(
            Axis(name=n, axis_type="time" if n == "t" else "space") for n in spec.axes
        ),
    )
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=[f"s{k}" for k in range(spec.n_levels)],
        level_scales=[tuple(f) for f in spec.factors],
        level_translations=[spec.level_translation(k) for k in range(spec.n_levels)],
        data_coordinate_system=system,
        name=name,
    )
    return store, spec.voxel_size


def channel_spec(n_channels: int, base: PyramidSpec = ANISO) -> PyramidSpec:
    """*base* with a leading channel axis that no level downsamples."""
    return PyramidSpec(
        shape0=(n_channels, *base.shape0),
        factors=tuple((1.0, *f) for f in base.factors),
        voxel_size=(1.0, *base.voxel_size),
        dtype=base.dtype,
        axes=("c", *base.axes),
    )
