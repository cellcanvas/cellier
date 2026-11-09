"""The contract a multiscale pyramid's level transforms must satisfy.

The renderer places every level through its ``level -> data`` transform (a
per-axis scale ``s_k`` and translation ``t_k`` in level-0 voxel units; see
``cellier.render._level_mapping``).  It supports coarser levels that sit over
the level-0 block they summarise, offset by less than one coarse voxel --
block averaging, plain striding and offset striding.  Cropped or shifted
levels would need per-level page tables and are rejected at load.

Per level ``k >= 1`` and axis (``N_k`` the level's size along the axis):

============  ================================================  ==================
check         rule                                              meaning
============  ================================================  ==================
``diagonal``  the transform is diagonal (scale + translation)   the models' form
``scale``     ``s_k >= 1`` and ``s_k >= s_{k-1}``               coarser as k grows
``offset``    ``-0.5 <= t_k <= s_k - 0.5``                      sub-voxel offset
``coverage``  ``|t_k + s_k (N_k - 0.5) - (N_0 - 0.5)| <= s_k``  covers level 0
============  ================================================  ==================

``offset`` accepts block average ``(s - 1) / 2``, offset striding ``s // 2``
and plain striding ``0``.  ``coverage`` allows each level's far edge to miss
level 0's by up to one coarse voxel (the last voxel of a non-divisible
level).  Comparisons use a tolerance of ``1e-6 * max(1, s_k)``.  An axis with
``s_k == 1`` and ``t_k == 0`` (time, channel) passes trivially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

_CONTRACT = (
    "every coarser level must be an axis-aligned scale >= 1 of level 0 "
    "(non-decreasing with level), offset by less than one of its own voxels "
    "(-0.5 <= t <= s - 0.5, level-0 voxel units), and cover level 0's extent "
    "to within one of its own voxels"
)


#: How :meth:`LevelContractIssue.describe` names each check's value.
_CHECK_LABELS = {
    "diagonal": "off-diagonal term",
    "scale": "scale",
    "offset": "translation",
    "coverage": "far edge",
}


@dataclass(frozen=True)
class LevelContractIssue:
    """One violated check.

    Attributes
    ----------
    level : int
        0-based level index.
    axis : int or None
        Data axis, or ``None`` for a whole-matrix check (``diagonal``).
    check : {"diagonal", "scale", "offset", "coverage"}
        The rule broken; see the module docstring.
    value : float
        The offending value.
    allowed : str
        The allowed range.
    """

    level: int
    axis: int | None
    check: str
    value: float
    allowed: str

    def describe(self) -> str:
        """One line naming the level, axis, value and allowed range."""
        where = f"level {self.level}" + (
            "" if self.axis is None else f", axis {self.axis}"
        )
        what = _CHECK_LABELS[self.check]
        return f"{where}: {what} is {self.value:.6g}, allowed {self.allowed}"


def level_contract_issues(
    level_transforms: Sequence,
    level_shapes: Sequence[Sequence[int]] | None = None,
) -> list[LevelContractIssue]:
    """Every contract violation of *level_transforms* (``level k -> level 0``).

    Parameters
    ----------
    level_transforms : sequence of AffineTransform
        One per level, level 0 first.  Anything with ``linear`` and
        ``translation`` array properties works.
    level_shapes : sequence of shapes, optional
        One per level.  ``coverage`` is skipped when not given.

    Returns
    -------
    list[LevelContractIssue]
        Empty when the pyramid satisfies the contract.
    """
    issues: list[LevelContractIssue] = []
    if len(level_transforms) < 2:
        return issues
    linears = [np.asarray(t.linear, dtype=np.float64) for t in level_transforms]
    offsets = [np.asarray(t.translation, dtype=np.float64) for t in level_transforms]
    scales = []
    for level, linear in enumerate(linears):
        diagonal = np.diag(linear) if linear.shape[0] == linear.shape[1] else None
        if diagonal is None or np.abs(linear - np.diag(diagonal)).max() > 1e-9:
            if level > 0:
                issues.append(
                    LevelContractIssue(
                        level, None, "diagonal", 1.0, "a diagonal matrix"
                    )
                )
            scales.append(None)
        else:
            scales.append(diagonal)
    for level in range(1, len(level_transforms)):
        scale, previous = scales[level], scales[level - 1]
        if scale is None:
            continue
        for axis, s in enumerate(scale):
            s = float(s)
            t = float(offsets[level][axis])
            tol = 1e-6 * max(1.0, abs(s))
            if s < 1.0 - tol:
                issues.append(LevelContractIssue(level, axis, "scale", s, ">= 1"))
            if previous is not None and s < float(previous[axis]) - tol:
                issues.append(
                    LevelContractIssue(
                        level,
                        axis,
                        "scale",
                        s,
                        f">= {float(previous[axis]):g} (the previous level's)",
                    )
                )
            if not (-0.5 - tol <= t <= s - 0.5 + tol):
                issues.append(
                    LevelContractIssue(level, axis, "offset", t, f"[-0.5, {s - 0.5:g}]")
                )
            if level_shapes is not None:
                far_k = t + s * (float(level_shapes[level][axis]) - 0.5)
                far_0 = float(level_shapes[0][axis]) - 0.5
                if abs(far_k - far_0) > s + tol:
                    issues.append(
                        LevelContractIssue(
                            level,
                            axis,
                            "coverage",
                            far_k,
                            f"{far_0:g} +- {s:g}",
                        )
                    )
    return issues


def validate_level_transforms(
    level_transforms: Sequence,
    level_shapes: Sequence[Sequence[int]] | None = None,
    *,
    name: str = "?",
) -> None:
    """Raise when a pyramid's level transforms break the contract.

    Parameters
    ----------
    level_transforms : sequence of AffineTransform
        One per level (``level k -> level 0``), level 0 first.
    level_shapes : sequence of shapes, optional
        One per level; ``coverage`` is skipped without them.
    name : str
        The store's name, for the message.

    Raises
    ------
    ValueError
        Naming every violation: level, axis, value and allowed range.
    """
    issues = level_contract_issues(level_transforms, level_shapes)
    if not issues:
        return
    lines = "\n".join(f"  - {issue.describe()}" for issue in issues)
    raise ValueError(
        f"Data store '{name}' has pyramid levels the renderer cannot place: "
        f"{_CONTRACT}.\n{lines}"
    )


def validate_store_levels(store: object) -> None:
    """Check a multi-level store's installed ``level_transforms`` against the contract.

    A no-op for stores with fewer than two transforms.  ``level_shapes`` is
    read when the store exposes it; without it ``coverage`` is skipped.

    Raises
    ------
    ValueError
        See :func:`validate_level_transforms`.
    """
    transforms = list(getattr(store, "level_transforms", None) or [])
    if len(transforms) < 2:
        return
    try:
        shapes = list(store.level_shapes)
    except (AttributeError, RuntimeError, ValueError):
        shapes = None
    if shapes is not None and len(shapes) != len(transforms):
        shapes = None
    validate_level_transforms(transforms, shapes, name=str(getattr(store, "name", "?")))
