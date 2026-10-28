"""Place a pygfx node with a matrix that may carry shear (design v2, D2).

pygfx's ``WorldObject.local`` defaults to ``state_basis="components"``:
assigning ``local.matrix`` decomposes the matrix into position, rotation and
scale and recomposes it, so a shear is silently replaced by the nearest
rotation-and-scale (pygfx issue #920).  A ``state_basis="matrix"`` local
transform stores the matrix as given.

pygfx offers no way to choose the basis when a node is built, so
:func:`ensure_matrix_basis` swaps it **in place**: a node's ``world`` is a
``RecursiveTransform`` holding the local transform in its ``own`` slot and
linked to its parent's and children's ``world`` objects, so replacing
``own`` keeps every link, the node's identity and the uniform-buffer update
(``WorldObject._update_object`` polls ``world.last_modified``).  It works on
a node at any time, attached or not, and needs no change to pygfx.  This is
the one place cellier reaches into that pygfx internal;
``tests/render/test_node_transform.py`` is its canary.  If pygfx ever keeps
shear itself, :func:`set_node_matrix` becomes ``node.local.matrix = m``.

**The one rule:** :func:`set_node_matrix` is the only writer of a data
node's transform.  On a matrix-basis node the component setters
(``local.position``, ``rotation``, ``scale``) still recompose from the
decomposed parts and would drop the shear again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygfx as gfx
from pygfx.utils.transform import AffineTransform as _PygfxAffine

if TYPE_CHECKING:
    import numpy as np

__all__ = ["ensure_matrix_basis", "set_node_matrix"]


def ensure_matrix_basis(node: gfx.WorldObject) -> bool:
    """Give *node* a matrix-basis local transform, in place.

    The current local matrix is carried over.  Idempotent.

    Parameters
    ----------
    node : gfx.WorldObject
        The node.

    Returns
    -------
    bool
        Whether the basis changed.
    """
    old = node.local
    if old.state_basis == "matrix":
        return False
    new = _PygfxAffine(
        state_basis="matrix",
        matrix=old.matrix.copy(),
        is_camera_space=old.is_camera_space,
    )
    world = node.world
    old._set_wrapper(None)
    world.own = new
    new._set_wrapper(world)
    node.local = new
    new.flag_update()
    return True


def set_node_matrix(node: gfx.WorldObject, matrix: np.ndarray) -> None:
    """Set *node*'s local matrix exactly, shear included.

    Parameters
    ----------
    node : gfx.WorldObject
        The node to place.
    matrix : np.ndarray
        A ``(4, 4)`` matrix, as :func:`cellier.render._spaces.node_matrix`
        returns.
    """
    if isinstance(node, gfx.WorldObject):
        # Anything else -- a test's stand-in node, say -- has no basis to
        # choose and takes the matrix as given.
        ensure_matrix_basis(node)
    node.local.matrix = matrix
