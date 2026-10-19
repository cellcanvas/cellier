"""``set_node_matrix``: shear survives on a pygfx node (design v2, D2 and K4).

pygfx's default local transform decomposes an assigned matrix into position,
rotation and scale and silently drops any shear.  The helper swaps the node
to a matrix-basis transform in place, with no change to pygfx.  It reaches
one pygfx internal (``node.world.own``); the canary tests here fail loudly if
a pygfx upgrade moves it -- see ``cellier/render/_node_transform.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pygfx as gfx

import cellier.render.visuals
from cellier.render._node_transform import ensure_matrix_basis, set_node_matrix

SHEAR = np.array(
    [
        [1.0, 0.5, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_canary_the_pygfx_internal_the_helper_reaches_is_there():
    """If this fails, pygfx moved ``world.own``: revisit ``_node_transform``."""
    node = gfx.Group()
    assert node.world.own is node.local
    assert node.local.state_basis == "components"
    assert hasattr(node.local, "_set_wrapper")


def test_canary_pygfx_alone_drops_the_shear():
    """What the helper is for: if this starts failing, pygfx keeps shear itself.

    Then ``set_node_matrix`` can become ``node.local.matrix = m``.
    """
    node = gfx.Group()
    node.local.matrix = SHEAR
    assert not np.allclose(node.local.matrix, SHEAR)


def test_the_helper_keeps_the_shear():
    node = gfx.Group()
    set_node_matrix(node, SHEAR)
    np.testing.assert_allclose(node.local.matrix, SHEAR)
    np.testing.assert_allclose(node.world.matrix, SHEAR)
    assert node.world.own is node.local


def test_the_swap_keeps_parent_and_children_linked():
    parent, node, child = gfx.Group(), gfx.Group(), gfx.Group()
    parent.add(node)
    node.add(child)
    parent.local.position = (10.0, 0.0, 0.0)
    child.local.position = (0.0, 0.0, 7.0)
    set_node_matrix(node, SHEAR)

    to_parent = np.eye(4)
    to_parent[0, 3] = 10.0
    to_child = np.eye(4)
    to_child[2, 3] = 7.0
    assert node.parent is parent and child.parent is node
    assert node.world.parent is parent.world and child.world.parent is node.world
    np.testing.assert_allclose(node.world.matrix, to_parent @ SHEAR)
    np.testing.assert_allclose(child.world.matrix, to_parent @ SHEAR @ to_child)


def test_a_later_matrix_reaches_the_world_transform():
    node = gfx.Group()
    set_node_matrix(node, SHEAR)
    before = node.world.last_modified
    moved = SHEAR.copy()
    moved[0, 3] = 5.0
    set_node_matrix(node, moved)
    assert node.world.last_modified > before
    np.testing.assert_allclose(node.world.matrix, moved)


def test_the_swap_is_idempotent():
    node = gfx.Group()
    assert ensure_matrix_basis(node)
    assert not ensure_matrix_basis(node)
    assert node.local.state_basis == "matrix"


def test_the_swap_carries_the_current_matrix_over():
    node = gfx.Group()
    node.local.position = (1.0, 2.0, 3.0)
    ensure_matrix_basis(node)
    np.testing.assert_allclose(node.local.matrix[:3, 3], [1.0, 2.0, 3.0])


def test_the_world_bounding_box_follows_the_shear():
    """pygfx's ``show_object`` fits a camera to this box."""
    mesh = gfx.Mesh(gfx.box_geometry(2.0, 2.0, 2.0), gfx.MeshBasicMaterial())
    set_node_matrix(mesh, SHEAR)
    box = mesh.get_world_bounding_box()
    # x spans 1 +/- (1 + 0.5) for y in [-1, 1].
    np.testing.assert_allclose(box[:, 0], [-0.5, 2.5])


def test_set_node_matrix_is_the_only_writer_of_a_data_node():
    """No data node in ``render/visuals`` is placed any other way.

    A component setter on a matrix-basis node recomposes from the decomposed
    parts and drops the shear again (design v2 section 11).  Inner child
    nodes (the 2D tile images) are separate nodes and may use them.
    """
    pattern = re.compile(
        r"\b(node|node_2d|node_3d)\.local\.(position|rotation|scale|matrix)\s*="
    )
    raw_matrix = re.compile(r"\.local\.matrix\s*=")
    root = Path(cellier.render.visuals.__file__).parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line) or raw_matrix.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert offenders == []
