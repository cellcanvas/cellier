"""A mesh in a 2D view: its cross-section, through the controller.

``plans/mesh_refactor_v3.md`` Phase 5.  The slice plane cuts the mesh; the
2D child draws the outline and the fill.  Frames are real frames of an
offscreen canvas.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.mesh._mesh_requests import MeshData, MeshSectionData
from cellier.data.mesh._mesh_slicing import slice_mesh
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events import MeshPickInfo, MeshSectionChangedEvent, MeshSectionUpdateEvent
from cellier.scene.dims import spatial_axes
from cellier.visuals import MeshFlatAppearance, MeshSectionConfig
from tests._meshes import uv_sphere
from tests._v2 import bound
from tests.render.test_mesh_loading import Reads, Rig

RED = (1.0, 0.0, 0.0, 1.0)
CENTRE = (16.0, 16.0, 16.0)
RADIUS = 10.0


class Rig2D(Rig):
    """A ``zyx`` scene with one offscreen canvas, 2D by default."""

    def __init__(self, dim: str = "2d", axes=None, render_modes=None) -> None:
        self.controller = CellierController(gui="offscreen")
        self.controller.camera_reslice_enabled = False
        kwargs = {} if render_modes is None else {"render_modes": render_modes}
        self.scene = self.controller.add_scene(
            dim=dim,
            coordinate_system=axes or spatial_axes("z", "y", "x"),
            name="scene",
            **kwargs,
        )
        self.controller.add_canvas(self.scene.id, canvas_size=(160, 160))
        self.canvas_id = self.controller.get_canvas_ids(self.scene.id)[0]
        self.view = self.controller.get_canvas_view(self.canvas_id)
        self.scheduler = self.controller._render_manager.scheduler

    def add(self, store, section=None, **appearance):
        appearance.setdefault("color", RED)
        return self.controller.add_mesh(
            data=store,
            scene_id=self.scene.id,
            appearance=MeshFlatAppearance(**appearance),
            name=store.name,
            section=section,
        )

    def nodes(self, visual):
        return self.gfx(visual)._levels[0]

    def set_z(self, z: float) -> None:
        self.controller.update_slice_indices(self.scene.id, {0: float(z)})

    def red(self) -> np.ndarray:
        """A frame's mask of red pixels."""
        frame = np.asarray(self.view._canvas.draw())
        return (frame[..., 0] > 150) & (frame[..., 1] < 90) & (frame[..., 2] < 90)


def _sphere(name="sphere") -> MeshMemoryStore:
    positions, indices = uv_sphere(RADIUS, CENTRE, n_lat=24, n_lon=48)
    return MeshMemoryStore(positions=positions, indices=indices, name=name)


@pytest.fixture
def reads(monkeypatch) -> Reads:
    return Reads(monkeypatch)


@pytest.fixture
def rig():
    rig = Rig2D()
    yield rig
    rig.controller.close()


async def _load(rig, z=16.0):
    rig.set_z(z)
    rig.controller.reslice_all()
    await rig.settle()
    rig.controller.fit_camera(rig.scene.id)
    await rig.settle()


# -- what is read -------------------------------------------------------------


async def test_a_2d_view_reads_a_section(rig, reads):
    mesh = rig.add(_sphere())
    await _load(rig)
    _name, request = reads.started[-1]
    assert request.section is not None
    assert request.section.mode == "cut"
    np.testing.assert_allclose(request.section.normal, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(request.section.offsets, (16.0,))
    # The cut replaces the z constraint: nothing is left to filter.
    assert request.region.half_spaces == ()
    nodes = rig.nodes(mesh)
    assert nodes.holds == "2d"
    assert nodes.fill.visible and nodes.outline.visible
    assert (nodes.fill_face_ids == -1).all()  # a cap: no face lies in the plane


async def test_a_3d_view_reads_whole_faces(reads):
    rig = Rig2D(dim="3d")
    try:
        mesh = rig.add(_sphere())
        rig.controller.reslice_all()
        await rig.settle()
        assert reads.started[-1][1].section is None
        assert rig.nodes(mesh).holds == "3d"
    finally:
        rig.controller.close()


async def test_a_plane_missing_the_mesh_draws_nothing(rig, reads):
    mesh = rig.add(_sphere())
    await _load(rig, z=30.0)
    assert rig.nodes(mesh).is_empty
    assert rig.drawn(mesh) is None
    assert not rig.red().any()


async def test_an_anisotropic_transform_cuts_in_data_space(rig, reads):
    """World z is twice data z: world z = 34 is the plane data z = 17."""
    store = _sphere()
    mesh = rig.add(store)
    scaled = bound(rig.controller, rig.scene.id, store, (2.0, 1.0, 1.0))
    rig.controller.set_visual_transform(mesh.id, scaled)
    await _load(rig, z=34.0)
    request = reads.started[-1][1]
    normal = np.array(request.section.normal)
    assert request.section.offsets[0] / normal[0] == pytest.approx(17.0)
    # One data unit off the equator: radius sqrt(R^2 - 1).
    positions = rig.nodes(mesh).outline.geometry.positions.data
    radius = np.linalg.norm(positions[:, :2] - [16.0, 16.0], axis=1)
    # (A polygon inscribed in the circle: a little inside it.)
    assert radius.max() == pytest.approx(np.sqrt(RADIUS**2 - 1.0), rel=5e-3)


async def test_a_time_axis_filters_and_z_cuts(reads):
    axes = [("t", "time"), *spatial_axes("z", "y", "x")]
    rig = Rig2D(axes=axes)
    try:
        positions, indices, offset = [], [], 0
        for t in range(3):
            p, i = uv_sphere(4.0 + 2 * t, CENTRE, n_lat=12, n_lon=24)
            positions.append(
                np.concatenate([np.full((len(p), 1), t, np.float32), p], axis=1)
            )
            indices.append(i + offset)
            offset += len(p)
        store = MeshMemoryStore(
            positions=np.concatenate(positions),
            indices=np.concatenate(indices).astype(np.int32),
            name="series",
        )
        mesh = rig.add(store)
        for t in range(3):
            rig.controller.update_slice_indices(rig.scene.id, {0: float(t), 1: 16.0})
            await rig.settle()
            request = reads.started[-1][1]
            # t is a filter (two half-spaces); z is the cut.
            assert len(request.region.half_spaces) == 2
            np.testing.assert_allclose(request.section.normal, (0.0, 1.0, 0.0, 0.0))
            outline = rig.nodes(mesh).outline.geometry.positions.data
            radius = np.linalg.norm(outline[:, :2] - [16.0, 16.0], axis=1)
            assert radius.max() == pytest.approx(4.0 + 2 * t, rel=1e-3)
    finally:
        rig.controller.close()


async def test_a_mesh_without_the_sliced_axis_is_drawn_whole(rig, reads):
    """A ``yx`` mesh in a ``zyx`` world has nothing to cut."""
    flat = MeshMemoryStore(
        positions=np.array([[4, 4], [4, 28], [28, 28], [28, 4]], dtype=np.float32),
        indices=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        name="flat",
    )
    mesh = rig.add(flat)
    await _load(rig)
    assert reads.started[-1][1].section is None
    nodes = rig.nodes(mesh)
    assert nodes.fill.visible and not nodes.outline.visible
    assert rig.red().sum() > 500
    # At every z: it has no extent to miss.
    rig.set_z(3.0)
    await rig.settle()
    assert reads.count("flat") == 1


async def test_a_mesh_at_a_fixed_z_shows_only_on_its_plane(rig, reads):
    flat = MeshMemoryStore(
        positions=np.array(
            [[9, 4, 4], [9, 4, 28], [9, 28, 28], [9, 28, 4]], dtype=np.float32
        ),
        indices=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        name="sheet",
    )
    mesh = rig.add(flat)
    await _load(rig, z=9.0)
    nodes = rig.nodes(mesh)
    # The faces themselves, and their border as the outline.
    assert sorted(nodes.fill_face_ids.tolist()) == [0, 1]
    assert len(nodes.outline_face_ids) == 4
    assert rig.red().sum() > 500
    rig.set_z(9.5)
    await rig.settle()
    assert not rig.red().any()


# -- the request key (L1) -----------------------------------------------------


async def test_toggling_a_part_reads_again_and_hides_until_loaded(rig, reads):
    mesh = rig.add(_sphere())
    await _load(rig)
    before = reads.count("sphere")
    reads.hold = True
    rig.controller.update_section_field(mesh.id, "fill", False)
    # A new key: not drawn until the new section is there.
    assert rig.drawn(mesh) is None
    reads.hold = False
    reads.release_all()
    await rig.settle()
    assert reads.count("sphere") == before + 1
    nodes = rig.nodes(mesh)
    assert nodes.outline.visible and not nodes.fill.visible


async def test_a_thickness_change_in_cut_mode_reads_nothing(rig, reads):
    mesh = rig.add(_sphere())
    await _load(rig)
    before = reads.count("sphere")
    for half in (1.0, 2.5, 0.0):
        rig.controller.update_thickness(rig.scene.id, {0: half})
        assert rig.drawn(mesh) is not None  # never hidden
        await rig.settle()
    assert reads.count("sphere") == before


async def test_a_thickness_change_in_slab_mode_reads_again(rig, reads):
    mesh = rig.add(_sphere(), section=MeshSectionConfig(mode="slab"))
    await _load(rig)
    before = reads.count("sphere")
    # No thickness yet: the slab is the cut.
    assert reads.started[-1][1].section.mode == "cut"
    rig.controller.update_thickness(rig.scene.id, {0: 2.0})
    await rig.settle()
    assert reads.count("sphere") == before + 1
    section = reads.started[-1][1].section
    assert section.mode == "slab"
    np.testing.assert_allclose(section.offsets, (14.0, 18.0))
    nodes = rig.nodes(mesh)
    assert (nodes.fill_face_ids >= 0).any()  # the surface inside the slab


async def test_outline_width_applies_live(rig, reads):
    mesh = rig.add(_sphere(), section=MeshSectionConfig(fill=False, outline_width=1.0))
    await _load(rig)
    before = reads.count("sphere")
    thin = int(rig.red().sum())
    rig.controller.update_section_field(mesh.id, "outline_width", 8.0)
    assert rig.gfx(mesh)._material_outline.thickness == pytest.approx(8.0)
    thick = int(rig.red().sum())
    await rig.settle()
    assert reads.count("sphere") == before
    assert thick > 3 * thin > 0


async def test_section_changes_travel_on_the_bus(rig, reads):
    mesh = rig.add(_sphere())
    seen: list[MeshSectionChangedEvent] = []
    rig.controller.on_section_changed(mesh.id, seen.append, owner_id=mesh.id)
    rig.controller._incoming_events.emit(
        MeshSectionUpdateEvent(
            source_id=mesh.id, visual_id=mesh.id, field="mode", value="slab"
        )
    )
    assert mesh.section.mode == "slab"
    assert [(e.field_name, e.new_value, e.source_id) for e in seen] == [
        ("mode", "slab", mesh.id)
    ]
    points = rig.controller.add_points(
        PointsMemoryStore(positions=np.zeros((1, 3), np.float32)),
        rig.scene.id,
        None,
        "points",
    )
    with pytest.raises(TypeError, match="only a mesh"):
        rig.controller.update_section_field(points.id, "fill", False)
    await rig.settle()


# -- pixels -------------------------------------------------------------------


def _disc_fraction(mask: np.ndarray) -> float:
    """Red pixels as a share of their own bounding box."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    box = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
    return float(mask.sum()) / box


async def test_the_equator_draws_a_disc(rig, reads):
    rig.add(_sphere())
    await _load(rig)
    mask = rig.red()
    # A filled circle covers pi / 4 of its bounding box.
    assert _disc_fraction(mask) == pytest.approx(np.pi / 4, abs=0.04)
    height, width = mask.shape
    assert mask[height // 2, width // 2]


async def test_outline_only_draws_a_ring(rig, reads):
    rig.add(_sphere(), section=MeshSectionConfig(fill=False))
    await _load(rig)
    mask = rig.red()
    assert mask.any()
    assert _disc_fraction(mask) < 0.2
    height, width = mask.shape
    assert not mask[height // 2, width // 2]


async def test_fill_only_draws_the_disc(rig, reads):
    mesh = rig.add(_sphere(), section=MeshSectionConfig(outline=False))
    await _load(rig)
    nodes = rig.nodes(mesh)
    assert nodes.fill.visible and not nodes.outline.visible
    assert _disc_fraction(rig.red()) == pytest.approx(np.pi / 4, abs=0.04)


async def test_the_section_shrinks_away_from_the_equator(rig, reads):
    rig.add(_sphere())
    await _load(rig)
    equator = int(rig.red().sum())
    rig.set_z(24.0)  # 8 of 10 from the centre: radius 6
    await rig.settle()
    smaller = int(rig.red().sum())
    assert smaller == pytest.approx(equator * 0.36, rel=0.12)


async def test_vertex_colours_reach_the_section(rig, reads):
    positions, indices = uv_sphere(RADIUS, CENTRE, n_lat=24, n_lon=48)
    green = np.tile(np.array([0.0, 1.0, 0.0, 1.0], np.float32), (len(positions), 1))
    store = MeshMemoryStore(
        positions=positions,
        indices=indices,
        colors=green,
        colors_layout="vertex",
        name="green",
    )
    rig.add(store, color_mode="vertex")
    await _load(rig)
    frame = np.asarray(rig.view._canvas.draw())
    is_green = (frame[..., 1] > 150) & (frame[..., 0] < 90)
    assert is_green.sum() > 500
    assert not rig.red().any()


# -- picks (5.10) -------------------------------------------------------------


async def test_picks_name_the_part_that_was_drawn(rig, reads):
    mesh = rig.add(_sphere())
    await _load(rig)
    gfx, nodes = rig.gfx(mesh), rig.nodes(mesh)

    fill = gfx.decode_pick(nodes.fill, {"face_index": 0})
    assert fill == MeshPickInfo(face_index=None, part="fill", level=0)

    outline = gfx.decode_pick(nodes.outline, {"vertex_index": 5})
    assert outline.part == "outline"
    # The face really is one the plane crosses.
    store = next(iter(rig.controller._model.data.stores.values()))
    crossed = store.positions[store.indices[outline.face_index], 0]
    assert crossed.min() <= 16.0 <= crossed.max()
    # The render manager reaches the visual from the leaf that was hit.
    details = rig.controller._render_manager._extract_pick_details(
        rig.scene.id, nodes.outline, {"vertex_index": 5}
    )
    assert details == outline


async def test_an_in_plane_face_picks_as_a_face(rig, reads):
    sheet = MeshMemoryStore(
        positions=np.array(
            [[9, 4, 4], [9, 4, 28], [9, 28, 28], [9, 28, 4]], dtype=np.float32
        ),
        indices=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        name="sheet",
    )
    mesh = rig.add(sheet)
    await _load(rig, z=9.0)
    gfx, nodes = rig.gfx(mesh), rig.nodes(mesh)
    picked = {gfx.decode_pick(nodes.fill, {"face_index": row}) for row in (0, 1)}
    assert picked == {
        MeshPickInfo(face_index=0, part="face", level=0),
        MeshPickInfo(face_index=1, part="face", level=0),
    }


# -- 2D and 3D ----------------------------------------------------------------


async def test_a_2d_to_3d_toggle_swaps_the_node_and_reads_whole_faces(reads):
    rig = Rig2D(render_modes={"2d", "3d"})
    try:
        mesh = rig.add(_sphere())
        await _load(rig)
        gfx = rig.gfx(mesh)
        scene_node = rig.controller._render_manager._scenes[rig.scene.id]
        assert scene_node.get_active_node(mesh.id) is gfx.node_2d
        assert isinstance(_last_data(rig, reads), MeshSectionData)

        rig.controller.set_displayed_axes(rig.scene.id, (0, 1, 2))
        await rig.settle()
        assert scene_node.get_active_node(mesh.id) is gfx.node_3d
        assert rig.nodes(mesh).holds == "3d"
        assert reads.started[-1][1].section is None
        assert isinstance(_last_data(rig, reads), MeshData)

        rig.controller.set_displayed_axes(rig.scene.id, (1, 2))
        await rig.settle()
        assert scene_node.get_active_node(mesh.id) is gfx.node_2d
        assert rig.nodes(mesh).holds == "2d"
        assert rig.nodes(mesh).outline.visible
    finally:
        rig.controller.close()


def _last_data(rig, reads):
    """Read the last request again, straight from the store."""
    _name, request = reads.started[-1]
    store = next(iter(rig.controller._model.data.stores.values()))
    return slice_mesh(store.level_arrays(), request, None)
