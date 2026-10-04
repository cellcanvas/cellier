"""Clipping planes through the controller: the model, the events, the reads.

Pixel accuracy of the cut is in ``test_clipping_planes_render.py`` and
``test_clipping_cut_face.py``.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pygfx as gfx
import pytest

from cellier.data import (
    GraphMemoryStore,
    ImageMemoryStore,
    LabelMemoryStore,
    LinesMemoryStore,
    MeshMemoryStore,
    PointsMemoryStore,
)
from cellier.data.lines._lines_requests import LinesSliceRequest
from cellier.events import ClippingPlanesChangedEvent, ClippingPlanesUpdateEvent
from cellier.render.scheduling import _scheduler
from cellier.transform import Axis, ConvexRegion, DataCoordinateSystem
from cellier.visuals import (
    ClippingPlane,
    MeshFlatAppearance,
    MeshSectionConfig,
    PointsMarkerAppearance,
)
from tests._meshes import uv_sphere
from tests.render import _clipping as h
from tests.render.conftest import drain_loading

SIZE = 128


def _system(names="zyx") -> DataCoordinateSystem:
    return DataCoordinateSystem(
        name="data",
        datastore_id=uuid4(),
        axes=tuple(Axis(name=n, axis_type="space") for n in names),
    )


def _plane(system, x=16.0, normal=(0, 0, 1), enabled=True) -> ClippingPlane:
    return ClippingPlane.from_point_normal(
        system, (16, 16, x), normal, axes=("z", "y", "x"), enabled=enabled
    )


def _points_store(n=4000, extent=32.0) -> PointsMemoryStore:
    rng = np.random.default_rng(0)
    return PointsMemoryStore(
        positions=(rng.random((n, 3)) * extent).astype(np.float32),
        data_coordinate_systems=[_system()],
    )


def _render_visual(controller, visual):
    scene_id = controller._visual_to_scene[visual.id]
    return controller._render_manager._scenes[scene_id].get_visual(visual.id)


def _live_materials(render_visual) -> list:
    """The materials of everything a render visual draws, in 2D and 3D."""
    found = {}
    for mode in ("2d", "3d"):
        node = render_visual.get_node(mode)
        if node is None:
            continue
        lines = {
            id(getattr(render_visual, name, None))
            for name in ("_aabb_line", "_aabb_line_2d", "_aabb_line_3d")
        }
        for slot in getattr(render_visual, "_slots", ()):
            lines |= {
                id(getattr(slot, n, None)) for n in ("_aabb_line_2d", "_aabb_line_3d")
            }
        for obj in node.iter(
            lambda o: isinstance(
                o, (gfx.Volume, gfx.Image, gfx.Mesh, gfx.Points, gfx.Line)
            )
        ):
            if id(obj) in lines or getattr(obj, "material", None) is None:
                continue  # the bounding-box wireframe is not clipped
            found[id(obj.material)] = obj.material
    return list(found.values())


# ---------------------------------------------------------------------------
# The model, the check and the events
# ---------------------------------------------------------------------------


def test_planes_given_at_construction_reach_the_materials(controller):
    store = _points_store()
    system = store.data_coordinate_systems[0]
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_points(
        data=store, scene_id=scene.id, clipping_planes=(_plane(system),)
    )
    controller.add_canvas(scene_id=scene.id)
    render_visual = _render_visual(controller, visual)
    assert render_visual.clipping_planes == visual.clipping_planes
    assert all(len(m.clipping_planes) == 1 for m in _live_materials(render_visual))


def test_an_assignment_emits_one_event_and_updates_the_materials(controller):
    store = _points_store()
    system = store.data_coordinate_systems[0]
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_points(data=store, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)
    events: list[ClippingPlanesChangedEvent] = []
    controller.on_clipping_planes_changed(visual.id, events.append, owner_id=uuid4())

    visual.clipping_planes = (_plane(system), _plane(system, 20.0, enabled=False))
    assert len(events) == 1
    assert events[0].clipping_planes == visual.clipping_planes
    assert events[0].source_id == controller._id
    material = _render_visual(controller, visual).node.material
    planes = [tuple(float(v) for v in p) for p in material.clipping_planes]
    # World x = data x here; the disabled plane keeps everything.
    assert planes == [(1.0, 0.0, 0.0, 16.0), (0.0, 0.0, 0.0, -1.0)]

    visual.clipping_planes = ()
    assert len(events) == 2
    assert len(material.clipping_planes) == 0


def test_set_clipping_planes_stamps_the_source(controller):
    store = _points_store()
    system = store.data_coordinate_systems[0]
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_points(data=store, scene_id=scene.id)
    events: list = []
    controller.on_clipping_planes_changed(visual.id, events.append, owner_id=uuid4())
    source = uuid4()
    controller.set_clipping_planes(visual.id, [_plane(system)], source_id=source)
    assert events[-1].source_id == source
    # The update event is the same request, from a widget.
    controller._incoming_events.emit(
        ClippingPlanesUpdateEvent(
            source_id=source, visual_id=visual.id, clipping_planes=()
        )
    )
    assert visual.clipping_planes == ()
    assert len(events) == 2


def test_a_plane_of_another_system_is_refused(controller):
    store = _points_store()
    system = store.data_coordinate_systems[0]
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_points(data=store, scene_id=scene.id)
    good = (_plane(system),)
    visual.clipping_planes = good
    with pytest.raises(ValueError, match="data coordinate system"):
        controller.set_clipping_planes(visual.id, [_plane(_system())])
    # A direct assignment is put back; psygnal wraps the error.
    with pytest.raises(Exception, match="data coordinate system"):
        visual.clipping_planes = (_plane(_system()),)
    assert visual.clipping_planes == good

    other_scene = controller.add_scene(dim="3d", name="t")
    with pytest.raises(ValueError, match="data coordinate system"):
        controller.add_points(
            data=_points_store(),
            scene_id=other_scene.id,
            clipping_planes=(_plane(system),),
        )


def test_a_plane_of_the_wrong_rank_is_refused(controller):
    from cellier.transform import Plane

    store = _points_store()
    system = store.data_coordinate_systems[0]
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_points(data=store, scene_id=scene.id)
    short = ClippingPlane(
        plane=Plane(coordinate_system=system.id, normal=(1, 0), offset=0)
    )
    with pytest.raises(ValueError, match="components"):
        controller.set_clipping_planes(visual.id, [short])


def test_two_planes_with_one_id_are_refused(controller):
    store = _points_store()
    system = store.data_coordinate_systems[0]
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_points(data=store, scene_id=scene.id)
    plane = _plane(system)
    twin = plane.model_copy(update={"enabled": False})
    with pytest.raises(ValueError, match="share an id"):
        controller.set_clipping_planes(visual.id, [plane, twin])
    assert visual.clipping_planes == ()


def test_one_plane_is_read_and_moved_by_its_id(controller):
    store = _points_store()
    system = store.data_coordinate_systems[0]
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_points(data=store, scene_id=scene.id)
    first, second = _plane(system), _plane(system, x=20.0, enabled=False)
    controller.set_clipping_planes(visual.id, [first, second])
    assert controller.get_clipping_plane(visual.id, second.id) == second

    events: list = []
    controller.on_clipping_planes_changed(visual.id, events.append, owner_id=uuid4())
    source = uuid4()
    new_plane = second.plane.model_copy(update={"offset": second.plane.offset + 3.0})
    moved = controller.set_clipping_plane(
        visual.id, second.id, new_plane, source_id=source
    )
    # The plane moved; its id, its place and its flag did not.
    assert visual.clipping_planes == (first, moved)
    assert moved.id == second.id
    assert moved.enabled is False
    assert moved.plane == new_plane
    assert [event.source_id for event in events] == [source]

    # The same plane again changes nothing and announces nothing.
    controller.set_clipping_plane(visual.id, second.id, new_plane, source_id=source)
    assert len(events) == 1

    with pytest.raises(KeyError):
        controller.get_clipping_plane(visual.id, uuid4())
    with pytest.raises(KeyError):
        controller.set_clipping_plane(visual.id, uuid4(), new_plane)


# ---------------------------------------------------------------------------
# Planes survive every material change (design 4.6)
# ---------------------------------------------------------------------------


async def test_planes_survive_material_changes(controller, tmp_path):
    scene = controller.add_scene(dim="3d", name="everything")
    rng = np.random.default_rng(0)
    shape = (16, 16, 16)
    positions, indices = uv_sphere(
        radius=6.0, centre=(8.0, 8.0, 8.0), n_lat=8, n_lon=16
    )
    visuals = {}
    stores = {
        "image": ImageMemoryStore(data=rng.random(shape, dtype=np.float32)),
        "labels": LabelMemoryStore(data=np.ones(shape, np.int32)),
        "mesh": MeshMemoryStore(positions=positions, indices=indices),
        "points": PointsMemoryStore(
            positions=(rng.random((50, 3)) * 16).astype(np.float32)
        ),
        "lines": LinesMemoryStore(
            positions=(rng.random((40, 3)) * 16).astype(np.float32)
        ),
        "graph": GraphMemoryStore(
            positions=(rng.random((20, 3)) * 16).astype(np.float32),
            edges=np.stack([np.arange(19), np.arange(1, 20)], axis=1),
        ),
    }
    visuals["image"] = controller.add_image(data=stores["image"], scene_id=scene.id)
    visuals["labels"] = controller.add_labels(data=stores["labels"], scene_id=scene.id)
    visuals["mesh"] = controller.add_mesh(
        data=stores["mesh"], scene_id=scene.id, appearance=MeshFlatAppearance()
    )
    visuals["points"] = controller.add_points(data=stores["points"], scene_id=scene.id)
    visuals["lines"] = controller.add_lines(data=stores["lines"], scene_id=scene.id)
    visuals["graph"] = controller.add_graph(data=stores["graph"], scene_id=scene.id)
    for kind in ("image_multiscale_mip", "labels_multiscale"):
        visuals[kind], stores[kind] = h.add_visual(
            kind, controller, scene.id, h.data_for(kind), tmp_path, kind
        )
    controller.add_canvas(scene_id=scene.id)

    # Before anything has loaded: points, lines and graph show placeholders.
    for name, visual in visuals.items():
        system = stores[name].data_coordinate_systems[0]
        visual.clipping_planes = (
            ClippingPlane.from_point_normal(system, (8, 8, 8), (0, 0, 1)),
        )

    def unclipped() -> dict:
        missing = {}
        for name, visual in visuals.items():
            materials = _live_materials(_render_visual(controller, visual))
            assert materials, name
            bad = [type(m).__name__ for m in materials if len(m.clipping_planes) != 1]
            if bad:
                missing[name] = bad
        return missing

    async def settle() -> None:
        controller.reslice_all()
        await drain_loading(controller)

    controller.fit_camera(scene.id)
    await settle()
    assert unclipped() == {}, "first load"
    for axes in ((1, 2), (0, 1, 2), (0, 2), (0, 1, 2)):
        controller.set_displayed_axes(scene.id, axes)
        await settle()
        assert unclipped() == {}, f"displayed axes {axes}"
    for mode in ("iso", "minip", "mip"):
        visuals["image"].single.render_mode = mode
        await settle()
        assert unclipped() == {}, f"image render mode {mode}"
    visuals["image_multiscale_mip"].single.render_mode = "iso"
    await settle()
    assert unclipped() == {}, "multiscale render mode"

    # A material that is swapped out must not bring stale planes back.
    for name in ("points", "lines"):
        store, full = stores[name], stores[name].positions
        store.positions = np.zeros((0, 3), dtype=np.float32)
        await settle()
        system = store.data_coordinate_systems[0]
        visuals[name].clipping_planes = (
            ClippingPlane.from_point_normal(system, (8, 8, 3), (0, 0, 1)),
        )
        store.positions = full
        await settle()
        material = _render_visual(controller, visuals[name]).node.material
        assert float(material.clipping_planes[0][3]) == 3.0, name


# ---------------------------------------------------------------------------
# Geometry in 3D
# ---------------------------------------------------------------------------


async def test_a_point_is_kept_or_dropped_whole(controller, render_scene):
    """A marker straddling the plane is not cut in half (design 4.4)."""
    system = _system()
    store = PointsMemoryStore(
        positions=np.array([[16, 16, 16]], dtype=np.float32),
        data_coordinate_systems=[system],
    )
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(size=40.0, size_space="screen"),
    )
    controller.add_canvas(scene_id=scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    def drawn() -> int:
        return int((render_scene(controller, scene.id, (SIZE, SIZE))[..., 3] > 0).sum())

    whole = drawn()
    assert whole > 300
    visual.clipping_planes = (_plane(system, 15.9),)  # the centre is kept
    assert drawn() == whole
    visual.clipping_planes = (_plane(system, 16.1),)  # the centre is clipped
    assert drawn() == 0


async def test_a_mesh_is_cut_open(controller, render_scene):
    system = _system()
    positions, indices = uv_sphere(
        radius=12.0, centre=(16.0, 16.0, 16.0), n_lat=24, n_lon=48
    )
    store = MeshMemoryStore(
        positions=positions, indices=indices, data_coordinate_systems=[system]
    )
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_mesh(
        data=store, scene_id=scene.id, appearance=MeshFlatAppearance()
    )
    controller.add_canvas(scene_id=scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    whole = render_scene(controller, scene.id, (SIZE, SIZE))[..., 3] > 0
    events: list = []
    controller.on_clipping_planes_changed(visual.id, events.append, owner_id=uuid4())
    visual.clipping_planes = (_plane(system, 16.0),)
    # The shader clips a 3D mesh: nothing is read again.
    assert not _render_visual(controller, visual).clipping_planes_affect_request
    cut = render_scene(controller, scene.id, (SIZE, SIZE))[..., 3] > 0
    assert 0 < cut.sum() < whole.sum()
    assert not (cut & ~whole).any()


@pytest.mark.parametrize("kind", ["lines", "graph"])
async def test_lines_and_graph_edges_are_cut_in_3d(kind, controller, render_scene):
    system = _system()
    rng = np.random.default_rng(1)
    positions = (rng.random((600, 3)) * 32).astype(np.float32)
    scene = controller.add_scene(dim="3d", name="s")
    if kind == "lines":
        store = LinesMemoryStore(positions=positions, data_coordinate_systems=[system])
        visual = controller.add_lines(data=store, scene_id=scene.id)
    else:
        store = GraphMemoryStore(
            positions=positions,
            edges=np.stack([np.arange(599), np.arange(1, 600)], axis=1),
            data_coordinate_systems=[system],
        )
        visual = controller.add_graph(data=store, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    whole = render_scene(controller, scene.id, (SIZE, SIZE))[..., 3] > 0
    visual.clipping_planes = (_plane(system, 16.0),)
    render_visual = _render_visual(controller, visual)
    assert not render_visual.clipping_planes_affect_request
    assert all(len(m.clipping_planes) == 1 for m in _live_materials(render_visual))
    cut = render_scene(controller, scene.id, (SIZE, SIZE))[..., 3] > 0
    # A cut segment ends in a cap, so a few pixels at the plane are new.
    assert 0.2 * whole.sum() < cut.sum() < 0.8 * whole.sum()


async def test_the_clipped_part_of_a_mesh_cannot_be_picked(controller):
    from rendercanvas.offscreen import RenderCanvas as OffscreenRenderCanvas

    system = _system()
    positions, indices = uv_sphere(
        radius=12.0, centre=(16.0, 16.0, 16.0), n_lat=24, n_lon=48
    )
    store = MeshMemoryStore(
        positions=positions, indices=indices, data_coordinate_systems=[system]
    )
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_mesh(
        data=store, scene_id=scene.id, appearance=MeshFlatAppearance()
    )
    controller.add_canvas(scene_id=scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    controller.fit_camera(scene.id)
    gfx_scene, camera = h.gfx_scene(controller, scene.id)

    canvas = OffscreenRenderCanvas(size=(SIZE, SIZE), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    canvas.request_draw(lambda: renderer.render(gfx_scene, camera))
    whole = np.asarray(canvas.draw())[..., 3] > 0
    visual.clipping_planes = (_plane(system, 16.0),)
    cut = np.asarray(canvas.draw())[..., 3] > 0

    def picked(mask) -> int:
        rows, cols = np.nonzero(mask)
        step = max(1, len(rows) // 60)
        return sum(
            renderer.get_pick_info((col + 0.5, row + 0.5)).get("world_object")
            is not None
            for row, col in zip(rows[::step], cols[::step])
        )

    from scipy.ndimage import binary_erosion

    removed = binary_erosion(whole & ~cut, iterations=2)
    kept = binary_erosion(cut, iterations=2)
    assert removed.sum() > 30
    assert picked(removed) == 0
    assert picked(kept) > 20


# ---------------------------------------------------------------------------
# Clipped regions are not fetched (design 5.1)
# ---------------------------------------------------------------------------


@pytest.fixture
def reads(monkeypatch) -> list:
    """Every chunk read the scheduler issues, as ``(cache id, key)``."""
    issued: list = []
    original = _scheduler.ChunkScheduler._read

    async def _read(scheduler, ticket):
        issued.append((ticket.cache_id, int(ticket.key)))
        await original(scheduler, ticket)

    monkeypatch.setattr(_scheduler.ChunkScheduler, "_read", _read)
    return issued


@pytest.mark.parametrize("dim", ["3d", "2d"])
@pytest.mark.parametrize("kind", ["image_multiscale_mip", "labels_multiscale"])
async def test_clipped_bricks_and_tiles_are_not_read(
    kind, dim, controller, reslice, tmp_path, reads
):
    counts = {}
    for tag, clip in (("whole", False), ("clipped", True)):
        scene = controller.add_scene(dim=dim, name=f"{tag}")
        visual, store = h.add_visual(
            kind, controller, scene.id, h.data_for(kind), tmp_path, f"{kind}_{tag}"
        )
        if clip:
            # 16-voxel bricks: brick 0 covers x in [-0.5, 15.5].  A brick
            # that only touches a plane is kept, so cut just past its face.
            visual.clipping_planes = h.clipping_planes(
                store, [((0, 0, 16.5), (0, 0, 1))]
            )
        controller.add_canvas(scene_id=scene.id)
        controller.update_slice_indices(scene.id, {0: 16 * h.Z_SCALE})
        before = len(reads)
        await reslice(controller, scene.id)
        counts[tag] = len(reads) - before
    # Half of the level-0 chunks; the coarse fallback level is still read.
    whole_level0 = 8 if dim == "3d" else 4
    assert counts["whole"] == whole_level0 + 1
    assert counts["clipped"] == whole_level0 // 2 + 1


async def test_a_plane_change_reslices_a_multiscale_visual_only(
    controller, reslice, tmp_path, reads
):
    scene = controller.add_scene(dim="3d", name="s")
    visual, store = h.add_visual(
        "image_multiscale_mip", controller, scene.id, h.image_data(), tmp_path, "ms"
    )
    points_store = _points_store()
    points = controller.add_points(data=points_store, scene_id=scene.id)
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    assert _render_visual(controller, visual).clipping_planes_affect_request
    assert not _render_visual(controller, points).clipping_planes_affect_request

    # Clip, then reveal: the bricks that come back into view are read.
    visual.clipping_planes = h.clipping_planes(store, [((0, 0, 16.5), (0, 0, 1))])
    await drain_loading(controller)
    before = len(reads)
    visual.clipping_planes = ()
    await drain_loading(controller)
    assert len(reads) == before  # still resident: nothing is read twice


# ---------------------------------------------------------------------------
# Geometry flattened in a 2D slab is clipped in the read (design 5.2)
# ---------------------------------------------------------------------------


async def _slab_scene(controller, add):
    scene = controller.add_scene(dim="2d", name="slab")
    visual, store = add(scene)
    controller.add_canvas(scene_id=scene.id)
    controller.update_slice_indices(scene.id, {0: 16.0})
    controller.update_thickness(scene.id, {0: 6.0})
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    return scene, visual, store


async def test_points_in_a_slab_are_clipped_by_their_true_position(
    controller, render_scene
):
    store = _points_store(20000)
    system = store.data_coordinate_systems[0]
    scene, visual, _ = await _slab_scene(
        controller,
        lambda scene: (controller.add_points(data=store, scene_id=scene.id), store),
    )
    render_visual = _render_visual(controller, visual)

    def drawn() -> int:
        return int((render_scene(controller, scene.id, (SIZE, SIZE))[..., 3] > 0).sum())

    whole = drawn()
    # A plane with a z component: the flattened slab has lost z.
    visual.clipping_planes = (
        ClippingPlane.from_point_normal(system, (16, 16, 16), (2, 0, 1)),
    )
    assert render_visual.clipping_planes_affect_request
    await drain_loading(controller)
    assert render_visual._clip_on_cpu
    assert len(render_visual.node.material.clipping_planes) == 0
    on_cpu = drawn()
    assert 0 < on_cpu < whole
    # The read kept exactly the points of the slab on the kept side.
    positions = store.positions
    in_slab = np.abs(positions[:, 0] - 16.0) <= 6.0
    kept = in_slab & (2 * positions[:, 0] + positions[:, 2] >= 48.0)
    assert render_visual.node.geometry.positions.nitems == int(kept.sum())

    # A plane on the displayed axes only: the shader is exact again.
    visual.clipping_planes = (
        ClippingPlane.from_point_normal(system, (16, 16), (0, 1), axes=("y", "x")),
    )
    await drain_loading(controller)
    assert not render_visual._clip_on_cpu
    assert len(render_visual.node.material.clipping_planes) == 1
    assert render_visual.node.geometry.positions.nitems == int(in_slab.sum())
    assert 0 < drawn() < whole


async def test_a_slab_mesh_section_is_clipped_with_its_fill(controller, render_scene):
    system = _system()
    positions, indices = uv_sphere(
        radius=12.0, centre=(16.0, 16.0, 16.0), n_lat=32, n_lon=64
    )
    store = MeshMemoryStore(
        positions=positions, indices=indices, data_coordinate_systems=[system]
    )
    scene, visual, _ = await _slab_scene(
        controller,
        lambda scene: (
            controller.add_mesh(
                data=store,
                scene_id=scene.id,
                appearance=MeshFlatAppearance(),
                section=MeshSectionConfig(mode="slab"),
            ),
            store,
        ),
    )

    def drawn() -> np.ndarray:
        return render_scene(controller, scene.id, (SIZE, SIZE))[..., 3] > 0

    whole = drawn()
    visual.clipping_planes = (
        ClippingPlane.from_point_normal(system, (16, 16, 16), (0.5, 0, 1)),
    )
    await drain_loading(controller)
    cut = drawn()
    # About half of the disc is left, filled: far more than an outline.
    assert 0.3 * whole.sum() < cut.sum() < 0.7 * whole.sum()


async def test_a_plane_change_keeps_a_slab_mesh_on_screen(controller):
    """D31: the last result stays until the new one lands; a slice move hides."""
    system = _system()
    positions, indices = uv_sphere(
        radius=12.0, centre=(16.0, 16.0, 16.0), n_lat=16, n_lon=32
    )
    store = MeshMemoryStore(
        positions=positions, indices=indices, data_coordinate_systems=[system]
    )
    scene, visual, _ = await _slab_scene(
        controller,
        lambda scene: (
            controller.add_mesh(
                data=store,
                scene_id=scene.id,
                appearance=MeshFlatAppearance(),
                section=MeshSectionConfig(mode="slab"),
            ),
            store,
        ),
    )
    residency = _render_visual(controller, visual)._residency
    assert residency.level_to_draw() is not None

    def plane(x):
        return (ClippingPlane.from_point_normal(system, (16, 16, x), (0.5, 0, 1)),)

    visual.clipping_planes = plane(14.0)
    await drain_loading(controller)
    visual.clipping_planes = plane(18.0)  # not drained: the read is in flight
    assert residency.level_to_draw() is not None
    await drain_loading(controller)
    assert residency.level_to_draw() is not None

    controller.update_slice_indices(scene.id, {0: 20.0})  # not drained
    assert residency.level_to_draw() is None
    await drain_loading(controller)
    assert residency.level_to_draw() is not None


async def test_the_lines_read_cuts_segments_at_the_plane():
    system = _system()
    positions = np.array(
        [[0, 0, 0], [0, 0, 10], [0, 5, 20], [0, 5, 30], [0, 9, 1], [0, 9, 2]],
        dtype=np.float32,
    )
    store = LinesMemoryStore(positions=positions, data_coordinate_systems=[system])
    request = LinesSliceRequest(
        slice_request_id=uuid4(),
        chunk_request_id=uuid4(),
        scale_index=0,
        displayed_axes=(1, 2),
        retained_axes=(1, 2),
        region=ConvexRegion(coordinate_system=system.id, ndim=3),
        clip_planes=(((0.0, 0.0, 1.0), 4.0),),  # keeps x >= 4
    )
    data = await store.get_data(request)
    np.testing.assert_allclose(data.positions, [[0, 4], [0, 10], [5, 20], [5, 30]])
    np.testing.assert_array_equal(data.original_edge_indices, [0, 1])


# ---------------------------------------------------------------------------
# Painting (D26)
# ---------------------------------------------------------------------------


async def test_the_brush_writes_only_what_is_drawn(controller):
    system = _system()
    store = LabelMemoryStore(
        data=np.zeros((16, 16, 16), dtype=np.int32), data_coordinate_systems=[system]
    )
    scene = controller.add_scene(dim="2d", name="paint")
    visual = controller.add_labels(data=store, scene_id=scene.id)
    canvas_id = controller.add_canvas(scene_id=scene.id)
    controller.update_slice_indices(scene.id, {0: 8.0})
    controller.reslice_all()
    await drain_loading(controller)
    visual.clipping_planes = (
        ClippingPlane.from_point_normal(system, (0, 0, 8), (0, 0, 1)),  # x >= 8
    )
    paint = controller.add_paint_controller(
        visual.id,
        controller.get_canvas_ids(scene.id)[0],
        brush_value=5,
        brush_radius_voxels=3.0,
    )
    assert canvas_id is not None
    # The pointer's world coordinate: z, then pygfx (x, y) on the two
    # displayed axes.  Centred on the plane.
    paint._apply_brush(np.array([8.0, 8.0, 8.0]))
    painted = np.argwhere(store.data == 5)
    assert len(painted) > 10
    assert painted[:, 2].min() == 8
    assert painted[:, 2].max() == 11
