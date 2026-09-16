"""``pick_write=`` and ``outline_mode=`` on the ``Viewer.add_*`` methods.

Both are written onto the model before the visual is registered, through the
same path as ``outline=`` and ``ambient_occlusion=``, so the render layer sees
them on the first frame and the pick-buffer warnings fire from one place.
"""

from __future__ import annotations

import inspect

import pytest

from cellier.convenience import Viewer
from cellier.render import OutlineConfig, RenderManagerConfig
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageAppearance,
    InMemoryImageChannelAppearance,
    InMemoryImageSingleAppearance,
    MeshFlatAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelsAppearance,
    VisualOutline,
)

_ZYX = spatial_axes("z", "y", "x")
_ZCYX = [("z", "space"), ("c", "channel"), ("y", "space"), ("x", "space")]
_CZYX = [("c", "channel"), ("z", "space"), ("y", "space"), ("x", "space")]


def _channels(cls) -> dict:
    return {i: cls(color_map="viridis") for i in range(2)}


# method -> (world, dim, store fixture, how to call it)
_ADDERS = {
    "add_image": (
        _ZYX,
        "2d",
        "image_store",
        lambda v, s, **kw: v.add_image(
            s,
            appearance=InMemoryImageAppearance(),
            **kw,
            single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
        ),
    ),
    "add_labels": (
        _ZYX,
        "2d",
        "labels_store",
        lambda v, s, **kw: v.add_labels(s, **kw),
    ),
    "add_mesh": (
        _ZYX,
        "3d",
        "mesh_store",
        lambda v, s, **kw: v.add_mesh(s, appearance=MeshFlatAppearance(), **kw),
    ),
    "add_points": (
        _ZYX,
        "3d",
        "points_store",
        lambda v, s, **kw: v.add_points(s, **kw),
    ),
    "add_graph": (_ZYX, "3d", "graph_store", lambda v, s, **kw: v.add_graph(s, **kw)),
    "add_lines": (_ZYX, "3d", "lines_store", lambda v, s, **kw: v.add_lines(s, **kw)),
    "add_image_multiscale": (
        _ZYX,
        "2d",
        "multiscale_image_store",
        lambda v, s, **kw: v.add_image_multiscale(
            s,
            appearance=MultiscaleImageAppearance(),
            **kw,
            single=MultiscaleImageSingleAppearance(
                color_map="viridis", render_mode="mip"
            ),
        ),
    ),
    "add_labels_multiscale": (
        _ZYX,
        "2d",
        "multiscale_labels_store",
        lambda v, s, **kw: v.add_labels_multiscale(
            s, appearance=MultiscaleLabelsAppearance(), **kw
        ),
    ),
    "add_image[composite]": (
        _ZCYX,
        "2d",
        "multichannel_store",
        lambda v, s, **kw: v.add_image(
            s,
            channel_axis=1,
            composite=True,
            channels=_channels(InMemoryImageChannelAppearance),
            **kw,
        ),
    ),
    "add_image_multiscale[composite]": (
        _CZYX,
        "2d",
        "multichannel_multiscale_store",
        lambda v, s, **kw: v.add_image_multiscale(
            s,
            channel_axis=0,
            composite=True,
            channels=_channels(MultiscaleImageChannelAppearance),
            **kw,
        ),
    ),
}

_LABELS_METHODS = ("add_labels", "add_labels_multiscale")


def _add(method: str, request, viewer_kwargs: dict | None = None, **kwargs):
    world, dim, fixture, call = _ADDERS[method]
    viewer = Viewer(world, dim=dim, **(viewer_kwargs or {}))
    return call(viewer, request.getfixturevalue(fixture), **kwargs)


def test_every_add_method_takes_pick_write_and_only_labels_take_outline_mode():
    parameters = {
        name: inspect.signature(getattr(Viewer, name.split("[")[0])).parameters
        for name in _ADDERS
    }
    assert all("pick_write" in params for params in parameters.values())
    assert {
        name for name, params in parameters.items() if "outline_mode" in params
    } == set(_LABELS_METHODS)


@pytest.mark.parametrize("method", list(_ADDERS))
def test_pick_write_defaults_to_true(method, request):
    assert _add(method, request).pick_write is True


@pytest.mark.parametrize("method", list(_ADDERS))
def test_pick_write_false_reaches_the_model(method, request):
    assert _add(method, request, pick_write=False).pick_write is False


def test_an_outline_turns_pick_write_back_on_with_a_warning(request):
    """An outline is drawn from the pick buffer, so the outline wins.

    The outline pass is on, so the pick_write warning is the only one.
    """
    outlines_on = RenderManagerConfig(outline=OutlineConfig(enabled=True))
    with pytest.warns(RuntimeWarning, match="pick_write"):
        visual = _add(
            "add_image",
            request,
            viewer_kwargs={"render_config": outlines_on},
            pick_write=False,
            outline=VisualOutline(slot=1),
        )

    assert visual.pick_write is True


@pytest.mark.parametrize("method", _LABELS_METHODS)
def test_outline_mode_defaults_to_per_label(method, request):
    assert _add(method, request).outline_mode == "per_label"


@pytest.mark.parametrize("mode", ["whole_object", "all_boundaries"])
@pytest.mark.parametrize("method", _LABELS_METHODS)
def test_outline_mode_reaches_the_model(method, mode, request):
    assert _add(method, request, outline_mode=mode).outline_mode == mode


def test_an_unknown_outline_mode_is_refused(request):
    with pytest.raises(ValueError, match="outline_mode"):
        _add("add_labels", request, outline_mode="silhouette")


def _gfx_pick_writes(viewer, visual) -> list[bool]:
    """``pick_write`` on every material the renderer built for *visual*."""
    import pygfx as gfx

    scene_manager = viewer.controller._render_manager._scenes[viewer.scene.id]
    gfx_visual = scene_manager.get_visual(visual.id)
    seen: set[int] = set()
    found: list[bool] = []
    for value in vars(gfx_visual).values():
        if not isinstance(value, gfx.WorldObject):
            continue
        for obj in value.iter():
            material = getattr(obj, "material", None)
            if material is not None and id(material) not in seen:
                seen.add(id(material))
                found.append(material.pick_write)
    return found


@pytest.mark.parametrize("method", list(_ADDERS))
def test_pick_write_false_reaches_every_render_material(method, request):
    """The model alone is not enough: the renderer must build unpickable too.

    Multiscale labels used to build their label materials pickable whatever
    the model said, which only a later change to ``pick_write`` corrected.
    """
    world, dim, fixture, call = _ADDERS[method]
    viewer = Viewer(world, dim=dim)
    visual = call(viewer, request.getfixturevalue(fixture), pick_write=False)

    pick_writes = _gfx_pick_writes(viewer, visual)

    assert pick_writes
    assert not any(pick_writes)
