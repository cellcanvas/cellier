"""TypedDict companions for appearance and render-config model classes.

Each TypedDict mirrors the fields of its corresponding Pydantic model so that
callers can pass plain dicts to ``Viewer.add_*`` methods without importing any
model class.  IDEs can autocomplete the keys and mypy can type-check them.

These types are a convenience-layer concern only -- they are not used by
``CellierController`` or any other non-convenience code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from cmap import Colormap


class BaseAppearanceKwargs(TypedDict, total=False):
    """Keys shared by all appearance models (mirrors ``BaseAppearance``)."""

    visible: bool
    opacity: float
    render_order: int
    depth_test: bool
    depth_write: bool
    depth_compare: Literal["<", "<=", "==", "!=", ">=", ">"]
    transparency_mode: Literal["blend", "add", "weighted_blend", "weighted_solid"]


class BaseImageAppearanceKwargs(BaseAppearanceKwargs, total=False):
    """Keys shared by all image appearance models (mirrors ``BaseImageAppearance``)."""

    color_map: Colormap | str
    clim: tuple[float, float]
    interpolation: Literal["linear", "nearest"]


class InMemoryImageAppearanceKwargs(BaseImageAppearanceKwargs, total=False):
    """Dict form of ``InMemoryImageAppearance``.

    Accepted by ``Viewer.add_image`` in place of an
    ``InMemoryImageAppearance`` instance.

    Keys
    ----
    color_map : Colormap or str
        Colormap applied after contrast normalisation. Accepts any
        cmap-registered name string (e.g. ``"viridis"``).
    clim : tuple[float, float]
        Contrast limits ``(min, max)``. Default ``(0.0, 1.0)``.
    interpolation : "linear" or "nearest"
        Texture sampler filter. Default ``"nearest"``.
    render_mode : "mip", "iso", or "minip"
        Volume rendering mode for the 3D view. Default ``"mip"``.
        Ignored by the 2D view.
    iso_threshold : float
        Isosurface threshold used when ``render_mode == "iso"``.
        Default ``0.5``.
    visible : bool
        Default ``True``.
    opacity : float
        Master opacity in [0, 1]. Default ``1.0``.
    render_order : int
        Pygfx render order. Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Depth comparison operator. Default ``"<"``.
    transparency_mode : str
        Default ``"blend"``.
    """

    render_mode: Literal["mip", "iso", "minip"]
    iso_threshold: float


class MultiscaleImageAppearanceKwargs(BaseImageAppearanceKwargs, total=False):
    """Dict form of ``MultiscaleImageAppearance``.

    Accepted by ``Viewer.add_image_multiscale`` and
    ``Viewer.add_multichannel_image_multiscale`` in place of a
    ``MultiscaleImageAppearance`` instance.

    Keys
    ----
    color_map : Colormap or str
        Colormap applied after contrast normalisation.
    clim : tuple[float, float]
        Contrast limits ``(min, max)``. Default ``(0.0, 1.0)``.
    interpolation : "linear" or "nearest"
        Texture sampler filter. Default ``"nearest"``.
    lod_bias : float
        Multiplier on the screen-space LOD threshold. Default ``1.0``.
    force_level : int or None
        Overrides automatic LOD selection when set. Default ``None``.
    frustum_cull : bool
        Skip bricks outside the camera frustum. Default ``True``.
    iso_threshold : float
        Isosurface threshold for 3D raycast rendering. Default ``0.2``.
    render_mode : str
        Volume rendering mode. Default ``"iso"``.
    attenuation : float
        Depth attenuation coefficient for ``"attenuated_mip"`` mode.
        Default ``1.0``.
    visible : bool
        Default ``True``.
    opacity : float
        Master opacity in [0, 1]. Default ``1.0``.
    render_order : int
        Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Default ``"<"``.
    transparency_mode : str
        Default ``"blend"``.
    """

    lod_bias: float
    force_level: int | None
    frustum_cull: bool
    iso_threshold: float
    render_mode: Literal["iso", "mip", "smooth_iso", "attenuated_mip"]
    attenuation: float


class BaseLabelsAppearanceKwargs(BaseAppearanceKwargs, total=False):
    """Keys shared by all label appearance models (mirrors ``BaseLabelsAppearance``)."""

    colormap_mode: Literal["random", "direct"]
    background_label: int
    salt: int
    color_dict: dict[int, tuple[float, float, float, float]]
    render_mode: str


class InMemoryLabelsAppearanceKwargs(BaseLabelsAppearanceKwargs, total=False):
    """Dict form of ``InMemoryLabelsAppearance``.

    Accepted by ``Viewer.add_labels`` in place of an
    ``InMemoryLabelsAppearance`` instance.

    Keys
    ----
    colormap_mode : "random" or "direct"
        Frozen after construction. Default ``"random"``.
    background_label : int
        Label ID treated as transparent. Default ``0``.
    salt : int
        Hash seed for random colormap mode. Default ``0``.
    color_dict : dict[int, tuple[float, float, float, float]]
        Explicit label-ID to RGBA mapping for direct mode.
    render_mode : "iso_categorical" or "flat_categorical"
        3D rendering mode. Default ``"iso_categorical"``.
    visible : bool
        Default ``True``.
    opacity : float
        Master opacity in [0, 1]. Default ``1.0``.
    render_order : int
        Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Default ``"<"``.
    transparency_mode : str
        Default ``"blend"``.
    """

    render_mode: Literal["iso_categorical", "flat_categorical"]


class MultiscaleLabelsAppearanceKwargs(BaseLabelsAppearanceKwargs, total=False):
    """Dict form of ``MultiscaleLabelsAppearance``.

    Accepted by ``Viewer.add_labels_multiscale`` in place of a
    ``MultiscaleLabelsAppearance`` instance.

    Keys
    ----
    colormap_mode : "random" or "direct"
        Frozen after construction. Default ``"random"``.
    background_label : int
        Default ``0``.
    salt : int
        Default ``0``.
    color_dict : dict[int, tuple[float, float, float, float]]
        Explicit label-ID to RGBA mapping.
    render_mode : str
        Wider than in-memory variant; also accepts ``"gradient_debug"``
        and ``"smooth_iso"``. Default ``"iso_categorical"``.
    lod_bias : float
        Multiplier on the screen-space LOD threshold. Default ``1.0``.
    force_level : int or None
        Overrides automatic LOD selection when set. Default ``None``.
    frustum_cull : bool
        Skip bricks outside the camera frustum. Default ``True``.
    visible : bool
        Default ``True``.
    opacity : float
        Master opacity in [0, 1]. Default ``1.0``.
    render_order : int
        Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Default ``"<"``.
    transparency_mode : str
        Default ``"blend"``.
    """

    render_mode: Literal[
        "iso_categorical", "flat_categorical", "gradient_debug", "smooth_iso"
    ]
    lod_bias: float
    force_level: int | None
    frustum_cull: bool


class MeshFlatAppearanceKwargs(BaseAppearanceKwargs, total=False):
    """Dict form of ``MeshFlatAppearance``.

    Accepted by ``Viewer.add_mesh`` to select flat (unlit) shading.
    Must include ``appearance_type: "flat"`` so Pydantic can pick the
    correct variant.

    Keys
    ----
    appearance_type : "flat"
        Required discriminator key; must be ``"flat"``.
    color : tuple[float, float, float, float]
        Uniform RGBA color. Default ``(0.7, 0.7, 0.7, 1.0)``.
    color_mode : "uniform", "vertex", or "face"
        Default ``"uniform"``.
    wireframe : bool
        Render only edges. Default ``False``.
    wireframe_thickness : float
        Edge thickness in screen pixels. Default ``1.0``.
    side : "both", "front", or "back"
        Default ``"both"``.
    visible : bool
        Default ``True``.
    opacity : float
        Master opacity in [0, 1]. Default ``1.0``.
    render_order : int
        Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Default ``"<"``.
    transparency_mode : str
        Default ``"blend"``.
    """

    appearance_type: Literal["flat"]
    color: tuple[float, float, float, float]
    color_mode: Literal["uniform", "vertex", "face"]
    wireframe: bool
    wireframe_thickness: float
    side: Literal["both", "front", "back"]


class MeshPhongAppearanceKwargs(BaseAppearanceKwargs, total=False):
    """Dict form of ``MeshPhongAppearance``.

    Accepted by ``Viewer.add_mesh`` to select Phong shading.
    Must include ``appearance_type: "phong"`` so Pydantic can pick the
    correct variant.

    Keys
    ----
    appearance_type : "phong"
        Required discriminator key; must be ``"phong"``.
    color : tuple[float, float, float, float]
        Uniform RGBA diffuse color. Default ``(0.4, 0.6, 0.9, 1.0)``.
    color_mode : "uniform", "vertex", or "face"
        Default ``"uniform"``.
    shininess : float
        Specular exponent. Default ``30.0``.
    side : "both", "front", or "back"
        Default ``"front"``.
    flat_shading : bool
        Use face normals instead of smooth vertex normals. Default ``False``.
    visible : bool
        Default ``True``.
    opacity : float
        Master opacity in [0, 1]. Default ``1.0``.
    render_order : int
        Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Default ``"<"``.
    transparency_mode : str
        Default ``"blend"``.
    """

    appearance_type: Literal["phong"]
    color: tuple[float, float, float, float]
    color_mode: Literal["uniform", "vertex", "face"]
    shininess: float
    side: Literal["both", "front", "back"]
    flat_shading: bool


class PointsMarkerAppearanceKwargs(BaseAppearanceKwargs, total=False):
    """Dict form of ``PointsMarkerAppearance``.

    Accepted by ``Viewer.add_points`` in place of a
    ``PointsMarkerAppearance`` instance.

    Keys
    ----
    color : tuple[float, float, float, float]
        Uniform RGBA fallback color. Default ``(1.0, 1.0, 1.0, 1.0)``.
    size : float
        Uniform point size in ``size_space`` units. Default ``5.0``.
    size_space : "screen" or "world"
        Coordinate space for point size. Default ``"screen"``.
    color_mode : "uniform" or "vertex"
        Where RGB comes from. A declaration, never inferred from the data.
        Default ``"uniform"``.
    size_mode : "uniform" or "vertex"
        Where the point size comes from, on the same terms as
        ``color_mode``. Default ``"uniform"``.
    visible : bool
        Default ``True``.
    opacity : float
        Master opacity in [0, 1]. Default ``1.0``.
    render_order : int
        Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Default ``"<"``.
    transparency_mode : str
        Default ``"blend"``.
    """

    color: tuple[float, float, float, float]
    size: float
    size_space: Literal["screen", "world"]
    color_mode: Literal["uniform", "vertex"]
    size_mode: Literal["uniform", "vertex"]


class GraphAppearanceKwargs(BaseAppearanceKwargs, total=False):
    """Dict form of ``GraphAppearance``.

    Accepted by ``Viewer.add_graph`` in place of a ``GraphAppearance``
    instance.  Flat with ``node_`` / ``edge_`` prefixes, mirroring the
    model: psygnal does not propagate nested model changes, so the
    appearance is not composed from the points and lines models.

    Keys
    ----
    node_color : tuple[float, float, float, float]
        Uniform RGBA node color. Default ``(1.0, 1.0, 1.0, 1.0)``.
    node_size : float
        Uniform node size in ``node_size_space`` units. Default ``5.0``.
    node_size_space : "screen" or "world"
        Default ``"screen"``.
    node_color_mode : "uniform" or "vertex"
        Where node RGB comes from. A declaration, never inferred from the
        data. Default ``"uniform"``.
    node_size_mode : "uniform" or "vertex"
        Where the node size comes from, on the same terms as
        ``node_color_mode``. Default ``"uniform"``.
    node_visible : bool
        Show the node sub-visual. Nests under ``visible``. Default ``True``.
    node_pick_write : bool
        Default ``True``.
    node_depth_compare : str
        Default ``"<="``, so a node coplanar with an edge reads on top.
        Deliberately differs from every other appearance model's ``"<"``.
    edge_color : tuple[float, float, float, float]
        Uniform RGBA edge color. Default ``(0.7, 0.7, 0.7, 1.0)`` -- dimmer
        than the node default so nodes read on top.
    edge_thickness : float
        Default ``2.0``.
    edge_thickness_space : "screen" or "world"
        Default ``"screen"``.
    edge_color_mode : "uniform" or "vertex"
        Default ``"uniform"``.
    edge_visible : bool
        Default ``True``.
    edge_pick_write : bool
        Default ``True``.
    edge_depth_compare : str
        Default ``"<"``.
    visible : bool
        Default ``True``.
    opacity : float
        Master opacity in [0, 1], applied to both sub-visuals. Default ``1.0``.
    render_order : int
        Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Default ``"<"``. Overridden per child by ``node_depth_compare`` and
        ``edge_depth_compare``.
    transparency_mode : str
        Default ``"blend"``.
    """

    node_color: tuple[float, float, float, float]
    node_size: float
    node_size_space: Literal["screen", "world"]
    node_color_mode: Literal["uniform", "vertex"]
    node_size_mode: Literal["uniform", "vertex"]
    node_visible: bool
    node_pick_write: bool
    node_depth_compare: Literal["<", "<=", "==", "!=", ">=", ">"]
    edge_color: tuple[float, float, float, float]
    edge_thickness: float
    edge_thickness_space: Literal["screen", "world"]
    edge_color_mode: Literal["uniform", "vertex"]
    edge_visible: bool
    edge_pick_write: bool
    edge_depth_compare: Literal["<", "<=", "==", "!=", ">=", ">"]


class LinesMemoryAppearanceKwargs(BaseAppearanceKwargs, total=False):
    """Dict form of ``LinesMemoryAppearance``.

    Accepted by ``Viewer.add_lines`` in place of a
    ``LinesMemoryAppearance`` instance.

    Keys
    ----
    color : tuple[float, float, float, float]
        Uniform RGBA fallback color. Default ``(1.0, 1.0, 1.0, 1.0)``.
    thickness : float
        Line thickness in ``thickness_space`` units. Default ``2.0``.
    thickness_space : "screen" or "world"
        Coordinate space for line thickness. Default ``"screen"``.
    color_mode : "uniform" or "vertex"
        Default ``"uniform"``.
    visible : bool
        Default ``True``.
    opacity : float
        Master opacity in [0, 1]. Default ``1.0``.
    render_order : int
        Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Default ``"<"``.
    transparency_mode : str
        Default ``"blend"``.
    """

    color: tuple[float, float, float, float]
    thickness: float
    thickness_space: Literal["screen", "world"]
    color_mode: Literal["uniform", "vertex"]


class ChannelAppearanceKwargs(BaseAppearanceKwargs, total=False):
    """Dict form of ``ChannelAppearance``.

    Accepted in the ``channels`` dict of ``Viewer.add_multichannel_image``
    and ``Viewer.add_multichannel_image_multiscale`` in place of a
    ``ChannelAppearance`` instance.

    Keys
    ----
    color_map : Colormap or str
        Colormap for this channel.
    clim : tuple[float, float]
        Contrast limits ``(min, max)``. Default ``(0.0, 1.0)``.
    render_mode_3d : "mip" or "iso"
        Volume render mode for 3D. Default ``"mip"``.
    iso_threshold : float
        Isosurface threshold used when ``render_mode_3d == "iso"``.
        Default ``0.5``.
    transparency_mode : str
        Default ``"add"`` (overrides ``BaseAppearance`` default of
        ``"blend"``).
    visible : bool
        Default ``True``.
    opacity : float
        Per-channel opacity in [0, 1]. Default ``1.0``.
    render_order : int
        Default ``0``.
    depth_test : bool
        Default ``True``.
    depth_write : bool
        Default ``True``.
    depth_compare : str
        Default ``"<"``.
    """

    color_map: Colormap | str
    clim: tuple[float, float]
    render_mode_3d: Literal["mip", "iso"]
    iso_threshold: float
    transparency_mode: Literal["blend", "add", "weighted_blend", "weighted_solid"]


class MultiscaleImageRenderConfigKwargs(TypedDict, total=False):
    """Dict form of ``MultiscaleImageRenderConfig``.

    Accepted by ``Viewer.add_image_multiscale`` and
    ``Viewer.add_multichannel_image_multiscale`` in place of a
    ``MultiscaleImageRenderConfig`` instance.

    Keys
    ----
    block_size : int
        Brick / tile side length in voxels. Default ``32``.
    gpu_budget_bytes : int
        Maximum GPU memory for the 3D brick cache. Default 1 GiB.
    gpu_budget_bytes_2d : int
        Maximum GPU memory for the 2D tile cache. Default 64 MiB.
    paint_max_tiles : int
        Maximum number of finest-level tiles paintable in one session.
        Default ``512``.
    """

    block_size: int
    gpu_budget_bytes: int
    gpu_budget_bytes_2d: int
    paint_max_tiles: int


class MultiscaleLabelRenderConfigKwargs(TypedDict, total=False):
    """Dict form of ``MultiscaleLabelRenderConfig``.

    Accepted by ``Viewer.add_labels_multiscale`` in place of a
    ``MultiscaleLabelRenderConfig`` instance.

    Keys
    ----
    block_size : int
        Brick / tile side length in voxels. Default ``32``.
    gpu_budget_bytes : int
        Maximum GPU memory for the 3D brick cache. Default 1 GiB.
    gpu_budget_bytes_2d : int
        Maximum GPU memory for the 2D tile cache. Default 64 MiB.
    paint_max_tiles : int
        Maximum number of finest-level tiles paintable in one session.
        Default ``512``.
    """

    block_size: int
    gpu_budget_bytes: int
    gpu_budget_bytes_2d: int
    paint_max_tiles: int


class InMemoryImageControlsKwargs(TypedDict, total=False):
    """Dict form of ``InMemoryImageControlsConfig``.

    Accepted by ``Viewer.add_image`` in place of an
    ``InMemoryImageControlsConfig`` instance.

    Keys
    ----
    appearance : list[str] or False
        Appearance fields to show, in display order.  ``False`` hides the
        panel.  Example: ``["color_map", "clim", "render_mode"]``.
    colormap_names : list[str]
        Names available in the colormap dropdown.
    clim_range : tuple[float, float]
        ``(min, max)`` bounds for the contrast-limits slider.
    """

    appearance: list[str] | bool
    colormap_names: list[str]
    clim_range: tuple[float, float]


class MultiscaleImageControlsKwargs(TypedDict, total=False):
    """Dict form of ``MultiscaleImageControlsConfig``.

    Accepted by ``Viewer.add_image_multiscale`` in place of a
    ``MultiscaleImageControlsConfig`` instance.

    Keys
    ----
    appearance : list[str] or False
        Appearance fields to show, in display order.  ``False`` hides the
        panel.  Example: ``["color_map", "clim", "render_mode",
        "iso_threshold", "attenuation", "lod_bias"]``.
    colormap_names : list[str]
        Names available in the colormap dropdown.
    clim_range : tuple[float, float]
        ``(min, max)`` bounds for the contrast-limits slider.
    dataset_info : str
        Pre-formatted HTML for the dataset-info detail block.
    """

    appearance: list[str] | bool
    colormap_names: list[str]
    clim_range: tuple[float, float]
    dataset_info: str


class ChannelControlsKwargs(TypedDict, total=False):
    """Dict form of ``ChannelControlsConfig``.

    Accepted by ``Viewer.add_multichannel_image`` /
    ``add_multichannel_image_multiscale`` (and the ``OrthoViewer`` equivalents)
    in place of a ``ChannelControlsConfig`` instance.

    Keys
    ----
    fields : list[str]
        Per-channel fields to expose, in display order.  Defaults to
        ``["visible", "color_map", "clim", "opacity"]``.
    colormap_names : list[str]
        Names available in each channel's colormap control.
    clim_range : tuple[float, float]
        ``(min, max)`` bounds for the contrast-limits sliders.
    channel_labels : dict[int, str]
        Optional per-channel display labels keyed by channel index.
    """

    fields: list[str]
    colormap_names: list[str]
    clim_range: tuple[float, float]
    channel_labels: dict[int, str]


class SidecarKwargs(TypedDict, total=False):
    """Dict form of ``SidecarOptions``.

    Accepted by ``display()`` in place of a ``SidecarOptions`` instance.

    Keys
    ----
    title : str
        Tab title. Defaults to ``"Cellier"``.
    anchor : str
        Placement of the sidecar panel. One of ``"right"``, ``"split-right"``,
        ``"split-left"``, ``"split-top"``, ``"split-bottom"``,
        ``"tab-before"``, ``"tab-after"``. Default ``"right"``.
    ref : SidecarOptions.ref
        Anchor relative to another sidecar's ``DisplayHandle``.
    """

    title: str
    anchor: Literal[
        "right",
        "split-right",
        "split-left",
        "split-top",
        "split-bottom",
        "tab-before",
        "tab-after",
    ]
    ref: object


__all__ = [
    "BaseAppearanceKwargs",
    "BaseImageAppearanceKwargs",
    "BaseLabelsAppearanceKwargs",
    "ChannelAppearanceKwargs",
    "ChannelControlsKwargs",
    "GraphAppearanceKwargs",
    "InMemoryImageAppearanceKwargs",
    "InMemoryImageControlsKwargs",
    "InMemoryLabelsAppearanceKwargs",
    "LinesMemoryAppearanceKwargs",
    "MeshFlatAppearanceKwargs",
    "MeshPhongAppearanceKwargs",
    "MultiscaleImageAppearanceKwargs",
    "MultiscaleImageControlsKwargs",
    "MultiscaleImageRenderConfigKwargs",
    "MultiscaleLabelRenderConfigKwargs",
    "MultiscaleLabelsAppearanceKwargs",
    "PointsMarkerAppearanceKwargs",
    "SidecarKwargs",
]
