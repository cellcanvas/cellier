# src/cellier/v2/visuals/_mesh_memory.py
from typing import Annotated, Literal, Union

from psygnal import EventedModel
from pydantic import ConfigDict, Field

from cellier.visuals._base_visual import BaseAppearance, BaseVisual


class MeshFlatAppearance(BaseAppearance):
    """Flat (unlit) mesh — maps to MeshBasicMaterial.

    No lights required.  Suitable for false-color meshes, wireframe
    overlays, and rapid inspection.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        Uniform RGBA, used when color_mode is ``"uniform"``.
    color_mode : str
        ``"uniform"``, ``"vertex"``, or ``"face"``.
    wireframe : bool
        Render only edges.  Default False.
    wireframe_thickness : float
        Edge thickness in screen pixels.  Default 1.0.
    opacity : float
        0-1 alpha multiplier.  Default 1.0.
    side : str
        ``"both"``, ``"front"``, or ``"back"``.  Default ``"both"``.
    """

    appearance_type: Literal["flat"] = "flat"
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    color_mode: Literal["uniform", "vertex", "face"] = "uniform"
    wireframe: bool = False
    wireframe_thickness: float = 1.0
    side: Literal["both", "front", "back"] = "both"


class MeshPhongAppearance(BaseAppearance):
    """Phong-shaded mesh — maps to MeshPhongMaterial.

    Requires lights in the scene.  Pass ``lighting="default"`` to
    ``controller.add_scene()`` to add ambient + directional lights.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        Uniform RGBA diffuse color, used when color_mode is
        ``"uniform"``.
    color_mode : str
        ``"uniform"``, ``"vertex"``, or ``"face"``.
    shininess : float
        Specular exponent.  Default 30.
    opacity : float
        0-1 alpha multiplier.  Default 1.0.
    side : str
        ``"both"``, ``"front"``, or ``"back"``.  Default ``"front"``.
    flat_shading : bool
        Use face normals instead of smooth vertex normals.
        Default False.
    """

    appearance_type: Literal["phong"] = "phong"
    color: tuple[float, float, float, float] = (0.4, 0.6, 0.9, 1.0)
    color_mode: Literal["uniform", "vertex", "face"] = "uniform"
    shininess: float = 30.0
    side: Literal["both", "front", "back"] = "front"
    flat_shading: bool = False


MeshAppearance = Annotated[
    Union[MeshFlatAppearance, MeshPhongAppearance],
    Field(discriminator="appearance_type"),
]


class MeshSectionConfig(EventedModel):
    """How a mesh is drawn in a 2D view: its cross-section.

    A 2D view cuts the mesh with the slice plane and draws the cut: an
    **outline** where the surface crosses the plane, and a **fill** where
    the outline closes into loops.  Both use the mesh's appearance (colour,
    opacity).  An open surface has no closed loop, so it draws its outline
    only.  A 3D view is not affected.

    Only spatial, continuous axes are cut.  Along any other sliced axis (a
    time axis, a discrete axis) the mesh is filtered face by face, as in 3D.

    Parameters
    ----------
    mode : {"cut", "slab"}
        ``"cut"`` (default) draws the cut by the slice plane itself; the
        scene's thickness on the cut axis is ignored, so a thick slab that
        shows many image planes still shows one mesh outline.  ``"slab"``
        draws what lies inside the scene's slab: the surface between its two
        faces, flattened, with a cap at each face and both cuts as the
        outline.  With no thickness the two are the same.
    outline : bool
        Draw the outline.  Default ``True``.
    fill : bool
        Draw the fill.  Default ``True``.
    outline_width : float
        Outline thickness in screen pixels.  Default 2.

    Changing ``mode``, ``outline`` or ``fill`` reads the mesh again;
    ``outline_width`` applies at once.
    """

    # A setting is checked when it is assigned too, so a width of zero or
    # an unknown mode is refused rather than drawn.
    model_config = ConfigDict(validate_assignment=True)

    mode: Literal["cut", "slab"] = "cut"
    outline: bool = True
    fill: bool = True
    outline_width: float = Field(default=2.0, gt=0)


#: ``MeshSectionConfig`` fields that change what is read (the request key).
SECTION_RESLICE_FIELDS: frozenset[str] = frozenset({"mode", "outline", "fill"})


class BaseMeshVisual(BaseVisual):
    """What every mesh visual model has: an appearance and a 2D section.

    Parameters
    ----------
    appearance : MeshFlatAppearance | MeshPhongAppearance
        Appearance parameters.
    section : MeshSectionConfig
        How the mesh is drawn in a 2D view.
    requires_camera_reslice : bool
        Always False; frozen.  Camera movement does not trigger reslicing.
    """

    appearance: MeshAppearance
    section: MeshSectionConfig = Field(default_factory=MeshSectionConfig)
    requires_camera_reslice: bool = Field(default=False, frozen=True)


class MeshVisual(BaseMeshVisual):
    """Model-layer visual for an in-memory triangle mesh.

    Parameters
    ----------
    visual_type : Literal["mesh_memory"]
        Discriminator field; always ``"mesh_memory"``.
    data_store_id : str
        UUID string of the associated ``MeshMemoryStore``.
    appearance : MeshFlatAppearance | MeshPhongAppearance
        Appearance parameters.
    section : MeshSectionConfig
        How the mesh is drawn in a 2D view: outline and fill of its
        cross-section.
    requires_camera_reslice : bool
        Always False; frozen.  Camera movement does not trigger reslicing.
    """

    visual_type: Literal["mesh_memory"] = "mesh_memory"

    def draws_nothing(self) -> bool:
        """Whether the mesh is hidden, and so loads nothing.

        A hidden mesh is not planned, and its reads stop.  Shown again at
        the same position it reads nothing: what it held is still there.
        """
        return not self.appearance.visible
