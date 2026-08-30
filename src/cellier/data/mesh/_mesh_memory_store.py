# src/cellier/v2/data/mesh/_mesh_memory_store.py
from __future__ import annotations

import asyncio
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator, model_validator

from cellier.data._base_data_store import BaseDataStore
from cellier.data.mesh._mesh_requests import MeshData, MeshSliceRequest

# Single degenerate triangle used when the slab contains no surviving faces.
_PLACEHOLDER_INDICES = np.array([[0, 1, 2]], dtype=np.int32)


def _compute_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area-weighted per-vertex normals from triangle soup.

    Parameters
    ----------
    positions : np.ndarray
        (n_vertices, 3) float32 vertex positions.
    indices : np.ndarray
        (n_faces, 3) int32 triangle face indices.

    Returns
    -------
    np.ndarray
        (n_vertices, 3) float32 unit normals.  Degenerate vertices
        (zero accumulated normal) get [0, 0, 1].
    """
    e1 = positions[indices[:, 1]] - positions[indices[:, 0]]
    e2 = positions[indices[:, 2]] - positions[indices[:, 0]]
    face_normals = np.cross(e1, e2)  # area-weighted, (n_faces, 3)

    vertex_normals = np.zeros_like(positions)
    np.add.at(vertex_normals, indices[:, 0], face_normals)
    np.add.at(vertex_normals, indices[:, 1], face_normals)
    np.add.at(vertex_normals, indices[:, 2], face_normals)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    safe = norms > 0
    vertex_normals = np.where(
        safe, vertex_normals / np.where(safe, norms, 1.0), [0.0, 0.0, 1.0]
    )
    return vertex_normals.astype(np.float32)


class MeshMemoryStore(BaseDataStore):
    """In-memory triangle mesh data store.

    Parameters
    ----------
    positions : np.ndarray
        (n_vertices, N) float32 vertex positions for any world
        dimensionality N ≥ 2.  For a 3-D scene use shape (n_vertices, 3);
        for a 5-D scene (t, z, y, x, c) use shape (n_vertices, 5).
    indices : np.ndarray
        (n_faces, 3) int32 triangle face indices.
        **Must be int32** — pygfx rejects int64 at upload time.
        int64 input is coerced silently by the validator.
    colors : np.ndarray | None
        Per-vertex (n_vertices, 4) or per-face (n_faces, 4) float32 RGBA.
        Which of the two is declared by ``colors_layout``, never inferred.
    colors_layout : str | None
        ``"vertex"`` or ``"face"``.  **Required whenever ``colors`` is
        set**, and rejected when it is not.

        This used to be inferred by comparing ``colors.shape[0]`` against
        ``n_faces``, which is ambiguous whenever a mesh has as many
        vertices as faces -- a tetrahedron has four of each, so its
        per-vertex colours were reported as per-face and gathered wrongly.
        The layout also decides which rows ``get_data`` gathers, so it is
        not a rendering preference the appearance can supply: it is a fact
        about the array that only the caller knows.
    name : str
        Human-readable label.
    """

    store_type: Literal["mesh_memory"] = "mesh_memory"
    name: str = "mesh_memory_store"
    positions: np.ndarray
    indices: np.ndarray
    colors: np.ndarray | None = None
    colors_layout: Literal["vertex", "face"] | None = None

    # validate_assignment so `store.colors = ...` re-runs the layout check.
    # Without it the invariant only holds at construction, and assigning
    # colours later leaves colors_layout None -- which `colors_mode` then
    # returns, and `get_data`'s `== "face"` test silently reads as vertex.
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("positions", mode="before")
    @classmethod
    def _coerce_positions(cls, v: Any) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @field_validator("indices", mode="before")
    @classmethod
    def _coerce_indices(cls, v: Any) -> np.ndarray:
        """Coerce to int32 — pygfx rejects int64 index buffers."""
        return np.ascontiguousarray(np.asarray(v, dtype=np.int32))

    @field_validator("colors", mode="before")
    @classmethod
    def _coerce_colors(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @model_validator(mode="after")
    def _check_colors_layout(self) -> MeshMemoryStore:
        """Require an explicit layout for colours, and sanity-check it.

        The length check catches a transposed or wrongly declared array, but
        cannot adjudicate a mesh with as many vertices as faces -- both
        counts are valid there, which is exactly the case the old inference
        got wrong.  In that case the declaration is simply authoritative,
        which is the point of having one.
        """
        if self.colors is None:
            if self.colors_layout is not None:
                raise ValueError(
                    "colors_layout was given without colors. Drop it, or "
                    "pass the colors it describes."
                )
            return self
        if self.colors_layout is None:
            raise ValueError(
                "colors requires an explicit colors_layout of 'vertex' or "
                "'face'. It is not inferred from the array length: a mesh "
                "with as many vertices as faces (a tetrahedron, say) is "
                "ambiguous, and the layout decides which rows are gathered "
                "when slicing."
            )
        expected = self.n_vertices if self.colors_layout == "vertex" else self.n_faces
        other = self.n_faces if self.colors_layout == "vertex" else self.n_vertices
        got = self.colors.shape[0]
        if got != expected and got != other:
            raise ValueError(
                f"colors_layout='{self.colors_layout}' expects {expected} "
                f"rows, but colors has {got}."
            )
        return self

    # ------------------------------------------------------------------
    # Serializers
    # ------------------------------------------------------------------

    @field_serializer("positions")
    def _ser_positions(self, v: np.ndarray, _info: Any) -> list:
        return v.tolist()

    @field_serializer("indices")
    def _ser_indices(self, v: np.ndarray, _info: Any) -> list:
        return v.tolist()

    @field_serializer("colors")
    def _ser_colors(self, v: np.ndarray | None, _info: Any) -> list | None:
        return v.tolist() if v is not None else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return self.positions.shape[0]

    @property
    def n_faces(self) -> int:
        return self.indices.shape[0]

    @property
    def colors_mode(self) -> str:
        """``'vertex'``, ``'face'``, or ``'none'`` -- the declared layout.

        Reads ``colors_layout`` rather than comparing array lengths.  The
        old inference reported per-vertex colours as per-face on any mesh
        with as many vertices as faces.
        """
        if self.colors is None:
            return "none"
        return self.colors_layout

    # ------------------------------------------------------------------
    # Async data access — three checkpoints for cancellability
    # ------------------------------------------------------------------

    async def get_data(self, request: MeshSliceRequest) -> MeshData:
        """Return slab-filtered, reindexed mesh data for *request*.

        Checkpoints
        -----------
        A  After Phase 1 (slab mask built, no reindexing yet).
           Fires if slider moved before reindexing begins.
        B  After Phase 2 (reindex complete, before projection).
           Fires if slider moved before projection.

        If CancelledError fires at either checkpoint the callback is
        never called, preventing stale geometry from reaching the GPU.

        Inclusion rule
        -------------
        A face survives only when **all** of its vertices satisfy
        ``slice_index - thickness <= coord <= slice_index + thickness``
        on every sliced axis.  This is the mesh analogue of the lines
        store's "both endpoints must pass" rule and avoids projecting
        off-slice vertices onto the slice plane with the wrong colors.

        Parameters
        ----------
        request : MeshSliceRequest
            Built by GFXMeshMemoryVisual.build_slice_request[_2d].

        Returns
        -------
        MeshData
            Filtered, reindexed, projected mesh ready for GPU upload.
            Normals are computed from the projected geometry.
            ``is_empty=True`` when the slab contained no faces.
        """
        positions = self.positions  # (n_vertices, N)
        indices = self.indices  # (n_faces, 3)
        colors = self.colors
        n_vertices = positions.shape[0]
        displayed = list(request.displayed_axes)
        n_display = len(displayed)

        # ── Phase 1: build slab mask ─────────────────────────────────
        face_mask = np.ones(self.n_faces, dtype=bool)

        for axis, idx in request.slice_indices.items():
            lo = float(idx) - request.thickness
            hi = float(idx) + request.thickness
            # Include face only if ALL vertices are in the slab on this axis.
            vertex_in = (positions[:, axis] >= lo) & (positions[:, axis] <= hi)
            face_mask &= vertex_in[indices].all(axis=1)

        # ── Checkpoint A ─────────────────────────────────────────────
        await asyncio.sleep(0)

        # ── Phase 2: reindex surviving faces ─────────────────────────
        surviving = indices[face_mask]  # (n_surv, 3)
        surviving_faces = np.where(face_mask)[0]  # original face index per row

        if surviving.shape[0] == 0:
            # Empty slab — return placeholder so the node stays valid.
            ph_pos = np.zeros((3, n_display), dtype=np.float32)
            ph_nor = np.zeros((3, n_display), dtype=np.float32)
            return MeshData(
                request_id=request.slice_request_id,
                positions=ph_pos,
                indices=_PLACEHOLDER_INDICES,
                normals=ph_nor,
                colors=None,
                color_mode="vertex",
                is_empty=True,
            )

        unique_old = np.unique(surviving.ravel())
        remap = np.full(n_vertices, -1, dtype=np.int32)
        remap[unique_old] = np.arange(len(unique_old), dtype=np.int32)

        new_positions = positions[unique_old]  # (n_surv_v, N)
        new_indices = remap[surviving]  # (n_surv_f, 3)

        if colors is not None:
            if self.colors_mode == "face":
                new_colors = colors[face_mask]  # (n_surv_f, 4)
                color_mode = "face"
            else:
                new_colors = colors[unique_old]  # (n_surv_v, 4)
                color_mode = "vertex"
        else:
            new_colors = None
            color_mode = "vertex"

        # ── Checkpoint B ─────────────────────────────────────────────
        await asyncio.sleep(0)

        # ── Phase 3: project onto displayed axes ─────────────────────
        proj_positions = new_positions[:, displayed]  # (n_surv_v, n_display)

        # Normals are computed from the projected geometry so they are
        # always valid in the display space regardless of world dimension.
        # In 2D the material is unlit so normals are unused; emit zeros.
        if n_display == 3:
            proj_normals = _compute_vertex_normals(proj_positions, new_indices)
        else:
            proj_normals = np.zeros_like(proj_positions)

        return MeshData(
            request_id=request.slice_request_id,
            positions=proj_positions,
            indices=new_indices,
            normals=proj_normals,
            colors=new_colors,
            color_mode=color_mode,
            is_empty=False,
            original_face_indices=surviving_faces,
        )
