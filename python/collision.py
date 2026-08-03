"""
URDF collision-geometry loading and world-frame assembly.

Complements mesh_utils.py, which samples a link's VISUAL mesh surface for
contact points. This instead loads a link's own <collision> geometry
(typically a simpler mesh, or a primitive box/sphere/cylinder, standing in
for its more detailed <visual> counterpart) and assembles it into a single
world-frame trimesh -- e.g. for testing sphere-vs-palm collision with
trimesh.proximity.signed_distance() against the actual collision surface,
rather than a sampled point cloud.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh

from .finger import FingerSet, _resolve_mesh_path
from .kinematics import _mesh_offset_matrix, get_transform
from .urdf_parser import CollisionGeometry, LinkProperty


def _load_collision_primitive(
    geom: CollisionGeometry,
    mesh_folder: str,
    urdf_base: Optional[str],
) -> Optional[trimesh.Trimesh]:
    """Build a link-local-frame trimesh for one <collision> geometry element."""
    if geom.kind == "mesh":
        if geom.mesh_file is None:
            return None
        resolved = _resolve_mesh_path(geom.mesh_file, mesh_folder, urdf_base)
        if resolved is None:
            return None
        mesh = trimesh.load_mesh(resolved, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh

    if geom.kind == "box" and geom.box_size is not None:
        return trimesh.creation.box(extents=geom.box_size)

    if geom.kind == "sphere" and geom.sphere_radius is not None:
        return trimesh.creation.icosphere(subdivisions=2, radius=geom.sphere_radius)

    if geom.kind == "cylinder" and geom.cylinder_radius is not None:
        return trimesh.creation.cylinder(radius=geom.cylinder_radius, height=geom.cylinder_length)

    return None


def build_collision_mesh(
    body_names: list[str],
    link_props: list[LinkProperty],
    model,
    data,
    mesh_folder: str,
    urdf_base: Optional[str] = None,
) -> Optional[trimesh.Trimesh]:
    """
    Assemble a set of bodies' URDF collision geometry into one world-frame
    trimesh, at whatever pose `data` currently holds (call
    kinematics.compute_fk() first).

    Args:
        body_names:  URDF link names to include (e.g. a hand's palm bodies).
        link_props:  From urdf_parser.parse_mimic_and_offsets().
        model, data: Pinocchio model/data, with FK already computed for the
                    pose these bodies should be assembled at.
        mesh_folder: Directory to prepend when a collision mesh filename is bare.
        urdf_base:   URDF file's directory (resolves package:// URIs and
                    paths written relative to the URDF, which is how
                    collision meshes are usually referenced).

    Returns:
        A single trimesh.Trimesh combining every collision piece found for
        `body_names`, in world frame, or None if none of them have any
        collision geometry (or none of it could be loaded).
    """
    link_prop_by_name = {lp.body: lp for lp in link_props}
    pieces: list[trimesh.Trimesh] = []

    for name in body_names:
        lp = link_prop_by_name.get(name)
        if lp is None or not lp.collision_geometries:
            continue

        T_world_link = get_transform(model, data, name)

        for geom in lp.collision_geometries:
            local = _load_collision_primitive(geom, mesh_folder, urdf_base)
            if local is None:
                continue
            T_offset = _mesh_offset_matrix(geom.rpy, geom.xyz)
            piece = local.copy()
            piece.apply_transform(T_world_link @ T_offset)
            pieces.append(piece)

    if not pieces:
        return None

    combined = trimesh.util.concatenate(pieces)
    # Rebuild fresh: concatenate() should already hand back a clean mesh, but
    # see workspace.py's generate_alphashape() docstring for why a
    # freshly-constructed Trimesh is worth insisting on before this gets
    # queried with proximity/containment tests -- cached spatial-index state
    # on a mesh built up from repeated transforms/concatenation has bitten
    # us before.
    return trimesh.Trimesh(vertices=combined.vertices.copy(), faces=combined.faces.copy(), process=True)


def build_visual_mesh(
    finger_set: FingerSet,
    model,
    data,
    mesh_folder: str,
    urdf_base: Optional[str] = None,
) -> Optional[trimesh.Trimesh]:
    """
    Assemble a FingerSet's URDF <visual> mesh geometry into one world-frame
    trimesh, at whatever pose `data` currently holds (call
    kinematics.compute_fk() first).

    Complements build_collision_mesh(), which does the same for a link's
    (usually simpler) <collision> geometry -- this uses each link's
    <visual> mesh instead, e.g. for rendering an accurate-looking hand model
    rather than a sampled surface point cloud.

    Args:
        finger_set: FingerSet with each link's mesh_file/rot_offset/
                    trans_offset already populated (i.e. from finger.build_hand()).
        model, data: Pinocchio model/data, with FK already computed for the
                    pose this mesh should be assembled at.
        mesh_folder: Directory to prepend when a mesh filename is bare.
        urdf_base:   URDF file's directory (resolves package:// URIs and
                    paths written relative to the URDF).

    Returns:
        A single trimesh.Trimesh combining every link's visual mesh, in
        world frame, or None if the finger is empty or none of its links
        have a visual mesh.
    """
    if finger_set.is_empty():
        return None

    pieces: list[trimesh.Trimesh] = []
    for lnk in finger_set.links:
        if lnk.mesh_file is None:
            continue
        resolved = _resolve_mesh_path(lnk.mesh_file, mesh_folder, urdf_base)
        if resolved is None:
            continue

        mesh = trimesh.load_mesh(resolved, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        T_world_link = get_transform(model, data, lnk.body_name)
        T_offset = _mesh_offset_matrix(lnk.rot_offset, lnk.trans_offset)
        piece = mesh.copy()
        piece.apply_transform(T_world_link @ T_offset)
        pieces.append(piece)

    if not pieces:
        return None

    combined = trimesh.util.concatenate(pieces)
    # Rebuild fresh -- see build_collision_mesh()'s identical comment above.
    return trimesh.Trimesh(vertices=combined.vertices.copy(), faces=combined.faces.copy(), process=True)
