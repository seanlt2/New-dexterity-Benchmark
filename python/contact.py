"""
Contact point detection and object reference frame.
Replaces MATLAB's Closest_2_Points.m and Object_Reference_Frame.m.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .finger import FingerLink, FingerSet
from .kinematics import (
    _mesh_offset_matrix,
    compute_fk,
    get_transform,
    points_transform,
)


# ---------------------------------------------------------------------------
# Closest point detection
# ---------------------------------------------------------------------------

def closest_two_points(
    model,
    data,
    q: np.ndarray,
    finger: FingerSet,
    body_indices: list[int],
    extrinsic_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Find the finger surface point closest to an extrinsic object.

    Replicates MATLAB's Closest_2_Points().

    Args:
        model, data:    Pinocchio model and data (FK must be current for q).
        q:              Current full joint configuration.
        finger:         FingerSet for one finger.
        body_indices:   0-indexed list of link indices within finger.links to
                        consider (e.g. tip links only, or all links).
        extrinsic_pts:  (M, 3) surface points of the external object in world
                        frame.

    Returns:
        T_world_contact: (4, 4) contact frame in world frame.
            - z-axis  = outward surface normal at the contact point.
            - x-axis  = tangent (constructed to be perpendicular to z).
            - origin  = contact point position.
        contact_in_body: (3,) contact point in the link body frame.
        closest_body_idx: Index into finger.links of the link containing the
                          closest point.
    """
    compute_fk(model, data, q)

    all_pts: list[np.ndarray] = []
    all_nrms: list[np.ndarray] = []
    all_body_ids: list[int] = []

    for i in body_indices:
        lnk = finger.links[i]
        if lnk.points is None or lnk.points.shape[1] != 3:
            continue
        world_pts = points_transform(
            model, data,
            lnk.body_name,
            lnk.rot_offset,
            lnk.trans_offset,
            lnk.points,
        )
        all_pts.append(world_pts)
        all_nrms.append(lnk.surf_norm)
        all_body_ids.extend([i] * len(world_pts))

    if not all_pts:
        raise ValueError("No valid surface points found for the given body indices.")

    pts = np.vstack(all_pts)
    nrms = np.vstack(all_nrms)
    body_ids = np.array(all_body_ids)

    # Closest point in pts to any point in extrinsic_pts
    D = cdist(pts, extrinsic_pts)
    row_a, _col_b = np.unravel_index(np.argmin(D), D.shape)

    contact_pt_world = pts[row_a]
    closest_body_idx = int(body_ids[row_a])
    closest_lnk: FingerLink = finger.links[closest_body_idx]

    # Build contact frame: z = outward normal, x/y = tangents
    T_world_body = get_transform(model, data, closest_lnk.body_name)
    T_link_mesh = _mesh_offset_matrix(closest_lnk.rot_offset, closest_lnk.trans_offset)
    T_world_mesh = T_world_body @ T_link_mesh
    R_world_mesh = T_world_mesh[:3, :3]

    normal_world = R_world_mesh @ nrms[row_a]
    normal_world /= np.linalg.norm(normal_world)

    # Build orthonormal frame around normal (same heuristic as MATLAB)
    if abs(np.dot(normal_world, [0, 0, 1])) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])

    x_vec = np.cross(ref, normal_world)
    x_vec /= np.linalg.norm(x_vec)
    y_vec = np.cross(normal_world, x_vec)

    R_contact = np.column_stack([x_vec, y_vec, normal_world])  # 3×3
    T_world_contact = np.eye(4)
    T_world_contact[:3, :3] = R_contact
    T_world_contact[:3, 3] = contact_pt_world

    # Contact point in body frame (for optimization use)
    R_wb = T_world_mesh[:3, :3]
    p_wb = T_world_mesh[:3, 3]
    T_body_world = np.eye(4)
    T_body_world[:3, :3] = R_wb.T
    T_body_world[:3, 3] = -R_wb.T @ p_wb
    contact_in_body_h = T_body_world @ np.append(contact_pt_world, 1.0)
    contact_in_body = contact_in_body_h[:3]

    return T_world_contact, contact_in_body, closest_body_idx


# ---------------------------------------------------------------------------
# Object reference frame
# ---------------------------------------------------------------------------

def object_reference_frame(
    contact_pts_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the object reference frame from a set of contact points.

    The object frame is centred at the centroid of the contact points and
    aligned with the world frame (identity rotation).

    Replicates MATLAB's Object_Reference_Frame().

    Args:
        contact_pts_world: (N, 3) contact point positions in world frame.

    Returns:
        object_center:  (3,) centroid position.
        R_world_object: (3, 3) rotation (identity — aligned with world).
        T_world_object: (4, 4) homogeneous transform.
    """
    object_center = contact_pts_world.mean(axis=0)
    R_world_object = np.eye(3)
    T_world_object = np.eye(4)
    T_world_object[:3, 3] = object_center
    return object_center, R_world_object, T_world_object
