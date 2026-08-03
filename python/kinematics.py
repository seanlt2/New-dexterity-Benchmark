"""
Forward kinematics and Jacobian helpers using Pinocchio.
Replaces MATLAB's Points_Transform.m, Points_and_Surf_Norm_Transform.m,
Surf_Norm_Transform.m, and True_Jacobian.m.

Convention note
---------------
MATLAB's rigidtform3d(rad2deg(rpy), xyz) treats the 3-element rpy vector as
ZYX Euler angles [yaw=rpy[0], pitch=rpy[1], roll=rpy[2]] in degrees.
This module replicates that exact convention so results match.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


try:
    import pinocchio as pin
    _PIN_AVAILABLE = True
except ImportError:
    _PIN_AVAILABLE = False
    print("Warning: pinocchio not found. Kinematics functions will not work.")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(urdf_path: str) -> tuple:
    """
    Load a URDF into a pinocchio model.

    Returns:
        (model, data): pinocchio.Model and pinocchio.Data
    """
    if not _PIN_AVAILABLE:
        raise ImportError("pinocchio is required for kinematics.")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def get_non_fixed_joint_names(model) -> list[str]:
    """
    Return ordered list of all non-fixed joint names (pinocchio joint order).
    Equivalent to MATLAB's non_fix_joint_names.
    """
    # model.names[0] is 'universe'; skip it
    return list(model.names[1:])


def get_joint_limits(model) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lower, upper) position limit arrays aligned with get_non_fixed_joint_names().
    """
    return np.array(model.lowerPositionLimit), np.array(model.upperPositionLimit)


# ---------------------------------------------------------------------------
# Mesh-offset transform (replicates rigidtform3d behaviour)
# ---------------------------------------------------------------------------

def _mesh_offset_matrix(rot_rpy_rad: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """
    Build the 4x4 mesh-offset transform from a URDF <visual>/<collision>
    <origin rpy="..." xyz="..."/>.

    URDF's rpy is standard roll-pitch-yaw about FIXED (extrinsic) x, y, z
    axes in that order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), i.e. scipy's
    lowercase, extrinsic 'xyz' sequence with radians (scipy's default).
    Verified against Pinocchio's own URDF visual-geometry parser
    (pin.buildGeomFromUrdf) on this project's URDFs: matches exactly.

    An earlier version of this function used scipy's uppercase 'ZYX'
    sequence (a different, INTRINSIC rotation order that also swaps which
    rpy component maps to which axis) with the intent of replicating
    MATLAB's rigidtform3d(rad2deg(rpy), trans) -- but MATLAB's own URDF
    reader (Mimic_and_Offset_Extraction.m) passes rpy through unmodified
    from the URDF's roll-pitch-yaw attribute, so whatever rigidtform3d's
    argument convention is, the correct end-to-end rotation is the standard
    URDF one used here, not 'ZYX'. That version also fed rad2deg(rpy) to
    from_euler without degrees=True, so the degree-valued numbers were
    silently reinterpreted as radians -- both bugs together produced a wrong
    rotation whenever a link had a nonzero visual rpy offset, most visibly
    on links with a large offset (e.g. this project's Ability Hand thumb,
    where it showed up as a ~4cm gap between the two thumb link meshes).
    """
    R = Rotation.from_euler("xyz", rot_rpy_rad).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T


# ---------------------------------------------------------------------------
# FK utilities
# ---------------------------------------------------------------------------

def compute_fk(model, data, q: np.ndarray) -> None:
    """
    Run forward kinematics and update all frame placements in-place.

    Args:
        model: pinocchio.Model
        data:  pinocchio.Data  (modified in-place)
        q:     Configuration vector (length model.nq), full joint space.
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)


def get_transform(model, data, link_name: str) -> np.ndarray:
    """
    Return the 4×4 world-to-link transform for a named link/frame.
    compute_fk() must be called first.

    Replicates MATLAB's getTransform(Robot, Config, link_name).
    """
    fid = model.getFrameId(link_name)
    return data.oMf[fid].homogeneous


def get_transform_by_id(model, data, frame_id: int) -> np.ndarray:
    """
    Same as get_transform(), but takes an already-resolved pinocchio frame id
    instead of a link name.

    model.getFrameId() does a name lookup every call; in a hot loop that
    calls this once per (configuration, link) pair -- e.g.
    workspace.finger_workspace()'s joint-space sweep -- that lookup is pure
    overhead, since a link's frame id is fixed and can be resolved once
    before the loop starts.
    """
    return data.oMf[frame_id].homogeneous


# ---------------------------------------------------------------------------
# Point/normal transforms
# ---------------------------------------------------------------------------

def points_transform(
    model,
    data,
    link_name: str,
    rot_offset: np.ndarray,
    trans_offset: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Transform mesh points from body frame to world frame.

    Replicates MATLAB's Points_Transform().

    Args:
        model, data:   Pinocchio model/data with FK already computed.
        link_name:     Name of the URDF link/frame.
        rot_offset:    RPY mesh offset in radians (from URDF visual origin).
        trans_offset:  XYZ mesh offset in meters (from URDF visual origin).
        points:        (N, 3) array of points in mesh/body frame.

    Returns:
        (N, 3) array of points in world frame.
    """
    T_world_link = get_transform(model, data, link_name)          # 4×4
    T_link_mesh = _mesh_offset_matrix(rot_offset, trans_offset)   # 4×4
    T = T_world_link @ T_link_mesh

    N = len(points)
    pts_h = np.hstack([points, np.ones((N, 1))]).T   # 4×N
    world_h = T @ pts_h                               # 4×N
    return (world_h[:3] / world_h[3]).T               # N×3


def points_transform_precomputed(
    model,
    data,
    frame_id: int,
    T_link_mesh: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Same as points_transform(), but takes an already-resolved frame id and a
    precomputed mesh-offset matrix instead of a link name + rot/trans offset.

    Both of those are static per link (independent of the configuration q),
    so a caller that transforms the same link's points at many
    configurations -- e.g. workspace.finger_workspace()'s joint-space sweep
    -- can resolve the frame id and build T_link_mesh once before the sweep
    and reuse them here on every iteration, instead of this function (via
    points_transform()) redoing a scipy Rotation.from_euler() construction
    and a model.getFrameId() name lookup on every single call. Measured on a
    real 320,000-pose sweep: rebuilding T_link_mesh from scratch every call
    accounted for 60% of the sweep's total runtime.

    Args:
        model, data:  Pinocchio model/data with FK already computed.
        frame_id:     Pinocchio frame id for the link (e.g. from
                      model.getFrameId(link_name), resolved once).
        T_link_mesh:  4×4 mesh-offset transform (e.g. from
                      kinematics._mesh_offset_matrix(rot_offset, trans_offset),
                      built once).
        points:       (N, 3) array of points in mesh/body frame.

    Returns:
        (N, 3) array of points in world frame -- identical to what
        points_transform() would return for the same link/offsets/points.
    """
    T_world_link = get_transform_by_id(model, data, frame_id)
    T = T_world_link @ T_link_mesh

    N = len(points)
    pts_h = np.hstack([points, np.ones((N, 1))]).T
    world_h = T @ pts_h
    return (world_h[:3] / world_h[3]).T


def points_and_normals_transform(
    model,
    data,
    link_name: str,
    rot_offset: np.ndarray,
    trans_offset: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform mesh points AND surface normals from body frame to world frame.

    Replicates MATLAB's Points_and_Surf_Norm_Transform().

    Normals are transformed by the rotation component only (w=0 homogeneous).
    """
    T_world_link = get_transform(model, data, link_name)
    T_link_mesh = _mesh_offset_matrix(rot_offset, trans_offset)
    T = T_world_link @ T_link_mesh

    N = len(points)
    pts_h = np.hstack([points, np.ones((N, 1))]).T
    world_pts_h = T @ pts_h
    world_pts = (world_pts_h[:3] / world_pts_h[3]).T

    nrm_h = np.hstack([normals, np.zeros((N, 1))]).T
    world_nrm_h = T @ nrm_h
    # For direction vectors the homogeneous w should stay 0; divide by it only
    # if it's non-zero (degenerate case).
    w = world_nrm_h[3]
    world_nrm = np.where(
        np.abs(w) > 1e-12,
        world_nrm_h[:3] / w,
        world_nrm_h[:3],
    ).T

    return world_pts, world_nrm


def normals_transform(
    model,
    data,
    link_name: str,
    rot_offset: np.ndarray,
    trans_offset: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """
    Transform surface normals from body frame to world frame (rotation only).

    Replicates MATLAB's Surf_Norm_Transform().
    """
    T_world_link = get_transform(model, data, link_name)
    T_link_mesh = _mesh_offset_matrix(rot_offset, trans_offset)
    T = T_world_link @ T_link_mesh

    N = len(normals)
    nrm_h = np.hstack([normals, np.zeros((N, 1))]).T
    world_nrm_h = T @ nrm_h
    w = world_nrm_h[3]
    return np.where(np.abs(w) > 1e-12, world_nrm_h[:3] / w, world_nrm_h[:3]).T


# ---------------------------------------------------------------------------
# Jacobian helpers
# ---------------------------------------------------------------------------

def compute_geometric_jacobian(
    model,
    data,
    q: np.ndarray,
    link_name: str,
) -> np.ndarray:
    """
    Compute the 6×nq geometric Jacobian for a link, expressed in world frame.

    Rows 0-2 : angular velocity components
    Rows 3-5 : linear velocity components

    Replicates MATLAB's geometricJacobian(Robot, Config, link_name), which
    returns [angular; linear] rows in that order. Pinocchio's own convention
    is the opposite ([linear; angular] -- verified against pin.Motion, whose
    constructor is Motion(linear, angular) and whose .vector is
    [linear, angular] in that order), so the two blocks are swapped here to
    match the row order this function's docstring -- and every caller that
    slices rows 3:6 expecting linear velocity (e.g. force_ellipsoid()) --
    was written against.
    """
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId(link_name)
    J = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return np.vstack([J[3:6], J[0:3]])   # pin native [linear;angular] -> [angular;linear]


def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix for cross product with v."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])


def contact_point_jacobian(
    model,
    data,
    q: np.ndarray,
    link_name: str,
    contact_point_world: np.ndarray,
) -> np.ndarray:
    """
    Geometric Jacobian of a point rigidly attached to `link_name`, evaluated
    at an arbitrary world-frame position (e.g. a contact point on the link's
    mesh surface), rather than at the link's own frame origin.

    MATLAB's Manipulation_capacity_region.m gets this by temporarily adding
    a fixed child body at the contact point to the rigidBodyTree, calling
    geometricJacobian(), then removing it. Pinocchio doesn't need a body in
    the model to differentiate an arbitrary rigidly-attached point: for a
    rigid body, velocity at point p equals velocity at the link origin o
    plus omega x (p - o), so the linear rows of the link's own Jacobian are
    shifted by that same relation; the angular rows are unchanged (a rigid
    body has one angular velocity everywhere).

    Args:
        model, data:          Pinocchio model/data with FK already valid for q.
        q:                    Configuration vector (length model.nq).
        link_name:            Name of the URDF link/frame the contact point
                              is rigidly attached to.
        contact_point_world:  (3,) contact point position in world frame.

    Returns:
        (6, nq) Jacobian at the contact point (rows 0-2 angular, 3-5 linear),
        in the same LOCAL_WORLD_ALIGNED convention as compute_geometric_jacobian().
    """
    J = compute_geometric_jacobian(model, data, q, link_name)
    J_ang = J[:3]
    J_lin = J[3:]

    origin_world = get_transform(model, data, link_name)[:3, 3]
    r = contact_point_world - origin_world
    J_lin_contact = J_lin - _skew(r) @ J_ang
    return np.vstack([J_ang, J_lin_contact])


def true_jacobian(
    J_full: np.ndarray,
    coupling_matrix: np.ndarray,
    actuated_joint_ids: list[int],
) -> np.ndarray:
    """
    Project the full geometric Jacobian to the actuated joint subspace using
    the coupling matrix.

    Replicates MATLAB's True_Jacobian().

    Args:
        J_full:            (6, n_full) geometric Jacobian in full joint space.
        coupling_matrix:   (n_full, n_actuated+1) coupling matrix from
                           CouplingInfo.matrix.
        actuated_joint_ids: 0-indexed list of actuated joint positions.

    Returns:
        (6, n_active_actuated) Jacobian projected onto actuated joints that
        actually influence this link.
    """
    n_full = coupling_matrix.shape[0]
    n_actuated = coupling_matrix.shape[1] - 1

    # Indicator vector for the joints belonging to this link/subtree
    actuated_index = np.zeros(n_full)
    actuated_index[actuated_joint_ids] = 1.0

    # Which actuated DOFs actually influence these joints?
    C = coupling_matrix[:, :n_actuated]
    actuated_relevance = C.T @ actuated_index
    active_cols = np.where(actuated_relevance)[0]

    J_projected = J_full @ C          # (6, n_actuated)
    return J_projected[:, active_cols]  # (6, n_active)
