"""
Per-finger workspace computation and alpha-shape volume.
Replaces MATLAB's Finger_Workspace_Efficient_2.m and Alphashape_generation.m.

The workspace is swept on a joint-space grid, applying the coupling matrix at
each configuration to get the full q, then FK-transforming each link's surface
point cloud into world frame. This module computes and persists the reachable
workspace volume of ONE finger at a time, from its selected surface points;
it does not compute multi-finger intersections (see opposability.py for the
sphere-based multi-finger grasp volume that replaces that workflow).

Step-halving scheme (matching MATLAB):
  actuated joint 1 → N_pts samples
  actuated joint 2 → N_pts samples
  actuated joint 3 → N_pts // 2 samples  (every 2nd index)
  actuated joint k → N_pts // 2^(k-2) samples  (every 2^(k-2)-th index)
"""

from __future__ import annotations

import itertools
import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .coupling import CouplingInfo
from .finger import FingerSet
from .kinematics import (
    _mesh_offset_matrix,
    compute_fk,
    get_transform_by_id,
    points_transform_precomputed,
)


# ---------------------------------------------------------------------------
# Workspace computation
# ---------------------------------------------------------------------------

def finger_workspace(
    finger: FingerSet,
    actuated_joint_space: np.ndarray,
    coupling: CouplingInfo,
    model,
    data,
    q_home: np.ndarray,
) -> np.ndarray:
    """
    Compute the reachable workspace point cloud for one finger.

    Replicates MATLAB's Finger_Workspace_Efficient_2().

    Args:
        finger:               FingerSet for one finger.
        actuated_joint_space: (n_actuated_total, N_pts) array.  Row i holds
                              N_pts evenly-spaced samples for actuated joint i
                              (all hand actuated joints, not just this finger's).
        coupling:             CouplingInfo from build_coupling_matrix().
        model, data:          Pinocchio model and data.
        q_home:               Home/neutral pinocchio configuration (length
                              model.nq).  Used as the base; only the finger's
                              joints are swept.

    Returns:
        (M, 3) float32 array of workspace points in world frame.
    """
    if finger.is_empty():
        return np.empty((0, 3), dtype=np.float32)

    # Unique full-q indices for joints that actuate this finger
    finger_act_full_ids = sorted(set(finger.actuated_joint_ids_flat))
    if not finger_act_full_ids:
        return np.empty((0, 3), dtype=np.float32)

    # Map full-q index → row in actuated_joint_space
    full_id_to_row = {fid: i for i, fid in enumerate(coupling.actuated_joint_ids)}
    finger_rows = [full_id_to_row[fid] for fid in finger_act_full_ids
                   if fid in full_id_to_row]
    if not finger_rows:
        return np.empty((0, 3), dtype=np.float32)

    N_pts = actuated_joint_space.shape[1]
    n_joints = len(finger_rows)

    # Step sizes matching MATLAB's halving scheme
    # joint index k (0-based): step = 1 for k<2, else 2^(k-1)
    pt_ranges = []
    for k in range(n_joints):
        step = 1 if k < 2 else 2 ** (k - 1)
        pt_ranges.append(list(range(0, N_pts, step)))

    links_with_pts = [
        lnk for lnk in finger.links
        if lnk.points is not None and len(lnk.points) > 0
    ]
    if not links_with_pts:
        return np.empty((0, 3), dtype=np.float32)

    # Each link's frame id and mesh-offset transform are fixed -- independent
    # of q -- so resolve/build them once here rather than inside the sweep
    # loop below, which calls this once per (configuration, link) pair.
    # Rebuilding the mesh-offset matrix from scratch every call (via scipy's
    # Rotation.from_euler) was measured to account for 60% of this sweep's
    # total runtime; see points_transform_precomputed()'s docstring.
    frame_ids = {lnk.body_name: model.getFrameId(lnk.body_name) for lnk in links_with_pts}
    mesh_offsets = {
        lnk.body_name: _mesh_offset_matrix(lnk.rot_offset, lnk.trans_offset)
        for lnk in links_with_pts
    }

    pts_per_config = sum(len(lnk.points) for lnk in links_with_pts)
    max_configs = 1
    for r in pt_ranges:
        max_configs *= len(r)

    workspace = np.zeros((pts_per_config * max_configs, 3), dtype=np.float32)
    cursor = 0

    total = max_configs
    count = 0
    report_every = max(1, total // 20)

    for combo in itertools.product(*pt_ranges):
        # Build the full actuated_q vector (zeros for non-finger joints)
        actuated_q = np.zeros(coupling.n_actuated)
        for k, row in enumerate(finger_rows):
            actuated_q[row] = actuated_joint_space[row, combo[k]]

        # Full joint vector via coupling matrix: C @ [q_act; 1]
        full_q = coupling.matrix @ np.append(actuated_q, 1.0)

        # Pinocchio q: copy home config, overwrite non-fixed joint positions
        q = q_home.copy()
        q[:len(full_q)] = full_q

        compute_fk(model, data, q)

        for lnk in links_with_pts:
            pts = points_transform_precomputed(
                model, data,
                frame_ids[lnk.body_name],
                mesh_offsets[lnk.body_name],
                lnk.points,
            )
            n_new = len(pts)
            workspace[cursor:cursor + n_new] = pts
            cursor += n_new

        count += 1
        if count % report_every == 0:
            print(f"  Workspace {100 * count / total:.1f}% complete")

    return workspace[:cursor]


# ---------------------------------------------------------------------------
# Alpha-shape workspace volume
# ---------------------------------------------------------------------------

def _voxelize_points(
    points: np.ndarray,
    voxel_size: float,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """
    Round points onto a voxel_size grid and deduplicate, processing in
    chunks so this never needs a full-size float64 copy of the whole cloud
    at once.

    A dense joint-space sweep's raw point cloud can be hundreds of millions
    of points (multiple GB even as float32). The direct one-liner this
    replaces -- `np.round(points / voxel_size) * voxel_size` -- divides by
    a Python float, silently upcasting the WHOLE array to float64, and each
    of round/multiply allocates its own fresh full-size array on top of
    that; measured on a real hand with several large actuated fingers, this
    was enough to OOM-kill the process outright (~35 GB already resident
    from other fingers' raw clouds, then a single finger's voxelization
    needing up to ~3x that finger's float64 size again on top).

    Deduplicating per-chunk (as integer voxel indices, not floats) and
    merging keeps peak memory bounded by chunk_size plus however many
    *unique* voxels have been found so far -- which for a 1mm grid over a
    finger-sized workspace is typically a small fraction of the raw sample
    count -- rather than by the raw cloud's total size. Mathematically
    identical to deduplicating the whole array at once: the union of each
    chunk's uniques, uniqued again, equals the unique set of the whole.

    Returns:
        (M, 3) float64 array of unique voxel-center coordinates.
    """
    accumulated: Optional[np.ndarray] = None
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        idx = np.round(chunk.astype(np.float64) / voxel_size).astype(np.int64)
        idx = np.unique(idx, axis=0)
        accumulated = idx if accumulated is None else np.unique(
            np.vstack([accumulated, idx]), axis=0
        )

    if accumulated is None:
        return np.empty((0, 3), dtype=np.float64)
    return accumulated.astype(np.float64) * voxel_size


def generate_alphashape(workspace_points: np.ndarray, max_fit_points: int = 2000):
    """
    Generate an alpha shape that tightly bounds the workspace point cloud.

    Replicates MATLAB's Alphashape_generation() (the "critical alpha for a
    single connected region" it computes via alphaShape/criticalAlpha).

    Voxelises the cloud at 1 mm resolution (via the chunked/memory-bounded
    _voxelize_points(), see its docstring), then searches for the critical
    alpha ourselves rather than via alphashape.optimizealpha() -- see
    _fit_alphashape()'s docstring for why. That search is O(a few) fits of a
    Delaunay-based algorithm that doesn't scale well, so for large clouds it
    runs on a random subsample (a few thousand points already characterize
    a shape's boundary about as well as hundreds of thousands do); the
    voxelized (but not subsampled) points are what's actually returned to
    the caller as the workspace point cloud elsewhere in this module.

    Args:
        workspace_points: (N, 3) point cloud (metres).
        max_fit_points:   Cap on how many (voxelized) points the alpha-shape
                          search itself runs on.

    Returns:
        trimesh.Trimesh alpha shape, or None if the cloud is too small, or
        genuinely too flat/degenerate (e.g. a single-DOF finger's sweep,
        which is a thin 2-D ribbon in 3-D space) for any alpha to give a
        valid enclosed volume.
    """
    voxel_size = 0.001  # 1 mm
    voxelized = _voxelize_points(workspace_points, voxel_size)

    if len(voxelized) < 4:
        return None

    if len(voxelized) > max_fit_points:
        rng = np.random.default_rng(0)
        fit_points = voxelized[rng.choice(len(voxelized), max_fit_points, replace=False)]
    else:
        fit_points = voxelized

    return _fit_alphashape(fit_points)


def sample_workspace_grid(
    shape,
    resolution: float = 0.001,
    chunk_size: int = 20_000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Resample an alpha-shape volume onto a uniform 3-D grid.

    Builds a regular meshgrid over the shape's bounding box at `resolution`
    and keeps only the grid points that lie inside the shape. The raw
    joint-space-swept point cloud from finger_workspace() has wildly
    uneven density -- points bunch up near some configurations and thin
    out elsewhere, purely as an artifact of the sweep's kinematics, not of
    the finger's actual reachable geometry -- so anything measuring local
    point density downstream (e.g. opposability.graspable_volume()'s
    default candidate-grid spacing) is more meaningful against an evenly
    sampled cloud than the raw one.

    Replicates the meshgrid + inShape/point-in-alphashape step from the
    original MATLAB pipeline.

    `shape.contains()` is a ray-casting test whose cost (and, for some
    shapes, correctness -- see generate_alphashape()'s docstring) can
    degrade badly when handed a huge point batch at once; this queries it
    in bounded-size chunks instead, which also gives progress output for
    what can otherwise be a multi-minute call on a finger-sized volume at
    1mm resolution.

    Args:
        shape:      trimesh.Trimesh alpha shape, e.g. from generate_alphashape().
        resolution: Grid spacing (metres).
        chunk_size: How many grid points to test against the shape per
                   shape.contains() call.
        verbose:    Print progress every ~10 chunks.

    Returns:
        (M, 3) float32 array of grid points inside the shape. Empty if
        shape is None, or if the shape is smaller than `resolution` in some
        dimension (no grid point happens to fall inside it).
    """
    if shape is None:
        return np.empty((0, 3), dtype=np.float32)

    lo, hi = shape.bounds
    axes = [np.arange(lo[i], hi[i] + resolution, resolution) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    n = len(grid_pts)
    inside = np.zeros(n, dtype=bool)
    n_chunks = max(1, -(-n // chunk_size))  # ceil div
    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        inside[start:end] = shape.contains(grid_pts[start:end])
        if verbose and (i % 10 == 0 or end == n):
            print(f"    sample_workspace_grid: {end:,}/{n:,} grid points tested "
                  f"({100 * end / n:.0f}%)")

    return grid_pts[inside].astype(np.float32)


def _safe_alphashape(points: np.ndarray, alpha: float):
    """
    alphashape.alphashape(), but treats "no valid shape at this alpha" as a
    clean None instead of letting it crash or return a non-3-D result.

    A degenerate candidate alpha (too tight for this point cloud) can make
    alphashape's own call into trimesh.repair.fix_normals() raise an
    IndexError outright -- indexing an empty faces array -- rather than
    returning an empty/invalid shape as its docstring implies it should.
    alpha=0 is also a special case in this library: it can return a 2-D
    shapely Polygon instead of a 3-D trimesh.Trimesh, which isn't usable
    as a workspace volume either.
    """
    import alphashape as _alphashape
    import trimesh as _trimesh

    try:
        shape = _alphashape.alphashape(points, alpha)
    except Exception:
        return None
    if not isinstance(shape, _trimesh.Trimesh) or len(shape.faces) == 0:
        return None
    return shape


def _encloses_all_points(shape, points: np.ndarray, tol: float) -> bool:
    """
    True if `shape` is a single watertight region and every point in
    `points` lies on or inside it (within `tol`).

    Matches MATLAB alphaShape's criticalAlpha(shape,'all-points') criterion:
    a single region containing every input point -- not just "some valid
    mesh", which an alpha-shape can be while still leaving individual points
    (e.g. ones on a thin protrusion) stranded outside it.
    """
    import trimesh as _trimesh

    if shape is None or not shape.is_watertight:
        return False
    dist = _trimesh.proximity.signed_distance(shape, points)
    return bool(np.all(dist >= -tol))


def _fit_alphashape(points: np.ndarray, max_steps: int = 40):
    """
    Find the critical alpha for `points`: the tightest alpha shape that
    still contains every point in a single connected region (MATLAB's
    alphaShape criticalAlpha(shape,'all-points'), which is what
    Alphashape_generation.m used).

    This doesn't use alphashape.optimizealpha(): that function always tests
    an astronomically large alpha (sys.float_info.max) first, as a sanity
    upper bound, before bisecting downward. For the densely, regularly
    sampled point clouds this project's joint-space sweeps produce, that
    first probe reliably crashes (see _safe_alphashape) rather than
    returning cleanly, which optimizealpha doesn't handle -- so the search
    never gets to the small, perfectly good alpha values below it.

    Instead: start from the convex hull (alpha=0), which by definition
    contains every point in one region, then tighten (alpha x1.5 per step)
    for as long as the shape stays a single region containing every point --
    stopping and returning the last shape that did, the moment a tighter
    alpha would either fragment the shape, crash, or leave any point
    stranded outside it.
    """
    import trimesh as _trimesh

    try:
        # alphashape.alphashape(points, 0) is documented as "safely returns
        # a convex hull", but in practice it can hand back a 2-D shapely
        # Polygon instead of a 3-D mesh -- go straight to trimesh's own
        # convex hull instead, which is unambiguously a 3-D Trimesh.
        hull = _trimesh.convex.convex_hull(points)
    except Exception:
        hull = None
    if hull is None or not isinstance(hull, _trimesh.Trimesh) or len(hull.faces) == 0:
        return None  # too small / too degenerate for even a convex hull

    tree = cKDTree(points)
    median_nn = float(np.median(tree.query(points, k=2)[0][:, 1]))
    tol = 2 * median_nn if median_nn > 0 else 1e-6

    if not _encloses_all_points(hull, points, tol):
        # Shouldn't happen for a true convex hull, but stay defensive.
        return hull

    best = hull
    alpha = 1.0 / (4 * median_nn) if median_nn > 0 else 1.0
    for _ in range(max_steps):
        candidate = _safe_alphashape(points, alpha)
        if not _encloses_all_points(candidate, points, tol):
            break
        best = candidate
        alpha *= 1.5

    # Rebuild fresh from vertices/faces before returning: after many rounds
    # of alphashape.alphashape() constructing intermediate meshes during the
    # search above, the winning mesh's cached spatial-index/proximity
    # structures can end up corrupted -- observed directly on a real
    # workspace shape, where even mesh.contains([mesh.centroid]) came back
    # False and proximity.signed_distance() raised an IndexError deep in
    # trimesh's nearby-faces lookup. A clean Trimesh(..., process=True)
    # forces every cache to be rebuilt from the (valid) topology, and fixes
    # both of those; a plausible cause of an OOM crash further downstream
    # too, since it's very likely the same corrupted structure that gets
    # queried against a large point batch rather than a small one.
    return _trimesh.Trimesh(vertices=best.vertices.copy(), faces=best.faces.copy(), process=True)


def workspace_volume(shape) -> float:
    """Return the volume (m³) of an alpha shape."""
    if shape is None:
        return 0.0
    if hasattr(shape, "volume"):
        return float(shape.volume)
    return 0.0


def save_finger_workspace(
    save_dir: str,
    finger_label: str,
    points: np.ndarray,
    shape=None,
) -> float:
    """
    Persist a computed finger workspace (points + alpha shape + volume) to disk.

    Writes:
        <finger_label>_wkspace_pts.npy    the raw point cloud
        <finger_label>_wkspace.csv        the raw point cloud, CSV
        <finger_label>_wkspace_alpha.pkl  the alpha shape (if provided)
        <finger_label>_wkspace_volume.json  {"volume_m3": ...}

    Args:
        save_dir:     Directory to write into (created if missing).
        finger_label: File stem, e.g. "Finger_1" or "Thumb".
        points:       (N, 3) workspace point cloud from finger_workspace().
        shape:        Alpha shape from generate_alphashape(), or None.

    Returns:
        The workspace volume (m^3) that was saved (0.0 if shape is None).
    """
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"{finger_label}_wkspace_pts.npy"), points)
    np.savetxt(os.path.join(save_dir, f"{finger_label}_wkspace.csv"), points, delimiter=",")

    vol = workspace_volume(shape)
    with open(os.path.join(save_dir, f"{finger_label}_wkspace_volume.json"), "w") as f:
        json.dump({"volume_m3": vol}, f)

    if shape is not None:
        with open(os.path.join(save_dir, f"{finger_label}_wkspace_alpha.pkl"), "wb") as f:
            pickle.dump(shape, f)

    return vol


# ---------------------------------------------------------------------------
# Per-body transform sweep (for later grasp synthesis)
# ---------------------------------------------------------------------------

@dataclass
class FingerWorkspaceTransforms:
    """
    The pose of every body in a finger, at every sampled workspace configuration.

    Mirrors MATLAB's Finger_Workspace_2() output struct: `Jointspace` plus one
    `Body_N` field per link. Here the per-body arrays are keyed by body name
    instead of positional Body_1/Body_2/... fields.
    """
    jointspace: np.ndarray                    # (n_configs, model.nq)
    body_transforms: dict[str, np.ndarray]    # body_name -> (n_configs, 4, 4)


def finger_workspace_transforms(
    finger: FingerSet,
    actuated_joint_space: np.ndarray,
    coupling: CouplingInfo,
    model,
    data,
    q_home: np.ndarray,
) -> FingerWorkspaceTransforms:
    """
    Sweep a finger's actuated joints and record each body's world-frame
    transform at every sampled pose (rather than its transformed surface
    points), so a later grasp-synthesis step can place fingertip bodies at
    specific workspace configurations.

    Replicates MATLAB's Finger_Workspace_2().

    Args:
        finger:               FingerSet for one finger.
        actuated_joint_space: (n_actuated_total, N_pts) array, same layout as
                              finger_workspace().
        coupling:             CouplingInfo from build_coupling_matrix().
        model, data:          Pinocchio model and data.
        q_home:               Home/neutral pinocchio configuration.

    Returns:
        FingerWorkspaceTransforms with one row per sampled pose.
    """
    empty = FingerWorkspaceTransforms(jointspace=np.empty((0, len(q_home))), body_transforms={})

    if finger.is_empty():
        return empty

    finger_act_full_ids = sorted(set(finger.actuated_joint_ids_flat))
    if not finger_act_full_ids:
        return empty

    full_id_to_row = {fid: i for i, fid in enumerate(coupling.actuated_joint_ids)}
    finger_rows = [full_id_to_row[fid] for fid in finger_act_full_ids
                   if fid in full_id_to_row]
    if not finger_rows:
        return empty

    N_pts = actuated_joint_space.shape[1]
    n_joints = len(finger_rows)

    # Step sizes matching MATLAB's halving scheme (same as finger_workspace())
    pt_ranges = []
    for k in range(n_joints):
        step = 1 if k < 2 else 2 ** (k - 1)
        pt_ranges.append(list(range(0, N_pts, step)))

    bodies = [lnk for lnk in finger.links if lnk.points is not None and len(lnk.points) > 0]
    if not bodies:
        return empty

    # Resolve/build once -- see finger_workspace()'s identical comment.
    frame_ids = {lnk.body_name: model.getFrameId(lnk.body_name) for lnk in bodies}
    mesh_offsets = {
        lnk.body_name: _mesh_offset_matrix(lnk.rot_offset, lnk.trans_offset)
        for lnk in bodies
    }

    max_configs = 1
    for r in pt_ranges:
        max_configs *= len(r)

    jointspace = np.zeros((max_configs, len(q_home)), dtype=np.float32)
    body_transforms = {
        lnk.body_name: np.zeros((max_configs, 4, 4), dtype=np.float32) for lnk in bodies
    }

    total = max_configs
    count = 0
    report_every = max(1, total // 20)

    for combo in itertools.product(*pt_ranges):
        actuated_q = np.zeros(coupling.n_actuated)
        for k, row in enumerate(finger_rows):
            actuated_q[row] = actuated_joint_space[row, combo[k]]

        full_q = coupling.matrix @ np.append(actuated_q, 1.0)

        q = q_home.copy()
        q[:len(full_q)] = full_q

        compute_fk(model, data, q)
        jointspace[count] = q

        for lnk in bodies:
            T_world_link = get_transform_by_id(model, data, frame_ids[lnk.body_name])
            body_transforms[lnk.body_name][count] = T_world_link @ mesh_offsets[lnk.body_name]

        count += 1
        if count % report_every == 0:
            print(f"  Workspace transforms {100 * count / total:.1f}% complete")

    return FingerWorkspaceTransforms(
        jointspace=jointspace[:count],
        body_transforms={name: arr[:count] for name, arr in body_transforms.items()},
    )


def save_finger_workspace_transforms(
    save_dir: str,
    finger_label: str,
    result: FingerWorkspaceTransforms,
) -> str:
    """
    Persist a FingerWorkspaceTransforms to a single compressed .npz file.

    Replicates MATLAB's save(fullfile(saveFolder,'Finger_N_wkspace_pts.mat'), 'Workspace').

    Returns:
        The path written.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{finger_label}_wkspace_transforms.npz")
    np.savez_compressed(
        path,
        jointspace=result.jointspace,
        body_names=np.array(list(result.body_transforms.keys())),
        **{f"body__{name}": arr for name, arr in result.body_transforms.items()},
    )
    return path


def load_finger_workspace_transforms(path: str) -> FingerWorkspaceTransforms:
    """Load a FingerWorkspaceTransforms previously written by save_finger_workspace_transforms()."""
    with np.load(path) as data:
        body_names = data["body_names"]
        body_transforms = {name: data[f"body__{name}"] for name in body_names}
        return FingerWorkspaceTransforms(
            jointspace=data["jointspace"],
            body_transforms=body_transforms,
        )
