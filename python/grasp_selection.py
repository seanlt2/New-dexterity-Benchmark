"""
Finger pose selection and two-finger opposable grasp construction.
Replaces MATLAB's Finger_Pose_Selection.m, select_diverse_grasps.m, and
Grasp_2_finger.m.

finger_pose_selection() and grasp_2_finger()'s opposition test were
originally ported as a chain of hard pass/fail filters (contact-distance
range, surface-normal-vs-sphere-center angle, a sharpness threshold, an
opposition-angle threshold) faithfully mirroring the MATLAB source -- any
one of those returning zero survivors killed the whole search ("No Finger 1
poses -- stopping."), which happened often enough in practice (e.g. every
grasp attempt failing for a given hand/sphere combination) to be a real
usability problem. Both were rewritten to rank most candidates instead of
thresholding them, keeping only one hard filter each:

  - finger_pose_selection() keeps the contact-to-sphere distance range
    (+/- 5%) as a hard cutoff -- a candidate whose nearest surface point
    isn't actually near the target sphere isn't a real contact, no matter
    how it ranks against the rest, so this one stays a filter rather than
    becoming "the least-bad of a set that never touches the sphere at
    all." Everything else (the surface-normal-vs-center angle, the
    sharpness threshold) is no longer filtered on; of the poses that DO
    pass the distance cutoff, the n_candidates closest to the sphere
    surface are kept, ranked best first.
  - grasp_2_finger() ranks (rather than thresholds at 135 deg) every
    joint-compatible finger-pair by opposition angle and keeps the best
    n_pose_pool before diverse-subset selection.

Both return *something* (down to a single pose) whenever at least one
candidate exists within contact range of the sphere, so downstream force-
closure/force-ellipsoid evaluation always has real candidates to judge --
including imperfect ones, which is the caller's (or
force_closure_check()'s) job to reject, not this selection stage's. A
finger that genuinely can't reach the sphere's surface at all still
correctly reports no poses found.

One known inconsistency in the MATLAB source was carried over faithfully
rather than silently "fixed" -- called out again in the porting summary:

  select_diverse_grasps.m hardcodes body index 4 (the 4th body) as "the
  fingertip" when building its diversity feature vector. The hand config
  actually active in Manipulation_capacity_region.m (Ability Hand) only has
  2 bodies per finger, so that literal index would be out of bounds. Ported
  with the contact body (finger*_selected_body) passed in explicitly
  instead of a hardcoded index 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial import cKDTree

from .finger import FingerSet
from .workspace import FingerWorkspaceTransforms


# ---------------------------------------------------------------------------
# Finger pose selection
# ---------------------------------------------------------------------------

@dataclass
class FingerPoseSelectionResult:
    """
    Result of finger_pose_selection(): the n_candidates best of a finger's
    swept workspace poses, ranked by contact-to-sphere distance (best/
    closest-to-the-target-surface first).

    All arrays are aligned along axis 0 (one row per returned pose).
    """
    pose_found: bool
    jointspace: np.ndarray                  # (n_poses, nq)
    transforms: dict[str, np.ndarray]       # body_name -> (n_poses, 4, 4), every body
    distances: np.ndarray                   # (n_poses,)
    sharpness: np.ndarray                   # (n_poses,)
    closest_point_ids: np.ndarray           # (n_poses,) int, index into the body's points
    center_body: np.ndarray                 # (n_poses, 3), sphere center in body frame


def _empty_pose_selection(workspace_transforms: FingerWorkspaceTransforms) -> FingerPoseSelectionResult:
    nq = workspace_transforms.jointspace.shape[1] if workspace_transforms.jointspace.ndim == 2 else 0
    return FingerPoseSelectionResult(
        pose_found=False,
        jointspace=np.empty((0, nq)),
        transforms={name: np.empty((0, 4, 4)) for name in workspace_transforms.body_transforms},
        distances=np.empty(0),
        sharpness=np.empty(0),
        closest_point_ids=np.empty(0, dtype=int),
        center_body=np.empty((0, 3)),
    )


def finger_pose_selection(
    finger: FingerSet,
    body_idx: int,
    workspace_transforms: FingerWorkspaceTransforms,
    sphere_center: np.ndarray,
    radius: float,
    n_candidates: int = 500,
) -> FingerPoseSelectionResult:
    """
    Keep the swept workspace poses whose contact point actually lies on the
    target sphere (+/- 5%), and return the n_candidates closest.

    Replaces Finger_Pose_Selection.m's chain of hard pass/fail filters
    (distance range, surface-normal-vs-center angle, a sharpness threshold)
    with just one hard filter -- the contact-to-sphere distance range -- plus
    a ranking by that same distance among the poses that pass it. A pose
    whose nearest surface point isn't actually near the sphere isn't a real
    contact candidate regardless of how it ranks against the others, so
    that one filter stays; the sharpness/angle thresholds don't, so a search
    no longer comes back empty just because no candidate happened to
    satisfy every threshold simultaneously among otherwise-valid contacts.
    See the module docstring for the full rationale.

    Args:
        finger:               FingerSet the body belongs to.
        body_idx:             Index into finger.links of the contact body.
        workspace_transforms: FingerWorkspaceTransforms for this finger.
        sphere_center:        (3,) target sphere center, world frame.
        radius:                Target sphere radius.
        n_candidates:          Max number of best-ranked poses to return.

    Returns:
        FingerPoseSelectionResult with up to n_candidates poses (closest to
        the target sphere surface first). pose_found=False if
        workspace_transforms has zero swept configurations, or none of them
        bring the contact point within +/- 5% of the target sphere surface.
    """
    n_configs = len(workspace_transforms.jointspace)
    if n_configs == 0:
        return _empty_pose_selection(workspace_transforms)

    lnk = finger.links[body_idx]
    body_pts = lnk.points
    body_sharpness = lnk.sharpness

    tree = cKDTree(body_pts)

    body_T = workspace_transforms.body_transforms[lnk.body_name]   # (n_configs, 4, 4)
    R = body_T[:, :3, :3]
    t = body_T[:, :3, 3]

    # sphere_center_body[i] = R[i]' @ (sphere_center - t[i])
    diff = sphere_center[None, :] - t                         # (n, 3)
    sphere_center_body = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), diff)

    distances, closest_point_ids = tree.query(sphere_center_body, k=1)

    # ---- Distance cutoff: contact point must actually lie on the sphere
    # surface (+/- 5%) -- the one hard filter kept from the original
    # pipeline (see module docstring). Without it, a finger whose workspace
    # doesn't reach anywhere near the sphere would still hand back
    # "candidates" that are merely the least-bad of a set that never
    # touches the target at all, which isn't a real contact.
    in_range = (distances <= radius * 1.05) & (distances >= radius * 0.95)
    if not np.any(in_range):
        return _empty_pose_selection(workspace_transforms)

    candidate_ids = np.where(in_range)[0]
    distance_score = np.abs(distances[candidate_ids] - radius)

    n_keep = min(n_candidates, len(candidate_ids))
    order = np.argpartition(distance_score, n_keep - 1)[:n_keep]
    order = order[np.argsort(distance_score[order])]   # best (closest) first
    best = candidate_ids[order]

    print(f"  {n_keep} Finger poses kept (of {len(candidate_ids):,} within contact range, "
          f"{n_configs:,} total)")

    return FingerPoseSelectionResult(
        pose_found=True,
        jointspace=workspace_transforms.jointspace[best],
        transforms={name: arr[best] for name, arr in workspace_transforms.body_transforms.items()},
        distances=distances[best],
        sharpness=body_sharpness[closest_point_ids[best]],
        closest_point_ids=closest_point_ids[best],
        center_body=sphere_center_body[best],
    )


# ---------------------------------------------------------------------------
# Diverse grasp subset selection
# ---------------------------------------------------------------------------

def select_diverse_grasps(
    pose_id: np.ndarray,
    finger2_transforms: dict[str, np.ndarray],
    finger1_transforms: dict[str, np.ndarray],
    finger2_contact_body: str,
    finger1_contact_body: str,
    finger2_norm_vec: np.ndarray,
    finger1_norm_vec: np.ndarray,
    n_select: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick n_select maximally-distinct grasps via farthest-point sampling over
    a per-grasp feature (both contact positions + both contact normals).

    Replicates MATLAB's select_diverse_grasps(). See the module docstring
    (item 2) for why the contact body is passed explicitly instead of
    MATLAB's hardcoded body index 4.

    Args:
        pose_id:              (N, 2) int, [finger1_pose_idx, finger2_pose_idx]
                              per candidate grasp (column order matches
                              grasp_2_finger()'s pose_id).
        finger2_transforms:   body_name -> (n2_poses, 4, 4), from finger 2's
                              FingerPoseSelectionResult.transforms.
        finger1_transforms:   body_name -> (n1_poses, 4, 4), from finger 1's
                              FingerPoseSelectionResult.transforms.
        finger2_contact_body: Body name used as finger 2's contact-position feature.
        finger1_contact_body: Body name used as finger 1's contact-position feature.
        finger2_norm_vec:     (n2_poses, 3) world-frame contact normals for finger 2.
        finger1_norm_vec:     (n1_poses, 3) world-frame contact normals for finger 1.
        n_select:             How many grasps to return.

    Returns:
        (selected_pose_ids, selected_rows): selected_pose_ids is (n_select, 2)
        (a subset of pose_id's rows), selected_rows is the (n_select,) index
        into pose_id.
    """
    N = len(pose_id)
    finger2_pos = finger2_transforms[finger2_contact_body][:, :3, 3]
    finger1_pos = finger1_transforms[finger1_contact_body][:, :3, 3]

    feat = np.zeros((N, 12))
    for r in range(N):
        i1, i2 = pose_id[r, 0], pose_id[r, 1]
        feat[r] = np.concatenate([
            finger2_pos[i2], finger1_pos[i1],
            finger2_norm_vec[i2], finger1_norm_vec[i1],
        ])

    # Standardize each column so positions (metres) and unit normals
    # contribute on comparable scales.
    mu = feat.mean(axis=0)
    sig = feat.std(axis=0)
    sig[sig < np.finfo(float).eps] = 1.0
    F = (feat - mu) / sig

    # --- Farthest-point sampling (max-min distance) ---
    selected = np.zeros(n_select, dtype=int)
    seed = int(np.argmin(np.sum(F ** 2, axis=1)))   # start near the feature-space centre
    selected[0] = seed
    min_dist = np.sum((F - F[seed]) ** 2, axis=1)
    for k in range(1, n_select):
        next_idx = int(np.argmax(min_dist))         # the grasp most unlike anything picked
        selected[k] = next_idx
        min_dist = np.minimum(min_dist, np.sum((F - F[next_idx]) ** 2, axis=1))

    return pose_id[selected], selected


# ---------------------------------------------------------------------------
# Two-finger opposable grasp
# ---------------------------------------------------------------------------

@dataclass
class Grasp2FingerResult:
    """Result of grasp_2_finger()."""
    grasp_found: bool
    selected_jointspace: np.ndarray             # (n_selected, nq)
    finger1_contact_transforms: np.ndarray      # (n_selected, 4, 4)
    finger2_contact_transforms: np.ndarray      # (n_selected, 4, 4)
    finger1_closest_point_body: np.ndarray      # (n_selected, 3), contact pt in body frame
    finger2_closest_point_body: np.ndarray      # (n_selected, 3)


# A collision checker gets both fingers' full pose-selection results and the
# pair of surviving-pose indices being tested, and returns True if that pair
# collides. See grasp_2_finger()'s docstring for why this is pluggable.
CollisionChecker = Callable[
    [FingerPoseSelectionResult, int, FingerPoseSelectionResult, int], bool
]


def _no_collision_found(
    r1: FingerPoseSelectionResult, i1: int, r2: FingerPoseSelectionResult, i2: int
) -> bool:
    return False


def grasp_2_finger(
    finger1: FingerSet,
    finger1_selected_body: int,
    finger1_workspace: FingerWorkspaceTransforms,
    finger2: FingerSet,
    finger2_selected_body: int,
    finger2_workspace: FingerWorkspaceTransforms,
    sphere_center: np.ndarray,
    sphere_radius: float,
    n_target_poses: int = 10,
    n_candidates: int = 500,
    n_pose_pool: int = 2000,
    collision_checker: Optional[CollisionChecker] = None,
) -> Grasp2FingerResult:
    """
    Find opposable two-finger grasps of a sphere at sphere_center.

    Replicates MATLAB's Grasp_2_finger(), with three differences:

    1. Per-finger pose selection and opposition testing are rank-based
       rather than hard pass/fail filters -- see the module docstring for
       why. finger_pose_selection() (per finger) and the opposition test
       below (across fingers) both keep their best-ranked candidates rather
       than eliminating everything below a threshold.
    2. MATLAB's collision-check stage builds solid collision geometry for
       the palm and both fingers (collectHandObjects.m / checkHandCollisions.m,
       built on fitCollisionGeometry.m) and rejects any pose pair that
       collides. That collision-geometry pipeline hasn't been ported to
       Python (see opposability.py's palm-collision docstring for the same
       caveat), so collision filtering here is pluggable: pass a
       `collision_checker(finger1_result, i1, finger2_result, i2) -> bool`
       callable to reject specific pose pairs, or leave it as None to treat
       every candidate pair as collision-free (matching MATLAB behavior only
       when nothing actually collides).
    3. MATLAB's Palm/palmtrans1 arguments exist solely to feed that
       collision-geometry construction; since collision-checking is now the
       caller's responsibility (via collision_checker), they aren't part of
       this signature -- a caller wiring up a real collision_checker should
       close over whatever palm/hand data it needs.

    Args:
        finger1, finger2:               FingerSets for the two contacting fingers.
        finger1_selected_body,
        finger2_selected_body:          Index into each finger's .links of the contact body.
        finger1_workspace,
        finger2_workspace:               FingerWorkspaceTransforms for each finger
                                        (from workspace.finger_workspace_transforms()).
        sphere_center:                   (3,) target sphere center, world frame.
        sphere_radius:                   Target sphere radius.
        n_target_poses:                  Max number of diverse grasps to return.
        n_candidates:                    Max best-by-distance poses kept per finger
                                        (passed through to finger_pose_selection()).
        n_pose_pool:                     Max best-by-opposition-angle finger-pairs
                                        kept before diverse-subset selection.
        collision_checker:               See note 2 above.

    Returns:
        Grasp2FingerResult. grasp_found=False (with empty arrays) if either
        finger has no pose bringing its contact point within the sphere's
        +/- 5% distance range (finger_pose_selection()'s one hard filter),
        or no joint-compatible finger-pair exists at all (a kinematic
        impossibility, not a quality threshold).
    """
    not_found = Grasp2FingerResult(
        grasp_found=False,
        selected_jointspace=np.empty((0, 0)),
        finger1_contact_transforms=np.empty((0, 4, 4)),
        finger2_contact_transforms=np.empty((0, 4, 4)),
        finger1_closest_point_body=np.empty((0, 3)),
        finger2_closest_point_body=np.empty((0, 3)),
    )

    # ---- Finger 1 pose selection ------------------------------------------
    r1 = finger_pose_selection(finger1, finger1_selected_body, finger1_workspace,
                               sphere_center, sphere_radius, n_candidates)
    print(f"  {r1.jointspace.shape[0]} Finger 1 poses")
    if not r1.pose_found:
        print("  No Finger 1 poses within contact range of the sphere -- stopping.")
        return not_found

    # ---- Finger 2 pose selection ------------------------------------------
    r2 = finger_pose_selection(finger2, finger2_selected_body, finger2_workspace,
                               sphere_center, sphere_radius, n_candidates)
    print(f"  {r2.jointspace.shape[0]} Finger 2 poses")
    if not r2.pose_found:
        print("  No Finger 2 poses within contact range of the sphere -- stopping.")
        return not_found

    n1 = r1.jointspace.shape[0]
    n2 = r2.jointspace.shape[0]

    body1_name = finger1.links[finger1_selected_body].body_name
    body2_name = finger2.links[finger2_selected_body].body_name

    finger1_norm_vec = np.zeros((n1, 3))
    for i in range(n1):
        R = r1.transforms[body1_name][i, :3, :3]
        n_body = finger1.links[finger1_selected_body].surf_norm[r1.closest_point_ids[i]]
        finger1_norm_vec[i] = R @ n_body

    finger2_norm_vec = np.zeros((n2, 3))
    for j in range(n2):
        R = r2.transforms[body2_name][j, :3, :3]
        n_body = finger2.links[finger2_selected_body].surf_norm[r2.closest_point_ids[j]]
        finger2_norm_vec[j] = R @ n_body

    # ---- Opposition ranking + shared-joint compatibility --------------------
    # Shared-joint agreement is a kinematic feasibility constraint (both
    # fingers physically share that actuator, so they can't disagree on its
    # position simultaneously) -- kept as a hard mask. Opposition angle is a
    # quality measure, not a feasibility one, so it's ranked (largest/most-
    # opposing first) and truncated to the best n_pose_pool pairs instead of
    # thresholded at a fixed angle -- see the module docstring.
    joint_ids_1 = {lnk.joint_id for lnk in finger1.links if lnk.joint_id != 0}
    joint_ids_2 = {lnk.joint_id for lnk in finger2.links if lnk.joint_id != 0}
    common_ids = sorted(joint_ids_1 & joint_ids_2)

    cos_matrix = np.clip(finger1_norm_vec @ finger2_norm_vec.T, -1.0, 1.0)   # (n1, n2)
    angle_matrix = np.arccos(cos_matrix)

    if common_ids:
        js1_common = r1.jointspace[:, common_ids]           # (n1, len(common_ids))
        js2_common = r2.jointspace[:, common_ids]            # (n2, len(common_ids))
        compatible = np.all(js1_common[:, None, :] == js2_common[None, :, :], axis=2)
        angle_matrix = np.where(compatible, angle_matrix, -np.inf)

    flat = angle_matrix.ravel()
    n_feasible = int(np.count_nonzero(np.isfinite(flat)))
    print(f"  {n_feasible:,} joint-compatible finger-pairs (of {n1 * n2:,} total)")
    if n_feasible == 0:
        print("  No joint-compatible finger-pairs -- stopping.")
        return not_found

    n_pool = min(n_pose_pool, n_feasible)
    best_flat = np.argpartition(-flat, n_pool - 1)[:n_pool]
    best_flat = best_flat[np.argsort(-flat[best_flat])]   # best (most-opposing) first
    pose_id = np.column_stack(np.unravel_index(best_flat, (n1, n2)))   # (P, 2): [finger1_idx, finger2_idx]

    print(f"  {len(pose_id)} potential grasp poses (best by opposition angle)")

    # ---- Collision check (pluggable; see docstring note 2) -----------------
    checker = collision_checker if collision_checker is not None else _no_collision_found
    collides = np.array([checker(r1, i, r2, j) for i, j in pose_id])
    valid_pose_id = pose_id[~collides]

    print(f"  {valid_pose_id.shape[0]} valid grasp poses")
    if valid_pose_id.shape[0] == 0:
        print("  All poses collided -- stopping.")
        return not_found

    # ---- Diverse subset selection -------------------------------------------
    n_selected = min(n_target_poses, valid_pose_id.shape[0])
    selected_pose_ids, _ = select_diverse_grasps(
        valid_pose_id, r2.transforms, r1.transforms, body2_name, body1_name,
        finger2_norm_vec, finger1_norm_vec, n_selected,
    )

    if not common_ids:
        selected_jointspace = (
            r1.jointspace[selected_pose_ids[:, 0]] + r2.jointspace[selected_pose_ids[:, 1]]
        )
    else:
        js2 = r2.jointspace[selected_pose_ids[:, 1]].copy()
        js2[:, common_ids] = 0
        selected_jointspace = r1.jointspace[selected_pose_ids[:, 0]] + js2

    finger1_contact_transforms = np.zeros((n_selected, 4, 4))
    finger2_contact_transforms = np.zeros((n_selected, 4, 4))
    finger1_closest_point_body = np.zeros((n_selected, 3))
    finger2_closest_point_body = np.zeros((n_selected, 3))

    for i in range(n_selected):
        i1, i2 = selected_pose_ids[i]

        pt_body_1 = finger1.links[finger1_selected_body].points[r1.closest_point_ids[i1]]
        T1 = r1.transforms[body1_name][i1]
        pt_world_1 = (T1 @ np.append(pt_body_1, 1.0))[:3]
        n1v = finger1_norm_vec[i1]
        ref1 = np.array([0.0, 0.0, 1.0]) if abs(np.dot(n1v, [0, 0, 1])) < 0.9 else np.array([1.0, 0.0, 0.0])
        x1 = np.cross(ref1, n1v); x1 /= np.linalg.norm(x1)
        y1 = np.cross(n1v, x1)
        T_contact_1 = np.eye(4)
        T_contact_1[:3, :3] = np.column_stack([x1, y1, n1v])
        T_contact_1[:3, 3] = pt_world_1
        finger1_contact_transforms[i] = T_contact_1

        pt_body_2 = finger2.links[finger2_selected_body].points[r2.closest_point_ids[i2]]
        T2 = r2.transforms[body2_name][i2]
        pt_world_2 = (T2 @ np.append(pt_body_2, 1.0))[:3]
        n2v = finger2_norm_vec[i2]
        ref2 = np.array([0.0, 0.0, 1.0]) if abs(np.dot(n2v, [0, 0, 1])) < 0.9 else np.array([1.0, 0.0, 0.0])
        x2 = np.cross(ref2, n2v); x2 /= np.linalg.norm(x2)
        y2 = np.cross(n2v, x2)
        T_contact_2 = np.eye(4)
        T_contact_2[:3, :3] = np.column_stack([x2, y2, n2v])
        T_contact_2[:3, 3] = pt_world_2
        finger2_contact_transforms[i] = T_contact_2

        finger1_closest_point_body[i] = pt_body_1
        finger2_closest_point_body[i] = pt_body_2

    return Grasp2FingerResult(
        grasp_found=True,
        selected_jointspace=selected_jointspace,
        finger1_contact_transforms=finger1_contact_transforms,
        finger2_contact_transforms=finger2_contact_transforms,
        finger1_closest_point_body=finger1_closest_point_body,
        finger2_closest_point_body=finger2_closest_point_body,
    )
