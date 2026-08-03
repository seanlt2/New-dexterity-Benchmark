"""
Manipulation capacity region: per-point evaluation of how well two fingers
can jointly apply force to an object at a set of candidate positions.
Replaces (lines 1-487 of) MATLAB's Manipulation_capacity_region.m.

Only the reusable computation is ported here -- hand/URDF setup and legacy
.mat-file loading from that script are superseded by the objects this
package already produces (build_hand(), finger_workspace_transforms(),
opposability.compute_opposable_grasp_volume()); see the module docstring in
opposability_calculation.py for the equivalent pattern.

For each candidate point, this:
  1. Finds opposable two-finger grasps of a sphere centered there
     (grasp_selection.grasp_2_finger()).
  2. Filters to force-closure grasps (grasp.force_closure_check()).
  3. Computes each contact's force ellipsoid from its true (actuated-space)
     Jacobian (kinematics.contact_point_jacobian() + true_jacobian(),
     force_ellipsoid.force_ellipsoid()).
  4. Finds the common projection basis of the two force ellipsoids
     (force_ellipsoid.force_ellipsoid_common_basis()), and keeps the pose
     with the highest basis rank (ties broken by total fraction) as that
     point's manipulation capacity.

One behavioral deviation from the MATLAB source, made because the
generalization from 1 point to N points was evidently unfinished there --
see manipulation_capacity_region()'s docstring for details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .coupling import CouplingInfo
from .finger import FingerSet
from .force_ellipsoid import force_ellipsoid, force_ellipsoid_common_basis
from .grasp import force_closure_check, grasp_wrench
from .grasp_selection import CollisionChecker, grasp_2_finger
from .kinematics import compute_fk, contact_point_jacobian, true_jacobian
from .workspace import FingerWorkspaceTransforms


# ---------------------------------------------------------------------------
# Manipulation capacity region
# ---------------------------------------------------------------------------

@dataclass
class ManipulationCapacityResult:
    """Result of manipulation_capacity_region(), one row per input point."""
    basis: np.ndarray             # (N, 3, 3); zero rows where no basis was found
    rank: np.ndarray              # (N,) int; common-basis dimension (0 = none found)
    ellipsoid_ratio: np.ndarray   # (N, 3); per-axis min/max magnitude fraction


def manipulation_capacity_region(
    points: np.ndarray,
    finger1: FingerSet,
    finger1_selected_body: int,
    finger1_workspace: FingerWorkspaceTransforms,
    finger2: FingerSet,
    finger2_selected_body: int,
    finger2_workspace: FingerWorkspaceTransforms,
    model,
    data,
    coupling: CouplingInfo,
    sphere_radius: float,
    n_target_poses: int = 10,
    angle_tolerance_deg: float = 5.0,
    collision_checker: Optional[CollisionChecker] = None,
) -> ManipulationCapacityResult:
    """
    Evaluate two-finger manipulation capacity at each of N candidate object
    positions.

    Replicates the computational core of Manipulation_capacity_region.m
    (through line 487): for each point, find opposable grasps of a sphere
    there, keep the force-closure ones, and record the common force-ellipsoid
    basis of the pose with the best (rank, then total fraction).

    Deviation from MATLAB: MATLAB creates ONE target_sphere, poses it once
    at a fixed center (outside the point loop), and never updates that pose
    per-point inside the loop -- so every point i is evaluated against
    contacts on the same fixed sphere, regardless of points[i]. That only
    matters when `points` has more than one row, which the MATLAB script
    never actually does (it always evaluates a single point). Since the
    natural use here is evaluating several candidate centers, e.g. from
    opposability.compute_opposable_grasp_volume(), this instead re-centers
    the target sphere at points[i] on every iteration -- matching what the
    per-point array preallocation in the MATLAB source clearly anticipates,
    even though the MATLAB loop body itself doesn't do it. Pass a single-row
    `points` array to reproduce the original script's exact behavior.

    Args:
        points:                (N, 3) candidate object/sphere-center positions,
                               world frame.
        finger1, finger2:       FingerSets for the two contacting fingers.
        finger1_selected_body,
        finger2_selected_body:  Index into each finger's .links of the contact body.
        finger1_workspace,
        finger2_workspace:      FingerWorkspaceTransforms for each finger.
        model, data:             Pinocchio model/data.
        coupling:                CouplingInfo from build_coupling_matrix().
        sphere_radius:           Radius of the test sphere at each point.
        n_target_poses:          Max diverse grasps considered per point
                                (passed through to grasp_2_finger()).
        angle_tolerance_deg:     Passed through to force_ellipsoid_common_basis().
        collision_checker:       Passed through to grasp_2_finger(); see its
                                docstring -- collision filtering is unported
                                and pluggable, defaulting to "no collisions".

    Returns:
        ManipulationCapacityResult with one row per input point. A point
        where no grasp/force-closure/common-basis was found gets rank=0,
        a zero basis, and a zero ellipsoid_ratio.
    """
    N = len(points)
    basis = np.zeros((N, 3, 3))
    rank = np.zeros(N, dtype=int)
    ellipsoid_ratio = np.zeros((N, 3))

    actuated_ids_1 = sorted({aid for lnk in finger1.links for aid in lnk.actuated_joint_ids})
    actuated_ids_2 = sorted({aid for lnk in finger2.links for aid in lnk.actuated_joint_ids})

    for i in range(N):
        point = points[i]

        grasp = grasp_2_finger(
            finger1, finger1_selected_body, finger1_workspace,
            finger2, finger2_selected_body, finger2_workspace,
            sphere_center=point, sphere_radius=sphere_radius,
            n_target_poses=n_target_poses, collision_checker=collision_checker,
        )
        if not grasp.grasp_found:
            continue

        best_rank = 0
        best_ratio_sum = 0.0

        n_poses = grasp.selected_jointspace.shape[0]
        for j in range(n_poses):
            q = grasp.selected_jointspace[j]
            compute_fk(model, data, q)

            G1 = grasp_wrench(finger1, finger1_selected_body,
                              grasp.finger1_contact_transforms[j], point, np.eye(3))
            G2 = grasp_wrench(finger2, finger2_selected_body,
                              grasp.finger2_contact_transforms[j], point, np.eye(3))
            W = np.hstack([G1, G2])

            if force_closure_check(W) != "Pass":
                continue

            J1 = contact_point_jacobian(
                model, data, q, finger1.links[finger1_selected_body].body_name,
                grasp.finger1_contact_transforms[j][:3, 3],
            )
            true_J1 = true_jacobian(J1, coupling.matrix, actuated_ids_1)

            J2 = contact_point_jacobian(
                model, data, q, finger2.links[finger2_selected_body].body_name,
                grasp.finger2_contact_transforms[j][:3, 3],
            )
            true_J2 = true_jacobian(J2, coupling.matrix, actuated_ids_2)

            fe1 = force_ellipsoid(true_J1)
            fe2 = force_ellipsoid(true_J2)

            result = force_ellipsoid_common_basis(
                [fe1, fe2], angle_tolerance_deg=angle_tolerance_deg,
            )
            if result is None:
                continue

            ratio_sum = float(result.fraction.sum())
            best_rank = max(best_rank, result.dim)
            best_ratio_sum = max(best_ratio_sum, ratio_sum)

            if result.dim == best_rank and ratio_sum == best_ratio_sum:
                basis[i] = result.basis
                rank[i] = result.dim
                ellipsoid_ratio[i] = result.fraction

    return ManipulationCapacityResult(basis=basis, rank=rank, ellipsoid_ratio=ellipsoid_ratio)


# ---------------------------------------------------------------------------
# Weighted sampling (standalone utility)
# ---------------------------------------------------------------------------

def sample_weighted_without_replacement(points: np.ndarray, m: int) -> np.ndarray:
    """
    Sample up to m points without replacement, weighted so points later in
    the input array are exponentially more likely to be picked
    (weights = exp(linspace(0, 5, n))).

    Replicates the weighted-sampling snippet in Manipulation_capacity_region.m
    (~lines 171-195). In that script its result only drives a separate
    visualization and isn't fed into the manipulation-capacity computation,
    so it's ported here as a standalone utility rather than wired into
    manipulation_capacity_region().

    Args:
        points: (N, 3) array to sample from -- meaningful only if already in
                some rank order (e.g. by distance or quality), since later
                rows are weighted higher.
        m:      Number of points to sample (clamped to len(points)).

    Returns:
        (min(m, N), 3) sampled points.
    """
    n = len(points)
    m = min(m, n)
    weights = np.exp(np.linspace(0, 5, n))
    available = list(range(n))
    idx = np.zeros(m, dtype=int)

    for i in range(m):
        p = weights / weights.sum()
        r = np.random.rand()
        c = np.cumsum(p)
        chosen = min(int(np.searchsorted(c, r)), len(available) - 1)
        idx[i] = available[chosen]
        del available[chosen]
        weights = np.delete(weights, chosen)

    return points[idx]
