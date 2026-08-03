"""
Grasp wrench matrices, contact wrench cones, and force-closure check.
Replaces MATLAB's Contact_Wrench.m, Grasp_Wrench.m, and Force_Closure_Check.m.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .finger import FingerLink, FingerSet


# ---------------------------------------------------------------------------
# Contact wrench cone
# ---------------------------------------------------------------------------

def contact_wrench(
    mu: float,
    mu_tor: float,
    cone_approx: str = "eight-sided",
) -> np.ndarray:
    """
    Build the contact wrench matrix W for a single contact point.

    Each column is a primitive (extremal) wrench from the friction / torque cone.
    The matrix is expressed in the contact frame where the z-axis is the
    inward contact normal.

    Replicates MATLAB's Contact_Wrench().

    Args:
        mu:          Coulomb friction coefficient (tangential).
        mu_tor:      Torsional friction coefficient (about normal). 0 = frictionless
                     about normal.
        cone_approx: "four-sided" or "eight-sided" polyhedral approximation.

    Returns:
        W: (6, k) matrix, top 3 rows = forces, bottom 3 = moments.
    """
    if mu == 0 and mu_tor == 0:
        # Point contact without friction — only normal force
        return np.array([[0, 0, 1, 0, 0, 0]]).T

    s2 = 1.0 / np.sqrt(2)

    if cone_approx == "four-sided":
        forces = np.array([
            [ mu,  0, 1],
            [  0, mu, 1],
            [-mu,  0, 1],
            [  0,-mu, 1],
        ]).T  # shape (3, 4)
    elif cone_approx == "eight-sided":
        forces = np.array([
            [ mu,      0, 1],
            [  0,     mu, 1],
            [-mu,      0, 1],
            [  0,    -mu, 1],
            [ mu*s2,  mu*s2, 1],
            [ mu*s2, -mu*s2, 1],
            [-mu*s2,  mu*s2, 1],
            [-mu*s2, -mu*s2, 1],
        ]).T  # shape (3, 8)
    else:
        raise ValueError(f"Unsupported cone_approx: '{cone_approx}'")

    n_f = forces.shape[1]

    if mu_tor == 0:
        moments = np.zeros((3, n_f))
        return np.vstack([forces, moments])

    # Add torsional wrench columns (±mu_tor about normal)
    m_pos = np.array([[0, 0,  mu_tor]]).T
    m_neg = np.array([[0, 0, -mu_tor]]).T
    moments = np.hstack([np.zeros((3, n_f)), m_pos, m_neg])
    forces_ext = np.hstack([forces, np.zeros((3, 2))])
    return np.vstack([forces_ext, moments])


# ---------------------------------------------------------------------------
# Grasp wrench matrix
# ---------------------------------------------------------------------------

def grasp_wrench(
    finger: FingerSet,
    body_idx: int,
    T_world_contact: np.ndarray,
    object_center: np.ndarray,
    R_world_object: np.ndarray,
    cone_approx: str = "eight-sided",
) -> np.ndarray:
    """
    Compute the grasp map contribution G for one contact.

    G maps primitive contact wrenches to object wrenches via the adjoint
    transformation at the contact point.

    Replicates MATLAB's Grasp_Wrench().

    Args:
        finger:          FingerSet.
        body_idx:        Index into finger.links of the contacting link.
        T_world_contact: (4, 4) contact frame in world frame.
        object_center:   (3,) object centroid in world frame.
        R_world_object:  (3, 3) object frame rotation in world frame.
        cone_approx:     "four-sided" or "eight-sided".

    Returns:
        G: (6, k) grasp map matrix.
    """
    lnk: FingerLink = finger.links[body_idx]
    mu, mu_tor = lnk.contact_assumption

    W_f = contact_wrench(mu, mu_tor, cone_approx)

    # Rotation from object frame to contact frame
    R_object_contact = T_world_contact[:3, :3] @ R_world_object  # 3×3

    # Contact position relative to object centre
    p_object_contact = T_world_contact[:3, 3] - object_center

    # Skew-symmetric cross-product matrix of p_object_contact
    S = _skew(p_object_contact)

    # Grasp map (adjoint): maps contact wrenches → object wrenches
    G_adj = np.block([
        [R_object_contact,       np.zeros((3, 3))],
        [S @ R_object_contact,   R_object_contact],
    ])  # (6, 6)

    return G_adj @ W_f   # (6, k)


def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for cross product with v."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])


# ---------------------------------------------------------------------------
# Force closure check
# ---------------------------------------------------------------------------

def force_closure_check(W: np.ndarray) -> str:
    """
    Check force closure of a grasp wrench matrix.

    Step 1: rank test (need rank 6 for full force closure).
    Step 2: linear program to find positive primitive wrench intensities that
            yield zero net wrench on the object.

    Replicates MATLAB's Force_Closure_Check().

    Args:
        W: (6, k) grasp wrench matrix (concatenated G matrices for all contacts).

    Returns:
        "Pass" if force closure holds, "Fail" otherwise.
    """
    if np.linalg.matrix_rank(W) < 6:
        print(f"  Rank is {np.linalg.matrix_rank(W)}. Force closure fails.")
        return "Fail"

    k = W.shape[1]

    # Equality: W @ lambda = 0 (zero net wrench) AND sum(lambda) = 1
    A_eq = np.vstack([W, np.ones((1, k))])   # (7, k)
    b_eq = np.zeros(7)
    b_eq[-1] = 1.0

    # Each lambda >= epsilon > 0 (strict positivity)
    lb = 1e-6 * np.ones(k)
    c = np.zeros(k)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, [None] * k)),
                     method="highs")

    if result.status == 0:
        return "Pass"
    return "Fail"
