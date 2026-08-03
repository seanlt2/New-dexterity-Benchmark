"""
Force ellipsoid computation, normal correction, and manipulation evaluation.
Replaces MATLAB's Force_Ellipsoid.m, Force_Ellipsoid_vector_correction.m,
Force_Manipulation_Evaluation.m, and forceEllipsoidCommonBasis.m.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Force ellipsoid
# ---------------------------------------------------------------------------

@dataclass
class ForceEllipsoid:
    """Result of force_ellipsoid()."""
    eigenvectors: np.ndarray   # (3, 3) columns are principal force directions
    force_scale: np.ndarray    # (3,) scaled eigenvalues (1/sqrt(lambda)); 0 for null dirs
    contact_frame: np.ndarray  # (4, 4) contact frame T_world_contact


def force_ellipsoid(jacobian: np.ndarray) -> ForceEllipsoid:
    """
    Compute the force ellipsoid from a Jacobian matrix.

    Uses the linear (translational) sub-block of the Jacobian to compute
    A = J_lin @ J_lin.T, then eigendecompose A to get force directions and
    magnitudes.

    Replicates MATLAB's Force_Ellipsoid().

    Args:
        jacobian: (6, n) Jacobian matrix.  Rows 0-2 = angular, 3-5 = linear.
                  (MATLAB convention: rows 1-3 = angular, 4-6 = linear.)

    Returns:
        ForceEllipsoid with eigenvectors, force_scale, and an empty 4×4 I
        contact_frame (populated later by the caller if needed).
    """
    J_lin = jacobian[3:6, :]          # linear sub-block (3 × n)
    rank_J = np.linalg.matrix_rank(J_lin)

    A = J_lin @ J_lin.T               # 3×3 manipulability matrix
    eigenvalues, eigenvectors = np.linalg.eigh(A)  # ascending order

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Force scale = 1/sqrt(lambda) for non-null directions; 0 elsewhere
    force_scale = np.zeros(3)
    for i in range(rank_J):
        if eigenvalues[i] > 0:
            force_scale[i] = 1.0 / np.sqrt(eigenvalues[i])

    # Normalise
    norm_fs = np.linalg.norm(force_scale)
    if norm_fs > 0:
        force_scale /= norm_fs

    # Re-sort by force_scale descending
    idx2 = np.argsort(force_scale)[::-1]
    force_scale = force_scale[idx2]
    eigenvectors = eigenvectors[:, idx2]

    return ForceEllipsoid(
        eigenvectors=eigenvectors,
        force_scale=force_scale,
        contact_frame=np.eye(4),
    )


# ---------------------------------------------------------------------------
# Normal correction
# ---------------------------------------------------------------------------

def force_ellipsoid_vector_correction(
    fe: ForceEllipsoid,
    T_world_contact: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flip ellipsoid eigenvectors to point away from the contact surface (i.e.
    in the same half-space as the contact normal).

    Replicates MATLAB's Force_Ellipsoid_vector_correction().

    Args:
        fe:               ForceEllipsoid from force_ellipsoid().
        T_world_contact:  (4, 4) contact frame; column 2 (z) = outward normal.
        scale:            Scalar to apply to force_scale.

    Returns:
        (v_corr, D_corr): corrected eigenvectors (3, 3) and scaled force scales (3,).
    """
    contact_normal = T_world_contact[:3, 2]   # z-axis of contact frame
    v_corr = fe.eigenvectors.copy()

    for j in range(3):
        if np.dot(v_corr[:, j], contact_normal) < 0:
            v_corr[:, j] = -v_corr[:, j]

    D_corr = scale * fe.force_scale
    return v_corr, D_corr


# ---------------------------------------------------------------------------
# Manipulation evaluation
# ---------------------------------------------------------------------------

def force_manipulation_evaluation(
    force_ellipsoids: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    acceptable_error_angle: float = 85.0,
) -> np.ndarray:
    """
    Evaluate manipulation capability from a pair of force ellipsoids.

    Assesses whether the two contacts together can generate forces in all
    necessary directions.

    Replicates MATLAB's Force_Manipulation_Evaluation().

    Args:
        force_ellipsoids: List of (V, D, T) tuples — one per contact.
                          V: (3, 3) eigenvectors
                          D: (3,)   force_scale
                          T: (4, 4) contact frame
        acceptable_error_angle: Maximum angle deviation (degrees) for a
                                direction to be considered "compatible".

    Returns:
        output: (3, 3) matrix whose non-zero columns are the manipulable
                force directions, or np.eye(3) for full mobility.
    """
    if len(force_ellipsoids) != 2:
        raise ValueError("Exactly 2 force ellipsoids are required.")

    V1, D1, _T1 = force_ellipsoids[0]
    V2, D2, _T2 = force_ellipsoids[1]

    dim1 = int(np.count_nonzero(D1))
    dim2 = int(np.count_nonzero(D2))

    # Ensure (V1, D1) belongs to the contact with higher dimension
    if dim2 > dim1:
        V1, D1, V2, D2 = V2, D2, V1, D1
        dim1, dim2 = dim2, dim1

    # Masked eigenvector matrices (zero out null-space columns)
    mask1 = (D1 != 0).astype(float)
    mask2 = (D2 != 0).astype(float)
    v1 = V1 * mask1[np.newaxis, :]   # (3, 3) with zeroed null cols
    v2 = V2 * mask2[np.newaxis, :]

    if dim1 == 3 and dim2 == 3:
        return np.eye(3)

    if dim1 == 3 and dim2 < 3:
        return v2

    output = np.zeros((3, 3))

    if dim1 == 2 and dim2 >= 1:
        normal_to_v1_plane = np.cross(v1[:, 0], v1[:, 1])
        for col in range(dim2):
            vec = v2[:, col]
            n_vec = np.linalg.norm(vec)
            n_normal = np.linalg.norm(normal_to_v1_plane)
            if n_vec < 1e-12 or n_normal < 1e-12:
                continue
            angle = np.degrees(
                np.arccos(np.clip(
                    np.dot(normal_to_v1_plane, vec) / (n_normal * n_vec), -1, 1
                ))
            )
            err = abs(angle - 90.0) if angle > 90.0 else abs(90.0 - angle)
            if err < acceptable_error_angle:
                output[:, col] = vec

    elif dim1 == 1 and dim2 == 1:
        vec1 = v1[:, 0]
        vec2 = v2[:, 0]
        n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if n1 > 1e-12 and n2 > 1e-12:
            angle = np.degrees(
                np.arccos(np.clip(np.dot(vec1, vec2) / (n1 * n2), -1, 1))
            )
            err = abs(angle - 90.0) if angle > 90.0 else abs(90.0 - angle)
            if err < acceptable_error_angle:
                output[:, 0] = vec1

    return output


# ---------------------------------------------------------------------------
# Common projection basis for K >= 2 force ellipsoids
# ---------------------------------------------------------------------------

@dataclass
class CommonBasisResult:
    """Result of force_ellipsoid_common_basis()."""
    mode: str                  # 'aligned' | 'deficient' | 'intersection'
    rank: np.ndarray           # (K,) int, rank of each input ellipsoid
    basis: np.ndarray          # (3, 3); columns 0:dim are the common basis axes, rest zero
    dim: int                   # number of genuine axes (1, 2, or 3)
    active_axes: np.ndarray    # (3,) bool, True for the first `dim` entries
    projected: list[np.ndarray]  # K x (3, 3); projected[k][i] is ellipsoid k's
                                  # i-th principal force vector in the common basis
    magnitude: np.ndarray      # (3, K); magnitude[j, k] = ellipsoid k's reach along axis j
    deviation_deg: np.ndarray  # (3, K); angle between basis axis j and ellipsoid k's column space
    numerator: np.ndarray      # (3,); least magnitude on each axis
    denominator: np.ndarray    # (3,); greatest magnitude on each axis
    fraction: np.ndarray       # (3,); numerator / denominator, in [0, 1]
    min_ellipsoid: np.ndarray  # (3,) int; index of the ellipsoid holding the least magnitude
    max_ellipsoid: np.ndarray  # (3,) int; index of the ellipsoid holding the greatest magnitude
    objective: float           # alignment objective value ('aligned' mode only; NaN otherwise)


_PERMS3 = list(itertools.permutations(range(3)))


def _orth(A: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the column space of A (SVD-based)."""
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0))
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * np.finfo(float).eps * max(float(s.max()), 0.0)
    r = int(np.sum(s > tol))
    return U[:, :r]


def _aligned_basis(A: list[np.ndarray], max_iter: int) -> tuple[np.ndarray, float]:
    """
    Orthonormal 3-frame maximizing the total retained magnitude when each
    ellipsoid's principal force vectors are each assigned to one basis axis.

    Alternating maximization: the inner permutation/sign assignment is exact
    (6 cases), and with those fixed, the objective is maximized over
    orthogonal U by the polar factor (via SVD) of the accumulated matrix.
    """
    K = len(A)
    U_best = np.eye(3)
    J_best = -np.inf

    for start in range(K):   # multi-start: each ellipsoid's own frame
        U = _orth(A[start])
        if U.shape[1] < 3:
            U = np.eye(3)
        J_prev = -np.inf

        for _ in range(max_iter):
            M = np.zeros((3, 3))
            J = 0.0
            for k in range(K):
                C = U.T @ A[k]
                aC = np.abs(C)
                best_val = -np.inf
                best_sig = (0, 1, 2)
                for sig in _PERMS3:
                    val = aC[sig[0], 0] + aC[sig[1], 1] + aC[sig[2], 2]
                    if val > best_val:
                        best_val = val
                        best_sig = sig
                J += best_val

                S = np.zeros((3, 3))
                for i in range(3):
                    j = best_sig[i]
                    sgn = np.sign(C[j, i])
                    if sgn == 0:
                        sgn = 1.0
                    S[j, i] = sgn
                M += A[k] @ S.T

            Wm, _, Zmt = np.linalg.svd(M)
            U = Wm @ Zmt   # argmax_{U orthogonal} trace(U'*M)

            if J - J_prev <= 1e-12 * max(1.0, abs(J)):
                break
            J_prev = J

        # final objective for this start
        J = 0.0
        for k in range(K):
            aC = np.abs(U.T @ A[k])
            best_val = max(
                aC[sig[0], 0] + aC[sig[1], 1] + aC[sig[2], 2] for sig in _PERMS3
            )
            J += best_val

        if J > J_best:
            J_best = J
            U_best = U

    if np.linalg.det(U_best) < 0:      # right-handed frame
        U_best = U_best.copy()
        U_best[:, 2] = -U_best[:, 2]

    return U_best, J_best


def _common_subspace(
    B: list[np.ndarray],
    ranks: np.ndarray,
    angle_tolerance_deg: float,
) -> tuple[np.ndarray, bool]:
    """
    m-dimensional subspace lying within angle_tolerance_deg of every
    ellipsoid's column space, m = min(rank). Candidates are the leading
    eigenvectors of the mean projector S = (1/K) * sum_k B_k*B_k', which
    maximize the mean squared cosine to every column space.
    """
    K = len(B)
    m = int(ranks.min())
    if m < 1:
        return np.zeros((3, 0)), False

    S = np.zeros((3, 3))
    for k in range(K):
        S += B[k] @ B[k].T
    S /= K
    S = (S + S.T) / 2

    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    U = eigvecs[:, order[:m]]

    cos_tol = np.cos(np.deg2rad(angle_tolerance_deg))
    ok = True
    for j in range(m):
        for k in range(K):
            if np.linalg.norm(B[k].T @ U[:, j]) < cos_tol - 1e-9:
                ok = False
                break
        if not ok:
            break

    if not ok:
        return np.zeros((3, 0)), False
    return U, True


def force_ellipsoid_common_basis(
    ellipsoids: list[ForceEllipsoid],
    angle_tolerance_deg: float = 0.0,
    rank_tol: float = 1e-6,
    max_iter: int = 200,
    verbose: bool = False,
) -> Optional[CommonBasisResult]:
    """
    Common projection basis for K >= 2 force ellipsoids: a shared orthonormal
    frame, each ellipsoid's principal force vectors projected onto it, and
    the min/max magnitude fraction along each basis axis.

    Replicates MATLAB's forceEllipsoidCommonBasis(V1,D1, V2,D2, ..., 'AngleTolerance', deg)
    calling convention. MATLAB additionally accepts several other input
    shapes (struct arrays, cell arrays, mixed forms) purely to make the
    function ergonomic from different call sites — that argument-format
    flexibility isn't ported; pass a plain list of ForceEllipsoid instead.
    Input validation MATLAB does for arbitrary external callers (re-
    orthonormalizing a non-orthonormal V, warning on negative D) is also
    skipped here, since ForceEllipsoid.eigenvectors/force_scale from
    force_ellipsoid() are already well-formed by construction.

    Behavior (mirrors the MATLAB docstring's three cases):
      - All ellipsoids rank 3 -> 'aligned': match each ellipsoid's principal
        force vectors to basis axes (signed permutation) and find the
        orthonormal frame maximizing total retained magnitude.
      - Exactly one ellipsoid rank < 3 -> 'deficient': that ellipsoid's own
        column space IS the common basis.
      - Two or more ellipsoids rank < 3 -> 'intersection': the top
        min(rank)-dimensional eigenspace of the mean column-space projector,
        accepted only if within angle_tolerance_deg of every deficient
        ellipsoid's column space; otherwise None (mirrors MATLAB's `OUT = []`).

    Args:
        ellipsoids: >= 2 ForceEllipsoid (from force_ellipsoid()).
        angle_tolerance_deg: Max angle (deg) a common-basis axis may deviate
            from a rank-deficient ellipsoid's column space (used only in the
            'intersection' case).
        rank_tol: Relative tolerance on force_scale used to decide which
            principal magnitudes count as zero.
        max_iter: Iterations of the alignment solver ('aligned' case).
        verbose: Print progress, mirroring MATLAB's 'Verbose' option.

    Returns:
        CommonBasisResult, or None if no admissible common basis exists (a
        rank-0 ellipsoid, or no basis within angle_tolerance_deg of every
        rank-deficient ellipsoid).
    """
    K = len(ellipsoids)
    if K < 2:
        raise ValueError(f"At least two force ellipsoids are required (got {K}).")

    ranks = np.zeros(K, dtype=int)
    A: list[np.ndarray] = [np.zeros((3, 3))] * K
    B: list[np.ndarray] = [np.zeros((3, 0))] * K
    for k, ell in enumerate(ellipsoids):
        V = ell.eigenvectors
        D = ell.force_scale
        d = np.abs(D)
        A[k] = V @ np.diag(D)
        if d.max() <= 0:
            ranks[k] = 0
            B[k] = np.zeros((3, 0))
        else:
            keep = d > rank_tol * d.max()
            ranks[k] = int(keep.sum())
            B[k] = _orth(V[:, keep])

    if np.any(ranks == 0):
        zero_idx = np.where(ranks == 0)[0]
        print(f"Warning: ellipsoid(s) {zero_idx.tolist()} have rank 0; no common basis is defined.")
        return None

    if verbose:
        print(f"Ranks: {ranks.tolist()}")

    n_deficient = int(np.sum(ranks < 3))
    objective = float("nan")

    if n_deficient == 0:
        U, objective = _aligned_basis(A, max_iter)
        mode = "aligned"
    elif n_deficient == 1:
        kd = int(np.where(ranks < 3)[0][0])
        U = B[kd]
        mode = "deficient"
    else:
        U, ok = _common_subspace(B, ranks, angle_tolerance_deg)
        if not ok:
            print(
                f"Warning: no {int(ranks.min())}-dimensional basis lies within "
                f"{angle_tolerance_deg} deg of every rank-deficient ellipsoid."
            )
            return None
        mode = "intersection"

    m = U.shape[1]
    U_full = np.zeros((3, 3))
    U_full[:, :m] = U
    active_axes = np.zeros(3, dtype=bool)
    active_axes[:m] = True

    projected: list[np.ndarray] = []
    magnitude = np.zeros((3, K))
    deviation_deg = np.zeros((3, K))
    for k in range(K):
        Pk = U_full.T @ A[k]
        projected.append(Pk)
        magnitude[:, k] = np.sqrt(np.sum(Pk ** 2, axis=1))
        for j in range(m):
            c = np.linalg.norm(B[k].T @ U_full[:, j])
            deviation_deg[j, k] = np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))

    numerator = np.zeros(3)
    denominator = np.zeros(3)
    fraction = np.zeros(3)
    min_ell = np.zeros(3, dtype=int)
    max_ell = np.zeros(3, dtype=int)
    for j in range(m):
        max_ell[j] = int(np.argmax(magnitude[j]))
        denominator[j] = magnitude[j, max_ell[j]]
        min_ell[j] = int(np.argmin(magnitude[j]))
        numerator[j] = magnitude[j, min_ell[j]]
        fraction[j] = numerator[j] / denominator[j] if denominator[j] > 0 else 0.0

    result = CommonBasisResult(
        mode=mode, rank=ranks, basis=U_full, dim=m, active_axes=active_axes,
        projected=projected, magnitude=magnitude, deviation_deg=deviation_deg,
        numerator=numerator, denominator=denominator, fraction=fraction,
        min_ellipsoid=min_ell, max_ellipsoid=max_ell, objective=objective,
    )

    if verbose:
        print(f"Mode: {mode}, basis dimension {m} (of 3)")
        for j in range(3):
            if active_axes[j]:
                print(f"  axis {j}: {U_full[:, j]}  fraction = {fraction[j]:.4f}")
            else:
                print(f"  axis {j}: (padding)")

    return result
