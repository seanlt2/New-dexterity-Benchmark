"""
Force polytope vertex search via the parallelotope-face algorithm of
Skuric, Padois & Daney, "On-line force capability evaluation based on
efficient polytope vertex search" (ICRA 2021) -- specifically their
Algorithm 1.

Not used anywhere in this project's pipeline yet -- this project's existing
manipulability analysis (python/force_ellipsoid.py) uses the ellipsoid
approximation the paper argues underestimates a mechanism's true force
capability; this module computes the EXACT force polytope instead, should
that ever be worth switching to.

Background (paper section II-III): for an n-DOF mechanism with Jacobian
J ∈ R^(m x n) and per-joint torque bounds tau_lower <= tau <= tau_upper, the
feasible torque set is an n-dimensional box (a "parallelotope"). The
feasible task-space force set (the "force polytope") is the image, under
the pseudo-inverse of J^T, of that box intersected with Im(J^T) -- the
subspace of torques that actually produce a task-space wrench rather than
just stressing the mechanism internally (Ker(J), which J^T can't see).

The paper's key result: the polytope's vertices always lie on the box's
(n-m)-dimensional faces, so instead of an exhaustive O(2^n) search over
every box corner, it's enough to walk each of the C(n,m) sets of parallel
(n-m)-dimensional faces (m of the n joint torques pinned to their lower or
upper bound, the rest left free) and, for each of a face set's 2^m
positions, solve a small (n-m)x(n-m) linear system (via the SVD of J) for
whether that face actually contains a vertex.
"""

from __future__ import annotations

import itertools

import numpy as np


def force_polytope_vertices(
    J: np.ndarray,
    tau_lower: np.ndarray,
    tau_upper: np.ndarray,
    alpha_tol: float = 1e-9,
    rank_tol: float = 1e-10,
    dedup_decimals: int = 9,
) -> np.ndarray:
    """
    Vertices of the feasible task-space force polytope for a mechanism with
    Jacobian J and joint-torque bounds [tau_lower, tau_upper].

    Implements Algorithm 1 of Skuric, Padois & Daney (ICRA 2021, see this
    module's docstring). Assumes J has full row rank (a non-singular
    configuration, m <= n) -- the paper makes this same assumption (its
    footnote 3); a rank-deficient J is rejected outright rather than
    silently producing a wrong/empty result.

    To evaluate a RESIDUAL force polytope (accounting for gravity, dynamics,
    or an already-applied nominal wrench -- paper section III-D, eq. 22),
    just pass tau_lower/tau_upper already shifted by those terms
    (tau_lower - tau_g - tau_d - tau_n, tau_upper - tau_g - tau_d - tau_n)
    rather than the raw actuator limits -- no change to this function is
    needed for that.

    Args:
        J:          (m, n) Jacobian (task-space dim m, DOF n), m <= n.
        tau_lower:  (n,) per-joint lower torque bound.
        tau_upper:  (n,) per-joint upper torque bound, >= tau_lower elementwise.
        alpha_tol:  Numerical tolerance for the alpha2 in [0, 1] face-
                   membership test and the quick-reject bounds check (eq. 17).
        rank_tol:   J is rejected as singular if its smallest singular value
                   is below this (relative to its largest).
        dedup_decimals: Vertices found via different faces that turn out to
                   coincide (e.g. a vertex lying on more than one face) are
                   deduplicated by rounding to this many decimals -- not
                   part of the paper's pseudocode, which returns one f_vert
                   per admissible face, but needed here to return the actual
                   distinct vertex SET.

    Returns:
        (V, m) array of distinct force-polytope vertices, V >= 0. Row order
        is the order faces were found in, not geometrically sorted.
    """
    J = np.asarray(J, dtype=float)
    tau_lower = np.asarray(tau_lower, dtype=float).ravel()
    tau_upper = np.asarray(tau_upper, dtype=float).ravel()

    m, n = J.shape
    if tau_lower.shape != (n,) or tau_upper.shape != (n,):
        raise ValueError(
            f"tau_lower/tau_upper must have shape ({n},) to match J's {n} columns, "
            f"got {tau_lower.shape} and {tau_upper.shape}."
        )
    if np.any(tau_upper < tau_lower):
        raise ValueError("tau_upper must be >= tau_lower elementwise.")
    if m > n:
        raise ValueError(f"J must have at least as many columns (DOF={n}) as rows (task-space dim={m}).")

    # --- SVD of J (eq. 12): U (m,m), S (m,), V = [V1 | V2], V1 (n,m), V2 (n,n-m) ---
    U, S, Vt = np.linalg.svd(J, full_matrices=True)
    if S[-1] < rank_tol * S[0]:
        raise ValueError(
            "J does not have full row rank (singular/near-singular configuration) -- "
            "Algorithm 1 assumes a non-singular Jacobian (see paper footnote 3)."
        )
    V = Vt.T
    V1 = V[:, :m]
    V2 = V[:, m:]  # (n, n-m); Ker(J), used to project out J^T f (eq. 13)

    # J^{T+} = U @ diag(1/S) @ V1^T (m, n) -- the pseudo-inverse of J^T,
    # built from J's own SVD (already computed above) rather than a second,
    # separate pseudo-inverse call.
    J_pinv_T = (U * (1.0 / S)) @ V1.T

    ranges = tau_upper - tau_lower  # (n,); magnitude of each base vector tau_i (eq. 7)

    vertices: list[np.ndarray] = []

    for fixed_idx in itertools.combinations(range(n), m):
        fixed_idx = list(fixed_idx)
        free_idx = [i for i in range(n) if i not in fixed_idx]

        # T (n-m, n-m): column k = -ranges[free_idx[k]] * V2[free_idx[k], :] (eq. 14)
        V2_free = V2[free_idx, :]
        T = -(V2_free.T) * ranges[free_idx]

        # Quick-reject bounding box on T @ alpha2 for alpha2 in [0,1]^(n-m) (eq. 17).
        t_ub = np.sum(np.maximum(T, 0.0), axis=1)
        t_lb = np.sum(np.minimum(T, 0.0), axis=1)

        # Check every one of this face-set's 2^m positions against the quick-
        # reject bounds BEFORE inverting T -- the point of eq. 17 is to skip
        # the inversion entirely when none of them can possibly contain a
        # vertex.
        candidates = []
        for bits in itertools.product((0, 1), repeat=m):
            tau_o = tau_lower.copy()
            for idx, b in zip(fixed_idx, bits):
                if b:
                    tau_o[idx] = tau_upper[idx]
            rhs = V2.T @ tau_o  # (n-m,)
            if np.all(rhs >= t_lb - alpha_tol) and np.all(rhs <= t_ub + alpha_tol):
                candidates.append((bits, rhs))

        if not candidates:
            continue

        try:
            T_inv = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            continue

        for bits, rhs in candidates:
            alpha2 = T_inv @ rhs
            if np.any(alpha2 < -alpha_tol) or np.any(alpha2 > 1.0 + alpha_tol):
                continue
            alpha2 = np.clip(alpha2, 0.0, 1.0)

            alpha_full = np.zeros(n)
            alpha_full[fixed_idx] = bits
            alpha_full[free_idx] = alpha2

            tau_vert = tau_lower + alpha_full * ranges  # eq. 7/9
            f_vert = J_pinv_T @ tau_vert                # eq. 16
            vertices.append(f_vert)

    if not vertices:
        return np.empty((0, m))

    verts = np.array(vertices)
    rounded = np.round(verts, decimals=dedup_decimals)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return verts[np.sort(unique_idx)]
