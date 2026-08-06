"""
Sphere-based multi-finger opposable grasp volume.
Replaces MATLAB's graspable_volume.m and workspace_palm_collision.m.

Given the per-finger reachable-workspace point clouds from workspace.py
(finger_workspace()), this module finds the set of positions where a sphere
of radius r can be placed such that every selected finger can reach its
surface AND the resulting contacts form an opposable (friction-cone force
closure) grasp, then removes any of those positions whose sphere would
overlap the palm.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Optional

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Candidate grid helper
# ---------------------------------------------------------------------------

def _colon(lo: float, step: float, hi: float) -> np.ndarray:
    """Inclusive float range, matching MATLAB's lo:step:hi colon operator."""
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    n = max(n, 1)
    return lo + step * np.arange(n)


# ---------------------------------------------------------------------------
# Parallel per-point opposability test (3+ fingers)
# ---------------------------------------------------------------------------
# graspable_volume()'s opposability test for 3+ finger groups (below) has no
# vectorized formulation, unlike the 2-finger antipodal case (one dot
# product): it's a hemisphere test answered by solving a small linear
# program per candidate point. For any group with more than a few thousand
# surviving candidates this dominates graspable_volume()'s total time by a
# wide margin, and every point is fully independent (only reads its own row
# of U, writes its own row of opp), so it's dispatched across a worker pool
# here instead of a serial `for m in range(M)` loop.
#
# U/n_fingers/margin are stashed as plain module globals (not passed via
# Pool(initargs=...), which would pickle U once per WORKER through the task
# queue) so that forked worker processes inherit them via Linux's
# copy-on-write fork() semantics for free -- only each chunk's small
# (start, end) index range actually gets pickled through the queue.
_POOL_U: Optional[np.ndarray] = None
_POOL_N_FINGERS: Optional[int] = None
_POOL_MARGIN: Optional[float] = None

# Below this many surviving candidates, the fork/IPC overhead of spinning up
# a worker pool isn't worth it -- just run the loop serially.
_MIN_PARALLEL_OPPOSABILITY_POINTS = 2000


def _opposability_chunk(index_range: tuple[int, int]) -> tuple[int, np.ndarray]:
    """Runs in a worker process: solve the hemisphere-opposability LP for
    every point in _POOL_U[start:end], returning (start, that chunk's mask)
    so the caller can place it back at the right offset in the full array."""
    start, end = index_range
    U, n_fingers, margin = _POOL_U, _POOL_N_FINGERS, _POOL_MARGIN
    c = np.array([0.0, 0.0, 0.0, -1.0])
    bounds = [(-1, 1), (-1, 1), (-1, 1), (None, None)]
    opp_chunk = np.zeros(end - start, dtype=bool)
    for i, m in enumerate(range(start, end)):
        Um = U[m]
        A_ub = np.hstack([-Um.T, np.ones((n_fingers, 1))])
        b_ub = np.zeros(n_fingers)
        sol = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if sol.success:
            opp_chunk[i] = sol.x[3] <= margin
    return start, opp_chunk


# ---------------------------------------------------------------------------
# Graspable volume (reach + opposability)
# ---------------------------------------------------------------------------

def graspable_volume(
    finger_workspaces: dict[str, np.ndarray],
    r: float,
    mu: float = 0.5,
    res: float | None = None,
    verbose: bool = True,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Sphere-center positions the hand can both REACH and OPPOSABLY grasp.

    Replicates MATLAB's graspable_volume().

    Args:
        finger_workspaces: {finger_name: (Nk, 3) array} reachable workspace
                           point cloud per selected finger (from
                           workspace.finger_workspace()). Must have >= 2 entries.
        r:                 Sphere radius.
        mu:                Friction coefficient (default 0.5).
        res:               Candidate grid spacing. Defaults to the COARSEST
                           (largest) median nearest-neighbour spacing across
                           all the given fingers' workspaces -- using just
                           one finger's spacing is fragile whenever fingers
                           have very different point densities, since a
                           single densely-sampled finger would otherwise
                           force a needlessly fine grid over the whole
                           (possibly much larger) combined bounding box.
        verbose:           Print candidate counts after each filtering
                           stage, to help see which finger/test is actually
                           doing the filtering.
        n_workers:         Worker processes for the 3+ finger opposability
                           test's per-point linear-program solve (see
                           _opposability_chunk() above). No effect on the
                           2-finger case, which is already fully vectorized.
                           1 (the default) runs it serially.

    Returns:
        (M, 3) array of sphere-center positions achieving an opposable grasp
        (may be empty).
    """
    fields = list(finger_workspaces.keys())
    n_fingers = len(fields)
    if n_fingers < 2:
        raise ValueError("graspable_volume requires at least 2 finger workspaces")

    all_pts = np.vstack([finger_workspaces[f] for f in fields])

    if res is None:
        spacings = []
        for fname in fields:
            Wf = finger_workspaces[fname]
            tree_f = cKDTree(Wf)
            dd, _ = tree_f.query(Wf, k=2)
            nn_dist = dd[:, 1]
            nonzero = nn_dist[nn_dist > 0]
            if len(nonzero) > 0:
                spacings.append(float(np.median(nonzero)))
        # Dense joint sweeps can produce many exactly-coincident samples,
        # which would zero out the median and leave no valid grid spacing.
        res = max(spacings) if spacings else r / 4.0

    lo = all_pts.min(axis=0) - r
    hi = all_pts.max(axis=0) + r
    axes = [_colon(lo[i], res, hi[i]) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    survivors = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    if verbose:
        print(f"    graspable_volume[{'+'.join(fields)}]: res={res:.5g} m, "
              f"{len(survivors):,} candidate grid points")

    # --- REACH phase: intersect r-dilations, carrying nearest contact points ---
    nearest = np.zeros((len(survivors), 3, n_fingers))
    for k, fname in enumerate(fields):
        Wk = finger_workspaces[fname]
        tree = cKDTree(Wk)
        dmin, nn_idx = tree.query(survivors, k=1)
        keep = dmin <= r
        survivors = survivors[keep]
        nearest = nearest[keep]
        nearest[:, :, k] = Wk[nn_idx[keep]]
        if verbose:
            print(f"    after {fname} reach (r={r:.4g} m): {len(survivors):,} candidates")
        if len(survivors) == 0:
            return survivors

    # --- contact directions u_k for every survivor ---
    M = len(survivors)
    U = np.zeros((M, 3, n_fingers))
    for k in range(n_fingers):
        d = nearest[:, :, k] - survivors
        norms = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), np.finfo(float).eps)
        U[:, :, k] = d / norms

    # --- OPPOSABILITY phase ---
    if n_fingers == 2:
        # antipodal test, fully vectorized
        opp = np.sum(U[:, :, 0] * U[:, :, 1], axis=1) <= -np.cos(np.arctan(mu))
    else:
        # hemisphere test per point: is there a v with u_k . v > 0 for all k?
        margin = np.sin(np.arctan(mu))
        opp = np.zeros(M, dtype=bool)

        if n_workers > 1 and M >= _MIN_PARALLEL_OPPOSABILITY_POINTS:
            global _POOL_U, _POOL_N_FINGERS, _POOL_MARGIN
            _POOL_U, _POOL_N_FINGERS, _POOL_MARGIN = U, n_fingers, margin

            # ~4 chunks per worker so a worker that finishes its share early
            # can pick up more via imap_unordered, rather than everyone
            # waiting on however many points landed in the single slowest
            # of exactly n_workers equal-sized chunks.
            n_chunks = max(1, min(M, n_workers * 4))
            edges = np.linspace(0, M, n_chunks + 1, dtype=int)
            chunks = [(int(edges[i]), int(edges[i + 1]))
                      for i in range(n_chunks) if edges[i] < edges[i + 1]]

            with mp.get_context("fork").Pool(processes=min(n_workers, len(chunks))) as pool:
                for start, opp_chunk in pool.imap_unordered(_opposability_chunk, chunks):
                    opp[start:start + len(opp_chunk)] = opp_chunk
        else:
            c = np.array([0.0, 0.0, 0.0, -1.0])   # maximize t  (x = [v(1:3); t])
            bounds = [(-1, 1), (-1, 1), (-1, 1), (None, None)]
            for m in range(M):
                Um = U[m]                                  # (3, n_fingers)
                A_ub = np.hstack([-Um.T, np.ones((n_fingers, 1))])
                b_ub = np.zeros(n_fingers)
                sol = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
                if sol.success:
                    opp[m] = sol.x[3] <= margin   # t* <= 0 frictionless; margin adds friction

    if verbose:
        print(f"    after opposability (mu={mu:.3g}): {int(opp.sum()):,} candidates")

    return survivors[opp]


# ---------------------------------------------------------------------------
# Palm collision filter
# ---------------------------------------------------------------------------

def filter_palm_collisions(
    candidates: np.ndarray,
    palm_mesh,
    r: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    Remove candidate sphere-center points whose radius-r sphere would
    intersect the palm.

    Approximates MATLAB's workspace_palm_collision(), which tests a candidate
    sphere against the palm's solid collision geometry (collisionBox /
    collisionCapsule primitives from fitCollisionGeometry.m). This uses the
    palm's actual URDF <collision> geometry too (collision.build_collision_mesh()),
    assembled into one world-frame trimesh, and trimesh.proximity.signed_distance()
    against it: signed_distance is positive when a point is inside the mesh
    and negative outside, so a candidate's sphere overlaps the palm whenever
    its center is within r of the surface OR inside it, i.e.
    signed_distance >= -r. (An earlier version of this function tested
    against the palm's sampled VISUAL point cloud instead of its actual
    collision surface -- nearest-sampled-point distance can under-report
    collisions in gaps between samples, which signed distance against the
    real surface doesn't have.)

    A bounding-sphere broad phase (mirroring the MATLAB function's
    broad/narrow-phase split) skips the signed-distance query for candidates
    that are clearly far from the palm, since it's the more expensive test.

    Args:
        candidates: (N, 3) sphere-center positions to test, e.g. the output
                   of graspable_volume().
        palm_mesh:  World-frame trimesh.Trimesh of the palm's URDF collision
                   geometry (collision.build_collision_mesh()), or None to
                   skip filtering entirely.
        r:          Sphere radius.

    Returns:
        (M, 3) subset of candidates whose sphere does not touch the palm.
    """
    if len(candidates) == 0 or palm_mesh is None:
        return candidates

    import trimesh as _trimesh

    # --- broad phase: one bounding sphere over the whole palm ---
    center = palm_mesh.bounds.mean(axis=0)
    bounding_radius = float(np.linalg.norm(palm_mesh.vertices - center, axis=1).max())
    dist_to_palm_center = np.linalg.norm(candidates - center, axis=1)
    broad_phase = dist_to_palm_center <= (bounding_radius + r)

    # --- narrow phase: colliding if within r of (or inside) the palm surface ---
    free = np.ones(len(candidates), dtype=bool)
    idx = np.where(broad_phase)[0]
    if len(idx) > 0:
        dist = _trimesh.proximity.signed_distance(palm_mesh, candidates[idx])
        free[idx] = dist < -r

    if verbose:
        print(f"    after palm collision: {int(free.sum()):,} candidates")

    return candidates[free]


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_opposable_grasp_volume(
    finger_workspaces: dict[str, np.ndarray],
    palm_mesh,
    r: float,
    mu: float = 0.5,
    res: float | None = None,
    verbose: bool = True,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Sphere-center positions where the selected fingers can form an opposable
    grasp AND the sphere does not collide with the palm.

    Combines graspable_volume() and filter_palm_collisions().

    Args:
        finger_workspaces: {finger_name: (Nk, 3) array} per selected finger.
        palm_mesh:          World-frame trimesh.Trimesh of the palm's URDF
                            collision geometry (collision.build_collision_mesh()),
                            or None to skip palm-collision filtering.
        r:                  Sphere radius.
        mu:                 Friction coefficient (default 0.5).
        res:                Candidate grid spacing (see graspable_volume()).
        verbose:            Print candidate counts after each filtering stage.
        n_workers:          See graspable_volume() -- only matters for 3+
                            finger groups.

    Returns:
        (M, 3) array of valid sphere-center grasp positions.
    """
    candidates = graspable_volume(finger_workspaces, r, mu, res, verbose=verbose, n_workers=n_workers)
    return filter_palm_collisions(candidates, palm_mesh, r, verbose=verbose)
