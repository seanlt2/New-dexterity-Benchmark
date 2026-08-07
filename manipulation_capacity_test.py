#!/home/sean/miniconda3/bin/python3
"""
manipulation_capacity_test.py
Evaluates two-finger opposable-grasp poses of a test sphere placed at one or
more locations in a previously computed graspable volume. With
N_SPHERE_LOCATIONS == 1 (the default), it tests just the volume's centroid
and visualizes that single result two ways: the hand at the grasp pose with
each contact's surface normal, and the hand at the same pose with each
contact's force-ellipsoid principal directions. With N_SPHERE_LOCATIONS > 1,
it searches that many sphere locations (the centroid plus more sampled from
the graspable volume), reusing the same loaded hand/workspace data for all
of them, and produces a summary (rank per location, saved as one .npz plus
a lightweight scatter plot) with full hand-mesh figures rendered only for
the best N_VISUALIZE locations -- rendering one for every location isn't
needed while just sanity-checking the search, and is not cheap (mesh
rendering dominates this script's per-location wall time; see
N_VISUALIZE's docstring below).

A harness for python/manipulation_capacity.py's underlying pieces
(grasp_selection.grasp_2_finger(), grasp.grasp_wrench(),
grasp.force_closure_check(), force_ellipsoid.force_ellipsoid()) -- rather
than manipulation_capacity_region()'s batch-over-many-points loop, which
only returns a per-point rank/fraction summary and discards exactly the
contact-level detail (contact transforms, force-ellipsoid vectors) the
detailed figures need. Reuses this project's active hand config directly
from opposability_calculation.py, so there's one source of truth for which
hand/URDF is active -- edit the config block there, not here.

Usage:
    ./run_manipulation_capacity_test.sh
    (or directly: ~/miniconda3/bin/python3 manipulation_capacity_test.py)

Prerequisites:
    1. Run opposability_calculation.py first, for the SAME active hand config
       and the SAME MANIPULATION_GROUP finger pair below -- this script reads
       that run's saved graspable-volume point cloud rather than recomputing
       it:
           <SAVE_FOLDER>/graspable_volume/graspable_volume_<Finger1>_<Finger2>_pts.npy
    2. Run compute_workspace_transforms.py first, for the SAME active hand
       config -- this script loads its precomputed per-finger workspace
       transforms rather than re-sweeping them itself:
           <SAVE_FOLDER>/Workspace_Transforms/<Finger>_wkspace_transforms.npz

Outputs (relative to project root), all under <SAVE_FOLDER>/manipulation_capacity/:
    <Finger1>_<Finger2>_pose.npz
        (N_SPHERE_LOCATIONS == 1 only) sphere_center, radius, q, both
        contact transforms, both force ellipsoids (eigenvectors +
        force_scale), and whether a force-closure + common-basis pose was
        actually found.
    <Finger1>_<Finger2>_grasp_pose_normals.png / _force_ellipsoids.png
        (N_SPHERE_LOCATIONS == 1 only) see the module summary above.
    <Finger1>_<Finger2>_search_summary.npz
        (N_SPHERE_LOCATIONS > 1 only) points, grasp_found, force_closure_found,
        found_common_basis, status, rank, fraction_sum, min_fraction (lowest
        per-axis common-basis balance score, 0 if no basis found), q -- one
        row per searched sphere location. status is one of "no_grasp" (no
        candidate pose reached/opposed the sphere at all), "no_force_closure"
        (candidate poses existed but none passed force closure),
        "no_common_basis" (force closure passed for at least one pose, but
        none of those also had a common force-ellipsoid basis), or "success"
        (both). q is that location's chosen jointspace vector, but only when
        status == "success" -- every other row is NaN, since a pose that
        never passed force closure/common-basis isn't a validated grasp
        configuration worth persisting.
    <Finger1>_<Finger2>_search_summary.png
        (N_SPHERE_LOCATIONS > 1 only) all searched locations, colored by
        status (see above), alongside the hand at its home/neutral pose for
        spatial context (not any particular grasp pose -- the searched
        locations span many different candidate poses, so there's no single
        "the" pose to show here the way the per-location figures do).
    <Finger1>_<Finger2>_loc<i>_pose.npz / _grasp_pose_normals.png / _force_ellipsoids.png
        (N_SPHERE_LOCATIONS > 1 only) same detail as the N==1 case above,
        for each of the best N_VISUALIZE locations (by found_common_basis,
        then rank, then fraction_sum) -- <i> is that location's index into
        the search_summary arrays.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Optional

if sys.version_info < (3, 10):
    sys.exit(
        f"ERROR: Python {sys.version} detected.\n"
        "This script requires Python 3.10+ (the conda environment).\n"
        "Run with:\n"
        "  ~/miniconda3/bin/python3 manipulation_capacity_test.py\n"
        "or activate conda first:\n"
        "  conda activate base && python3 manipulation_capacity_test.py"
    )

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pinocchio as pin
from python import (
    build_hand,
    build_visual_mesh,
    compute_fk,
    contact_point_jacobian,
    force_closure_check,
    force_ellipsoid,
    force_ellipsoid_common_basis,
    force_ellipsoid_vector_correction,
    grasp_2_finger,
    grasp_wrench,
    load_finger_workspace_transforms,
    load_model,
    parse_mimic_and_offsets,
    true_jacobian,
)

# Reuse the active hand config, and the grasp-radius-per-group table, from
# opposability_calculation.py directly rather than duplicating it -- change
# the hand there, and this script follows automatically. That module only
# defines things at import time (no work runs until its own main() is
# called), so importing it here is side-effect-free.
from opposability_calculation import (
    COLORS,
    FINGER_BODIES,
    URDF_FILE,
    MESH_FOLDER,
    MESH_SCALE,
    SAVE_FOLDER,
    N_MESH_SAMPLES,
    GRASP_SPHERE_RADIUS,
    FRICTION_MU,
    MU_TOR,
    OPPOSABILITY_GROUP_RADII,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config specific to this script
# ─────────────────────────────────────────────────────────────────────────────

# Which two fingers to evaluate -- must be a finger pair that
# opposability_calculation.py already computed a graspable volume for (its
# OPPOSABILITY_GROUPS list), since this script reads that saved output.
MANIPULATION_GROUP: tuple[str, str] = ("Thumb", "Index")

# Which body (0-indexed into FINGER_BODIES[finger]) each finger contacts
# with -- -1 (the default) means the last/most distal body, i.e. the
# fingertip.
FINGER_SELECTED_BODY: dict[str, int] = {
    "Thumb": -1,
    "Index": -1,
}

# Max diverse candidate grasp poses of the test sphere to consider (passed
# through to grasp_selection.grasp_2_finger()); the best one (force closure
# + highest common-basis rank/fraction) is kept for visualization.
N_TARGET_POSES = 50

# Angle tolerance (deg) for the common force-ellipsoid basis, used only to
# rank candidate poses here (force_ellipsoid.force_ellipsoid_common_basis()).
ANGLE_TOLERANCE_DEG = 5.0

# How many sphere locations in the graspable volume to search. 1 (the
# default) tests just the volume's centroid, matching this script's
# original single-pose behavior exactly. >1 additionally selects
# (N_SPHERE_LOCATIONS - 1) more locations, spread evenly across the
# graspable volume's point cloud via farthest-point sampling (see
# _farthest_point_sample()), and searches every one of them, reusing the
# same loaded hand/workspace-transform data for all of them -- so the fixed
# setup cost (parsing the URDF, sampling meshes, loading both fingers'
# workspace transforms) is paid once, not once per location. Location 0 is
# the graspable-volume point closest to the centroid.
#
# Only consulted when POINTS_PER_SPHERE_LOCATION (below) is None.
N_SPHERE_LOCATIONS = 1

# Alternative to a fixed N_SPHERE_LOCATIONS: pick the number of sphere
# locations relative to the size of the loaded graspable volume instead --
# e.g. 100 means roughly 1 location per 100 graspable-volume points,
# regardless of how big that volume's point cloud actually is (which varies
# hugely across finger groups and hands, hence "roughly": there's no reason
# a fixed location count would represent a 500-point graspable volume and a
# 500,000-point one equally well). None (the default) disables this and
# falls back to the fixed N_SPHERE_LOCATIONS above.
POINTS_PER_SPHERE_LOCATION: Optional[int] = None

# How many worker processes to search sphere locations 1..N_SPHERE_LOCATIONS-1
# with in parallel (location 0, the centroid, always runs first in the main
# process -- see main()). 1 disables multiprocessing entirely (useful for
# debugging). Only consulted when N_SPHERE_LOCATIONS > 1.
#
# This machine's 20 logical CPUs are NOT 20 uniform cores: it's a 12th-gen
# Intel hybrid part (6 Performance cores w/ hyperthreading = 12 threads, + 8
# Efficiency cores = 8 threads, 14 physical cores total), and it's a laptop
# chip, so sustained all-core load risks thermal throttling. 10 was chosen
# (rather than 20) to roughly match the P-core thread count, leave the
# E-cores/other hyperthread free for the OS and this process's own main
# thread, and reduce throttling risk -- measure before assuming higher is
# strictly better on this specific machine.
N_WORKERS = 10

# How many of the searched locations (best first, by found_common_basis
# then rank then fraction_sum) get full hand-mesh figures rendered. Only
# consulted when N_SPHERE_LOCATIONS > 1. Keep this small: mesh rendering
# (build_visual_mesh() + plot_trisurf()/savefig() on the full URDF meshes)
# is the expensive part of this script per location -- roughly a minute for
# a hand with dense STL meshes -- while the search itself is only a few
# seconds per location, so N_VISUALIZE, not N_SPHERE_LOCATIONS, is what
# controls how long a batch run's visualization phase takes.
N_VISUALIZE = 3

# Display-only arrow length (m) for the surface-normal and force-ellipsoid
# vectors in the plots below -- purely a visualization scale, doesn't affect
# any saved/computed result.
VIS_VECTOR_LENGTH = 0.02

AXIS_COLORS = ["red", "green", "blue"]   # force-ellipsoid principal axes 0/1/2

# Colors for _plot_search_summary()'s per-location status scatter -- see
# _evaluate_sphere_center()'s docstring for what each status means. Ordered
# worst to best so the legend reads as a severity gradient.
STATUS_COLORS = {
    "no_grasp": "lightgray",
    "no_force_closure": "red",
    "no_common_basis": "orange",
    "success": "green",
}


def _farthest_point_sample(points: np.ndarray, n: int, seed_point: np.ndarray) -> np.ndarray:
    """
    Greedy farthest-point sampling: returns indices of `n` points from
    `points`, spread as evenly as possible across the point cloud (each pick
    maximizes the minimum distance to every point already selected).

    Used instead of uniform random sampling (the previous approach) because
    random sampling doesn't guarantee spatial coverage -- with a modest
    sample count relative to a large cloud, chance clustering can leave
    whole regions of the graspable volume unsampled while other regions get
    several nearby picks. Deterministic (no RNG): the first pick is always
    the point nearest `seed_point`, so re-running with the same inputs
    reproduces the same locations.

    O(n * len(points)) via an incrementally maintained per-point "distance
    to nearest selected point so far" array -- standard for this algorithm,
    and fast enough here since graspable-volume point clouds run at most in
    the low hundreds of thousands.
    """
    n = min(n, len(points))
    selected = np.empty(n, dtype=np.int64)

    selected[0] = int(np.argmin(np.linalg.norm(points - seed_point, axis=1)))
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)

    for i in range(1, n):
        selected[i] = int(np.argmax(min_dist))
        new_dist = np.linalg.norm(points - points[selected[i]], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return selected


def main() -> None:
    if len(MANIPULATION_GROUP) != 2:
        sys.exit(f"MANIPULATION_GROUP must have exactly 2 fingers, got {MANIPULATION_GROUP!r}.")

    finger1_name, finger2_name = MANIPULATION_GROUP
    urdf_path = os.path.join(ROOT, URDF_FILE)
    mesh_folder = os.path.join(ROOT, MESH_FOLDER)
    urdf_base = os.path.dirname(urdf_path)

    # ── 1. Load the saved graspable volume; pick sphere location(s) to test ──
    group_stem = "_".join(MANIPULATION_GROUP)
    grasp_pts_path = os.path.join(
        ROOT, SAVE_FOLDER, "graspable_volume", f"graspable_volume_{group_stem}_pts.npy"
    )
    if not os.path.exists(grasp_pts_path):
        sys.exit(
            f"Could not find {grasp_pts_path}\n"
            f"Run opposability_calculation.py first (with MANIPULATION_GROUP={MANIPULATION_GROUP} "
            f"included in its OPPOSABILITY_GROUPS) to generate it."
        )
    grasp_pts = np.load(grasp_pts_path)
    print(f"Loaded {len(grasp_pts):,} graspable-volume points from {grasp_pts_path}")

    if len(grasp_pts) == 0:
        sys.exit("Graspable volume is empty -- nothing to test.")

    if POINTS_PER_SPHERE_LOCATION is not None:
        n_locations = max(1, len(grasp_pts) // POINTS_PER_SPHERE_LOCATION)
        print(f"POINTS_PER_SPHERE_LOCATION={POINTS_PER_SPHERE_LOCATION}: "
              f"{len(grasp_pts):,} points -> {n_locations} location(s).")
    else:
        n_locations = N_SPHERE_LOCATIONS

    centroid = grasp_pts.mean(axis=0)
    if n_locations <= 1:
        sphere_centers = centroid[None, :]
        print(f"Testing a single grasp pose at the graspable volume's centroid: "
              f"[{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}] m")
    else:
        idx = _farthest_point_sample(grasp_pts, n_locations, seed_point=centroid)
        sphere_centers = grasp_pts[idx]
        print(f"Testing {len(sphere_centers)} sphere locations spread evenly across the "
              f"graspable volume via farthest-point sampling "
              f"(location 0 = the point closest to the centroid).")

    radius = OPPOSABILITY_GROUP_RADII.get(MANIPULATION_GROUP, GRASP_SPHERE_RADIUS)
    print(f"Using sphere radius r={radius:g} m (same as this group's graspable volume).")

    # ── 2. Parse URDF, load model, build fingers ──────────────────────────────
    print("Parsing URDF...")
    mimic_props, link_props = parse_mimic_and_offsets(urdf_path)

    print("Loading pinocchio model...")
    model, data = load_model(urdf_path)

    print("Building finger structs...")
    # Each body's (mu, mu_tor) feeds grasp_wrench()'s contact-wrench cone
    # below -- with contact_assumption left at [0, 0] (as
    # opposability_calculation.py's contact_specs are, since it never needs
    # this), force_closure_check() only ever sees a frictionless, single-ray
    # cone per contact, which two contacts can never satisfy (rank <= 2 of
    # the 6 needed). Use FRICTION_MU/MU_TOR here so the friction cone
    # actually matches what's configured for this evaluation -- see
    # MU_TOR's docstring in opposability_calculation.py for why a 2-finger
    # grasp needs MU_TOR > 0 (a "soft finger" contact) to ever pass force
    # closure at all, regardless of FRICTION_MU.
    contact_specs = {
        name: np.tile([FRICTION_MU, MU_TOR], (len(bodies), 1))
        for name, bodies in FINGER_BODIES.items()
    }
    fingers, coupling = build_hand(
        finger_specs=FINGER_BODIES,
        contact_specs=contact_specs,
        model=model,
        link_props=link_props,
        mimic_props=mimic_props,
        mesh_folder=mesh_folder,
        n_mesh_samples=N_MESH_SAMPLES,
        mesh_scale=MESH_SCALE,
        urdf_base=urdf_base,
    )

    finger1 = fingers[finger1_name]
    finger2 = fingers[finger2_name]
    if finger1.is_empty() or finger2.is_empty():
        sys.exit(f"{finger1_name} or {finger2_name} is not present on this hand.")

    body1_idx = FINGER_SELECTED_BODY.get(finger1_name, -1) % len(finger1.links)
    body2_idx = FINGER_SELECTED_BODY.get(finger2_name, -1) % len(finger2.links)
    print(f"  Contact bodies: {finger1_name}[{body1_idx}]={finger1.links[body1_idx].body_name}, "
          f"{finger2_name}[{body2_idx}]={finger2.links[body2_idx].body_name}")

    # ── 3. Load precomputed per-body workspace transforms for the two fingers ──
    transforms_dir = os.path.join(ROOT, SAVE_FOLDER, "Workspace_Transforms")

    def _load_transforms(finger_name: str):
        path = os.path.join(transforms_dir, f"{finger_name}_wkspace_transforms.npz")
        if not os.path.exists(path):
            sys.exit(
                f"Could not find {path}\n"
                f"Run compute_workspace_transforms.py first (with the same active "
                f"hand config) to precompute it."
            )
        return load_finger_workspace_transforms(path)

    print(f"Loading precomputed workspace transforms for {finger1_name} and {finger2_name}...")
    wt1 = _load_transforms(finger1_name)
    wt2 = _load_transforms(finger2_name)
    print(f"  {finger1_name}: {wt1.jointspace.shape[0]:,} poses")
    print(f"  {finger2_name}: {wt2.jointspace.shape[0]:,} poses")

    actuated_ids_1 = sorted({aid for lnk in finger1.links for aid in lnk.actuated_joint_ids})
    actuated_ids_2 = sorted({aid for lnk in finger2.links for aid in lnk.actuated_joint_ids})

    # ── 4. Search + evaluate every sphere location ────────────────────────────
    # All of steps 1-3 above (URDF parse, mesh sampling, workspace-transform
    # loading) ran once, regardless of how many locations are searched here.
    multi = len(sphere_centers) > 1
    results = []

    if not multi:
        print("Searching for an opposable grasp of the test sphere...")
        results.append(_evaluate_sphere_center(
            sphere_centers[0], radius,
            finger1, body1_idx, wt1, finger2, body2_idx, wt2,
            model, data, coupling, actuated_ids_1, actuated_ids_2,
        ))
    else:
        # Location 0 (centroid) always runs synchronously here first, for two
        # reasons: it's needed either way, and -- more importantly -- this
        # call is what lazily builds and caches each finger's broad-phase
        # KD-tree (grasp_selection._broad_phase_tree(), ~6s one-time cost per
        # finger). Doing that BEFORE the worker pool forks means every worker
        # inherits the already-built trees via copy-on-write; forking first
        # would make each worker independently rebuild its own copy.
        print(f"[1/{len(sphere_centers)}] sphere center "
              f"[{sphere_centers[0][0]:.4f}, {sphere_centers[0][1]:.4f}, {sphere_centers[0][2]:.4f}] m")
        results.append(_evaluate_sphere_center(
            sphere_centers[0], radius,
            finger1, body1_idx, wt1, finger2, body2_idx, wt2,
            model, data, coupling, actuated_ids_1, actuated_ids_2,
        ))

        remaining = sphere_centers[1:]
        if len(remaining) > 0 and N_WORKERS > 1:
            n_workers = min(N_WORKERS, len(remaining))
            print(f"Searching remaining {len(remaining)} location(s) across {n_workers} worker process(es)...")

            global _POOL_RADIUS, _POOL_FINGER1, _POOL_BODY1_IDX, _POOL_WT1
            global _POOL_FINGER2, _POOL_BODY2_IDX, _POOL_WT2, _POOL_MODEL
            global _POOL_DATA, _POOL_COUPLING, _POOL_ACTUATED_IDS_1, _POOL_ACTUATED_IDS_2
            _POOL_RADIUS = radius
            _POOL_FINGER1, _POOL_BODY1_IDX, _POOL_WT1 = finger1, body1_idx, wt1
            _POOL_FINGER2, _POOL_BODY2_IDX, _POOL_WT2 = finger2, body2_idx, wt2
            _POOL_MODEL, _POOL_DATA = model, data
            _POOL_COUPLING = coupling
            _POOL_ACTUATED_IDS_1, _POOL_ACTUATED_IDS_2 = actuated_ids_1, actuated_ids_2

            with mp.get_context("fork").Pool(
                processes=n_workers, initializer=_pool_worker_init,
            ) as pool:
                # chunksize=1: per-location cost varies enormously (measured
                # ~11K to ~4M broad-phase candidates depending on location --
                # a >350x range), so a small chunksize lets a worker that
                # finishes an easy location immediately pick up the next one,
                # rather than sitting idle while another worker chews through
                # a pre-assigned batch that happened to be expensive.
                for n_done, result in enumerate(
                    pool.imap_unordered(_pool_worker, remaining, chunksize=1), start=2
                ):
                    print(f"[{n_done}/{len(sphere_centers)}] done")
                    results.append(result)
        else:
            for i, center in enumerate(remaining, start=2):
                print(f"[{i}/{len(sphere_centers)}] sphere center "
                      f"[{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] m")
                results.append(_evaluate_sphere_center(
                    center, radius,
                    finger1, body1_idx, wt1, finger2, body2_idx, wt2,
                    model, data, coupling, actuated_ids_1, actuated_ids_2,
                ))

    out_dir = os.path.join(ROOT, SAVE_FOLDER, "manipulation_capacity")
    os.makedirs(out_dir, exist_ok=True)

    if not multi:
        result = results[0]
        if not result["found_common_basis"]:
            if result["status"] == "no_grasp":
                sys.exit("No opposable grasp found at the graspable volume's centroid: no "
                         "candidate pose reached/opposed the sphere at all.")
            elif result["status"] == "no_force_closure":
                sys.exit("Found candidate grasp poses at the centroid, but none satisfied "
                         "force closure.")
            else:
                sys.exit("Found force-closure poses at the centroid, but none had a common "
                         "force-ellipsoid basis.")
        _save_and_visualize_one(result, radius, group_stem, out_dir, fingers, model, data, mesh_folder, urdf_base)
        print("Done.")
        return

    # ── 5. Batch summary + save ────────────────────────────────────────────
    n_success = sum(r["status"] == "success" for r in results)
    n_no_basis = sum(r["status"] == "no_common_basis" for r in results)
    n_no_closure = sum(r["status"] == "no_force_closure" for r in results)
    n_no_grasp = sum(r["status"] == "no_grasp" for r in results)
    print(f"\n{len(results)} location(s) searched:")
    print(f"  {n_success} succeeded (force closure + common force basis)")
    print(f"  {n_no_basis} found force closure but no common force basis")
    print(f"  {n_no_closure} found candidate poses but none satisfied force closure")
    print(f"  {n_no_grasp} found no candidate grasp pose at all")

    # q is only meaningful for a validated ("success") pose -- see
    # _evaluate_sphere_center()'s docstring -- so every other row is left as
    # NaN rather than persisting a fallback pose that never passed force
    # closure/common-basis as if it were one.
    q_matrix = np.full((len(results), model.nq), np.nan)
    for i, r in enumerate(results):
        if r["status"] == "success":
            q_matrix[i] = r["q"]

    summary_path = os.path.join(out_dir, f"{group_stem}_search_summary.npz")
    np.savez(
        summary_path,
        points=np.array([r["sphere_center"] for r in results]),
        grasp_found=np.array([r["grasp_found"] for r in results]),
        force_closure_found=np.array([r["force_closure_found"] for r in results]),
        found_common_basis=np.array([r["found_common_basis"] for r in results]),
        status=np.array([r["status"] for r in results]),
        rank=np.array([r["rank"] for r in results]),
        fraction_sum=np.array([r["fraction_sum"] for r in results]),
        min_fraction=np.array([r["min_fraction"] for r in results]),
        q=q_matrix,
    )
    print(f"Saved {summary_path}")
    _plot_search_summary(results, out_dir, group_stem, fingers, model, data, mesh_folder, urdf_base)

    # ── 6. Full hand-mesh figures for the best N_VISUALIZE locations ─────────
    found_indices = [i for i, r in enumerate(results) if r["grasp_found"]]
    ranked = sorted(
        found_indices,
        key=lambda i: (results[i]["found_common_basis"], results[i]["rank"], results[i]["fraction_sum"]),
        reverse=True,
    )
    n_vis = min(N_VISUALIZE, len(ranked))
    print(f"Rendering detailed figures for the best {n_vis} location(s)...")
    for rank_i, i in enumerate(ranked[:n_vis]):
        print(f"  [{rank_i}] location {i}:")
        loc_stem = f"{group_stem}_loc{i}"
        _save_and_visualize_one(results[i], radius, loc_stem, out_dir, fingers, model, data, mesh_folder, urdf_base)
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Worker-pool search (see N_WORKERS)
# ─────────────────────────────────────────────────────────────────────────────
# main() forks N_WORKERS worker processes to search sphere locations
# 1..N_SPHERE_LOCATIONS-1 in parallel (location 0, the centroid, always runs
# synchronously in the main process first -- see main()). The big read-only
# search inputs (model, both fingers' FingerSet/workspace-transform data,
# etc.) are stashed in these module-level globals rather than passed as
# per-task arguments: multiprocessing.Pool pickles each task's function and
# arguments through an internal queue regardless of start method, so passing
# the ~hundreds-of-MB workspace transforms per task would re-serialize them
# for every one of the N_SPHERE_LOCATIONS-1 tasks. Setting them as globals
# before the Pool is created means each forked worker inherits them for free
# via Linux's copy-on-write fork() semantics instead -- only the small
# per-task sphere_center (3 floats) actually gets pickled through the queue.
_POOL_RADIUS: Optional[float] = None
_POOL_FINGER1 = None
_POOL_BODY1_IDX: Optional[int] = None
_POOL_WT1 = None
_POOL_FINGER2 = None
_POOL_BODY2_IDX: Optional[int] = None
_POOL_WT2 = None
_POOL_MODEL = None
_POOL_DATA = None
_POOL_COUPLING = None
_POOL_ACTUATED_IDS_1: Optional[list[int]] = None
_POOL_ACTUATED_IDS_2: Optional[list[int]] = None


def _pool_worker_init() -> None:
    """
    Runs once per worker process at startup. Limits each worker's own BLAS
    thread pool to 1 -- otherwise every worker's numpy/scipy calls would
    ALSO try to spawn threads across all cores, oversubscribing this
    machine's 14 physical cores roughly N_WORKERS-fold on top of the
    N_WORKERS processes themselves. (threadpoolctl found no controllable
    BLAS thread pool in this environment when checked, i.e. this is
    currently a no-op safety net rather than a fix for an observed problem
    here -- kept in case that ever changes, e.g. a different numpy/BLAS
    build.)
    """
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass


def _pool_worker(center: np.ndarray) -> dict:
    """Runs in a forked worker process -- evaluates one sphere center using
    the shared search inputs stashed in the _POOL_* globals above."""
    return _evaluate_sphere_center(
        center, _POOL_RADIUS,
        _POOL_FINGER1, _POOL_BODY1_IDX, _POOL_WT1,
        _POOL_FINGER2, _POOL_BODY2_IDX, _POOL_WT2,
        _POOL_MODEL, _POOL_DATA, _POOL_COUPLING,
        _POOL_ACTUATED_IDS_1, _POOL_ACTUATED_IDS_2,
    )


def _evaluate_sphere_center(
    sphere_center: np.ndarray,
    radius: float,
    finger1, body1_idx: int, wt1,
    finger2, body2_idx: int, wt2,
    model, data, coupling,
    actuated_ids_1: list[int],
    actuated_ids_2: list[int],
) -> dict:
    """
    Search for opposable grasps of a test sphere at sphere_center, then
    evaluate each candidate pose and keep the best (force closure + highest
    common-basis rank/fraction) -- mirroring
    manipulation_capacity.manipulation_capacity_region()'s selection, but,
    unlike that function, keeping the winning pose's full contact detail
    (transforms, force ellipsoids) instead of just a summary.

    Returns a dict with grasp_found/force_closure_found/found_common_basis/
    status/rank/fraction_sum always present, and q/T1/T2/fe1/fe2 present
    (non-None) whenever grasp_found is True -- q is the chosen candidate
    pose's jointspace vector regardless of status, for this function's own
    internal callers (e.g. rendering the "best available" pose even when no
    location fully succeeds), but callers persisting results to disk should
    gate on status == "success" instead (see main()'s search_summary saving,
    per the "only store q for common-basis poses" requirement this was
    built for).

    status is one of:
      "no_grasp"          -- no candidate pose reached/opposed the sphere at
                             all (grasp_2_finger found nothing).
      "no_force_closure"  -- candidate poses existed, but none passed force
                             closure (the "grasping criteria").
      "no_common_basis"   -- force closure passed for at least one pose, but
                             none of those also had a common force-ellipsoid
                             basis.
      "success"           -- both force closure and a common basis were
                             found for at least one pose.
    These are deliberately distinct failure modes: "no_force_closure" means
    the contacts themselves can't resist arbitrary disturbance wrenches
    (a geometric/frictional fact about the two contact points), while
    "no_common_basis" means force closure is fine but the two fingers can't
    be independently actuated along a shared set of directions (a
    manipulability property, checked only once closure already passed).
    """
    grasp = grasp_2_finger(
        finger1, body1_idx, wt1,
        finger2, body2_idx, wt2,
        sphere_center=sphere_center, sphere_radius=radius,
        n_target_poses=N_TARGET_POSES,
    )
    if not grasp.grasp_found:
        return {
            "sphere_center": sphere_center, "grasp_found": False,
            "force_closure_found": False, "found_common_basis": False,
            "status": "no_grasp",
            "rank": 0, "fraction_sum": 0.0, "min_fraction": 0.0,
            "q": None, "T1": None, "T2": None, "fe1": None, "fe2": None,
        }

    n_poses = grasp.selected_jointspace.shape[0]
    best_score: Optional[tuple[int, float]] = None
    best = None   # (j, fe1, fe2, basis)
    any_closure_pass = False
    for j in range(n_poses):
        q = grasp.selected_jointspace[j]
        compute_fk(model, data, q)

        G1 = grasp_wrench(finger1, body1_idx, grasp.finger1_contact_transforms[j], sphere_center, np.eye(3))
        G2 = grasp_wrench(finger2, body2_idx, grasp.finger2_contact_transforms[j], sphere_center, np.eye(3))
        closure = force_closure_check(np.hstack([G1, G2]))

        J1 = contact_point_jacobian(model, data, q, finger1.links[body1_idx].body_name,
                                     grasp.finger1_contact_transforms[j][:3, 3])
        true_J1 = true_jacobian(J1, coupling.matrix, actuated_ids_1)
        J2 = contact_point_jacobian(model, data, q, finger2.links[body2_idx].body_name,
                                     grasp.finger2_contact_transforms[j][:3, 3])
        true_J2 = true_jacobian(J2, coupling.matrix, actuated_ids_2)

        fe1 = force_ellipsoid(true_J1)
        fe2 = force_ellipsoid(true_J2)

        if best is None:
            best = (j, fe1, fe2, None)   # fallback if nothing below ever qualifies

        if closure != "Pass":
            continue
        any_closure_pass = True
        basis = force_ellipsoid_common_basis([fe1, fe2], angle_tolerance_deg=ANGLE_TOLERANCE_DEG)
        if basis is None:
            continue
        score = (basis.dim, float(basis.fraction.sum()))
        if best_score is None or score > best_score:
            best_score = score
            best = (j, fe1, fe2, basis)

    chosen_j, fe1, fe2, basis = best
    found_common_basis = best_score is not None
    if found_common_basis:
        status = "success"
        print(f"    Chosen pose {chosen_j}/{n_poses}: common-basis rank={best_score[0]}, "
              f"fraction_sum={best_score[1]:.4f}")
    elif any_closure_pass:
        status = "no_common_basis"
        print(f"    {n_poses} candidate pose(s) checked: force closure passed for at least one, "
              f"but none had a common force-ellipsoid basis -- falling back to candidate pose "
              f"{chosen_j}/{n_poses} for internal use. Its force ellipsoids are still each "
              f"contact's own true Jacobian-derived ellipsoid, just not validated against a "
              f"common basis.")
    else:
        status = "no_force_closure"
        print(f"    {n_poses} candidate pose(s) checked: none passed force closure -- falling "
              f"back to candidate pose {chosen_j}/{n_poses} for internal use.")

    # Lowest per-axis balance score across the common basis (1.0 = the
    # weaker contact matches the stronger one exactly on every axis; lower
    # = some axis is dominated by one contact). 0.0 when no basis was found.
    min_fraction = float(basis.fraction[:basis.dim].min()) if (basis is not None and basis.dim > 0) else 0.0

    return {
        "sphere_center": sphere_center,
        "grasp_found": True,
        "force_closure_found": any_closure_pass,
        "found_common_basis": found_common_basis,
        "status": status,
        "rank": best_score[0] if best_score is not None else 0,
        "fraction_sum": best_score[1] if best_score is not None else 0.0,
        "min_fraction": min_fraction,
        "q": grasp.selected_jointspace[chosen_j],
        "T1": grasp.finger1_contact_transforms[chosen_j],
        "T2": grasp.finger2_contact_transforms[chosen_j],
        "fe1": fe1,
        "fe2": fe2,
    }


def _save_and_visualize_one(
    result: dict,
    radius: float,
    stem: str,
    out_dir: str,
    fingers: dict,
    model, data,
    mesh_folder: str,
    urdf_base: str,
) -> None:
    """Save one _evaluate_sphere_center() result and render its two hand-mesh figures."""
    sphere_center = result["sphere_center"]
    q, T1, T2, fe1, fe2 = result["q"], result["T1"], result["T2"], result["fe1"], result["fe2"]

    out_path = os.path.join(out_dir, f"{stem}_pose.npz")
    np.savez(
        out_path,
        sphere_center=sphere_center,
        radius=radius,
        q=q,
        finger1_contact_transform=T1,
        finger2_contact_transform=T2,
        finger1_eigenvectors=fe1.eigenvectors,
        finger1_force_scale=fe1.force_scale,
        finger2_eigenvectors=fe2.eigenvectors,
        finger2_force_scale=fe2.force_scale,
        found_common_basis=result["found_common_basis"],
    )
    print(f"    Saved {out_path}")

    compute_fk(model, data, q)
    hand_meshes = _collect_hand_meshes(fingers, model, data, mesh_folder, urdf_base)

    _plot_grasp_pose_normals(hand_meshes, T1, T2, sphere_center, radius, out_dir, stem)
    _plot_grasp_pose_force_ellipsoids(hand_meshes, T1, T2, fe1, fe2, sphere_center, radius, out_dir, stem)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_search_summary(
    results: list[dict],
    out_dir: str,
    group_stem: str,
    fingers: dict,
    model, data,
    mesh_folder: str,
    urdf_base: str,
) -> None:
    """
    Scatter of every searched sphere location, colored by status (see
    _evaluate_sphere_center()'s docstring for the four categories), alongside
    the hand at its home/neutral pose for spatial context.

    The hand render is a one-time cost (same order as a single detailed
    figure, see N_VISUALIZE's docstring) regardless of how many locations
    were searched -- unlike per-location detailed figures, this isn't
    repeated per point. Home/neutral pose is used rather than any one
    candidate grasp pose since the searched locations span many different
    poses; there's no single "the" pose to show here.
    """
    points = np.array([r["sphere_center"] for r in results])
    status = [r["status"] for r in results]

    q_neutral = pin.neutral(model)
    compute_fk(model, data, q_neutral)
    hand_meshes = _collect_hand_meshes(fingers, model, data, mesh_folder, urdf_base)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, hand_meshes)

    for label, color in STATUS_COLORS.items():
        mask = np.array([s == label for s in status])
        if not np.any(mask):
            continue
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                  c=color, s=40, zorder=5, label=f"{label} ({int(mask.sum())})")

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"Sphere-location search summary — {group_stem.replace('_', '+')} "
                 f"({len(points)} locations)\n(hand shown at home/neutral pose, for context only)")
    # loc="best" (the default) searches for the placement that overlaps the
    # least plotted data -- an O(n) collision check against every artist on
    # the axes, which becomes extremely expensive with a dense hand mesh or
    # a large scatter (measured: ~130s of a ~160s figure-render on a 5-point
    # search summary, all in legend placement, none of it affecting any
    # saved/computed result). A fixed corner is just as readable here.
    ax.legend(markerscale=5, loc="upper left")
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{group_stem}_search_summary.png"), dpi=150)
    plt.show()


def _collect_hand_meshes(fingers: dict, model, data, mesh_folder: str, urdf_base: str) -> dict[str, object]:
    """World-frame URDF visual mesh for every non-empty finger, at whatever pose `data` currently holds."""
    hand_meshes: dict[str, object] = {}
    for fname, fs in fingers.items():
        if fs.is_empty():
            continue
        mesh = build_visual_mesh(fs, model, data, mesh_folder, urdf_base)
        if mesh is not None:
            hand_meshes[fname] = mesh
    return hand_meshes


def _draw_sphere(
    ax, center: np.ndarray, radius: float,
    color: str = "orange", alpha: float = 0.35, label: str | None = None,
) -> None:
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:14j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True, zorder=4)
    ax.plot_wireframe(x, y, z, color=color, alpha=min(alpha * 1.5, 0.7), linewidth=0.4, zorder=4)
    if label is not None:
        # plot_surface()/plot_wireframe() don't produce a usable legend
        # handle on their own -- add an invisible proxy artist so the sphere
        # still gets a legend entry.
        ax.plot([], [], color=color, marker="o", linestyle="None", markersize=8, label=label)


def _render_hand_meshes(ax, hand_meshes: dict[str, object]) -> None:
    for fname, mesh in hand_meshes.items():
        v = mesh.vertices
        color = COLORS.get(fname, "black")
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=mesh.faces,
                        color=color, alpha=0.6, edgecolor="none")
        # plot_trisurf's Poly3DCollection doesn't produce a usable legend
        # handle on its own (same issue as _draw_sphere's plot_surface) --
        # add an invisible proxy artist instead.
        ax.plot([], [], color=color, marker="s", linestyle="None", markersize=8, label=fname)


def _plot_grasp_pose_normals(
    hand_meshes: dict[str, object],
    T1: np.ndarray,
    T2: np.ndarray,
    sphere_center: np.ndarray,
    radius: float,
    out_dir: str,
    group_stem: str,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, hand_meshes)
    _draw_sphere(ax, sphere_center, radius, label="Test sphere")

    for T, label in [(T1, "Contact normal"), (T2, None)]:
        p, n = T[:3, 3], T[:3, 2]   # contact position, outward world-frame normal
        ax.quiver(*p, *(n * VIS_VECTOR_LENGTH), color="black", linewidth=2, label=label)
        ax.scatter(*p, s=60, c="magenta", marker="o", zorder=5,
                   label="Contact points" if label else None)

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"Grasp pose + contact normals — {group_stem.replace('_', '+')}")
    # loc="best" (the default) searches for the placement that overlaps the
    # least plotted data -- an O(n) collision check against every artist on
    # the axes, which becomes extremely expensive with a dense hand mesh or
    # a large scatter (measured: ~130s of a ~160s figure-render on a 5-point
    # search summary, all in legend placement, none of it affecting any
    # saved/computed result). A fixed corner is just as readable here.
    ax.legend(markerscale=5, loc="upper left")
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{group_stem}_grasp_pose_normals.png"), dpi=150)
    plt.show()


def _plot_grasp_pose_force_ellipsoids(
    hand_meshes: dict[str, object],
    T1: np.ndarray,
    T2: np.ndarray,
    fe1,
    fe2,
    sphere_center: np.ndarray,
    radius: float,
    out_dir: str,
    group_stem: str,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, hand_meshes)
    _draw_sphere(ax, sphere_center, radius, label="Test sphere")

    for T, fe, first in [(T1, fe1, True), (T2, fe2, False)]:
        v_corr, d_corr = force_ellipsoid_vector_correction(fe, T)
        d_max = d_corr.max()
        # Normalize each ellipsoid to its OWN greatest-magnitude axis --
        # display-only, so all three axes are visible at a comparable scale
        # regardless of the ellipsoid's actual (possibly tiny or rank-
        # deficient) force_scale values.
        d_norm = d_corr / d_max if d_max > 0 else d_corr
        p = T[:3, 3]
        for axis in range(3):
            vec = v_corr[:, axis] * d_norm[axis] * VIS_VECTOR_LENGTH
            ax.quiver(*p, *vec, color=AXIS_COLORS[axis], linewidth=2,
                     label=f"axis {axis}" if first else None)
        ax.scatter(*p, s=60, c="magenta", marker="o", zorder=5,
                   label="Contact points" if first else None)

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"Force ellipsoids at contact — {group_stem.replace('_', '+')}\n"
                 f"(each normalized to its own largest-magnitude axis, for display only)")
    # loc="best" (the default) searches for the placement that overlaps the
    # least plotted data -- an O(n) collision check against every artist on
    # the axes, which becomes extremely expensive with a dense hand mesh or
    # a large scatter (measured: ~130s of a ~160s figure-render on a 5-point
    # search summary, all in legend placement, none of it affecting any
    # saved/computed result). A fixed corner is just as readable here.
    ax.legend(markerscale=5, loc="upper left")
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{group_stem}_force_ellipsoids.png"), dpi=150)
    plt.show()


def _set_axes_equal(ax) -> None:
    """Force equal scale on all three axes of a 3-D plot (see opposability_calculation.py)."""
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    half_range = max(x1 - x0, y1 - y0, z1 - z0) / 2
    xm, ym, zm = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    ax.set_xlim3d(xm - half_range, xm + half_range)
    ax.set_ylim3d(ym - half_range, ym + half_range)
    ax.set_zlim3d(zm - half_range, zm + half_range)


if __name__ == "__main__":
    main()
