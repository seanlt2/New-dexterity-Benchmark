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
        (N_SPHERE_LOCATIONS > 1 only) points, grasp_found, found_common_basis,
        rank, fraction_sum, min_fraction (lowest per-axis common-basis
        balance score, 0 if no basis found) -- one row per searched sphere
        location.
    <Finger1>_<Finger2>_search_summary.png
        (N_SPHERE_LOCATIONS > 1 only) all searched locations, colored by
        rank, alongside the hand at its home/neutral pose for spatial
        context (not any particular grasp pose -- the searched locations
        span many different candidate poses, so there's no single "the"
        pose to show here the way the per-location figures do).
    <Finger1>_<Finger2>_loc<i>_pose.npz / _grasp_pose_normals.png / _force_ellipsoids.png
        (N_SPHERE_LOCATIONS > 1 only) same detail as the N==1 case above,
        for each of the best N_VISUALIZE locations (by found_common_basis,
        then rank, then fraction_sum) -- <i> is that location's index into
        the search_summary arrays.
"""

from __future__ import annotations

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
# original single-pose behavior exactly. >1 additionally samples
# (N_SPHERE_LOCATIONS - 1) more locations uniformly at random (without
# replacement) from the graspable volume's point cloud, and searches every
# one of them, reusing the same loaded hand/workspace-transform data for
# all of them -- so the fixed setup cost (parsing the URDF, sampling
# meshes, loading both fingers' workspace transforms) is paid once, not
# once per location. Location 0 is always the centroid.
N_SPHERE_LOCATIONS = 1

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

    centroid = grasp_pts.mean(axis=0)
    if N_SPHERE_LOCATIONS <= 1:
        sphere_centers = centroid[None, :]
        print(f"Testing a single grasp pose at the graspable volume's centroid: "
              f"[{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}] m")
    else:
        n_extra = min(N_SPHERE_LOCATIONS - 1, len(grasp_pts))
        extra_idx = np.random.choice(len(grasp_pts), size=n_extra, replace=False)
        extra = grasp_pts[extra_idx]
        sphere_centers = np.vstack([centroid[None, :], extra])
        print(f"Testing {len(sphere_centers)} sphere locations in the graspable volume "
              f"(location 0 = centroid, {len(sphere_centers) - 1} more sampled from it).")

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
    for i, center in enumerate(sphere_centers):
        if multi:
            print(f"[{i + 1}/{len(sphere_centers)}] sphere center "
                  f"[{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] m")
        else:
            print("Searching for an opposable grasp of the test sphere...")
        results.append(_evaluate_sphere_center(
            center, radius,
            finger1, body1_idx, wt1, finger2, body2_idx, wt2,
            model, data, coupling, actuated_ids_1, actuated_ids_2,
        ))

    out_dir = os.path.join(ROOT, SAVE_FOLDER, "manipulation_capacity")
    os.makedirs(out_dir, exist_ok=True)

    if not multi:
        result = results[0]
        if not result["grasp_found"]:
            sys.exit("No opposable grasp found at the graspable volume's centroid.")
        _save_and_visualize_one(result, radius, group_stem, out_dir, fingers, model, data, mesh_folder, urdf_base)
        print("Done.")
        return

    # ── 5. Batch summary + save ────────────────────────────────────────────
    n_found = sum(r["grasp_found"] for r in results)
    n_closure = sum(r["found_common_basis"] for r in results)
    print(f"\n{n_found}/{len(results)} locations found an opposable grasp; "
          f"{n_closure}/{len(results)} achieved force closure + a common basis.")

    summary_path = os.path.join(out_dir, f"{group_stem}_search_summary.npz")
    np.savez(
        summary_path,
        points=np.array([r["sphere_center"] for r in results]),
        grasp_found=np.array([r["grasp_found"] for r in results]),
        found_common_basis=np.array([r["found_common_basis"] for r in results]),
        rank=np.array([r["rank"] for r in results]),
        fraction_sum=np.array([r["fraction_sum"] for r in results]),
        min_fraction=np.array([r["min_fraction"] for r in results]),
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

    Returns a dict with grasp_found/found_common_basis/rank/fraction_sum
    always present, and q/T1/T2/fe1/fe2 present (non-None) whenever
    grasp_found is True.
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
            "found_common_basis": False, "rank": 0, "fraction_sum": 0.0, "min_fraction": 0.0,
            "q": None, "T1": None, "T2": None, "fe1": None, "fe2": None,
        }

    n_poses = grasp.selected_jointspace.shape[0]
    best_score: Optional[tuple[int, float]] = None
    best = None   # (j, fe1, fe2, basis)
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
        basis = force_ellipsoid_common_basis([fe1, fe2], angle_tolerance_deg=ANGLE_TOLERANCE_DEG)
        if basis is None:
            continue
        score = (basis.dim, float(basis.fraction.sum()))
        if best_score is None or score > best_score:
            best_score = score
            best = (j, fe1, fe2, basis)

    chosen_j, fe1, fe2, basis = best
    if best_score is not None:
        print(f"    Chosen pose {chosen_j}/{n_poses}: common-basis rank={best_score[0]}, "
              f"fraction_sum={best_score[1]:.4f}")
    else:
        print(f"    No candidate pose achieved force closure + a common force-ellipsoid basis -- "
              f"falling back to candidate pose {chosen_j}/{n_poses}. Its force ellipsoids are "
              f"still each contact's own true Jacobian-derived ellipsoid, just not validated "
              f"against force closure/a common basis.")

    # Lowest per-axis balance score across the common basis (1.0 = the
    # weaker contact matches the stronger one exactly on every axis; lower
    # = some axis is dominated by one contact). 0.0 when no basis was found.
    min_fraction = float(basis.fraction[:basis.dim].min()) if (basis is not None and basis.dim > 0) else 0.0

    return {
        "sphere_center": sphere_center,
        "grasp_found": True,
        "found_common_basis": best_score is not None,
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
    Scatter of every searched sphere location, colored by rank, alongside
    the hand at its home/neutral pose for spatial context.

    The hand render is a one-time cost (same order as a single detailed
    figure, see N_VISUALIZE's docstring) regardless of how many locations
    were searched -- unlike per-location detailed figures, this isn't
    repeated per point. Home/neutral pose is used rather than any one
    candidate grasp pose since the searched locations span many different
    poses; there's no single "the" pose to show here.
    """
    points = np.array([r["sphere_center"] for r in results])
    rank = np.array([r["rank"] for r in results])

    q_neutral = pin.neutral(model)
    compute_fk(model, data, q_neutral)
    hand_meshes = _collect_hand_meshes(fingers, model, data, mesh_folder, urdf_base)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, hand_meshes)

    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=rank, cmap="viridis", vmin=0, vmax=3, s=40, zorder=5)
    fig.colorbar(sc, ax=ax, label="common-basis rank (0 = none found)")

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
