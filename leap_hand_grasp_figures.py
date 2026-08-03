#!/home/sean/miniconda3/bin/python3
"""
leap_hand_grasp_figures.py
Generates an illustrated walkthrough of the opposability/manipulation
pipeline for a two-finger grasp: per-finger workspace volumes, the graspable
volume they combine into, then N_GRASP_SETS distinct concrete grasps of a
test sphere (so several can be compared and the best-looking one picked),
each shown with increasing contact-mechanics detail (friction cones, force
ellipsoids, common force-ellipsoid basis). Written for the Leap Hand's
Thumb+Index pair, but (like every other script here) actually follows
whatever hand config is active in opposability_calculation.py -- switch the
hand there, not here.

Figures (all saved under <SAVE_FOLDER>/pipeline_figures/):
    01_home_workspaces.png
        The hand at its home pose, with the two fingers' workspace alpha
        shapes (from opposability_calculation.py) rendered translucent.
        Pose-independent -- generated once, not per grasp pose.
    02_home_workspaces_graspable_volume.png
        Same as 01, plus the two fingers' graspable-volume alpha shape.
        Also generated once.
    pose_<i>/03_grasp_pose_graspable_volume.png  (i = 0..N_GRASP_SETS-1)
        The hand at grasp pose i (found by searching the graspable volume,
        same as manipulation_capacity_test.py), holding the target sphere,
        shown inside the graspable volume. No workspace shapes.
    pose_<i>/04_grasp_pose_friction_cones.png
        Same grasp, sphere now translucent, with each contact's 8-sided
        friction cone (grasp.contact_wrench()'s actual primitive directions,
        not a generic cone) rendered as a translucent pyramid.
    pose_<i>/05_grasp_pose_force_ellipsoids.png
        Same grasp, translucent sphere, with each contact's force-ellipsoid
        principal directions (as in manipulation_capacity_test.py).
    pose_<i>/06_grasp_pose_common_basis.png
        Same grasp, translucent sphere, with the two contacts' common
        force-ellipsoid basis vectors drawn at the SPHERE CENTER (not the
        contacts) -- this basis is a property of the pair of contacts
        jointly, not either one individually.
    07_search_summary_rank.png / 08_search_summary_balance.png
        OPTIONAL, only produced if <SAVE_FOLDER>/manipulation_capacity/
        <Finger1>_<Finger2>_search_summary.npz exists (from running
        manipulation_capacity_test.py with N_SPHERE_LOCATIONS > 1 first --
        a separate, much more expensive batch search over many sphere
        locations, not run by this script). The hand at home pose, with
        every searched sphere location plotted, colored by common-basis
        rank (07) or by that pose's lowest per-axis balance score, i.e.
        min(common_basis.fraction) (08).

Usage:
    ./run_leap_hand_grasp_figures.sh
    (or directly: ~/miniconda3/bin/python3 leap_hand_grasp_figures.py)

Prerequisites (same as manipulation_capacity_test.py):
    1. opposability_calculation.py, for the SAME active hand config and the
       SAME MANIPULATION_GROUP below -- this script reads its saved
       workspace/graspable-volume alpha shapes and point clouds rather than
       recomputing them.
    2. compute_workspace_transforms.py, for the SAME active hand config.
    3. (Optional, for figures 7-8 only) manipulation_capacity_test.py, run
       with N_SPHERE_LOCATIONS > 1 for the same MANIPULATION_GROUP.
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Optional

if sys.version_info < (3, 10):
    sys.exit(
        f"ERROR: Python {sys.version} detected.\n"
        "This script requires Python 3.10+ (the conda environment).\n"
        "Run with:\n"
        "  ~/miniconda3/bin/python3 leap_hand_grasp_figures.py\n"
        "or activate conda first:\n"
        "  conda activate base && python3 leap_hand_grasp_figures.py"
    )

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pinocchio as pin
from python import (
    build_hand,
    compute_fk,
    contact_point_jacobian,
    contact_wrench,
    force_closure_check,
    force_ellipsoid,
    force_ellipsoid_common_basis,
    force_ellipsoid_vector_correction,
    grasp_2_finger,
    grasp_wrench,
    load_finger_workspace_transforms,
    load_model,
    parse_mimic_and_offsets,
    sample_weighted_without_replacement,
    true_jacobian,
)

# Reuse the active hand config from opposability_calculation.py (single
# source of truth -- edit the hand there, not here), and the low-level
# rendering building blocks from manipulation_capacity_test.py rather than
# duplicating them (hand-mesh collection/rendering, sphere drawing, equal-
# axis scaling).
from opposability_calculation import (
    COLORS,
    FINGER_BODIES,
    FINGER_ID,
    URDF_FILE,
    MESH_FOLDER,
    MESH_SCALE,
    SAVE_FOLDER,
    N_MESH_SAMPLES,
    GRASP_SPHERE_RADIUS,
    FRICTION_MU,
    MU_TOR,
    OPPOSABILITY_GROUP_RADII,
    HOME_ACTUATED,
)
from manipulation_capacity_test import (
    _collect_hand_meshes,
    _draw_sphere,
    _set_axes_equal,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MANIPULATION_GROUP: tuple[str, str] = ("Thumb", "Index")
FINGER_SELECTED_BODY: dict[str, int] = {"Thumb": -1, "Index": -1}

N_TARGET_POSES = 50
ANGLE_TOLERANCE_DEG = 5.0

# How many distinct grasp poses to fully illustrate (each gets its own set
# of figures 3-6, in its own pose_<i> subfolder) -- so you can compare
# several and pick whichever looks best, rather than being stuck with
# whatever the centroid happens to produce.
N_GRASP_SETS = 5

# How many extra sphere centers (beyond the centroid, sampled from the
# graspable volume) to try while hunting for N_GRASP_SETS poses with a
# common force-ellipsoid basis -- needs to comfortably exceed N_GRASP_SETS
# since not every sphere center yields one (see the "Rank is 5" force-
# closure discussion this session). Candidates that never find a common
# basis are still kept as fallback filler if not enough good ones turn up.
N_SEARCH_RETRY = 20

# If False (default), reuse a previous run's found grasp poses from
# <out_dir>/grasp_poses_cache.pkl instead of re-searching, whenever that
# cache exists and matches N_GRASP_SETS -- so re-running this script just to
# tweak a visual constant (a color, CONE_HEIGHT, VIEW_ELEV/AZIM, ...) is
# fast. Set True to force a fresh search (needed after changing something
# the search itself depends on: MANIPULATION_GROUP, FRICTION_MU, MU_TOR,
# N_GRASP_SETS, N_TARGET_POSES, or the active hand config).
FORCE_NEW_SEARCH = False

VIS_VECTOR_LENGTH = 0.02      # display-only arrow length (m)
AXIS_COLORS = ["red", "green", "blue"]   # principal axes 0/1/2, both force
                                          # ellipsoids (fig 5) and common basis (fig 6)

WORKSPACE_ALPHA = 0.15
GRASPABLE_VOLUME_COLOR = "green"
GRASPABLE_VOLUME_ALPHA = 0.15
SPHERE_ALPHA_SOLID = 0.6        # figure 3
SPHERE_ALPHA_TRANSLUCENT = 0.35  # figures 4-6 ("now translucent")
CONE_HEIGHT = 0.02               # display-only friction-cone edge length (m)
CONE_ALPHA = 0.35

# Local override of opposability_calculation.py's COLORS, just for this
# script's hand rendering -- Middle/Pinkie aren't part of the Thumb+Index
# grasp these figures illustrate, so they're greyed out like the palm
# instead of keeping their own (green/cyan) colors, to keep visual focus on
# the two active fingers (Index recolored to yellow for the same reason:
# visually distinct from Thumb and everything greyed-out). Doesn't affect
# COLORS itself, so other scripts (opposability_calculation.py,
# manipulation_capacity_test.py) are unchanged.
PIPELINE_COLORS = {**COLORS, "Middle": "gray", "Pinkie": "gray", "Index": "yellow"}

# Show a legend on each figure -- off "for now" per request; the label=
# kwargs are left in place on every plotted element below so re-enabling is
# just flipping this back to True.
SHOW_LEGEND = False

# 3-D view angle (degrees) applied to every figure via Axes3D.view_init();
# matplotlib's own defaults (elev=30, azim=-60) unless changed here.
VIEW_ELEV = 10
VIEW_AZIM = 60


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(MANIPULATION_GROUP) != 2:
        sys.exit(f"MANIPULATION_GROUP must have exactly 2 fingers, got {MANIPULATION_GROUP!r}.")

    finger1_name, finger2_name = MANIPULATION_GROUP
    urdf_path = os.path.join(ROOT, URDF_FILE)
    mesh_folder = os.path.join(ROOT, MESH_FOLDER)
    urdf_base = os.path.dirname(urdf_path)
    group_stem = "_".join(MANIPULATION_GROUP)

    # ── 1. Parse URDF, load model, build the whole hand ──────────────────────
    print("Parsing URDF...")
    mimic_props, link_props = parse_mimic_and_offsets(urdf_path)

    print("Loading pinocchio model...")
    model, data = load_model(urdf_path)

    print("Building finger structs...")
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

    # ── 2. Home configuration (identical to opposability_calculation.py) ─────
    q_neutral = pin.neutral(model)
    if HOME_ACTUATED is not None and len(HOME_ACTUATED) == coupling.n_actuated:
        full_q_home = coupling.matrix @ np.append(HOME_ACTUATED, 1.0)
        q_home = q_neutral.copy()
        q_home[:len(full_q_home)] = full_q_home
    else:
        q_home = q_neutral.copy()

    # ── 3. Load per-finger workspace alpha shapes ─────────────────────────────
    wvol_dir = os.path.join(ROOT, SAVE_FOLDER, "workspace_volumes")
    alpha1 = _load_pickle(os.path.join(wvol_dir, f"Finger_{FINGER_ID[finger1_name]}_wkspace_alpha.pkl"))
    alpha2 = _load_pickle(os.path.join(wvol_dir, f"Finger_{FINGER_ID[finger2_name]}_wkspace_alpha.pkl"))
    if alpha1 is None or alpha2 is None:
        sys.exit(
            f"Missing {finger1_name}/{finger2_name} workspace alpha shape(s) in {wvol_dir}\n"
            f"Run opposability_calculation.py first (with the same active hand config)."
        )
    print(f"Loaded {finger1_name}/{finger2_name} workspace alpha shapes.")

    # ── 4. Load the graspable volume's alpha shape + point cloud ─────────────
    grasp_dir = os.path.join(ROOT, SAVE_FOLDER, "graspable_volume")
    grasp_alpha = _load_pickle(os.path.join(grasp_dir, f"graspable_volume_{group_stem}_alpha.pkl"))
    grasp_pts_path = os.path.join(grasp_dir, f"graspable_volume_{group_stem}_pts.npy")
    if grasp_alpha is None or not os.path.exists(grasp_pts_path):
        sys.exit(
            f"Missing graspable-volume alpha shape or points for {group_stem} in {grasp_dir}\n"
            f"Run opposability_calculation.py first (with {MANIPULATION_GROUP} included in its "
            f"OPPOSABILITY_GROUPS)."
        )
    grasp_pts = np.load(grasp_pts_path)
    print(f"Loaded {group_stem} graspable-volume alpha shape and {len(grasp_pts):,} points.")

    # ── 5. Load precomputed per-body workspace transforms ────────────────────
    transforms_dir = os.path.join(ROOT, SAVE_FOLDER, "Workspace_Transforms")

    def _load_transforms(finger_name: str):
        path = os.path.join(transforms_dir, f"{finger_name}_wkspace_transforms.npz")
        if not os.path.exists(path):
            sys.exit(
                f"Could not find {path}\n"
                f"Run compute_workspace_transforms.py first (with the same active hand config)."
            )
        return load_finger_workspace_transforms(path)

    print(f"Loading precomputed workspace transforms for {finger1_name} and {finger2_name}...")
    wt1 = _load_transforms(finger1_name)
    wt2 = _load_transforms(finger2_name)

    radius = OPPOSABILITY_GROUP_RADII.get(MANIPULATION_GROUP, GRASP_SPHERE_RADIUS)
    print(f"Using sphere radius r={radius:g} m.")

    out_dir = os.path.join(ROOT, SAVE_FOLDER, "pipeline_figures")
    os.makedirs(out_dir, exist_ok=True)

    # ── 6. Find N_GRASP_SETS distinct grasp poses to illustrate figures 3-6 with ──
    # Cached to disk (pickled -- ForceEllipsoid/CommonBasisResult are plain
    # dataclasses, so this round-trips cleanly) so that re-running this
    # script purely to tweak a visual constant (a color, CONE_HEIGHT,
    # VIEW_ELEV/AZIM, ...) doesn't have to re-run the search -- the search
    # itself doesn't depend on any of those, only on the hand/physics config
    # (MANIPULATION_GROUP, FRICTION_MU, MU_TOR, N_GRASP_SETS, ...). Set
    # FORCE_NEW_SEARCH=True (or just delete the cache file) whenever one of
    # THOSE changes and the poses actually need to be found again.
    actuated_ids_1 = sorted({aid for lnk in finger1.links for aid in lnk.actuated_joint_ids})
    actuated_ids_2 = sorted({aid for lnk in finger2.links for aid in lnk.actuated_joint_ids})

    cache_path = os.path.join(out_dir, "grasp_poses_cache.pkl")
    results = None
    if not FORCE_NEW_SEARCH and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if len(cached) == N_GRASP_SETS:
            results = cached
            print(f"Loaded {len(results)} cached grasp pose(s) from {cache_path} "
                  f"(set FORCE_NEW_SEARCH=True to search again).")
        else:
            print(f"Cached poses ({len(cached)}) don't match N_GRASP_SETS ({N_GRASP_SETS}) -- searching again.")

    if results is None:
        results = _find_grasp_poses(
            grasp_pts, radius,
            finger1, body1_idx, wt1, finger2, body2_idx, wt2,
            model, data, coupling, actuated_ids_1, actuated_ids_2,
            n_sets=N_GRASP_SETS,
        )
        if not results:
            sys.exit("No opposable grasp found anywhere tried in the graspable volume -- nothing to illustrate.")
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved {len(results)} grasp pose(s) to {cache_path} for reuse.")

    # ── Figures 1-2: home pose (pose-independent, generated once) ────────────
    compute_fk(model, data, q_home)
    home_meshes = _collect_hand_meshes(fingers, model, data, mesh_folder, urdf_base)

    _figure_1(home_meshes, alpha1, alpha2, finger1_name, finger2_name, out_dir)
    _figure_2(home_meshes, alpha1, alpha2, grasp_alpha, finger1_name, finger2_name, out_dir)

    # ── Figures 3-6: one set per grasp pose, in its own pose_<i> subfolder ───
    for i, result in enumerate(results):
        print(f"Rendering figure set {i} "
              f"({'common basis found' if result['common_basis'] is not None else 'no common basis'})...")
        pose_dir = os.path.join(out_dir, f"pose_{i}")
        os.makedirs(pose_dir, exist_ok=True)

        compute_fk(model, data, result["q"])
        grasp_meshes = _collect_hand_meshes(fingers, model, data, mesh_folder, urdf_base)

        _figure_3(grasp_meshes, grasp_alpha, result, radius, finger1_name, finger2_name, pose_dir)
        _figure_4(grasp_meshes, result, radius, finger1_name, finger2_name, pose_dir)
        _figure_5(grasp_meshes, result, radius, pose_dir)
        _figure_6(grasp_meshes, result, radius, pose_dir)

    # ── Figures 7-8: search-summary scatter plots, if available (optional --
    #      produced by a separate, much more expensive script) ───────────────
    _figure_search_summary(home_meshes, group_stem, out_dir)

    print("Done.")


def _load_pickle(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Grasp search (keeps the full CommonBasisResult, unlike
# manipulation_capacity_test.py's version, since figure 6 needs it)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_sphere_center(
    sphere_center: np.ndarray,
    radius: float,
    finger1, body1_idx: int, wt1,
    finger2, body2_idx: int, wt2,
    model, data, coupling,
    actuated_ids_1: list[int],
    actuated_ids_2: list[int],
) -> Optional[dict]:
    grasp = grasp_2_finger(
        finger1, body1_idx, wt1, finger2, body2_idx, wt2,
        sphere_center=sphere_center, sphere_radius=radius, n_target_poses=N_TARGET_POSES,
    )
    if not grasp.grasp_found:
        return None

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
            best = (j, fe1, fe2, None)

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
    return {
        "sphere_center": sphere_center,
        "q": grasp.selected_jointspace[chosen_j],
        "T1": grasp.finger1_contact_transforms[chosen_j],
        "T2": grasp.finger2_contact_transforms[chosen_j],
        "fe1": fe1,
        "fe2": fe2,
        "common_basis": basis,   # CommonBasisResult, or None
        "rank": basis.dim if basis is not None else 0,
        "fraction_sum": float(basis.fraction.sum()) if basis is not None else 0.0,
    }


def _find_grasp_poses(
    grasp_pts: np.ndarray, radius: float,
    finger1, body1_idx: int, wt1, finger2, body2_idx: int, wt2,
    model, data, coupling, actuated_ids_1: list[int], actuated_ids_2: list[int],
    n_sets: int,
) -> list[dict]:
    """
    Try the graspable volume's centroid, then (if needed) more sampled
    centers, collecting up to n_sets distinct poses that each have a common
    force-ellipsoid basis. If fewer than n_sets such poses turn up among all
    candidates tried, the remainder are padded with the best-ranked
    candidates that didn't (same "always return something" philosophy as
    _evaluate_sphere_center() itself).

    Returns up to n_sets results, best (found_common_basis, then rank, then
    fraction_sum) first.
    """
    centroid = grasp_pts.mean(axis=0)
    candidates = [centroid]
    if N_SEARCH_RETRY > 0:
        candidates.extend(list(sample_weighted_without_replacement(grasp_pts, N_SEARCH_RETRY)))

    good: list[dict] = []       # candidates with a common basis
    fallback: list[dict] = []   # candidates without one, kept as filler
    n_tried = 0

    for i, center in enumerate(candidates):
        if len(good) >= n_sets:
            break
        n_tried += 1
        tag = "centroid" if i == 0 else f"candidate {i}"
        print(f"Searching sphere center [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] ({tag})...")
        result = _evaluate_sphere_center(
            center, radius, finger1, body1_idx, wt1, finger2, body2_idx, wt2,
            model, data, coupling, actuated_ids_1, actuated_ids_2,
        )
        if result is None:
            print("  No opposable grasp at this center.")
            continue
        print(f"  rank={result['rank']}, fraction_sum={result['fraction_sum']:.4f}")

        if result["common_basis"] is not None:
            good.append(result)
        else:
            fallback.append(result)

    print(f"Found {len(good)}/{n_sets} poses with a common force-ellipsoid basis "
          f"(tried {n_tried} candidate sphere center(s)).")

    results = good[:n_sets]
    if len(results) < n_sets:
        missing = n_sets - len(results)
        fallback.sort(key=lambda r: (r["rank"], r["fraction_sum"]), reverse=True)
        print(f"  Padding with the {min(missing, len(fallback))} best fallback candidate(s) "
              f"(no common basis -- their figure 6 will note that).")
        results.extend(fallback[:missing])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Shared drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_hand_meshes(ax, hand_meshes: dict) -> None:
    """
    Same as manipulation_capacity_test.py's _render_hand_meshes(), but
    colored via this script's local PIPELINE_COLORS (Middle/Pinkie greyed
    out) instead of the shared COLORS -- kept local so it doesn't change
    that script's own figures.
    """
    for fname, mesh in hand_meshes.items():
        v = mesh.vertices
        color = PIPELINE_COLORS.get(fname, "black")
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=mesh.faces,
                        color=color, alpha=0.6, edgecolor="none")
        ax.plot([], [], color=color, marker="s", linestyle="None", markersize=8, label=fname)


def _finish_axes(ax) -> None:
    """Apply the shared view angle, optional legend, and equal-axis scaling -- call
    right before plt.tight_layout()/savefig() in every figure."""
    if SHOW_LEGEND:
        ax.legend(markerscale=5)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    _set_axes_equal(ax)


def _render_alpha_shape(ax, shape, color: str, alpha: float, label: str) -> None:
    """Render a trimesh.Trimesh alpha shape (workspace/graspable volume), translucent."""
    if shape is None:
        return
    v = shape.vertices
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=shape.faces,
                    color=color, alpha=alpha, edgecolor="none")
    # plot_trisurf's Poly3DCollection doesn't produce a usable legend handle
    # on its own -- add an invisible proxy artist instead (same pattern as
    # manipulation_capacity_test.py's _draw_sphere()/_render_hand_meshes()).
    ax.plot([], [], color=color, marker="^", linestyle="None", markersize=8, label=label)


def _mark_contacts(ax, result: dict) -> None:
    for i, T in enumerate([result["T1"], result["T2"]]):
        p = T[:3, 3]
        ax.scatter(*p, s=60, c="magenta", marker="o", zorder=5,
                   label="Contact points" if i == 0 else None)


def _draw_friction_cone(
    ax, apex: np.ndarray, T_contact: np.ndarray, mu: float, mu_tor: float,
    height: float, color: str, alpha: float, label: str,
) -> None:
    """
    Render one contact's actual 8-sided friction-cone primitive directions
    (grasp.contact_wrench()'s force sub-block, the same directions
    force_closure_check() tests against) as a translucent pyramid from the
    contact point.
    """
    W_f = contact_wrench(mu, mu_tor, "eight-sided")
    dirs_contact = W_f[:3, :].T                      # (k, 3), contact frame

    # With mu_tor > 0, contact_wrench() appends 2 pure-torsion primitives
    # (zero force, +/-mu_tor moment about the normal) -- those aren't force
    # directions and would divide-by-zero below; the cone shape itself is
    # only the force-cone edges, so drop any zero-magnitude rows first.
    force_norms = np.linalg.norm(dirs_contact, axis=1)
    dirs_contact = dirs_contact[force_norms > 1e-12]

    dirs_world = dirs_contact @ T_contact[:3, :3].T   # rotate each row into world frame
    dirs_unit = dirs_world / np.linalg.norm(dirs_world, axis=1, keepdims=True)
    tips = apex + dirs_unit * height

    n = len(dirs_unit)
    triangles = [[apex, tips[i], tips[(i + 1) % n]] for i in range(n)]
    poly = Poly3DCollection(triangles, facecolor=color, alpha=alpha, edgecolor="none")
    ax.add_collection3d(poly)
    ax.plot([], [], color=color, marker="^", linestyle="None", markersize=8, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _figure_1(home_meshes, alpha1, alpha2, name1, name2, out_dir) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, home_meshes)
    _render_alpha_shape(ax, alpha1, PIPELINE_COLORS.get(name1, "blue"), WORKSPACE_ALPHA, f"{name1} workspace")
    _render_alpha_shape(ax, alpha2, PIPELINE_COLORS.get(name2, "red"), WORKSPACE_ALPHA, f"{name2} workspace")

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"1. Home pose + {name1}/{name2} workspace volumes")
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_home_workspaces.png"), dpi=150)
    plt.show()


def _figure_2(home_meshes, alpha1, alpha2, grasp_alpha, name1, name2, out_dir) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, home_meshes)
    _render_alpha_shape(ax, alpha1, PIPELINE_COLORS.get(name1, "blue"), WORKSPACE_ALPHA, f"{name1} workspace")
    _render_alpha_shape(ax, alpha2, PIPELINE_COLORS.get(name2, "red"), WORKSPACE_ALPHA, f"{name2} workspace")
    _render_alpha_shape(ax, grasp_alpha, GRASPABLE_VOLUME_COLOR, GRASPABLE_VOLUME_ALPHA,
                        f"{name1}+{name2} graspable volume")

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"2. Home pose + workspace volumes + graspable volume")
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_home_workspaces_graspable_volume.png"), dpi=150)
    plt.show()


def _figure_3(grasp_meshes, grasp_alpha, result, radius, name1, name2, out_dir) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, grasp_meshes)
    _render_alpha_shape(ax, grasp_alpha, GRASPABLE_VOLUME_COLOR, GRASPABLE_VOLUME_ALPHA,
                        f"{name1}+{name2} graspable volume")
    _draw_sphere(ax, result["sphere_center"], radius, color="orange",
                alpha=SPHERE_ALPHA_SOLID, label="Test sphere")
    _mark_contacts(ax, result)

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"3. Grasp pose — {name1}+{name2} grasping the target sphere\nwithin the graspable volume")
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_grasp_pose_graspable_volume.png"), dpi=150)
    plt.show()


def _figure_4(grasp_meshes, result, radius, name1, name2, out_dir) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, grasp_meshes)
    _draw_sphere(ax, result["sphere_center"], radius, color="orange",
                alpha=SPHERE_ALPHA_TRANSLUCENT, label="Test sphere")
    _mark_contacts(ax, result)
    _draw_friction_cone(ax, result["T1"][:3, 3], result["T1"], FRICTION_MU, MU_TOR,
                        CONE_HEIGHT, PIPELINE_COLORS.get(name1, "blue"), CONE_ALPHA, f"{name1} friction cone")
    _draw_friction_cone(ax, result["T2"][:3, 3], result["T2"], FRICTION_MU, MU_TOR,
                        CONE_HEIGHT, PIPELINE_COLORS.get(name2, "red"), CONE_ALPHA, f"{name2} friction cone")

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"4. Grasp pose — friction cones at the {name1}/{name2} contacts")
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_grasp_pose_friction_cones.png"), dpi=150)
    plt.show()


def _figure_5(grasp_meshes, result, radius, out_dir) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, grasp_meshes)
    _draw_sphere(ax, result["sphere_center"], radius, color="orange",
                alpha=SPHERE_ALPHA_TRANSLUCENT, label="Test sphere")

    for T, fe, first in [(result["T1"], result["fe1"], True), (result["T2"], result["fe2"], False)]:
        v_corr, d_corr = force_ellipsoid_vector_correction(fe, T)
        d_max = d_corr.max()
        d_norm = d_corr / d_max if d_max > 0 else d_corr
        p = T[:3, 3]
        for axis in range(3):
            vec = v_corr[:, axis] * d_norm[axis] * VIS_VECTOR_LENGTH
            ax.quiver(*p, *vec, color=AXIS_COLORS[axis], linewidth=2,
                     label=f"axis {axis}" if first else None)
        ax.scatter(*p, s=60, c="magenta", marker="o", zorder=5,
                   label="Contact points" if first else None)

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title("5. Grasp pose — force-ellipsoid principal directions at each contact")
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_grasp_pose_force_ellipsoids.png"), dpi=150)
    plt.show()


def _figure_6(grasp_meshes, result, radius, out_dir) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _render_hand_meshes(ax, grasp_meshes)
    _draw_sphere(ax, result["sphere_center"], radius, color="orange",
                alpha=SPHERE_ALPHA_TRANSLUCENT, label="Test sphere")
    _mark_contacts(ax, result)

    basis = result["common_basis"]
    center = result["sphere_center"]
    if basis is not None and basis.dim > 0:
        max_frac = basis.fraction[:basis.dim].max()
        for j in range(basis.dim):
            scale = (basis.fraction[j] / max_frac) if max_frac > 0 else 1.0
            vec = basis.basis[:, j] * scale * VIS_VECTOR_LENGTH
            ax.quiver(*center, *vec, color=AXIS_COLORS[j], linewidth=2, label=f"common basis axis {j}")
        ax.scatter(*center, s=80, c="gold", marker="*", zorder=6, label="Sphere center")
        title = "6. Grasp pose — common force-ellipsoid basis at the sphere center"
    else:
        ax.scatter(*center, s=80, c="gold", marker="*", zorder=6, label="Sphere center")
        title = ("6. Grasp pose — no common force-ellipsoid basis found for this grasp\n"
                 "(sphere center shown, but no basis vectors to draw)")
        print("  Note: figure 6 has no common-basis vectors -- this grasp never found one; "
              "see _find_grasp_poses()'s log above.")

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(title)
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_grasp_pose_common_basis.png"), dpi=150)
    plt.show()


def _figure_search_summary(home_meshes: dict, group_stem: str, out_dir: str) -> None:
    """
    Optional figures 7-8: every sphere location manipulation_capacity_test.py
    searched (in its N_SPHERE_LOCATIONS > 1 batch mode), plotted at the
    hand's home pose and colored by common-basis rank (07) or by that
    location's lowest per-axis balance score (08) -- i.e. how close the
    weaker of the two contacts' reach is to matching the stronger one along
    the worst common-basis axis; 1.0 = perfectly matched, 0 = no basis found
    (or that axis's weaker contact contributes nothing).

    A no-op (with an explanatory message) if that batch search hasn't been
    run, since it's a separate, much more expensive script this one doesn't
    invoke itself.
    """
    summary_path = os.path.join(ROOT, SAVE_FOLDER, "manipulation_capacity", f"{group_stem}_search_summary.npz")
    if not os.path.exists(summary_path):
        print(f"Skipping figures 7-8 (search-summary scatter plots) -- {summary_path} not found.\n"
              f"  Run manipulation_capacity_test.py with N_SPHERE_LOCATIONS > 1 first to generate it.")
        return

    data = np.load(summary_path)
    points = data["points"]
    rank = data["rank"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    _render_hand_meshes(ax, home_meshes)
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=rank, cmap="viridis", vmin=0, vmax=3, s=40, zorder=5)
    fig.colorbar(sc, ax=ax, label="common-basis rank (0 = none found)")
    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"7. Search summary ({len(points)} locations) — colored by common-basis rank")
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "07_search_summary_rank.png"), dpi=150)
    plt.show()

    if "min_fraction" not in data.files:
        print(f"Skipping figure 8 (balance-score scatter) -- {summary_path} predates the "
              f"min_fraction field. Re-run manipulation_capacity_test.py's batch mode to regenerate it.")
        return

    min_fraction = data["min_fraction"]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    _render_hand_meshes(ax, home_meshes)
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=min_fraction, cmap="viridis", vmin=0, vmax=1, s=40, zorder=5)
    fig.colorbar(sc, ax=ax, label="lowest common-basis balance score (min fraction, 0-1)")
    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"8. Search summary ({len(points)} locations) — colored by lowest balance score")
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "08_search_summary_balance.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
