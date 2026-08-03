#!/home/sean/miniconda3/bin/python3
"""
hand_dimensions_test.py
Reports physical-dimension statistics for the active hand config (see
opposability_calculation.py), as a first step toward normalizing the
opposability sphere radius (GRASP_SPHERE_RADIUS / OPPOSABILITY_GROUP_RADII)
against the hand's own scale instead of picking it by hand per hand model.

For every body in FINGER_BODIES:
  1. Samples surface points via mesh_utils_2.mesh_to_points_2 (density-based
     sampling -- see python/mesh_utils_2.py).
  2. Fits the smallest-radius cylinder, over all possible axis orientations,
     that encompasses all of that body's sampled points.

Then, per finger (every FINGER_BODIES key except "Palm", which is the
non-finger base/palm group):
  3. Reports the average of its bodies' cylinder radii.
  4. Locates the finger's first joint -- the joint attaching its first
     (most-proximal) body to the rest of the hand -- at the home pose, and
     reports the smallest, greatest, and average pairwise distance between
     different fingers' first-joint positions.

Usage:
    ./run_hand_dimensions_test.sh
    (or directly: ~/miniconda3/bin/python3 hand_dimensions_test.py)

Outputs:
    <SAVE_FOLDER>/hand_dimensions/hand_dimensions.json
"""

from __future__ import annotations

import itertools
import json
import os
import sys

if sys.version_info < (3, 10):
    sys.exit(
        f"ERROR: Python {sys.version} detected.\n"
        "This script requires Python 3.10+ (the conda environment).\n"
        "Run with:\n"
        "  ~/miniconda3/bin/python3 hand_dimensions_test.py\n"
        "or activate conda first:\n"
        "  conda activate base && python3 hand_dimensions_test.py"
    )

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pinocchio as pin
from python import (
    build_coupling_matrix,
    compute_fk,
    load_model,
    mesh_to_points_2,
    parse_mimic_and_offsets,
)
# Reused rather than duplicated -- same mesh-path resolution rules (package://
# URIs, relative-to-URDF paths, bare filenames) already relied on by
# finger.load_meshes() and collision.build_collision_mesh().
from python.finger import _resolve_mesh_path

# Reuse the active hand config directly from opposability_calculation.py --
# change the hand there, and this script follows automatically. That module
# only defines things at import time (no work runs until its own main() is
# called), so importing it here is side-effect-free.
from opposability_calculation import (
    FINGER_BODIES,
    URDF_FILE,
    MESH_FOLDER,
    MESH_SCALE,
    SAVE_FOLDER,
    HOME_ACTUATED,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config specific to this script
# ─────────────────────────────────────────────────────────────────────────────

# Surface-sample density passed to mesh_to_points_2, in points per unit area
# in the mesh file's own coordinate units (see mesh_utils_2.py) -- NOT
# affected by MESH_SCALE. Tuned so this project's meter-scale finger-link
# meshes yield a few hundred to a couple thousand points per body; the
# per-body point count is printed below so this can be retuned per hand.
POINTS_PER_AREA = 500_000.0

# Candidate axis directions sampled on the unit sphere before local
# refinement, when fitting each body's smallest enclosing cylinder.
N_CANDIDATE_DIRECTIONS = 300


# ─────────────────────────────────────────────────────────────────────────────
# Smallest enclosing cylinder (arbitrary axis orientation)
# ─────────────────────────────────────────────────────────────────────────────

def _fibonacci_sphere(n: int) -> np.ndarray:
    """n roughly-uniform unit directions on the sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    golden_angle = np.pi * (1 + 5 ** 0.5)
    theta = golden_angle * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def smallest_enclosing_cylinder(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Fit the smallest-radius cylinder (any axis orientation) that encompasses
    every point in `points`.

    For a fixed axis direction, the minimal enclosing-cylinder radius is just
    the largest perpendicular distance from any point to the axis line
    through the centroid -- so this reduces to a 2-parameter search (axis
    direction) minimizing that radius. Solved by sampling candidate
    directions on the sphere (plus the PCA principal axis, a good initial
    guess for elongated link geometry), then locally refining the best one.

    Args:
        points: (N, 3) points, in any consistent frame (result is rigid-frame
                invariant).

    Returns:
        (radius, axis, centroid) -- axis is a unit vector, centroid is (3,).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    def radius_for_axis(axis: np.ndarray) -> float:
        axis = axis / np.linalg.norm(axis)
        proj = centered @ axis
        perp = centered - np.outer(proj, axis)
        return float(np.sqrt((perp ** 2).sum(axis=1)).max())

    # PCA principal axis -- a strong initial guess since finger/palm links
    # are elongated along their own length.
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    candidates = [eigvecs[:, -1]]
    candidates.extend(_fibonacci_sphere(N_CANDIDATE_DIRECTIONS))

    radii = [radius_for_axis(a) for a in candidates]
    best_idx = int(np.argmin(radii))
    best_axis = np.asarray(candidates[best_idx])
    best_radius = radii[best_idx]

    # Local polish in spherical coordinates around the best candidate.
    theta0 = np.arccos(np.clip(best_axis[2], -1.0, 1.0))
    phi0 = np.arctan2(best_axis[1], best_axis[0])

    def _obj(params: np.ndarray) -> float:
        theta, phi = params
        axis = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
        return radius_for_axis(axis)

    res = minimize(_obj, x0=[theta0, phi0], method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 300})
    if res.fun < best_radius:
        theta, phi = res.x
        best_axis = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
        best_radius = float(res.fun)

    return best_radius, best_axis, centroid


# ─────────────────────────────────────────────────────────────────────────────
# Joint position
# ─────────────────────────────────────────────────────────────────────────────

def _first_joint_position(model, data, body_name: str) -> np.ndarray:
    """World-frame position of the joint attaching body_name to its parent."""
    fid = model.getFrameId(body_name)
    joint_id_pin = model.frames[fid].parent
    return np.array(data.oMi[joint_id_pin].translation)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    urdf_path = os.path.join(ROOT, URDF_FILE)
    mesh_folder = os.path.join(ROOT, MESH_FOLDER)
    urdf_base = os.path.dirname(urdf_path)

    print("Parsing URDF...")
    mimic_props, link_props = parse_mimic_and_offsets(urdf_path)
    link_prop_by_name = {lp.body: lp for lp in link_props}

    print("Loading pinocchio model...")
    model, data = load_model(urdf_path)

    # ── Home configuration (identical to opposability_calculation.py) ────────
    non_fix_joint_names = list(model.names[1:])
    coupling = build_coupling_matrix(non_fix_joint_names, mimic_props)
    q_neutral = pin.neutral(model)
    if HOME_ACTUATED is not None and len(HOME_ACTUATED) == coupling.n_actuated:
        full_q_home = coupling.matrix @ np.append(HOME_ACTUATED, 1.0)
        q_home = q_neutral.copy()
        q_home[:len(full_q_home)] = full_q_home
    else:
        q_home = q_neutral.copy()
    compute_fk(model, data, q_home)

    # ── 1 & 2. Per-body surface sampling + smallest enclosing cylinder ───────
    print(f"\nFitting smallest enclosing cylinders (points_per_area={POINTS_PER_AREA:g})...")
    body_radius: dict[str, dict[str, float]] = {}   # finger -> body -> radius
    body_npts: dict[str, dict[str, int]] = {}        # finger -> body -> n points

    for finger_name, bodies in FINGER_BODIES.items():
        if bodies == ["none"]:
            continue
        body_radius[finger_name] = {}
        body_npts[finger_name] = {}

        for bname in bodies:
            lp = link_prop_by_name.get(bname)
            if lp is None or lp.mesh_file is None:
                print(f"  {finger_name}/{bname}: no visual mesh, skipping.")
                continue

            resolved = _resolve_mesh_path(lp.mesh_file, mesh_folder, urdf_base)
            if resolved is None:
                print(f"  {finger_name}/{bname}: could not resolve mesh '{lp.mesh_file}', skipping.")
                continue

            folder, fname = os.path.split(resolved)
            points, _normals, _sharp, _pass = mesh_to_points_2(
                folder + os.sep, fname, POINTS_PER_AREA, MESH_SCALE,
            )
            if len(points) < 4:
                print(f"  {finger_name}/{bname}: too few sampled points ({len(points)}), skipping.")
                continue

            radius, _axis, _centroid = smallest_enclosing_cylinder(points.astype(np.float64))
            body_radius[finger_name][bname] = radius
            body_npts[finger_name][bname] = len(points)
            print(f"  {finger_name}/{bname}: radius={radius * 1000:.3f} mm  ({len(points)} points)")

    # ── 3. Average cylinder radius per finger ─────────────────────────────────
    print("\nAverage cylinder radius per finger:")
    finger_avg_radius: dict[str, float] = {}
    for finger_name, radii in body_radius.items():
        if not radii:
            continue
        avg = float(np.mean(list(radii.values())))
        finger_avg_radius[finger_name] = avg
        print(f"  {finger_name}: {avg * 1000:.3f} mm  (over {len(radii)} bodies)")

    # ── 4. First-joint positions + pairwise distances between fingers ────────
    print("\nFirst-joint positions (home pose, world frame):")
    first_joint_pos: dict[str, np.ndarray] = {}
    for finger_name, bodies in FINGER_BODIES.items():
        if finger_name == "Palm" or bodies == ["none"]:
            continue
        first_body = bodies[0]
        try:
            pos = _first_joint_position(model, data, first_body)
        except Exception as exc:
            print(f"  {finger_name}: could not locate first joint of '{first_body}' ({exc}), skipping.")
            continue
        first_joint_pos[finger_name] = pos
        print(f"  {finger_name} (first body '{first_body}'): "
              f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

    pair_distances: list[tuple[str, str, float]] = []
    for a, b in itertools.combinations(sorted(first_joint_pos.keys()), 2):
        d = float(np.linalg.norm(first_joint_pos[a] - first_joint_pos[b]))
        pair_distances.append((a, b, d))

    print("\nPairwise distance between fingers' first-joint positions:")
    if pair_distances:
        for a, b, d in pair_distances:
            print(f"  {a} <-> {b}: {d * 1000:.3f} mm")
        d_values = np.array([d for _, _, d in pair_distances])
        print(f"  min:  {d_values.min() * 1000:.3f} mm")
        print(f"  max:  {d_values.max() * 1000:.3f} mm")
        print(f"  mean: {d_values.mean() * 1000:.3f} mm")
    else:
        print("  Fewer than 2 fingers present -- nothing to compare.")

    # ── Save summary ───────────────────────────────────────────────────────
    out_dir = os.path.join(ROOT, SAVE_FOLDER, "hand_dimensions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hand_dimensions.json")
    summary = {
        "points_per_area": POINTS_PER_AREA,
        "body_cylinder_radius_m": body_radius,
        "body_n_points": body_npts,
        "finger_avg_cylinder_radius_m": finger_avg_radius,
        "first_joint_position_m": {k: v.tolist() for k, v in first_joint_pos.items()},
        "first_joint_pairwise_distance_m": [
            {"finger_a": a, "finger_b": b, "distance": d} for a, b, d in pair_distances
        ],
    }
    if pair_distances:
        d_values = np.array([d for _, _, d in pair_distances])
        summary["first_joint_pairwise_distance_stats_m"] = {
            "min": float(d_values.min()),
            "max": float(d_values.max()),
            "mean": float(d_values.mean()),
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
