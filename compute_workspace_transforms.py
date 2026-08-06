#!/home/sean/miniconda3/bin/python3
"""
compute_workspace_transforms.py
Precomputes and saves each finger's per-body workspace transform sweep
(workspace.finger_workspace_transforms()), for every finger on the active
hand config (see opposability_calculation.py), to disk.

manipulation_capacity_test.py used to compute this sweep itself, inline, on
every run -- but it's one of the most expensive steps in that script and is
identical for a given finger regardless of which graspable-volume region or
finger pairing is being evaluated. Splitting it out here means:
  - it only has to be computed once per finger, not once per
    manipulation_capacity_test.py run,
  - it can be precomputed while other parts of the pipeline (workspace/
    opposability analysis) are still being iterated on, and
  - evaluating several different graspable regions/finger pairs (e.g. sweeping
    MANIPULATION_GROUP or OPPOSABILITY_GROUP_RADII) no longer re-pays this
    cost per finger pair -- each finger's sweep is computed exactly once here
    and shared across every pair that includes it.

Usage:
    ./run_compute_workspace_transforms.sh
    (or directly: ~/miniconda3/bin/python3 compute_workspace_transforms.py)

Outputs (relative to project root):
    <SAVE_FOLDER>/Workspace_Transforms/<Finger>_wkspace_transforms.npz
        one file per finger present on the hand (all FINGER_BODIES keys
        except "Palm" and any finger set to ["none"])
"""

from __future__ import annotations

import os
import sys
import time

if sys.version_info < (3, 10):
    sys.exit(
        f"ERROR: Python {sys.version} detected.\n"
        "This script requires Python 3.10+ (the conda environment).\n"
        "Run with:\n"
        "  ~/miniconda3/bin/python3 compute_workspace_transforms.py\n"
        "or activate conda first:\n"
        "  conda activate base && python3 compute_workspace_transforms.py"
    )

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pinocchio as pin
from python import (
    build_hand,
    compute_fk,
    finger_workspace_transforms,
    load_model,
    parse_mimic_and_offsets,
    save_finger_workspace_transforms,
)

# Reuse the active hand config directly from opposability_calculation.py --
# change the hand there, and this script follows automatically.
from opposability_calculation import (
    FINGER_BODIES,
    URDF_FILE,
    MESH_FOLDER,
    MESH_SCALE,
    SAVE_FOLDER,
    N_MESH_SAMPLES,
    N_PTS,
    HOME_ACTUATED,
    N_WORKERS,
)


def main() -> None:
    urdf_path = os.path.join(ROOT, URDF_FILE)
    mesh_folder = os.path.join(ROOT, MESH_FOLDER)
    urdf_base = os.path.dirname(urdf_path)

    print("Parsing URDF...")
    mimic_props, link_props = parse_mimic_and_offsets(urdf_path)

    print("Loading pinocchio model...")
    model, data = load_model(urdf_path)

    print("Building finger structs...")
    contact_specs = {name: np.zeros((len(bodies), 2)) for name, bodies in FINGER_BODIES.items()}
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

    # ── Home configuration (identical to opposability_calculation.py) ────────
    q_neutral = pin.neutral(model)
    if HOME_ACTUATED is not None and len(HOME_ACTUATED) == coupling.n_actuated:
        full_q_home = coupling.matrix @ np.append(HOME_ACTUATED, 1.0)
        q_home = q_neutral.copy()
        q_home[:len(full_q_home)] = full_q_home
    else:
        q_home = q_neutral.copy()
    compute_fk(model, data, q_home)

    # ── Actuated joint space grid (identical to opposability_calculation.py) ──
    n_act = coupling.n_actuated
    act_space = np.zeros((n_act, N_PTS))
    for i, full_id in enumerate(coupling.actuated_joint_ids):
        lo = float(model.lowerPositionLimit[full_id])
        hi = float(model.upperPositionLimit[full_id])
        act_space[i, :] = np.linspace(lo, hi, N_PTS)

    out_dir = os.path.join(ROOT, SAVE_FOLDER, "Workspace_Transforms")

    # ── Sweep + save each finger's per-body transforms ────────────────────────
    for finger_name, bodies in FINGER_BODIES.items():
        if finger_name == "Palm" or bodies == ["none"]:
            continue
        finger = fingers[finger_name]
        if finger.is_empty():
            continue

        print(f"Sweeping {finger_name}'s joint-space transforms...")
        t0 = time.time()
        result = finger_workspace_transforms(finger, act_space, coupling, model, data, q_home, n_workers=N_WORKERS)
        elapsed = time.time() - t0
        print(f"  {result.jointspace.shape[0]:,} poses in {elapsed:.1f}s")

        path = save_finger_workspace_transforms(out_dir, finger_name, result)
        print(f"  Saved {path}")

    print("Done.")


if __name__ == "__main__":
    main()
