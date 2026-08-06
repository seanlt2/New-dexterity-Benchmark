#!/home/sean/miniconda3/bin/python3
"""
opposability_calculation.py
Computes per-finger workspace volumes for a robotic hand, then finds sphere
positions where the selected fingers form an opposable grasp that doesn't
collide with the palm. Saves everything and visualises it.

Python port of the MATLAB Workspace_calculation.m / graspable_volume.m /
workspace_palm_collision.m pipeline. (Supersedes the older alpha-shape
workspace-INTERSECTION approach from Opposability_calculation.m — that
voxel-grid pairwise/N-way intersection method is no longer computed here;
see python/opposability.py for the sphere-based replacement.)

Usage:
    ~/miniconda3/bin/python3 opposability_calculation.py [--hand NAME]
    (or: conda activate base && python3 opposability_calculation.py --hand NAME)

Pass --hand to select which hand config to run (see HAND_CONFIGS below for
valid names; --help lists them); defaults to DEFAULT_HAND. manipulation_
capacity_test.py and leap_hand_grasp_figures.py both import their hand
config from this module too, so passing --hand to either of their
run_*.sh wrappers selects the same hand for them as well.

Outputs (relative to project root):
    <save_folder>/workspace_volumes/
        Finger_<N>_wkspace_pts.npy
        Finger_<N>_wkspace_alpha.pkl
        Finger_<N>_wkspace.csv
    <save_folder>/graspable_volume/
        graspable_volume_pts.npy
        graspable_volume_alpha.pkl
        graspable_volume.csv
"""

from __future__ import annotations

import sys

# Guard: must run under Python 3.10+ (conda environment).
# The system python3 on this machine is 3.8 and has incompatible packages.
if sys.version_info < (3, 10):
    sys.exit(
        f"ERROR: Python {sys.version} detected.\n"
        "This script requires Python 3.10+ (the conda environment).\n"
        "Run with:\n"
        "  ~/miniconda3/bin/python3 opposability_calculation.py\n"
        "or activate conda first:\n"
        "  conda activate base && python3 opposability_calculation.py"
    )

import argparse
import multiprocessing as mp
import os
import pickle
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import trimesh

# ── make `python` package importable from project root ───────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pinocchio as pin
from python import (
    actuated_to_full_jointspace,
    build_collision_mesh,
    build_hand,
    compute_fk,
    compute_opposable_grasp_volume,
    finger_workspace,
    generate_alphashape,
    load_model,
    parse_mimic_and_offsets,
    points_transform,
    sample_workspace_grid,
    workspace_volume,
)

# ─────────────────────────────────────────────────────────────────────────────
# Finger-number labels (matching MATLAB file-naming convention)
# ─────────────────────────────────────────────────────────────────────────────
FINGER_ID = {"Thumb": 1, "Index": 2, "Pinkie": 3, "Middle": 4, "Ring": 5}

# ─────────────────────────────────────────────────────────────────────────────
# Hand configurations
# ─────────────────────────────────────────────────────────────────────────────
# Select which one is active with --hand NAME (see HAND_CONFIGS' keys below
# for valid names; defaults to DEFAULT_HAND). Everything a hand config used
# to need as a bare module-level constant -- FINGER_BODIES, MESH_SCALE,
# URDF_FILE, MESH_FOLDER, SAVE_FOLDER, HOME_ACTUATED -- now lives together in
# one dict entry, so switching hands is a single flag instead of
# hand-editing which block is commented out.
#
# opposability_groups: list[tuple[group, radius]] -- which finger
# combinations to test for an opposable grasp, and the sphere radius (m) to
# test each one with. Each group is computed (and plotted/saved) separately,
# rather than requiring every active finger to simultaneously reach and
# oppose the same sphere. Not every reachable-and-opposable combination is a
# *useful* grasp for a given hand's mechanism (e.g. two fingers that can
# only flex in parallel won't form an effective pinch even if they can
# geometrically reach the same sphere from roughly opposite sides), so pick
# the combinations that make sense for the hand. Group names must be keys
# of finger_bodies. Hands that were never actually run with a specific set
# leave this as []. A radius of None means "no override" -- a Thumb+Pinkie
# pinch is only ever going to close around something much smaller than a
# full Thumb+Index+Middle+Ring+Pinkie power grasp, for example, so most
# hands below give every group its own explicit radius, but a group can
# fall back to the shared GRASP_SPHERE_RADIUS by using None instead.
# ─────────────────────────────────────────────────────────────────────────────

HAND_CONFIGS: dict[str, dict] = {
    "soft_hand": dict(
        finger_bodies={
            "Thumb":  ["CMC", "Link1_Thumb", "Fingertip_Thumb"],
            "Index":  ["Abd_Add_Index", "Link1_Index", "Fingertip_Index"],
            "Middle": ["none"],
            "Ring":   ["none"],
            "Pinkie": ["Abd_Add_Pinkie", "Link1_Pinkie", "Fingertip_Pinkie"],
            "Palm":   ["none"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/URDF_v4_Right/urdf/URDF_v4_Right.urdf",
        mesh_folder="URDF_Files/URDF_v4_Right/meshes/",
        save_folder="Opposability/URDF_v4_Right",
        home_actuated=None,   # use zero/neutral
        opposability_groups=[],
    ),
    "ability_hand": dict(
        finger_bodies={
            "Thumb":  ["thumb_L1", "thumb_L2"],
            "Index":  ["index_L1", "index_L2"],
            "Middle": ["middle_L1", "middle_L2"],
            "Ring":   ["ring_L1", "ring_L2"],
            "Pinkie": ["pinky_L1", "pinky_L2"],
            "Palm":   ["base", "thumb_base"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/ability_hand/ability_hand_right_stl.urdf",
        mesh_folder="URDF_Files/ability_hand/meshes/visual/",
        save_folder="Opposability/ability_hand",
        home_actuated=None,
        opposability_groups=[
            (("Thumb", "Index"), 0.011),
            (("Thumb", "Middle"), 0.011),
            (("Thumb", "Ring"), 0.011),
            (("Thumb", "Pinkie"), 0.011),
            (("Thumb", "Index", "Middle"), 0.016),
            (("Thumb", "Index", "Ring"), 0.016),
            (("Thumb", "Index", "Pinkie"), 0.016),
            (("Thumb", "Middle", "Ring"), 0.016),
            (("Thumb", "Middle", "Pinkie"), 0.016),
            (("Thumb", "Ring", "Pinkie"), 0.016),
            (("Thumb", "Index", "Middle", "Ring"), 0.022),
            (("Thumb", "Index", "Middle", "Pinkie"), 0.022),
            (("Thumb", "Index", "Ring", "Pinkie"), 0.022),
            (("Thumb", "Middle", "Ring", "Pinkie"), 0.022),
            (("Thumb", "Index", "Middle", "Ring", "Pinkie"), 0.027),
        ],
    ),
    "allegro_hand": dict(
        finger_bodies={
            "Thumb":  ["link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"],
            "Index":  ["link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip"],
            "Middle": ["link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"],
            "Ring":   ["none"],
            "Pinkie": ["link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip"],
            "Palm":   ["base_link"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/allegro_hand/allegro_hand_right_stl.urdf",
        mesh_folder="URDF_Files/allegro_hand/meshes/visual/",
        save_folder="Opposability/allegro_hand",
        home_actuated=None,
        opposability_groups=[
            (("Thumb", "Index"), 0.016),
            (("Thumb", "Middle"), 0.016),
            (("Thumb", "Pinkie"), 0.016),
            (("Index", "Middle"), 0.016),
            (("Index", "Pinkie"), 0.016),
            (("Middle", "Pinkie"), 0.016),
            (("Thumb", "Index", "Middle"), 0.024),
            (("Thumb", "Index", "Pinkie"), 0.024),
            (("Thumb", "Middle", "Pinkie"), 0.024),
            (("Index", "Middle", "Pinkie"), 0.024),
            (("Thumb", "Index", "Middle", "Pinkie"), 0.032),
        ],
    ),
    "allegro_hand_3_finger": dict(
        finger_bodies={
            "Thumb":  ["link_6_0", "link_7_0", "link_8_0", "link_8_0_tip"],
            "Index":  ["link_0_0", "link_1_0", "link_2_0", "link_2_0_tip"],
            "Middle": ["none"],
            "Ring":   ["none"],
            "Pinkie": ["link_4_0", "link_4_0", "link_5_0", "link_5_0_tip"],
            "Palm":   ["palm_link"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/allegro_hand_3_finger/allegro_hand_3F.urdf",
        mesh_folder="URDF_Files/allegro_hand_3_finger/meshes/",
        save_folder="Opposability/allegro_hand_3_finger",
        home_actuated=None,
        opposability_groups=[],
    ),
    "barrett_hand": dict(
        finger_bodies={
            "Thumb":  ["finger_3_med_link", "finger_3_dist_link"],
            "Index":  ["finger_1_prox_link", "finger_1_med_link", "finger_1_dist_link"],
            "Middle": ["none"],
            "Ring":   ["none"],
            "Pinkie": ["finger_2_prox_link", "finger_2_med_link", "finger_2_dist_link"],
            "Palm":   ["base_link"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/barrett_hand/bhand_model_stl.urdf",
        mesh_folder="URDF_Files/barrett_hand/meshes/visual/",
        save_folder="Opposability/barrett_hand",
        home_actuated=None,
        opposability_groups=[],
    ),
    "craft_hand": dict(
        finger_bodies={
            "Thumb":  ["thumbmcp", "thumb2", "thumb3", "thumb4"],
            "Index":  ["index1", "index2", "index3", "index4"],
            "Middle": ["middle1", "middle2", "middle3", "middle4"],
            "Ring":   ["ring1", "ring2", "ring3", "ring4"],
            "Pinkie": ["pinky1", "pinky2", "pinky3", "pinky4"],
            "Palm":   ["base"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/CRAFT_Hand/craft_hand.urdf",
        mesh_folder="URDF_Files/CRAFT_Hand/meshes/",
        save_folder="Opposability/CRAFT_Hand",
        home_actuated=None,
        opposability_groups=[],
    ),
    "dclaw_gripper": dict(
        finger_bodies={
            "Thumb":  ["link_f1_1", "link_f1_2", "link_f1_3", "link_f1_head"],
            "Index":  ["link_f2_1", "link_f2_2", "link_f2_3", "link_f2_head"],
            "Middle": ["none"],
            "Ring":   ["none"],
            "Pinkie": ["link_f3_1", "link_f3_2", "link_f3_3", "link_f3_head"],
            "Palm":   ["base_link"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/dclaw_gripper/dclaw_gripper_stl.urdf",
        mesh_folder="URDF_Files/dclaw_gripper/meshes/visual/",
        save_folder="Opposability/dclaw_gripper",
        # Home actuated pose -- matches MATLAB: [(-pi/2), -1.7, -1.7, 0]
        home_actuated=np.array([-np.pi / 2, -1.7, -1.7, 0.0]),
        opposability_groups=[],
    ),
    "inspire_hand": dict(
        finger_bodies={
            "Thumb":  ["thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal"],
            "Index":  ["index_proximal", "index_intermediate"],
            "Middle": ["middle_proximal", "middle_intermediate"],
            "Ring":   ["ring_proximal", "ring_intermediate"],
            "Pinkie": ["pinky_proximal", "pinky_intermediate"],
            "Palm":   ["hand_base_link"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/inspire_hand/inspire_hand_right_stl.urdf",
        mesh_folder="URDF_Files/inspire_hand/meshes/visual/",
        save_folder="Opposability/inspire_hand",
        home_actuated=None,
        opposability_groups=[],
    ),
    "krysalis_hand": dict(
        finger_bodies={
            "Thumb":  ["T_CMC", "T_MCP", "T_IP"],
            "Index":  ["I_Below_MCP", "I_MCP", "I_PIP", "I_DIP"],
            "Middle": ["M_MCP", "M_PIP", "M_DIP"],
            "Ring":   ["R_Below_MCP", "R_MCP", "R_PIP", "R_DIP"],
            "Pinkie": ["P_Below_MCP", "P_MCP", "P_PIP", "P_DIP"],
            "Palm":   ["Knuckle_Holder"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Krysalis_hand/hand.urdf",
        mesh_folder="URDF_Files/Krysalis_hand/meshes/",
        save_folder="Opposability/Krysalis_hand",
        home_actuated=None,
        opposability_groups=[
            (("Thumb", "Index"), 123456),
            (("Thumb", "Middle"), 123456),
            (("Thumb", "Ring"), 123456),
            (("Thumb", "Pinkie"), 123456),
            (("Index", "Middle"), 123456),
            (("Index", "Pinkie"), 123456),
            (("Middle", "Ring"), 123456),
            (("Middle", "Pinkie"), 123456),
            (("Thumb", "Index", "Middle"), 123456),
            (("Thumb", "Index", "Ring"), 123456),
            (("Thumb", "Index", "Pinkie"), 123456),
            (("Thumb", "Middle", "Ring"), 123456),
            (("Thumb", "Middle", "Pinkie"), 123456),
            (("Thumb", "Ring", "Pinkie"), 123456),
            (("Thumb", "Index", "Middle", "Ring"), 123456),
            (("Thumb", "Index", "Middle", "Pinkie"), 123456),
            (("Thumb", "Index", "Ring", "Pinkie"), 123456),
            (("Thumb", "Middle", "Ring", "Pinkie"), 123456),
            (("Thumb", "Index", "Middle", "Ring", "Pinkie"), 123456),
        ],
    ),
    "leap_hand": dict(
        finger_bodies={
            "Thumb":  ["thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip"],
            "Index":  ["mcp_joint", "pip", "dip", "fingertip"],
            "Middle": ["mcp_joint_2", "pip_2", "dip_2", "fingertip_2"],
            "Ring":   ["none"],
            "Pinkie": ["mcp_joint_3", "pip_3", "dip_3", "fingertip_3"],
            "Palm":   ["palm_lower"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/leap_hand/leap_hand_right_stl.urdf",
        mesh_folder="URDF_Files/leap_hand/meshes/visual/",
        save_folder="Opposability/leap_hand",
        home_actuated=None,
        opposability_groups=[
            (("Thumb", "Index"), 0.021),
            (("Thumb", "Middle"), 0.021),
            (("Thumb", "Pinkie"), 0.021),
            (("Index", "Middle"), 0.021),
            (("Index", "Pinkie"), 0.021),
            (("Middle", "Pinkie"), 0.021),
            (("Thumb", "Index", "Middle"), 0.032),
            (("Thumb", "Index", "Pinkie"), 0.032),
            (("Thumb", "Middle", "Pinkie"), 0.032),
            (("Index", "Middle", "Pinkie"), 0.032),
            (("Thumb", "Index", "Middle", "Pinkie"), 0.043),
        ],
    ),
    "orca_hand": dict(
        finger_bodies={
            "Thumb":  ["none",],
            "Index":  ["none",],
            "Middle": ["none",],
            "Ring":   ["none",],
            "Pinkie": ["none",],
            "Palm":   ["none",],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Orca_Hand/models/urdf/orcahand_right.urdf",
        mesh_folder="URDF_Files/Orca_Hand/assets/urdf/",
        save_folder="Opposability/Orca_Hand",
        home_actuated=None,
        opposability_groups=[],
    ),
    "rapid_hand": dict(
        finger_bodies={
            "Thumb":  ["TMCPF","TMCPS","TPIP","TDIP"],
            "Index":  ["IMCPF","IMCPS","IPIP","IDIP"],
            "Middle": ["MMCPF","MMCPS","MPIP","MDIP"],
            "Ring":   ["RMCPF","RMCPS","RPIP","RDIP"],
            "Pinkie": ["LMCPF","LMCPS","LPIP","LDIP"],
            "Palm":   ["BASE",],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Rapid_Hand/urdf/rapidhand.urdf",
        mesh_folder="URDF_Files/Rapid_Hand/meshes/",
        save_folder="Opposability/Rapid_Hand",
        home_actuated=None,
        opposability_groups=[],
    ),
    "HX5_D20": dict(
        finger_bodies={
            "Thumb":  ["finger_r_link1","finger_r_link2","finger_r_link3","finger_r_link4"],
            "Index":  ["finger_r_link5","finger_r_link6","finger_r_link7","finger_r_link8"],
            "Middle": ["finger_r_link9","finger_r_link10","finger_r_link11","finger_r_link12"],
            "Ring":   ["finger_r_link13","finger_r_link14","finger_r_link15","finger_r_link16"],
            "Pinkie": ["finger_r_link17","finger_r_link18","finger_r_link19","finger_r_link20"],
            "Palm":   ["hx5_d20_right_base",],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/HX5_D20_Right/hx5_d20_right.urdf",
        mesh_folder="URDF_Files/HX5_D20_Right/meshes/",
        save_folder="Opposability/HX5_D20",
        home_actuated=None,
        opposability_groups=[],
    ),
    "ruka_hand": dict(
        finger_bodies={
            "Thumb":  ["Thumb_MCP_Link", "Thumb_DIP_Link", "Thumb_PIP_Link"],
            "Index":  ["Index_MCP_Link", "Index_DIP_Link", "Index_PIP_Link"],
            "Middle": ["Middle_MCP_Link", "Middle_DIP_Link", "Middle_PIP_Link"],
            "Ring":   ["Ring_MCP_Link", "Ring_DIP_Link", "Ring_PIP_Link"],
            "Pinkie": ["Pinky_MCP_Link", "Pinky_DIP_Link", "Pinky_PIP_Link"],
            "Palm":   ["Palm_Link"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/ruka_hand/ruka_hand.urdf",
        mesh_folder="URDF_Files/ruka_hand/meshes/",
        save_folder="Opposability/ruka_hand",
        home_actuated=None,
        opposability_groups=[
            (("Thumb", "Index"), 123456),
            (("Thumb", "Middle"), 123456),
            (("Thumb", "Ring"), 123456),
            (("Thumb", "Pinkie"), 123456),
            (("Thumb", "Index", "Middle"), 123456),
            (("Thumb", "Index", "Ring"), 123456),
            (("Thumb", "Index", "Pinkie"), 123456),
            (("Thumb", "Middle", "Ring"), 123456),
            (("Thumb", "Middle", "Pinkie"), 123456),
            (("Thumb", "Ring", "Pinkie"), 123456),
            (("Thumb", "Index", "Middle", "Ring"), 123456),
            (("Thumb", "Index", "Middle", "Pinkie"), 123456),
            (("Thumb", "Index", "Ring", "Pinkie"), 123456),
            (("Thumb", "Middle", "Ring", "Pinkie"), 123456),
            (("Thumb", "Index", "Middle", "Ring", "Pinkie"), 123456),
        ],
    ),
    "shadow_hand": dict(
        finger_bodies={
            "Thumb":  ["thbase", "thproximal", "thhub", "thmiddle", "thdistal"],
            "Index":  ["ffknuckle", "ffproximal", "ffmiddle", "ffdistal"],
            "Middle": ["mfknuckle", "mfproximal", "mfmiddle", "mfdistal"],
            "Ring":   ["rfknuckle", "rfproximal", "rfmiddle", "rfdistal"],
            "Pinkie": ["lfmetacarpal", "lfknuckle", "lfproximal", "lfmiddle", "lfdistal"],
            "Palm":   ["palm"],
        },
        mesh_scale=0.001,
        urdf_file="URDF_Files/shadow_hand/shadow_hand_right_coupled_stl.urdf",
        mesh_folder="URDF_Files/shadow_hand/meshes/visual/",
        save_folder="Opposability/shadow_hand",
        home_actuated=None,
        opposability_groups=[],
    ),
    "shadow_dexee": dict(
        finger_bodies={
            "Thumb":  ["none"],
            "Index":  ["none"],
            "Middle": ["none"],
            "Ring":   ["none"],
            "Pinkie": ["none"],
            "Palm":   ["none"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/shadow_dexee/shadow_dexee.urdf",
        mesh_folder="URDF_Files/shadow_dexee/meshes/",
        save_folder="Opposability/shadow_dexee",
        home_actuated=None,
        opposability_groups=[],
    ),
    "sharpa_wave_hand": dict(
        finger_bodies={
            "Thumb":  ["right_thumb_CMC_VL", "right_thumb_MC", "right_thumb_MCP_VL", "right_thumb_PP", "right_thumb_DP"],
            "Index":  ["right_index_MCP_VL", "right_index_PP", "right_index_MP", "right_index_DP"],
            "Middle": ["right_middle_MCP_VL", "right_middle_PP", "right_middle_MP", "right_middle_DP"],
            "Ring":   ["right_ring_MCP_VL", "right_ring_PP", "right_ring_MP", "right_ring_DP"],
            "Pinkie": ["right_pinky_MC", "right_pinky_MCP_VL", "right_pinky_PP", "right_pinky_MP", "right_pinky_DP"],
            "Palm":   ["right_hand_C_MC"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/sharpa_wave_hand/right_hand/right_hand.urdf",
        mesh_folder="URDF_Files/sharpa_wave_hand/right_hand/meshes/",
        save_folder="Opposability/sharpa_wave_hand/right_hand",
        home_actuated=None,
        opposability_groups=[],
    ),
    "schunk_hand": dict(
        finger_bodies={
            "Thumb":  ["right_hand_a", "right_hand_b", "right_hand_c"],
            "Index":  ["right_hand_l", "right_hand_p", "right_hand_t"],
            "Middle": ["right_hand_k", "right_hand_o", "right_hand_s"],
            "Ring":   ["right_hand_j", "right_hand_n", "right_hand_r"],
            "Pinkie": ["right_hand_i", "right_hand_m", "right_hand_q"],
            "Palm":   ["right_hand_e1", "right_hand_z", "right_hand_e2",
                       "right_hand_virtual_k", "right_hand_virtual_l",
                       "right_hand_virtual_i", "right_hand_virtual_j"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/schunk_hand/schunk_svh_hand_right_stl.urdf",
        mesh_folder="URDF_Files/schunk_hand/meshes/visual/",
        save_folder="Opposability/schunk_hand",
        home_actuated=None,
        opposability_groups=[
            (("Thumb", "Index"), 123456),
            (("Thumb", "Middle"), 123456),
            (("Thumb", "Ring"), 123456),
            (("Thumb", "Pinkie"), 123456),
            (("Index", "Middle"), 123456),
            (("Index", "Ring"), 123456),
            (("Index", "Pinkie"), 123456),
            (("Middle", "Ring"), 123456),
            (("Middle", "Pinkie"), 123456),
            (("Thumb", "Index", "Middle"), 123456),
            (("Thumb", "Index", "Ring"), 123456),
            (("Thumb", "Index", "Pinkie"), 123456),
            (("Thumb", "Middle", "Ring"), 123456),
            (("Thumb", "Middle", "Pinkie"), 123456),
            (("Thumb", "Ring", "Pinkie"), 123456),
            (("Thumb", "Index", "Middle", "Ring"), 123456),
            (("Thumb", "Index", "Middle", "Pinkie"), 123456),
            (("Thumb", "Index", "Ring", "Pinkie"), 123456),
            (("Thumb", "Middle", "Ring", "Pinkie"), 123456),
            (("Thumb", "Index", "Middle", "Ring", "Pinkie"), 123456),
        ],
    ),
    "tesollo_dg3f": dict(
        finger_bodies={
            "Thumb":  ["l_dg_1_1", "l_dg_1_2", "l_dg_1_3", "l_dg_1_4", "l_dg_1_tip"],
            "Index":  ["l_dg_2_1", "l_dg_2_2", "l_dg_2_3", "l_dg_2_4", "l_dg_2_tip"],
            "Middle": ["none"],
            "Ring":   ["none"],
            "Pinkie": ["l_dg_3_1", "l_dg_3_2", "l_dg_3_3", "l_dg_3_4", "l_dg_3_tip"],
            "Palm":   ["l_dg_base"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Tesollo_URDF/dg3fm/dg3f_m_stl.urdf",
        mesh_folder="URDF_Files/Tesollo_URDF/dg3fm/meshes/",
        save_folder="Opposability/Tesollo/dg3fm",
        home_actuated=None,
        opposability_groups=[
            (("Thumb", "Index"), 0.015),
            (("Thumb", "Pinkie"), 0.015),
            (("Index", "Pinkie"), 0.015),
            (("Thumb", "Index", "Pinkie"), 0.023),
        ],
    ),
    "tesollo_dg4f": dict(
        finger_bodies={
            "Thumb":  ["l_dg_1_inner", "l_dg_1_1", "l_dg_1_2", "l_dg_1_3", "l_dg_1_4", "l_dg_1_5", "l_dg_1_tip"],
            "Index":  ["l_dg_2_1", "l_dg_2_2", "l_dg_2_3", "l_dg_2_4", "l_dg_2_tip"],
            "Middle": ["l_dg_3_1", "l_dg_3_2", "l_dg_3_3", "l_dg_3_4", "l_dg_3_tip"],
            "Ring":   ["none"],
            "Pinkie": ["l_dg_4_inner", "l_dg_4_1", "l_dg_4_2", "l_dg_4_3", "l_dg_4_4", "l_dg_4_5", "l_dg_4_tip"],
            "Palm":   ["l_dg_base"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Tesollo_URDF/dg4f/dg3f.urdf",
        mesh_folder="URDF_Files/Tesollo_URDF/dg4f/meshes/",
        save_folder="Opposability/Tesollo_URDF/dg4f",
        home_actuated=None,
        opposability_groups=[],
    ),
    "tesollo_dg5f": dict(
        finger_bodies={
            "Thumb":  ["rl_dg_1_1", "rl_dg_1_2", "rl_dg_1_3", "rl_dg_1_4", "rl_dg_1_tip"],
            "Index":  ["rl_dg_2_1", "rl_dg_2_2", "rl_dg_2_3", "rl_dg_2_4", "rl_dg_2_tip"],
            "Middle": ["rl_dg_3_1", "rl_dg_3_2", "rl_dg_3_3", "rl_dg_3_4", "rl_dg_3_tip"],
            "Ring":   ["rl_dg_4_1", "rl_dg_4_2", "rl_dg_4_3", "rl_dg_4_4", "rl_dg_4_tip"],
            "Pinkie": ["rl_dg_5_1", "rl_dg_5_2", "rl_dg_5_3", "rl_dg_5_4", "rl_dg_5_tip"],
            "Palm":   ["rl_dg_base", "rl_dg_palm"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Tesollo_URDF/dg5f/dg5f_right.urdf",
        mesh_folder="URDF_Files/Tesollo_URDF/dg5f/meshes/",
        save_folder="Opposability/Tesollo_URDF/dg5f",
        home_actuated=None,
        opposability_groups=[],
    ),
    "tesollo_dg5fs": dict(
        finger_bodies={
            "Thumb":  ["link_1_1", "link_1_2", "link_1_3", "link_1_4", "link_1_tip"],
            "Index":  ["link_2_1", "link_2_2", "link_2_3", "link_2_4", "link_2_tip"],
            "Middle": ["link_3_1", "link_3_2", "link_3_3", "link_3_4", "link_3_tip"],
            "Ring":   ["link_4_1", "link_4_2", "link_4_3", "link_4_4", "link_4_tip"],
            "Pinkie": ["link_5_1", "link_5_2", "link_5_3", "link_5_4", "link_5_tip"],
            "Palm":   ["link_base"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Tesollo_URDF/dg5fs/dg5fs_right.urdf",
        mesh_folder="URDF_Files/Tesollo_URDF/dg5fs/meshes/",
        save_folder="Opposability/Tesollo_URDF/dg5fs",
        home_actuated=None,
        opposability_groups=[],
    ),
    "tesollo_dg5fs_15dof": dict(
        finger_bodies={
            "Thumb":  ["link_1_1", "link_1_2", "link_1_3", "link_1_tip"],
            "Index":  ["link_2_1", "link_2_2", "link_2_3", "link_2_tip"],
            "Middle": ["link_3_1", "link_3_2", "link_3_3", "link_3_tip"],
            "Ring":   ["link_4_1", "link_4_2", "link_4_3", "link_4_tip"],
            "Pinkie": ["link_5_1", "link_5_2", "link_5_3", "link_5_tip"],
            "Palm":   ["link_base"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Tesollo_URDF/dg5fs/dg5fs_15dof_right.urdf",
        mesh_folder="URDF_Files/Tesollo_URDF/dg5fs/meshes/",
        save_folder="Opposability/Tesollo_URDF/dg5fs_15dof",
        home_actuated=None,
        opposability_groups=[],
    ),
    "unitree_dex3": dict(
        finger_bodies={
            "Thumb":  ["right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link"],
            "Index":  ["right_hand_index_0_link", "right_hand_index_1_link"],
            "Middle": ["none"],
            "Ring":   ["none"],
            "Pinkie": ["right_hand_middle_0_link", "right_hand_middle_1_link"],
            "Palm":   ["right_hand_palm_link"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Unitree/dex3_1/dex3_1_r.urdf",
        mesh_folder="URDF_Files/Unitree/dex3_1/meshes/",
        save_folder="Opposability/Unitree/dex3_1",
        home_actuated=None,
        opposability_groups=[],
    ),
    "unitree_dex5": dict(
        finger_bodies={
            "Thumb":  ["Link_11R", "Link_12R", "Link_13R", "Link_14R"],
            "Index":  ["Link_21R", "Link_22R", "Link_23R", "Link_24R"],
            "Middle": ["Link_31R", "Link_32R", "Link_33R", "Link_34R"],
            "Ring":   ["Link_41R", "Link_42R", "Link_43R", "Link_44R"],
            "Pinkie": ["Link_51R", "Link_52R", "Link_53R", "Link_54R"],
            "Palm":   ["base_link00"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Unitree/dex5_1/Dex5-URDF-R/Dex5-URDF-R.urdf",
        mesh_folder="URDF_Files/Unitree/dex5_1/Dex5-URDF-R/meshes/",
        save_folder="Opposability/Unitree/dex5_1",
        home_actuated=None,
        opposability_groups=[],
    ),
    "wuji_hand": dict(
        finger_bodies={
            "Thumb":  ["right_finger1_link1", "right_finger1_link2", "right_finger1_link3", "right_finger1_link4","right_finger1_tip_link"],
            "Index":  ["right_finger2_link1", "right_finger2_link2", "right_finger2_link3", "right_finger2_link4","right_finger2_tip_link"],
            "Middle": ["right_finger3_link1", "right_finger3_link2", "right_finger3_link3", "right_finger3_link4","right_finger3_tip_link"],
            "Ring":   ["right_finger4_link1", "right_finger4_link2", "right_finger4_link3", "right_finger4_link4","right_finger4_tip_link"],
            "Pinkie": ["right_finger5_link1", "right_finger5_link2", "right_finger5_link3", "right_finger5_link4","right_finger5_tip_link"],
            "Palm":   ["right_palm_link"],
        },
        mesh_scale=1.0,
        urdf_file="URDF_Files/Wuji_Hand/urdf/right.urdf",
        mesh_folder="URDF_Files/Wuji_Hand/meshes/right/",
        save_folder="Opposability/Wuji_Hand",
        home_actuated=None,
        opposability_groups=[],
    ),
}

DEFAULT_HAND = "leap_hand"

_parser = argparse.ArgumentParser(
    description="Compute per-finger workspace volumes and opposable grasp "
                 "volumes for a robotic hand.",
)
_parser.add_argument(
    "--hand",
    choices=sorted(HAND_CONFIGS),
    default=DEFAULT_HAND,
    help=f"Which hand config to use (default: {DEFAULT_HAND}).",
)
# parse_known_args(), not parse_args(): this module is imported (re-running
# everything above) by manipulation_capacity_test.py and
# leap_hand_grasp_figures.py too, so it must tolerate whatever other args
# the invoking script's own sys.argv happens to carry rather than erroring
# out on them. --hand and --help/-h still work normally either way, so
# passing --hand to any of the three scripts' run_*.sh wrappers selects the
# same hand for all of them.
_cli_args, _ = _parser.parse_known_args()
ACTIVE_HAND = _cli_args.hand

_active_cfg = HAND_CONFIGS[ACTIVE_HAND]
FINGER_BODIES  = _active_cfg["finger_bodies"]
MESH_SCALE     = _active_cfg["mesh_scale"]
URDF_FILE      = _active_cfg["urdf_file"]
MESH_FOLDER    = _active_cfg["mesh_folder"]
SAVE_FOLDER    = _active_cfg["save_folder"]
HOME_ACTUATED  = _active_cfg["home_actuated"]

# opposability_groups is stored as a single (group, radius) list per hand
# (one source of truth instead of a separate groups list + radii dict that
# had to be kept in sync); expand it here into the two forms the rest of
# this module -- and manipulation_capacity_test.py / leap_hand_grasp_figures
# .py, which import OPPOSABILITY_GROUP_RADII directly -- already expect.
# A radius of None means "no override", i.e. use GRASP_SPHERE_RADIUS.
OPPOSABILITY_GROUPS = [group for group, _radius in _active_cfg["opposability_groups"]]
OPPOSABILITY_GROUP_RADII = {
    group: radius
    for group, radius in _active_cfg["opposability_groups"]
    if radius is not None
}

print(f"Using hand config {ACTIVE_HAND!r} (pass --hand NAME to use another; "
      f"available: {', '.join(sorted(HAND_CONFIGS))})")

# Grid resolution for workspace sweep
N_PTS = 40
# Number of surface samples per mesh link
N_MESH_SAMPLES = 600
# Grid spacing used to resample each finger's workspace to an evenly spaced
# point cloud, once its alpha shape is known (m)
WORKSPACE_GRID_RESOLUTION = 0.001
# Radius of the sphere tested for opposable grasps (m); the default for
# every group in OPPOSABILITY_GROUPS unless overridden per-group in
# OPPOSABILITY_GROUP_RADII below.
GRASP_SPHERE_RADIUS = 0.01
# Friction coefficient used in the opposability test
FRICTION_MU = 0.2
# Torsional friction coefficient (resistance to twisting about the contact
# normal) for manipulation_capacity_test.py's force-closure check -- 0.0 is
# a "hard finger" contact (translational friction only), which this
# project's opposability/graspable-volume pipeline above assumes throughout
# and is the historical default. Note two hard-finger point contacts can
# never reach full 6-DOF force closure regardless of geometry or FRICTION_MU
# (each contributes at most rank 3, and two such contacts always share one
# redundant direction -- the wrench of a force along the line between them
# -- capping the pair at rank 5); a nonzero MU_TOR ("soft finger") adds a
# genuinely new wrench direction per contact and lets 2-finger grasps pass.
MU_TOR = 0.1
# Worker processes for this pipeline's parallelized hot spots:
#   - main()'s OPPOSABILITY_GROUPS loop (step 7 below, see _group_worker()):
#     every group's graspable-volume computation is independent, so they're
#     dispatched across a worker pool instead of run one at a time. This is
#     the only place graspable_volume()'s own n_workers parameter (next
#     item) is deliberately left at 1 rather than also using N_WORKERS --
#     multiprocessing.Pool workers are daemonic, and daemonic processes
#     cannot spawn children of their own, so nesting two pools crashes
#     outright (hit this for real: "AssertionError: daemonic processes are
#     not allowed to have children"). Outer-only still benefits every
#     group, not just 3+finger ones, so nothing is lost by not nesting.
#   - python/opposability.py's graspable_volume(): the 3+ finger
#     opposability test's per-point linear-program solve (see its
#     _opposability_chunk() docstring) -- the only unvectorized, and usually
#     dominant, cost in this pipeline for any OPPOSABILITY_GROUPS entry with
#     3+ fingers. No effect on 2-finger groups (already fully vectorized).
#     Available for direct use outside this pipeline's per-group dispatch
#     (e.g. a single ad-hoc call on one group) -- just not from inside
#     _group_worker(), per the above.
#   - python/workspace.py's sample_workspace_grid(): the chunked
#     shape.contains() point-in-alphashape test used to resample each
#     finger's workspace onto an evenly spaced grid (see its own docstring
#     -- "can otherwise be a multi-minute call on a finger-sized volume at
#     1mm resolution").
#   - python/workspace.py's generate_alphashape() -> _voxelize_points():
#     the chunked rounding/deduplication step that voxelizes a raw swept
#     point cloud before the critical-alpha search runs (see its own
#     docstring for the real ~35GB OOM incident this chunking exists to
#     avoid, and why the parallel path's merge strategy is deliberately
#     different from the serial fallback's). No effect on the critical-
#     alpha search itself, which stays serial (small input, sequential
#     bisection-style search). Same daemon restriction as above applies --
#     only called with n_workers=N_WORKERS from main(), never from inside
#     _group_worker().
#   - python/workspace.py's finger_workspace(): the joint-space grid sweep
#     itself (step 6's `ws = finger_workspace(...)` call) -- one FK solve
#     plus per-link point transform per sampled joint configuration. This is
#     what *generates* each finger's raw swept point cloud in the first
#     place (up to hundreds of thousands of independent configs for a
#     multi-DOF finger), so it's usually the single largest cost the other
#     parallelized steps above merely consume the output of. Below
#     _MIN_PARALLEL_CONFIGS combos it always runs serially regardless of
#     N_WORKERS. Only called from main() (never from inside _group_worker(),
#     which doesn't call it at all). Both this and finger_workspace_transforms()
#     just below write each worker's results directly into a disk-backed
#     memmap rather than returning them through the pool -- see
#     _workspace_chunk()'s docstring in workspace.py: returning each chunk's
#     (large) array instead was measured to cap real speedup at ~1.7x on 10
#     workers, since pickling/piping it back to the main process dominated
#     over the actual (cheap, ~75us/config) FK+transform compute.
#   - python/workspace.py's finger_workspace_transforms(): same joint-space
#     sweep as finger_workspace() above, but recording each body's world-
#     frame transform instead of transformed surface points -- used by
#     compute_workspace_transforms.py (a separate script, not called from
#     this one), not by main() here.
# 1 disables multiprocessing entirely for all of the above.
N_WORKERS = 10
# Seconds each figure window stays open before auto-closing on its own, so
# this script can run unattended (e.g. OPPOSABILITY_GROUPS with several
# entries pops up a new blocking figure per group) -- still closes early if
# you close a window yourself.
FIGURE_TIMEOUT_SECONDS = 60

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _makedirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _save_folder(sub: str) -> str:
    path = os.path.join(ROOT, SAVE_FOLDER, sub)
    os.makedirs(path, exist_ok=True)
    return path


def _save_volume(folder: str, stem: str, pts: np.ndarray, alpha) -> None:
    """Save point cloud (.npy + .csv) and alphashape (.pkl, if available)."""
    np.save(os.path.join(folder, f"{stem}_pts.npy"), pts)
    np.savetxt(os.path.join(folder, f"{stem}.csv"), pts, delimiter=",")
    if alpha is not None:
        with open(os.path.join(folder, f"{stem}_alpha.pkl"), "wb") as f:
            pickle.dump(alpha, f)


# ─────────────────────────────────────────────────────────────────────────────
# Parallel OPPOSABILITY_GROUPS dispatch
# ─────────────────────────────────────────────────────────────────────────────
# Each OPPOSABILITY_GROUPS entry's graspable-volume computation is
# independent of every other entry (see main() step 7's comment), so
# main() dispatches them across an outer worker pool below instead of a
# serial loop. workspaces/palm_mesh/grasp_dir are stashed as plain module
# globals (not passed via Pool(initargs=...), which would pickle them once
# per worker through the task queue) so forked workers inherit them for
# free via Linux's copy-on-write fork() semantics -- only each task's small
# `group` tuple actually gets pickled through the queue.
#
# _group_worker() always calls compute_opposable_grasp_volume() with
# n_workers=1 (no inner pool), even for 3+ finger groups that could
# otherwise use graspable_volume()'s own per-point opposability parallelism
# (see python/opposability.py) -- NOT a performance choice, a hard Python
# restriction: multiprocessing.Pool's worker processes are always daemonic,
# and daemonic processes are forbidden from spawning children of their own
# ("AssertionError: daemonic processes are not allowed to have children"),
# so an outer worker landing on a 3+finger group would crash outright the
# moment it tried to build its own inner pool. Confirmed by hitting exactly
# this crash on a real run. Outer-only still benefits every group (not just
# 3+finger ones), so nothing is lost by not nesting -- graspable_volume()'s
# n_workers parameter remains available for direct use outside this
# pipeline's per-group dispatch (e.g. a single ad-hoc call on one group).
_POOL_WORKSPACES: Optional[dict] = None
_POOL_PALM_MESH = None
_POOL_GRASP_DIR: Optional[str] = None


def _group_worker(group: tuple[str, ...]) -> dict:
    """
    Runs in a forked worker process: computes and saves one
    OPPOSABILITY_GROUPS entry's graspable volume. Deliberately does NOT
    call _plot_graspable_volume() -- matplotlib figure creation isn't safe
    to do concurrently across multiple processes, so main() does all
    plotting itself afterward, once each worker's result comes back.
    """
    radius = OPPOSABILITY_GROUP_RADII.get(group, GRASP_SPHERE_RADIUS)
    finger_workspaces = {fname: _POOL_WORKSPACES[fname] for fname in group}

    # verbose=False: with multiple groups all running at once,
    # graspable_volume()'s per-stage progress prints would interleave into
    # an unreadable mess across processes. main() prints one clean summary
    # line per group instead, once this returns. n_workers=1: see the
    # no-nested-pools comment above this function.
    t0 = time.time()
    grasp_pts = compute_opposable_grasp_volume(
        finger_workspaces, _POOL_PALM_MESH, r=radius, mu=FRICTION_MU,
        n_workers=1, verbose=False,
    )
    opposability_check_seconds = time.time() - t0

    stem = "graspable_volume_" + "_".join(group)
    grasp_alpha = generate_alphashape(grasp_pts) if len(grasp_pts) > 0 else None
    grasp_vol = workspace_volume(grasp_alpha)
    _save_volume(_POOL_GRASP_DIR, stem, grasp_pts, grasp_alpha)

    return {
        "group": group,
        "radius": radius,
        "grasp_pts": grasp_pts,
        "grasp_vol": grasp_vol,
        "stem": stem,
        "opposability_check_seconds": opposability_check_seconds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _t_main_start = time.time()
    urdf_path = os.path.join(ROOT, URDF_FILE)
    mesh_folder = os.path.join(ROOT, MESH_FOLDER)
    urdf_base = os.path.dirname(urdf_path)

    # _plot_home_pose() (step 4) and _plot_workspace_volumes() (step 6) save
    # directly into SAVE_FOLDER, before _save_folder() ever gets called for
    # any of its subdirectories -- only relied on SAVE_FOLDER already
    # existing from a previous run until now, which a clean checkout (or a
    # deleted output dir) doesn't have.
    _makedirs(os.path.join(ROOT, SAVE_FOLDER))

    # ── 1. Parse URDF and load pinocchio model ───────────────────────────────
    print("Parsing URDF...")
    mimic_props, link_props = parse_mimic_and_offsets(urdf_path)

    print("Loading pinocchio model...")
    model, data = load_model(urdf_path)
    print(f"  {model.njoints - 1} non-fixed joints, nq={model.nq}")

    # ── 2. Build finger sets + coupling matrix ───────────────────────────────
    print("Building finger structs...")
    contact_specs = {name: np.zeros((len(bodies), 2))
                     for name, bodies in FINGER_BODIES.items()}
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
    print(f"  {coupling.n_actuated} actuated joints")

    # ── 3. Home configuration ────────────────────────────────────────────────
    q_neutral = pin.neutral(model)

    if HOME_ACTUATED is not None and len(HOME_ACTUATED) == coupling.n_actuated:
        full_q_home = coupling.matrix @ np.append(HOME_ACTUATED, 1.0)
        q_home = q_neutral.copy()
        q_home[:len(full_q_home)] = full_q_home
    else:
        q_home = q_neutral.copy()

    # ── 4. Visualise home pose mesh points ──────────────────────────────────
    print("Collecting home-pose mesh points for visualisation...")
    compute_fk(model, data, q_home)

    home_pts: dict[str, np.ndarray] = {}
    for fname, fs in fingers.items():
        if fs.is_empty():
            continue
        collected = []
        for lnk in fs.links:
            if lnk.points is not None and lnk.points.shape[1] == 3:
                wp = points_transform(model, data, lnk.body_name,
                                      lnk.rot_offset, lnk.trans_offset,
                                      lnk.points)
                collected.append(wp)
        if collected:
            home_pts[fname] = np.vstack(collected)

    _plot_home_pose(home_pts)

    # ── 5. Actuated joint space grid ─────────────────────────────────────────
    print("Building actuated joint-space grid...")
    n_act = coupling.n_actuated
    act_space = np.zeros((n_act, N_PTS))
    for i, full_id in enumerate(coupling.actuated_joint_ids):
        # Get joint position limits from pinocchio (0-indexed)
        lo = float(model.lowerPositionLimit[full_id])
        hi = float(model.upperPositionLimit[full_id])
        act_space[i, :] = np.linspace(lo, hi, N_PTS)

    # ── 6. Compute each finger's workspace, alpha shape, and evenly-sampled
    #      grid, one finger at a time ────────────────────────────────────────
    # Deliberately merged into a single per-finger loop, rather than
    # computing every finger's raw swept cloud first and only alpha-shaping
    # them afterward: a dense, actuated finger's raw cloud can be hundreds of
    # millions of points (multiple GB even as float32), and holding every
    # finger's raw cloud alive at once -- as the previous two-pass version
    # did -- was measured to reach ~35 GB baseline for just 4 fingers on the
    # Allegro Hand config, before generate_alphashape() (see its own memory
    # fix in workspace.py) even started allocating on top of that; the OOM
    # killer ended the process outright. Here, each finger's huge raw `ws`
    # is rebound to (and replaced by) its much smaller resampled grid before
    # moving to the next finger, so at most one finger's raw cloud is ever
    # resident at a time.
    workspaces: dict[str, np.ndarray | None] = {}
    alphas: dict[str, object] = {}
    finger_names_active: list[str] = []
    wvol_dir = _save_folder("workspace_volumes")

    # Per-stage wall-clock timings, reported in a summary table at the end
    # of main() (see STAGE_TIMINGS below). Keyed by finger name / group.
    finger_timings: dict[str, dict[str, float]] = {}
    group_timings: dict[tuple[str, ...], float] = {}

    for fname in ["Thumb", "Index", "Middle", "Ring", "Pinkie"]:
        fs = fingers.get(fname)
        if fs is None or fs.is_empty():
            workspaces[fname] = None
            continue

        print(f"Computing workspace: {fname}...")
        t0 = time.time()
        ws = finger_workspace(fs, act_space, coupling, model, data, q_home, n_workers=N_WORKERS)
        sweep_seconds = time.time() - t0
        print(f"  {len(ws):,} points ({sweep_seconds:.1f}s)")
        if len(ws) == 0:
            workspaces[fname] = None
            continue

        print(f"Generating alpha shape: {fname}...")
        t0 = time.time()
        alpha = generate_alphashape(ws, n_workers=N_WORKERS)
        alphashape_seconds = time.time() - t0
        alphas[fname] = alpha
        vol = workspace_volume(alpha)
        print(f"  {fname} alpha shape done ({vol:.6g} m^3, {alphashape_seconds:.1f}s)")

        # Resample onto an evenly spaced grid (points inside the alpha
        # shape) -- the raw swept cloud's density varies hugely with the
        # sweep's kinematics, not with the finger's actual reachable shape,
        # which throws off anything downstream that measures local point
        # density (e.g. opposability.graspable_volume()'s default grid
        # spacing). Falls back to the raw swept points if no alpha shape
        # was found, or the grid resample came back empty.
        t0 = time.time()
        ws_uniform = sample_workspace_grid(alpha, resolution=WORKSPACE_GRID_RESOLUTION, n_workers=N_WORKERS)
        resample_seconds = time.time() - t0
        if len(ws_uniform) > 0:
            print(f"  {fname} resampled to {len(ws_uniform):,} evenly-spaced points "
                  f"(from {len(ws):,} raw, {resample_seconds:.1f}s)")
            ws = ws_uniform   # drops the only reference to the raw cloud --
                               # free to be garbage-collected before the next
                               # finger's sweep starts.
        else:
            print(f"  {fname}: grid resample unavailable, keeping {len(ws):,} raw swept points "
                  f"({resample_seconds:.1f}s)")

        workspaces[fname] = ws
        finger_names_active.append(fname)
        finger_timings[fname] = {
            "sweep": sweep_seconds,
            "alphashape": alphashape_seconds,
            "resample": resample_seconds,
        }

        fid = FINGER_ID[fname]
        stem = f"Finger_{fid}_wkspace"
        _save_volume(wvol_dir, stem, ws, alpha)
        print(f"  Saved {stem}")

    _plot_workspace_volumes(workspaces, alphas, finger_names_active)

    # ── 7. Opposable grasp volume per selected finger combination: sphere ───
    #      positions the group can reach and grip, that don't collide with
    #      the palm. Each entry in OPPOSABILITY_GROUPS is computed and
    #      plotted separately rather than requiring every active finger to
    #      simultaneously reach the same sphere. ──────────────────────────
    palm_points = home_pts.get("Palm", np.empty((0, 3)))  # for visualisation only

    # finger_workspace() (step 6) left `data` at whatever pose its last swept
    # configuration was, not q_home -- recompute FK at q_home before reading
    # the palm's collision-geometry transform from it, same pose home_pts
    # (and palm_points above) were built at.
    compute_fk(model, data, q_home)
    palm_mesh = build_collision_mesh(
        FINGER_BODIES["Palm"], link_props, model, data, mesh_folder, urdf_base,
    )
    if palm_mesh is None:
        print("No palm collision geometry found in the URDF -- grasp candidates "
              "will not be filtered against the palm.")

    grasp_dir = _save_folder("graspable_volume")

    active_groups = []
    for group in OPPOSABILITY_GROUPS:
        missing = [f for f in group if f not in finger_names_active]
        if missing:
            print(f"Skipping {'+'.join(group)}: {', '.join(missing)} not active for this hand.")
            continue
        active_groups.append(group)

    if N_WORKERS > 1 and len(active_groups) > 1:
        # One worker per OPPOSABILITY_GROUPS entry running concurrently
        # (see _group_worker()'s no-nested-pools comment above for why this
        # is outer-only, not split with an inner pool per group).
        outer_workers = min(len(active_groups), N_WORKERS)
        print(f"Computing {len(active_groups)} opposable grasp volume(s) across "
              f"{outer_workers} worker process(es)...")

        global _POOL_WORKSPACES, _POOL_PALM_MESH, _POOL_GRASP_DIR
        _POOL_WORKSPACES = workspaces
        _POOL_PALM_MESH = palm_mesh
        _POOL_GRASP_DIR = grasp_dir

        with mp.get_context("fork").Pool(processes=outer_workers) as pool:
            for result in pool.imap_unordered(_group_worker, active_groups):
                group = result["group"]
                group_timings[group] = result["opposability_check_seconds"]
                print(f"  {'+'.join(group)} (r={result['radius']:g} m): "
                      f"{len(result['grasp_pts']):,} candidates, "
                      f"saved {result['stem']} ({result['grasp_vol']:.6g} m^3, "
                      f"opposability check {result['opposability_check_seconds']:.1f}s)")
                # ── 8. Visualisation (main process only) ───────────────────
                _plot_graspable_volume(workspaces, palm_points, result["grasp_pts"], list(group))
    else:
        for group in active_groups:
            radius = OPPOSABILITY_GROUP_RADII.get(group, GRASP_SPHERE_RADIUS)
            print(f"Computing opposable grasp volume: {'+'.join(group)} (r={radius:g} m)...")
            finger_workspaces = {fname: workspaces[fname] for fname in group}

            t0 = time.time()
            grasp_pts = compute_opposable_grasp_volume(
                finger_workspaces, palm_mesh, r=radius, mu=FRICTION_MU, n_workers=N_WORKERS,
            )
            group_timings[group] = time.time() - t0
            print(f"  {len(grasp_pts):,} candidate grasp-sphere positions "
                  f"(opposability check {group_timings[group]:.1f}s)")

            stem = "graspable_volume_" + "_".join(group)
            grasp_alpha = generate_alphashape(grasp_pts, n_workers=N_WORKERS) if len(grasp_pts) > 0 else None
            grasp_vol = workspace_volume(grasp_alpha)
            _save_volume(grasp_dir, stem, grasp_pts, grasp_alpha)
            print(f"  Saved {stem} ({grasp_vol:.6g} m^3)")

            # ── 8. Visualisation ─────────────────────────────────────────
            _plot_graspable_volume(workspaces, palm_points, grasp_pts, list(group))

    # ── Timing summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE TIMING SUMMARY")
    print("=" * 70)
    print(f"{'Finger':<10}{'Sweep (s)':>14}{'Alphashape (s)':>18}{'Resample (s)':>16}")
    for fname in finger_names_active:
        t = finger_timings[fname]
        print(f"{fname:<10}{t['sweep']:>14.1f}{t['alphashape']:>18.1f}{t['resample']:>16.1f}")
    total_sweep = sum(t["sweep"] for t in finger_timings.values())
    total_alphashape = sum(t["alphashape"] for t in finger_timings.values())
    total_resample = sum(t["resample"] for t in finger_timings.values())
    print(f"{'TOTAL':<10}{total_sweep:>14.1f}{total_alphashape:>18.1f}{total_resample:>16.1f}")

    print(f"\n{'Group':<30}{'Opposability check (s)':>24}")
    for group in active_groups:
        print(f"{'+'.join(group):<30}{group_timings[group]:>24.1f}")
    total_opposability = sum(group_timings.values())
    print(f"{'TOTAL':<30}{total_opposability:>24.1f}")

    grand_total = time.time() - _t_main_start
    print(f"\nGrand total (this function's wall time): {grand_total:.1f}s "
          f"({grand_total / 60:.1f} min)")
    print("=" * 70)

    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Thumb":  "blue",
    "Index":  "red",
    "Middle": "green",
    "Ring":   "orange",
    "Pinkie": "cyan",
    "Palm":   "gray",
}


def _show_and_close(fig, timeout: float = FIGURE_TIMEOUT_SECONDS) -> None:
    """
    Show a figure, auto-closing it after `timeout` seconds if it's still
    open -- lets this script run unattended through every OPPOSABILITY_GROUPS
    figure instead of blocking forever on plt.show() waiting for someone to
    close each window by hand. Closing a window yourself still unblocks
    immediately, same as plain plt.show().
    """
    timer = threading.Timer(timeout, plt.close, args=(fig,))
    timer.start()
    try:
        plt.show()
    finally:
        timer.cancel()


def _set_axes_equal(ax) -> None:
    """
    Force equal scale on all three axes of a 3-D plot.

    matplotlib's 3-D axes don't keep x/y/z at the same data-to-pixel scale
    by default (each axis is stretched to fill the figure independently),
    which visually distorts shapes. This recenters each axis on the data's
    midpoint and gives all three the same half-range.
    """
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    half_range = max(x1 - x0, y1 - y0, z1 - z0) / 2
    xm, ym, zm = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    ax.set_xlim3d(xm - half_range, xm + half_range)
    ax.set_ylim3d(ym - half_range, ym + half_range)
    ax.set_zlim3d(zm - half_range, zm + half_range)


def _plot_home_pose(home_pts: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for fname, pts in home_pts.items():
        if fname == "Palm":
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=1, label=fname, c=COLORS.get(fname, "black"))
    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title("Hand — home pose mesh points")
    ax.legend(markerscale=5)
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, SAVE_FOLDER, "home_pose.png"), dpi=150)
    _show_and_close(fig)


def _plot_workspace_volumes(
    workspaces: dict,
    alphas: dict[str, object],
    active: list[str],
) -> None:
    """Plot each finger's swept workspace point cloud and its alpha-shape volume."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for fname in active:
        ws = workspaces.get(fname)
        if ws is None or len(ws) == 0:
            continue
        color = COLORS.get(fname, "black")

        ax.scatter(ws[::10, 0], ws[::10, 1], ws[::10, 2],
                   s=1, c=color, alpha=0.2, label=fname)

        shape = alphas.get(fname)
        if shape is not None and hasattr(shape, "vertices") and hasattr(shape, "faces") and len(shape.faces) > 0:
            v = shape.vertices
            ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=shape.faces,
                            color=color, alpha=0.15, edgecolor="none")

    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title("Per-finger workspace volumes")
    ax.legend(markerscale=5)
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, SAVE_FOLDER, "workspace_volumes.png"), dpi=150)
    _show_and_close(fig)


def _plot_graspable_volume(
    workspaces: dict,
    palm_points: np.ndarray,
    grasp_pts: np.ndarray,
    group: list[str],
) -> None:
    """
    Plot only the given group's finger workspaces, the palm, and the
    resulting graspable sphere volume for that group.

    Only plotting `group`'s own fingers (not every active finger) keeps a
    dense finger's point cloud from visually drowning out a smaller group's
    grasp result -- with hundreds of thousands of background points even at
    low alpha, matplotlib's blending saturates to a solid blob that a
    thinner magenta overlay can barely show through.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for fname in group:
        ws = workspaces.get(fname)
        if ws is None or len(ws) == 0:
            continue
        ax.scatter(ws[::50, 0], ws[::50, 1], ws[::50, 2],
                   s=1, c=COLORS.get(fname, "black"), alpha=0.05, label=fname)

    if len(palm_points) > 0:
        ax.scatter(palm_points[:, 0], palm_points[:, 1], palm_points[:, 2],
                   s=1, c=COLORS["Palm"], alpha=0.2, label="Palm")

    if len(grasp_pts) > 0:
        ax.scatter(grasp_pts[:, 0], grasp_pts[:, 1], grasp_pts[:, 2],
                   s=4, c="magenta", alpha=0.9, label="Graspable volume", zorder=5)
    else:
        print(f"    (no graspable-volume points to plot for {'+'.join(group)})")

    group_label = "+".join(group)
    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
    ax.set_title(f"Opposable grasp volume — {group_label}")
    ax.legend(markerscale=5)
    _set_axes_equal(ax)
    plt.tight_layout()
    fname = f"graspable_volume_{'_'.join(group)}.png"
    plt.savefig(os.path.join(ROOT, SAVE_FOLDER, fname), dpi=150)
    _show_and_close(fig)


if __name__ == "__main__":
    main()
