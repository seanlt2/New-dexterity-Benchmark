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
    ~/miniconda3/bin/python3 opposability_calculation.py
    (or: conda activate base && python3 opposability_calculation.py)

To switch hands, uncomment the desired config block and comment out the active one.

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

import os
import pickle
import threading

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
# Hand configurations — uncomment the block you want to run
# ─────────────────────────────────────────────────────────────────────────────

# ## Soft Hand
# FINGER_BODIES = {
#     "Thumb":  ["CMC", "Link1_Thumb", "Fingertip_Thumb"],
#     "Index":  ["Abd_Add_Index", "Link1_Index", "Fingertip_Index"],
#     "Middle": ["none"],
#     "Ring":   ["none"],
#     "Pinkie": ["Abd_Add_Pinkie", "Link1_Pinkie", "Fingertip_Pinkie"],
#     "Palm":   ["none"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/URDF_v4_Right/urdf/URDF_v4_Right.urdf"
# MESH_FOLDER  = "URDF_Files/URDF_v4_Right/meshes/"
# SAVE_FOLDER  = "Opposability/URDF_v4_Right"
# HOME_ACTUATED = None   # use zero/neutral

## Leap Hand
FINGER_BODIES = {
    "Thumb":  ["thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip"],
    "Index":  ["mcp_joint", "pip", "dip", "fingertip"],
    "Middle": ["mcp_joint_2", "pip_2", "dip_2", "fingertip_2"],
    "Ring":   ["none"],
    "Pinkie": ["mcp_joint_3", "pip_3", "dip_3", "fingertip_3"],
    "Palm":   ["palm_lower"],
}
MESH_SCALE   = 1.0
URDF_FILE    = "URDF_Files/leap_hand/leap_hand_right_stl.urdf"
MESH_FOLDER  = "URDF_Files/leap_hand/meshes/visual/"
SAVE_FOLDER  = "Opposability/leap_hand"
HOME_ACTUATED = None

# ## Schunk Hand
# FINGER_BODIES = {
#     "Thumb":  ["right_hand_a", "right_hand_b", "right_hand_c"],
#     "Index":  ["right_hand_l", "right_hand_p", "right_hand_t"],
#     "Middle": ["right_hand_k", "right_hand_o", "right_hand_s"],
#     "Ring":   ["right_hand_j", "right_hand_n", "right_hand_r"],
#     "Pinkie": ["right_hand_i", "right_hand_m", "right_hand_q"],
#     "Palm":   ["right_hand_e1", "right_hand_z", "right_hand_e2",
#                "right_hand_virtual_k", "right_hand_virtual_l",
#                "right_hand_virtual_i", "right_hand_virtual_j"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/schunk_hand/schunk_svh_hand_right_stl.urdf"
# MESH_FOLDER  = "URDF_Files/schunk_hand/meshes/visual/"
# SAVE_FOLDER  = "Opposability/schunk_hand"
# HOME_ACTUATED = None

# ## Barrett Hand
# FINGER_BODIES = {
#     "Thumb":  ["finger_3_med_link", "finger_3_dist_link"],
#     "Index":  ["finger_1_prox_link", "finger_1_med_link", "finger_1_dist_link"],
#     "Middle": ["none"],
#     "Ring":   ["none"],
#     "Pinkie": ["finger_2_prox_link", "finger_2_med_link", "finger_2_dist_link"],
#     "Palm":   ["base_link"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/barrett_hand/bhand_model_stl.urdf"
# MESH_FOLDER  = "URDF_Files/barrett_hand/meshes/visual/"
# SAVE_FOLDER  = "Opposability/barrett_hand"
# HOME_ACTUATED = None

# ## Shadow Hand
# FINGER_BODIES = {
#     "Thumb":  ["thbase", "thproximal", "thhub", "thmiddle", "thdistal"],
#     "Index":  ["ffknuckle", "ffproximal", "ffmiddle", "ffdistal"],
#     "Middle": ["mfknuckle", "mfproximal", "mfmiddle", "mfdistal"],
#     "Ring":   ["rfknuckle", "rfproximal", "rfmiddle", "rfdistal"],
#     "Pinkie": ["lfmetacarpal", "lfknuckle", "lfproximal", "lfmiddle", "lfdistal"],
#     "Palm":   ["palm"],
# }
# MESH_SCALE   = 0.001
# URDF_FILE    = "URDF_Files/shadow_hand/shadow_hand_right_coupled_stl.urdf"
# MESH_FOLDER  = "URDF_Files/shadow_hand/meshes/visual/"
# SAVE_FOLDER  = "Opposability/shadow_hand"
# HOME_ACTUATED = None

# ## Allegro Hand
# FINGER_BODIES = {
#     "Thumb":  ["link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"],
#     "Index":  ["link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip"],
#     "Middle": ["link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"],
#     "Ring":   ["none"],
#     "Pinkie": ["link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip"],
#     "Palm":   ["none"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/allegro_hand/allegro_hand_right_stl.urdf"
# MESH_FOLDER  = "URDF_Files/allegro_hand/meshes/visual/"
# SAVE_FOLDER  = "Opposability/allegro_hand"
# HOME_ACTUATED = None

# ## Allegro Hand 3 Fingers
# FINGER_BODIES = {
#     "Thumb":  ["link_6_0", "link_7_0", "link_8_0", "link_8_0_tip"],
#     "Index":  ["link_0_0", "link_1_0", "link_2_0", "link_2_0_tip"],
#     "Middle": ["none"],
#     "Ring":   ["none"],
#     "Pinkie": ["link_4_0", "link_4_0", "link_5_0", "link_5_0_tip"],
#     "Palm":   ["palm_link"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/allegro_hand_3_finger/allegro_hand_3F.urdf"
# MESH_FOLDER  = "URDF_Files/allegro_hand_3_finger/meshes/"
# SAVE_FOLDER  = "Opposability/allegro_hand_3_finger"
# HOME_ACTUATED = None

# ## Ability Hand
# FINGER_BODIES = {
#     "Thumb":  ["thumb_L1", "thumb_L2"],
#     "Index":  ["index_L1", "index_L2"],
#     "Middle": ["middle_L1", "middle_L2"],
#     "Ring":   ["ring_L1", "ring_L2"],
#     "Pinkie": ["pinky_L1", "pinky_L2"],
#     "Palm":   ["base", "thumb_base"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/ability_hand/ability_hand_right_stl.urdf"
# MESH_FOLDER  = "URDF_Files/ability_hand/meshes/visual/"
# SAVE_FOLDER  = "Opposability/ability_hand"
# HOME_ACTUATED = None

# ## Inspire Hand
# FINGER_BODIES = {
#     "Thumb":  ["thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal"],
#     "Index":  ["index_proximal", "index_intermediate"],
#     "Middle": ["middle_proximal", "middle_intermediate"],
#     "Ring":   ["ring_proximal", "ring_intermediate"],
#     "Pinkie": ["pinky_proximal", "pinky_intermediate"],
#     "Palm":   ["hand_base_link"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/inspire_hand/inspire_hand_right_stl.urdf"
# MESH_FOLDER  = "URDF_Files/inspire_hand/meshes/visual/"
# SAVE_FOLDER  = "Opposability/inspire_hand"
# HOME_ACTUATED = None

## D'Claw Gripper  ← ACTIVE
# FINGER_BODIES = {
#     "Thumb":  ["link_f1_1", "link_f1_2", "link_f1_3", "link_f1_head"],
#     "Index":  ["link_f2_1", "link_f2_2", "link_f2_3", "link_f2_head"],
#     "Middle": ["none"],
#     "Ring":   ["none"],
#     "Pinkie": ["link_f3_1", "link_f3_2", "link_f3_3", "link_f3_head"],
#     "Palm":   ["base_link"],
# }
# MESH_SCALE    = 1.0
# URDF_FILE     = "URDF_Files/dclaw_gripper/dclaw_gripper_stl.urdf"
# MESH_FOLDER   = "URDF_Files/dclaw_gripper/meshes/visual/"
# SAVE_FOLDER   = "Opposability/dclaw_gripper"
# # Home actuated pose — matches MATLAB: [(-pi/2), -1.7, -1.7, 0]
# HOME_ACTUATED = np.array([-np.pi / 2, -1.7, -1.7, 0.0])

# ## Ruka Hand
# FINGER_BODIES = {
#     "Thumb":  ["Thumb_MCP_Link", "Thumb_DIP_Link", "Thumb_PIP_Link"],
#     "Index":  ["Index_MCP_Link", "Index_DIP_Link", "Index_PIP_Link"],
#     "Middle": ["Middle_MCP_Link", "Middle_DIP_Link", "Middle_PIP_Link"],
#     "Ring":   ["Ring_MCP_Link", "Ring_DIP_Link", "Ring_PIP_Link"],
#     "Pinkie": ["Pinky_MCP_Link", "Pinky_DIP_Link", "Pinky_PIP_Link"],
#     "Palm":   ["Palm_Link"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/ruka_hand/ruka_hand.urdf"
# MESH_FOLDER  = "URDF_Files/ruka_hand/meshes/"
# SAVE_FOLDER  = "Opposability/ruka_hand"
# HOME_ACTUATED = None

# ## Sharpa Wave Hand
# FINGER_BODIES = {
#     "Thumb":  ["right_thumb_CMC_VL", "right_thumb_MC", "right_thumb_MCP_VL", "right_thumb_PP", "right_thumb_DP"],
#     "Index":  ["right_index_MCP_VL", "right_index_PP", "right_index_MP", "right_index_DP"],
#     "Middle": ["right_middle_MCP_VL", "right_middle_PP", "right_middle_MP", "right_middle_DP"],
#     "Ring":   ["right_ring_MCP_VL", "right_ring_PP", "right_ring_MP", "right_ring_DP"],
#     "Pinkie": ["right_pinky_MC", "right_pinky_MCP_VL", "right_pinky_PP", "right_pinky_MP", "right_pinky_DP"],
#     "Palm":   ["right_hand_C_MC"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/sharpa_wave_hand/right_hand/right_hand.urdf"
# MESH_FOLDER  = "URDF_Files/sharpa_wave_hand/right_hand/meshes/"
# SAVE_FOLDER  = "Opposability/sharpa_wave_hand/right_hand"
# HOME_ACTUATED = None

# ## Tesollo DG3F
# FINGER_BODIES = {
#     "Thumb":  ["l_dg_1_1", "l_dg_1_2", "l_dg_1_3", "l_dg_1_4","l_dg_1_tip"],
#     "Index":  ["l_dg_2_1", "l_dg_2_2", "l_dg_2_3", "l_dg_2_4","l_dg_2_tip"],
#     "Middle": ["none"],
#     "Ring":   ["none"],
#     "Pinkie": ["l_dg_3_1", "l_dg_3_2", "l_dg_3_3", "l_dg_3_4","l_dg_3_tip"],
#     "Palm":   ["l_dg_base"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/Tesollo_URDF/dg3fm/dg3f_m_stl.urdf"
# MESH_FOLDER  = "URDF_Files/Tesollo_URDF/dg3fm/meshes/"
# SAVE_FOLDER  = "Opposability/Tesollo/dg3fm"
# HOME_ACTUATED = None

# ## Tesollo DG4F
# FINGER_BODIES = {
#     "Thumb":  ["l_dg_1_inner","l_dg_1_1", "l_dg_1_2", "l_dg_1_3", "l_dg_1_4","l_dg_1_5","l_dg_1_tip"],
#     "Index":  ["l_dg_2_1", "l_dg_2_2", "l_dg_2_3", "l_dg_2_4","l_dg_2_tip"],
#     "Middle": ["l_dg_3_1", "l_dg_3_2", "l_dg_3_3", "l_dg_3_4","l_dg_3_tip"],
#     "Ring":   ["none"],
#     "Pinkie": ["l_dg_4_inner","l_dg_4_1", "l_dg_4_2", "l_dg_4_3", "l_dg_4_4","l_dg_4_5","l_dg_4_tip"],
#     "Palm":   ["l_dg_base"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/Tesollo_URDF/dg4f/dg3f.urdf"
# MESH_FOLDER  = "URDF_Files/Tesollo_URDF/dg4f/meshes/"
# SAVE_FOLDER  = "Opposability/Tesollo_URDF/dg4f"
# HOME_ACTUATED = None

# ## Tesollo DG5F
# FINGER_BODIES = {
#     "Thumb":  ["rl_dg_1_1", "rl_dg_1_2", "rl_dg_1_3", "rl_dg_1_4","rl_dg_1_tip"],
#     "Index":  ["rl_dg_2_1", "rl_dg_2_2", "rl_dg_2_3", "rl_dg_2_4","rl_dg_2_tip"],
#     "Middle": ["rl_dg_3_1", "rl_dg_3_2", "rl_dg_3_3", "rl_dg_3_4","rl_dg_3_tip"],
#     "Ring":   ["rl_dg_4_1", "rl_dg_4_2", "rl_dg_4_3", "rl_dg_4_4","rl_dg_4_tip"],
#     "Pinkie": ["rl_dg_5_1", "rl_dg_5_2", "rl_dg_5_3", "rl_dg_5_4","rl_dg_5_tip"],
#     "Palm":   ["rl_dg_base","rl_dg_palm"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/Tesollo_URDF/dg5f/dg5f_right.urdf"
# MESH_FOLDER  = "URDF_Files/Tesollo_URDF/dg5f/meshes/"
# SAVE_FOLDER  = "Opposability/Tesollo_URDF/dg5f"
# HOME_ACTUATED = None

# ## Tesollo DG5FS
# FINGER_BODIES = {
#     "Thumb":  ["link_1_1", "link_1_2", "link_1_3", "link_1_4","link_1_tip"],
#     "Index":  ["link_2_1", "link_2_2", "link_2_3", "link_2_4","link_2_tip"],
#     "Middle": ["link_3_1", "link_3_2", "link_3_3", "link_3_4","link_3_tip"],
#     "Ring":   ["link_4_1", "link_4_2", "link_4_3", "link_4_4","link_4_tip"],
#     "Pinkie": ["link_5_1", "link_5_2", "link_5_3", "link_5_4","link_5_tip"],
#     "Palm":   ["link_base"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/Tesollo_URDF/dg5fs/dg5fs_right.urdf"
# MESH_FOLDER  = "URDF_Files/Tesollo_URDF/dg5fs/meshes/"
# SAVE_FOLDER  = "Opposability/Tesollo_URDF/dg5fs"
# HOME_ACTUATED = None

# ## Tesollo DG5FS 15dof
# FINGER_BODIES = {
#     "Thumb":  ["link_1_1", "link_1_2", "link_1_3","link_1_tip"],
#     "Index":  ["link_2_1", "link_2_2", "link_2_3","link_2_tip"],
#     "Middle": ["link_3_1", "link_3_2", "link_3_3","link_3_tip"],
#     "Ring":   ["link_4_1", "link_4_2", "link_4_3","link_4_tip"],
#     "Pinkie": ["link_5_1", "link_5_2", "link_5_3","link_5_tip"],
#     "Palm":   ["link_base"],
# }
# MESH_SCALE   = 1.0
# URDF_FILE    = "URDF_Files/Tesollo_URDF/dg5fs/dg5fs_15dof_right.urdf"
# MESH_FOLDER  = "URDF_Files/Tesollo_URDF/dg5fs/meshes/"
# SAVE_FOLDER  = "Opposability/Tesollo_URDF/dg5fs_15dof"
# HOME_ACTUATED = None

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
# Seconds each figure window stays open before auto-closing on its own, so
# this script can run unattended (e.g. OPPOSABILITY_GROUPS with several
# entries pops up a new blocking figure per group) -- still closes early if
# you close a window yourself.
FIGURE_TIMEOUT_SECONDS = 60

# Which finger combinations to test for an opposable grasp -- each tuple is
# computed (and plotted/saved) separately, rather than requiring every
# active finger to simultaneously reach and oppose the same sphere. Not
# every reachable-and-opposable combination is a *useful* grasp for a given
# hand's mechanism (e.g. two fingers that can only flex in parallel won't
# form an effective pinch even if they can geometrically reach the same
# sphere from roughly opposite sides), so pick the combinations that make
# sense for the active hand. Names must be keys of FINGER_BODIES.

# # Ability Hand
# OPPOSABILITY_GROUPS: list[tuple[str, ...]] = [
#     ("Thumb", "Index"),
#     ("Thumb", "Middle"),
#     ("Thumb", "Ring"),
#     ("Thumb", "Pinkie"),
#     ("Thumb", "Index", "Middle"),
#     ("Thumb", "Index", "Ring"),
#     ("Thumb", "Index", "Pinkie"),
#     ("Thumb", "Middle", "Ring"),
#     ("Thumb", "Middle", "Pinkie"),
#     ("Thumb", "Ring", "Pinkie"),
#     ("Thumb", "Index", "Middle", "Ring"),
#     ("Thumb", "Index", "Middle", "Pinkie"),
#     ("Thumb", "Index", "Ring", "Pinkie"),
#     ("Thumb", "Middle", "Ring", "Pinkie"),
#     ("Thumb", "Index", "Middle", "Ring", "Pinkie"),
# ]

# Leap & Allegro Hands
OPPOSABILITY_GROUPS: list[tuple[str, ...]] = [
    ("Thumb", "Index"),
    ("Thumb", "Middle"),
    ("Thumb", "Pinkie"),
    ("Index", "Middle"),
    ("Index", "Pinkie"),
    ("Middle", "Pinkie"),
    ("Thumb", "Index", "Middle"),
    ("Thumb", "Index", "Pinkie"),
    ("Thumb", "Middle", "Pinkie"),
    ("Index", "Middle", "Pinkie"),
    ("Thumb", "Index", "Middle", "Pinkie"),
]

# # Tesollo DG3FM Hand
# OPPOSABILITY_GROUPS: list[tuple[str, ...]] = [
#     ("Thumb", "Index"),
#     ("Thumb", "Pinkie"),    
#     ("Index", "Pinkie"),
#     ("Thumb", "Index", "Pinkie"),
# ]

# Per-group sphere radius overrides (m) -- a Thumb+Pinkie pinch is only ever
# going to close around something much smaller than a full
# Thumb+Index+Middle+Ring+Pinkie power grasp, for example. Any group in
# OPPOSABILITY_GROUPS not listed here uses GRASP_SPHERE_RADIUS.

# #Ability Hand
# OPPOSABILITY_GROUP_RADII: dict[tuple[str, ...], float] = {
#     ("Thumb", "Index"): 0.011,
#     ("Thumb", "Middle"): 0.011,
#     ("Thumb", "Ring"): 0.011,
#     ("Thumb", "Pinkie"): 0.011,
#     ("Thumb", "Index", "Middle"): 0.016,
#     ("Thumb", "Index", "Ring"): 0.016,
#     ("Thumb", "Index", "Pinkie"): 0.016,
#     ("Thumb", "Middle", "Ring"): 0.016,
#     ("Thumb", "Middle", "Pinkie"): 0.016,
#     ("Thumb", "Ring", "Pinkie"): 0.016,
#     ("Thumb", "Index", "Middle", "Ring"): 0.022,
#     ("Thumb", "Index", "Middle", "Pinkie"): 0.022,
#     ("Thumb", "Index", "Ring", "Pinkie"): 0.022,
#     ("Thumb", "Middle", "Ring", "Pinkie"): 0.022,
#     ("Thumb", "Index", "Middle", "Ring", "Pinkie"): 0.027,
# }

# #Leap Hand
# OPPOSABILITY_GROUP_RADII: dict[tuple[str, ...], float] = {
#     ("Thumb", "Index"): 0.021,
#     ("Thumb", "Middle"): 0.021,
#     ("Thumb", "Pinkie"): 0.021,
#     ("Index", "Middle"): 0.021,
#     ("Index", "Pinkie"): 0.021,
#     ("Middle", "Pinkie"): 0.021,
#     ("Thumb", "Index", "Middle"): 0.032,
#     ("Thumb", "Index", "Pinkie"): 0.032,
#     ("Thumb", "Middle", "Pinkie"): 0.032,
#     ("Index", "Middle", "Pinkie"): 0.032,
#     ("Thumb", "Index", "Middle", "Pinkie"): 0.043,
# }

#Allegro Hand
OPPOSABILITY_GROUP_RADII: dict[tuple[str, ...], float] = {
    ("Thumb", "Index"): 0.016,
    ("Thumb", "Middle"): 0.016,
    ("Thumb", "Pinkie"): 0.016,
    ("Index", "Middle"): 0.016,
    ("Index", "Pinkie"): 0.016,
    ("Middle", "Pinkie"): 0.016,
    ("Thumb", "Index", "Middle"): 0.024,
    ("Thumb", "Index", "Pinkie"): 0.024,
    ("Thumb", "Middle", "Pinkie"): 0.024,
    ("Index", "Middle", "Pinkie"): 0.024,
    ("Thumb", "Index", "Middle", "Pinkie"): 0.032,
}


# # Tesollo DG3FM Hand
# OPPOSABILITY_GROUP_RADII: dict[tuple[str, ...], float] = {
#     ("Thumb", "Index"): 0.015,
#     ("Thumb", "Pinkie"): 0.015,
#     ("Index", "Pinkie"): 0.015,
#     ("Thumb", "Index", "Pinkie"): 0.023,
# }

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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    urdf_path = os.path.join(ROOT, URDF_FILE)
    mesh_folder = os.path.join(ROOT, MESH_FOLDER)
    urdf_base = os.path.dirname(urdf_path)

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

    for fname in ["Thumb", "Index", "Middle", "Ring", "Pinkie"]:
        fs = fingers.get(fname)
        if fs is None or fs.is_empty():
            workspaces[fname] = None
            continue

        print(f"Computing workspace: {fname}...")
        ws = finger_workspace(fs, act_space, coupling, model, data, q_home)
        print(f"  {len(ws):,} points")
        if len(ws) == 0:
            workspaces[fname] = None
            continue

        print(f"Generating alpha shape: {fname}...")
        alpha = generate_alphashape(ws)
        alphas[fname] = alpha
        vol = workspace_volume(alpha)
        print(f"  {fname} alpha shape done ({vol:.6g} m^3)")

        # Resample onto an evenly spaced grid (points inside the alpha
        # shape) -- the raw swept cloud's density varies hugely with the
        # sweep's kinematics, not with the finger's actual reachable shape,
        # which throws off anything downstream that measures local point
        # density (e.g. opposability.graspable_volume()'s default grid
        # spacing). Falls back to the raw swept points if no alpha shape
        # was found, or the grid resample came back empty.
        ws_uniform = sample_workspace_grid(alpha, resolution=WORKSPACE_GRID_RESOLUTION)
        if len(ws_uniform) > 0:
            print(f"  {fname} resampled to {len(ws_uniform):,} evenly-spaced points "
                  f"(from {len(ws):,} raw)")
            ws = ws_uniform   # drops the only reference to the raw cloud --
                               # free to be garbage-collected before the next
                               # finger's sweep starts.
        else:
            print(f"  {fname}: grid resample unavailable, keeping {len(ws):,} raw swept points")

        workspaces[fname] = ws
        finger_names_active.append(fname)

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

    for group in OPPOSABILITY_GROUPS:
        missing = [f for f in group if f not in finger_names_active]
        if missing:
            print(f"Skipping {'+'.join(group)}: {', '.join(missing)} not active for this hand.")
            continue

        radius = OPPOSABILITY_GROUP_RADII.get(group, GRASP_SPHERE_RADIUS)
        print(f"Computing opposable grasp volume: {'+'.join(group)} (r={radius:g} m)...")
        finger_workspaces = {fname: workspaces[fname] for fname in group}

        grasp_pts = compute_opposable_grasp_volume(
            finger_workspaces, palm_mesh, r=radius, mu=FRICTION_MU,
        )
        print(f"  {len(grasp_pts):,} candidate grasp-sphere positions")

        stem = "graspable_volume_" + "_".join(group)
        grasp_alpha = generate_alphashape(grasp_pts) if len(grasp_pts) > 0 else None
        grasp_vol = workspace_volume(grasp_alpha)
        _save_volume(grasp_dir, stem, grasp_pts, grasp_alpha)
        print(f"  Saved {stem} ({grasp_vol:.6g} m^3)")

        # ── 8. Visualisation ─────────────────────────────────────────────
        _plot_graspable_volume(workspaces, palm_points, grasp_pts, list(group))

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
