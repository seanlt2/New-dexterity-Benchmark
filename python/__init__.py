"""
New Dexterity Benchmark — Python port of the MATLAB implementation.

Modules
-------
urdf_parser      parse_mimic_and_offsets()
coupling         build_coupling_matrix(), actuated_to_full_jointspace()
mesh_utils       mesh_to_points()
mesh_utils_2     mesh_to_points_2()
collision        build_collision_mesh(), build_visual_mesh()
kinematics       load_model(), compute_fk(), get_transform(),
                 points_transform(), points_and_normals_transform(),
                 compute_geometric_jacobian(), contact_point_jacobian(),
                 true_jacobian()
finger           FingerLink, FingerSet, build_hand()
workspace        finger_workspace(), generate_alphashape(), sample_workspace_grid(),
                 workspace_volume(), save_finger_workspace(), finger_workspace_transforms(),
                 save/load_finger_workspace_transforms()
opposability     graspable_volume(), filter_palm_collisions(),
                 compute_opposable_grasp_volume()
contact          closest_two_points(), object_reference_frame()
grasp            contact_wrench(), grasp_wrench(), force_closure_check()
force_ellipsoid  force_ellipsoid(), force_ellipsoid_vector_correction(),
                 force_manipulation_evaluation(), force_ellipsoid_common_basis()
grasp_selection  finger_pose_selection(), select_diverse_grasps(), grasp_2_finger()
manipulation_capacity  manipulation_capacity_region(),
                 sample_weighted_without_replacement()
dynamics         compute_mass_matrix(), compute_coriolis_matrix(),
                 mass_and_coriolis_from_urdf() -- not used anywhere yet, see
                 its module docstring.
"""

from .urdf_parser import (
    parse_mimic_and_offsets,
    MimicProperty,
    LinkProperty,
    CollisionGeometry,
)
from .coupling import (
    build_coupling_matrix,
    actuated_to_full_jointspace,
    CouplingInfo,
)
from .mesh_utils import mesh_to_points
from .mesh_utils_2 import mesh_to_points_2
from .collision import build_collision_mesh, build_visual_mesh
from .kinematics import (
    load_model,
    get_non_fixed_joint_names,
    get_joint_limits,
    compute_fk,
    get_transform,
    get_transform_by_id,
    points_transform,
    points_transform_precomputed,
    points_and_normals_transform,
    normals_transform,
    compute_geometric_jacobian,
    contact_point_jacobian,
    true_jacobian,
)
from .finger import (
    FingerLink,
    FingerSet,
    build_finger_set,
    populate_joint_info,
    load_meshes,
    build_hand,
)
from .workspace import (
    finger_workspace,
    generate_alphashape,
    sample_workspace_grid,
    workspace_volume,
    save_finger_workspace,
    FingerWorkspaceTransforms,
    finger_workspace_transforms,
    save_finger_workspace_transforms,
    load_finger_workspace_transforms,
)
from .opposability import (
    graspable_volume,
    filter_palm_collisions,
    compute_opposable_grasp_volume,
)
from .contact import (
    closest_two_points,
    object_reference_frame,
)
from .grasp import (
    contact_wrench,
    grasp_wrench,
    force_closure_check,
)
from .force_ellipsoid import (
    ForceEllipsoid,
    force_ellipsoid,
    force_ellipsoid_vector_correction,
    force_manipulation_evaluation,
    CommonBasisResult,
    force_ellipsoid_common_basis,
)
from .grasp_selection import (
    FingerPoseSelectionResult,
    finger_pose_selection,
    select_diverse_grasps,
    Grasp2FingerResult,
    grasp_2_finger,
)
from .manipulation_capacity import (
    ManipulationCapacityResult,
    manipulation_capacity_region,
    sample_weighted_without_replacement,
)
from .dynamics import (
    compute_mass_matrix,
    compute_coriolis_matrix,
    mass_and_coriolis_from_urdf,
)

__all__ = [
    # urdf_parser
    "parse_mimic_and_offsets", "MimicProperty", "LinkProperty", "CollisionGeometry",
    # coupling
    "build_coupling_matrix", "actuated_to_full_jointspace", "CouplingInfo",
    # mesh_utils
    "mesh_to_points",
    # mesh_utils_2
    "mesh_to_points_2",
    # collision
    "build_collision_mesh", "build_visual_mesh",
    # kinematics
    "load_model", "get_non_fixed_joint_names", "get_joint_limits",
    "compute_fk", "get_transform", "get_transform_by_id",
    "points_transform", "points_transform_precomputed",
    "points_and_normals_transform", "normals_transform",
    "compute_geometric_jacobian", "contact_point_jacobian", "true_jacobian",
    # finger
    "FingerLink", "FingerSet",
    "build_finger_set", "populate_joint_info", "load_meshes", "build_hand",
    # workspace
    "finger_workspace", "generate_alphashape", "sample_workspace_grid", "workspace_volume",
    "save_finger_workspace", "FingerWorkspaceTransforms",
    "finger_workspace_transforms", "save_finger_workspace_transforms",
    "load_finger_workspace_transforms",
    # opposability
    "graspable_volume", "filter_palm_collisions", "compute_opposable_grasp_volume",
    # contact
    "closest_two_points", "object_reference_frame",
    # grasp
    "contact_wrench", "grasp_wrench", "force_closure_check",
    # force_ellipsoid
    "ForceEllipsoid", "force_ellipsoid",
    "force_ellipsoid_vector_correction", "force_manipulation_evaluation",
    "CommonBasisResult", "force_ellipsoid_common_basis",
    # grasp_selection
    "FingerPoseSelectionResult", "finger_pose_selection", "select_diverse_grasps",
    "Grasp2FingerResult", "grasp_2_finger",
    # manipulation_capacity
    "ManipulationCapacityResult", "manipulation_capacity_region",
    "sample_weighted_without_replacement",
    # dynamics
    "compute_mass_matrix", "compute_coriolis_matrix", "mass_and_coriolis_from_urdf",
]
