"""
Joint coupling matrix and actuated-to-full-jointspace conversion.
Replaces MATLAB's Coupling_matrix.m and Actuated_to_Full_Jointspace.m.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .urdf_parser import MimicProperty


@dataclass
class CouplingInfo:
    """Result of build_coupling_matrix — holds all joint space bookkeeping."""
    n_actuated: int
    actuated_joint_ids: list[int]       # 0-indexed positions in the full q vector
    non_actuated_joint_ids: list[int]   # 0-indexed positions in the full q vector
    actuated_joint_names: list[str]
    # Shape: (n_full_joints, n_actuated + 1)
    # Last column holds offsets for mimic joints.
    # full_q = matrix[:, :-1] @ actuated_q + matrix[:, -1]
    matrix: np.ndarray


def build_coupling_matrix(
    non_fix_joint_names: list[str],
    mimic_props: list[MimicProperty],
) -> CouplingInfo:
    """
    Build a coupling matrix that maps actuated joints → all non-fixed joints.

    Replicates MATLAB's Coupling_matrix().

    The last column of the matrix stores constant offsets (for mimic joints
    with a non-zero offset). To convert:
        full_q = matrix[:, :-1] @ actuated_q + matrix[:, -1]
    which is equivalent to:
        full_q = matrix @ np.append(actuated_q, 1.0)

    Args:
        non_fix_joint_names: Ordered list of all non-fixed joint names as they
            appear in the pinocchio model (model.names[1:]).
        mimic_props: List of MimicProperty parsed from the URDF.

    Returns:
        CouplingInfo with the matrix and index bookkeeping.
    """
    mimic_joint_names = {mp.joint for mp in mimic_props}
    n_full = len(non_fix_joint_names)
    n_actuated = n_full - len(mimic_props)

    actuated_joint_names: list[str] = []
    actuated_joint_ids: list[int] = []
    non_actuated_joint_ids: list[int] = []

    for i, name in enumerate(non_fix_joint_names):
        if name in mimic_joint_names:
            non_actuated_joint_ids.append(i)
        else:
            actuated_joint_names.append(name)
            actuated_joint_ids.append(i)

    matrix = np.zeros((n_full, n_actuated + 1))

    if n_actuated == n_full:
        # Fully actuated — identity mapping
        np.fill_diagonal(matrix[:, :n_actuated], 1.0)
    else:
        mimic_by_name = {mp.joint: mp for mp in mimic_props}

        for i, name in enumerate(non_fix_joint_names):
            if name in actuated_joint_names:
                col = actuated_joint_names.index(name)
                matrix[i, col] = 1.0
            elif name in mimic_by_name:
                mp = mimic_by_name[name]
                if mp.parent in actuated_joint_names:
                    parent_col = actuated_joint_names.index(mp.parent)
                    matrix[i, parent_col] = mp.multiplier
                    matrix[i, n_actuated] = mp.offset
                else:
                    print(f"Warning: mimic parent '{mp.parent}' is not directly actuated.")

    return CouplingInfo(
        n_actuated=n_actuated,
        actuated_joint_ids=actuated_joint_ids,
        non_actuated_joint_ids=non_actuated_joint_ids,
        actuated_joint_names=actuated_joint_names,
        matrix=matrix,
    )


def actuated_to_full_jointspace(
    actuated_q: np.ndarray,
    coupling: CouplingInfo,
) -> np.ndarray:
    """
    Convert an actuated-joint configuration vector to the full joint vector.

    Replicates MATLAB's Actuated_to_Full_Jointspace().

    Args:
        actuated_q: 1-D array of actuated joint positions, length n_actuated.
        coupling:   CouplingInfo from build_coupling_matrix().

    Returns:
        full_q: 1-D array of all non-fixed joint positions.
    """
    augmented = np.append(actuated_q, 1.0)
    return coupling.matrix @ augmented
