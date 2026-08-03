"""
Finger data structures and setup.
Replaces MATLAB's Finger_Struct.m.

A hand is described as a set of named fingers, each consisting of one or more
FingerLink objects (one per URDF body/link that belongs to that finger).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .coupling import CouplingInfo, build_coupling_matrix
from .mesh_utils import mesh_to_points
from .urdf_parser import LinkProperty, MimicProperty


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FingerLink:
    """
    One URDF link belonging to a finger.

    Mirrors the fields of MATLAB's Finger struct array element:
      Bodies, Mesh, Points, Surf_Norm, Contact_Assumption, Collision_Size,
      Body_id, Link_id, Joint_ids, Joint_Limits, Actuated_Joint_ids,
      Coup_matrix, Rot_Offset, Trans_Offset.
    """
    body_name: str                                     # URDF link name
    mesh_file: Optional[str] = None                    # path from URDF <visual>
    points: Optional[np.ndarray] = None               # (N,3) surface pts (world)
    surf_norm: Optional[np.ndarray] = None            # (N,3) unit normals
    sharpness: Optional[np.ndarray] = None            # (N,) sharpness score [0,1]
    contact_assumption: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )                                                  # [mu, mu_tor]
    collision_size: int = 1
    body_id: int = 0        # pinocchio frame id for this link
    link_id: int = 1        # sequential link number within the finger
    joint_id: int = 0       # 0-indexed position in full q; 0 means fixed/unset
    joint_limits: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )                                                  # [lower, upper]
    actuated_joint_ids: list[int] = field(default_factory=list)
    coup_matrix: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )                                                  # [slope, offset]
    rot_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )                                                  # RPY rad (from URDF visual origin)
    trans_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )                                                  # XYZ m (from URDF visual origin)

    # internal: joint name (used during build, not needed after)
    _joint_name: str = field(default="", repr=False)


@dataclass
class FingerSet:
    """
    A named group of FingerLink objects representing one finger.

    Equivalent to one of MATLAB's Thumb/Index/Middle/Ring/Pinkie/Palm arrays.
    """
    name: str                    # e.g. "Thumb", "Index"
    links: list[FingerLink] = field(default_factory=list)

    def __iter__(self):
        return iter(self.links)

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        return self.links[idx]

    @property
    def joint_ids(self) -> list[int]:
        return [lnk.joint_id for lnk in self.links]

    @property
    def actuated_joint_ids_flat(self) -> list[int]:
        ids: list[int] = []
        for lnk in self.links:
            ids.extend(lnk.actuated_joint_ids)
        return sorted(set(ids))

    def is_empty(self) -> bool:
        return len(self.links) == 0 or self.links[0].body_name == "none"


# ---------------------------------------------------------------------------
# Building finger sets from a pinocchio model + URDF metadata
# ---------------------------------------------------------------------------

def build_finger_set(
    finger_name: str,
    body_names: list[str],
    contact_assumptions: np.ndarray,
    model,
    link_props: list[LinkProperty],
) -> FingerSet:
    """
    Build a FingerSet from a list of body names.

    Args:
        finger_name:         e.g. "Thumb".
        body_names:          Ordered list of URDF link names belonging to this finger.
                             Pass ["none"] for fingers not present on the hand.
        contact_assumptions: (N, 2) array of [mu, mu_tor] per link.
        model:               Loaded pinocchio model.
        link_props:          From urdf_parser.parse_mimic_and_offsets().

    Returns:
        FingerSet with body metadata filled in (mesh not yet loaded).
    """
    if body_names == ["none"]:
        return FingerSet(name=finger_name, links=[
            FingerLink(body_name="none")
        ])

    link_prop_by_name = {lp.body: lp for lp in link_props}

    links: list[FingerLink] = []
    for i, bname in enumerate(body_names):
        lp = link_prop_by_name.get(bname)
        rot_offset = lp.mesh_rpy if lp is not None else np.zeros(3)
        trans_offset = lp.mesh_xyz if lp is not None else np.zeros(3)
        mesh_file = lp.mesh_file if lp is not None else None

        # Pinocchio frame id for this link
        try:
            fid = model.getFrameId(bname)
        except Exception:
            fid = 0

        mu = float(contact_assumptions[i, 0]) if i < len(contact_assumptions) else 0.0
        mu_tor = float(contact_assumptions[i, 1]) if i < len(contact_assumptions) else 0.0

        links.append(FingerLink(
            body_name=bname,
            mesh_file=mesh_file,
            contact_assumption=np.array([mu, mu_tor]),
            body_id=fid,
            rot_offset=rot_offset,
            trans_offset=trans_offset,
        ))

    return FingerSet(name=finger_name, links=links)


def populate_joint_info(
    finger_set: FingerSet,
    model,
    coupling: CouplingInfo,
) -> None:
    """
    Fill in joint_id, joint_limits, actuated_joint_ids, and coup_matrix for
    each link in a FingerSet, using the pinocchio model and coupling info.

    Modifies finger_set.links in-place.
    """
    if finger_set.is_empty():
        return

    non_fix_names = list(model.names[1:])   # pinocchio joint order
    n_actuated = coupling.n_actuated

    link_counter = 0
    for lnk in finger_set.links:
        # Find the joint of this link in pinocchio
        # The parent joint of a frame is model.frames[fid].parent
        fid = lnk.body_id
        if fid == 0 or fid >= len(model.frames):
            continue

        frame = model.frames[fid]
        joint_id_pin = frame.parent  # pinocchio joint index (1-indexed in model.joints)

        if joint_id_pin == 0:
            # attached to universe → fixed-to-world, no movable joint
            continue

        joint = model.joints[joint_id_pin]
        joint_name = model.names[joint_id_pin]

        if joint_name not in non_fix_names:
            continue

        full_q_idx = non_fix_names.index(joint_name)  # 0-indexed position in full q
        lower = model.lowerPositionLimit[joint.idx_q]
        upper = model.upperPositionLimit[joint.idx_q]

        lnk.joint_id = full_q_idx
        lnk.joint_limits = np.array([lower, upper])
        lnk._joint_name = joint_name

        # Actuated joint ids and coupling slope/offset
        id_vector = np.zeros(len(non_fix_names))
        id_vector[full_q_idx] = 1.0
        C = coupling.matrix[:, :n_actuated]
        actuated_id_vector = C.T @ id_vector
        active_actuated = np.where(actuated_id_vector)[0]

        lnk.actuated_joint_ids = [coupling.actuated_joint_ids[j] for j in active_actuated]
        if len(active_actuated) > 0:
            slope = actuated_id_vector[active_actuated[0]]
            offset = coupling.matrix[full_q_idx, n_actuated]
            lnk.coup_matrix = np.array([slope, offset])

    # Assign link_id (increments when a new joint is encountered)
    link_counter = 1
    for i, lnk in enumerate(finger_set.links):
        if i == 0:
            lnk.link_id = 1
        else:
            prev = finger_set.links[i - 1]
            lnk.link_id = prev.link_id + (1 if lnk.joint_id != 0 else 0)


def load_meshes(
    finger_set: FingerSet,
    mesh_folder: str,
    n_samples: int,
    mesh_scale: float,
    urdf_base: Optional[str] = None,
) -> None:
    """
    Load and sample surface meshes for each link in a FingerSet.

    Modifies finger_set.links in-place, filling .points and .surf_norm.

    Args:
        mesh_folder: Directory to prepend when mesh_file is a bare filename.
        n_samples:   Number of barycentric surface samples.
        mesh_scale:  Scale factor for the point cloud.
        urdf_base:   Base directory of the URDF file (used to resolve
                     package:// URIs or relative paths in mesh_file).
    """
    if finger_set.is_empty():
        return

    for lnk in finger_set.links:
        if lnk.mesh_file is None:
            continue

        resolved = _resolve_mesh_path(lnk.mesh_file, mesh_folder, urdf_base)
        if resolved is None:
            print(f"  Could not resolve mesh for {lnk.body_name}: {lnk.mesh_file}")
            continue

        folder, fname = os.path.split(resolved)
        pts, nrms, sharp = mesh_to_points(folder + os.sep, fname, n_samples, mesh_scale)
        lnk.points = pts
        lnk.surf_norm = nrms
        lnk.sharpness = sharp


def _resolve_mesh_path(
    mesh_file: str,
    mesh_folder: str,
    urdf_base: Optional[str],
) -> Optional[str]:
    """
    Resolve a mesh filename that may be an absolute path, package:// URI,
    or relative path.
    """
    # Strip package:// prefix (common in ROS URDFs)
    if mesh_file.startswith("package://"):
        rest = mesh_file[len("package://"):]
        # rest = "package_name/meshes/..." — use the URDF base dir as root
        if urdf_base:
            candidate = os.path.join(urdf_base, rest)
            if os.path.exists(candidate):
                return candidate
        # Also try stripping the first path component (package name)
        parts = rest.split("/", 1)
        if len(parts) == 2 and urdf_base:
            candidate = os.path.join(urdf_base, parts[1])
            if os.path.exists(candidate):
                return candidate

    if os.path.isabs(mesh_file) and os.path.exists(mesh_file):
        return mesh_file

    # Bare filename in mesh_folder
    candidate = os.path.join(mesh_folder, mesh_file)
    if os.path.exists(candidate):
        return candidate

    # Relative to URDF base
    if urdf_base:
        candidate = os.path.join(urdf_base, mesh_file)
        if os.path.exists(candidate):
            return candidate

    # Basename only in mesh_folder (strips any directory prefix from mesh_file)
    basename = os.path.basename(mesh_file)
    candidate = os.path.join(mesh_folder, basename)
    if os.path.exists(candidate):
        return candidate

    return None


# ---------------------------------------------------------------------------
# Convenience: build an entire hand's fingers from spec dicts
# ---------------------------------------------------------------------------

def build_hand(
    finger_specs: dict[str, list[str]],
    contact_specs: dict[str, np.ndarray],
    model,
    link_props: list[LinkProperty],
    mimic_props: list[MimicProperty],
    mesh_folder: str,
    n_mesh_samples: int = 300,
    mesh_scale: float = 1.0,
    urdf_base: Optional[str] = None,
) -> tuple[dict[str, FingerSet], CouplingInfo]:
    """
    High-level entry point: build all FingerSets for a hand.

    Args:
        finger_specs:   {"Thumb": [body1, body2, ...], "Index": [...], ...}
        contact_specs:  {"Thumb": np.array([[mu, mu_tor], ...]), ...}
        model:          Pinocchio model.
        link_props:     From urdf_parser.parse_mimic_and_offsets().
        mimic_props:    From urdf_parser.parse_mimic_and_offsets().
        mesh_folder:    Directory containing STL/OBJ files.
        n_mesh_samples: Number of surface samples per mesh.
        mesh_scale:     Scale factor for mesh points.
        urdf_base:      URDF file directory (used to resolve package:// URIs).

    Returns:
        (fingers, coupling) where fingers is a dict name→FingerSet.
    """
    non_fix_joint_names = list(model.names[1:])
    coupling = build_coupling_matrix(non_fix_joint_names, mimic_props)

    fingers: dict[str, FingerSet] = {}
    for fname, bodies in finger_specs.items():
        contact = contact_specs.get(fname, np.zeros((len(bodies), 2)))
        fs = build_finger_set(fname, bodies, contact, model, link_props)
        populate_joint_info(fs, model, coupling)
        load_meshes(fs, mesh_folder, n_mesh_samples, mesh_scale, urdf_base)
        fingers[fname] = fs

    return fingers, coupling
