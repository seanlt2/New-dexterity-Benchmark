"""
URDF parser for mimic joint relationships, link visual offsets, and
per-link collision geometry.
Replaces MATLAB's Mimic_and_Offset_Extraction.m
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MimicProperty:
    """One mimic joint relationship: child = multiplier * parent + offset."""
    joint: str        # child joint name (the mimic)
    parent: str       # parent/leader joint name
    multiplier: float
    offset: float


@dataclass
class CollisionGeometry:
    """
    One <collision> element's geometry, in the link's own frame.

    kind is one of "mesh", "box", "sphere", "cylinder"; only the fields
    relevant to that kind are populated.
    """
    kind: str
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))       # translation
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))       # roll-pitch-yaw
    mesh_file: Optional[str] = None            # kind == "mesh"
    box_size: Optional[np.ndarray] = None      # kind == "box": (3,) full extents
    sphere_radius: Optional[float] = None      # kind == "sphere"
    cylinder_radius: Optional[float] = None    # kind == "cylinder"
    cylinder_length: Optional[float] = None    # kind == "cylinder"


@dataclass
class LinkProperty:
    """Visual origin offset and collision geometry for a URDF link."""
    body: str                                # link name
    mesh_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))  # translation
    mesh_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll-pitch-yaw
    mesh_file: Optional[str] = None          # path as written in URDF (may be package:// URI)
    collision_geometries: list[CollisionGeometry] = field(default_factory=list)


def parse_mimic_and_offsets(urdf_path: str) -> tuple[list[MimicProperty], list[LinkProperty]]:
    """
    Parse a URDF file and extract:
      - mimic joint relationships (child = multiplier * parent + offset)
      - per-link visual origin offsets (xyz, rpy) and mesh filename
      - per-link collision geometry (every <collision> element: mesh, box,
        sphere, or cylinder, with its own origin)

    Args:
        urdf_path: Absolute path to the .urdf file.

    Returns:
        (mimic_props, link_props)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # ---- mimic joints -------------------------------------------------------
    mimic_props: list[MimicProperty] = []
    for joint_el in root.iter("joint"):
        mimic_el = joint_el.find("mimic")
        if mimic_el is None:
            continue
        child_joint = joint_el.get("name", "")
        parent_joint = mimic_el.get("joint", "")
        multiplier = float(mimic_el.get("multiplier", "1.0"))
        offset = float(mimic_el.get("offset", "0.0"))
        print(f"  {child_joint} = {multiplier:.4f} * {parent_joint} + {offset:.4f}")
        mimic_props.append(MimicProperty(child_joint, parent_joint, multiplier, offset))

    # ---- link visual origins ------------------------------------------------
    link_props: list[LinkProperty] = []
    for link_el in root.iter("link"):
        name = link_el.get("name", "")
        prop = LinkProperty(body=name)

        visual_el = link_el.find("visual")
        if visual_el is not None:
            origin_el = visual_el.find("origin")
            if origin_el is not None:
                xyz_str = origin_el.get("xyz", "0 0 0")
                rpy_str = origin_el.get("rpy", "0 0 0")
                prop.mesh_xyz = np.fromstring(xyz_str, sep=" ")
                prop.mesh_rpy = np.fromstring(rpy_str, sep=" ")

            geom_el = visual_el.find("geometry")
            if geom_el is not None:
                mesh_el = geom_el.find("mesh")
                if mesh_el is not None:
                    prop.mesh_file = mesh_el.get("filename", None)

        # ---- collision geometry (may be several <collision> elements) -------
        for collision_el in link_el.findall("collision"):
            origin_el = collision_el.find("origin")
            xyz = np.fromstring(origin_el.get("xyz", "0 0 0"), sep=" ") if origin_el is not None else np.zeros(3)
            rpy = np.fromstring(origin_el.get("rpy", "0 0 0"), sep=" ") if origin_el is not None else np.zeros(3)

            geom_el = collision_el.find("geometry")
            if geom_el is None:
                continue

            mesh_el = geom_el.find("mesh")
            if mesh_el is not None:
                prop.collision_geometries.append(CollisionGeometry(
                    kind="mesh", xyz=xyz, rpy=rpy, mesh_file=mesh_el.get("filename", None),
                ))
                continue

            box_el = geom_el.find("box")
            if box_el is not None:
                prop.collision_geometries.append(CollisionGeometry(
                    kind="box", xyz=xyz, rpy=rpy,
                    box_size=np.fromstring(box_el.get("size", "0 0 0"), sep=" "),
                ))
                continue

            sphere_el = geom_el.find("sphere")
            if sphere_el is not None:
                prop.collision_geometries.append(CollisionGeometry(
                    kind="sphere", xyz=xyz, rpy=rpy,
                    sphere_radius=float(sphere_el.get("radius", "0")),
                ))
                continue

            cylinder_el = geom_el.find("cylinder")
            if cylinder_el is not None:
                prop.collision_geometries.append(CollisionGeometry(
                    kind="cylinder", xyz=xyz, rpy=rpy,
                    cylinder_radius=float(cylinder_el.get("radius", "0")),
                    cylinder_length=float(cylinder_el.get("length", "0")),
                ))
                continue

        link_props.append(prop)

    return mimic_props, link_props
