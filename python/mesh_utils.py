"""
Mesh-to-point-cloud sampling with surface normals and sharpness ranking.
Replaces MATLAB's Mesh2Pts_3.m.

Supports STL and OBJ files. For each mesh:
  1. Sample N points proportional to face area (barycentric sampling).
  2. Filter interior-facing points by testing if a point offset along its raw
     face normal is outside the mesh convex interior (Delaunay test). The
     offset distance (test_len) is set relative to the mesh's own scale:
     max pairwise distance between sample points, divided by 50.
  3. Estimate outward-pointing surface normals via PCA on local neighborhoods,
     and score each point's local sharpness from the same neighborhood (the
     ratio of the smallest to total eigenvalue spread: ~0 on flat patches,
     ~1 at sharp corners/edges).
  4. Keep only the largest connected component (removes isolated fragments),
     then rescale sharpness to [0, 1] over the kept points.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import NearestNeighbors
import trimesh


def mesh_to_points(
    mesh_folder: str,
    mesh_name: str,
    n_samples: int,
    mesh_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample surface points, estimate outward normals, and score sharpness.

    Replicates MATLAB's Mesh2Pts_3().

    Args:
        mesh_folder: Directory containing the mesh file.
        mesh_name:   Filename (e.g. "finger_tip.stl" or "palm.obj").
        n_samples:   Desired number of surface samples before filtering.
        mesh_scale:  Scale factor applied to the final point cloud.

    Returns:
        points:    (M, 3) float32 surface point cloud (scaled).
        normals:   (M, 3) float64 unit outward surface normals.
        sharpness: (M,) float64 sharpness score in [0, 1] (0 = flat, 1 = sharpest
                   point in this cloud), for ranking candidate contact points.
        Returns (empty, empty, empty) arrays if the mesh cannot be read.
    """
    ext = Path(mesh_name).suffix.lower()
    mesh_path = os.path.join(mesh_folder, mesh_name)

    if ext not in (".stl", ".obj"):
        print(f"  mesh_to_points: unsupported format '{ext}', skipping.")
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)

    mesh: trimesh.Trimesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # ---- area-weighted barycentric sampling ---------------------------------
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)

    total_area = face_areas.sum()
    if total_area == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)

    face_probs = face_areas / total_area
    chosen = np.random.choice(len(faces), size=n_samples, replace=True, p=face_probs)

    r1 = np.sqrt(np.random.rand(n_samples))
    r2 = np.random.rand(n_samples)
    u = 1 - r1
    v = r1 * (1 - r2)
    w = r1 * r2

    A = vertices[faces[chosen, 0]]
    B = vertices[faces[chosen, 1]]
    C = vertices[faces[chosen, 2]]
    sample_pts = u[:, None] * A + v[:, None] * B + w[:, None] * C

    # ---- per-face normals for STL (trimesh preserves them) ------------------
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    sample_normals_raw = face_normals[chosen]

    # ---- filter: keep only exterior-facing points (Delaunay hull test) ------
    try:
        dt = Delaunay(vertices)
    except Exception:
        dt = None

    # test_len is relative to the sample cloud's own scale, not a fixed
    # constant, so it works across meshes of very different sizes.
    max_dist = pdist(sample_pts).max() if n_samples > 1 else 0.0
    test_len = max_dist / 50.0

    valid_pts = []
    valid_norms = []

    for i in range(n_samples):
        if dt is not None:
            test_pt = sample_pts[i] + test_len * sample_normals_raw[i]
            inside = dt.find_simplex(test_pt) >= 0
            if inside:
                continue
        valid_pts.append(sample_pts[i])
        valid_norms.append(sample_normals_raw[i])

    if len(valid_pts) == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)

    valid_pts = np.array(valid_pts)
    valid_norms = np.array(valid_norms)

    # ---- PCA-based normal refinement + sharpness scoring ---------------------
    n_valid = len(valid_pts)
    k = min(20, n_valid - 1)
    if k >= 1:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(valid_pts)
        _, indices = nbrs.kneighbors(valid_pts)

        centroid = valid_pts.mean(axis=0)
        refined_norms = np.zeros_like(valid_pts)
        sharpness = np.zeros(n_valid)

        for i in range(n_valid):
            neighbors = valid_pts[indices[i]]
            cov = np.cov(neighbors.T)
            # eigh returns eigenvalues ascending; guard tiny negative values
            # from numerical noise before using them as a variation ratio.
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 0, None)
            normal = eigvecs[:, 0]  # eigenvector for smallest eigenvalue
            r = valid_pts[i] - centroid
            if np.dot(normal, r) < 0:
                normal = -normal
            refined_norms[i] = normal

            # Surface variation: smallest-to-total eigenvalue ratio. ~0 on a
            # flat/planar patch, ~1/3 at an isotropic corner-like
            # neighborhood; rescaled to [0, 1] as a sharpness score.
            eigval_sum = eigvals.sum()
            surf_variation = eigvals[0] / eigval_sum if eigval_sum > 0 else 0.0
            sharpness[i] = min(3 * surf_variation, 1.0)
    else:
        refined_norms = valid_norms
        sharpness = np.zeros(n_valid)

    # ---- keep largest connected component -----------------------------------
    if n_valid > 1:
        D = cdist(valid_pts, valid_pts)
        np.fill_diagonal(D, np.inf)
        nearest_dist = D.min(axis=1)
        mean_spacing = nearest_dist.mean()
        epsilon = 40 * mean_spacing

        adjacency = D < epsilon
        filtered_pts, filtered_norms, filtered_sharpness = _largest_component(
            valid_pts, refined_norms, sharpness, adjacency
        )
    else:
        filtered_pts, filtered_norms, filtered_sharpness = (
            valid_pts, refined_norms, sharpness
        )

    # ---- rescale sharpness to [0, 1] over the kept points --------------------
    shifted = filtered_sharpness - filtered_sharpness.min()
    shifted_max = shifted.max()
    final_sharpness = shifted / shifted_max if shifted_max > 0 else shifted

    return (
        (mesh_scale * filtered_pts).astype(np.float32),
        filtered_norms,
        final_sharpness,
    )


def _largest_component(
    pts: np.ndarray,
    norms: np.ndarray,
    sharpness: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pts/norms/sharpness belonging to the largest connected component."""
    n = len(pts)
    visited = np.zeros(n, dtype=bool)
    components: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            neighbors = np.where(adjacency[node])[0]
            stack.extend(neighbors.tolist())
        components.append(component)

    if not components:
        return pts, norms, sharpness

    largest = max(components, key=len)
    idx = np.array(largest)
    return pts[idx], norms[idx], sharpness[idx]
