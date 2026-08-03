"""
Mesh-to-point-cloud sampling with surface normals and sharpness ranking --
variant of mesh_utils.py with two behavioral changes:

  1. Sample density is specified per unit surface area (points_per_area)
     instead of as a fixed total point count, so meshes of very different
     sizes get comparably dense sampling instead of the same sample budget
     stretched thin or wasted.
  2. Points that fail the exterior-facing (concavity) test are no longer
     dropped. Every sampled point is returned, alongside a same-length
     array flagging which ones passed (1) or failed (0) that test, so the
     caller can decide what to do with the failing points instead of
     having them silently discarded.

Everything else -- barycentric area-weighted sampling, PCA-based normal
refinement, sharpness scoring, largest-connected-component filtering, final
sharpness rescaling -- is identical to mesh_utils.mesh_to_points(), and now
runs over the full (unfiltered) point set rather than just the points that
passed the concavity test.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import NearestNeighbors
import trimesh


def mesh_to_points_2(
    mesh_folder: str,
    mesh_name: str,
    points_per_area: float,
    mesh_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample surface points at a fixed density, estimate outward normals,
    score sharpness, and flag (without dropping) points that fail the
    exterior-facing test.

    Variant of mesh_utils.mesh_to_points() -- see the module docstring for
    the two differences.

    Args:
        mesh_folder:     Directory containing the mesh file.
        mesh_name:       Filename (e.g. "finger_tip.stl" or "palm.obj").
        points_per_area: Desired samples per unit area, in the mesh file's
                         own coordinate units (e.g. if the file is in
                         millimetres, this is points per mm^2, regardless
                         of mesh_scale -- mesh_scale only rescales the
                         returned points, it doesn't change the mesh
                         geometry used for sampling/area).
        mesh_scale:      Scale factor applied to the final point cloud.

    Returns:
        points:         (N, 3) float32 surface point cloud (scaled). N is
                        every sampled point -- none are dropped.
        normals:        (N, 3) float64 unit outward surface normals.
        sharpness:      (N,) float64 sharpness score in [0, 1] (0 = flat,
                        1 = sharpest point in this cloud).
        concavity_pass: (N,) int array, 1 where the point passed the
                        exterior-facing (concavity) test, 0 where it didn't.
        Returns four empty arrays if the mesh cannot be read.
    """
    ext = Path(mesh_name).suffix.lower()
    mesh_path = os.path.join(mesh_folder, mesh_name)

    if ext not in (".stl", ".obj"):
        print(f"  mesh_to_points_2: unsupported format '{ext}', skipping.")
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0, dtype=int)

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
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0, dtype=int)

    n_samples = max(1, round(total_area * points_per_area))

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

    # ---- concavity test: flag (don't drop) exterior-facing points -----------
    try:
        dt = Delaunay(vertices)
    except Exception:
        dt = None

    # test_len is relative to the sample cloud's own scale, not a fixed
    # constant, so it works across meshes of very different sizes.
    max_dist = pdist(sample_pts).max() if n_samples > 1 else 0.0
    test_len = max_dist / 50.0

    concavity_pass = np.ones(n_samples, dtype=int)
    if dt is not None:
        for i in range(n_samples):
            test_pt = sample_pts[i] + test_len * sample_normals_raw[i]
            inside = dt.find_simplex(test_pt) >= 0
            if inside:
                concavity_pass[i] = 0

    all_pts = sample_pts
    all_norms = sample_normals_raw

    # ---- PCA-based normal refinement + sharpness scoring ---------------------
    n_pts = len(all_pts)
    k = min(20, n_pts - 1)
    if k >= 1:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(all_pts)
        _, indices = nbrs.kneighbors(all_pts)

        centroid = all_pts.mean(axis=0)
        refined_norms = np.zeros_like(all_pts)
        sharpness = np.zeros(n_pts)

        for i in range(n_pts):
            neighbors = all_pts[indices[i]]
            cov = np.cov(neighbors.T)
            # eigh returns eigenvalues ascending; guard tiny negative values
            # from numerical noise before using them as a variation ratio.
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 0, None)
            normal = eigvecs[:, 0]  # eigenvector for smallest eigenvalue
            r = all_pts[i] - centroid
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
        refined_norms = all_norms
        sharpness = np.zeros(n_pts)

    # ---- keep largest connected component -----------------------------------
    if n_pts > 1:
        D = cdist(all_pts, all_pts)
        np.fill_diagonal(D, np.inf)
        nearest_dist = D.min(axis=1)
        mean_spacing = nearest_dist.mean()
        epsilon = 40 * mean_spacing

        adjacency = D < epsilon
        filtered_pts, filtered_norms, filtered_sharpness, filtered_pass = _largest_component(
            all_pts, refined_norms, sharpness, concavity_pass, adjacency
        )
    else:
        filtered_pts, filtered_norms, filtered_sharpness, filtered_pass = (
            all_pts, refined_norms, sharpness, concavity_pass
        )

    # ---- rescale sharpness to [0, 1] over the kept points --------------------
    shifted = filtered_sharpness - filtered_sharpness.min()
    shifted_max = shifted.max()
    final_sharpness = shifted / shifted_max if shifted_max > 0 else shifted

    return (
        (mesh_scale * filtered_pts).astype(np.float32),
        filtered_norms,
        final_sharpness,
        filtered_pass,
    )


def _largest_component(
    pts: np.ndarray,
    norms: np.ndarray,
    sharpness: np.ndarray,
    concavity_pass: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pts/norms/sharpness/concavity_pass belonging to the largest connected component."""
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
        return pts, norms, sharpness, concavity_pass

    largest = max(components, key=len)
    idx = np.array(largest)
    return pts[idx], norms[idx], sharpness[idx], concavity_pass[idx]
