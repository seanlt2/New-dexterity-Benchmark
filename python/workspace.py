"""
Per-finger workspace computation and alpha-shape volume.
Replaces MATLAB's Finger_Workspace_Efficient_2.m and Alphashape_generation.m.

The workspace is swept on a joint-space grid, applying the coupling matrix at
each configuration to get the full q, then FK-transforming each link's surface
point cloud into world frame. This module computes and persists the reachable
workspace volume of ONE finger at a time, from its selected surface points;
it does not compute multi-finger intersections (see opposability.py for the
sphere-based multi-finger grasp volume that replaces that workflow).

Step-halving scheme (matching MATLAB):
  actuated joint 1 → N_pts samples
  actuated joint 2 → N_pts samples
  actuated joint 3 → N_pts // 2 samples  (every 2nd index)
  actuated joint k → N_pts // 2^(k-2) samples  (every 2^(k-2)-th index)
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import pickle
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .coupling import CouplingInfo
from .finger import FingerSet
from .kinematics import (
    _mesh_offset_matrix,
    compute_fk,
    get_transform_by_id,
    points_transform_precomputed,
)


# ---------------------------------------------------------------------------
# Workspace computation
# ---------------------------------------------------------------------------

# Module globals for _workspace_chunk() -- the parallel path for
# finger_workspace()'s joint-space sweep (see n_workers below). Set once,
# right before the pool is created, so forked workers inherit model/data/
# coupling-matrix/etc via Linux's copy-on-write fork() semantics instead of
# having them re-pickled per chunk; only each chunk's small (start, end)
# combo-index range actually gets pickled through the task queue. `data` in
# particular is a mutable Pinocchio object that compute_fk() writes into on
# every call -- forking gives each worker its own private copy of it, so
# there's no cross-process aliasing even though every worker calls
# compute_fk() on "the same" object as far as the source code reads.
_POOL_FW_MODEL = None
_POOL_FW_DATA = None
_POOL_FW_Q_HOME: Optional[np.ndarray] = None
_POOL_FW_COUPLING_MATRIX: Optional[np.ndarray] = None
_POOL_FW_N_ACTUATED: Optional[int] = None
_POOL_FW_FINGER_ROWS: Optional[list[int]] = None
_POOL_FW_ACT_JOINT_SPACE: Optional[np.ndarray] = None
_POOL_FW_PT_RANGES: Optional[list[list[int]]] = None
_POOL_FW_PT_RANGE_LENS: Optional[list[int]] = None
_POOL_FW_LINKS: Optional[list[tuple]] = None
_POOL_FW_PTS_PER_CONFIG: Optional[int] = None
# Path/shape of the disk-backed memmap each worker writes its chunk's points
# directly into (see finger_workspace()'s parallel branch for why this is a
# memmap and not a returned array).
_POOL_FW_MMAP_PATH: Optional[str] = None
_POOL_FW_MMAP_SHAPE: Optional[tuple[int, int]] = None

_MIN_PARALLEL_CONFIGS = 2000


def _combo_indices(flat_index: int, lens: list[int]) -> list[int]:
    """
    Decode a flat combo index into itertools.product(*pt_ranges)'s per-axis
    indices -- i.e. the same mixed-radix counting product() does internally
    (right-most axis varies fastest), just computed directly for an
    arbitrary index instead of by stepping through every combo before it.
    Lets a chunk worker start at combo `start` in O(n_joints), not O(start).
    """
    n = len(lens)
    idxs = [0] * n
    rem = flat_index
    for k in range(n - 1, -1, -1):
        idxs[k] = rem % lens[k]
        rem //= lens[k]
    return idxs


def _workspace_chunk(index_range: tuple[int, int]) -> tuple[int, int]:
    """Runs in a worker process: sweeps combo indices [start, end) of
    _POOL_FW_PT_RANGES (the same combos itertools.product(*pt_ranges) would
    yield at those positions), writing each combo's points directly into its
    slot -- flat_index * _POOL_FW_PTS_PER_CONFIG -- of the shared
    _POOL_FW_MMAP_PATH memmap, and returning only (start, end) back through
    the task queue.

    Returning the chunk's own (potentially huge) point array instead was the
    first version of this function; measured on a real 320k-config, 718M-
    point Allegro Hand thumb sweep, it capped the parallel speedup at only
    ~1.7x on 10 workers, because every worker's multi-hundred-MB result still
    had to be pickled and piped back to the main process one at a time --
    that serialized transfer, not the FK/transform compute itself, was the
    actual bottleneck. Writing straight into a shared memmap lets every
    worker's writes happen concurrently and removes that transfer almost
    entirely, since only the tiny (start, end) tuple crosses the queue."""
    start, end = index_range
    model, data = _POOL_FW_MODEL, _POOL_FW_DATA
    q_home = _POOL_FW_Q_HOME
    coupling_matrix = _POOL_FW_COUPLING_MATRIX
    n_actuated = _POOL_FW_N_ACTUATED
    finger_rows = _POOL_FW_FINGER_ROWS
    act_space = _POOL_FW_ACT_JOINT_SPACE
    pt_ranges = _POOL_FW_PT_RANGES
    lens = _POOL_FW_PT_RANGE_LENS
    links = _POOL_FW_LINKS
    pts_per_config = _POOL_FW_PTS_PER_CONFIG

    mmap_arr = np.memmap(_POOL_FW_MMAP_PATH, dtype=np.float32, mode="r+",
                          shape=_POOL_FW_MMAP_SHAPE)

    for flat_index in range(start, end):
        idxs = _combo_indices(flat_index, lens)

        actuated_q = np.zeros(n_actuated)
        for k, row in enumerate(finger_rows):
            actuated_q[row] = act_space[row, pt_ranges[k][idxs[k]]]

        full_q = coupling_matrix @ np.append(actuated_q, 1.0)
        q = q_home.copy()
        q[:len(full_q)] = full_q

        compute_fk(model, data, q)

        cursor = flat_index * pts_per_config
        for frame_id, mesh_offset, pts in links:
            transformed = points_transform_precomputed(model, data, frame_id, mesh_offset, pts)
            n_new = len(transformed)
            mmap_arr[cursor:cursor + n_new] = transformed
            cursor += n_new

    # No explicit flush()/msync() here: that forces a synchronous write to
    # the backing file, which is unneeded and expensive -- the main process
    # reads the same file's pages back through the same OS page cache once
    # every worker has exited, which is already coherent without one.
    # Measured: adding flush() here cut the 1.66x speedup an unflushed
    # memmap gets down to ~1.3x.
    del mmap_arr
    return start, end


def finger_workspace(
    finger: FingerSet,
    actuated_joint_space: np.ndarray,
    coupling: CouplingInfo,
    model,
    data,
    q_home: np.ndarray,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Compute the reachable workspace point cloud for one finger.

    Replicates MATLAB's Finger_Workspace_Efficient_2().

    Args:
        finger:               FingerSet for one finger.
        actuated_joint_space: (n_actuated_total, N_pts) array.  Row i holds
                              N_pts evenly-spaced samples for actuated joint i
                              (all hand actuated joints, not just this finger's).
        coupling:             CouplingInfo from build_coupling_matrix().
        model, data:          Pinocchio model and data.
        q_home:               Home/neutral pinocchio configuration (length
                              model.nq).  Used as the base; only the finger's
                              joints are swept.
        n_workers:            Worker processes to sweep joint-space combos
                              with in parallel (each combo -- one FK solve
                              plus per-link point transform -- is
                              independent of every other). 1 (the default)
                              runs the original serial loop; below
                              _MIN_PARALLEL_CONFIGS combos, always runs
                              serially regardless of n_workers, since a
                              small finger's sweep isn't worth pool-startup
                              overhead.

    Returns:
        (M, 3) float32 array of workspace points in world frame.
    """
    if finger.is_empty():
        return np.empty((0, 3), dtype=np.float32)

    # Unique full-q indices for joints that actuate this finger
    finger_act_full_ids = sorted(set(finger.actuated_joint_ids_flat))
    if not finger_act_full_ids:
        return np.empty((0, 3), dtype=np.float32)

    # Map full-q index → row in actuated_joint_space
    full_id_to_row = {fid: i for i, fid in enumerate(coupling.actuated_joint_ids)}
    finger_rows = [full_id_to_row[fid] for fid in finger_act_full_ids
                   if fid in full_id_to_row]
    if not finger_rows:
        return np.empty((0, 3), dtype=np.float32)

    N_pts = actuated_joint_space.shape[1]
    n_joints = len(finger_rows)

    # Step sizes matching MATLAB's halving scheme
    # joint index k (0-based): step = 1 for k<2, else 2^(k-1)
    pt_ranges = []
    for k in range(n_joints):
        step = 1 if k < 2 else 2 ** (k - 1)
        pt_ranges.append(list(range(0, N_pts, step)))

    links_with_pts = [
        lnk for lnk in finger.links
        if lnk.points is not None and len(lnk.points) > 0
    ]
    if not links_with_pts:
        return np.empty((0, 3), dtype=np.float32)

    # Each link's frame id and mesh-offset transform are fixed -- independent
    # of q -- so resolve/build them once here rather than inside the sweep
    # loop below, which calls this once per (configuration, link) pair.
    # Rebuilding the mesh-offset matrix from scratch every call (via scipy's
    # Rotation.from_euler) was measured to account for 60% of this sweep's
    # total runtime; see points_transform_precomputed()'s docstring.
    frame_ids = {lnk.body_name: model.getFrameId(lnk.body_name) for lnk in links_with_pts}
    mesh_offsets = {
        lnk.body_name: _mesh_offset_matrix(lnk.rot_offset, lnk.trans_offset)
        for lnk in links_with_pts
    }

    pts_per_config = sum(len(lnk.points) for lnk in links_with_pts)
    max_configs = 1
    for r in pt_ranges:
        max_configs *= len(r)

    total = max_configs

    if n_workers > 1 and total >= _MIN_PARALLEL_CONFIGS:
        pt_range_lens = [len(r) for r in pt_ranges]
        link_tuples = [
            (frame_ids[lnk.body_name], mesh_offsets[lnk.body_name], lnk.points)
            for lnk in links_with_pts
        ]
        mmap_shape = (pts_per_config * max_configs, 3)

        global _POOL_FW_MODEL, _POOL_FW_DATA, _POOL_FW_Q_HOME, _POOL_FW_COUPLING_MATRIX
        global _POOL_FW_N_ACTUATED, _POOL_FW_FINGER_ROWS, _POOL_FW_ACT_JOINT_SPACE
        global _POOL_FW_PT_RANGES, _POOL_FW_PT_RANGE_LENS, _POOL_FW_LINKS, _POOL_FW_PTS_PER_CONFIG
        global _POOL_FW_MMAP_PATH, _POOL_FW_MMAP_SHAPE
        _POOL_FW_MODEL = model
        _POOL_FW_DATA = data
        _POOL_FW_Q_HOME = q_home
        _POOL_FW_COUPLING_MATRIX = coupling.matrix
        _POOL_FW_N_ACTUATED = coupling.n_actuated
        _POOL_FW_FINGER_ROWS = finger_rows
        _POOL_FW_ACT_JOINT_SPACE = actuated_joint_space
        _POOL_FW_PT_RANGES = pt_ranges
        _POOL_FW_PT_RANGE_LENS = pt_range_lens
        _POOL_FW_LINKS = link_tuples
        _POOL_FW_PTS_PER_CONFIG = pts_per_config
        _POOL_FW_MMAP_SHAPE = mmap_shape

        # ~4 chunks/worker, same granularity as opposability.py's LP-solve
        # pool, for reasonable load balancing without excessive per-task
        # dispatch overhead (per-config cost here is fairly uniform, unlike
        # that pool's highly variable per-point candidate counts).
        n_chunks = max(1, min(total, n_workers * 4))
        chunk_len = -(-total // n_chunks)  # ceil
        chunk_bounds = [(start, min(start + chunk_len, total))
                         for start in range(0, total, chunk_len)]

        # Disk-backed (not /dev/shm-backed) so a large sweep's memory
        # footprint stays exactly what it already is elsewhere in this
        # pipeline -- evictable page-cache pages, not a fixed RAM
        # reservation -- rather than adding a second, tmpfs-capped copy of
        # the same data on top. See _workspace_chunk()'s docstring for why
        # this replaced returning each chunk's array through the pool.
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="finger_workspace_", suffix=".dat")
        _POOL_FW_MMAP_PATH = tmp_path
        try:
            nbytes = mmap_shape[0] * mmap_shape[1] * np.dtype(np.float32).itemsize
            # posix_fallocate (not truncate/ftruncate), so every block is
            # actually allocated on disk up front: workers write to
            # different offsets of this same file concurrently, and a
            # sparse file would make each worker's first touch of a new
            # region extend the file's on-disk extent map -- an inode-level
            # metadata change that can serialize concurrent writers on the
            # same file even though their byte ranges never overlap.
            # Preallocating means every write instead lands on already-
            # mapped blocks, so it needs no metadata lock.
            os.posix_fallocate(tmp_fd, 0, nbytes)
            os.close(tmp_fd)

            n_done = 0
            with mp.get_context("fork").Pool(processes=min(n_workers, len(chunk_bounds))) as pool:
                for _start, _end in pool.imap_unordered(_workspace_chunk, chunk_bounds):
                    n_done += 1
                    print(f"  Workspace {100 * n_done / len(chunk_bounds):.1f}% complete "
                          f"({n_done}/{len(chunk_bounds)} chunks)")

            workspace = np.array(np.memmap(tmp_path, dtype=np.float32, mode="r", shape=mmap_shape))
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return workspace

    workspace = np.zeros((pts_per_config * max_configs, 3), dtype=np.float32)
    cursor = 0
    count = 0
    report_every = max(1, total // 20)

    for combo in itertools.product(*pt_ranges):
        # Build the full actuated_q vector (zeros for non-finger joints)
        actuated_q = np.zeros(coupling.n_actuated)
        for k, row in enumerate(finger_rows):
            actuated_q[row] = actuated_joint_space[row, combo[k]]

        # Full joint vector via coupling matrix: C @ [q_act; 1]
        full_q = coupling.matrix @ np.append(actuated_q, 1.0)

        # Pinocchio q: copy home config, overwrite non-fixed joint positions
        q = q_home.copy()
        q[:len(full_q)] = full_q

        compute_fk(model, data, q)

        for lnk in links_with_pts:
            pts = points_transform_precomputed(
                model, data,
                frame_ids[lnk.body_name],
                mesh_offsets[lnk.body_name],
                lnk.points,
            )
            n_new = len(pts)
            workspace[cursor:cursor + n_new] = pts
            cursor += n_new

        count += 1
        if count % report_every == 0:
            print(f"  Workspace {100 * count / total:.1f}% complete")

    return workspace[:cursor]


# ---------------------------------------------------------------------------
# Alpha-shape workspace volume
# ---------------------------------------------------------------------------

# Module globals for _voxelize_points()'s parallel path (see n_workers
# below), set right before the pool is created so forked workers inherit
# the (potentially hundreds-of-millions-of-points) array via Linux's
# copy-on-write fork() semantics instead of having it re-pickled per chunk.
_POOL_VOXELIZE_POINTS: Optional[np.ndarray] = None
_POOL_VOXEL_SIZE: Optional[float] = None


def _voxelize_chunk(index_range: tuple[int, int]) -> np.ndarray:
    """Runs in a worker process: round + dedupe one chunk of
    _POOL_VOXELIZE_POINTS onto the voxel grid, returning that chunk's own
    local unique voxel indices. Deduplication is associative (see
    _voxelize_points()'s docstring), so the caller still does one final
    merge across every chunk's result -- the same voxel can legitimately
    appear in more than one chunk's local output."""
    start, end = index_range
    chunk = _POOL_VOXELIZE_POINTS[start:end]
    idx = np.round(chunk.astype(np.float64) / _POOL_VOXEL_SIZE).astype(np.int64)
    return np.unique(idx, axis=0)


def _voxelize_points(
    points: np.ndarray,
    voxel_size: float,
    chunk_size: int = 2_000_000,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Round points onto a voxel_size grid and deduplicate, processing in
    chunks so this never needs a full-size float64 copy of the whole cloud
    at once.

    A dense joint-space sweep's raw point cloud can be hundreds of millions
    of points (multiple GB even as float32). The direct one-liner this
    replaces -- `np.round(points / voxel_size) * voxel_size` -- divides by
    a Python float, silently upcasting the WHOLE array to float64, and each
    of round/multiply allocates its own fresh full-size array on top of
    that; measured on a real hand with several large actuated fingers, this
    was enough to OOM-kill the process outright (~35 GB already resident
    from other fingers' raw clouds, then a single finger's voxelization
    needing up to ~3x that finger's float64 size again on top).

    Deduplicating per-chunk (as integer voxel indices, not floats) and
    merging keeps peak memory bounded by chunk_size plus however many
    *unique* voxels have been found so far -- which for a 1mm grid over a
    finger-sized workspace is typically a small fraction of the raw sample
    count -- rather than by the raw cloud's total size. Mathematically
    identical to deduplicating the whole array at once: the union of each
    chunk's uniques, uniqued again, equals the unique set of the whole.

    With n_workers > 1 (and more than one chunk), chunks are deduplicated
    independently across a worker pool, then merged once at the end --
    NOT the same incremental one-chunk-at-a-time merge the serial path
    below uses. That's a deliberate memory/speed tradeoff scoped to the
    parallel path only: it holds every chunk's (already-deduplicated, so
    typically much smaller than raw) result at once rather than folding
    each one in and discarding it immediately, bounded by chunk_size *
    n_chunks of *unique* voxels rather than raw points, not by the huge raw
    cloud this function exists to avoid holding onto. The n_workers<=1
    fallback is untouched from the original incremental version, to keep
    its memory profile exactly as before by default.

    Args:
        points:     Raw point cloud (metres).
        voxel_size: Voxel grid spacing (metres).
        chunk_size: How many points to round + dedupe per chunk.
        n_workers:  Worker processes to dedupe chunks with in parallel. 1
                   (the default) runs the original serial loop.

    Returns:
        (M, 3) float64 array of unique voxel-center coordinates.
    """
    if n_workers > 1 and len(points) > chunk_size:
        chunk_bounds = [(start, min(start + chunk_size, len(points)))
                         for start in range(0, len(points), chunk_size)]
        global _POOL_VOXELIZE_POINTS, _POOL_VOXEL_SIZE
        _POOL_VOXELIZE_POINTS, _POOL_VOXEL_SIZE = points, voxel_size
        with mp.get_context("fork").Pool(processes=min(n_workers, len(chunk_bounds))) as pool:
            chunk_uniques = list(pool.imap_unordered(_voxelize_chunk, chunk_bounds))
        if not chunk_uniques:
            return np.empty((0, 3), dtype=np.float64)
        accumulated = np.unique(np.vstack(chunk_uniques), axis=0)
        return accumulated.astype(np.float64) * voxel_size

    accumulated: Optional[np.ndarray] = None
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        idx = np.round(chunk.astype(np.float64) / voxel_size).astype(np.int64)
        idx = np.unique(idx, axis=0)
        accumulated = idx if accumulated is None else np.unique(
            np.vstack([accumulated, idx]), axis=0
        )

    if accumulated is None:
        return np.empty((0, 3), dtype=np.float64)
    return accumulated.astype(np.float64) * voxel_size


def generate_alphashape(workspace_points: np.ndarray, max_fit_points: int = 2000, n_workers: int = 1):
    """
    Generate an alpha shape that tightly bounds the workspace point cloud.

    Replicates MATLAB's Alphashape_generation() (the "critical alpha for a
    single connected region" it computes via alphaShape/criticalAlpha).

    Voxelises the cloud at 1 mm resolution (via the chunked/memory-bounded
    _voxelize_points(), see its docstring), then searches for the critical
    alpha ourselves rather than via alphashape.optimizealpha() -- see
    _fit_alphashape()'s docstring for why. That search is O(a few) fits of a
    Delaunay-based algorithm that doesn't scale well, so for large clouds it
    runs on a random subsample (a few thousand points already characterize
    a shape's boundary about as well as hundreds of thousands do); the
    voxelized (but not subsampled) points are what's actually returned to
    the caller as the workspace point cloud elsewhere in this module.

    Args:
        workspace_points: (N, 3) point cloud (metres).
        max_fit_points:   Cap on how many (voxelized) points the alpha-shape
                          search itself runs on.
        n_workers:        Worker processes for the voxelization step (see
                          _voxelize_points()) -- no effect on the critical-
                          alpha search itself, which is a small-input,
                          sequential bisection-style search and isn't
                          parallelized. 1 (the default) runs it serially.

    Returns:
        trimesh.Trimesh alpha shape, or None if the cloud is too small, or
        genuinely too flat/degenerate (e.g. a single-DOF finger's sweep,
        which is a thin 2-D ribbon in 3-D space) for any alpha to give a
        valid enclosed volume.
    """
    voxel_size = 0.001  # 1 mm
    voxelized = _voxelize_points(workspace_points, voxel_size, n_workers=n_workers)

    if len(voxelized) < 4:
        return None

    if len(voxelized) > max_fit_points:
        rng = np.random.default_rng(0)
        fit_points = voxelized[rng.choice(len(voxelized), max_fit_points, replace=False)]
    else:
        fit_points = voxelized

    return _fit_alphashape(fit_points)


# Module globals used by _contains_chunk() when sample_workspace_grid() runs
# its chunks across a worker pool (see n_workers below) -- set once, right
# before the pool is created, so forked workers inherit shape/grid_pts via
# Linux's copy-on-write fork() semantics instead of having them re-pickled
# per chunk. Only each chunk's small (start, end) index range actually gets
# pickled through the task queue.
_POOL_SHAPE = None
_POOL_GRID_PTS: Optional[np.ndarray] = None


def _contains_chunk(index_range: tuple[int, int]) -> tuple[int, np.ndarray]:
    """Runs in a worker process: shape.contains() for one chunk of
    _POOL_GRID_PTS, returning (start, that chunk's boolean mask) so the
    caller can place it back at the right offset in the full array."""
    start, end = index_range
    return start, _POOL_SHAPE.contains(_POOL_GRID_PTS[start:end])


def sample_workspace_grid(
    shape,
    resolution: float = 0.001,
    chunk_size: int = 20_000,
    verbose: bool = True,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Resample an alpha-shape volume onto a uniform 3-D grid.

    Builds a regular meshgrid over the shape's bounding box at `resolution`
    and keeps only the grid points that lie inside the shape. The raw
    joint-space-swept point cloud from finger_workspace() has wildly
    uneven density -- points bunch up near some configurations and thin
    out elsewhere, purely as an artifact of the sweep's kinematics, not of
    the finger's actual reachable geometry -- so anything measuring local
    point density downstream (e.g. opposability.graspable_volume()'s
    default candidate-grid spacing) is more meaningful against an evenly
    sampled cloud than the raw one.

    Replicates the meshgrid + inShape/point-in-alphashape step from the
    original MATLAB pipeline.

    `shape.contains()` is a ray-casting test whose cost (and, for some
    shapes, correctness -- see generate_alphashape()'s docstring) can
    degrade badly when handed a huge point batch at once; this queries it
    in bounded-size chunks instead, which also gives progress output for
    what can otherwise be a multi-minute call on a finger-sized volume at
    1mm resolution. With n_workers > 1, those same chunks (still capped at
    chunk_size each, for the correctness/performance reason above -- this
    doesn't change how big any single shape.contains() call is, only how
    many of them run at once) are dispatched across a worker pool instead
    of run one after another.

    Args:
        shape:      trimesh.Trimesh alpha shape, e.g. from generate_alphashape().
        resolution: Grid spacing (metres).
        chunk_size: How many grid points to test against the shape per
                   shape.contains() call.
        verbose:    Print progress every ~10 chunks.
        n_workers:  Worker processes to test chunks with in parallel. 1 (the
                   default) runs the original serial loop.

    Returns:
        (M, 3) float32 array of grid points inside the shape. Empty if
        shape is None, or if the shape is smaller than `resolution` in some
        dimension (no grid point happens to fall inside it).
    """
    if shape is None:
        return np.empty((0, 3), dtype=np.float32)

    lo, hi = shape.bounds
    axes = [np.arange(lo[i], hi[i] + resolution, resolution) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    n = len(grid_pts)
    inside = np.zeros(n, dtype=bool)
    chunk_bounds = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]
    n_chunks = len(chunk_bounds)

    if n_workers > 1 and n_chunks > 1:
        global _POOL_SHAPE, _POOL_GRID_PTS
        _POOL_SHAPE, _POOL_GRID_PTS = shape, grid_pts

        n_done = 0
        with mp.get_context("fork").Pool(processes=min(n_workers, n_chunks)) as pool:
            for start, mask in pool.imap_unordered(_contains_chunk, chunk_bounds):
                inside[start:start + len(mask)] = mask
                n_done += 1
                if verbose and (n_done % 10 == 0 or n_done == n_chunks):
                    print(f"    sample_workspace_grid: {n_done}/{n_chunks} chunks tested "
                          f"({100 * n_done / n_chunks:.0f}%)")
    else:
        for i, (start, end) in enumerate(chunk_bounds):
            inside[start:end] = shape.contains(grid_pts[start:end])
            if verbose and (i % 10 == 0 or end == n):
                print(f"    sample_workspace_grid: {end:,}/{n:,} grid points tested "
                      f"({100 * end / n:.0f}%)")

    return grid_pts[inside].astype(np.float32)


def _safe_alphashape(points: np.ndarray, alpha: float):
    """
    alphashape.alphashape(), but treats "no valid shape at this alpha" as a
    clean None instead of letting it crash or return a non-3-D result.

    A degenerate candidate alpha (too tight for this point cloud) can make
    alphashape's own call into trimesh.repair.fix_normals() raise an
    IndexError outright -- indexing an empty faces array -- rather than
    returning an empty/invalid shape as its docstring implies it should.
    alpha=0 is also a special case in this library: it can return a 2-D
    shapely Polygon instead of a 3-D trimesh.Trimesh, which isn't usable
    as a workspace volume either.
    """
    import alphashape as _alphashape
    import trimesh as _trimesh

    try:
        shape = _alphashape.alphashape(points, alpha)
    except Exception:
        return None
    if not isinstance(shape, _trimesh.Trimesh) or len(shape.faces) == 0:
        return None
    return shape


def _encloses_all_points(shape, points: np.ndarray, tol: float) -> bool:
    """
    True if `shape` is a single watertight region and every point in
    `points` lies on or inside it (within `tol`).

    Matches MATLAB alphaShape's criticalAlpha(shape,'all-points') criterion:
    a single region containing every input point -- not just "some valid
    mesh", which an alpha-shape can be while still leaving individual points
    (e.g. ones on a thin protrusion) stranded outside it.
    """
    import trimesh as _trimesh

    if shape is None or not shape.is_watertight:
        return False
    dist = _trimesh.proximity.signed_distance(shape, points)
    return bool(np.all(dist >= -tol))


def _fit_alphashape(points: np.ndarray, max_steps: int = 40):
    """
    Find the critical alpha for `points`: the tightest alpha shape that
    still contains every point in a single connected region (MATLAB's
    alphaShape criticalAlpha(shape,'all-points'), which is what
    Alphashape_generation.m used).

    This doesn't use alphashape.optimizealpha(): that function always tests
    an astronomically large alpha (sys.float_info.max) first, as a sanity
    upper bound, before bisecting downward. For the densely, regularly
    sampled point clouds this project's joint-space sweeps produce, that
    first probe reliably crashes (see _safe_alphashape) rather than
    returning cleanly, which optimizealpha doesn't handle -- so the search
    never gets to the small, perfectly good alpha values below it.

    Instead: start from the convex hull (alpha=0), which by definition
    contains every point in one region, then tighten (alpha x1.5 per step)
    for as long as the shape stays a single region containing every point --
    stopping and returning the last shape that did, the moment a tighter
    alpha would either fragment the shape, crash, or leave any point
    stranded outside it.
    """
    import trimesh as _trimesh

    try:
        # alphashape.alphashape(points, 0) is documented as "safely returns
        # a convex hull", but in practice it can hand back a 2-D shapely
        # Polygon instead of a 3-D mesh -- go straight to trimesh's own
        # convex hull instead, which is unambiguously a 3-D Trimesh.
        hull = _trimesh.convex.convex_hull(points)
    except Exception:
        hull = None
    if hull is None or not isinstance(hull, _trimesh.Trimesh) or len(hull.faces) == 0:
        return None  # too small / too degenerate for even a convex hull

    tree = cKDTree(points)
    median_nn = float(np.median(tree.query(points, k=2)[0][:, 1]))
    tol = 2 * median_nn if median_nn > 0 else 1e-6

    if not _encloses_all_points(hull, points, tol):
        # Shouldn't happen for a true convex hull, but stay defensive.
        return hull

    best = hull
    alpha = 1.0 / (4 * median_nn) if median_nn > 0 else 1.0
    for _ in range(max_steps):
        candidate = _safe_alphashape(points, alpha)
        if not _encloses_all_points(candidate, points, tol):
            break
        best = candidate
        alpha *= 1.5

    # Rebuild fresh from vertices/faces before returning: after many rounds
    # of alphashape.alphashape() constructing intermediate meshes during the
    # search above, the winning mesh's cached spatial-index/proximity
    # structures can end up corrupted -- observed directly on a real
    # workspace shape, where even mesh.contains([mesh.centroid]) came back
    # False and proximity.signed_distance() raised an IndexError deep in
    # trimesh's nearby-faces lookup. A clean Trimesh(..., process=True)
    # forces every cache to be rebuilt from the (valid) topology, and fixes
    # both of those; a plausible cause of an OOM crash further downstream
    # too, since it's very likely the same corrupted structure that gets
    # queried against a large point batch rather than a small one.
    return _trimesh.Trimesh(vertices=best.vertices.copy(), faces=best.faces.copy(), process=True)


def workspace_volume(shape) -> float:
    """Return the volume (m³) of an alpha shape."""
    if shape is None:
        return 0.0
    if hasattr(shape, "volume"):
        return float(shape.volume)
    return 0.0


def save_finger_workspace(
    save_dir: str,
    finger_label: str,
    points: np.ndarray,
    shape=None,
) -> float:
    """
    Persist a computed finger workspace (points + alpha shape + volume) to disk.

    Writes:
        <finger_label>_wkspace_pts.npy    the raw point cloud
        <finger_label>_wkspace.csv        the raw point cloud, CSV
        <finger_label>_wkspace_alpha.pkl  the alpha shape (if provided)
        <finger_label>_wkspace_volume.json  {"volume_m3": ...}

    Args:
        save_dir:     Directory to write into (created if missing).
        finger_label: File stem, e.g. "Finger_1" or "Thumb".
        points:       (N, 3) workspace point cloud from finger_workspace().
        shape:        Alpha shape from generate_alphashape(), or None.

    Returns:
        The workspace volume (m^3) that was saved (0.0 if shape is None).
    """
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"{finger_label}_wkspace_pts.npy"), points)
    np.savetxt(os.path.join(save_dir, f"{finger_label}_wkspace.csv"), points, delimiter=",")

    vol = workspace_volume(shape)
    with open(os.path.join(save_dir, f"{finger_label}_wkspace_volume.json"), "w") as f:
        json.dump({"volume_m3": vol}, f)

    if shape is not None:
        with open(os.path.join(save_dir, f"{finger_label}_wkspace_alpha.pkl"), "wb") as f:
            pickle.dump(shape, f)

    return vol


# ---------------------------------------------------------------------------
# Per-body transform sweep (for later grasp synthesis)
# ---------------------------------------------------------------------------

@dataclass
class FingerWorkspaceTransforms:
    """
    The pose of every body in a finger, at every sampled workspace configuration.

    Mirrors MATLAB's Finger_Workspace_2() output struct: `Jointspace` plus one
    `Body_N` field per link. Here the per-body arrays are keyed by body name
    instead of positional Body_1/Body_2/... fields.
    """
    jointspace: np.ndarray                    # (n_configs, model.nq)
    body_transforms: dict[str, np.ndarray]    # body_name -> (n_configs, 4, 4)


# Module globals for _workspace_transforms_chunk() -- the parallel path for
# finger_workspace_transforms()'s joint-space sweep. Same rationale/pattern
# as finger_workspace()'s _POOL_FW_* globals above (fork-inherited
# read-only state, disk-backed memmap for the actual output instead of
# returning arrays through the pool -- see _workspace_chunk()'s docstring
# for why). Every sampled pose's jointspace row and every body's flattened
# 4x4 transform are packed into one wide memmap row per pose (columns
# [0:nq) = jointspace, [nq+16*i : nq+16*(i+1)) = body i's transform,
# raveled row-major) rather than one memmap file per array, purely to avoid
# managing N+1 temp files/paths for what's a single per-pose write.
_POOL_FWT_MODEL = None
_POOL_FWT_DATA = None
_POOL_FWT_Q_HOME: Optional[np.ndarray] = None
_POOL_FWT_COUPLING_MATRIX: Optional[np.ndarray] = None
_POOL_FWT_N_ACTUATED: Optional[int] = None
_POOL_FWT_FINGER_ROWS: Optional[list[int]] = None
_POOL_FWT_ACT_JOINT_SPACE: Optional[np.ndarray] = None
_POOL_FWT_PT_RANGES: Optional[list[list[int]]] = None
_POOL_FWT_PT_RANGE_LENS: Optional[list[int]] = None
_POOL_FWT_BODY_INFO: Optional[list[tuple]] = None  # [(frame_id, mesh_offset), ...]
_POOL_FWT_NQ: Optional[int] = None
_POOL_FWT_MMAP_PATH: Optional[str] = None
_POOL_FWT_MMAP_SHAPE: Optional[tuple[int, int]] = None


def _workspace_transforms_chunk(index_range: tuple[int, int]) -> tuple[int, int]:
    """Runs in a worker process: sweeps combo indices [start, end), writing
    each pose's jointspace row + every body's flattened transform directly
    into row `flat_index` of the shared _POOL_FWT_MMAP_PATH memmap. See
    _workspace_chunk()'s docstring -- same reasoning for writing into a
    memmap instead of returning arrays through the pool."""
    start, end = index_range
    model, data = _POOL_FWT_MODEL, _POOL_FWT_DATA
    q_home = _POOL_FWT_Q_HOME
    coupling_matrix = _POOL_FWT_COUPLING_MATRIX
    n_actuated = _POOL_FWT_N_ACTUATED
    finger_rows = _POOL_FWT_FINGER_ROWS
    act_space = _POOL_FWT_ACT_JOINT_SPACE
    pt_ranges = _POOL_FWT_PT_RANGES
    lens = _POOL_FWT_PT_RANGE_LENS
    body_info = _POOL_FWT_BODY_INFO
    nq = _POOL_FWT_NQ

    mmap_arr = np.memmap(_POOL_FWT_MMAP_PATH, dtype=np.float32, mode="r+",
                          shape=_POOL_FWT_MMAP_SHAPE)

    for flat_index in range(start, end):
        idxs = _combo_indices(flat_index, lens)

        actuated_q = np.zeros(n_actuated)
        for k, row in enumerate(finger_rows):
            actuated_q[row] = act_space[row, pt_ranges[k][idxs[k]]]

        full_q = coupling_matrix @ np.append(actuated_q, 1.0)
        q = q_home.copy()
        q[:len(full_q)] = full_q

        compute_fk(model, data, q)

        mmap_arr[flat_index, :nq] = q
        col = nq
        for frame_id, mesh_offset in body_info:
            T_world_link = get_transform_by_id(model, data, frame_id)
            mmap_arr[flat_index, col:col + 16] = (T_world_link @ mesh_offset).ravel()
            col += 16

    # No flush()/msync() -- see _workspace_chunk()'s identical comment.
    del mmap_arr
    return start, end


def finger_workspace_transforms(
    finger: FingerSet,
    actuated_joint_space: np.ndarray,
    coupling: CouplingInfo,
    model,
    data,
    q_home: np.ndarray,
    n_workers: int = 1,
) -> FingerWorkspaceTransforms:
    """
    Sweep a finger's actuated joints and record each body's world-frame
    transform at every sampled pose (rather than its transformed surface
    points), so a later grasp-synthesis step can place fingertip bodies at
    specific workspace configurations.

    Replicates MATLAB's Finger_Workspace_2().

    Args:
        finger:               FingerSet for one finger.
        actuated_joint_space: (n_actuated_total, N_pts) array, same layout as
                              finger_workspace().
        coupling:             CouplingInfo from build_coupling_matrix().
        model, data:          Pinocchio model and data.
        q_home:               Home/neutral pinocchio configuration.
        n_workers:            Worker processes to sweep joint-space combos
                              with in parallel -- see finger_workspace()'s
                              identical parameter. 1 (the default) runs the
                              original serial loop; below
                              _MIN_PARALLEL_CONFIGS combos, always runs
                              serially regardless of n_workers.

    Returns:
        FingerWorkspaceTransforms with one row per sampled pose.
    """
    empty = FingerWorkspaceTransforms(jointspace=np.empty((0, len(q_home))), body_transforms={})

    if finger.is_empty():
        return empty

    finger_act_full_ids = sorted(set(finger.actuated_joint_ids_flat))
    if not finger_act_full_ids:
        return empty

    full_id_to_row = {fid: i for i, fid in enumerate(coupling.actuated_joint_ids)}
    finger_rows = [full_id_to_row[fid] for fid in finger_act_full_ids
                   if fid in full_id_to_row]
    if not finger_rows:
        return empty

    N_pts = actuated_joint_space.shape[1]
    n_joints = len(finger_rows)

    # Step sizes matching MATLAB's halving scheme (same as finger_workspace())
    pt_ranges = []
    for k in range(n_joints):
        step = 1 if k < 2 else 2 ** (k - 1)
        pt_ranges.append(list(range(0, N_pts, step)))

    bodies = [lnk for lnk in finger.links if lnk.points is not None and len(lnk.points) > 0]
    if not bodies:
        return empty

    # Resolve/build once -- see finger_workspace()'s identical comment.
    frame_ids = {lnk.body_name: model.getFrameId(lnk.body_name) for lnk in bodies}
    mesh_offsets = {
        lnk.body_name: _mesh_offset_matrix(lnk.rot_offset, lnk.trans_offset)
        for lnk in bodies
    }

    max_configs = 1
    for r in pt_ranges:
        max_configs *= len(r)

    total = max_configs
    nq = len(q_home)

    if n_workers > 1 and total >= _MIN_PARALLEL_CONFIGS:
        pt_range_lens = [len(r) for r in pt_ranges]
        body_info = [(frame_ids[lnk.body_name], mesh_offsets[lnk.body_name]) for lnk in bodies]
        row_width = nq + 16 * len(bodies)
        mmap_shape = (total, row_width)

        global _POOL_FWT_MODEL, _POOL_FWT_DATA, _POOL_FWT_Q_HOME, _POOL_FWT_COUPLING_MATRIX
        global _POOL_FWT_N_ACTUATED, _POOL_FWT_FINGER_ROWS, _POOL_FWT_ACT_JOINT_SPACE
        global _POOL_FWT_PT_RANGES, _POOL_FWT_PT_RANGE_LENS, _POOL_FWT_BODY_INFO, _POOL_FWT_NQ
        global _POOL_FWT_MMAP_PATH, _POOL_FWT_MMAP_SHAPE
        _POOL_FWT_MODEL = model
        _POOL_FWT_DATA = data
        _POOL_FWT_Q_HOME = q_home
        _POOL_FWT_COUPLING_MATRIX = coupling.matrix
        _POOL_FWT_N_ACTUATED = coupling.n_actuated
        _POOL_FWT_FINGER_ROWS = finger_rows
        _POOL_FWT_ACT_JOINT_SPACE = actuated_joint_space
        _POOL_FWT_PT_RANGES = pt_ranges
        _POOL_FWT_PT_RANGE_LENS = pt_range_lens
        _POOL_FWT_BODY_INFO = body_info
        _POOL_FWT_NQ = nq
        _POOL_FWT_MMAP_SHAPE = mmap_shape

        n_chunks = max(1, min(total, n_workers * 4))
        chunk_len = -(-total // n_chunks)  # ceil
        chunk_bounds = [(start, min(start + chunk_len, total))
                         for start in range(0, total, chunk_len)]

        tmp_fd, tmp_path = tempfile.mkstemp(prefix="finger_workspace_transforms_", suffix=".dat")
        _POOL_FWT_MMAP_PATH = tmp_path
        try:
            nbytes = mmap_shape[0] * mmap_shape[1] * np.dtype(np.float32).itemsize
            os.posix_fallocate(tmp_fd, 0, nbytes)
            os.close(tmp_fd)

            n_done = 0
            with mp.get_context("fork").Pool(processes=min(n_workers, len(chunk_bounds))) as pool:
                for _start, _end in pool.imap_unordered(_workspace_transforms_chunk, chunk_bounds):
                    n_done += 1
                    print(f"  Workspace transforms {100 * n_done / len(chunk_bounds):.1f}% complete "
                          f"({n_done}/{len(chunk_bounds)} chunks)")

            big = np.array(np.memmap(tmp_path, dtype=np.float32, mode="r", shape=mmap_shape))
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        jointspace = big[:, :nq]
        body_transforms = {}
        col = nq
        for lnk in bodies:
            body_transforms[lnk.body_name] = big[:, col:col + 16].reshape(total, 4, 4)
            col += 16

        return FingerWorkspaceTransforms(jointspace=jointspace, body_transforms=body_transforms)

    jointspace = np.zeros((max_configs, nq), dtype=np.float32)
    body_transforms = {
        lnk.body_name: np.zeros((max_configs, 4, 4), dtype=np.float32) for lnk in bodies
    }

    count = 0
    report_every = max(1, total // 20)

    for combo in itertools.product(*pt_ranges):
        actuated_q = np.zeros(coupling.n_actuated)
        for k, row in enumerate(finger_rows):
            actuated_q[row] = actuated_joint_space[row, combo[k]]

        full_q = coupling.matrix @ np.append(actuated_q, 1.0)

        q = q_home.copy()
        q[:len(full_q)] = full_q

        compute_fk(model, data, q)
        jointspace[count] = q

        for lnk in bodies:
            T_world_link = get_transform_by_id(model, data, frame_ids[lnk.body_name])
            body_transforms[lnk.body_name][count] = T_world_link @ mesh_offsets[lnk.body_name]

        count += 1
        if count % report_every == 0:
            print(f"  Workspace transforms {100 * count / total:.1f}% complete")

    return FingerWorkspaceTransforms(
        jointspace=jointspace[:count],
        body_transforms={name: arr[:count] for name, arr in body_transforms.items()},
    )


def save_finger_workspace_transforms(
    save_dir: str,
    finger_label: str,
    result: FingerWorkspaceTransforms,
) -> str:
    """
    Persist a FingerWorkspaceTransforms to a single compressed .npz file.

    Replicates MATLAB's save(fullfile(saveFolder,'Finger_N_wkspace_pts.mat'), 'Workspace').

    Returns:
        The path written.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{finger_label}_wkspace_transforms.npz")
    np.savez_compressed(
        path,
        jointspace=result.jointspace,
        body_names=np.array(list(result.body_transforms.keys())),
        **{f"body__{name}": arr for name, arr in result.body_transforms.items()},
    )
    return path


def load_finger_workspace_transforms(path: str) -> FingerWorkspaceTransforms:
    """Load a FingerWorkspaceTransforms previously written by save_finger_workspace_transforms()."""
    with np.load(path) as data:
        body_names = data["body_names"]
        body_transforms = {name: data[f"body__{name}"] for name in body_names}
        return FingerWorkspaceTransforms(
            jointspace=data["jointspace"],
            body_transforms=body_transforms,
        )
