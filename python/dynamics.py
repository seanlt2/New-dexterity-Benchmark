"""
Joint-space dynamics matrices (mass matrix, Coriolis matrix) from a URDF,
via Pinocchio's Composite Rigid Body Algorithm and Coriolis Matrix
algorithm.

Not used anywhere in this project's pipeline yet -- built ahead of a
possible future move from this project's current force-ellipsoid
(quasi-static, velocity/force-mapping) manipulability analysis to a dynamic
force analysis, which would need M(q) and C(q, qdot) for the full rigid-body
equations of motion:

    M(q) qddot + C(q, qdot) qdot + g(q) = tau

Everything here works in the FULL Pinocchio joint space (size model.nq /
model.nv), not this project's reduced actuated joint space. That's
deliberately different from kinematics.true_jacobian(), which projects a
Jacobian onto actuated DOFs via the coupling matrix: this project's coupling
is a CONSTANT linear map q_full = C @ [q_actuated; 1] (see
coupling.CouplingInfo), so the analogous reduction for the mass/Coriolis
matrices -- M_reduced = C^T M C, C_reduced(q, qdot_act) = C^T C(q, C @
qdot_act) C, g_reduced = C^T g -- is a real, valid derivation given that
constant-coupling structure, but it's a separate step this module doesn't
do, since it's not needed yet.

Also worth knowing before relying on these numbers for anything: they're
only physically meaningful if every link in the URDF has a real <inertial>
tag (mass, center of mass, inertia tensor). Not every hand config's URDF in
this project is guaranteed to -- some converted/downloaded URDFs omit or
zero it out -- so check a given hand's URDF (or model.inertias after
loading it) before trusting M/C for that hand specifically.
"""

from __future__ import annotations

import numpy as np

try:
    import pinocchio as pin
    _PIN_AVAILABLE = True
except ImportError:
    _PIN_AVAILABLE = False

from .kinematics import load_model


def compute_mass_matrix(model, data, q: np.ndarray) -> np.ndarray:
    """
    Joint-space mass (inertia) matrix M(q), full Pinocchio joint space.

    Uses the Composite Rigid Body Algorithm (pin.crba()). Explicitly
    symmetrized (M = triu(M) + triu(M, 1).T) rather than trusting the raw
    result to already be fully populated below the diagonal -- Pinocchio's
    documented crba() behavior only guarantees the upper triangle, even
    though the specific binding this project currently uses (Pinocchio
    4.0.0) already returns a fully symmetric matrix in practice (verified
    directly against a real hand model before writing this).

    Args:
        model, data: Pinocchio model/data (e.g. from kinematics.load_model()).
        q:           Configuration vector (length model.nq).

    Returns:
        (model.nv, model.nv) symmetric mass matrix.
    """
    if not _PIN_AVAILABLE:
        raise ImportError("pinocchio is required for dynamics functions.")
    M = np.array(pin.crba(model, data, q))
    return np.triu(M) + np.triu(M, 1).T


def compute_coriolis_matrix(model, data, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """
    Joint-space Coriolis/centrifugal matrix C(q, qdot), full Pinocchio joint
    space, such that C(q, qdot) @ qdot gives the Coriolis/centrifugal
    generalized-force term in the equations of motion. NOT symmetric in
    general (unlike the mass matrix) -- that's expected, not a bug.

    Args:
        model, data: Pinocchio model/data.
        q:           Configuration vector (length model.nq).
        qdot:        Joint velocity vector (length model.nv).

    Returns:
        (model.nv, model.nv) Coriolis matrix.
    """
    if not _PIN_AVAILABLE:
        raise ImportError("pinocchio is required for dynamics functions.")
    return np.array(pin.computeCoriolisMatrix(model, data, q, qdot))


def mass_and_coriolis_from_urdf(
    urdf_path: str,
    q: np.ndarray | None = None,
    qdot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Self-contained convenience entry point: parse a URDF and compute its
    mass matrix M(q) and Coriolis matrix C(q, qdot) in one call, without
    needing an already-loaded pinocchio model/data around.

    See compute_mass_matrix() / compute_coriolis_matrix() for the lower-
    level versions that take an already-loaded model/data instead -- for a
    caller (e.g. a future dynamic-force-analysis script) that's already
    parsed the URDF once via kinematics.load_model() and doesn't want to
    reparse it on every call.

    Args:
        urdf_path: Path to the URDF file.
        q:         Configuration vector (length model.nq). Defaults to
                  pin.neutral(model) (the model's zero/home configuration).
        qdot:      Joint velocity vector (length model.nv). Defaults to all
                  zeros -- note this makes C(q, 0) the zero matrix (Coriolis
                  terms vanish at zero velocity by definition), so pass an
                  actual qdot if the Coriolis matrix itself, not just M(q),
                  is what's wanted.

    Returns:
        (M, C): mass matrix and Coriolis matrix, each (model.nv, model.nv).
    """
    if not _PIN_AVAILABLE:
        raise ImportError("pinocchio is required for dynamics functions.")
    model, data = load_model(urdf_path)
    if q is None:
        q = pin.neutral(model)
    if qdot is None:
        qdot = np.zeros(model.nv)
    M = compute_mass_matrix(model, data, q)
    C = compute_coriolis_matrix(model, data, q, qdot)
    return M, C
