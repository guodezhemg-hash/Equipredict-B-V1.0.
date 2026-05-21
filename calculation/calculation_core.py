"""
Calculation core preview for a reaction-extent equilibrium calculation.

This file intentionally shows only the central numerical expressions:

    C = C0 + V @ xi
    logQ = V.T @ log(C)
    J = V.T @ (V / C[:, None])

It does not include data loading, batch processing, model construction,
real-data solver tuning, file export, or any project-specific chemistry data.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import least_squares


DEFAULT_EPS = 1e-15

__all__ = [
    "concentration_from_extent",
    "log_reaction_quotient",
    "reaction_extent_jacobian",
    "log_equilibrium_residual",
    "make_least_squares_callbacks",
    "solve_toy_problem_with_least_squares",
]


def concentration_from_extent(
    initial_concentration: np.ndarray,
    v_matrix: np.ndarray,
    reaction_extent: np.ndarray,
) -> np.ndarray:
    """
    Compute the concentration vector from reaction extents.

    Parameters
    ----------
    initial_concentration:
        Initial concentration vector C0 with shape (n_species,).
    v_matrix:
        Stoichiometric/coefficient matrix V with shape
        (n_species, n_reactions).
    reaction_extent:
        Reaction extent vector xi with shape (n_reactions,).

    Returns
    -------
    np.ndarray
        Concentration vector C.
    """
    c0 = np.asarray(initial_concentration, dtype=float)
    v = np.asarray(v_matrix, dtype=float)
    xi = np.asarray(reaction_extent, dtype=float)
    return c0 + v @ xi


def log_reaction_quotient(
    concentration: np.ndarray,
    v_matrix: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """
    Compute log reaction quotients for all reactions.

    The small `eps` guard is used only to avoid log(0) in this numerical
    preview.
    """
    c = np.asarray(concentration, dtype=float)
    v = np.asarray(v_matrix, dtype=float)
    safe_c = np.maximum(c, eps)
    return v.T @ np.log(safe_c)


def reaction_extent_jacobian(
    concentration: np.ndarray,
    v_matrix: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """
    Compute the analytic Jacobian of logQ with respect to reaction extent.

    This corresponds to:

        J = V.T @ (V / C[:, None])
    """
    c = np.asarray(concentration, dtype=float)
    v = np.asarray(v_matrix, dtype=float)
    safe_c = np.maximum(c, eps)
    return v.T @ (v / safe_c[:, None])


def log_equilibrium_residual(
    reaction_extent: np.ndarray,
    initial_concentration: np.ndarray,
    v_matrix: np.ndarray,
    log_equilibrium_constant: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """
    Compute the residual vector logQ(xi) - logK.

    This is a compact preview of the residual used by a nonlinear least-squares
    solver.
    """
    c = concentration_from_extent(
        initial_concentration,
        v_matrix,
        reaction_extent,
    )
    if np.any(c <= eps):
        return np.full_like(log_equilibrium_constant, 1e6, dtype=float)

    log_q = log_reaction_quotient(c, v_matrix, eps=eps)
    return log_q - np.asarray(log_equilibrium_constant, dtype=float)


def make_least_squares_callbacks(
    initial_concentration: np.ndarray,
    v_matrix: np.ndarray,
    log_equilibrium_constant: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Build residual and Jacobian callbacks for a nonlinear least-squares solver.

    This mirrors the essential structure used in the full calculation:

        residual(xi) = logQ(C0 + V @ xi) - logK
        jacobian(xi) = V.T @ (V / C[:, None])

    The full project data interface and tuned solver policy are intentionally
    not included in this preview.
    """
    c0 = np.asarray(initial_concentration, dtype=float)
    v = np.asarray(v_matrix, dtype=float)
    log_k = np.asarray(log_equilibrium_constant, dtype=float)
    n_reactions = v.shape[1]

    def residual(reaction_extent: np.ndarray) -> np.ndarray:
        return log_equilibrium_residual(
            reaction_extent,
            c0,
            v,
            log_k,
            eps=eps,
        )

    def jacobian(reaction_extent: np.ndarray) -> np.ndarray:
        c = concentration_from_extent(c0, v, reaction_extent)
        if np.any(c <= eps):
            return np.eye(n_reactions) * 1e6
        return reaction_extent_jacobian(c, v, eps=eps)

    return residual, jacobian


def solve_toy_problem_with_least_squares() -> dict[str, np.ndarray | float | int | bool]:
    """
    Run a small artificial least-squares example.

    This is intentionally limited to one generic toy reaction. It demonstrates
    how the residual and Jacobian callbacks can be handed to SciPy without
    exposing project-specific data, preprocessing, parameter tables, or export
    logic.
    """
    c0 = np.array([1.0, 1.0, 0.01])
    v = np.array([[-1.0], [-1.0], [1.0]])
    log_k = np.log(np.array([10.0]))

    residual_fn, jacobian_fn = make_least_squares_callbacks(c0, v, log_k)

    result = least_squares(
        residual_fn,
        x0=np.zeros(v.shape[1]),
        jac=jacobian_fn,
        bounds=(-1.0, 1.0),
        method="trf",
        max_nfev=100,
    )

    equilibrium_concentration = concentration_from_extent(c0, v, result.x)

    return {
        "success": result.success,
        "nfev": result.nfev,
        "cost": result.cost,
        "xi": result.x,
        "equilibrium_concentration": equilibrium_concentration,
        "residual": residual_fn(result.x),
    }


def _demo() -> None:
    """Run a tiny artificial example with no project-specific data."""
    c0 = np.array([1.0, 1.0, 0.01])
    v = np.array([[-1.0], [-1.0], [1.0]])
    xi = np.array([0.5])
    log_k = np.log(np.array([10.0]))

    c = concentration_from_extent(c0, v, xi)
    log_q = log_reaction_quotient(c, v)
    jacobian = reaction_extent_jacobian(c, v)
    residual = log_equilibrium_residual(xi, c0, v, log_k)
    residual_fn, jacobian_fn = make_least_squares_callbacks(c0, v, log_k)
    solver_result = solve_toy_problem_with_least_squares()

    print("C =", c)
    print("logQ =", log_q)
    print("J =", jacobian)
    print("residual =", residual)
    print("least_squares residual callback =", residual_fn(xi))
    print("least_squares jacobian callback =", jacobian_fn(xi))
    print("solver success =", solver_result["success"])
    print("solver xi =", solver_result["xi"])
    print("solver equilibrium C =", solver_result["equilibrium_concentration"])
    print("solver residual =", solver_result["residual"])


if __name__ == "__main__":
    _demo()
