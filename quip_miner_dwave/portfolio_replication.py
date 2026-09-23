"""Offline reproduction of P (qportfolio)'s portfolio scoring pipeline.

Every portfolio solver arm (QPU, dwave-neal, scalar SA, MSA) will pass its raw
reads through the adapter here (:func:`score_reads`) before a Round 2 result is
trusted. It reproduces, without modifying P, the shared finish every provider
in P's own race goes through: ``qpo.encoders.encode_select`` -> a dimod
``BinaryQuadraticModel`` -> ``qpo.solvers.sampling.select_solution`` -> the true
portfolio objective and ``qpo.solvers.feasibility.check_feasibility``.

P (``qpo``, ``dimod``, ``scipy``) lives only in its own pinned environment, so
every P import in this module is LAZY (inside a function body). Importing this
module never requires P; calling most of its functions does.

Fixtures built here are always ``provenance = "synthetic-test"``: Task 1 of
the Round 2 plan found no historical portfolio market capture, so historical
replication stays blocked and no live market data is ever fetched.

A fixture's ``diversification_beta`` is P's cardinality-mode frustration term
(``PortfolioProblem.frustration_beta``, the ``beta`` in ``qpo.problem``'s
module docstring). It is NOT an annealing inverse temperature; nothing here
touches an annealer's schedule. Two ways to arrive at that value are kept
distinct via ``beta_mode``:

    "manuscript-fixed"        an explicit absolute value set directly on the
                               fixture, independent of P's live config ramp.
    "repository-default-ramp" P's shipped ramp, as it stands in its pinned
                               ``qpo.config`` today
                               (``qpo.encoders.resolve_frustration_beta``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

#: Historical provenance is blocked (Task 1); every fixture built here carries this.
SYNTHETIC_PROVENANCE = "synthetic-test"

BETA_MODES = ("manuscript-fixed", "repository-default-ramp")


def make_problem(
    mu: Sequence[float],
    Sigma: Sequence[Sequence[float]],
    gamma: float,
    w_min: float,
    w_max: float,
    cardinality_k: Optional[int],
    frustration_beta: float = 0.0,
    asset_tickers: Optional[Sequence[str]] = None,
):
    """Build a ``qpo.problem.PortfolioProblem`` from plain arrays. Imports qpo lazily."""
    from qpo.problem import PortfolioProblem

    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(Sigma, dtype=np.float64)
    n = mu_arr.shape[0]
    tickers = list(asset_tickers) if asset_tickers is not None else [f"A{i}" for i in range(n)]
    return PortfolioProblem(
        mu=mu_arr,
        Sigma=sigma_arr,
        gamma=float(gamma),
        w_max=float(w_max),
        w_min=float(w_min),
        asset_tickers=tickers,
        cardinality_k=int(cardinality_k) if cardinality_k is not None else None,
        frustration_beta=float(frustration_beta),
    )


def repository_default_beta(problem) -> float:
    """P's shipped ramp: ``qpo.encoders.resolve_frustration_beta(problem)``.

    Zero below ``qpo.config.CARDINALITY_BETA_N_MIN`` assets; the caller decides
    whether to apply the result to ``problem.frustration_beta``.
    """
    from qpo.encoders import resolve_frustration_beta

    return float(resolve_frustration_beta(problem))


def encode_to_ising(problem) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]:
    """``encode_select(problem)``, converted to Ising via dimod, offset retained.

    Returns ``(qubo, h, edges, j, offset)``: ``h`` is float64 shape ``(n,)``,
    ``edges`` is int64 shape ``(m, 2)`` with every pair ordered ``(min, max)``,
    ``j`` is float64 shape ``(m,)`` aligned with ``edges``, and ``offset`` is a
    plain float. This is the Round 2 bundle array contract
    (:mod:`quip_miner_dwave.round2_io`).
    """
    import dimod
    from qpo.encoders import encode_select

    qubo = encode_select(problem)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())
    h_dict, j_dict, offset = bqm.to_ising()

    n = qubo.n
    h = np.array([float(h_dict.get(i, 0.0)) for i in range(n)], dtype=np.float64)

    pairs = sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in j_dict})
    if pairs:
        edges = np.array(pairs, dtype=np.int64)
        j = np.array([_pair_value(j_dict, u, v) for u, v in pairs], dtype=np.float64)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        j = np.zeros(0, dtype=np.float64)

    return qubo, h, edges, j, float(offset)


def _pair_value(j_dict: Dict[Tuple[int, int], float], u: int, v: int) -> float:
    """dimod stores each undirected pair under exactly one of ``(u, v)``/``(v, u)``."""
    if (u, v) in j_dict:
        return float(j_dict[(u, v)])
    return float(j_dict[(v, u)])


def portfolio_fixture_manifest(
    problem,
    beta_mode: str,
    provenance: str = SYNTHETIC_PROVENANCE,
) -> Dict[str, Any]:
    """Manifest fields for a portfolio fixture bundle. See the module docstring for ``beta_mode``."""
    if beta_mode not in BETA_MODES:
        raise ValueError(f"beta_mode must be one of {BETA_MODES}, got {beta_mode!r}")
    return {
        "provenance": provenance,
        "n_assets": int(problem.N),
        "mu": problem.mu.tolist(),
        "sigma": problem.Sigma.tolist(),
        "gamma": float(problem.gamma),
        "cardinality_k": problem.cardinality_k,
        "w_min": float(problem.w_min),
        "w_max": float(problem.w_max),
        "asset_tickers": list(problem.asset_tickers),
        # The select encoding's diversification term, never an annealing schedule beta.
        "diversification_beta": {
            "effective": float(problem.frustration_beta),
            "mode": beta_mode,
        },
        "encoding_scheme": "select",
    }


def write_portfolio_fixture(
    path: PathLike,
    problem,
    beta_mode: str,
    provenance: str = SYNTHETIC_PROVENANCE,
) -> str:
    """Encode ``problem`` to Ising and write it as a Round 2 bundle.

    Uses :func:`quip_miner_dwave.round2_io.write_bundle`, so the on-disk
    contract (``h``/``edges``/``j``/``offset`` arrays, hashed manifest) is
    identical to the Task 2 model bundles; only the manifest's portfolio
    fields (mu, Sigma, K, gamma, bounds, diversification beta) are specific to
    this fixture.
    """
    from quip_miner_dwave.round2_io import write_bundle

    qubo, h, edges, j, offset = encode_to_ising(problem)
    manifest = portfolio_fixture_manifest(problem, beta_mode=beta_mode, provenance=provenance)
    manifest["offset"] = offset
    arrays = {"h": h, "edges": edges, "j": j, "offset": np.asarray(offset, dtype=np.float64)}
    return write_bundle(path, manifest, arrays)


def _bits_from_spins(spins: Any, n_vars: Optional[int] = None) -> np.ndarray:
    """Validate and convert +-1 spin reads to {0,1} bits.

    Accepts a single read, shape ``(n_vars,)``, or many, shape
    ``(n_reads, n_vars)``. Raises on empty, nonfinite, or non-+-1 input: this
    is the "validate nonempty finite input before scoring" gate from the task
    brief, run before anything is handed to P.
    """
    arr = np.asarray(spins, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("spins must be a nonempty (n_reads, n_vars) array")
    if not np.isfinite(arr).all():
        raise ValueError("spins must be finite")
    if not np.isin(arr, (-1.0, 1.0)).all():
        raise ValueError("spins must be +-1 valued")
    if n_vars is not None and arr.shape[1] != n_vars:
        raise ValueError(f"spins carry {arr.shape[1]} variables, expected {n_vars}")
    return ((arr.astype(np.int16) + 1) // 2).astype(np.int8)


def score_reads(problem, qubo, spins: Any) -> Dict[str, Any]:
    """Reproduce P's scoring adapter for one batch of raw reads.

    Matches the task brief's ``score_reads`` sketch against P's real API:
    ``select_solution`` returns ``(weights, bits)`` where ``bits`` is the RAW
    read that produced the best final portfolio, not a post-repair support
    indicator (P repairs and weights every read internally; only the winning
    read's original bits come back). ``check_feasibility`` takes
    ``(weights, w_max, w_min, ..., cardinality_k=...)`` positionally, matching
    the sketch.

    P's own feasibility gate does not reject NaN (``nan > eps`` is False in
    NumPy, so a NaN budget residual silently reads as "ok"). This adapter
    overrides the gate to infeasible whenever the resulting weights or
    objective are nonfinite, per the task brief's explicit requirement, even
    though P's gate alone would have called it feasible.
    """
    import dimod
    from qpo.solvers.feasibility import check_feasibility
    from qpo.solvers.sampling import select_solution

    bits = _bits_from_spins(spins, n_vars=qubo.n)

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())
    response = dimod.SampleSet.from_samples_bqm((bits, list(range(qubo.n))), bqm)
    weights, selected_bits = select_solution(response, qubo, problem)
    objective = problem.objective(weights)
    gate = check_feasibility(
        weights,
        problem.w_max,
        problem.w_min,
        cardinality_k=problem.cardinality_k,
    )
    finite = bool(np.isfinite(weights).all() and np.isfinite(objective))
    raw_feasible_count = (
        int(np.count_nonzero(bits.sum(axis=1) == problem.cardinality_k))
        if problem.cardinality_k is not None
        else None
    )
    return {
        "weights": weights,
        "selected_bits": selected_bits,
        "objective": objective,
        "feasible": bool(gate.feasible) and finite,
        "raw_feasible_count": raw_feasible_count,
        "returned_reads": int(bits.shape[0]),
    }


def per_read_diagnostics(problem, spins: Any) -> Dict[str, Any]:
    """Per-read repair/weighting diagnostics, cardinality (select-encoding) problems only.

    Calls the same reference ``qpo.projection.greedy_project`` and
    ``qpo.projection.weights_for_support`` every provider's finish uses.  For
    every read this records the pre-repair support (K before repair), the
    post-repair support (always exactly K after repair), what the repair
    added/removed, post-repair feasibility (with the same nonfinite override
    as :func:`score_reads`), and a weighting-failure heuristic.

    P's ``qpo.weighting.optimal_weights`` falls back to the exact equal-weight
    vector on the support when SLSQP's result violates the budget or box, but
    exposes no flag for that fallback having fired. An exact match to the
    equal-weight vector, for a support of two or more assets, is recorded as a
    likely weighting failure; this is a heuristic proxy, not an instrumented
    signal from P (a genuinely symmetric instance could also land exactly on
    equal weight). Kept off for a single-asset support, where the budget
    constraint alone forces weight 1.0 regardless of anything going wrong.
    """
    from qpo.projection import greedy_project, weights_for_support
    from qpo.solvers.feasibility import check_feasibility

    if problem.cardinality_k is None:
        raise ValueError("per_read_diagnostics requires a cardinality problem")

    bits = _bits_from_spins(spins)
    n = problem.N
    rho = problem.rho if problem.frustration_beta else None

    records: List[Dict[str, Any]] = []
    multiplicities: Dict[Tuple[int, ...], int] = {}
    for row in bits:
        raw_support = frozenset(int(i) for i in np.nonzero(row[:n])[0])
        support = greedy_project(row, problem, rho)
        multiplicities[support] = multiplicities.get(support, 0) + 1

        weights = weights_for_support(support, problem)
        objective = problem.objective(weights)
        gate = check_feasibility(
            weights, problem.w_max, problem.w_min, cardinality_k=problem.cardinality_k
        )
        finite = bool(np.isfinite(weights).all() and np.isfinite(objective))

        k = len(support)
        equal_weight = 1.0 / k if k else 0.0
        weighting_failed = k > 1 and bool(
            np.allclose(weights[list(support)], equal_weight, atol=1e-12, rtol=0.0)
        )

        support_set = frozenset(support)
        records.append(
            {
                "raw_support": sorted(raw_support),
                "raw_k": len(raw_support),
                "repaired_support": list(support),
                "added": sorted(support_set - raw_support),
                "removed": sorted(raw_support - support_set),
                "post_repair_feasible": bool(gate.feasible) and finite,
                "post_repair_reason": gate.reason if finite else "nonfinite weights or objective",
                "weighting_failed": weighting_failed,
                "objective": objective,
            }
        )

    return {"records": records, "support_multiplicities": dict(multiplicities)}
