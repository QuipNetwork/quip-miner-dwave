#!/usr/bin/env python3
"""Write the Task 3 portfolio-scoring fixture bundles.

Task 3 of docs/superpowers/plans/2026-09-22-regime-search-round2.md: builds
the 4/18/28-asset synthetic-test fixtures the task brief calls for (exhaustive
tests use the 4-asset problem directly; this script is what actually freezes
all three onto disk) and writes each one, twice, under two distinct
diversification-beta modes so a manuscript-fixed fixture is never confused
with one built from P's live repository-default ramp
(``qpo.encoders.resolve_frustration_beta``).

Requires P's pinned environment on the path (``qpo``, ``dimod``), so it must
run under P's interpreter, e.g.:

    PYTHONPATH=<this repo> <P>/.venv/bin/python scripts/portfolio_replication.py

Historical replication stays blocked (Task 1 found no historical portfolio
market capture), so every fixture here is built from a deterministic,
seeded synthetic instance -- never live market data -- and carries
``provenance = "synthetic-test"``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quip_miner_dwave import portfolio_replication as pr  # noqa: E402

DEFAULT_ROOT = Path("/home/carback1/quip-data/regimes/round2/bundles-portfolio-fixtures")

#: (n_assets, cardinality_k, manuscript-fixed beta as a fraction of the objective scale).
FIXTURES: Tuple[Tuple[int, int, float], ...] = (
    (4, 2, 0.5),
    (18, 6, 0.5),
    (28, 9, 0.5),
)


def _synthetic_problem(n: int, k: int, seed: int):
    """Deterministic one-factor PSD instance, the same shape used by the test suite."""
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.5, 1.5, size=n)
    idiosyncratic = rng.uniform(0.0002, 0.0020, size=n)
    Sigma = 0.3 * 0.0004 * np.outer(loadings, loadings) + np.diag(idiosyncratic)
    mu = rng.normal(0.0, 0.0006, size=n)
    return pr.make_problem(
        mu=mu,
        Sigma=Sigma,
        gamma=5.0,
        w_min=0.25 / k,
        w_max=0.5,
        cardinality_k=k,
        asset_tickers=[f"A{i}" for i in range(n)],
    )


def write_fixtures(root: Path) -> None:
    from qpo.encoders import resolve_frustration_beta, select_objective_scale

    for n, k, manuscript_fraction in FIXTURES:
        base = _synthetic_problem(n, k, seed=n)

        manuscript_problem = _synthetic_problem(n, k, seed=n)
        manuscript_problem.frustration_beta = manuscript_fraction * select_objective_scale(base)
        manuscript_problem.invalidate_cache()
        manuscript_path = root / f"n{n}-k{k}" / "manuscript-fixed"
        manuscript_hash = pr.write_portfolio_fixture(
            manuscript_path, manuscript_problem, beta_mode="manuscript-fixed"
        )
        print(f"n={n} k={k} manuscript-fixed beta={manuscript_problem.frustration_beta:.6g} -> "
              f"{manuscript_path} ({manuscript_hash})")

        default_problem = _synthetic_problem(n, k, seed=n)
        default_problem.frustration_beta = resolve_frustration_beta(default_problem)
        default_problem.invalidate_cache()
        default_path = root / f"n{n}-k{k}" / "repository-default-ramp"
        default_hash = pr.write_portfolio_fixture(
            default_path, default_problem, beta_mode="repository-default-ramp"
        )
        print(f"n={n} k={k} repository-default-ramp beta={default_problem.frustration_beta:.6g} "
              f"-> {default_path} ({default_hash})")


def main() -> int:
    root = DEFAULT_ROOT
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    write_fixtures(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
