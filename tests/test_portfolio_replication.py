"""Tests for the offline reproduction of P (qportfolio)'s scoring pipeline.

Task 3 of docs/superpowers/plans/2026-09-22-regime-search-round2.md: this is
the shared scoring adapter every portfolio solver arm (QPU, dwave-neal,
scalar SA, MSA) will pass its raw reads through. Every test here calls the
pinned P implementation directly (``qpo``, ``dimod``), so the whole module
skips cleanly when P is not on the path -- P lives only in its own pinned
environment (see the task-3 report for the exact invocation).

Tolerance: coefficients and objectives are compared at 1e-12 absolute /
1e-10 relative, the software tolerance the task brief declares for this
comparison.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quip_miner_dwave import portfolio_replication as pr
from quip_miner_dwave import round2_io

qpo = pytest.importorskip("qpo")
dimod = pytest.importorskip("dimod")

encode_select = qpo.encoders.encode_select
resolve_frustration_beta = qpo.encoders.resolve_frustration_beta
select_objective_scale = qpo.encoders.select_objective_scale
PortfolioProblem = qpo.problem.PortfolioProblem

ABS_TOL = 1e-12
REL_TOL = 1e-10


def assert_close(actual, expected, *, abs_tol=ABS_TOL, rel_tol=REL_TOL):
    np.testing.assert_allclose(actual, expected, atol=abs_tol, rtol=rel_tol)


# --------------------------------------------------------------------- fixtures


def four_asset_problem(beta: float = 0.0) -> PortfolioProblem:
    """4-asset cardinality problem, small enough to enumerate exhaustively."""
    mu = np.array([0.02, 0.015, 0.01, 0.005])
    Sigma = np.array(
        [
            [0.04, 0.02, 0.01, 0.005],
            [0.02, 0.03, 0.015, 0.005],
            [0.01, 0.015, 0.02, 0.002],
            [0.005, 0.005, 0.002, 0.01],
        ]
    )
    return pr.make_problem(
        mu=mu,
        Sigma=Sigma,
        gamma=3.0,
        w_min=0.1,
        w_max=0.7,
        cardinality_k=2,
        frustration_beta=beta,
        asset_tickers=["A", "B", "C", "D"],
    )


def _random_psd_problem(n: int, k: int, seed: int) -> PortfolioProblem:
    """A larger deterministic, PSD-by-construction fixture for integration tests."""
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


# -------------------------------------------------------- step 2: exhaustive 16-state check


class TestExhaustiveSmallStates:
    """4-asset problem: 2**4 = 16 states, checked against direct binary energy."""

    @pytest.mark.parametrize("beta_fraction", [0.0, 0.5])
    def test_qubo_dict_matches_direct_binary_energy(self, beta_fraction):
        problem = four_asset_problem()
        if beta_fraction:
            problem.frustration_beta = beta_fraction * select_objective_scale(problem)
            problem.invalidate_cache()
        qubo = encode_select(problem)
        qdict = qubo.to_dict()

        for bits_tuple in itertools.product((0, 1), repeat=4):
            x = np.array(bits_tuple, dtype=float)
            direct = float(x @ qubo.Q @ x)
            from_dict = sum(
                coeff * (x[i] if i == j else x[i] * x[j]) for (i, j), coeff in qdict.items()
            )
            assert_close(from_dict, direct)

    @pytest.mark.parametrize("beta_fraction", [0.0, 0.5])
    def test_ising_conversion_retains_offset_and_energy(self, beta_fraction):
        problem = four_asset_problem()
        if beta_fraction:
            problem.frustration_beta = beta_fraction * select_objective_scale(problem)
            problem.invalidate_cache()
        qubo, h, edges, j, offset = pr.encode_to_ising(problem)

        # Independently recompute the Ising conversion via dimod to check our
        # (h, edges, j, offset) extraction round-trips, then compare Ising
        # energy against the direct QUBO energy for every one of the 16 states.
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())
        for bits_tuple in itertools.product((0, 1), repeat=4):
            bits = np.array(bits_tuple, dtype=float)
            qubo_energy = float(bits @ qubo.Q @ bits)

            spins = 2.0 * bits - 1.0
            ising_energy = float(h @ spins) + offset
            for idx, (u, v) in enumerate(edges):
                ising_energy += j[idx] * spins[u] * spins[v]
            assert_close(ising_energy, qubo_energy)

            dimod_energy = bqm.energy({i: int(b) for i, b in enumerate(bits_tuple)})
            assert_close(qubo_energy, dimod_energy)


# ------------------------------------------------------------- score_reads adapter


class TestScoreReads:
    def test_rejects_empty_response(self):
        problem = four_asset_problem()
        qubo = encode_select(problem)
        with pytest.raises(ValueError):
            pr.score_reads(problem, qubo, np.zeros((0, qubo.n)))

    def test_rejects_nonfinite_input(self):
        problem = four_asset_problem()
        qubo = encode_select(problem)
        spins = np.array([[1.0, -1.0, np.nan, 1.0]])
        with pytest.raises(ValueError):
            pr.score_reads(problem, qubo, spins)

    def test_rejects_non_pm1_input(self):
        problem = four_asset_problem()
        qubo = encode_select(problem)
        spins = np.array([[1.0, 0.0, -1.0, 1.0]])
        with pytest.raises(ValueError):
            pr.score_reads(problem, qubo, spins)

    def test_accepts_a_single_1d_read(self):
        problem = four_asset_problem()
        qubo = encode_select(problem)
        spins = np.array([-1.0, -1.0, -1.0, -1.0])
        result = pr.score_reads(problem, qubo, spins)
        assert result["returned_reads"] == 1

    def test_minimum_raw_energy_read_can_differ_from_best_final_portfolio(self):
        """[1,1,0,0] has lower raw select-QUBO energy than [0,1,1,1], but its
        post-repair portfolio is worse: this is P's documented select-encoding
        caveat (energy is the equal-weight surrogate, not the true objective)."""
        problem = four_asset_problem(beta=0.0)
        problem.frustration_beta = 0.5 * select_objective_scale(problem)
        problem.invalidate_cache()
        qubo = encode_select(problem)
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())

        low_energy_worse = (1, 1, 0, 0)
        high_energy_better = (0, 1, 1, 1)
        e_low = bqm.energy({i: b for i, b in enumerate(low_energy_worse)})
        e_high = bqm.energy({i: b for i, b in enumerate(high_energy_better)})
        assert e_low < e_high  # confirms the raw-energy ordering used below

        spins = np.array(
            [
                [2 * b - 1 for b in low_energy_worse],
                [2 * b - 1 for b in high_energy_better],
            ],
            dtype=float,
        )
        result = pr.score_reads(problem, qubo, spins)
        selected_bits = tuple(int((s + 1) // 2) for s in result["selected_bits"])
        assert selected_bits == high_energy_better
        assert selected_bits != low_energy_worse

    def test_raw_feasible_count_before_repair(self):
        """raw_feasible_count checks K on the RAW bits, before any repair."""
        problem = four_asset_problem()
        qubo = encode_select(problem)
        spins = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],  # raw K=0
                [1.0, 1.0, -1.0, -1.0],  # raw K=2 == cardinality_k
                [1.0, 1.0, 1.0, 1.0],  # raw K=4
            ]
        )
        result = pr.score_reads(problem, qubo, spins)
        assert result["raw_feasible_count"] == 1

    def test_final_score_is_identical_across_hypothetical_adapters(self):
        """Same raw reads through 'QPU', 'neal', 'scalar', 'MSA' must score identically:
        score_reads is a pure function of (problem, qubo, spins)."""
        problem = four_asset_problem()
        problem.frustration_beta = 0.3 * select_objective_scale(problem)
        problem.invalidate_cache()
        qubo = encode_select(problem)
        rng = np.random.default_rng(0)
        spins = rng.choice([-1.0, 1.0], size=(12, qubo.n))

        adapters = ["qpu", "neal", "scalar", "msa"]
        results = [pr.score_reads(problem, qubo, spins) for _ in adapters]
        base = results[0]
        for other in results[1:]:
            assert_close(other["weights"], base["weights"])
            assert_close(other["objective"], base["objective"])
            assert other["feasible"] == base["feasible"]

    def test_post_repair_feasibility_holds_even_when_gate_alone_would_pass_nan(self):
        """P's own check_feasibility does not reject NaN (nan > eps is False in
        NumPy). A NaN objective must still flip score_reads' feasible flag to
        False even though P's gate reports ok. The QUBO is built from a clean
        problem first (encode_select itself rejects a NaN coefficient via its
        symmetry check), then mu is corrupted in place afterwards so only the
        objective step, not the encoding, sees the NaN."""
        problem = four_asset_problem()
        qubo = encode_select(problem)
        problem.mu[1] = np.nan
        spins = np.array([-1.0, -1.0, 1.0, 1.0])
        result = pr.score_reads(problem, qubo, spins)
        assert not np.isfinite(result["objective"])
        assert result["feasible"] is False


# --------------------------------------------------------- per-read diagnostics


class TestPerReadDiagnostics:
    def test_requires_a_cardinality_problem(self):
        problem = pr.make_problem(
            mu=[0.01, 0.02],
            Sigma=[[1e-4, 0.0], [0.0, 1e-4]],
            gamma=1.0,
            w_min=0.0,
            w_max=1.0,
            cardinality_k=None,
        )
        with pytest.raises(ValueError):
            pr.per_read_diagnostics(problem, np.array([-1.0, 1.0]))

    def test_k_before_and_after_repair(self):
        problem = four_asset_problem()
        spins = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],  # raw K=0, must be repaired up to K=2
                [1.0, 1.0, 1.0, -1.0],  # raw K=3, must be repaired down to K=2
                [1.0, 1.0, -1.0, -1.0],  # raw K=2 already
            ]
        )
        diag = pr.per_read_diagnostics(problem, spins)
        records = diag["records"]
        assert [r["raw_k"] for r in records] == [0, 3, 2]
        assert all(len(r["repaired_support"]) == problem.cardinality_k for r in records)
        assert len(records[0]["added"]) == 2 and not records[0]["removed"]
        assert len(records[1]["removed"]) == 1 and not records[1]["added"]
        assert not records[2]["added"] and not records[2]["removed"]

    def test_support_multiplicities_count_repaired_supports(self):
        problem = four_asset_problem()
        spins = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0],
            ]
        )
        diag = pr.per_read_diagnostics(problem, spins)
        total = sum(diag["support_multiplicities"].values())
        assert total == len(spins)
        for support in diag["support_multiplicities"]:
            assert len(support) == problem.cardinality_k

    def test_post_repair_feasibility_is_recorded(self):
        problem = four_asset_problem()
        spins = np.array([1.0, 1.0, -1.0, -1.0])
        diag = pr.per_read_diagnostics(problem, spins)
        assert diag["records"][0]["post_repair_feasible"] is True

    def test_weighting_failure_is_flagged(self, monkeypatch):
        """Force P's documented equal-weight fallback path and confirm the
        diagnostic's heuristic (an exact match to the fallback vector) catches
        it. P exposes no flag for the fallback, so this monkeypatches
        qpo.projection.weights_for_support -- the exact function
        per_read_diagnostics imports lazily -- to return the fallback vector
        it would have produced, rather than trying to force scipy to fail."""
        problem = four_asset_problem()
        support = (2, 3)
        equal_weight_fallback = np.zeros(problem.N)
        equal_weight_fallback[list(support)] = 1.0 / len(support)

        def fake_weights_for_support(sup, prob):
            assert sup == support
            return equal_weight_fallback

        monkeypatch.setattr(qpo.projection, "weights_for_support", fake_weights_for_support)

        spins = np.array([-1.0, -1.0, 1.0, 1.0])  # already repairs to (2, 3)
        diag = pr.per_read_diagnostics(problem, spins)
        assert diag["records"][0]["weighting_failed"] is True

    def test_single_asset_support_is_never_flagged_as_a_weighting_failure(self):
        """K=1: the budget alone forces weight 1.0, which trivially equals the
        'equal weight' heuristic value; that must not read as a failure."""
        problem = pr.make_problem(
            mu=[0.01, 0.02, 0.03],
            Sigma=np.eye(3) * 1e-4,
            gamma=1.0,
            w_min=0.0,
            w_max=1.0,
            cardinality_k=1,
        )
        spins = np.array([-1.0, -1.0, 1.0])
        diag = pr.per_read_diagnostics(problem, spins)
        assert diag["records"][0]["weighting_failed"] is False


# --------------------------------------------------------------------- beta handling


class TestBeta:
    def test_zero_beta_matches_encoding_and_objective(self):
        problem = four_asset_problem(beta=0.0)
        assert problem.frustration_beta == 0.0
        qubo = encode_select(problem)
        # No frustration coupling: off-diagonal is exactly (gamma/K^2) Sigma_ij.
        k, gamma = problem.cardinality_k, problem.gamma
        coef = gamma / (2.0 * k * k)
        expected_offdiag = coef * problem.Sigma[0, 1]
        assert_close(qubo.Q[0, 1], expected_offdiag)

        weights = np.array([0.6, 0.0, 0.4, 0.0])
        objective = problem.objective(weights)
        expected = 0.5 * gamma * weights @ problem.Sigma @ weights - problem.mu @ weights
        assert_close(objective, expected)

    def test_positive_beta_is_used_by_both_encoding_and_objective(self):
        problem = four_asset_problem()
        beta = 0.4 * select_objective_scale(problem)
        problem.frustration_beta = beta
        problem.invalidate_cache()
        qubo = encode_select(problem)

        # Encoding: Q[i, j] and Q[j, i] each hold coef*Sigma_ij already (full
        # symmetric matrix, not a triangle), plus half of beta*rho_ij; to_dict's
        # pair coefficient is their sum, i.e. 2*coef*Sigma_ij + beta*rho_ij.
        k, gamma = problem.cardinality_k, problem.gamma
        coef = gamma / (2.0 * k * k)
        expected_offdiag = 2.0 * coef * problem.Sigma[0, 1] + beta * problem.rho[0, 1]
        assert_close(qubo.Q[0, 1] + qubo.Q[1, 0], expected_offdiag)

        # Objective: the same recorded beta drives the held-pair penalty.
        weights = np.array([0.6, 0.0, 0.4, 0.0])
        held = (weights > 1e-9).astype(float)
        pair_corr = 0.5 * (held @ problem.rho @ held - held.sum())
        expected = (
            0.5 * gamma * weights @ problem.Sigma @ weights
            - problem.mu @ weights
            + beta * pair_corr
        )
        assert_close(problem.objective(weights), expected)

    def test_manuscript_fixed_and_repository_default_ramp_fixtures_stay_separate(self, tmp_path):
        # N=18 clears qpo.config.CARDINALITY_BETA_N_MIN (15), so the repository
        # ramp is nonzero and distinguishable from an arbitrary manuscript value.
        n, k = 18, 6
        manuscript_problem = _random_psd_problem(n, k, seed=1)
        manuscript_problem.frustration_beta = 0.25
        manuscript_problem.invalidate_cache()

        default_problem = _random_psd_problem(n, k, seed=1)
        default_problem.frustration_beta = resolve_frustration_beta(default_problem)
        default_problem.invalidate_cache()

        assert default_problem.frustration_beta > 0.0
        assert manuscript_problem.frustration_beta != default_problem.frustration_beta

        manuscript_path = tmp_path / "manuscript-fixed"
        default_path = tmp_path / "repository-default-ramp"
        pr.write_portfolio_fixture(manuscript_path, manuscript_problem, beta_mode="manuscript-fixed")
        pr.write_portfolio_fixture(
            default_path, default_problem, beta_mode="repository-default-ramp"
        )

        manuscript_manifest, _ = round2_io.read_bundle(manuscript_path)
        default_manifest, _ = round2_io.read_bundle(default_path)
        assert manuscript_manifest["diversification_beta"]["mode"] == "manuscript-fixed"
        assert default_manifest["diversification_beta"]["mode"] == "repository-default-ramp"
        assert manuscript_manifest["diversification_beta"]["effective"] == pytest.approx(0.25)
        assert default_manifest["diversification_beta"]["effective"] == pytest.approx(
            default_problem.frustration_beta
        )
        assert (
            manuscript_manifest["diversification_beta"]["effective"]
            != default_manifest["diversification_beta"]["effective"]
        )


# ------------------------------------------------------------- fixture round-trip


class TestFixtureBundles:
    @pytest.mark.parametrize("n,k", [(4, 2), (18, 6), (28, 9)])
    def test_fixture_round_trips_through_write_bundle_and_matches_direct_energy(
        self, tmp_path, n, k
    ):
        if n == 4:
            problem = four_asset_problem()
            problem.frustration_beta = 0.3 * select_objective_scale(problem)
            problem.invalidate_cache()
        else:
            problem = _random_psd_problem(n, k, seed=n)
            problem.frustration_beta = resolve_frustration_beta(problem)
            problem.invalidate_cache()

        path = tmp_path / f"n{n}"
        pr.write_portfolio_fixture(path, problem, beta_mode="repository-default-ramp")
        manifest, arrays = round2_io.read_bundle(path)

        assert manifest["provenance"] == "synthetic-test"
        assert manifest["n_assets"] == n
        assert manifest["cardinality_k"] == k

        # Rebuild the BQM from the bundle's h/edges/j/offset and check its
        # energy against encode_select's own Q for a batch of random reads.
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {i: float(v) for i, v in enumerate(arrays["h"])},
            {
                (int(u), int(v)): float(w)
                for (u, v), w in zip(arrays["edges"], arrays["j"])
            },
            float(arrays["offset"]),
        )
        qubo = encode_select(problem)
        rng = np.random.default_rng(n)
        for _ in range(5):
            bits = rng.integers(0, 2, size=n)
            spins = {i: (2 * int(b) - 1) for i, b in enumerate(bits)}
            ising_energy = bqm.energy(spins)
            qubo_energy = float(bits.astype(float) @ qubo.Q @ bits.astype(float))
            assert_close(ising_energy, qubo_energy)

    @pytest.mark.parametrize("n,k", [(4, 2), (18, 6), (28, 9)])
    def test_score_reads_on_the_written_fixture(self, tmp_path, n, k):
        if n == 4:
            problem = four_asset_problem()
        else:
            problem = _random_psd_problem(n, k, seed=100 + n)
        problem.frustration_beta = resolve_frustration_beta(problem) if n >= 15 else 0.0
        problem.invalidate_cache()

        path = tmp_path / f"n{n}-score"
        pr.write_portfolio_fixture(path, problem, beta_mode="repository-default-ramp")
        round2_io.read_bundle(path)  # exercises the hash/round-trip contract

        qubo = encode_select(problem)
        rng = np.random.default_rng(n + 1)
        spins = rng.choice([-1.0, 1.0], size=(8, n))
        result = pr.score_reads(problem, qubo, spins)
        assert result["returned_reads"] == 8
        assert np.isfinite(result["weights"]).all()
        assert np.isfinite(result["objective"])
        assert result["weights"].shape == (n,)
