import numpy as np
import pytest

quip_msa = pytest.importorskip("quip_msa")


def test_a_seeded_run_returns_each_read_in_the_row_of_its_seed():
    # A ferromagnetic ring has two ground states. At a very cold start beta
    # neither seed can move, so the output rows show the input order.
    n = 16
    h = np.zeros(n)
    edges = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int64)
    j = -np.ones(n)
    seeds = np.stack([np.ones(n, dtype=np.int8), -np.ones(n, dtype=np.int8)])
    spins, _ = quip_msa.Msa().sample(
        h, edges, j, num_sweeps=8, num_reads=2, seed=5,
        beta_range=(0.1, 50.0), initial_spins=seeds, start_beta=49.0,
    )
    assert np.array_equal(spins, seeds)
