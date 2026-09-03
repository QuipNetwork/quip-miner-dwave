"""The budget gate must be reachable, and it must say so when it is not."""

import logging

import pytest

from quip_miner_dwave.budget import (
    DWAVE_CONFIG_KEYS,
    QPUTimeConfig,
    QPUTimeManager,
    budget_from_backend_toml,
)


def test_cap_below_threshold_is_lifted_and_reported(caplog):
    """A cap under min_block_budget makes should_mine() false forever.

    daily 80m / min_block 20m / cap 5m: the pool tops out at 300s and the gate
    opens at 1200s, so every job is rejected for the life of the process.
    """
    caplog.set_level(logging.ERROR)
    m = QPUTimeManager(
        QPUTimeConfig(
            daily_budget_seconds=4800.0,
            min_block_budget_seconds=1200.0,
            budget_cap_seconds=300.0,
            initial_budget_seconds=0.0,
        )
    )
    assert m._pool_cap_us == 1200.0 * 1_000_000
    assert "budget_cap" in caplog.text
    assert "min_block_budget" in caplog.text


def test_lifted_cap_lets_the_gate_open():
    m = QPUTimeManager(
        QPUTimeConfig(
            daily_budget_seconds=4800.0,
            min_block_budget_seconds=1200.0,
            budget_cap_seconds=300.0,
            initial_budget_seconds=0.0,
        )
    )
    # 1200s of pool at 4800s/86400s per second takes 6 hours of wall clock.
    start = m._last_accrual_s
    assert not m.should_mine(start).should_mine
    assert m.should_mine(start + 6 * 3600).should_mine


def test_cap_at_or_above_threshold_is_untouched(caplog):
    caplog.set_level(logging.ERROR)
    m = QPUTimeManager(
        QPUTimeConfig(
            daily_budget_seconds=4800.0,
            min_block_budget_seconds=1200.0,
            budget_cap_seconds=1800.0,
        )
    )
    assert m._pool_cap_us == 1800.0 * 1_000_000
    assert caplog.text == ""


def test_seconds_until_can_mine_is_finite_once_the_cap_is_lifted():
    m = QPUTimeManager(
        QPUTimeConfig(
            daily_budget_seconds=4800.0,
            min_block_budget_seconds=1200.0,
            budget_cap_seconds=300.0,
            initial_budget_seconds=0.0,
        )
    )
    est = m.should_mine(m._last_accrual_s)
    assert est.seconds_until_can_mine == pytest.approx(21600.0, rel=1e-3)


def test_zero_daily_budget_never_opens():
    m = QPUTimeManager(
        QPUTimeConfig(daily_budget_seconds=0.0, min_block_budget_seconds=90.0)
    )
    assert m.should_mine().seconds_until_can_mine == float("inf")


def test_initial_budget_is_a_recognized_key():
    """budget_from_backend_toml reads it, so the schema must list it or every
    config that sets it draws an unknown-field warning."""
    assert "initial_budget" in DWAVE_CONFIG_KEYS
    assert "initial_budget_seconds" in DWAVE_CONFIG_KEYS
    m = budget_from_backend_toml('daily_budget = "80m"\ninitial_budget = "10m"\n')
    assert m is not None
    assert m.should_mine().should_mine
