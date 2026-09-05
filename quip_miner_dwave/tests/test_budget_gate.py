"""Tests for the qblock participation gate.

The gate is the rule that the QPU never half-joins a round: credits go out on a
qblock boundary or not at all, and they come back the moment the budget line
is crossed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from quip_miner_dwave.budget import BudgetConfig, BudgetPacer
from quip_miner_dwave.session_loop import ParticipationGate
from quip_miner_dwave.usage import UsageLedger


def ts(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()


def gate_with(budget_s=3000.0, reset_day=9):
    led = UsageLedger(":memory:")
    pacer = BudgetPacer(
        BudgetConfig(budget_seconds=budget_s, reset_day=reset_day), led
    )
    return ParticipationGate(pacer), led


# Ten days into a 30-day period at 3000s/month: 1000s of allowance earned.
MID_PERIOD = ts(2026, 9, 19)


def test_a_fresh_gate_is_not_participating():
    # Nothing is granted until a qblock boundary says so, even with a full
    # allowance sitting there.
    gate, _ = gate_with()
    assert gate.participating is False


def test_a_job_arriving_before_any_boundary_is_refused():
    # The coordinator can only dispatch against credits we granted, but a stale
    # credit must not turn into a mid-round join.
    gate, _ = gate_with()
    result = gate.on_job(MID_PERIOD)
    assert result.allowed is False
    assert result.changed is False


def test_a_boundary_with_headroom_starts_participation():
    gate, _ = gate_with()
    result = gate.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None
    assert result.allowed is True
    assert result.changed is True  # caller grants credits on the change
    assert gate.participating is True


def test_a_boundary_without_headroom_sits_the_round_out():
    gate, led = gate_with()
    led.record(1500 * 1_000_000, now=MID_PERIOD)  # 500s past the line
    result = gate.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None
    assert result.allowed is False
    assert gate.participating is False


def test_the_same_generation_is_not_a_new_boundary():
    # Cancel can repeat a watermark; only an advance is a fresh qblock.
    gate, _ = gate_with()
    assert gate.on_qblock_boundary(100, MID_PERIOD) is not None
    assert gate.on_qblock_boundary(100, MID_PERIOD) is None
    assert gate.on_qblock_boundary(99, MID_PERIOD) is None
    assert gate.on_qblock_boundary(101, MID_PERIOD) is not None


def test_staying_in_across_a_boundary_does_not_re_grant():
    # Credits survive a reseed, so a second grant would over-fund the pipeline.
    gate, _ = gate_with()
    first = gate.on_qblock_boundary(100, MID_PERIOD)
    second = gate.on_qblock_boundary(101, MID_PERIOD)
    assert first is not None and first.changed is True
    assert second is not None and second.allowed is True
    assert second.changed is False  # no state flip -> caller grants nothing


def test_crossing_the_line_mid_qblock_parks_credits_immediately():
    gate, led = gate_with()
    gate.on_qblock_boundary(100, MID_PERIOD)
    assert gate.participating is True

    led.record(1500 * 1_000_000, now=MID_PERIOD)  # blow past the line mid-round
    result = gate.on_job(MID_PERIOD)
    assert result.allowed is False
    assert result.changed is True  # caller logs the stop once
    assert gate.participating is False


def test_the_stop_is_logged_once_not_per_rejected_job():
    gate, led = gate_with()
    gate.on_qblock_boundary(100, MID_PERIOD)
    led.record(1500 * 1_000_000, now=MID_PERIOD)
    assert gate.on_job(MID_PERIOD).changed is True
    # Every later job in the same shut round is a silent refusal.
    assert gate.on_job(MID_PERIOD).changed is False
    assert gate.on_job(MID_PERIOD).changed is False


def test_recovery_waits_for_a_boundary_not_for_the_line():
    # The line recovers continuously, but rejoining mid-round is exactly what
    # the gate exists to prevent.
    gate, led = gate_with()
    gate.on_qblock_boundary(100, MID_PERIOD)
    led.record(1500 * 1_000_000, now=MID_PERIOD)
    gate.on_job(MID_PERIOD)
    assert gate.participating is False

    # Six days later the line has climbed past the spend, but a job alone must
    # not restart participation.
    recovered = ts(2026, 9, 25)
    assert gate.on_job(recovered).allowed is False
    assert gate.participating is False

    # The next boundary does.
    result = gate.on_qblock_boundary(101, recovered)
    assert result is not None and result.allowed is True
    assert gate.participating is True


def test_a_sat_out_round_reports_the_wait():
    gate, led = gate_with()
    led.record(1500 * 1_000_000, now=MID_PERIOD)
    result = gate.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None
    assert result.decision.seconds_until_headroom > 0
