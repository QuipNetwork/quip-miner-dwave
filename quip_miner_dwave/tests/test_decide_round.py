"""The single round decision: join now, or bank the headroom for a better hour."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Optional

import pytest

from quip_miner_dwave.profile import SLOTS, SlotStats, Snapshot, slot_of
from quip_miner_dwave.strategy import (
    REASON_BELOW_MIN,
    REASON_BETTER_SLOT,
    REASON_EXPLORE,
    REASON_GOOD_SHOT,
    REASON_NO_DATA,
    REASON_SATURATED,
    StrategyConfig,
    decide_round,
    describe_round_decision,
)

MONDAY = datetime(2026, 9, 7, tzinfo=timezone.utc).timestamp()
NOW = MONDAY + 13 * 3600  # slot 13
SLOT_NOW = slot_of(NOW)
NEXT_HOUR = SLOT_NOW + 1
LAM = 1.5e-4  # per job, about what qpu-1 shows: 131 wins in 845k jobs
ROUND_S = 600.0
ACCESS_S = 0.046
HEADROOM_US = 7.6e6  # one round's accrual at 0.76 s/min
ACCRUAL_US_PER_S = HEADROOM_US / ROUND_S
FAR_FUTURE = NOW + 30 * 86_400


def _snapshot(
    rates: Optional[Dict[int, float]] = None,
    *,
    default_rate: Optional[float] = 1.1,
    lam: Optional[float] = LAM,
    lam_slots: Optional[Dict[int, float]] = None,
    rtt_s: float = 3.0,
) -> Snapshot:
    rates = rates or {}
    slots = [
        SlotStats(jobs_per_s=rates.get(s, default_rate), rtt_s=rtt_s, queue_s=None, evidence=1000.0)
        for s in range(SLOTS)
    ]
    lam_by_slot = [(lam_slots or {}).get(s, lam if lam is not None else 0.0) for s in range(SLOTS)]
    return Snapshot(
        built_at=NOW,
        slots=slots,
        lam_global=lam,
        lam_by_slot=lam_by_slot,
        round_length_s=ROUND_S,
        access_s_per_job=ACCESS_S,
        rounds_joined=100,
        wins=10,
        jobs_in_rounds=60_000,
        margin_jobs=60_000,
    )


def _decide(snapshot, *, headroom_us=HEADROOM_US, config=None, explore_draw=0.5, period_end=FAR_FUTURE):
    return decide_round(
        now=NOW,
        headroom_us=headroom_us,
        accrual_us_per_s=ACCRUAL_US_PER_S,
        period_end=period_end,
        snapshot=snapshot,
        config=config or StrategyConfig(),
        explore_draw=explore_draw,
    )


def _p(lam, jobs):
    return 1.0 - math.exp(-lam * jobs)


def test_no_snapshot_or_no_evidence_joins_with_no_data():
    for snap in (None, _snapshot(lam=None), _snapshot(default_rate=None)):
        d = _decide(snap)
        assert d.join and d.reason == REASON_NO_DATA
        assert d.slot == SLOT_NOW


def test_deliverable_jobs_is_the_smaller_of_budget_and_rate():
    by_budget = _decide(_snapshot())
    assert by_budget.expected_jobs == pytest.approx(HEADROOM_US / 1e6 / ACCESS_S)  # 165.2
    by_rate = _decide(_snapshot(), headroom_us=100e6)
    assert by_rate.expected_jobs == pytest.approx(1.1 * (ROUND_S - 3.0))  # 656.7


def test_p_now_follows_the_slot_win_rate():
    d = _decide(_snapshot(lam_slots={SLOT_NOW: 3 * LAM}))
    assert d.lam == pytest.approx(3 * LAM)
    assert d.p_now == pytest.approx(_p(3 * LAM, HEADROOM_US / 1e6 / ACCESS_S))


def test_banking_in_an_identical_slot_is_never_better():
    # Same rate and win rate everywhere: concavity says spend now.
    d = _decide(_snapshot())
    assert d.join and d.reason == REASON_GOOD_SHOT
    assert d.p_best < d.p_now


def test_a_higher_win_rate_within_the_horizon_defers():
    # The next hour's slot wins three times as often per job and can take
    # 3 jobs/s, so the headroom cap is 82 s and banking is possible for ten
    # rounds. Six rounds from now is the first round in that slot.
    d = _decide(_snapshot({NEXT_HOUR: 3.0}, lam_slots={NEXT_HOUR: 3 * LAM}))
    assert not d.join and d.reason == REASON_BETTER_SLOT
    assert d.wait_slot == NEXT_HOUR and d.wait_rounds == 6
    assert d.p_best >= d.p_now * 1.25
    assert d.saturates_in_rounds == 10


def test_a_better_slot_past_the_saturation_horizon_is_out_of_reach():
    # Same win rate advantage, but every slot delivers 1.1 jobs/s: the
    # headroom cap is 30 s, three rounds away, and the better slot is six.
    d = _decide(_snapshot(lam_slots={NEXT_HOUR: 3 * LAM}))
    assert d.join and d.reason == REASON_GOOD_SHOT
    assert d.saturates_in_rounds == 3 and d.wait_rounds < 6


def test_a_lifted_rate_cap_defers_even_at_the_same_win_rate():
    # Now: 0.1 jobs/s, so the round can only take 60 of the 165 jobs the
    # headroom would fund. Next hour: 3 jobs/s. Same win rate per job.
    d = _decide(_snapshot({SLOT_NOW: 0.1, NEXT_HOUR: 3.0}))
    assert d.expected_jobs == pytest.approx(0.1 * (ROUND_S - 3.0))
    assert not d.join and d.reason == REASON_BETTER_SLOT
    assert d.wait_slot == NEXT_HOUR


def test_headroom_no_round_can_spend_joins_at_once():
    # 0.2 jobs/s caps a round at ~119 jobs, 5.5 s of headroom: less than
    # one round's accrual, so banking is impossible from the start.
    d = _decide(_snapshot(lam_slots={NEXT_HOUR: 3 * LAM}, default_rate=0.2))
    assert d.join and d.reason == REASON_SATURATED
    assert d.saturates_in_rounds == 0


def test_saturated_headroom_joins_whatever_the_minimum():
    d = _decide(_snapshot({NEXT_HOUR: 3.0}), headroom_us=90e6, config=StrategyConfig(min_win_probability=0.99))
    assert d.join and d.reason == REASON_SATURATED


def test_the_period_end_bounds_the_horizon():
    d = _decide(_snapshot(lam_slots={NEXT_HOUR: 3 * LAM}), period_end=NOW + 500)
    assert d.join and d.reason == REASON_SATURATED


def test_below_the_minimum_probability_skips_while_banking_is_possible():
    d = _decide(_snapshot(), config=StrategyConfig(min_win_probability=0.5))
    assert not d.join and d.reason == REASON_BELOW_MIN
    assert d.p_now < 0.5


def test_the_explore_draw_joins_regardless():
    d = _decide(_snapshot(), config=StrategyConfig(min_win_probability=0.99, explore_fraction=0.1), explore_draw=0.05)
    assert d.join and d.reason == REASON_EXPLORE


def test_verdict_order_is_explore_saturated_minimum_better_slot():
    cfg = StrategyConfig(min_win_probability=0.99)
    # explore beats saturated
    assert _decide(_snapshot(), headroom_us=90e6, config=cfg, explore_draw=0.0).reason == REASON_EXPLORE
    # saturated beats the minimum
    assert _decide(_snapshot(), headroom_us=90e6, config=cfg).reason == REASON_SATURATED
    # the minimum beats a better slot that would otherwise defer
    d = _decide(_snapshot({NEXT_HOUR: 3.0}, lam_slots={NEXT_HOUR: 3 * LAM}), config=cfg)
    assert d.reason == REASON_BELOW_MIN


def test_the_log_line_names_the_numbers():
    join = _decide(_snapshot())
    line = describe_round_decision(340, join, HEADROOM_US)
    assert line.startswith("[QPU] qblock 340: join (good-shot)")
    assert "165 jobs" in line and "1.10 jobs/s" in line and "Mon 13h" in line and "headroom 8s" in line
    skip = _decide(_snapshot({NEXT_HOUR: 3.0}, lam_slots={NEXT_HOUR: 3 * LAM}))
    line = describe_round_decision(341, skip, HEADROOM_US)
    assert line.startswith("[QPU] qblock 341: skip (better-slot)")
    assert "Mon 14h" in line and "6 round(s)" in line and "saturates in 10 round(s)" in line
