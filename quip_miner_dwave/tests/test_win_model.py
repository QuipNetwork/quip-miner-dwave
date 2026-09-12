"""The win model: a per-job win rate from margins and observed wins."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quip_miner_dwave.history import AttemptRoundSummary, HistoryStore, JobSample
from quip_miner_dwave.profile import (
    DEFAULT_ACCESS_S_PER_JOB,
    DEFAULT_ROUND_LENGTH_S,
    SnapshotRefresher,
    build_snapshot,
    posterior_win_rate,
    prior_win_rate,
    slot_of,
)

MONDAY = datetime(2026, 9, 7, tzinfo=timezone.utc).timestamp()
NOW = MONDAY + 6 * 86_400


def _round(store, start, generation, *, jobs, won, length=600):
    store.open_round(start, generation, joined=True, reason="budget", p_win=None, expected_jobs=None)
    store.close_round(start, start + length)
    for i in range(jobs):
        store.record_job(
            JobSample(
                completed_at=start + 1 + i,
                generation=generation,
                rtt_ms=3000,
                access_us=46_000,
                inflight_at_submit=1,
                reads=48,
                best_energy_milli=-14_540_000,
                target_milli=-14_554_000,
                hits=0,
            ),
            round_start_ts=int(start),
        )
    if won:
        store.apply_attempt_round(
            AttemptRoundSummary(
                generation=generation, first_ts_s=int(start) + 1, last_ts_s=int(start) + jobs,
                jobs=jobs, hits_coord=1, won=True, best_energy_milli=-14_560_000,
                threshold_milli=-14_554_000, access_us=46_000 * jobs, margins={},
            ),
            insert_missing=False,
        )


def test_lambda_prior_comes_from_the_margin_histogram():
    assert prior_win_rate({}) is None
    assert prior_win_rate({5: 100}) == pytest.approx(0.5 / 101)  # never cleared, still not zero
    assert prior_win_rate({-1: 2, 0: 1, 5: 97}) == pytest.approx(3.5 / 101)


def test_observed_wins_move_lambda_off_the_prior():
    # One pseudo-win at the prior rate: 1/prior pseudo-jobs.
    assert posterior_win_rate(0.001, wins=0, jobs=0) == pytest.approx(0.001)
    assert posterior_win_rate(0.001, wins=3, jobs=1000) == pytest.approx(4 / 2000)
    assert posterior_win_rate(0.001, wins=0, jobs=9000) == pytest.approx(1 / 10_000)


def test_no_history_yields_no_win_rate_and_defaults():
    snap = build_snapshot(HistoryStore(":memory:"), NOW)
    assert snap.lam_global is None
    assert snap.round_length_s == DEFAULT_ROUND_LENGTH_S
    assert snap.access_s_per_job == DEFAULT_ACCESS_S_PER_JOB
    assert all(s.jobs_per_s is None for s in snap.slots)


def test_a_slot_without_rounds_inherits_the_global_rate():
    store = HistoryStore(":memory:")
    monday_13 = MONDAY + 13 * 3600
    _round(store, monday_13, 2, jobs=200, won=True)
    _round(store, monday_13 + 700, 3, jobs=200, won=False)
    snap = build_snapshot(store, NOW)
    assert snap.lam_global is not None
    assert snap.rounds_joined == 2 and snap.wins == 1 and snap.jobs_in_rounds == 400
    assert snap.lam_by_slot[100] == pytest.approx(snap.lam_global)
    # The slot with a win in 400 jobs sits above a global rate that is the
    # same evidence plus the histogram prior.
    assert snap.lam_by_slot[slot_of(monday_13)] > snap.lam_global * 0.99


def test_the_global_rate_is_the_posterior_of_the_histogram_prior():
    store = HistoryStore(":memory:")
    _round(store, MONDAY + 13 * 3600, 2, jobs=200, won=True)
    snap = build_snapshot(store, NOW)
    prior = prior_win_rate(store.margin_counts(0))
    assert prior is not None
    assert snap.lam_global == pytest.approx(posterior_win_rate(prior, wins=1, jobs=200))


def test_rounds_with_no_margin_histogram_fall_back_to_the_round_rate_as_a_prior():
    # Seeded history from before margins were recorded: jobs and a won round
    # exist, but no job carried an energy to bin, so the histogram is empty.
    store = HistoryStore(":memory:")
    start = MONDAY + 13 * 3600
    store.open_round(start, 2, joined=True, reason="budget", p_win=None, expected_jobs=None)
    store.close_round(start, start + 600)
    for i in range(200):
        store.record_job(
            JobSample(
                completed_at=start + 1 + i,
                generation=2,
                rtt_ms=3000,
                access_us=46_000,
                inflight_at_submit=1,
                reads=48,
                best_energy_milli=None,
                target_milli=None,
                hits=0,
            ),
            round_start_ts=int(start),
        )
    store.apply_attempt_round(
        AttemptRoundSummary(
            generation=2, first_ts_s=int(start) + 1, last_ts_s=int(start) + 200,
            jobs=200, hits_coord=1, won=True, best_energy_milli=-14_560_000,
            threshold_milli=-14_554_000, access_us=46_000 * 200, margins={},
        ),
        insert_missing=False,
    )
    assert store.margin_counts(0) == {}
    snap = build_snapshot(store, NOW)
    assert snap.jobs_in_rounds == 200 and snap.wins == 1
    assert snap.lam_global == pytest.approx(posterior_win_rate((1 + 0.5) / (200 + 1), wins=1, jobs=200))


def test_round_length_is_the_median_of_closed_live_rounds():
    store = HistoryStore(":memory:")
    for i, length in enumerate((500, 900, 600)):
        _round(store, MONDAY + i * 3600, 10 + i, jobs=1, won=False, length=length)
    store.open_round(MONDAY + 5 * 3600, 20, joined=True, reason="budget", p_win=None, expected_jobs=None)  # still open
    assert build_snapshot(store, NOW).round_length_s == 600


def test_access_time_per_job_comes_from_the_hourly_sums():
    store = HistoryStore(":memory:")
    _round(store, MONDAY, 2, jobs=10, won=False)
    assert build_snapshot(store, NOW).access_s_per_job == pytest.approx(0.046)


def test_the_refresher_serves_the_latest_snapshot_without_a_thread():
    store = HistoryStore(":memory:")
    refresher = SnapshotRefresher(store, interval_s=3600.0)
    assert refresher.latest() is None
    snap = refresher.refresh_now()
    assert snap is not None and refresher.latest() is snap
    refresher.stop()


def test_a_failed_refresh_keeps_the_previous_snapshot(caplog):
    store = HistoryStore(":memory:")
    refresher = SnapshotRefresher(store, interval_s=3600.0)
    first = refresher.refresh_now()
    store.close()  # every query now raises
    assert refresher.refresh_now() is first
    assert any("snapshot" in r.getMessage() for r in caplog.records)
