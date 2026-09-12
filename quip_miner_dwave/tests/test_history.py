"""Tests for the throughput history: busy clock, hourly sums, margins."""

from __future__ import annotations

import dataclasses

from hypothesis import given, settings
from hypothesis import strategies as st

from quip_miner_dwave.history import (
    BusyClock,
    HistoryStore,
    JobSample,
    day_floor,
    margin_unit,
    split_by_hour,
)

HOUR = 1788796800  # 2026-09-06T00:00:00Z, an hour boundary

# A plain-dict-and-splat factory widens JobSample's mixed int/float fields to
# float under pyright, so the default sample is built once, typed, and
# per-test overrides go through dataclasses.replace instead.
_DEFAULT_SAMPLE = JobSample(
    completed_at=HOUR + 10.0,
    generation=5,
    rtt_ms=3050,
    access_us=46_000,
    inflight_at_submit=3,
    reads=48,
    best_energy_milli=-14_540_000,
    target_milli=-14_554_000,
    hits=0,
    sapi_ms=2900,
)


def _sample(**kw) -> JobSample:
    return dataclasses.replace(_DEFAULT_SAMPLE, **kw)


def test_split_by_hour_attributes_a_segment_to_each_hour_it_crosses():
    pieces = split_by_hour(HOUR + 3599.0, HOUR + 3601.5)
    assert pieces == [(HOUR, 1000), (HOUR + 3600, 1500)]


def test_split_by_hour_of_an_empty_segment_is_empty():
    assert split_by_hour(HOUR + 5.0, HOUR + 5.0) == []
    assert split_by_hour(HOUR + 6.0, HOUR + 5.0) == []


def test_busy_time_counts_overlapping_jobs_once():
    clock = BusyClock()
    clock.start(HOUR + 0.0)
    clock.start(HOUR + 1.0)
    first = clock.stop(HOUR + 2.0)
    second = clock.stop(HOUR + 4.0)
    assert first == [(HOUR, 2000)]
    assert second == [(HOUR, 2000)]
    assert clock.inflight == 0


def test_busy_time_is_emitted_at_every_completion():
    # A long run of overlapping jobs must not hold its busy time until the
    # very last one finishes: the hour row would lag the whole run.
    clock = BusyClock()
    clock.start(HOUR)
    clock.start(HOUR)
    assert clock.stop(HOUR + 1.0) == [(HOUR, 1000)]
    assert clock.inflight == 1
    assert clock.stop(HOUR + 1.5) == [(HOUR, 500)]


def test_a_stop_without_a_start_is_ignored():
    assert BusyClock().stop(HOUR) == []


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=0, max_value=20_000, allow_nan=False),
            st.floats(min_value=0.001, max_value=5_000, allow_nan=False),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_the_clock_s_busy_total_is_the_union_length_of_any_job_intervals(jobs):
    # Little's law needs busy time to be the measure of the union of the job
    # intervals, not their sum. Replay the intervals as events and compare
    # against an independent union computed by sorting and merging.
    intervals = [(HOUR + s, HOUR + s + d) for s, d in jobs]
    events = sorted(
        [(t0, 0) for t0, _ in intervals] + [(t1, 1) for _, t1 in intervals]
    )
    clock = BusyClock()
    total_ms = 0
    for t, kind in events:
        if kind == 0:
            clock.start(t)
        else:
            total_ms += sum(ms for _, ms in clock.stop(t))
    # Explicit merge, kept simple on purpose.
    union = 0.0
    cur_start, cur_end = None, None
    for t0, t1 in sorted(intervals):
        if cur_end is None or t0 > cur_end:
            if cur_end is not None:
                union += cur_end - cur_start
            cur_start, cur_end = t0, t1
        else:
            cur_end = max(cur_end, t1)
    # min_size=1 guarantees at least one interval, so the loop always sets
    # both; spell that out for the type checker.
    assert cur_start is not None and cur_end is not None
    union += cur_end - cur_start
    # One millisecond of rounding per emitted piece is the only slack.
    assert abs(total_ms / 1000.0 - union) <= 0.001 * (2 * len(events) + 1)


def test_a_job_sample_accumulates_into_its_hour_row():
    store = HistoryStore(":memory:")
    store.record_job(_sample(), round_start_ts=None)
    store.record_job(_sample(rtt_ms=2950, sapi_ms=None), round_start_ts=None)
    store.add_busy([(HOUR, 6000)])

    (row,) = store.hourly_rows(HOUR)
    assert row["jobs"] == 2
    assert row["busy_ms"] == 6000
    assert row["rtt_ms_sum"] == 6000
    assert row["rtt_ms_sumsq"] == 3050**2 + 2950**2
    assert row["rtt_ms_max"] == 3050
    assert row["sapi_ms_sum"] == 2900 and row["sapi_jobs"] == 1
    assert row["access_us_sum"] == 92_000
    assert row["inflight_sum"] == 6
    assert row["reads"] == 96
    assert row["source"] == "live"


def test_wasted_jobs_count_in_their_hour():
    store = HistoryStore(":memory:")
    store.record_wasted(HOUR + 100)
    store.record_wasted(HOUR + 3700)
    rows = store.hourly_rows(HOUR)
    assert [(r["hour_start_s"], r["wasted_jobs"], r["jobs"]) for r in rows] == [
        (HOUR, 1, 0),
        (HOUR + 3600, 1, 0),
    ]


def test_margin_unit_is_zero_or_negative_exactly_when_the_target_is_cleared():
    target = -14_554_000
    assert margin_unit(target, target) == 0
    assert margin_unit(target - 1, target) == 0  # one milli under: cleared
    assert margin_unit(target + 1, target) == 1  # one milli over: missed
    assert margin_unit(target - 1000, target) == -1
    assert margin_unit(target + 1000, target) == 1
    assert margin_unit(target + 1500, target) == 2


def test_margin_histogram_bins_the_job_best_against_the_round_target():
    store = HistoryStore(":memory:")
    store.record_job(_sample(best_energy_milli=-14_540_000), round_start_ts=None)
    store.record_job(_sample(best_energy_milli=-14_560_000), round_start_ts=None)
    store.record_job(_sample(best_energy_milli=None), round_start_ts=None)
    store.record_job(_sample(target_milli=None), round_start_ts=None)
    assert store.margin_counts(day_floor(HOUR)) == {14: 1, -6: 1}


def test_history_survives_reopening(tmp_path):
    path = str(tmp_path / "usage.db")
    store = HistoryStore(path)
    store.record_job(_sample(), round_start_ts=None)
    store.close()
    reopened = HistoryStore(path)
    assert reopened.hourly_rows(HOUR)[0]["jobs"] == 1
    reopened.close()


def test_history_shares_a_file_with_the_usage_ledger(tmp_path):
    from quip_miner_dwave.usage import UsageLedger

    path = str(tmp_path / "usage.db")
    ledger = UsageLedger(path)
    ledger.record(46_000, now=HOUR)
    store = HistoryStore(path)
    store.record_job(_sample(), round_start_ts=None)
    assert ledger.spent_us_since(HOUR) == 46_000
    assert store.hourly_rows(HOUR)[0]["jobs"] == 1
    store.close()
    ledger.close()
