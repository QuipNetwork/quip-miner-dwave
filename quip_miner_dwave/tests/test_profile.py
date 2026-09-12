"""Hour-of-week estimates: slot math, recency, shrinkage."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quip_miner_dwave.profile import (
    SECONDS_PER_WEEK,
    is_weekend,
    parent_of,
    recency_weight,
    slot_label,
    slot_of,
    slot_stats,
)

# 2026-09-07 is a Monday.
MONDAY = datetime(2026, 9, 7, tzinfo=timezone.utc).timestamp()
DAY = 86_400


def _row(hour, *, jobs, busy_ms, rtt_ms_sum=0, sapi_ms_sum=0, sapi_jobs=0,
         access_us_sum=0, source="live"):
    return {
        "hour_start_s": int(hour),
        "source": source,
        "jobs": jobs,
        "busy_ms": busy_ms,
        "rtt_ms_sum": rtt_ms_sum,
        "sapi_ms_sum": sapi_ms_sum,
        "sapi_jobs": sapi_jobs,
        "access_us_sum": access_us_sum,
    }


def test_monday_midnight_utc_is_slot_zero():
    assert slot_of(MONDAY) == 0
    assert slot_of(MONDAY + 13 * 3600) == 13
    assert slot_of(MONDAY + 5 * DAY + 13 * 3600) == 133  # Saturday 13h
    assert slot_of(MONDAY + 7 * DAY) == 0
    assert slot_of(MONDAY - 1) == 167


def test_slot_labels_name_the_day_and_hour():
    assert slot_label(0) == "Mon 00h"
    assert slot_label(133) == "Sat 13h"
    assert slot_label(167) == "Sun 23h"


def test_weekend_slots_share_a_parent_by_hour():
    assert is_weekend(133) and is_weekend(6 * 24) and not is_weekend(4 * 24 + 23)
    assert parent_of(133) == (True, 13)
    assert parent_of(6 * 24 + 13) == (True, 13)
    assert parent_of(13) == (False, 13)


def test_recency_weight_halves_every_half_life():
    assert recency_weight(0.0) == 1.0
    assert recency_weight(4 * SECONDS_PER_WEEK) == pytest.approx(0.5)
    assert recency_weight(8 * SECONDS_PER_WEEK) == pytest.approx(0.25)


def test_heavy_evidence_reports_the_slot_s_own_rate():
    rows = [_row(MONDAY, jobs=10_000, busy_ms=5_000_000)]  # 2 jobs/s
    stats = slot_stats(rows, now=MONDAY + 7 * DAY)
    assert stats[0].jobs_per_s == pytest.approx(2.0)
    assert stats[0].evidence == pytest.approx(10_000 * recency_weight(7 * DAY))


def test_a_slot_without_evidence_inherits_the_global_estimate():
    rows = [_row(MONDAY, jobs=10_000, busy_ms=5_000_000)]
    stats = slot_stats(rows, now=MONDAY + 7 * DAY)
    assert stats[100].jobs_per_s == pytest.approx(2.0)
    assert stats[100].evidence == 0.0


def test_thin_evidence_is_pulled_toward_the_parent():
    rows = [
        _row(MONDAY, jobs=10_000, busy_ms=5_000_000),  # Mon 00h, 2 jobs/s
        _row(MONDAY + 2 * DAY, jobs=20, busy_ms=5_000),  # Wed 00h, 4 jobs/s, 20 jobs
    ]
    stats = slot_stats(rows, now=MONDAY + 3 * DAY)
    wed = stats[48].jobs_per_s
    assert wed is not None and 2.0 < wed < 2.5  # 20 jobs cannot outvote 200 pseudo-jobs


def test_older_weeks_count_less():
    rows = [
        _row(MONDAY, jobs=1000, busy_ms=1_000_000),  # 1 job/s, now
        _row(MONDAY - 4 * SECONDS_PER_WEEK, jobs=1000, busy_ms=250_000),  # 4 jobs/s, half weight
    ]
    stats = slot_stats(rows, now=MONDAY)
    # Weighted: 1500 jobs over 1125 busy seconds, not 2000 over 1250.
    assert stats[0].jobs_per_s == pytest.approx(1500 / 1125)


def test_rows_outside_the_window_are_ignored():
    rows = [
        _row(MONDAY - 9 * SECONDS_PER_WEEK, jobs=1000, busy_ms=1_000_000),
        _row(MONDAY + 3600, jobs=1000, busy_ms=1_000_000),  # in the future
    ]
    stats = slot_stats(rows, now=MONDAY)
    assert stats[0].jobs_per_s is None and stats[0].rtt_s is None


def test_queue_wait_is_service_time_minus_access_time():
    rows = [_row(MONDAY, jobs=100, busy_ms=100_000, rtt_ms_sum=300_000,
                 sapi_ms_sum=290_000, sapi_jobs=100, access_us_sum=4_600_000)]
    stats = slot_stats(rows, now=MONDAY + 3600)
    assert stats[0].rtt_s == pytest.approx(3.0)
    assert stats[0].queue_s == pytest.approx(2.9 - 0.046)


def test_seeded_rows_carry_throughput_but_no_round_trip():
    rows = [_row(MONDAY, jobs=100, busy_ms=100_000, access_us_sum=4_600_000, source="attempts")]
    stats = slot_stats(rows, now=MONDAY + 3600)
    assert stats[0].jobs_per_s == pytest.approx(1.0)
    assert stats[0].rtt_s is None and stats[0].queue_s is None
