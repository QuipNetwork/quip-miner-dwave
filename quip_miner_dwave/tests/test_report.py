"""The --profile report reads back what the history recorded."""

from __future__ import annotations

from datetime import datetime, timezone

from quip_miner_dwave.history import HistoryStore, JobSample
from quip_miner_dwave.report import render_profile

MONDAY = datetime(2026, 9, 7, tzinfo=timezone.utc).timestamp()


def _store() -> HistoryStore:
    store = HistoryStore(":memory:")
    for i in range(300):
        store.record_job(
            JobSample(
                completed_at=MONDAY + 13 * 3600 + i,
                generation=2,
                rtt_ms=3000,
                access_us=46_000,
                inflight_at_submit=3,
                reads=48,
                best_energy_milli=-14_540_000,
                target_milli=-14_554_000,
                hits=0,
                sapi_ms=2900,
            ),
            round_start_ts=None,
        )
    store.add_busy([(int(MONDAY + 13 * 3600), 150_000)])  # 300 jobs in 150 s: 2.0 jobs/s
    store.open_round(int(MONDAY + 13 * 3600), 2, joined=True, reason="budget", p_win=0.042, expected_jobs=165.0)
    store.close_round(int(MONDAY + 13 * 3600), int(MONDAY + 13 * 3600 + 600))
    store.open_round(int(MONDAY + 5 * 86_400), 9, joined=False, reason="budget-sat-out", p_win=None, expected_jobs=None)
    return store


def test_the_report_shows_a_seven_by_twenty_four_grid():
    out = render_profile(_store(), now=MONDAY + 6 * 86_400)
    lines = out.splitlines()
    assert lines[0].startswith("QPU throughput by hour of week")
    mon = next(line for line in lines if line.startswith("Mon"))
    cells = mon.split()[1:]
    assert len(cells) == 24
    assert cells[13] == "2.0"
    assert cells[0] == "."  # no evidence in that slot
    assert sum(1 for line in lines if line[:3] in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")) == 14


def test_the_report_lists_recent_rounds_with_predicted_and_actual():
    out = render_profile(_store(), now=MONDAY + 6 * 86_400)
    assert "Weekdays: 1 rounds joined, 0 won" in out
    assert "Weekends: 0 rounds joined" in out
    rows = [line for line in out.splitlines() if " join " in line or " skip " in line]
    assert any("Mon 13h" in r and "join" in r and "4.2%" in r and "budget" in r for r in rows)
    assert any("Sat 00h" in r and "skip" in r and "budget-sat-out" in r for r in rows)


def test_the_hits_column_is_wide_enough_for_a_joined_round_s_read_count():
    # A joined round's hits (reads at or below target) can run into five
    # digits; a four-character column runs it into the next column.
    store = HistoryStore(":memory:")
    start = int(MONDAY + 13 * 3600)
    store.open_round(start, 2, joined=True, reason="budget", p_win=None, expected_jobs=None)
    store.record_job(
        JobSample(
            completed_at=MONDAY + 13 * 3600 + 1,
            generation=2,
            rtt_ms=3000,
            access_us=46_000,
            inflight_at_submit=1,
            reads=1,
            best_energy_milli=-100,
            target_milli=-50,
            hits=12_345,
        ),
        round_start_ts=start,
    )
    out = render_profile(store, now=MONDAY + 6 * 86_400)
    row = next(line for line in out.splitlines() if "Mon 13h" in line)
    assert "   12345" in row  # six-wide right-aligned: one pad space plus the two-space separator


def test_the_report_states_the_win_model():
    out = render_profile(_store(), now=MONDAY + 6 * 86_400)
    assert "Win model:" in out
    assert "/job" in out and "round length" in out


def test_the_report_says_when_there_is_no_win_model():
    from quip_miner_dwave.history import HistoryStore

    out = render_profile(HistoryStore(":memory:"), now=MONDAY)
    assert "Win model: no evidence yet" in out
