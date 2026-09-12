"""Text report behind ``quip-dwave-qa --profile``.

Two 7x24 grids (throughput, D-Wave queue wait), the win summary split by
weekday and weekend, and the last rounds with what the strategy predicted
beside what happened. The SQLite file stays the machine-readable surface.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Callable, List, Optional

from quip_miner_dwave.history import HistoryStore
from quip_miner_dwave.profile import (
    DAY_NAMES,
    DEFAULT_WEEKS,
    HOURS_PER_DAY,
    SECONDS_PER_WEEK,
    SlotStats,
    build_snapshot,
    is_weekend,
    slot_label,
    slot_of,
    slot_stats,
)

RECENT_ROUNDS = 20


def _cell(value: Optional[float], evidence: float) -> str:
    # A slot borrowing everything from its parent shows nothing of its own.
    if value is None or evidence < 1.0:
        return "   ."
    return f"{value:4.1f}"


def _grid(title: str, stats: List[SlotStats], pick: Callable[[SlotStats], Optional[float]]) -> List[str]:
    lines = [title, "     " + " ".join(f"{h:>4d}" for h in range(HOURS_PER_DAY))]
    for day in range(7):
        cells = []
        for hour in range(HOURS_PER_DAY):
            st = stats[day * HOURS_PER_DAY + hour]
            cells.append(_cell(pick(st), st.evidence))
        lines.append(f"{DAY_NAMES[day]}  " + " ".join(cells))
    return lines


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%m-%d %H:%M")


def _summary(rows: List[sqlite3.Row], name: str) -> str:
    jobs = sum(int(r["jobs"]) for r in rows)
    wins = sum(int(r["won"]) for r in rows)
    line = f"{name}: {len(rows)} rounds joined, {wins} won, {jobs} jobs"
    if wins:
        line += f", {jobs / wins:.0f} jobs per win"
    return line


def render_profile(store: HistoryStore, now: float) -> str:
    since = now - DEFAULT_WEEKS * SECONDS_PER_WEEK
    stats = slot_stats(store.hourly_rows(since), now)
    lines: List[str] = []
    lines += _grid(
        "QPU throughput by hour of week (UTC), jobs/s. '.' means no evidence in that slot.",
        stats,
        lambda s: s.jobs_per_s,
    )
    lines.append("")
    lines += _grid("D-Wave queue wait by hour of week (UTC), seconds.", stats, lambda s: s.queue_s)
    lines.append("")
    rounds = store.rounds(since_ts=since)
    joined = [r for r in rounds if r["joined"]]
    lines.append(_summary(joined, "All"))
    lines.append(_summary([r for r in joined if not is_weekend(slot_of(r["start_ts_s"]))], "Weekdays"))
    lines.append(_summary([r for r in joined if is_weekend(slot_of(r["start_ts_s"]))], "Weekends"))
    snap = build_snapshot(store, now)
    if snap.lam_global is None:
        lines.append("Win model: no evidence yet")
    else:
        lines.append(
            f"Win model: {snap.lam_global:.2e}/job ({snap.wins} wins in "
            f"{snap.jobs_in_rounds} jobs over {snap.rounds_joined} rounds), "
            f"round length {snap.round_length_s:.0f}s, "
            f"access {snap.access_s_per_job * 1000:.0f} ms/job"
        )
    lines.append("")
    lines.append(f"Last {RECENT_ROUNDS} rounds (UTC):")
    lines.append("start        slot     verdict  reason           P(win)  jobs    hits  won")
    for r in rounds[:RECENT_ROUNDS]:
        p = "-" if r["p_win"] is None else f"{100 * r['p_win']:.1f}%"
        lines.append(
            f"{_utc(r['start_ts_s']):<12} {slot_label(slot_of(r['start_ts_s'])):<8} "
            f"{'join' if r['joined'] else 'skip':<8} {r['reason']:<16} {p:>6}  "
            f"{r['jobs']:>4}  {r['hits']:>6}  {'W' if r['won'] else ''}"
        )
    return "\n".join(lines)
