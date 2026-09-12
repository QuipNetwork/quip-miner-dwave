"""Durable history behind the round strategy.

Three records, all in the usage database next to ``qpu_usage_hourly``:

* ``qpu_throughput_hourly``: operational sums per UTC hour (completions, busy
  time, round trips, D-Wave service time, access time, concurrency). Every
  throughput metric derives from these at query time, so the estimator can
  change without a migration.
* ``miner_rounds``: one row per qblock round the miner saw, with what the
  strategy predicted at the boundary and what then happened.
* ``energy_margin_daily``: histogram of each job's best energy relative to
  the round's target, in whole energy units. Margins pool across problem
  instances and difficulty changes; raw energies do not.

Sums per hour rather than a row per job, for the same reason as ``usage.py``:
a busy day is up to ~150k jobs and the estimator only ever wants aggregates.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from quip_miner_dwave.usage import SECONDS_PER_HOUR, hour_floor

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400
MILLI_PER_UNIT = 1000

SOURCE_LIVE = "live"
SOURCE_ATTEMPTS = "attempts"

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qpu_throughput_hourly (
    hour_start_s      INTEGER PRIMARY KEY,
    source            TEXT    NOT NULL,
    busy_ms           INTEGER NOT NULL DEFAULT 0,
    jobs              INTEGER NOT NULL DEFAULT 0,
    wasted_jobs       INTEGER NOT NULL DEFAULT 0,
    rtt_ms_sum        INTEGER NOT NULL DEFAULT 0,
    rtt_ms_sumsq      INTEGER NOT NULL DEFAULT 0,
    rtt_ms_max        INTEGER NOT NULL DEFAULT 0,
    sapi_ms_sum       INTEGER NOT NULL DEFAULT 0,
    sapi_jobs         INTEGER NOT NULL DEFAULT 0,
    access_us_sum     INTEGER NOT NULL DEFAULT 0,
    inflight_sum      INTEGER NOT NULL DEFAULT 0,
    reads             INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS miner_rounds (
    start_ts_s        INTEGER PRIMARY KEY,
    generation        INTEGER NOT NULL,
    source            TEXT    NOT NULL,
    end_ts_s          INTEGER,
    target_milli      INTEGER,
    joined            INTEGER NOT NULL DEFAULT 0,
    reason            TEXT    NOT NULL DEFAULT '',
    p_win             REAL,
    expected_jobs     REAL,
    jobs              INTEGER NOT NULL DEFAULT 0,
    reads             INTEGER NOT NULL DEFAULT 0,
    hits              INTEGER NOT NULL DEFAULT 0,
    hits_coord        INTEGER NOT NULL DEFAULT 0,
    best_energy_milli INTEGER,
    access_us         INTEGER NOT NULL DEFAULT 0,
    won               INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS energy_margin_daily (
    day_start_s       INTEGER NOT NULL,
    margin_unit       INTEGER NOT NULL,
    jobs              INTEGER NOT NULL,
    PRIMARY KEY (day_start_s, margin_unit)
);
CREATE TABLE IF NOT EXISTS seeded_dirs (
    dir_name          TEXT PRIMARY KEY,
    lines_seen        INTEGER NOT NULL
);
"""


def day_floor(ts: float) -> int:
    """Floor a unix timestamp to the start of its UTC day."""
    return int(ts) // SECONDS_PER_DAY * SECONDS_PER_DAY


def margin_unit(best_energy_milli: int, target_milli: int) -> int:
    """Whole energy units between a job's best energy and the round target.

    Zero or negative means the job cleared the target. Ceiling division on
    the shortfall keeps "one milli over" in bin 1 and "one milli under" in
    bin 0, so ``bin <= 0`` is exactly "cleared".
    """
    return -((target_milli - best_energy_milli) // MILLI_PER_UNIT)


def split_by_hour(start: float, end: float) -> List[Tuple[int, int]]:
    """Split ``[start, end)`` into ``(hour_start_s, milliseconds)`` pieces."""
    out: List[Tuple[int, int]] = []
    cursor = start
    while cursor < end:
        hour = hour_floor(cursor)
        piece_end = min(end, float(hour + SECONDS_PER_HOUR))
        out.append((hour, int(round((piece_end - cursor) * 1000))))
        cursor = piece_end
    return out


class BusyClock:
    """Wall time with at least one job in flight, attributed to UTC hours.

    Busy time is emitted at every completion rather than when the last job
    finishes, so an hour row is never more than one job behind while the
    miner runs continuously.
    """

    def __init__(self) -> None:
        self._inflight = 0
        self._since: Optional[float] = None

    @property
    def inflight(self) -> int:
        return self._inflight

    def start(self, now: float) -> None:
        if self._inflight == 0:
            self._since = now
        self._inflight += 1

    def stop(self, now: float) -> List[Tuple[int, int]]:
        """Close the busy time since the last emission; return its pieces."""
        if self._inflight == 0 or self._since is None:
            return []
        self._inflight -= 1
        pieces = split_by_hour(self._since, now)
        self._since = now if self._inflight > 0 else None
        return pieces


@dataclass(frozen=True)
class JobSample:
    """What one completed job contributes to the history."""

    completed_at: float
    generation: int
    rtt_ms: int
    access_us: int
    inflight_at_submit: int
    reads: int
    best_energy_milli: Optional[int]
    target_milli: Optional[int]
    hits: int
    sapi_ms: Optional[int] = None


class HistoryStore:
    """SQLite record of throughput, rounds and margins. One lock, one file."""

    def __init__(self, path: str):
        self._path = path
        if path != ":memory:":
            parent = Path(path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        # Worker threads write, the seed thread writes, the refresher reads:
        # one connection behind a lock, as the usage ledger does.
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        for pragma in _PRAGMAS:
            self._db.execute(pragma)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # -- hourly ---------------------------------------------------------

    def add_busy(self, pieces: Sequence[Tuple[int, int]]) -> None:
        if not pieces:
            return
        with self._lock:
            self._db.executemany(
                "INSERT INTO qpu_throughput_hourly (hour_start_s, source, busy_ms) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(hour_start_s) DO UPDATE SET "
                "source = excluded.source, busy_ms = busy_ms + excluded.busy_ms",
                [(hour, SOURCE_LIVE, ms) for hour, ms in pieces],
            )
            self._db.commit()

    def record_job(self, sample: JobSample, round_start_ts: Optional[int]) -> None:
        """Fold one completed job into its hour, its round and the histogram."""
        hour = hour_floor(sample.completed_at)
        sapi_ms = sample.sapi_ms if sample.sapi_ms is not None else 0
        sapi_jobs = 1 if sample.sapi_ms is not None else 0
        with self._lock:
            self._db.execute(
                "INSERT INTO qpu_throughput_hourly (hour_start_s, source, jobs, "
                "rtt_ms_sum, rtt_ms_sumsq, rtt_ms_max, sapi_ms_sum, sapi_jobs, "
                "access_us_sum, inflight_sum, reads) "
                "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(hour_start_s) DO UPDATE SET "
                "source = excluded.source, "
                "jobs = jobs + 1, "
                "rtt_ms_sum = rtt_ms_sum + excluded.rtt_ms_sum, "
                "rtt_ms_sumsq = rtt_ms_sumsq + excluded.rtt_ms_sumsq, "
                "rtt_ms_max = MAX(rtt_ms_max, excluded.rtt_ms_max), "
                "sapi_ms_sum = sapi_ms_sum + excluded.sapi_ms_sum, "
                "sapi_jobs = sapi_jobs + excluded.sapi_jobs, "
                "access_us_sum = access_us_sum + excluded.access_us_sum, "
                "inflight_sum = inflight_sum + excluded.inflight_sum, "
                "reads = reads + excluded.reads",
                (
                    hour,
                    SOURCE_LIVE,
                    sample.rtt_ms,
                    sample.rtt_ms * sample.rtt_ms,
                    sample.rtt_ms,
                    sapi_ms,
                    sapi_jobs,
                    sample.access_us,
                    sample.inflight_at_submit,
                    sample.reads,
                ),
            )
            if sample.best_energy_milli is not None and sample.target_milli is not None:
                self._db.execute(
                    "INSERT INTO energy_margin_daily (day_start_s, margin_unit, jobs) "
                    "VALUES (?, ?, 1) "
                    "ON CONFLICT(day_start_s, margin_unit) DO UPDATE SET jobs = jobs + 1",
                    (
                        day_floor(sample.completed_at),
                        margin_unit(sample.best_energy_milli, sample.target_milli),
                    ),
                )
            if round_start_ts is not None:
                self._db.execute(
                    "UPDATE miner_rounds SET "
                    "jobs = jobs + 1, reads = reads + ?, hits = hits + ?, "
                    "access_us = access_us + ?, "
                    "best_energy_milli = CASE WHEN ? IS NULL THEN best_energy_milli "
                    "WHEN best_energy_milli IS NULL OR ? < best_energy_milli THEN ? "
                    "ELSE best_energy_milli END "
                    "WHERE start_ts_s = ?",
                    (
                        sample.reads,
                        sample.hits,
                        sample.access_us,
                        sample.best_energy_milli,
                        sample.best_energy_milli,
                        sample.best_energy_milli,
                        round_start_ts,
                    ),
                )
            self._db.commit()

    def record_wasted(self, now: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO qpu_throughput_hourly (hour_start_s, source, wasted_jobs) "
                "VALUES (?, ?, 1) "
                "ON CONFLICT(hour_start_s) DO UPDATE SET "
                "source = excluded.source, wasted_jobs = wasted_jobs + 1",
                (hour_floor(now), SOURCE_LIVE),
            )
            self._db.commit()

    def hourly_rows(self, since_ts: float) -> List[sqlite3.Row]:
        with self._lock:
            return self._db.execute(
                "SELECT * FROM qpu_throughput_hourly WHERE hour_start_s >= ? "
                "ORDER BY hour_start_s",
                (hour_floor(since_ts),),
            ).fetchall()

    def margin_counts(self, since_ts: float) -> Dict[int, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT margin_unit, SUM(jobs) AS jobs FROM energy_margin_daily "
                "WHERE day_start_s >= ? GROUP BY margin_unit",
                (day_floor(since_ts),),
            ).fetchall()
        return {int(r["margin_unit"]): int(r["jobs"]) for r in rows}

    def close(self) -> None:
        with self._lock:
            self._db.close()
