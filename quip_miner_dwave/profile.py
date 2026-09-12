"""Hour-of-week estimates from the hourly history.

The week is the period. D-Wave's shared queue follows its customers'
working hours, and the working hypothesis is that weekends beat weekday
working hours. Each of the 168 slots is treated as stationary and pooled
across weeks, never across days of the week: the days differ (Kim and
Whitt, 2014), and averaging them hides the very thing being looked for.

Every number derives from the operational sums in ``qpu_throughput_hourly``
at query time. Throughput is completions over busy time, mean round trip is
round-trip sum over completions, and the D-Wave queue wait is server
service time minus access time. Thin slots shrink toward their parent
(weekday or weekend, same hour) and then toward the global mean, so a slot
with no evidence never looks better or worse than the average.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

from quip_miner_dwave.history import HistoryStore

logger = logging.getLogger(__name__)


class Row(Protocol):
    """What a slot needs from a row: field lookup by name.

    A plain ``dict`` (tests) and ``sqlite3.Row`` (``HistoryStore.hourly_rows``)
    both satisfy this; a ``dict`` is a ``typing.Mapping``, but ``sqlite3.Row``
    is not, which is why this is a narrower ``Protocol`` instead.
    """

    def __getitem__(self, key: str, /) -> Any: ...

SLOTS = 168
HOURS_PER_DAY = 24
SECONDS_PER_WEEK = 7 * 86_400
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# 1970-01-01 was a Thursday, so epoch hour 0 is Thursday 00:00 UTC, which is
# 72 hours after the Monday 00:00 that slot 0 names. Adding 72 is subtracting
# 96, modulo 168.
_EPOCH_OFFSET_HOURS = 72

DEFAULT_WEEKS = 8
DEFAULT_HALF_LIFE_WEEKS = 4.0
# Pseudo-evidence, in jobs, behind each level of the prior. About one
# round's worth: fewer jobs than that and the parent's estimate leads.
DEFAULT_PSEUDO_JOBS = 200.0


def slot_of(ts: float) -> int:
    """Hour of the week in UTC, Monday 00:00 is 0."""
    return int((int(ts) // 3600 + _EPOCH_OFFSET_HOURS) % SLOTS)


def slot_label(slot: int) -> str:
    return f"{DAY_NAMES[slot // HOURS_PER_DAY]} {slot % HOURS_PER_DAY:02d}h"


def is_weekend(slot: int) -> bool:
    return slot // HOURS_PER_DAY >= 5


def parent_of(slot: int) -> Tuple[bool, int]:
    """Weekday-or-weekend and hour: the level a thin slot borrows from."""
    return is_weekend(slot), slot % HOURS_PER_DAY


def recency_weight(age_s: float, half_life_weeks: float = DEFAULT_HALF_LIFE_WEEKS) -> float:
    return 0.5 ** (age_s / SECONDS_PER_WEEK / half_life_weeks)


@dataclass(frozen=True)
class SlotStats:
    """Shrunk estimates for one hour of the week.

    ``None`` means no evidence at any level. ``evidence`` is the weighted
    job count observed in this slot itself, before shrinkage.
    """

    jobs_per_s: Optional[float]
    rtt_s: Optional[float]
    queue_s: Optional[float]
    evidence: float


class _Acc:
    """Weighted operational sums for one slot, parent, or the whole window."""

    __slots__ = ("jobs", "live_jobs", "busy_s", "rtt_s", "sapi_s", "sapi_jobs", "access_s")

    def __init__(self) -> None:
        self.jobs = 0.0
        # Seeded rows carry completions but no round trips; only live rows
        # may serve as the denominator for the round-trip mean.
        self.live_jobs = 0.0
        self.busy_s = 0.0
        self.rtt_s = 0.0
        self.sapi_s = 0.0
        self.sapi_jobs = 0.0
        self.access_s = 0.0

    def add(self, row: Row, w: float) -> None:
        jobs = float(row["jobs"])
        self.jobs += w * jobs
        if row["source"] == "live":
            self.live_jobs += w * jobs
        self.busy_s += w * float(row["busy_ms"]) / 1000.0
        self.rtt_s += w * float(row["rtt_ms_sum"]) / 1000.0
        self.sapi_s += w * float(row["sapi_ms_sum"]) / 1000.0
        self.sapi_jobs += w * float(row["sapi_jobs"])
        self.access_s += w * float(row["access_us_sum"]) / 1_000_000.0

    def rate(self) -> Optional[float]:
        return self.jobs / self.busy_s if self.busy_s > 0 else None

    def rtt(self) -> Optional[float]:
        return self.rtt_s / self.live_jobs if self.live_jobs > 0 else None

    def queue(self) -> Optional[float]:
        if self.sapi_jobs <= 0 or self.jobs <= 0:
            return None
        return max(0.0, self.sapi_s / self.sapi_jobs - self.access_s / self.jobs)

    def evidence_for(self, measure: str) -> float:
        """The job count backing one measure: seeded rows carry no round
        trip and no SAPI timing, so ``rtt`` and ``queue`` have less evidence
        than ``rate`` does whenever seeded jobs are in the mix."""
        if measure == "rtt":
            return self.live_jobs
        if measure == "queue":
            return self.sapi_jobs
        return self.jobs


def _shrink(
    value: Optional[float], n: float, prior: Optional[float], pseudo: float
) -> Optional[float]:
    if prior is None:
        return value
    if value is None:
        return prior
    return (n * value + pseudo * prior) / (n + pseudo)


def slot_stats(
    rows: Iterable[Row],
    now: float,
    *,
    weeks: int = DEFAULT_WEEKS,
    half_life_weeks: float = DEFAULT_HALF_LIFE_WEEKS,
    pseudo_jobs: float = DEFAULT_PSEUDO_JOBS,
) -> List[SlotStats]:
    """168 slot estimates from hourly rows, recency-weighted and shrunk."""
    per_slot = [_Acc() for _ in range(SLOTS)]
    parents: Dict[Tuple[bool, int], _Acc] = {}
    everything = _Acc()
    horizon = now - weeks * SECONDS_PER_WEEK
    for row in rows:
        hour = int(row["hour_start_s"])
        if hour < horizon or hour > now:
            continue
        w = recency_weight(now - hour, half_life_weeks)
        slot = slot_of(hour)
        per_slot[slot].add(row, w)
        parents.setdefault(parent_of(slot), _Acc()).add(row, w)
        everything.add(row, w)

    out: List[SlotStats] = []
    for slot in range(SLOTS):
        parent = parents.get(parent_of(slot), _Acc())
        mine = per_slot[slot]
        estimates = []
        for measure in ("rate", "rtt", "queue"):
            overall = getattr(everything, measure)()
            parent_est = _shrink(
                getattr(parent, measure)(), parent.evidence_for(measure), overall, pseudo_jobs
            )
            estimates.append(
                _shrink(getattr(mine, measure)(), mine.evidence_for(measure), parent_est, pseudo_jobs)
            )
        out.append(
            SlotStats(
                jobs_per_s=estimates[0],
                rtt_s=estimates[1],
                queue_s=estimates[2],
                evidence=mine.jobs,
            )
        )
    return out


# --- win model ---------------------------------------------------------

# The chain's convergence target for a round, used until a live round has
# closed. Live rounds are the real measure and take over immediately.
DEFAULT_ROUND_LENGTH_S = 600.0
# Measured on Advantage2_system1 at production size (num_reads=48).
DEFAULT_ACCESS_S_PER_JOB = 0.046
ROUND_LENGTH_SAMPLE = 100


def prior_win_rate(margins: Mapping[int, int]) -> Optional[float]:
    """The per-job probability of clearing the round target, Laplace-smoothed.

    ``None`` without a histogram, so a caller can tell "no evidence" from
    "never cleared". Half a pseudo-clear keeps a QPU that has never cleared
    a target on a small non-zero rate instead of a certain zero.
    """
    total = sum(margins.values())
    if total <= 0:
        return None
    cleared = sum(n for margin, n in margins.items() if margin <= 0)
    return (cleared + 0.5) / (total + 1.0)


def posterior_win_rate(prior: float, wins: float, jobs: float) -> float:
    """Gamma-Poisson update of a per-job win rate: one pseudo-win at ``prior``.

    Wins in a round are Poisson in the jobs delivered. A Gamma(1, 1/prior)
    prior has mean ``prior`` and the weight of a single win, so observed wins
    take over after about ``1/prior`` jobs and a QPU with real wins is judged
    by them.
    """
    return (1.0 + wins) / (1.0 / prior + jobs)


@dataclass(frozen=True)
class Snapshot:
    """Everything the boundary decision reads. Built off the session thread.

    ``lam_by_slot`` is meaningful only when ``lam_global`` is set: without
    any evidence at all it is all zeros, and every caller that reads it
    already returns on ``lam_global is None`` first.
    """

    built_at: float
    slots: List[SlotStats]
    # Per-job win rate. None without any evidence at all.
    lam_global: Optional[float]
    lam_by_slot: List[float]
    round_length_s: float
    access_s_per_job: float
    rounds_joined: int
    wins: int
    jobs_in_rounds: int
    margin_jobs: int


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build_snapshot(
    store: HistoryStore,
    now: float,
    *,
    weeks: int = DEFAULT_WEEKS,
    half_life_weeks: float = DEFAULT_HALF_LIFE_WEEKS,
    pseudo_jobs: float = DEFAULT_PSEUDO_JOBS,
) -> Snapshot:
    since = now - weeks * SECONDS_PER_WEEK
    rows = store.hourly_rows(since)
    slots = slot_stats(rows, now, weeks=weeks, half_life_weeks=half_life_weeks, pseudo_jobs=pseudo_jobs)

    jobs = sum(int(r["jobs"]) for r in rows)
    access_us = sum(int(r["access_us_sum"]) for r in rows)
    access_s = access_us / jobs / 1_000_000.0 if jobs > 0 and access_us > 0 else DEFAULT_ACCESS_S_PER_JOB

    rounds = store.rounds(since_ts=since)
    lengths = [
        float(r["end_ts_s"] - r["start_ts_s"])
        for r in rounds
        if r["source"] == "live" and r["end_ts_s"] is not None and r["end_ts_s"] > r["start_ts_s"]
    ][:ROUND_LENGTH_SAMPLE]
    round_length = _median(lengths) if lengths else DEFAULT_ROUND_LENGTH_S

    # `won` defaults to 0 until the coordinator's attempts file publishes the
    # round's outcome, so the newest round or two here still read as a loss
    # for a while. Conservative, and negligible against a window this wide.
    joined = [r for r in rounds if r["joined"] and int(r["jobs"]) > 0]
    wins = sum(int(r["won"]) for r in joined)
    jobs_in_rounds = sum(int(r["jobs"]) for r in joined)
    margins = store.margin_counts(since)
    prior = prior_win_rate(margins)
    if prior is None and jobs_in_rounds > 0:
        # Rounds without a histogram: seeded history from before margins
        # were recorded. The rounds themselves are the only prior there is.
        prior = (wins + 0.5) / (jobs_in_rounds + 1.0)

    if prior is None:
        lam_global: Optional[float] = None
        lam_by_slot = [0.0] * SLOTS
    else:
        lam_global = posterior_win_rate(prior, wins, jobs_in_rounds)
        slot_wins = [0.0] * SLOTS
        slot_jobs = [0.0] * SLOTS
        for r in joined:
            s = slot_of(float(r["start_ts_s"]))
            slot_wins[s] += int(r["won"])
            slot_jobs[s] += int(r["jobs"])
        lam_by_slot = [
            posterior_win_rate(lam_global, slot_wins[s], slot_jobs[s]) for s in range(SLOTS)
        ]

    return Snapshot(
        built_at=now,
        slots=slots,
        lam_global=lam_global,
        lam_by_slot=lam_by_slot,
        round_length_s=round_length,
        access_s_per_job=access_s,
        rounds_joined=len(joined),
        wins=wins,
        jobs_in_rounds=jobs_in_rounds,
        margin_jobs=sum(margins.values()),
    )


class SnapshotRefresher:
    """Rebuilds the snapshot on its own thread; the boundary reads the latest.

    A rebuild is a few aggregate queries over a few hundred rows, but it is
    IO on the mounted volume, and the boundary decision runs on the session
    thread under the dispatch lock. The decision therefore reads memory only
    and this thread does the reading.
    """

    def __init__(self, store: HistoryStore, *, interval_s: float = 300.0):
        self._store = store
        self._interval = interval_s
        self._lock = threading.Lock()
        self._latest: Optional[Snapshot] = None
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="dwave-history-snapshot", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> Optional[Snapshot]:
        with self._lock:
            return self._latest

    def refresh_now(self) -> Optional[Snapshot]:
        """Rebuild synchronously. Returns the snapshot now current."""
        try:
            snap = build_snapshot(self._store, time.time())
        except Exception as exc:  # noqa: BLE001 - keep serving the last good one
            logger.warning("history: snapshot refresh failed: %s", exc)
            return self.latest()
        with self._lock:
            self._latest = snap
        return snap

    def request_refresh(self) -> None:
        """Wake the thread early, after a seed or an outcome pickup."""
        self._wake.set()

    def stop(self) -> None:
        """Signal the thread to stop and wait briefly for it to notice.

        A refresh in flight reads the store; joining here with a bounded
        timeout keeps that read from racing the store's close during
        shutdown, without risking a hang if the thread was never started.
        """
        self._stopping.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stopping.is_set():
            self.refresh_now()
            self._wake.wait(self._interval)
            self._wake.clear()
