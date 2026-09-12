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

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple


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
