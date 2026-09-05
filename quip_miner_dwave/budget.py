"""QPU budget with even distribution across the quota period.

Replaces the v0.2/v0.3 daily reservoir. That model accumulated credit and then
burnt it in one continuous run, which left the miner dark for a contiguous
multi-hour block once the pool drained — long enough for the rest of the field
to close the gap while the QPU sat idle.

This model paces instead of bursting. The allowance at any instant is the flat
share of the budget that the elapsed part of the period has earned::

    allowance = budget * (now - period_start) / (period_end - period_start)

Mining is allowed while cumulative spend sits under that line. Spend comes from
:mod:`quip_miner_dwave.usage`, so it survives a restart, and the period follows
D-Wave's own quota semantics: a fixed day of the month, UTC, clamped to the
month's length.

Because spend only ever runs a fraction of a qblock past the line before the
gate shuts, and the line keeps rising, the miner idles for one qblock at a time
rather than one night at a time.
"""

from __future__ import annotations

import calendar
import logging
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from quip_miner_dwave.config import warn_unknown_fields
from quip_miner_dwave.usage import UsageLedger

logger = logging.getLogger(__name__)

# Where the usage ledger lives when the coordinator does not name a path. The
# deployment already mounts /data for config.toml and the attempts dashboard,
# so it is the one directory known to outlive the container.
DEFAULT_USAGE_DB = "/data/qpu-usage.db"

# Config keys the dwave backend recognizes in Configure.backend_toml. Anything
# else (outside SESSION_KEYS) is a typo and gets warned about, uniform with the
# Rust backends' unknown-field handling. Connection credentials (token/solver/
# region) are NOT here: they come from D-Wave's own config (dwave.conf + env),
# not the coordinator.
DWAVE_CONFIG_KEYS = frozenset(
    {
        "budget",
        "budget_seconds",
        "budget_reset_day",
        "usage_db",
        "anneal_time_us",
        "num_reads",
    }
)


class BudgetUnavailable(Exception):
    """The budget cannot be enforced, so the miner must not mine.

    Raised for an invalid quota shape or a ledger that will not open. Both mean
    the same thing operationally: spend would go unmetered.
    """


def warn_unknown_backend_keys(toml_text: str) -> None:
    """Parse ``Configure.backend_toml`` and warn on keys outside the dwave
    schema, matching the Rust backends' unknown-field warnings."""
    if not toml_text or not toml_text.strip():
        return
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return
    warn_unknown_fields("dwave", data.keys(), DWAVE_CONFIG_KEYS)


def parse_duration(duration_str: str) -> float:
    """Parse a duration string (``30s``, ``5m``, ``2h``, ``1d``, ``1w``) to seconds."""
    s = duration_str.strip().lower()
    if s.endswith("s") and not s.endswith("ms"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60.0
    if s.endswith("h"):
        return float(s[:-1]) * 3600.0
    if s.endswith("d"):
        return float(s[:-1]) * 86400.0
    if s.endswith("w"):
        return float(s[:-1]) * 604800.0
    return float(s)


def _reset_at(year: int, month: int, day: int) -> datetime:
    """The reset instant for one month, clamped to that month's length.

    A reset day of 31 has to mean "the 28th" in February; clamping keeps every
    month in the year exactly one period long with no gaps or overlaps.
    """
    last = calendar.monthrange(year, month)[1]
    return datetime(year, month, min(day, last), tzinfo=timezone.utc)


def period_bounds(now: float, reset_day: int) -> tuple[float, float]:
    """UTC ``[start, end)`` of the quota month containing ``now``."""
    dt = datetime.fromtimestamp(now, timezone.utc)
    start = _reset_at(dt.year, dt.month, reset_day)
    if dt < start:
        # This month's reset has not happened yet, so the live period opened
        # with last month's.
        year, month = (dt.year - 1, 12) if dt.month == 1 else (dt.year, dt.month - 1)
        start = _reset_at(year, month, reset_day)
    year, month = (
        (start.year + 1, 1) if start.month == 12 else (start.year, start.month + 1)
    )
    return start.timestamp(), _reset_at(year, month, reset_day).timestamp()


@dataclass
class BudgetConfig:
    """Quota shape: how much QPU time per reset period."""

    budget_seconds: float
    reset_day: int = 1
    usage_db: str = DEFAULT_USAGE_DB


@dataclass
class ParticipationDecision:
    """Whether the pacer will fund the next qblock, and why."""

    participate: bool
    headroom_us: float
    allowance_us: float
    spent_us: float
    period_start: float
    period_end: float
    seconds_until_headroom: float


class BudgetPacer:
    """Even-distribution gate over one period's QPU allotment."""

    def __init__(self, config: BudgetConfig, ledger: UsageLedger):
        self.config = config
        self.ledger = ledger

    def decide(self, now: float) -> ParticipationDecision:
        """Evaluate spend against the flat allowance line at ``now``."""
        start, end = period_bounds(now, self.config.reset_day)
        span = end - start
        elapsed = min(max(now - start, 0.0), span)
        budget_us = self.config.budget_seconds * 1_000_000
        allowance_us = budget_us * elapsed / span
        spent_us = self.ledger.spent_us_since(start)
        headroom_us = allowance_us - spent_us

        if headroom_us > 0:
            until = 0.0
        else:
            rate_us_per_s = budget_us / span
            # Time for the rising line to reach current spend; never longer
            # than the wait for the reset, which zeroes spend outright.
            catch_up = (
                -headroom_us / rate_us_per_s if rate_us_per_s > 0 else float("inf")
            )
            until = min(catch_up, max(0.0, end - now))

        return ParticipationDecision(
            participate=headroom_us > 0,
            headroom_us=headroom_us,
            allowance_us=allowance_us,
            spent_us=spent_us,
            period_start=start,
            period_end=end,
            seconds_until_headroom=until,
        )

    def record_access_time(self, qpu_access_time_us: float, now: float) -> None:
        """Bill a completed job against the period."""
        self.ledger.record(qpu_access_time_us, now=now)

    def stats(self, now: float) -> Dict[str, Any]:
        decision = self.decide(now)
        return {
            "budget_seconds": self.config.budget_seconds,
            "reset_day": self.config.reset_day,
            "period_start": decision.period_start,
            "period_end": decision.period_end,
            "allowance_seconds": decision.allowance_us / 1_000_000,
            "spent_seconds": decision.spent_us / 1_000_000,
            "headroom_seconds": decision.headroom_us / 1_000_000,
            "jobs_this_period": self.ledger.jobs_since(decision.period_start),
            "seconds_until_headroom": decision.seconds_until_headroom,
        }


def budget_from_backend_toml(toml_text: str) -> Optional[BudgetPacer]:
    """Build a pacer from ``Configure.backend_toml``, or None if unbudgeted.

    Raises :class:`BudgetUnavailable` when a budget is configured but cannot
    be enforced: an unmetered miner would spend the period's quota in a day, so
    refusing to start is the safe failure.
    """
    if not toml_text or not toml_text.strip():
        return None
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return None

    raw = data.get("budget") or data.get("budget_seconds")
    if raw is None:
        return None
    amount = float(raw) if isinstance(raw, (int, float)) else parse_duration(str(raw))

    reset_day = int(data.get("budget_reset_day", 1))
    if not 1 <= reset_day <= 31:
        raise BudgetUnavailable(
            f"budget_reset_day must be 1-31, got {reset_day}"
        )

    db_path = str(data.get("usage_db") or DEFAULT_USAGE_DB)
    try:
        ledger = UsageLedger(db_path)
    except Exception as exc:
        raise BudgetUnavailable(
            f"cannot open the QPU usage ledger at {db_path}: {exc}. "
            "Set usage_db to a writable path on a persistent volume; the miner "
            "will not mine without a durable record of spend."
        ) from exc

    return BudgetPacer(
        BudgetConfig(
            budget_seconds=amount,
            reset_day=reset_day,
            usage_db=db_path,
        ),
        ledger,
    )
