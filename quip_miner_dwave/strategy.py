"""The round strategy: is this qblock worth the headroom, or is a better hour coming?

Runs after the budget said yes, once per qblock boundary, and answers from a
memory snapshot only. The design and its reasoning are in
docs/superpowers/specs/2026-09-11-qpu-time-of-week-strategy-design.md.
"""

from __future__ import annotations

import logging
import math
import random
import tomllib
from dataclasses import dataclass
from typing import Optional

from quip_miner_dwave.budget import ParticipationDecision
from quip_miner_dwave.profile import Snapshot, SnapshotRefresher, slot_label, slot_of

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyConfig:
    """Operator knobs from ``Configure.backend_toml``.

    With the defaults and no history the miner behaves as it does today:
    every round the budget allows is joined. ``slot_advantage`` is what lets
    a materially better hour of the week defer a round; ``explore_fraction``
    keeps every slot measured even when the strategy would skip it.
    """

    min_win_probability: float = 0.0
    slot_advantage: float = 0.25
    explore_fraction: float = 0.10


# key -> (lower bound, upper bound or None)
STRATEGY_KEYS = {
    "min_win_probability": (0.0, 1.0),
    "slot_advantage": (0.0, None),
    "explore_fraction": (0.0, 1.0),
}


def strategy_config_from_toml(toml_text: str) -> StrategyConfig:
    """Lenient like the sampling defaults: the budget parser reports a bad document."""
    defaults = StrategyConfig()
    if not toml_text or not toml_text.strip():
        return defaults
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return defaults
    values = {}
    for key, (low, high) in STRATEGY_KEYS.items():
        default = getattr(defaults, key)
        raw = data.get(key)
        if raw is None:
            values[key] = default
            continue
        ok = (
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and low <= float(raw)
            and (high is None or float(raw) <= high)
        )
        if not ok:
            bound = f"between {low:g} and {high:g}" if high is not None else f"at least {low:g}"
            logger.warning("ignoring %s=%r from config: expected a number %s", key, raw, bound)
            values[key] = default
            continue
        values[key] = float(raw)
    return StrategyConfig(**values)


REASON_NO_DATA = "no-data"
REASON_EXPLORE = "explore"
REASON_SATURATED = "saturated"
REASON_BELOW_MIN = "below-min-p"
REASON_BETTER_SLOT = "better-slot"
REASON_GOOD_SHOT = "good-shot"

# The profile is weekly, so no deferral looks further ahead than that.
_HORIZON_S = 7 * 86_400

# The banking loop below runs under the session's dispatch lock, so its
# iteration count needs a hard ceiling independent of the week cap: a
# degenerate median round length (seconds, not minutes) could otherwise
# turn one boundary decision into a long spin.
_HORIZON_ROUNDS_MAX = 2000


@dataclass(frozen=True)
class RoundDecision:
    """The verdict on one qblock boundary and the numbers behind it."""

    join: bool
    reason: str
    slot: int
    # None on a no-data verdict: no history means no prediction was made,
    # not a prediction of zero.
    p_now: Optional[float]
    expected_jobs: Optional[float]
    jobs_per_s: Optional[float]
    lam: Optional[float]
    # The marginal win probability banking the headroom buys at the best
    # later round within the horizon, and where that round is.
    p_best: float
    wait_slot: Optional[int]
    wait_rounds: int
    saturates_in_rounds: Optional[int]


def _no_data(slot: int) -> RoundDecision:
    return RoundDecision(
        join=True,
        reason=REASON_NO_DATA,
        slot=slot,
        p_now=None,
        expected_jobs=None,
        jobs_per_s=None,
        lam=None,
        p_best=0.0,
        wait_slot=None,
        wait_rounds=0,
        saturates_in_rounds=None,
    )


def decide_round(
    *,
    now: float,
    headroom_us: float,
    accrual_us_per_s: float,
    period_end: float,
    snapshot: Optional[Snapshot],
    config: StrategyConfig,
    explore_draw: float,
) -> RoundDecision:
    """Join this round, or bank the headroom for a better hour of the week.

    Runs after the budget said yes. The allotment is use-it-or-lose-it, so
    skipping only pays when the banked headroom buys more at a later round
    than it buys now, and only while banking is still possible: once the
    headroom exceeds what any round can spend, or the period is about to
    reset, waiting throws QPU time away.
    """
    slot = slot_of(now)
    if snapshot is None or snapshot.lam_global is None:
        return _no_data(slot)
    if all(st.jobs_per_s is None for st in snapshot.slots):
        return _no_data(slot)

    length = snapshot.round_length_s
    access_s = snapshot.access_s_per_job

    def deliverable(h_us: float, s: int) -> float:
        st = snapshot.slots[s]
        by_budget = max(0.0, h_us) / 1_000_000.0 / access_s
        if st.jobs_per_s is None:
            return by_budget
        by_rate = st.jobs_per_s * max(0.0, length - (st.rtt_s or 0.0))
        return min(by_budget, by_rate)

    def p_win(s: int, jobs: float) -> float:
        return 1.0 - math.exp(-snapshot.lam_by_slot[s] * jobs)

    jobs_now = deliverable(headroom_us, slot)
    p_now = p_win(slot, jobs_now)

    # How much headroom one round can spend at best, across the week. Past
    # that, accrual is lost: the guard that keeps a prepaid allotment spent.
    cap_jobs = max(
        st.jobs_per_s * max(0.0, length - (st.rtt_s or 0.0))
        for st in snapshot.slots
        if st.jobs_per_s is not None
    )
    cap_us = cap_jobs * access_s * 1_000_000.0
    per_round_us = accrual_us_per_s * length
    if headroom_us >= cap_us or per_round_us <= 0:
        k_sat = 0
    else:
        k_sat = math.ceil((cap_us - headroom_us) / per_round_us)
    k_period = int(max(0.0, period_end - now) // length)
    horizon = min(k_sat, k_period, int(_HORIZON_S // length), _HORIZON_ROUNDS_MAX)

    p_best, k_best = 0.0, 0
    for k in range(1, horizon + 1):
        s_k = slot_of(now + k * length)
        banked = k * per_round_us
        gain = p_win(s_k, deliverable(headroom_us + banked, s_k)) - p_win(
            s_k, deliverable(banked, s_k)
        )
        if gain > p_best:
            p_best, k_best = gain, k

    def verdict(join: bool, reason: str) -> RoundDecision:
        return RoundDecision(
            join=join,
            reason=reason,
            slot=slot,
            p_now=p_now,
            expected_jobs=jobs_now,
            jobs_per_s=snapshot.slots[slot].jobs_per_s,
            lam=snapshot.lam_by_slot[slot],
            p_best=p_best,
            wait_slot=slot_of(now + k_best * length) if k_best else None,
            wait_rounds=k_best,
            saturates_in_rounds=k_sat,
        )

    if explore_draw < config.explore_fraction:
        return verdict(True, REASON_EXPLORE)
    if horizon == 0:
        return verdict(True, REASON_SATURATED)
    if p_now < config.min_win_probability:
        return verdict(False, REASON_BELOW_MIN)
    if k_best > 0 and p_best >= p_now * (1.0 + config.slot_advantage):
        return verdict(False, REASON_BETTER_SLOT)
    return verdict(True, REASON_GOOD_SHOT)


def describe_round_decision(generation: int, d: RoundDecision, headroom_us: float) -> str:
    """One log line per boundary, with the numbers the verdict rests on."""
    head = f"[QPU] qblock {generation}: {'join' if d.join else 'skip'} ({d.reason})"
    if d.reason == REASON_NO_DATA:
        return f"{head} | no history yet, mining every round the budget allows"
    # Every other reason comes from a snapshot with evidence, so decide_round
    # always sets both to real floats here; narrow for the format calls below.
    assert d.p_now is not None and d.expected_jobs is not None
    rate = f"{d.jobs_per_s:.2f}" if d.jobs_per_s is not None else "?"
    lam = f"{d.lam:.1e}" if d.lam is not None else "?"
    headroom = f"headroom {headroom_us / 1_000_000:.0f}s"
    if d.join:
        return (
            f"{head} | P(win) {100 * d.p_now:.1f}% from {d.expected_jobs:.0f} jobs "
            f"at {rate} jobs/s in {slot_label(d.slot)} | win rate {lam}/job | {headroom}"
        )
    wait = slot_label(d.wait_slot) if d.wait_slot is not None else "a later slot"
    return (
        f"{head} | P(win) {100 * d.p_now:.1f}% now; banking buys {100 * d.p_best:.1f}% "
        f"at {wait} in {d.wait_rounds} round(s) | {headroom}, saturates in "
        f"{d.saturates_in_rounds} round(s)"
    )


class RoundStrategy:
    """The gate's adapter: config, the latest snapshot, and the explore draw."""

    def __init__(
        self,
        config: StrategyConfig,
        snapshots: SnapshotRefresher,
        rng: Optional[random.Random] = None,
    ):
        self._config = config
        self._snapshots = snapshots
        self._rng = rng or random.Random()

    @property
    def config(self) -> StrategyConfig:
        return self._config

    def decide(self, now: float, budget: ParticipationDecision) -> RoundDecision:
        return decide_round(
            now=now,
            headroom_us=budget.headroom_us,
            accrual_us_per_s=budget.accrual_us_per_s,
            period_end=budget.period_end,
            snapshot=self._snapshots.latest(),
            config=self._config,
            explore_draw=self._rng.random(),
        )
