"""The round strategy: is this qblock worth the headroom, or is a better hour coming?

Runs after the budget said yes, once per qblock boundary, and answers from a
memory snapshot only. The design and its reasoning are in
docs/superpowers/specs/2026-09-11-qpu-time-of-week-strategy-design.md.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass

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
