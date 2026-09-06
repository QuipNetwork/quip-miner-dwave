"""Uniform config-override discipline for the dwave miner.

Python mirror of the Rust ``quip_solver_core::config`` helpers so every miner
type behaves the same: config (from the coordinator's ``Configure``) overrides
CLI/env with a warning, and unrecognized keys warn. Credentials are not carried
here — the dwave miner uses D-Wave's own config (dwave.conf + env).
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from typing import Iterable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Keys consumed by the shared session layer, not any single backend; never
# warned about as unknown. Mirrors miner-core's SESSION_KEYS.
SESSION_KEYS = frozenset({"num_sweeps"})

T = TypeVar("T")


def config_override(name: str, cli: T, from_config: Optional[T]) -> T:
    """Resolve one setting with ``config > CLI`` precedence.

    Warns and returns the config value only on an effective change; otherwise
    returns ``cli`` silently.
    """
    if from_config is not None and from_config != cli:
        logger.warning(
            "config overrides %s: %s -> %s (from coordinator)", name, cli, from_config
        )
        return from_config
    return cli


def warn_unknown_fields(
    backend: str, present: Iterable[str], known: Iterable[str]
) -> None:
    """Warn once per unrecognized key. Session keys are never flagged."""
    known_all = set(known) | SESSION_KEYS
    for key in present:
        if key not in known_all:
            logger.warning("config: unknown field '%s' for %s (ignored)", key, backend)


@dataclass(frozen=True)
class SamplingDefaults:
    """Session-wide sampling parameters from ``Configure.backend_toml``.

    These are *defaults*, the lowest rung of the per-job precedence ladder in
    :func:`quip_miner_dwave.job._sampling_params`: the job's own
    ``IsingProblem`` wins, then the session ``SetTarget``, then these, then the
    hard-coded fallback. Zero means "not set" on the wire, so it means the same
    here and leaves the next rung down in charge.

    ``anneal_time_us`` at 0 leaves D-Wave's own default anneal in place.
    """

    num_reads: int = 0
    anneal_time_us: int = 0


def sampling_defaults_from_toml(toml_text: str) -> SamplingDefaults:
    """Read ``num_reads`` and ``anneal_time_us`` from ``backend_toml``.

    A malformed document yields the empty defaults rather than raising: the
    budget parser is the one that refuses to run on unparseable config (an
    unmetered miner is a spending risk, a default anneal time is not), so
    reporting the same failure twice would only obscure it.
    """
    if not toml_text or not toml_text.strip():
        return SamplingDefaults()
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return SamplingDefaults()

    def non_negative(key: str) -> int:
        raw = data.get(key)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            if raw is not None:
                logger.warning(
                    "ignoring %s=%r from config: expected a non-negative integer",
                    key,
                    raw,
                )
            return 0
        return raw

    return SamplingDefaults(
        num_reads=non_negative("num_reads"),
        anneal_time_us=non_negative("anneal_time_us"),
    )
