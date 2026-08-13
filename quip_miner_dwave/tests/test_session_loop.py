"""Tests for session-loop helpers — generation-based cancellation."""

from __future__ import annotations

import logging

from quip_miner_dwave.session_loop import (
    _is_abandoned,
    log_attempt,
    log_progress,
)


def test_mempool_generation_is_never_abandoned():
    # Generation 0 = mempool job, no PoW cancellation scope. It must survive any
    # watermark, or a reseed would wrongly drop live mempool work.
    assert _is_abandoned(0, 0) is False
    assert _is_abandoned(0, 5) is False
    assert _is_abandoned(0, 10_000) is False


def test_pow_generation_at_or_below_watermark_is_abandoned():
    assert _is_abandoned(3, 5) is True  # below
    assert _is_abandoned(5, 5) is True  # at the watermark


def test_pow_generation_above_watermark_survives():
    assert _is_abandoned(6, 5) is False  # newer than the reseed
    assert _is_abandoned(1, 0) is False  # no cancel yet (watermark 0)


# First 8 bytes of a longer job id, matching the observed CPU miner line.
_JOB_ID = bytes.fromhex("1858b8caf35ae0df") + b"\x00\x01"


def test_attempt_line_matches_the_shared_format(caplog):
    # A finished attempt is info. Energy is milli internally and whole units
    # on the line. Wall and device use the shared duration buckets.
    with caplog.at_level(logging.INFO):
        log_attempt(
            _JOB_ID,
            energy_milli=-14_245_000,
            valid=0,
            total=106,
            wall_ms=21_000,
            device_ms=21_000,
        )
    assert (
        "[quip-miner-dwave] attempt 1858b8caf35ae0df..: energy -14245, valid 0/106 | 21.0s wall, 21.0s device"
        in caplog.text
    )
    assert caplog.records[0].levelno == logging.INFO


def test_rejected_attempt_line_matches_the_shared_format(caplog):
    with caplog.at_level(logging.WARNING):
        log_attempt(_JOB_ID, rejected="MALFORMED", wall_ms=21_000)
    assert (
        "[quip-miner-dwave] attempt 1858b8caf35ae0df..: rejected MALFORMED | 21.0s wall"
        in caplog.text
    )
    assert caplog.records[0].levelno == logging.WARNING


def test_cancelled_attempt_line_is_debug(caplog):
    with caplog.at_level(logging.DEBUG):
        log_attempt(_JOB_ID, cancelled=True, wall_ms=21_000)
    assert (
        "[quip-miner-dwave] attempt 1858b8caf35ae0df..: cancelled after 21.0s"
        in caplog.text
    )
    assert caplog.records[0].levelno == logging.DEBUG


def test_progress_line_matches_the_shared_format(caplog):
    # 10 jobs in 20 seconds is 0.5 jobs/s. Best and the requirement use
    # whole energy units, not milli.
    with caplog.at_level(logging.INFO):
        log_progress(
            jobs_done=10,
            elapsed_s=20.0,
            reads=106,
            sweeps=0,
            best_energy_milli=-14_245_000,
            max_energy_milli=-14_000_000,
            min_solutions=1,
        )
    assert (
        "[quip-miner-dwave] progress: 10 jobs | 0.5 jobs/s | reads=106 sweeps=0 | best=-14245 | requires energy<=-14000, solutions>=1"
        in caplog.text
    )
    assert caplog.records[0].levelno == logging.INFO
