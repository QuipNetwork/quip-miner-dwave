"""Tests for session-loop helpers — generation-based cancellation."""

from __future__ import annotations

from quip_miner_dwave.session_loop import _is_abandoned


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
