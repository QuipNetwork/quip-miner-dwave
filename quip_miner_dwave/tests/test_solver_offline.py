"""End-to-end: an offline solver parks credits instead of spinning on rejects.

A reject normally carries a credit refund, and the coordinator dispatches the
next staged job the moment the refund lands. While the solver is offline that
loop runs at SAPI round-trip speed for the whole qblock: 7,623 rejects in one
11-minute round during the Advantage2_system1 outage. The miner now holds the
credit until the next qblock boundary, where one probe per round is enough to
notice the solver is back.
"""

from __future__ import annotations

import threading
import time
from typing import List, cast

import numpy as np
import pytest
from dwave.cloud.exceptions import SolverOfflineError

from quip_solver_core import miner_pb2, wire

from quip_miner_dwave import session_loop
from quip_miner_dwave.ocean import OceanSampler, SampleResult
from quip_miner_dwave.usage import UsageLedger

JOB_GENERATION = 5
QUEUE_DEPTH = 3


class FlakySampler:
    """Raises the offline error for the first ``offline_jobs`` jobs, then
    answers normally."""

    def __init__(self, offline_jobs: int):
        self._offline_left = offline_jobs
        self.native_topology_hash = None

    def ensure_connected(self) -> None:
        pass

    def set_session_topology(self, nodes, edges) -> None:
        pass

    def close(self) -> None:
        pass

    def cancel_inflight(self, keys) -> int:
        return 0

    def drain_unobserved_access_us(self) -> int:
        return 0

    def sample(self, nodes, h, edges, j, **kwargs) -> SampleResult:
        if self._offline_left > 0:
            self._offline_left -= 1
            raise SolverOfflineError("Solver is offline.")
        return SampleResult(
            spins=np.array([[1, -1]], dtype=np.int8),
            variables=[0, 1],
            energies=[-1.0],
            device_access_time_us=46_000,
            num_reads=1,
        )


def _configure(usage_db: str) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        configure=miner_pb2.Configure(
            queue_depth=QUEUE_DEPTH,
            idle_timeout_s=300,
            backend_toml=f'budget = "3000s"\nusage_db = "{usage_db}"\n',
        )
    )


def _job(job_id: bytes, generation: int) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        job=miner_pb2.Job(
            job_id=job_id,
            kind=miner_pb2.ISING_SAMPLE,
            generation=generation,
            deadline_ms=int(time.time() * 1000) + 3_600_000,
            ising=miner_pb2.IsingProblem(
                h_milli_le32=wire.encode_i32_le([1000, -1000]),
                j_milli_le32=wire.encode_i32_le([500]),
                edges=miner_pb2.EdgeList(u=[0], v=[1]),
                num_reads=1,
            ),
        )
    )


def _cancel(generation: int) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(cancel=miner_pb2.Cancel(max_generation=generation))


def _topology() -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        topology=miner_pb2.Topology(
            nodes=[0, 1], edges=miner_pb2.EdgeList(u=[0], v=[1]), hash=b""
        )
    )


def _wait_for(sent: List, kind: str, count: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if sum(1 for m in sent if m.WhichOneof("msg") == kind) >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"never saw {count} {kind} message(s)")


def _run(monkeypatch, sampler, usage_db: str, script) -> List:
    """Drive run_session against ``script``, a generator that yields CoordMsgs
    and may inspect ``sent`` between yields. Returns the outbound messages."""
    sent: List = []

    class _Stub:
        def __init__(self, channel):
            pass

        def Session(self, request_iter):
            threading.Thread(
                target=lambda: [sent.append(m) for m in request_iter], daemon=True
            ).start()
            return script(sent)

    class _Channel:
        def close(self):
            pass

    class _Ready:
        def result(self, timeout=None):
            return True

    monkeypatch.setattr(
        session_loop.grpc, "insecure_channel", lambda *a, **k: _Channel()
    )
    monkeypatch.setattr(
        session_loop.grpc, "channel_ready_future", lambda ch: _Ready()
    )
    monkeypatch.setattr(session_loop.miner_pb2_grpc, "MinerServiceStub", _Stub)
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "test-token")
    session_loop.run_session(
        "unix:///tmp/nope.sock", "qpu-test", cast(OceanSampler, sampler)
    )
    return sent


def _credits(msgs) -> List[int]:
    return [m.job_request.credits for m in msgs if m.WhichOneof("msg") == "job_request"]


@pytest.fixture
def usage_db(tmp_path) -> str:
    return str(tmp_path / "usage.db")


def test_an_offline_solver_rejects_the_job_without_a_refund(monkeypatch, usage_db):
    boundary_at: List[int] = []

    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        yield _cancel(1)  # the boundary the gate joins on: credits go out here
        yield _topology()
        yield _job(b"down", JOB_GENERATION)
        _wait_for(sent, "reject", 1)
        boundary_at.append(len(sent))
        yield _cancel(JOB_GENERATION + 1)
        _wait_for(sent, "job_request", 2)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    sent = _run(monkeypatch, FlakySampler(offline_jobs=1), usage_db, script)

    rejects = [m for m in sent if m.WhichOneof("msg") == "reject"]
    assert [r.reject.reason for r in rejects] == [miner_pb2.OVERLOADED]
    # The join granted the pipeline depth. The reject carried no refund, so
    # nothing else went out until the boundary re-granted the parked credit.
    assert _credits(sent[: boundary_at[0]]) == [QUEUE_DEPTH]
    assert _credits(sent[boundary_at[0] :]) == [1]


def test_an_offline_solver_bills_the_ledger_nothing(monkeypatch, usage_db):
    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        yield _cancel(1)
        yield _topology()
        yield _job(b"down", JOB_GENERATION)
        _wait_for(sent, "reject", 1)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    _run(monkeypatch, FlakySampler(offline_jobs=1), usage_db, script)

    assert UsageLedger(usage_db).spent_us_since(0.0) == 0


def test_the_parked_credit_is_the_probe_that_finds_the_solver_back(
    monkeypatch, usage_db
):
    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        yield _cancel(1)
        yield _topology()
        yield _job(b"down", JOB_GENERATION)
        _wait_for(sent, "reject", 1)
        yield _cancel(JOB_GENERATION + 1)
        _wait_for(sent, "job_request", 2)
        # The solver is back: the re-granted credit's job completes normally.
        yield _job(b"back", JOB_GENERATION + 2)
        _wait_for(sent, "result", 1)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    sent = _run(monkeypatch, FlakySampler(offline_jobs=1), usage_db, script)

    kinds = [m.WhichOneof("msg") for m in sent]
    assert kinds.count("result") == 1
    # A completed job refills its own credit as usual.
    assert _credits(sent)[-1] == 1
