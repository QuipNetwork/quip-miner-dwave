"""gRPC session loop: Hello → Welcome → Configure → credit/job cycle.

Mirrors the ``quip-mock-miner`` reference miner (quip-miner repo) using the
``quip_solver_core`` Python SDK. Uses the synchronous gRPC client with a request queue (reliable
over UDS).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional, Tuple

import grpc

from quip_solver_core import miner_pb2, miner_pb2_grpc, session as session_sdk

from quip_miner_dwave import (
    ALGORITHM,
    BACKEND,
    EXIT_CLEAN,
    EXIT_CONFIG_INVALID,
    EXIT_INTERNAL_FATAL,
    EXIT_TOKEN_REJECTED,
    FEATURES,
    MAX_EDGES,
    MAX_NODES,
)
from quip_miner_dwave.budget import (
    BudgetPacer,
    BudgetUnavailable,
    ParticipationDecision,
    budget_from_backend_toml,
    warn_unknown_backend_keys,
)
from quip_miner_dwave.config import (
    SamplingDefaults,
    queue_depth_from_toml,
    sampling_defaults_from_toml,
)
from quip_miner_dwave.job import handle_job
from quip_solver_core.session import DEFAULT_NUM_SWEEPS, num_sweeps_from_toml
from quip_miner_dwave.ocean import OceanSampler

logger = logging.getLogger(__name__)


def _surface_pool_failure(future) -> None:
    """Log a job-worker exception instead of letting the Future swallow it.

    ``handle_job`` answers every failure it can see (a sampler exception
    rejects OVERLOADED with a credit refund); this backstop catches the ones
    it cannot — a crash in the reply bookkeeping itself. Replies may be
    partially sent at that point, so no blind refund: the log line is the
    difference between a diagnosable incident and silence.
    """
    exc = future.exception()
    if exc is not None:
        logger.error("job worker crashed: %r", exc, exc_info=exc)

_STOP = object()

# How many submissions this backend keeps on the QPU at once when nobody says
# otherwise. Sized so that a QPU which becomes free can be fed at its own
# ceiling, which is Little's Law and has nothing to do with how contended the
# device happens to be today: a queue sized for a busy chip starves a free one.
#
#     depth = chip_throughput * round_trip
#
# Measured against Advantage2_system1 on a production-sized problem (4577
# nodes, 41514 couplers, num_reads=48): 43.2 ms of access time per job, so the
# chip tops out at 23.2 jobs/s, and an uncontended round trip is 1.57 s. That
# needs 36 in flight to saturate. Session logs show the round trip reaching
# 3.05 s under load, which needs 71.
#
# 96 holds the chip saturated through a 4.14 s round trip — 2.6x the clean
# figure and well past the worst observed — so a connectivity blip degrades
# throughput rather than stalling the device. Probed to 128 with no error at
# any depth: the Leap project's concurrency cap is nowhere near this.
#
# The cost of depth is reseed exposure: a Cancel catching a full pipeline
# strands this many jobs, and the ones D-Wave has already begun annealing are
# billed. That is affordable only because the Cancel path now reaches SAPI.
DEFAULT_QUEUE_DEPTH = 96

# Operator log token. Distinct from BACKEND ("dwave-qpu"), which is the
# protocol advertisement. Mixed-fleet lines use [quip-miner-dwave].
_LOG_BACKEND = "dwave"


def _short_job_id(job_id: bytes) -> str:
    """Render the leading bytes of a job id, matching quip-solver-core."""
    head = bytes(job_id[:8]).hex()
    if len(job_id) > 8:
        return head + ".."
    return head


def _energy_units(milli: int) -> int:
    """Floor-divide a milli-energy to whole units for a log line."""
    return milli // 1000


def _format_duration_ms(ms: int) -> str:
    """Render a millisecond duration with the shared miner buckets."""
    if ms < 1_000:
        return f"{ms}ms"
    if ms < 60_000:
        tenths = (ms + 50) // 100
        if tenths >= 600:
            return "1m 0s"
        secs = tenths // 10
        frac = tenths % 10
        return f"{secs}.{frac}s"
    if ms < 3_600_000:
        mins = ms // 60_000
        rem = ms % 60_000
        secs = (rem + 500) // 1_000
        if secs == 60:
            mins += 1
            secs = 0
        if mins >= 60:
            return f"{mins // 60}h {mins % 60}m"
        return f"{mins}m {secs}s"
    hours = ms // 3_600_000
    rem = ms % 3_600_000
    mins = (rem + 30_000) // 60_000
    if mins == 60:
        hours += 1
        mins = 0
    return f"{hours}h {mins}m"


def _reject_reason_name(reason: int) -> str:
    try:
        return miner_pb2.RejectReason.Name(reason)
    except ValueError:
        return str(reason)


@dataclass
class GateResult:
    """Outcome of asking the participation gate whether the QPU may sample."""

    allowed: bool
    changed: bool
    decision: ParticipationDecision


class ParticipationGate:
    """Funds whole qblocks, never half of one.

    The miner signals participation by granting credits: the coordinator
    dispatches against credits alone, so withholding them is how the QPU sits a
    round out. Two rules govern when they move, and both consult the same
    budget line:

    * A join only ever happens on a qblock boundary. Joining mid-round buys a
      share of a round the field started ahead of us, and the coordinator
      cancels whatever is still staged at the boundary anyway, so the access
      time it costs is spent for nothing.
    * A stop can happen at any time. Crossing the line mid-round parks the
      credits immediately; the next join still waits for a boundary.
    """

    def __init__(self, pacer: BudgetPacer):
        self._pacer = pacer
        self._participating = False
        self._boundary_generation = 0

    @property
    def participating(self) -> bool:
        return self._participating

    def on_qblock_boundary(
        self, generation: int, now: float
    ) -> Optional[GateResult]:
        """Re-decide at a new qblock. None when this is not a fresh boundary."""
        if generation <= self._boundary_generation:
            return None
        self._boundary_generation = generation
        decision = self._pacer.decide(now)
        changed = decision.participate != self._participating
        self._participating = decision.participate
        return GateResult(
            allowed=decision.participate, changed=changed, decision=decision
        )

    def on_job(self, now: float) -> GateResult:
        """May this job be sampled? Shuts participation the moment it may not."""
        if not self._participating:
            return GateResult(
                allowed=False, changed=False, decision=self._pacer.decide(now)
            )
        decision = self._pacer.decide(now)
        changed = not decision.participate
        if changed:
            self._participating = False
        return GateResult(
            allowed=decision.participate, changed=changed, decision=decision
        )


class InflightJobs:
    """Which job is on the QPU right now, and for which generation.

    The sampler cancels by job id and knows nothing about generations; the
    coordinator abandons generations and knows nothing about what this miner
    still has in flight. This map is the join between the two.

    Reads never consume: a worker thread owns the removal of its own job, so a
    Cancel that dropped entries here would strand their release and leak the
    map for the life of the session.
    """

    def __init__(self) -> None:
        self._generations: dict[bytes, int] = {}
        self._lock = threading.Lock()

    def add(self, job_id: bytes, generation: int) -> None:
        with self._lock:
            self._generations[bytes(job_id)] = generation

    def release(self, job_id: bytes) -> None:
        with self._lock:
            self._generations.pop(bytes(job_id), None)

    def abandoned(self, watermark: int) -> list[bytes]:
        """Job ids belonging to a generation the coordinator reseeded past."""
        with self._lock:
            return [
                job_id
                for job_id, generation in self._generations.items()
                if _is_abandoned(generation, watermark)
            ]


class CancelTally:
    """How often a SAPI cancel actually beat the anneal.

    D-Wave only refunds a problem it has not started annealing, so the value
    of cancelling is an empirical question this answers: ``requested`` counts
    the problems handed to SAPI, ``missed`` the ones that came back with a
    full result anyway, having been charged in full.
    """

    def __init__(self) -> None:
        self._requested = 0
        self._missed = 0

    def requested(self, count: int) -> None:
        self._requested += count

    def missed(self) -> None:
        self._missed += 1

    def summary(self) -> str:
        if self._requested == 0:
            return "no cancels yet"
        pct = 100.0 * self._missed / self._requested
        return (
            f"{self._requested} cancelled, "
            f"{self._missed} annealed anyway ({pct:.0f}% missed)"
        )


def resolve_queue_depth(*, coordinator: int, configured: int) -> int:
    """Pick the pipeline depth from the three rungs that can set it.

    Mirrors the per-job ladder in :func:`quip_miner_dwave.job._sampling_params`:
    the operator's ``backend_toml`` wins because QPU-specific knowledge lives
    there, then whatever the coordinator sent, then this backend's own measured
    default. Zero means "not set" on the wire and in the config, and the floor
    of 1 keeps a misconfiguration from granting no credits and mining nothing.
    """
    if configured > 0:
        return configured
    if coordinator > 0:
        return coordinator
    return max(1, DEFAULT_QUEUE_DEPTH)


def _bill_unobserved(sampler, pacer: Optional[BudgetPacer]) -> None:
    """Record access time D-Wave charged for but never reported back.

    A cancelled submission raises instead of returning samples, so it carries
    no ``device_access_time_us``. The sampler estimates the charge rather than
    assuming none (see ``OceanSampler._charge_unobserved``); this drains that
    estimate into the ledger. Draining on a job that ran unbudgeted would
    throw the estimate away, so the pacer's absence is checked first.
    """
    if pacer is None:
        return
    owed = sampler.drain_unobserved_access_us()
    if owed:
        pacer.record_access_time(owed, time.time())


def _log_qblock_joined(generation: int, decision: ParticipationDecision) -> None:
    logger.info(
        "[QPU] joining qblock %d: %.0fs of headroom on the budget line "
        "(spent %.0fs of %.0fs earned so far)",
        generation,
        decision.headroom_us / 1_000_000,
        decision.spent_us / 1_000_000,
        decision.allowance_us / 1_000_000,
    )


def _log_qblock_sat_out(generation: int, decision: ParticipationDecision) -> None:
    """One line when the gate shuts, with the reopening estimate.

    The per-attempt rejects sit at debug on purpose: a shut gate rejects every
    job the coordinator already staged, and narrating each one at warning
    buries the session log without adding anything this line did not say.
    """
    logger.info(
        "[QPU] sitting out qblock %d: %.0fs past the budget line, next window "
        "in %s",
        generation,
        -decision.headroom_us / 1_000_000,
        _format_duration_ms(int(decision.seconds_until_headroom * 1000)),
    )


def _log_line_crossed(decision: ParticipationDecision) -> None:
    logger.info(
        "[QPU] budget line crossed mid-qblock (%.0fs over); parking credits, "
        "next window in %s and the QPU rejoins at the qblock after that",
        -decision.headroom_us / 1_000_000,
        _format_duration_ms(int(decision.seconds_until_headroom * 1000)),
    )


def _utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _log_budget_configured(pacer: BudgetPacer, now: float) -> None:
    """State the quota, then say plainly why the QPU is not mining yet.

    A budgeted miner is idle between Configure and the next qblock boundary,
    and idle for whole qblocks whenever spend is ahead of the line. Both are
    correct and both look like a hang, so each one says why it is waiting and
    how long the wait is.
    """
    stats = pacer.stats(now)
    logger.info(
        "[QPU] budget %.0fs per period, resets day %d (period %s -> %s); "
        "spent %.0fs of %.0fs earned so far; jobs this period: %d",
        stats["budget_seconds"],
        stats["reset_day"],
        _utc_day(stats["period_start"]),
        _utc_day(stats["period_end"]),
        stats["spent_seconds"],
        stats["allowance_seconds"],
        stats["jobs_this_period"],
    )
    if stats["headroom_seconds"] > 0:
        logger.info(
            "[QPU] waiting: %.0fs of headroom is available, but credits are "
            "held until the next qblock boundary so the QPU joins a whole "
            "round rather than part of one",
            stats["headroom_seconds"],
        )
    else:
        logger.info(
            "[QPU] waiting: %.0fs past the budget line, so no credits are "
            "granted. Next window in %s, then the QPU joins at the following "
            "qblock boundary",
            -stats["headroom_seconds"],
            _format_duration_ms(int(stats["seconds_until_headroom"] * 1000)),
        )


def log_attempt(
    job_id: bytes,
    *,
    energy_milli: Optional[int] = None,
    valid: int = 0,
    total: int = 0,
    wall_ms: int = 0,
    device_ms: int = 0,
    rejected: Optional[str] = None,
    cancelled: bool = False,
) -> None:
    """Emit one per-attempt line in the shared miner format."""
    job = _short_job_id(job_id)
    wall = _format_duration_ms(wall_ms)
    if cancelled:
        logger.debug(
            "[quip-miner-%s] attempt %s: cancelled after %s",
            _LOG_BACKEND,
            job,
            wall,
        )
        return
    if rejected is not None:
        logger.warning(
            "[quip-miner-%s] attempt %s: rejected %s | %s wall",
            _LOG_BACKEND,
            job,
            rejected,
            wall,
        )
        return
    energy = "n/a" if energy_milli is None else str(_energy_units(energy_milli))
    device = _format_duration_ms(device_ms)
    logger.info(
        "[quip-miner-%s] attempt %s: energy %s, valid %d/%d | %s wall, %s device",
        _LOG_BACKEND,
        job,
        energy,
        valid,
        total,
        wall,
        device,
    )


def log_progress(
    *,
    jobs_done: int,
    elapsed_s: float,
    reads: int,
    sweeps: int,
    best_energy_milli: Optional[int],
    max_energy_milli: Optional[int],
    min_solutions: int,
) -> None:
    """Emit the shared progress line every N completed jobs."""
    rate = jobs_done / elapsed_s if elapsed_s > 0 else 0.0
    best = (
        "n/a"
        if best_energy_milli is None
        else str(_energy_units(best_energy_milli))
    )
    if max_energy_milli is None:
        requirement = "no target set"
    else:
        requirement = "requires energy<={}, solutions>={}".format(
            _energy_units(max_energy_milli),
            min_solutions,
        )
    logger.info(
        "[quip-miner-%s] progress: %d jobs | %.1f jobs/s | reads=%d sweeps=%d | best=%s | %s",
        _LOG_BACKEND,
        jobs_done,
        rate,
        reads,
        sweeps,
        best,
        requirement,
    )


def _status(
    miner_id: str, jobs_done: int = 0, abandoned: int = 0
) -> miner_pb2.MinerMsg:
    return miner_pb2.MinerMsg(
        status=miner_pb2.Status(
            miner_id=miner_id,
            utilization=0.0,
            jobs_done=jobs_done,
            abandoned_generation=abandoned,
        )
    )


def _is_abandoned(generation: int, watermark: int) -> bool:
    """True if a job's ``generation`` was abandoned by a Cancel(``watermark``).

    Generation ``0`` is a mempool job with no PoW cancellation scope and is
    never abandoned; a PoW job is abandoned once its generation is at or below
    the coordinator's reseed watermark. Mirrors the Rust ``CancelToken``.
    """
    return generation != 0 and generation <= watermark


def capabilities_message() -> miner_pb2.Capabilities:
    """Build the ``Capabilities`` message: what this backend supports.

    Must answer without touching the device, so it is a pure function of the
    same static numbers ``Hello`` advertises. ``--capabilities`` prints the
    protobuf JSON of this exact message and the in-session ``GetCapabilities``
    reply wraps it, so the two answers cannot drift apart (SPEC section 8).
    """
    return miner_pb2.Capabilities(
        backend=BACKEND,
        algorithm=ALGORITHM,
        supported_kinds=[miner_pb2.ISING_SAMPLE],
        max_nodes=MAX_NODES,
        max_edges=MAX_EDGES,
        features=list(FEATURES),
        protocol_version=session_sdk.PROTOCOL_VERSION,
        stream_width=1,
    )


def _unix_target(uri: str) -> str:
    """Normalize ``unix:///path`` or bare path to a grpc UDS target."""
    if uri.startswith("unix://"):
        path = uri[len("unix://") :]
    elif uri.startswith("unix:"):
        path = uri[len("unix:") :]
    else:
        path = uri
    if not path.startswith("/"):
        path = "/" + path
    # grpc-python accepts both unix:/abs and unix:///abs; prefer the triple-slash
    # form which matches tonic's unix:// advertisement after strip.
    return f"unix://{path}"


def run_session(
    coordinator_uri: str,
    miner_id: str,
    sampler: OceanSampler,
    *,
    budget: Optional[BudgetPacer] = None,
) -> int:
    """Run one miner session; return a process exit code."""
    target = _unix_target(coordinator_uri)
    try:
        hello = session_sdk.build_hello(
            miner_id,
            BACKEND,
            ALGORITHM,
            [miner_pb2.ISING_SAMPLE],
            MAX_NODES,
            MAX_EDGES,
            features=list(FEATURES),
        )
    except session_sdk.MissingToken:
        logger.error("QUIP_SESSION_TOKEN unset")
        return EXIT_TOKEN_REJECTED

    if sampler.native_topology_hash:
        hello.native_topology_hash = sampler.native_topology_hash

    out_q: queue.Queue = queue.Queue()
    # Pre-buffer Hello so the coordinator handshake has something immediately.
    out_q.put(miner_pb2.MinerMsg(hello=hello))

    def request_iter() -> Iterator[miner_pb2.MinerMsg]:
        while True:
            item = out_q.get()
            if item is _STOP:
                return
            yield item

    jobs_done = 0
    # Watermark set by Cancel(max_generation): any job at or below it belongs to
    # a generation the coordinator abandoned on reseed and is skipped, not sampled.
    cancel_watermark = 0
    grace_ms = 5000
    config: Optional[session_sdk.SessionConfig] = None
    session_nodes: list[int] = []
    session_edges: list[Tuple[int, int]] = []
    session_hash: Optional[bytes] = None
    session_target: Optional[miner_pb2.SetTarget] = None
    session_sweeps: int = DEFAULT_NUM_SWEEPS
    session_defaults = SamplingDefaults()
    pending_budget = budget
    # None until a budget is configured. While it is None the miner is
    # unmetered and mines every round, as it did before budget pacing.
    gate: Optional[ParticipationGate] = (
        ParticipationGate(pending_budget) if pending_budget is not None else None
    )
    queue_depth = DEFAULT_QUEUE_DEPTH
    # Pipeline: up to queue_depth QPU submissions in flight (overlaps cloud RTT).
    # jobs_done + pending_budget are shared with worker threads -> guard them.
    state_lock = threading.Lock()
    # What the QPU is chewing on right now, so a Cancel can reach it.
    inflight = InflightJobs()
    tally = CancelTally()
    job_pool: Optional[ThreadPoolExecutor] = None
    session_start = time.monotonic()
    best_energy_milli: Optional[int] = None
    PROGRESS_LOG_INTERVAL = 10

    def process_job(job, s_nodes, s_edges, s_hash, s_target, s_sweeps, s_defaults):
        # Runs on a pool thread: sample (blocking on the QPU), then enqueue
        # replies. Shared-state mutations are guarded by state_lock.
        nonlocal jobs_done, best_energy_milli
        started = time.monotonic()
        try:
            replies = handle_job(
                job,
                sampler,
                session_nodes=s_nodes,
                session_edges=s_edges,
                session_hash=s_hash,
                session_target=s_target,
                session_sweeps=s_sweeps,
                session_defaults=s_defaults,
            )
        finally:
            # Off the QPU one way or another: a later Cancel must not try to
            # drop a problem that has already finished.
            inflight.release(job.job_id)
        _bill_unobserved(sampler, pending_budget)
        wall_ms = int((time.monotonic() - started) * 1000)
        for reply in replies:
            kind = reply.WhichOneof("msg")
            if kind == "result" and pending_budget is not None:
                meta = reply.result.meta
                if meta is not None:
                    # Billed before the abandoned check, not after. D-Wave
                    # charged for this anneal whatever the coordinator decided
                    # to do with the answer, and a ledger that under-counts
                    # hands the pacer headroom the QPU has already spent.
                    #
                    # Billed outside state_lock, too. This commits to SQLite,
                    # which fsyncs the deployment's mounted volume, and the
                    # ledger already has its own lock. state_lock is what the
                    # session-loop thread takes to dispatch the next job, so
                    # holding it across this put every Cancel, Job and Ping
                    # behind one worker's disk write — at a pipeline depth of
                    # 96, behind all of them.
                    pending_budget.record_access_time(
                        meta.device_access_time_us, time.time()
                    )
            with state_lock:
                abandoned = _is_abandoned(job.generation, cancel_watermark)
                if kind in ("result", "reject") and abandoned:
                    # SPEC section 5: no Result for an abandoned generation.
                    # The Reject goes the same way — a cancelled submission
                    # surfaces as one, and answering a job the coordinator has
                    # already reseeded past is noise it cannot act on.
                    if kind == "result":
                        # It came back with samples, so SAPI ran the anneal
                        # despite the cancel. That is the miss rate.
                        tally.missed()
                    continue
                if kind == "result":
                    jobs_done += 1
                    meta = reply.result.meta
                    job_best = min(
                        (s.energy_milli for s in reply.result.solutions),
                        default=None,
                    )
                    if job_best is not None and (
                        best_energy_milli is None or job_best < best_energy_milli
                    ):
                        best_energy_milli = job_best
                    if s_target is not None:
                        valid = sum(
                            1
                            for s in reply.result.solutions
                            if s.energy_milli <= s_target.max_energy_milli
                        )
                    else:
                        valid = len(reply.result.solutions)
                    device_ms = (
                        meta.device_access_time_us // 1000
                        if meta is not None
                        else 0
                    )
                    log_attempt(
                        job.job_id,
                        energy_milli=job_best,
                        valid=valid,
                        total=len(reply.result.solutions),
                        wall_ms=wall_ms,
                        device_ms=device_ms,
                    )
                    if jobs_done % PROGRESS_LOG_INTERVAL == 0:
                        log_progress(
                            jobs_done=jobs_done,
                            elapsed_s=time.monotonic() - session_start,
                            reads=meta.reads if meta is not None else 0,
                            sweeps=meta.sweeps if meta is not None else 0,
                            best_energy_milli=best_energy_milli,
                            max_energy_milli=(
                                s_target.max_energy_milli
                                if s_target is not None
                                else None
                            ),
                            min_solutions=(
                                int(s_target.min_solutions)
                                if s_target is not None
                                else 0
                            ),
                        )
                elif kind == "reject":
                    log_attempt(
                        job.job_id,
                        rejected=_reject_reason_name(reply.reject.reason),
                        wall_ms=wall_ms,
                    )
                if kind == "job_request" and gate is not None:
                    # A finished job refills its own credit to keep the pipeline
                    # full mid-qblock. Dropping the refill is what parks the
                    # credits; the next grant waits for a qblock boundary.
                    result = gate.on_job(time.time())
                    if not result.allowed:
                        if result.changed:
                            _log_line_crossed(result.decision)
                        continue
            out_q.put(reply)

    exit_code = EXIT_CLEAN

    # tonic (Rust) rejects UDS streams whose :authority is the socket path;
    # pin a conventional authority so grpc-python ↔ tonic interop works.
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.default_authority", "localhost"),
            ("grpc.enable_http_proxy", 0),
        ],
    )
    try:
        # Wait briefly for the coordinator's listener (race with process spawn).
        deadline = time.time() + 10.0
        while True:
            try:
                grpc.channel_ready_future(channel).result(
                    timeout=max(0.05, deadline - time.time())
                )
                break
            except grpc.FutureTimeoutError:
                if time.time() >= deadline:
                    logger.error("timed out waiting for coordinator at %s", target)
                    return EXIT_INTERNAL_FATAL

        stub = miner_pb2_grpc.MinerServiceStub(channel)
        responses = stub.Session(request_iter())

        last_activity = time.monotonic()
        for cm in responses:
            last_activity = time.monotonic()
            which = cm.WhichOneof("msg")
            if which == "welcome":
                ver = cm.welcome.protocol_version
                if ver not in (0, 1):
                    logger.error("bad Welcome protocol_version=%s", ver)
                    out_q.put(
                        miner_pb2.MinerMsg(
                            fatal=miner_pb2.Fatal(
                                exit_code=EXIT_INTERNAL_FATAL,
                                reason="bad welcome protocol_version",
                                restart_required=False,
                            )
                        )
                    )
                    exit_code = EXIT_INTERNAL_FATAL
                    break
            elif which == "configure":
                config = session_sdk.session_config_from_configure(
                    miner_id, cm.configure
                )
                # The coordinator has engaged us: connect to the QPU now (a
                # no-op in mock mode / when already connected).
                sampler.ensure_connected()
                # Uniform config handling: warn on any key the dwave schema
                # doesn't recognize before consuming the ones it does.
                warn_unknown_backend_keys(cm.configure.backend_toml)
                # Session-wide sweep budget: a top-level num_sweeps key, or
                # the SDK default. Echoed per Result in SamplerMeta.sweeps.
                session_sweeps = num_sweeps_from_toml(cm.configure.backend_toml)
                # Operator-set sampling defaults for jobs the coordinator left
                # blank. Lowest rung of the precedence ladder in job.py.
                session_defaults = sampling_defaults_from_toml(
                    cm.configure.backend_toml
                )
                if session_defaults != SamplingDefaults():
                    logger.info(
                        "[budget] sampling defaults from config: num_reads=%s "
                        "anneal_time_us=%s (per-job and SetTarget values still win)",
                        session_defaults.num_reads or "unset",
                        session_defaults.anneal_time_us or "unset",
                    )
                if pending_budget is None and cm.configure.backend_toml:
                    try:
                        pending_budget = budget_from_backend_toml(
                            cm.configure.backend_toml
                        )
                    except BudgetUnavailable as exc:
                        # No durable meter, no mining: an unmetered miner burns
                        # the period's quota in a day. Name the fix and stop.
                        logger.error("%s", exc)
                        out_q.put(
                            miner_pb2.MinerMsg(
                                fatal=miner_pb2.Fatal(
                                    exit_code=EXIT_CONFIG_INVALID,
                                    reason=str(exc),
                                    restart_required=False,
                                )
                            )
                        )
                        exit_code = EXIT_CONFIG_INVALID
                        break
                    if pending_budget is not None:
                        gate = ParticipationGate(pending_budget)
                out_q.put(miner_pb2.MinerMsg(ready=miner_pb2.Ready()))
                # Read straight off the wire, not off SessionConfig: the SDK
                # substitutes its own default of 3 for an unset field, which
                # would hide the "coordinator said nothing" case this backend
                # wants to answer with its own measured depth.
                depth = resolve_queue_depth(
                    coordinator=cm.configure.queue_depth,
                    configured=queue_depth_from_toml(cm.configure.backend_toml),
                )
                queue_depth = depth
                logger.info(
                    "[QPU] pipeline depth %d (coordinator asked for %s)",
                    depth,
                    cm.configure.queue_depth or "nothing",
                )
                if job_pool is None:
                    job_pool = ThreadPoolExecutor(
                        max_workers=max(1, depth), thread_name_prefix="dwave-job"
                    )
                # Local binding so the None check narrows: pending_budget is
                # captured by process_job, which blocks narrowing on it.
                pacer = pending_budget
                if pacer is None:
                    out_q.put(
                        miner_pb2.MinerMsg(
                            job_request=miner_pb2.JobRequest(credits=depth)
                        )
                    )
                else:
                    # Ready says the session is established; credits say the QPU
                    # is participating. They are deliberately not the same
                    # message. A budgeted miner grants nothing until a qblock
                    # boundary gives it a whole round to decide about.
                    _log_budget_configured(pacer, time.time())
            elif which == "topology":
                topo = cm.topology
                session_nodes = list(topo.nodes)
                session_hash = bytes(topo.hash)
                if topo.HasField("edges"):
                    session_edges = list(zip(topo.edges.u, topo.edges.v))
                else:
                    session_edges = []
                sampler.set_session_topology(session_nodes, session_edges)
            elif which == "set_target":
                session_target = cm.set_target
            elif which == "job":
                with state_lock:
                    cancelled = _is_abandoned(cm.job.generation, cancel_watermark)
                if cancelled:
                    # Abandoned generation (coordinator reseeded): don't spend
                    # QPU access on it. Refund the credit so the coordinator
                    # keeps the pipeline full — mirrors the Rust miners' skip at
                    # dequeue. In-flight submissions are left to finish; the
                    # coordinator discards their stale-generation Results.
                    #
                    # The gate is deliberately not consulted: a skipped job bills
                    # no access time, so it must not be the thing that decides
                    # participation, and parking here would strand the refund.
                    log_attempt(cm.job.job_id, cancelled=True, wall_ms=0)
                    out_q.put(
                        miner_pb2.MinerMsg(
                            job_request=miner_pb2.JobRequest(credits=1)
                        )
                    )
                    continue
                with state_lock:
                    gate_result = None if gate is None else gate.on_job(time.time())
                    budget_ok = gate_result is None or gate_result.allowed
                if not budget_ok:
                    logger.debug(
                        "[quip-miner-%s] attempt %s: rejected OVERLOADED (budget)",
                        _LOG_BACKEND,
                        _short_job_id(cm.job.job_id),
                    )
                    if gate_result is not None and gate_result.changed:
                        _log_line_crossed(gate_result.decision)
                    out_q.put(
                        miner_pb2.MinerMsg(
                            reject=miner_pb2.Reject(
                                job_id=cm.job.job_id,
                                reason=miner_pb2.OVERLOADED,
                            )
                        )
                    )
                    # No replacement credit. The coordinator dispatches against
                    # credits alone, so refunding one here turns a shut gate into
                    # a reject/dispatch spin at memory speed — the credits come
                    # back at the next qblock boundary, if the line allows it.
                    continue
                # Submit for concurrent sampling; the pool bounds in-flight to
                # queue_depth (credits keep the coordinator dispatching that many).
                args = (
                    cm.job,
                    list(session_nodes),
                    list(session_edges),
                    session_hash,
                    session_target,
                    session_sweeps,
                    session_defaults,
                )
                # Tracked before the submit, not inside the worker: a Cancel
                # arriving while the job waits for a pool thread must still
                # reach it, and the sampler holds the note until it registers.
                inflight.add(cm.job.job_id, cm.job.generation)
                if job_pool is not None:
                    job_pool.submit(process_job, *args).add_done_callback(
                        _surface_pool_failure
                    )
                else:
                    process_job(*args)
            elif which == "cancel":
                # Raise the reseed watermark so every job at/below max_generation
                # is skipped at dequeue instead of sampled on the QPU. The Status
                # reply reports the real watermark, not just an ack flag, so the
                # coordinator can see the current cancel point outside a Cancel
                # round-trip too (Ping reports it the same way, below).
                with state_lock:
                    cancel_watermark = max(cancel_watermark, cm.cancel.max_generation)
                    done = jobs_done
                    watermark = cancel_watermark
                out_q.put(_status(miner_id, done, abandoned=watermark))
                # Every job of an abandoned generation still on the QPU is
                # access time buying a round the coordinator has thrown away.
                # Ask D-Wave to drop them; it refunds only the ones it has not
                # started annealing, which is what the tally measures.
                doomed = inflight.abandoned(watermark)
                if doomed:
                    live = sampler.cancel_inflight(doomed)
                    tally.requested(len(doomed))
                    logger.info(
                        "[QPU] cancel gen<=%d: asked D-Wave to drop %d in-flight "
                        "job(s), %d still running | session: %s",
                        watermark,
                        len(doomed),
                        live,
                        tally.summary(),
                    )
                # A reseed is the one qblock boundary the miner can see: it is
                # the only monotone round counter the coordinator sends, and it
                # arrives every round even while the QPU holds no credits. That
                # makes it the point at which participation is decided.
                if gate is not None:
                    with state_lock:
                        boundary = gate.on_qblock_boundary(
                            cm.cancel.max_generation, time.time()
                        )
                    if boundary is not None:
                        if not boundary.allowed:
                            # Every skipped round says so, not just the first:
                            # a run of silent boundaries is exactly what made
                            # the old blackout unreadable in the session log.
                            _log_qblock_sat_out(
                                cm.cancel.max_generation, boundary.decision
                            )
                        elif boundary.changed:
                            # Credits survive a reseed, so only the round that
                            # resumes mining needs a grant.
                            _log_qblock_joined(
                                cm.cancel.max_generation, boundary.decision
                            )
                            out_q.put(
                                miner_pb2.MinerMsg(
                                    job_request=miner_pb2.JobRequest(
                                        credits=queue_depth
                                    )
                                )
                            )
            elif which == "ping":
                with state_lock:
                    done = jobs_done
                    watermark = cancel_watermark
                out_q.put(_status(miner_id, done, abandoned=watermark))
            elif which == "get_capabilities":
                out_q.put(miner_pb2.MinerMsg(capabilities=capabilities_message()))
            elif which == "shutdown":
                grace_ms = cm.shutdown.grace_ms or 5000
                break

            # Soft idle timeout: if the coordinator stalls mid-session.
            idle_s = config.idle_timeout_s if config else 300
            if time.monotonic() - last_activity > idle_s:
                logger.info("idle timeout (%ss) — clean exit", idle_s)
                break

        # Drain in-flight submissions (they enqueue their Results) before the
        # end-of-outbound marker, bounded by the grace window.
        if job_pool is not None:
            job_pool.shutdown(wait=True)
        # Last sweep: a charge estimated after the final job's own drain would
        # otherwise die with the process and under-count the period.
        _bill_unobserved(sampler, pending_budget)
        # Signal end-of-outbound so the server can finish draining Results.
        out_q.put(_STOP)
        # Give the feeder thread a moment to flush (grace_ms).
        time.sleep(min(0.05, grace_ms / 1000.0))
        return exit_code
    except grpc.RpcError as exc:
        code = exc.code() if hasattr(exc, "code") else None
        if code == grpc.StatusCode.UNAUTHENTICATED:
            return EXIT_TOKEN_REJECTED
        logger.exception("rpc error: %s", exc)
        return EXIT_INTERNAL_FATAL
    except Exception:
        logger.exception("session failed")
        return EXIT_INTERNAL_FATAL
    finally:
        if job_pool is not None:
            job_pool.shutdown(wait=False, cancel_futures=True)
        try:
            out_q.put(_STOP)
        except Exception:  # noqa: BLE001
            pass
        channel.close()
        sampler.close()


def run_session_sync(
    coordinator_uri: str,
    miner_id: str,
    sampler: OceanSampler,
    *,
    budget: Optional[BudgetPacer] = None,
) -> int:
    """Sync entry (session is already synchronous)."""
    return run_session(coordinator_uri, miner_id, sampler, budget=budget)
