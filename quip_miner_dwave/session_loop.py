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
from typing import Iterator, Optional, Tuple

import grpc

from quip_solver_core import miner_pb2, miner_pb2_grpc, session as session_sdk

from quip_miner_dwave import (
    ALGORITHM,
    BACKEND,
    EXIT_CLEAN,
    EXIT_INTERNAL_FATAL,
    EXIT_TOKEN_REJECTED,
    FEATURES,
    MAX_EDGES,
    MAX_NODES,
)
from quip_miner_dwave.budget import (
    QPUTimeManager,
    budget_from_backend_toml,
    warn_unknown_backend_keys,
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


def _log_budget_closed(estimate) -> None:
    """One line when the QPU budget gate shuts, with the reopening estimate.

    The per-attempt rejects sit at debug on purpose: a shut gate rejects every
    job the coordinator already staged, and narrating each one at warning
    buries the session log without adding anything this line did not say.
    """
    if estimate is None:
        logger.info("[QPU] budget exhausted; parking credits until it recovers")
        return
    pool_s = estimate.pool_us / 1_000_000
    until = estimate.seconds_until_can_mine
    if until == float("inf"):
        logger.error(
            "[QPU] budget can never reach the mining threshold as configured "
            "(pool %.0fs): no job will ever be accepted. Check daily_budget, "
            "min_block_budget and budget_cap.",
            pool_s,
        )
        return
    logger.info(
        "[QPU] budget exhausted (pool %.0fs); parking credits, next window in %s",
        pool_s,
        _format_duration_ms(int(until * 1000)),
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
    budget: Optional[QPUTimeManager] = None,
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
    pending_budget = budget
    # Credits are parked while the QPU budget gate is shut: the coordinator only
    # dispatches against credits we grant, so withholding them is how a miner
    # applies backpressure. queue_depth is the grant to restore when the pool
    # reopens (the coordinator's liveness Ping drives that re-check).
    credits_parked = False
    queue_depth = 3
    # Pipeline: up to queue_depth QPU submissions in flight (overlaps cloud RTT).
    # jobs_done + pending_budget are shared with worker threads -> guard them.
    state_lock = threading.Lock()
    job_pool: Optional[ThreadPoolExecutor] = None
    session_start = time.monotonic()
    best_energy_milli: Optional[int] = None
    PROGRESS_LOG_INTERVAL = 10

    def process_job(job, s_nodes, s_edges, s_hash, s_target, s_sweeps):
        # Runs on a pool thread: sample (blocking on the QPU), then enqueue
        # replies. Shared-state mutations are guarded by state_lock.
        nonlocal jobs_done, best_energy_milli, credits_parked
        started = time.monotonic()
        replies = handle_job(
            job,
            sampler,
            session_nodes=s_nodes,
            session_edges=s_edges,
            session_hash=s_hash,
            session_target=s_target,
            session_sweeps=s_sweeps,
        )
        wall_ms = int((time.monotonic() - started) * 1000)
        for reply in replies:
            kind = reply.WhichOneof("msg")
            with state_lock:
                if kind == "result" and _is_abandoned(job.generation, cancel_watermark):
                    # SPEC section 5: no Result for an abandoned generation.
                    # The cancel landed while the QPU sampled this job; the
                    # coordinator reseeded past it and reclaims the credit
                    # itself, so the reply is dropped whole.
                    continue
                if kind == "result":
                    jobs_done += 1
                    meta = reply.result.meta
                    if pending_budget is not None and meta is not None:
                        pending_budget.record_access_time(meta.device_access_time_us)
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
                if kind == "job_request" and pending_budget is not None:
                    estimate = pending_budget.should_mine()
                    if not estimate.should_mine:
                        pending_budget.end_burst()
                        # Dropping the request is what parks the credit. Mark it
                        # so the Ping re-check knows to restore the full grant;
                        # without this the miner sits at zero credits for good.
                        if not credits_parked:
                            credits_parked = True
                            _log_budget_closed(estimate)
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
                if pending_budget is None and cm.configure.backend_toml:
                    pending_budget = budget_from_backend_toml(cm.configure.backend_toml)
                out_q.put(miner_pb2.MinerMsg(ready=miner_pb2.Ready()))
                depth = config.queue_depth if config else 3
                queue_depth = depth
                if job_pool is None:
                    job_pool = ThreadPoolExecutor(
                        max_workers=max(1, depth), thread_name_prefix="dwave-job"
                    )
                if pending_budget is None or pending_budget.should_mine().should_mine:
                    out_q.put(
                        miner_pb2.MinerMsg(
                            job_request=miner_pb2.JobRequest(credits=depth)
                        )
                    )
                else:
                    credits_parked = True
                    _log_budget_closed(pending_budget.should_mine())
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
                    budget_estimate = (
                        None if pending_budget is None else pending_budget.should_mine()
                    )
                    budget_ok = budget_estimate is None or budget_estimate.should_mine
                if cancelled:
                    # Abandoned generation (coordinator reseeded): don't spend
                    # QPU access on it. Refund the credit so the coordinator
                    # keeps the pipeline full — mirrors the Rust miners' skip at
                    # dequeue. In-flight submissions are left to finish; the
                    # coordinator discards their stale-generation Results.
                    log_attempt(cm.job.job_id, cancelled=True, wall_ms=0)
                    out_q.put(
                        miner_pb2.MinerMsg(
                            job_request=miner_pb2.JobRequest(credits=1)
                        )
                    )
                    continue
                if not budget_ok:
                    logger.debug(
                        "[quip-miner-%s] attempt %s: rejected OVERLOADED (budget)",
                        _LOG_BACKEND,
                        _short_job_id(cm.job.job_id),
                    )
                    with state_lock:
                        first_closure = not credits_parked
                        credits_parked = True
                    if first_closure:
                        _log_budget_closed(budget_estimate)
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
                    # a reject/dispatch spin at memory speed — the credit comes
                    # back when the pool reopens (the Ping re-check below).
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
                )
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

            # Parked credits come back the moment the pool reopens. Every
            # inbound message is a chance to notice, and the coordinator's
            # liveness Ping arrives every 15s even with nothing dispatched, so
            # a miner waiting on budget needs no timer of its own.
            if pending_budget is not None:
                with state_lock:
                    reopened = (
                        pending_budget.should_mine() if credits_parked else None
                    )
                    if reopened is not None and reopened.should_mine:
                        credits_parked = False
                    else:
                        reopened = None
                if reopened is not None:
                    logger.info(
                        "[QPU] budget available again (pool %.0fs); resuming at "
                        "%s credits",
                        reopened.pool_us / 1_000_000,
                        queue_depth,
                    )
                    out_q.put(
                        miner_pb2.MinerMsg(
                            job_request=miner_pb2.JobRequest(credits=queue_depth)
                        )
                    )

            # Soft idle timeout: if the coordinator stalls mid-session.
            idle_s = config.idle_timeout_s if config else 300
            if time.monotonic() - last_activity > idle_s:
                logger.info("idle timeout (%ss) — clean exit", idle_s)
                break

        # Drain in-flight submissions (they enqueue their Results) before the
        # end-of-outbound marker, bounded by the grace window.
        if job_pool is not None:
            job_pool.shutdown(wait=True)
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
    budget: Optional[QPUTimeManager] = None,
) -> int:
    """Sync entry (session is already synchronous)."""
    return run_session(coordinator_uri, miner_id, sampler, budget=budget)
