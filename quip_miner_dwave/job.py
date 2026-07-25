"""Job validation, sampling, and Result/Reject construction."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

from quip_proto import miner_pb2, wire

from quip_miner_dwave.ocean import OceanSampler, SampleResult

logger = logging.getLogger(__name__)


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def decode_milli_f64(raw: bytes) -> List[float]:
    """Decode little-endian i32 milli array to floats. Raises ValueError if bad."""
    return [v / 1000.0 for v in wire.decode_i32_le(raw)]


def edges_of(ising: miner_pb2.IsingProblem) -> List[Tuple[int, int]]:
    which = ising.WhichOneof("graph")
    if which == "edges":
        e = ising.edges
        return list(zip(e.u, e.v))
    return []


def resolve_graph(
    ising: miner_pb2.IsingProblem,
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    n_h: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return (nodes, edges) for a job.

    Inline EdgeList uses dense 0..n-1 node ids from ``n_h``; topology-hash
    jobs use the session-cached Topology.
    """
    which = ising.WhichOneof("graph")
    if which == "edges":
        return list(range(n_h)), edges_of(ising)
    return list(session_nodes), list(session_edges)


def spins_to_bytes(spins: Sequence[int]) -> bytes:
    return wire.encode_spins(list(spins))


def sample_dict_to_vector(
    sample: Dict[int, int], nodes: Sequence[int]
) -> List[int]:
    """Map a label-keyed sample into a dense spin vector in ``nodes`` order."""
    out: List[int] = []
    for n in nodes:
        s = sample.get(int(n), 1)
        out.append(1 if s >= 0 else -1)
    return out


class _Rejected(Exception):
    """Raised by a validation step to abort the job with a reject reason.

    Lets the guards read as straight-line code and keeps the conversion to a
    Reject message in one place, in :func:`handle_job`.
    """

    def __init__(self, reason: int):
        super().__init__(reason)
        self.reason = reason


def _reject(job_id: bytes, reason: int) -> List[miner_pb2.MinerMsg]:
    """Build the single-message reply that rejects ``job_id`` for ``reason``.

    A reject ends the job, so it is always the whole reply: no Result and no
    follow-up JobRequest. ``reason`` is a ``miner_pb2`` reject-reason enum.
    """
    return [
        miner_pb2.MinerMsg(reject=miner_pb2.Reject(job_id=job_id, reason=reason))
    ]


def _validate_job(
    job: miner_pb2.Job,
    session_hash: Optional[bytes],
) -> Tuple[miner_pb2.IsingProblem, List[float], List[float]]:
    """Check job-level invariants and decode the milli arrays.

    Args:
        job: The job to validate.
        session_hash: Hash of the cached session ``Topology``, or ``None`` if
            no ``Topology`` has arrived yet.

    Returns:
        The job's ``IsingProblem`` and its decoded ``h`` and ``j`` values.

    Raises:
        _Rejected: Unsupported kind, undecodable ``h``/``j``, a deadline
            already passed, or a ``topology_hash`` the session cannot satisfy.
    """
    if job.kind not in (
        miner_pb2.ISING_SAMPLE,
        miner_pb2.JOB_KIND_UNSPECIFIED,
    ):
        raise _Rejected(miner_pb2.UNSUPPORTED_KIND)

    ising = job.ising if job.HasField("ising") else miner_pb2.IsingProblem()

    # MALFORMED: h or j length not a multiple of 4
    try:
        h = decode_milli_f64(ising.h_milli_le32)
    except ValueError:
        raise _Rejected(miner_pb2.MALFORMED) from None
    try:
        j_vals = decode_milli_f64(ising.j_milli_le32) if ising.j_milli_le32 else []
    except ValueError:
        raise _Rejected(miner_pb2.MALFORMED) from None

    if job.deadline_ms and job.deadline_ms < now_unix_ms():
        raise _Rejected(miner_pb2.EXPIRED)

    if ising.WhichOneof("graph") == "topology_hash":
        if session_hash is None:
            raise _Rejected(miner_pb2.TOPOLOGY_MISSING)
        if bytes(ising.topology_hash) != bytes(session_hash):
            raise _Rejected(miner_pb2.TOPOLOGY_MISMATCH)

    return ising, h, j_vals


def _resolve_problem(
    ising: miner_pb2.IsingProblem,
    h: Sequence[float],
    j_vals: Sequence[float],
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    *,
    job_id: bytes,
) -> Tuple[List[int], Dict[int, float], Dict[Tuple[int, int], float]]:
    """Resolve the job's graph and build the sampler's ``h``/``J`` mappings.

    ``h_milli_le32`` and ``j_milli_le32`` are dense positional arrays over the
    resolved graph, so a length mismatch is a malformed job rather than
    something to trim to fit. Truncating would drop values the coordinator
    specified (or leave trailing nodes and edges untouched) and then report a
    credible energy for a different problem. A short ``j`` is not "the rest are
    zero" either: the wire format cannot express that, so it is a desync.

    Inline-EdgeList jobs derive their nodes from ``len(h)`` and so never trip
    the node check; it catches a job desynced from the session ``Topology``.

    Returns:
        The resolved node ordering and the ``h``/``J`` dicts keyed by qubit id.

    Raises:
        _Rejected: ``MALFORMED`` when ``h`` or ``j`` disagrees with the graph.
    """
    nodes, edges = resolve_graph(ising, session_nodes, session_edges, len(h))
    if not nodes and h:
        nodes = list(range(len(h)))

    if len(h) != len(nodes):
        logger.warning(
            "job %s: %d biases for a %d-node graph; rejecting MALFORMED",
            job_id.hex(),
            len(h),
            len(nodes),
        )
        raise _Rejected(miner_pb2.MALFORMED)

    if len(j_vals) != len(edges):
        logger.warning(
            "job %s: %d couplings for a %d-edge graph; rejecting MALFORMED",
            job_id.hex(),
            len(j_vals),
            len(edges),
        )
        raise _Rejected(miner_pb2.MALFORMED)

    h_dict = {int(nodes[i]): float(h[i]) for i in range(len(nodes))}
    j_dict: Dict[Tuple[int, int], float] = {
        (int(u), int(v)): float(j_vals[k]) for k, (u, v) in enumerate(edges)
    }
    return nodes, h_dict, j_dict


def _sampling_params(
    ising: miner_pb2.IsingProblem,
    session_target: Optional["miner_pb2.SetTarget"],
) -> Tuple[int, int]:
    """Resolve ``(num_reads, anneal_time_us)`` for one job.

    Both follow the same precedence: per-job override, then the session's
    ``SetTarget``, then the default. ``anneal_time_us`` defaults to 0, meaning
    the QPU applies its hardware-default anneal.

    (Full energy-based adapt for the QPU path is a follow-up; see quip-asx.* —
    it needs the shared GSE model and QPU credits.)
    """
    num_reads = int(ising.num_reads)
    if num_reads == 0 and session_target is not None and session_target.num_reads:
        num_reads = int(session_target.num_reads)
    if num_reads == 0:
        num_reads = 1

    anneal_time_us = int(ising.anneal_time_us)
    if (
        anneal_time_us == 0
        and session_target is not None
        and session_target.anneal_time_us
    ):
        anneal_time_us = int(session_target.anneal_time_us)

    return num_reads, anneal_time_us


def _build_result(
    job_id: bytes,
    nodes: Sequence[int],
    result: SampleResult,
) -> List[miner_pb2.MinerMsg]:
    """Turn a completed sample into a Result plus a follow-up JobRequest.

    Reports the sampler's own energy for the problem it actually annealed.
    Re-scoring here would buy nothing: the coordinator cannot trust a miner's
    energy regardless, so it re-scores whatever it accepts. ``energy_milli`` is
    an integer field, so the only transform is quantizing to milli.
    """
    solutions = []
    for sample, qpu_e in zip(result.samples, result.energies):
        spins = sample_dict_to_vector(sample, nodes if nodes else sorted(sample))
        solutions.append(
            miner_pb2.Solution(
                spins_bytes=spins_to_bytes(spins),
                energy_milli=int(round(qpu_e * 1000)),
            )
        )

    meta = miner_pb2.SamplerMeta(
        reads=result.num_reads,
        sweeps=0,
        device_access_time_us=result.device_access_time_us,
        qpu_access_us=result.device_access_time_us,
        extra=result.extra,
    )
    return [
        miner_pb2.MinerMsg(
            result=miner_pb2.Result(
                job_id=job_id,
                solutions=solutions,
                meta=meta,
            )
        ),
        miner_pb2.MinerMsg(job_request=miner_pb2.JobRequest(credits=1)),
    ]


def handle_job(
    job: miner_pb2.Job,
    sampler: OceanSampler,
    *,
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    session_hash: Optional[bytes] = None,
    session_target: Optional["miner_pb2.SetTarget"] = None,
) -> List[miner_pb2.MinerMsg]:
    """Validate and solve one job; return Result+JobRequest or Reject messages.

    Validation runs in two steps that raise :class:`_Rejected` rather than
    returning early, so every reject reason converges on one exit here.
    :func:`_validate_job` covers the job itself (kind, decodable h/j, deadline,
    topology hash) and :func:`_resolve_problem` covers agreement between the
    arrays and the resolved graph. ``session_hash`` is the hash of the cached
    ``Topology``, ``None`` until one arrives; a ``topology_hash`` job with no
    cache rejects ``TOPOLOGY_MISSING`` and a differing hash rejects
    ``TOPOLOGY_MISMATCH``, matching the Rust miners.
    """
    job_id = job.job_id
    try:
        ising, h, j_vals = _validate_job(job, session_hash)
        nodes, h_dict, j_dict = _resolve_problem(
            ising,
            h,
            j_vals,
            session_nodes,
            session_edges,
            job_id=job_id,
        )
    except _Rejected as exc:
        return _reject(job_id, exc.reason)

    num_reads, anneal_time_us = _sampling_params(ising, session_target)
    result: SampleResult = sampler.sample(
        h_dict,
        j_dict,
        num_reads=num_reads,
        # 0 leaves annealing_time unset so the QPU default applies.
        anneal_time_us=anneal_time_us or None,
        # Use job_id bytes as the defect-clamp seed when present.
        nonce_seed=bytes(job_id) if job_id else None,
        label=f"quip-{job_id.hex()[:8] if job_id else 'job'}",
    )
    return _build_result(job_id, nodes, result)
