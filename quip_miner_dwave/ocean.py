"""Ocean SDK wrapper with v0.2 lessons and offline mock support.

Design points carried from ``QPU/dwave_sampler.py``:
- Thread-pooled async submits (GIL-bound encode/submit off the main path)
- SampleSet decode happens only after the cloud future completes (not on submit)
- Defect-qubit clamping before submit; reconstruction after decode
- Real ``device_access_time_us`` extracted from sampleset timing info

Offline mode (``QUIP_DWAVE_MOCK=1`` or an injected sampler) uses a dimod
sampler so unit/conformance tests never hit a real QPU.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from quip_miner_dwave.defects import (
    DefectInfo,
    prepare_problem,
    reconstruct_samples,
)
from quip_miner_dwave.topology import native_topology_hash

logger = logging.getLogger(__name__)


# Ocean's canonical token variable, and the name the v0.2 stack shipped. The
# quip-node-manager compose files still deliver the Leap token as
# DWAVE_API_KEY, which Ocean never reads: DWaveSampler then raises "API token
# not defined" while a perfectly good token sits in the environment. Promote
# the old name at startup so an upgraded node keeps mining.
OCEAN_TOKEN_ENV = "DWAVE_API_TOKEN"
LEGACY_TOKEN_ENV = "DWAVE_API_KEY"


def adopt_legacy_token_env() -> bool:
    """Copy ``DWAVE_API_KEY`` to ``DWAVE_API_TOKEN`` when only the old name is set.

    Returns True when the promotion happened. ``DWAVE_API_TOKEN`` always wins:
    a value under the canonical name is never overwritten.
    """
    legacy = os.environ.get(LEGACY_TOKEN_ENV, "").strip()
    if not legacy or os.environ.get(OCEAN_TOKEN_ENV, "").strip():
        return False
    os.environ[OCEAN_TOKEN_ENV] = legacy
    logger.warning(
        "%s is deprecated and will be dropped; using its value as %s. "
        "Set %s instead.",
        LEGACY_TOKEN_ENV,
        OCEAN_TOKEN_ENV,
        OCEAN_TOKEN_ENV,
    )
    return True


def credentials_present() -> bool:
    """True if a D-Wave API token is available via env or SDK config file."""
    if os.environ.get(OCEAN_TOKEN_ENV):
        return True
    conf = os.path.expanduser("~/.config/dwave/dwave.conf")
    if os.path.isfile(conf):
        try:
            with open(conf, encoding="utf-8") as f:
                text = f.read()
            if "token" in text.lower() and "=" in text:
                return True
        except OSError:
            pass
    return False


def mock_mode_enabled() -> bool:
    return os.environ.get("QUIP_DWAVE_MOCK", "").strip() in ("1", "true", "yes")


def mock_backend() -> str:
    """Offline mock sampler backend: ``exact`` (default, brute-force — only
    tiny problems) or ``sa`` (SimulatedAnnealingSampler — scales to realistic
    topologies). Selected via ``QUIP_DWAVE_MOCK_BACKEND``."""
    b = os.environ.get("QUIP_DWAVE_MOCK_BACKEND", "").strip().lower()
    return "sa" if b == "sa" else "exact"


def ocean_importable() -> bool:
    try:
        import dimod  # noqa: F401
        import dwave  # noqa: F401

        return True
    except ImportError:
        return False


def qpu_access_time_us(sampleset: Any) -> int:
    """Sum qpu_programming_time + qpu_sampling_time (µs); 0 if missing."""
    info = getattr(sampleset, "info", None) or {}
    timing = info.get("timing") or {}
    prog = timing.get("qpu_programming_time") or 0
    sample = timing.get("qpu_sampling_time") or 0
    return int(prog) + int(sample)


@dataclass
class SampleResult:
    """One decoded sample batch from the QPU/mock.

    Spins stay in the array the sampler produced. Materialising a dict per
    read, keyed by qubit label, cost ~220k Python-level operations per job and
    bought nothing: the only consumer wants a dense vector in session node
    order, which is one vectorised reorder away from this form.

    ``spins`` is ``(reads, len(variables))`` of int8, normalised to +1/-1, and
    ``variables[i]`` is the qubit label of column ``i``.
    """

    spins: "np.ndarray"
    variables: List[int]
    energies: List[float]
    device_access_time_us: int
    num_reads: int
    defect_info: Optional[DefectInfo] = None
    extra: Dict[str, str] = field(default_factory=dict)


class SupportsSample(Protocol):
    """The one call :func:`quip_miner_dwave.job.handle_job` makes on a sampler.

    Naming the surface instead of the concrete class is what lets the tests
    drive job.py with a recording or exploding double, which is the only way to
    cover the failure paths without a live QPU.
    """

    def sample(
        self,
        h: Dict[int, float],
        j: Dict[Tuple[int, int], float],
        *,
        num_reads: int = 1,
        anneal_time_us: Optional[int] = None,
        nonce_seed: Optional[bytes] = None,
        label: str = "quip-dwave-qa",
        cancel_key: Optional[bytes] = None,
    ) -> SampleResult: ...


class SupportsClose(Protocol):
    """The surface the SIGTERM handler needs: release the cloud client."""

    def close(self) -> None: ...


class MockSampler:
    """Offline sampler backed by dimod ExactSolver / SimulatedAnnealingSampler.

    Returns a sampleset-like object with a synthetic timing dict so
    ``device_access_time_us`` is non-zero in tests.
    """

    def __init__(self, backend: str = "exact"):
        import dimod

        self._backend = backend
        if backend == "sa":
            self._sampler = dimod.SimulatedAnnealingSampler()
        else:
            self._sampler = dimod.ExactSolver()
        self.nodelist: List[int] = []
        self.edgelist: List[Tuple[int, int]] = []
        self.properties: Dict[str, Any] = {
            "chip_id": "mock-qpu",
            "num_qubits": 0,
        }

    def sample_ising(self, h, j, **kwargs):
        import dimod

        num_reads = int(kwargs.get("num_reads") or 1)
        t0 = time.monotonic()
        if self._backend == "exact":
            ss = self._sampler.sample_ising(h, j)
            # ExactSolver returns every state; keep the lowest-energy rows.
            if len(ss) > num_reads:
                ss = ss.truncate(num_reads)
        else:
            ss = self._sampler.sample_ising(h, j, num_reads=num_reads)
        elapsed_us = max(1, int((time.monotonic() - t0) * 1_000_000))
        # Attach synthetic QPU timing so SamplerMeta.device_access_time_us is real-ish.
        # SampleSet.info is read-only; rebuild with the same samples + new info.
        info = dict(getattr(ss, "info", None) or {})
        info["timing"] = {
            "qpu_programming_time": 100,
            "qpu_sampling_time": elapsed_us,
        }
        return dimod.SampleSet(ss.record, ss.variables, info, ss.vartype)


class OceanSampler:
    """Thin wrapper around a D-Wave (or mock) sampler.

    Callers always pass logical Ising problems. Defect clamping and sample
    reconstruction are handled here; energy scoring for consensus stays in
    the session layer via ``quip_solver_core.scoring``.
    """

    def __init__(
        self,
        *,
        sampler: Any = None,
        solver_name: Optional[str] = None,
        region: Optional[str] = None,
        token: Optional[str] = None,
        mock: Optional[bool] = None,
        submit_workers: int = 4,
        defective_qubits: Optional[Sequence[int]] = None,
        defective_edges: Optional[set] = None,
    ):
        self._submit_pool = ThreadPoolExecutor(
            max_workers=max(1, submit_workers),
            thread_name_prefix="dwave-submit",
        )
        # Live cloud problems, keyed by the caller's cancel key, so a
        # coordinator Cancel can reach a problem still sitting on the QPU.
        # Crossed by the submit pool and the session thread -> guard it.
        self._inflight: Dict[bytes, Any] = {}
        # Keys cancelled before their submit landed. The cloud Future only
        # exists once _submit_sync has run and SAPI has accepted the problem;
        # a Cancel inside that window has nothing to call yet, and that window
        # is exactly when cancelling saves the most access time.
        self._cancel_pending: set = set()
        self._inflight_lock = threading.Lock()
        # Access time D-Wave charged for problems whose timing we never saw.
        # See _charge_unobserved for why this exists and why it over-counts.
        self._unobserved_access_us = 0
        self._max_access_us = 0
        self._defective_qubits: List[int] = list(defective_qubits or [])
        self._defective_edges: set = set(defective_edges or set())
        self._live_nodes: List[int] = []
        self._live_edges: List[Tuple[int, int]] = []
        self._native_hash: Optional[bytes] = None
        self._is_mock = False
        self._qpu_solver = None
        self._connected = False
        # Connection overrides; unset values fall through to D-Wave's native
        # config resolution (dwave.conf + standard env) in `_connect_real`.
        self._solver_name = solver_name
        self._region = region
        self._token = token

        use_mock = mock if mock is not None else mock_mode_enabled()
        if sampler is not None:
            self.sampler = sampler
            self._is_mock = use_mock or isinstance(sampler, MockSampler)
            self._connected = True
            self._apply_native_hash()
        elif use_mock:
            self.sampler = MockSampler(backend=mock_backend())
            self._is_mock = True
            self._connected = True
            logger.info(
                "[QPU] mock sampler active (QUIP_DWAVE_MOCK, backend=%s)",
                mock_backend(),
            )
            self._apply_native_hash()
        else:
            # Real QPU: do NOT contact D-Wave here. The connection is deferred
            # until the coordinator engages us (Configure calls ensure_connected)
            # or another mode forces it (--check), so an idle or unconnected
            # miner never opens a QPU session.
            self.sampler = None
            logger.info("[QPU] real sampler deferred until Configure")

    def ensure_connected(self) -> None:
        """Connect to the real QPU if not already (idempotent).

        Mock and injected samplers are ready at construction; a real sampler
        connects here — invoked when the coordinator sends Configure, or eagerly
        by --check. Safe to call repeatedly.
        """
        if self._connected:
            return
        self.sampler = self._connect_real(self._solver_name, self._region, self._token)
        self._detect_defects()
        self._connected = True
        self._apply_native_hash()

    def _apply_native_hash(self) -> None:
        if self._live_nodes and self._live_edges is not None:
            self._native_hash = native_topology_hash(self._live_nodes, self._live_edges)

    def _connect_real(self, solver_name, region, token):
        from dwave.system import DWaveSampler

        # Comply with D-Wave's own config: DWaveSampler resolves credentials
        # from ~/.config/dwave/dwave.conf and the canonical DWAVE_API_TOKEN /
        # DWAVE_API_SOLVER / DWAVE_API_REGION / DWAVE_API_ENDPOINT env vars. Pass
        # only explicit overrides a caller supplied; everything else is the SDK's.
        kwargs: Dict[str, Any] = {"request_timeout": (60, 300)}
        if solver_name is not None:
            kwargs["solver"] = solver_name
        if region is not None:
            kwargs["region"] = region
        if token is not None:
            kwargs["token"] = token
        base = DWaveSampler(**kwargs)
        self._qpu_solver = base
        self._live_nodes = sorted(int(n) for n in base.nodelist)
        self._live_edges = [(int(u), int(v)) for u, v in base.edgelist]
        logger.info(
            "[QPU] connected solver=%s qubits=%d",
            base.properties.get("chip_id", "?"),
            len(self._live_nodes),
        )
        return base

    def _detect_defects(self) -> None:
        # Real QPU: live graph is authoritative; no stored-topology diff here.
        # Defects are only injected when the session Topology lists qubits that
        # the live solver does not expose (handled per-job against Topology).
        self._defective_qubits = []
        self._defective_edges = set()

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def native_topology_hash(self) -> Optional[bytes]:
        return self._native_hash

    @property
    def live_nodes(self) -> List[int]:
        return list(self._live_nodes)

    @property
    def live_edges(self) -> List[Tuple[int, int]]:
        return list(self._live_edges)

    def set_session_topology(
        self,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
    ) -> None:
        """Bind the session Topology; compute defects vs live QPU if known."""
        nodes_l = [int(n) for n in nodes]
        edges_l = [(int(u), int(v)) for u, v in edges]
        if self._live_nodes:
            live_set = set(self._live_nodes)
            self._defective_qubits = sorted(set(nodes_l) - live_set)
            live_edge_set = {(min(u, v), max(u, v)) for u, v in self._live_edges}
            def_q = set(self._defective_qubits)
            self._defective_edges = set()
            for u, v in edges_l:
                if u in def_q or v in def_q:
                    continue
                key = (min(u, v), max(u, v))
                if key not in live_edge_set:
                    self._defective_edges.add((u, v))
            self._log_topology_fit(len(nodes_l), len(edges_l))
        else:
            # Mock / no live hardware: treat session topology as native.
            self._live_nodes = nodes_l
            self._live_edges = edges_l
            self._defective_qubits = []
            self._defective_edges = set()
            self._native_hash = native_topology_hash(nodes_l, edges_l)

    def _log_topology_fit(self, session_nodes: int, session_edges: int) -> None:
        """Report how much of the session topology this solver can anneal.

        Qubits the live chip lacks are clamped to pseudorandom spins and the
        couplers it lacks are scored after the fact (see ``defects``), so a
        solver whose working graph is not the network's returns energies that
        are valid, accepted, and a fraction of what the graph allows. Without
        this line that shows up only as results that never reach target.
        """
        missing_nodes = len(self._defective_qubits)
        missing_edges = len(self._defective_edges)
        if not missing_nodes and not missing_edges:
            logger.info(
                "[QPU] session topology matches the live graph "
                "(%d nodes, %d couplers)",
                session_nodes,
                session_edges,
            )
            return
        node_pct = 100.0 * missing_nodes / session_nodes if session_nodes else 0.0
        edge_pct = 100.0 * missing_edges / session_edges if session_edges else 0.0
        # A production chip is missing a handful of qubits. Missing a tenth of
        # the graph is a different chip, not hardware defects.
        log = logger.error if max(node_pct, edge_pct) > 10.0 else logger.warning
        log(
            "[QPU] live solver is missing %d/%d nodes (%.1f%%) and %d/%d "
            "couplers (%.1f%%) of the session topology; the missing part is "
            "clamped, not annealed, so energies land short of target. Check "
            "that DWAVE_API_SOLVER names the chip the network's topology "
            "came from.",
            missing_nodes,
            session_nodes,
            node_pct,
            missing_edges,
            session_edges,
            edge_pct,
        )

    def _register_inflight(self, key: bytes, future: Any) -> None:
        """Record a live cloud problem, or drop it if a Cancel beat it here."""
        with self._inflight_lock:
            if key in self._cancel_pending:
                # The note is consumed by the registration it applies to; a
                # key left poisoned would kill the next job that reuses it.
                self._cancel_pending.discard(key)
                doomed = future
            else:
                self._inflight[key] = future
                doomed = None
        if doomed is not None:
            doomed.cancel()

    def _release_inflight(self, key: bytes) -> None:
        """Forget a problem that has finished, cancelled or not."""
        with self._inflight_lock:
            self._inflight.pop(key, None)
            self._cancel_pending.discard(key)

    def cancel_inflight(self, keys: Sequence[bytes]) -> int:
        """Ask SAPI to drop these problems; return how many were still live.

        Best-effort by construction: D-Wave only refunds a problem it has not
        started annealing, so the return value counts the ones that had not
        finished when we asked, which is the ceiling on what was saved — not a
        confirmed refund. Keys whose submit has not landed are noted so
        :meth:`_register_inflight` cancels them the moment it does.
        """
        with self._inflight_lock:
            found = []
            for key in keys:
                future = self._inflight.pop(key, None)
                if future is None:
                    self._cancel_pending.add(key)
                else:
                    found.append(future)
        live = 0
        for future in found:
            if future.done():
                continue
            future.cancel()
            live += 1
        return live

    def _observe_access_us(self, access_us: int) -> None:
        """Note a measured charge, so an unmeasurable one can be estimated."""
        with self._inflight_lock:
            self._max_access_us = max(self._max_access_us, int(access_us))

    def _charge_unobserved(self) -> None:
        """Bill a problem SAPI accepted but whose timing we never saw.

        A cancel that beat the anneal costs nothing; a cancel that lost is
        charged by D-Wave in full. The raised path carries no timing to tell
        those apart, so the ledger assumes the anneal ran. That over-counts
        every successful cancel, which is the safe direction to be wrong:
        under-counting lets the pacer hand out headroom D-Wave has already
        spent, and the drift compounds across the period.

        The estimate is the largest charge measured this session, which is
        exact on a fleet whose jobs all carry the same num_reads. It is 0
        until the first job completes, so a cancel in the opening seconds of a
        session is under-billed; that window is bounded and does not recur.
        """
        with self._inflight_lock:
            self._unobserved_access_us += self._max_access_us

    def drain_unobserved_access_us(self) -> int:
        """Take the estimated charges accrued since the last call."""
        with self._inflight_lock:
            owed, self._unobserved_access_us = self._unobserved_access_us, 0
        return owed

    def close(self) -> None:
        self._submit_pool.shutdown(wait=False)
        if self._qpu_solver is not None:
            try:
                self._qpu_solver.client.close(wait=False)
            except Exception:  # noqa: BLE001 — best-effort
                pass

    def _submit_sync(
        self,
        h,
        j,
        num_reads: int,
        label: str,
        anneal_time_us: Optional[int] = None,
        cancel_key: Optional[bytes] = None,
    ):
        """Run on a pool thread: build/submit only; do NOT touch .sampleset."""
        kwargs: Dict[str, Any] = {
            "num_reads": num_reads,
            "label": label,
        }
        # D-Wave's SAPI parameter is `annealing_time`, in microseconds — the
        # same unit as the proto's `anneal_time_us`, so no conversion needed.
        # Only set when the caller supplied an explicit override; otherwise
        # leave it out so the QPU's hardware-default anneal applies.
        if anneal_time_us:
            kwargs["annealing_time"] = anneal_time_us
        # Prefer sample_ising; mock ExactSolver also supports it.
        sample_fn = getattr(self.sampler, "sample_ising", None)
        if not callable(sample_fn):
            raise RuntimeError("sampler has no sample_ising")
        # For real cloud futures the SDK returns a Future when async is used;
        # dimod/mock return a SampleSet. We normalize in _decode.
        if hasattr(self.sampler, "sample_ising") and not self._is_mock:
            # Async path via underlying solver when available.
            solver = getattr(self.sampler, "solver", None)
            if solver is not None and hasattr(solver, "sample_ising"):
                future = solver.sample_ising(h, j, **kwargs)
                # Registered here, on the pool thread, because this is the
                # first moment the cloud Future exists. Anything earlier has
                # no handle to cancel; anything later widens the blind window.
                if cancel_key is not None:
                    self._register_inflight(cancel_key, future)
                return future
        return sample_fn(h, j, **kwargs)

    @staticmethod
    def _decode_future(future_or_ss: Any):
        """Decode sampleset OFF the submit path (main/consumer thread)."""
        if hasattr(future_or_ss, "sampleset"):
            return future_or_ss.sampleset
        return future_or_ss

    def sample(
        self,
        h: Dict[int, float],
        j: Dict[Tuple[int, int], float],
        *,
        num_reads: int = 1,
        anneal_time_us: Optional[int] = None,
        nonce_seed: Optional[bytes] = None,
        label: str = "quip-dwave-qa",
        cancel_key: Optional[bytes] = None,
    ) -> SampleResult:
        """Submit one Ising problem and return decoded, reconstructed samples.

        Submit work runs on the thread pool; sampleset decode runs here after
        the future completes (v0.2 lesson: never decode on the submit path).
        ``anneal_time_us`` (microseconds) maps directly to D-Wave's
        ``annealing_time`` SAPI parameter; ``None``/``0`` leaves it unset so
        the QPU's hardware-default anneal applies.

        ``cancel_key`` makes the submission reachable by
        :meth:`cancel_inflight` until it finishes. A cancelled problem raises
        out of here rather than returning samples, which is what the caller
        wants: there is no Result to send for a generation the coordinator has
        already abandoned.
        """
        h_eff, j_eff, defect_info = prepare_problem(
            h,
            j,
            defective_qubits=self._defective_qubits,
            defective_edges=self._defective_edges,
            # Either kind of defect needs the reduction. Withholding the seed
            # when only couplers are missing skipped it entirely and sent the
            # QPU a graph it does not have.
            nonce_seed=(
                nonce_seed
                if (self._defective_qubits or self._defective_edges)
                else None
            ),
        )
        # Thread-pooled submit
        fut = self._submit_pool.submit(
            self._submit_sync,
            h_eff,
            j_eff,
            max(1, int(num_reads)),
            label,
            anneal_time_us,
            cancel_key,
        )
        # Two failure boundaries, billed differently. A failure here is the
        # submit itself dying in this process: D-Wave never saw the problem,
        # so charging for it would burn quota the QPU never spent.
        try:
            raw = fut.result()
        except BaseException:
            if cancel_key is not None:
                self._release_inflight(cancel_key)
            raise
        # Decode off the submit path. A failure here is a problem SAPI already
        # accepted — a cancelled one, most often — and it may have annealed.
        try:
            ss = self._decode_future(raw)
        except BaseException:
            self._charge_unobserved()
            raise
        finally:
            if cancel_key is not None:
                self._release_inflight(cancel_key)
        access_us = qpu_access_time_us(ss)
        self._observe_access_us(access_us)

        variables = [int(v) for v in ss.variables]
        # ExactSolver / SA use ±1; coerce zeros just in case. One vectorised
        # pass, not a dict comprehension per read.
        spins = np.where(np.asarray(ss.record.sample) >= 0, 1, -1).astype(np.int8)
        energies = [float(e) for e in ss.record.energy]
        spins, variables, energies = reconstruct_samples(
            spins, variables, energies, defect_info
        )

        # The cloud client aggregates identical reads into one record row
        # carrying num_occurrences, so the row count is distinct solutions, not
        # anneals performed. Sum the occurrences to report reads actually run;
        # the offline samplers do not aggregate, where the sum degrades to the
        # row count anyway.
        occurrences = getattr(ss.record, "num_occurrences", None)
        reads_done = (
            int(sum(occurrences)) if occurrences is not None else len(energies)
        )

        return SampleResult(
            spins=spins,
            variables=variables,
            energies=energies,
            device_access_time_us=access_us,
            num_reads=reads_done,
            defect_info=defect_info,
            extra={"mock": "1" if self._is_mock else "0"},
        )
