from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_KIND_UNSPECIFIED: _ClassVar[JobKind]
    ISING_SAMPLE: _ClassVar[JobKind]
    GATE_CIRCUIT: _ClassVar[JobKind]

class RejectReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REJECT_REASON_UNSPECIFIED: _ClassVar[RejectReason]
    UNSUPPORTED_KIND: _ClassVar[RejectReason]
    TOO_LARGE: _ClassVar[RejectReason]
    EXPIRED: _ClassVar[RejectReason]
    OVERLOADED: _ClassVar[RejectReason]
    SHUTTING_DOWN: _ClassVar[RejectReason]
    MALFORMED: _ClassVar[RejectReason]
    TOPOLOGY_MISMATCH: _ClassVar[RejectReason]
    TOPOLOGY_MISSING: _ClassVar[RejectReason]
JOB_KIND_UNSPECIFIED: JobKind
ISING_SAMPLE: JobKind
GATE_CIRCUIT: JobKind
REJECT_REASON_UNSPECIFIED: RejectReason
UNSUPPORTED_KIND: RejectReason
TOO_LARGE: RejectReason
EXPIRED: RejectReason
OVERLOADED: RejectReason
SHUTTING_DOWN: RejectReason
MALFORMED: RejectReason
TOPOLOGY_MISMATCH: RejectReason
TOPOLOGY_MISSING: RejectReason

class MinerMsg(_message.Message):
    __slots__ = ("hello", "ready", "job_request", "result", "reject", "status", "fatal", "capabilities")
    HELLO_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    JOB_REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    REJECT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    FATAL_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    hello: Hello
    ready: Ready
    job_request: JobRequest
    result: Result
    reject: Reject
    status: Status
    fatal: Fatal
    capabilities: Capabilities
    def __init__(self, hello: _Optional[_Union[Hello, _Mapping]] = ..., ready: _Optional[_Union[Ready, _Mapping]] = ..., job_request: _Optional[_Union[JobRequest, _Mapping]] = ..., result: _Optional[_Union[Result, _Mapping]] = ..., reject: _Optional[_Union[Reject, _Mapping]] = ..., status: _Optional[_Union[Status, _Mapping]] = ..., fatal: _Optional[_Union[Fatal, _Mapping]] = ..., capabilities: _Optional[_Union[Capabilities, _Mapping]] = ...) -> None: ...

class CoordMsg(_message.Message):
    __slots__ = ("welcome", "configure", "topology", "job", "cancel", "ping", "shutdown", "set_target", "get_capabilities")
    WELCOME_FIELD_NUMBER: _ClassVar[int]
    CONFIGURE_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    PING_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_FIELD_NUMBER: _ClassVar[int]
    SET_TARGET_FIELD_NUMBER: _ClassVar[int]
    GET_CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    welcome: Welcome
    configure: Configure
    topology: Topology
    job: Job
    cancel: Cancel
    ping: Ping
    shutdown: Shutdown
    set_target: SetTarget
    get_capabilities: GetCapabilities
    def __init__(self, welcome: _Optional[_Union[Welcome, _Mapping]] = ..., configure: _Optional[_Union[Configure, _Mapping]] = ..., topology: _Optional[_Union[Topology, _Mapping]] = ..., job: _Optional[_Union[Job, _Mapping]] = ..., cancel: _Optional[_Union[Cancel, _Mapping]] = ..., ping: _Optional[_Union[Ping, _Mapping]] = ..., shutdown: _Optional[_Union[Shutdown, _Mapping]] = ..., set_target: _Optional[_Union[SetTarget, _Mapping]] = ..., get_capabilities: _Optional[_Union[GetCapabilities, _Mapping]] = ...) -> None: ...

class Hello(_message.Message):
    __slots__ = ("miner_id", "session_token", "protocol_version", "backend", "algorithm", "supported_kinds", "max_nodes", "max_edges", "native_topology_hash", "features")
    MINER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_KINDS_FIELD_NUMBER: _ClassVar[int]
    MAX_NODES_FIELD_NUMBER: _ClassVar[int]
    MAX_EDGES_FIELD_NUMBER: _ClassVar[int]
    NATIVE_TOPOLOGY_HASH_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    miner_id: str
    session_token: str
    protocol_version: int
    backend: str
    algorithm: str
    supported_kinds: _containers.RepeatedScalarFieldContainer[JobKind]
    max_nodes: int
    max_edges: int
    native_topology_hash: bytes
    features: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, miner_id: _Optional[str] = ..., session_token: _Optional[str] = ..., protocol_version: _Optional[int] = ..., backend: _Optional[str] = ..., algorithm: _Optional[str] = ..., supported_kinds: _Optional[_Iterable[_Union[JobKind, str]]] = ..., max_nodes: _Optional[int] = ..., max_edges: _Optional[int] = ..., native_topology_hash: _Optional[bytes] = ..., features: _Optional[_Iterable[str]] = ...) -> None: ...

class Welcome(_message.Message):
    __slots__ = ("protocol_version",)
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    def __init__(self, protocol_version: _Optional[int] = ...) -> None: ...

class Configure(_message.Message):
    __slots__ = ("queue_depth", "idle_timeout_s", "heartbeat_s", "reconnect_window_s", "backend_toml")
    QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    IDLE_TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_S_FIELD_NUMBER: _ClassVar[int]
    RECONNECT_WINDOW_S_FIELD_NUMBER: _ClassVar[int]
    BACKEND_TOML_FIELD_NUMBER: _ClassVar[int]
    queue_depth: int
    idle_timeout_s: int
    heartbeat_s: int
    reconnect_window_s: int
    backend_toml: str
    def __init__(self, queue_depth: _Optional[int] = ..., idle_timeout_s: _Optional[int] = ..., heartbeat_s: _Optional[int] = ..., reconnect_window_s: _Optional[int] = ..., backend_toml: _Optional[str] = ...) -> None: ...

class Ready(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Topology(_message.Message):
    __slots__ = ("hash", "nodes", "edges", "allowed_h_milli")
    HASH_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_H_MILLI_FIELD_NUMBER: _ClassVar[int]
    hash: bytes
    nodes: _containers.RepeatedScalarFieldContainer[int]
    edges: EdgeList
    allowed_h_milli: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, hash: _Optional[bytes] = ..., nodes: _Optional[_Iterable[int]] = ..., edges: _Optional[_Union[EdgeList, _Mapping]] = ..., allowed_h_milli: _Optional[_Iterable[int]] = ...) -> None: ...

class SetTarget(_message.Message):
    __slots__ = ("max_energy_milli", "min_solutions", "min_diversity_milli", "num_reads", "num_sweeps", "anneal_time_us")
    MAX_ENERGY_MILLI_FIELD_NUMBER: _ClassVar[int]
    MIN_SOLUTIONS_FIELD_NUMBER: _ClassVar[int]
    MIN_DIVERSITY_MILLI_FIELD_NUMBER: _ClassVar[int]
    NUM_READS_FIELD_NUMBER: _ClassVar[int]
    NUM_SWEEPS_FIELD_NUMBER: _ClassVar[int]
    ANNEAL_TIME_US_FIELD_NUMBER: _ClassVar[int]
    max_energy_milli: int
    min_solutions: int
    min_diversity_milli: int
    num_reads: int
    num_sweeps: int
    anneal_time_us: int
    def __init__(self, max_energy_milli: _Optional[int] = ..., min_solutions: _Optional[int] = ..., min_diversity_milli: _Optional[int] = ..., num_reads: _Optional[int] = ..., num_sweeps: _Optional[int] = ..., anneal_time_us: _Optional[int] = ...) -> None: ...

class EdgeList(_message.Message):
    __slots__ = ("u", "v")
    U_FIELD_NUMBER: _ClassVar[int]
    V_FIELD_NUMBER: _ClassVar[int]
    u: _containers.RepeatedScalarFieldContainer[int]
    v: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, u: _Optional[_Iterable[int]] = ..., v: _Optional[_Iterable[int]] = ...) -> None: ...

class IsingProblem(_message.Message):
    __slots__ = ("topology_hash", "edges", "h_milli_le32", "j_milli_le32", "num_reads", "num_sweeps", "anneal_time_us", "initial_spins", "start_beta_milli", "reversal_s_milli", "reversal_pause_us")
    TOPOLOGY_HASH_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    H_MILLI_LE32_FIELD_NUMBER: _ClassVar[int]
    J_MILLI_LE32_FIELD_NUMBER: _ClassVar[int]
    NUM_READS_FIELD_NUMBER: _ClassVar[int]
    NUM_SWEEPS_FIELD_NUMBER: _ClassVar[int]
    ANNEAL_TIME_US_FIELD_NUMBER: _ClassVar[int]
    INITIAL_SPINS_FIELD_NUMBER: _ClassVar[int]
    START_BETA_MILLI_FIELD_NUMBER: _ClassVar[int]
    REVERSAL_S_MILLI_FIELD_NUMBER: _ClassVar[int]
    REVERSAL_PAUSE_US_FIELD_NUMBER: _ClassVar[int]
    topology_hash: bytes
    edges: EdgeList
    h_milli_le32: bytes
    j_milli_le32: bytes
    num_reads: int
    num_sweeps: int
    anneal_time_us: int
    initial_spins: _containers.RepeatedScalarFieldContainer[bytes]
    start_beta_milli: int
    reversal_s_milli: int
    reversal_pause_us: int
    def __init__(self, topology_hash: _Optional[bytes] = ..., edges: _Optional[_Union[EdgeList, _Mapping]] = ..., h_milli_le32: _Optional[bytes] = ..., j_milli_le32: _Optional[bytes] = ..., num_reads: _Optional[int] = ..., num_sweeps: _Optional[int] = ..., anneal_time_us: _Optional[int] = ..., initial_spins: _Optional[_Iterable[bytes]] = ..., start_beta_milli: _Optional[int] = ..., reversal_s_milli: _Optional[int] = ..., reversal_pause_us: _Optional[int] = ...) -> None: ...

class Provenance(_message.Message):
    __slots__ = ("is_pow", "order_id")
    IS_POW_FIELD_NUMBER: _ClassVar[int]
    ORDER_ID_FIELD_NUMBER: _ClassVar[int]
    is_pow: bool
    order_id: bytes
    def __init__(self, is_pow: _Optional[bool] = ..., order_id: _Optional[bytes] = ...) -> None: ...

class Job(_message.Message):
    __slots__ = ("job_id", "kind", "generation", "deadline_ms", "ising", "provenance")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    ISING_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    job_id: bytes
    kind: JobKind
    generation: int
    deadline_ms: int
    ising: IsingProblem
    provenance: Provenance
    def __init__(self, job_id: _Optional[bytes] = ..., kind: _Optional[_Union[JobKind, str]] = ..., generation: _Optional[int] = ..., deadline_ms: _Optional[int] = ..., ising: _Optional[_Union[IsingProblem, _Mapping]] = ..., provenance: _Optional[_Union[Provenance, _Mapping]] = ...) -> None: ...

class JobRequest(_message.Message):
    __slots__ = ("credits",)
    CREDITS_FIELD_NUMBER: _ClassVar[int]
    credits: int
    def __init__(self, credits: _Optional[int] = ...) -> None: ...

class Cancel(_message.Message):
    __slots__ = ("max_generation",)
    MAX_GENERATION_FIELD_NUMBER: _ClassVar[int]
    max_generation: int
    def __init__(self, max_generation: _Optional[int] = ...) -> None: ...

class Ping(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Shutdown(_message.Message):
    __slots__ = ("grace_ms",)
    GRACE_MS_FIELD_NUMBER: _ClassVar[int]
    grace_ms: int
    def __init__(self, grace_ms: _Optional[int] = ...) -> None: ...

class Solution(_message.Message):
    __slots__ = ("spins_bytes", "energy_milli")
    SPINS_BYTES_FIELD_NUMBER: _ClassVar[int]
    ENERGY_MILLI_FIELD_NUMBER: _ClassVar[int]
    spins_bytes: bytes
    energy_milli: int
    def __init__(self, spins_bytes: _Optional[bytes] = ..., energy_milli: _Optional[int] = ...) -> None: ...

class SamplerMeta(_message.Message):
    __slots__ = ("reads", "sweeps", "device_access_time_us", "qpu_access_us", "extra")
    class ExtraEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    READS_FIELD_NUMBER: _ClassVar[int]
    SWEEPS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ACCESS_TIME_US_FIELD_NUMBER: _ClassVar[int]
    QPU_ACCESS_US_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    reads: int
    sweeps: int
    device_access_time_us: int
    qpu_access_us: int
    extra: _containers.ScalarMap[str, str]
    def __init__(self, reads: _Optional[int] = ..., sweeps: _Optional[int] = ..., device_access_time_us: _Optional[int] = ..., qpu_access_us: _Optional[int] = ..., extra: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Result(_message.Message):
    __slots__ = ("job_id", "solutions", "meta")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    SOLUTIONS_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    job_id: bytes
    solutions: _containers.RepeatedCompositeFieldContainer[Solution]
    meta: SamplerMeta
    def __init__(self, job_id: _Optional[bytes] = ..., solutions: _Optional[_Iterable[_Union[Solution, _Mapping]]] = ..., meta: _Optional[_Union[SamplerMeta, _Mapping]] = ...) -> None: ...

class Reject(_message.Message):
    __slots__ = ("job_id", "reason")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    job_id: bytes
    reason: RejectReason
    def __init__(self, job_id: _Optional[bytes] = ..., reason: _Optional[_Union[RejectReason, str]] = ...) -> None: ...

class Status(_message.Message):
    __slots__ = ("miner_id", "utilization", "jobs_done", "abandoned_generation", "sampler_stats")
    class SamplerStatsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MINER_ID_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    JOBS_DONE_FIELD_NUMBER: _ClassVar[int]
    ABANDONED_GENERATION_FIELD_NUMBER: _ClassVar[int]
    SAMPLER_STATS_FIELD_NUMBER: _ClassVar[int]
    miner_id: str
    utilization: float
    jobs_done: int
    abandoned_generation: int
    sampler_stats: _containers.ScalarMap[str, str]
    def __init__(self, miner_id: _Optional[str] = ..., utilization: _Optional[float] = ..., jobs_done: _Optional[int] = ..., abandoned_generation: _Optional[int] = ..., sampler_stats: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Fatal(_message.Message):
    __slots__ = ("exit_code", "reason", "restart_required")
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    RESTART_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    exit_code: int
    reason: str
    restart_required: bool
    def __init__(self, exit_code: _Optional[int] = ..., reason: _Optional[str] = ..., restart_required: _Optional[bool] = ...) -> None: ...

class GetCapabilities(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Capabilities(_message.Message):
    __slots__ = ("backend", "algorithm", "supported_kinds", "max_nodes", "max_edges", "features", "protocol_version", "stream_width", "native_topology_hash")
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_KINDS_FIELD_NUMBER: _ClassVar[int]
    MAX_NODES_FIELD_NUMBER: _ClassVar[int]
    MAX_EDGES_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    STREAM_WIDTH_FIELD_NUMBER: _ClassVar[int]
    NATIVE_TOPOLOGY_HASH_FIELD_NUMBER: _ClassVar[int]
    backend: str
    algorithm: str
    supported_kinds: _containers.RepeatedScalarFieldContainer[JobKind]
    max_nodes: int
    max_edges: int
    features: _containers.RepeatedScalarFieldContainer[str]
    protocol_version: int
    stream_width: int
    native_topology_hash: bytes
    def __init__(self, backend: _Optional[str] = ..., algorithm: _Optional[str] = ..., supported_kinds: _Optional[_Iterable[_Union[JobKind, str]]] = ..., max_nodes: _Optional[int] = ..., max_edges: _Optional[int] = ..., features: _Optional[_Iterable[str]] = ..., protocol_version: _Optional[int] = ..., stream_width: _Optional[int] = ..., native_topology_hash: _Optional[bytes] = ...) -> None: ...
