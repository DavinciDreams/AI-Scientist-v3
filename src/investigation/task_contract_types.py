"""Immutable schema-v1 task-contract value types."""

from dataclasses import dataclass


MIN_COMPUTE_WALL_CLOCK_SECONDS = 120
DEFAULT_MAX_CANDIDATE_RPC_CALLS = 100_000
HARD_MAX_CANDIDATE_RPC_CALLS = 10_000_000


@dataclass(frozen=True)
class CanonicalTask:
    path: str
    sha256: str


@dataclass(frozen=True)
class Method:
    class_name: str
    description: str


@dataclass(frozen=True)
class Candidate:
    interface: str
    editable_paths: tuple[str, ...]


@dataclass(frozen=True)
class Evaluators:
    search: str
    final: str


@dataclass(frozen=True)
class ResourceRoots:
    search_root: str
    final_root: str
    public_artifact_bundle: str | None


@dataclass(frozen=True)
class DataRoles:
    search_roles: tuple[str, ...]
    final_roles: tuple[str, ...]


@dataclass(frozen=True)
class Metric:
    name: str
    data_role: str


@dataclass(frozen=True)
class Metrics:
    search: tuple[Metric, ...]
    final: tuple[Metric, ...]


@dataclass(frozen=True)
class Upstream:
    repository: str
    revision: str
    package_version: str | None
    license: str
    entrypoints: tuple[str, ...]
    patches: tuple[str, ...]


@dataclass(frozen=True)
class Dependency:
    name: str
    version: str


@dataclass(frozen=True)
class Artifact:
    role: str
    path: str
    sha256: str
    source: str


@dataclass(frozen=True)
class Provenance:
    dependencies: tuple[Dependency, ...]
    artifacts: tuple[Artifact, ...]


@dataclass(frozen=True)
class ComputePlane:
    accelerator: str
    accelerator_model: str
    gpu_count: int
    evaluator_gpu_count: int
    candidate_gpu_count: int
    cpu_cores: int
    memory_gb: int
    precision: str
    wall_clock_seconds: int
    max_candidate_rpc_calls: int
    execution_mode: str = "isolated_v1"


@dataclass(frozen=True)
class Compute:
    search: ComputePlane
    final: ComputePlane


@dataclass(frozen=True)
class SearchProxy:
    name: str
    preserves: str
    omits: str


@dataclass(frozen=True)
class Protocol:
    search_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    search_reset_sequence: tuple[int, ...]
    search_smoke_reset_sequence: tuple[int, ...]
    final_reset_sequence: tuple[int, ...]
    final_smoke_reset_sequence: tuple[int, ...]
    baselines: tuple[str, ...]
    promotion_rule: str
    search_proxies: tuple[SearchProxy, ...]


@dataclass(frozen=True)
class Feedback:
    search_visible: tuple[str, ...]
    final_sealed: tuple[str, ...]


@dataclass(frozen=True)
class Publication:
    search_is_proxy: bool
    requires_final_evaluation: bool
    required_evidence: tuple[str, ...]


@dataclass(frozen=True)
class TaskContract:
    schema_version: int
    canonical_task: CanonicalTask
    method: Method
    candidate: Candidate
    evaluators: Evaluators
    resources: ResourceRoots
    data: DataRoles
    metrics: Metrics
    upstream: tuple[Upstream, ...]
    provenance: Provenance
    compute: Compute
    protocol: Protocol
    feedback: Feedback
    publication: Publication
