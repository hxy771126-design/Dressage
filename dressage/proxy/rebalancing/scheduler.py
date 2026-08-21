"""Proxy-side engine placement and turn-boundary rebalancing."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import socket
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Deque, Mapping
from urllib.parse import urlsplit

from ._batch_milp import (
    BatchProblem,
    BatchSolution,
    BatchSolverError,
    EngineBaseline,
    FeasibleEdge,
    SolverStatus,
    solve_batch_for_target_load,
    solve_batch_greedy,
    solve_batch_milp,
)
from .cache_hit_estimator import (
    CacheHitEstimator,
    CacheSource,
    ContextRecoveryEstimate,
    context_bucket,
    longest_common_prefix_length,
)
from .load_batch_trace import LoadBatchHistory, LoadBatchTrace
from .context_recovery_model import PerformanceHistory, percentile
from .model_cache_profile import ModelCacheProfile, canonical_fingerprint
from .ray_calibration import (
    MachineCalibrationConfig,
    RayTransferBenchmark,
    load_machine_calibration_config_from_env,
)
from .scheduler_state import (
    CompatibilityPoolStateMachine,
    EngineRebalancingConfig,
    PoolReadiness,
    SchedulerState,
)
from .snapshot_store import CalibrationSnapshotStore
from .transfer_calibrator import (
    Benchmark,
    CalibrationPlan,
    CalibrationState,
    ContextPathReadiness,
    TransferCalibrator,
)

logger = logging.getLogger(__name__)

_MIN_SGLANG_VERSION = (0, 5, 15, 1)


class _RouterUnavailableError(RuntimeError):
    """The SGLang Router control plane cannot be reached yet."""


def _sglang_version_key(value: str) -> tuple[int, int, int, int] | None:
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?", value)
    if match is None:
        return None
    major, minor, patch, post = match.groups()
    return int(major), int(minor), int(patch), int(post or 0)


def sglang_rebalancing_supported(value: str) -> bool:
    key = _sglang_version_key(value)
    return key is not None and key >= _MIN_SGLANG_VERSION


@dataclass(frozen=True)
class EngineDeploymentInfo:
    engine_id: str
    node_id: str
    worker_url: str
    model_id: str
    weight_version: str
    sglang_version: str
    tp_size: int
    pp_size: int
    dp_size: int
    kv_dtype: str
    state_dtype: str
    page_size: int
    swa_window_size: int | None
    mamba_backend: str | None
    mamba_track_interval: int | None
    hicache_args: tuple[tuple[str, str], ...]
    mooncake_args: tuple[tuple[str, str], ...]
    cache_fingerprint: str
    mooncake_protocol: str = ""
    shared_l3: bool = False
    host_staging: bool = False
    gpudirect: bool = False

    @classmethod
    def from_worker(
        cls,
        *,
        worker_url: str,
        server_info: Mapping[str, Any] | None,
        weight_version: str | None,
        model_id: str,
        fallback_sglang_version: str = "unknown",
    ) -> "EngineDeploymentInfo":
        info = dict(server_info or {})
        args = info.get("server_args") or info.get("args") or info
        if not isinstance(args, Mapping):
            args = {}
        extra = args.get("hicache_storage_backend_extra_config") or {}
        if isinstance(extra, str):
            try:
                parsed_extra = json.loads(extra)
            except (TypeError, ValueError):
                parsed_extra = None
            if isinstance(parsed_extra, Mapping):
                extra = parsed_extra
                mooncake_args = tuple(
                    sorted((str(k), str(v)) for k, v in extra.items())
                )
            else:
                mooncake_args = (("config", extra),)
        elif isinstance(extra, Mapping):
            mooncake_args = tuple(sorted((str(k), str(v)) for k, v in extra.items()))
        else:
            mooncake_args = ()
        enable_hicache = bool(args.get("enable_hierarchical_cache"))
        backend = str(args.get("hicache_storage_backend") or "").lower()
        shared_l3 = enable_hicache and backend == "mooncake"
        protocol = str(
            (extra.get("protocol") if isinstance(extra, Mapping) else None)
            or args.get("mooncake_protocol")
            or ""
        ).lower()
        device = str(
            (extra.get("device_name") if isinstance(extra, Mapping) else None)
            or args.get("mooncake_device")
            or ""
        ).lower()
        normalized_mooncake = dict(mooncake_args)
        if protocol:
            normalized_mooncake.setdefault("protocol", protocol)
        if device:
            normalized_mooncake.setdefault("device_name", device)
        mooncake_args = tuple(sorted(normalized_mooncake.items()))
        gpudirect = shared_l3 and bool(
            args.get("hicache_use_gpudirect")
            or (extra.get("use_gpudirect") if isinstance(extra, Mapping) else False)
            or (extra.get("gpudirect") if isinstance(extra, Mapping) else False)
        )
        host_staging = shared_l3 and not gpudirect
        mamba_backend = "/".join(
            str(value or "default")
            for value in (
                args.get("linear_attn_backend") or args.get("mamba_ssm_backend"),
                args.get("linear_attn_prefill_backend"),
                args.get("linear_attn_decode_backend"),
            )
        )
        deployment_payload = {
            "model_id": args.get("model_path") or model_id,
            "weight_version": weight_version or "unknown",
            "sglang_version": info.get("version") or fallback_sglang_version,
            "tp_size": args.get("tp_size", 1),
            "pp_size": args.get("pp_size", 1),
            "dp_size": args.get("dp_size", 1),
            "kv_dtype": args.get("kv_cache_dtype") or args.get("dtype") or "auto",
            "state_dtype": args.get("dtype") or "auto",
            "page_size": args.get("page_size") or 1,
            "swa_window_size": args.get("sliding_window_size"),
            "mamba_backend": mamba_backend,
            "mamba_track_interval": args.get("mamba_track_interval"),
            "hicache": {
                key: args.get(key)
                for key in (
                    "enable_hierarchical_cache",
                    "hicache_ratio",
                    "hicache_write_policy",
                    "hicache_mem_layout",
                    "hicache_storage_backend",
                )
            },
            "mooncake": dict(mooncake_args),
        }
        parsed = urlsplit(worker_url)
        node_id = parsed.hostname or worker_url
        return cls(
            engine_id=worker_url,
            node_id=node_id,
            worker_url=worker_url.rstrip("/"),
            model_id=str(args.get("model_path") or model_id),
            weight_version=str(weight_version or "unknown"),
            sglang_version=str(info.get("version") or fallback_sglang_version),
            tp_size=int(args.get("tp_size") or 1),
            pp_size=int(args.get("pp_size") or 1),
            dp_size=int(args.get("dp_size") or 1),
            kv_dtype=str(args.get("kv_cache_dtype") or args.get("dtype") or "auto"),
            state_dtype=str(args.get("dtype") or "auto"),
            page_size=int(args.get("page_size") or 1),
            swa_window_size=(
                None
                if args.get("sliding_window_size") is None
                else int(args["sliding_window_size"])
            ),
            mamba_backend=mamba_backend,
            mamba_track_interval=(
                None
                if args.get("mamba_track_interval") is None
                else int(args["mamba_track_interval"])
            ),
            hicache_args=tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in deployment_payload["hicache"].items()
                )
            ),
            mooncake_args=mooncake_args,
            cache_fingerprint=canonical_fingerprint(deployment_payload),
            mooncake_protocol=protocol,
            shared_l3=shared_l3,
            host_staging=host_staging,
            gpudirect=gpudirect,
        )


@dataclass
class EngineLoad:
    worker_url: str
    healthy: bool = True
    metrics_timestamp: float = 0.0
    running: int = 0
    queued: int = 0
    active_tokens: int = 0
    token_capacity: int = 0
    request_capacity: int = 0
    token_usage: float = 0.0
    waiting_uncached_tokens: int = 0
    gen_throughput: float = 0.0
    queue_waiting: int = 0
    queue_paused: int = 0
    queue_retracted: int = 0
    queue_grammar: int = 0
    live_queue_metrics_available: bool = False
    reserved_requests: int = 0
    reserved_tokens: int = 0
    reserved_prefill_tokens: int = 0

    def fresh(self, *, now: float, stale_seconds: float) -> bool:
        return (
            self.healthy
            and self.metrics_timestamp > 0
            and now - self.metrics_timestamp <= stale_seconds
        )

    def snapshot(self, *, now: float, stale_seconds: float) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics_fresh"] = self.fresh(now=now, stale_seconds=stale_seconds)
        payload["metrics_age_seconds"] = (
            None
            if self.metrics_timestamp <= 0
            else max(0.0, now - self.metrics_timestamp)
        )
        return payload


@dataclass
class SessionRoutingState:
    owner_worker_url: str | None = None
    pending_owner_worker_url: str | None = None
    fingerprint: str | None = None
    previous_committed_tokens: list[int] = field(default_factory=list)
    seen_engines: set[str] = field(default_factory=set)
    group_id: int | str | None = None
    group_size: int = 1
    task_key: str | None = None
    generated_tokens: int = 0
    default_step_max_tokens: int | None = None


@dataclass(frozen=True)
class StepGenerationBudget:
    step_max_tokens_source: str
    effective_step_max_tokens: int | None
    historical_step_tokens_p75: int | None
    group_remaining_tokens: int | None
    estimated_step_output_tokens: int | None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadScore:
    request_pressure: float
    token_pressure: float
    queue_pressure: float
    total: float


@dataclass(frozen=True)
class RoutingDecision:
    session_id: str
    source_worker_url: str | None
    target_worker_url: str | None
    cache_fingerprint: str | None
    state: SchedulerState
    reason: str
    source_context: ContextRecoveryEstimate | None = None
    target_context: ContextRecoveryEstimate | None = None
    source_queue_seconds: float | None = None
    target_queue_seconds: float | None = None
    source_queue_history_seconds: float | None = None
    source_queue_live_seconds: float | None = None
    target_queue_history_seconds: float | None = None
    target_queue_live_seconds: float | None = None
    stay_seconds: float | None = None
    move_seconds: float | None = None
    queue_risk_seconds: float = 0.0
    context_risk_seconds: float = 0.0
    decision_risk_seconds: float = 0.0
    source_decode_seconds: float | None = None
    target_decode_seconds: float | None = None
    source_base_load: LoadScore | None = None
    target_base_load: LoadScore | None = None
    source_projected_load: LoadScore | None = None
    target_projected_load: LoadScore | None = None
    load_improvement_ratio: float | None = None
    required_load_improvement_ratio: float | None = None
    step_max_tokens_source: str = "unavailable"
    effective_step_max_tokens: int | None = None
    historical_step_tokens_p75: int | None = None
    group_remaining_tokens: int | None = None
    estimated_step_output_tokens: int | None = None
    moved: bool = False

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        if self.source_context is not None:
            payload["source_context"]["cache_source"] = (
                self.source_context.cache_source.value
            )
        if self.target_context is not None:
            payload["target_context"]["cache_source"] = (
                self.target_context.cache_source.value
            )
        return payload


@dataclass(frozen=True)
class RoutingLease:
    decision: RoutingDecision
    worker_url: str | None
    reserved_tokens: int
    base_tokens: int
    started_monotonic: float
    projected_load_score: float | None = None
    context_tokens: int = 0
    expected_output_tokens: int = 0
    reserved_prefill_tokens: int = 0
    prefill_reservation_generation: int | None = None
    reservation_id: int | None = None
    batch_id: int | None = None


@dataclass
class _ReservationEntry:
    engine_url: str
    request_increment: int
    token_increment: int
    prefill_increment: int
    prefill_reservation_generation: int | None
    prefill_active: bool


@dataclass
class _PendingBatchStep:
    arrival_id: int
    session_id: str
    input_ids: tuple[int, ...]
    step_max_new_tokens: int | None
    context_remaining_tokens: int | None
    expected_version: str | None
    require_registered_context: bool
    future: asyncio.Future[RoutingLease]
    cancelled: bool = False
    lease: RoutingLease | None = None


@dataclass
class _OpenBatch:
    id: int
    started_monotonic: float
    steps: list[_PendingBatchStep] = field(default_factory=list)
    sealed_monotonic: float | None = None
    terminal_published: bool = False


@dataclass(frozen=True)
class _BatchFetchResult:
    url: str
    status: str
    duration_seconds: float
    row_count: int
    load: EngineLoad | None


@dataclass(frozen=True)
class _FrozenBatchStep:
    pending: _PendingBatchStep
    session_signature: tuple[Any, ...]
    source: str | None
    fingerprint: str | None
    edges: tuple[FeasibleEdge, ...]
    budgets: tuple[tuple[str, StepGenerationBudget], ...]
    base_tokens: tuple[tuple[str, int], ...]
    fixed_target: str | None = None
    failure: str | None = None


@dataclass(frozen=True)
class _FrozenBatchEngineTrace:
    url: str
    fetch_status: str
    fetch_duration_seconds: float
    health: bool
    version: str | None
    fingerprint: str | None
    row_count: int
    running: int | None
    active_tokens: int | None
    request_capacity: int | None
    token_capacity: int | None
    token_usage: float | None
    gen_throughput: float | None
    queued: int | None
    queue_pressure: float | None
    waiting_uncached_tokens: int | None
    live_ledger_requests: int
    live_ledger_tokens: int
    live_ledger_prefill: int
    base_requests: int | None
    base_tokens: int | None
    base_queue: int | None


@dataclass(frozen=True)
class _FrozenBatch:
    reservation_revision: int
    topology_signature: tuple[tuple[str, bool, str, str], ...]
    decision_engines: tuple[EngineBaseline, ...]
    engine_traces: tuple[_FrozenBatchEngineTrace, ...]
    steps: tuple[_FrozenBatchStep, ...]
    sticky_problem: BatchProblem | None
    optimized_problem: BatchProblem | None


@dataclass(frozen=True)
class _SolvedBatch:
    assignment: tuple[tuple[str, str], ...]
    sticky_greedy: BatchSolution | None
    sticky: BatchSolution | None
    optimized: BatchSolution | None
    adopted: str
    target_maximum_load: float | None
    improvement_ratio: float | None
    fallback_reason: str | None
    elapsed_seconds: float


@dataclass(frozen=True)
class _CompletionObservation:
    lease: RoutingLease
    engine_url: str
    fingerprint: str
    shared_l3: bool
    old_owner: str | None
    target_seen_before: bool
    running: int
    context_tokens: int
    response_meta: Mapping[str, Any]
    output_tokens: int
    elapsed_seconds: float


class GroupLengthEstimator:
    def __init__(self, *, history_size: int, min_task_samples: int = 32) -> None:
        self.min_task_samples = min_task_samples
        self._group: dict[int | str, Deque[int]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._task: dict[str, Deque[int]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def observe(
        self, *, group_id: int | str | None, task_key: str | None, final_length: int
    ) -> None:
        if group_id is not None:
            self._group[group_id].append(max(0, int(final_length)))
        if task_key:
            self._task[task_key].append(max(0, int(final_length)))

    def remaining(
        self,
        *,
        group_id: int | str | None,
        task_key: str | None,
        generated_tokens: int,
    ) -> int | None:
        samples: list[int] = []
        if group_id is not None and len(self._group[group_id]) >= 2:
            samples = list(self._group[group_id])
        elif task_key and len(self._task[task_key]) >= self.min_task_samples:
            samples = list(self._task[task_key])
        if not samples:
            return None
        final = percentile([float(value) for value in samples], 0.75)
        return max(0, int(math.ceil(final)) - max(0, int(generated_tokens)))


class StepLengthEstimator:
    """P75 successful output length for one model call.

    Output-length behavior belongs to the task/model, not to one Engine.  The
    per-task series therefore falls back to all tasks in the same compatible
    pool and max-token bucket when it is still sparse.
    """

    def __init__(self, *, history_size: int, min_samples: int) -> None:
        self.min_samples = min_samples
        self._task: dict[tuple[str, str, str], Deque[int]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._pool: dict[tuple[str, str], Deque[int]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    @staticmethod
    def _bucket(max_tokens: int | None) -> str:
        if max_tokens is None:
            return "unbounded"
        return context_bucket(max_tokens)

    def observe(
        self,
        *,
        fingerprint: str,
        task_key: str | None,
        max_tokens: int | None,
        output_tokens: int,
    ) -> None:
        bucket = self._bucket(max_tokens)
        value = max(0, int(output_tokens))
        if task_key:
            self._task[(fingerprint, task_key, bucket)].append(value)
        self._pool[(fingerprint, bucket)].append(value)

    def p75(
        self,
        *,
        fingerprint: str,
        task_key: str | None,
        max_tokens: int | None,
    ) -> int | None:
        bucket = self._bucket(max_tokens)
        samples: list[int] = []
        if task_key:
            samples = list(self._task[(fingerprint, task_key, bucket)])
        if len(samples) < self.min_samples:
            samples = list(self._pool[(fingerprint, bucket)])
        if len(samples) < self.min_samples:
            return None
        return max(0, int(math.ceil(percentile(list(map(float, samples)), 0.75))))

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_series": len(self._task),
            "pool_series": len(self._pool),
            "samples": sum(len(values) for values in self._pool.values()),
            "min_samples": self.min_samples,
        }


class EngineRebalancer:
    def __init__(
        self,
        client: Any,
        *,
        config: EngineRebalancingConfig,
        model_id: str,
        model_config: Mapping[str, Any] | Any | None = None,
        calibration_benchmark: Benchmark | None = None,
        calibration_snapshot_root: str | Path | None = None,
        calibration_snapshot_run_name: str = "dressage",
        calibration_snapshot_interval_requests: int = 128,
    ) -> None:
        self.client = client
        self.config = config
        self.model_id = model_id
        self.model_config = model_config
        self.calibration_benchmark = calibration_benchmark
        self.machine_calibration_config: MachineCalibrationConfig | None = (
            load_machine_calibration_config_from_env() if config.enabled else None
        )
        self.performance = PerformanceHistory(
            history_size=config.history_size,
            min_samples=config.min_samples,
        )
        self.cache_hits = CacheHitEstimator(
            history_size=config.history_size,
            min_samples=config.min_samples,
            cold_start_probability=config.cold_start_hit_probability,
        )
        self.calibrator = TransferCalibrator()
        self.group_lengths = GroupLengthEstimator(history_size=config.history_size)
        self.step_lengths = StepLengthEstimator(
            history_size=config.history_size,
            min_samples=config.min_samples,
        )
        self.deployments: dict[str, EngineDeploymentInfo] = {}
        self.loads: dict[str, EngineLoad] = {}
        self._load_generations: dict[str, int] = defaultdict(int)
        self._reservations: dict[int, _ReservationEntry] = {}
        self._next_reservation_id = 1
        self._reservation_revision = 0
        self.profiles: dict[str, ModelCacheProfile] = {}
        self.pools: dict[str, CompatibilityPoolStateMachine] = {}
        self.plans: dict[str, CalibrationPlan] = {}
        self.sessions: dict[str, SessionRoutingState] = {}
        self.excluded_engines: dict[str, str] = {}
        self._decisions: Deque[dict[str, Any]] = deque(maxlen=config.history_size)
        self._observations: Deque[dict[str, Any]] = deque(maxlen=config.history_size)
        self._runtime_restore_seconds: dict[tuple[str, str, str, str], Deque[float]] = (
            defaultdict(lambda: deque(maxlen=config.history_size))
        )
        self._runtime_restore_errors: dict[tuple[str, str, str, str], Deque[float]] = (
            defaultdict(lambda: deque(maxlen=config.history_size))
        )
        self._runtime_restore_throughputs: dict[
            tuple[str, str, str, str], Deque[float]
        ] = defaultdict(lambda: deque(maxlen=config.history_size))
        self._online_request_count = 0
        self._snapshot_store = (
            CalibrationSnapshotStore(
                root=calibration_snapshot_root,
                run_name=calibration_snapshot_run_name,
                interval_requests=calibration_snapshot_interval_requests,
            )
            if config.enabled and calibration_snapshot_root is not None
            else None
        )
        self._snapshot_tasks: set[asyncio.Task[None]] = set()
        self._observation_tasks: set[asyncio.Task[None]] = set()
        self._final_snapshot_written = False
        self._lock = asyncio.Lock()
        self._batch_lock = asyncio.Lock()
        self._batch_run_lock = asyncio.Lock()
        self._open_batch: _OpenBatch | None = None
        self._pending_acquires: dict[str, _PendingBatchStep] = {}
        self._next_batch_id = 1
        self._next_arrival_id = 1
        self._batch_runner_tasks: set[asyncio.Task[None]] = set()
        self._load_batch_history = LoadBatchHistory(config.history_size)
        self._refresh_lock = asyncio.Lock()
        self._poll_task: asyncio.Task | None = None
        self._calibration_task: asyncio.Task | None = None
        self._ray_calibration_backend: RayTransferBenchmark | None = None
        self._preflight_plan: CalibrationPlan | None = None
        self._preflight_node_ids: set[str] = set()
        self._preflight_node_addresses: set[str] = set()
        self._preflight_node_aliases: dict[str, str] = {}
        self._stopping = False
        self._deployment_checked_at: dict[str, float] = {}
        self._deployment_refresh_seconds = 10.0

    async def start(self) -> None:
        logger.info(
            "engine rebalancing startup: enabled=%s effective_config=%s",
            self.config.enabled,
            self.config.snapshot(),
        )
        if not self.config.enabled:
            self.calibrator.transition(
                CalibrationState.DISABLED, "engine_rebalancing_disabled"
            )
            return
        if self._snapshot_store is not None:
            logger.info(
                "engine rebalancing calibration snapshots: directory=%s "
                "interval_requests=%d",
                self._snapshot_store.directory,
                self._snapshot_store.interval_requests,
            )
        self.calibrator.transition(
            CalibrationState.WAITING_FOR_RAY,
            (
                "waiting_for_ray"
                if self.machine_calibration_config is not None
                else "deployment_config_unavailable"
            ),
        )
        self._calibration_task = asyncio.create_task(
            self._run_machine_preflight(),
            name="engine-rebalancing-machine-calibration",
        )
        # Router discovery deliberately starts only after machine calibration
        # has reached a terminal state and initial.json has been persisted.

    async def close(self) -> None:
        self._stopping = True
        async with self._batch_lock:
            runners = tuple(self._batch_runner_tasks)
            for task in runners:
                task.cancel()
            pending = tuple(self._pending_acquires.values())
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)
        for step in pending:
            if not step.future.done():
                step.future.set_exception(RuntimeError("engine rebalancer is closed"))
        if self._calibration_task is not None:
            self._calibration_task.cancel()
            try:
                await self._calibration_task
            except asyncio.CancelledError:
                pass
            self._calibration_task = None
        if self._ray_calibration_backend is not None:
            await self._ray_calibration_backend.close()
            self._ray_calibration_backend = None
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        await self._drain_observation_tasks()
        await self._drain_snapshot_tasks()
        if self._snapshot_store is not None and not self._final_snapshot_written:
            self._final_snapshot_written = True
            await self._persist_current_snapshot("final", self._online_request_count)

    def _capture_file_snapshot(self, kind: str) -> dict[str, Any] | None:
        try:
            return {
                "offline_calibration": self.calibration_snapshot(),
                "runtime_calibration": self._runtime_calibration_snapshot(),
            }
        except Exception:
            logger.warning(
                "failed to capture engine rebalancing %s calibration snapshot",
                kind,
                exc_info=True,
            )
            return None

    async def _persist_current_snapshot(
        self, kind: str, online_request_count: int
    ) -> None:
        if self._snapshot_store is None:
            return
        payload = self._capture_file_snapshot(kind)
        if payload is not None:
            await self._write_calibration_snapshot(
                kind,
                online_request_count,
                payload,
            )

    def _schedule_calibration_snapshot(
        self,
        kind: str,
        online_request_count: int,
        payload: Mapping[str, Any],
    ) -> None:
        if self._snapshot_store is None:
            return
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._write_calibration_snapshot(kind, online_request_count, payload),
                name=f"engine-rebalancing-{kind}-snapshot",
            )
        except Exception:
            logger.warning(
                "failed to schedule engine rebalancing %s calibration snapshot",
                kind,
                exc_info=True,
            )
            return
        self._snapshot_tasks.add(task)
        task.add_done_callback(self._snapshot_tasks.discard)

    async def _write_calibration_snapshot(
        self,
        kind: str,
        online_request_count: int,
        payload: Mapping[str, Any],
    ) -> None:
        store = self._snapshot_store
        if store is None:
            return
        try:
            path = await store.write(
                kind=kind,
                online_request_count=online_request_count,
                payload=payload,
            )
        except Exception:
            logger.warning(
                "failed to persist engine rebalancing %s calibration snapshot",
                kind,
                exc_info=True,
            )
        else:
            logger.info(
                "persisted engine rebalancing %s calibration snapshot: %s",
                kind,
                path,
            )

    async def _drain_snapshot_tasks(self) -> None:
        while self._snapshot_tasks:
            tasks = tuple(self._snapshot_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    def _schedule_completion_observation(
        self, observation: _CompletionObservation
    ) -> None:
        task = asyncio.create_task(
            self._run_completion_observation(observation),
            name="engine-rebalancing-completion-observation",
        )
        self._observation_tasks.add(task)
        task.add_done_callback(self._observation_tasks.discard)

    async def _run_completion_observation(
        self, observation: _CompletionObservation
    ) -> None:
        try:
            await self._record_completion_observation(observation)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "engine rebalancing completion observation failed",
                exc_info=True,
            )

    async def _drain_observation_tasks(self) -> None:
        while self._observation_tasks:
            tasks = tuple(self._observation_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    def _record_successful_online_request(self) -> None:
        self._online_request_count += 1
        store = self._snapshot_store
        if (
            store is not None
            and self._online_request_count % store.interval_requests == 0
        ):
            payload = self._capture_file_snapshot("periodic")
            if payload is None:
                return
            self._schedule_calibration_snapshot(
                "periodic",
                self._online_request_count,
                payload,
            )

    async def _run_machine_preflight(self) -> None:
        try:
            await self._run_machine_preflight_impl()
        finally:
            if self.calibrator.state in {
                CalibrationState.READY,
                CalibrationState.DEGRADED,
            }:
                await self._persist_current_snapshot(
                    "initial", self._online_request_count
                )
                if not self._stopping and self._poll_task is None:
                    self._poll_task = asyncio.create_task(
                        self._poll_loop(),
                        name="engine-rebalancing-load-poll",
                    )

    async def _run_machine_preflight_impl(self) -> None:
        config = self.machine_calibration_config
        if config is None:
            self.calibrator.transition(
                CalibrationState.DEGRADED,
                "deployment_config_unavailable; using full-prefill fallback",
            )
            return
        if not config.shared_l3:
            fingerprint = config.fingerprint([])
            self._preflight_plan = CalibrationPlan(
                fingerprint=fingerprint,
                tasks=(),
                skipped_links={"mooncake": "L3 disabled"},
            )
            self.calibrator.transition(
                CalibrationState.READY,
                "no transfer calibration required: L3 disabled",
                plan_fingerprint=fingerprint,
            )
            return
        model_config = self.model_config
        if model_config is None and config.model_config_path:
            try:
                from transformers import AutoConfig

                model_config = AutoConfig.from_pretrained(
                    config.model_config_path,
                    trust_remote_code=True,
                )
            except Exception:
                logger.warning(
                    "could not load calibration model profile from deployment config",
                    exc_info=True,
                )
        if model_config is None:
            self.calibrator.transition(
                CalibrationState.DEGRADED,
                "model cache profile unavailable; using full-prefill fallback",
            )
            return

        backend = RayTransferBenchmark(config)
        self._ray_calibration_backend = backend
        terminal_state: CalibrationState | None = None
        terminal_reason = ""
        try:
            discovered = await backend.connect()
            self._preflight_node_addresses = {
                str(item["address"]) for item in discovered
            }
            self._preflight_node_ids = {str(item["address"]) for item in discovered} | {
                str(item["node_id"]) for item in discovered
            }
            self._preflight_node_aliases = {
                alias: str(item["address"])
                for item in discovered
                for alias in (str(item["node_id"]), str(item["address"]))
            }
            for configured_node in config.nodes:
                configured = configured_node.node_id
                try:
                    resolved = socket.gethostbyname(configured)
                except OSError:
                    resolved = configured
                target = self._preflight_node_aliases.get(
                    configured
                ) or self._preflight_node_aliases.get(resolved)
                if target is None and len(self._preflight_node_addresses) == 1:
                    target = next(iter(self._preflight_node_addresses))
                if target is not None:
                    self._preflight_node_aliases[configured] = target
                    self._preflight_node_aliases[resolved] = target
            profile = ModelCacheProfile.from_model_config(
                model_config,
                deployment=config.model_deployment,
            )
            fingerprint = config.fingerprint(discovered)
            plan = CalibrationPlan.build(
                fingerprint=fingerprint,
                engine_deployments=backend.planned_engine_slots(),
                shared_l3=config.shared_l3,
                host_staging=config.host_staging,
                gpudirect=config.gpudirect,
                model_cache_profile=profile,
            )
            self._preflight_plan = plan
            self.calibrator.transition(
                CalibrationState.RUNNING,
                "executing machine calibration plan",
                plan_fingerprint=fingerprint,
            )
            await self.calibrator.execute(plan, backend)
            if self.calibrator.plan_complete(plan):
                terminal_state = CalibrationState.READY
                terminal_reason = "machine calibration plan complete"
            else:
                terminal_state = CalibrationState.DEGRADED
                terminal_reason = (
                    "machine calibration incomplete; using full-prefill fallback"
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("machine transfer calibration failed", exc_info=True)
            terminal_state = CalibrationState.DEGRADED
            terminal_reason = (
                f"machine calibration failed: {exc!r}; using full-prefill fallback"
            )
        finally:
            resources_recovered: bool | None = None
            try:
                resources_recovered = await backend.close()
            except Exception as exc:  # pragma: no cover - Ray/CUDA specific
                logger.warning("machine calibration cleanup failed", exc_info=True)
                if terminal_state is not None:
                    terminal_state = CalibrationState.DEGRADED
                    terminal_reason = (
                        f"machine calibration cleanup failed: {exc!r}; "
                        "using full-prefill fallback"
                    )
            self._ray_calibration_backend = None
            if terminal_state is not None:
                if resources_recovered is False:
                    terminal_state = CalibrationState.DEGRADED
                    terminal_reason = (
                        "Ray GPU resources did not recover after calibration; "
                        "using full-prefill fallback"
                    )
                # Publish READY/DEGRADED only after every actor has been closed
                # and Ray has confirmed that its GPU resources are available
                # again.  The launch script uses this transition as the gate
                # before submitting the rollout job.
                self.calibrator.transition(
                    terminal_state,
                    terminal_reason,
                    plan_fingerprint=(
                        None
                        if self._preflight_plan is None
                        else self._preflight_plan.fingerprint
                    ),
                )

    async def _poll_loop(self) -> None:
        base_delay = self.config.load_poll_interval_ms / 1000.0
        startup_backoff = (
            base_delay,
            max(base_delay, 1.0),
            max(base_delay, 2.0),
            max(base_delay, 5.0),
        )
        delay = base_delay
        startup_failures = 0
        connected = False
        outage = False
        while not self._stopping:
            await asyncio.sleep(delay)
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except _RouterUnavailableError as exc:
                cause = exc.__cause__ or exc
                if not connected:
                    if startup_failures == 0:
                        logger.info(
                            "waiting_for_router: SGLang Router is not available "
                            "yet (%r)",
                            cause,
                        )
                    else:
                        logger.debug("waiting_for_router: %r", cause)
                    startup_failures += 1
                    delay = startup_backoff[
                        min(startup_failures, len(startup_backoff) - 1)
                    ]
                else:
                    if not outage:
                        logger.warning("SGLang Router became unavailable (%r)", cause)
                    else:
                        logger.debug("SGLang Router remains unavailable: %r", cause)
                    outage = True
                    delay = base_delay
            except Exception:
                logger.warning("engine rebalancing load refresh failed", exc_info=True)
                delay = base_delay
            else:
                if not connected:
                    logger.info("SGLang Router discovery is ready")
                elif outage:
                    logger.info("SGLang Router connection recovered")
                connected = True
                outage = False
                startup_failures = 0
                delay = base_delay

    async def refresh(self) -> None:
        if not self.config.enabled:
            return
        if self._calibration_task is not None and not self._calibration_task.done():
            return
        async with self._refresh_lock:
            await self._refresh_unlocked()

    async def _refresh_unlocked(self) -> None:
        try:
            workers = await self.client.list_workers()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _RouterUnavailableError(str(exc)) from exc
        healthy = []
        for worker in workers:
            url = (
                worker.get("url")
                if isinstance(worker, Mapping)
                else getattr(worker, "url", None)
            )
            is_healthy = (
                worker.get("is_healthy")
                if isinstance(worker, Mapping)
                else getattr(worker, "is_healthy", False)
            )
            mode = (
                worker.get("connection_mode")
                if isinstance(worker, Mapping)
                else getattr(worker, "connection_mode", None)
            )
            if url and is_healthy and str(mode or "http").lower() == "http":
                healthy.append(str(url).rstrip("/"))

        inspection_started = time.monotonic()

        async def inspect_worker(
            url: str,
        ) -> tuple[str, EngineDeploymentInfo | None, bool]:
            last_checked = self._deployment_checked_at.get(url, 0.0)
            refresh_deployment = (
                url not in self.deployments
                or inspection_started - last_checked >= self._deployment_refresh_seconds
            )
            if not refresh_deployment:
                return url, None, False
            info, version = await asyncio.gather(
                self.client.get_server_info(url),
                self.client.get_worker_weight_version(url),
                return_exceptions=True,
            )
            if isinstance(info, Exception) or isinstance(version, Exception):
                return url, None, False
            deployment = EngineDeploymentInfo.from_worker(
                worker_url=url,
                server_info=info,
                weight_version=version,
                model_id=self.model_id,
            )
            return url, deployment, True

        inspected = await asyncio.gather(*(inspect_worker(url) for url in healthy))
        now = time.monotonic()
        new_plans: list[tuple[str, CalibrationPlan]] = []
        async with self._lock:
            for url, deployment, deployment_checked in inspected:
                if deployment is not None:
                    if sglang_rebalancing_supported(deployment.sglang_version):
                        self.deployments[url] = deployment
                        self.excluded_engines.pop(url, None)
                    else:
                        self.deployments.pop(url, None)
                        self.excluded_engines[url] = (
                            "SGLang "
                            f"{deployment.sglang_version} is older than v0.5.15.post1"
                        )
                if deployment_checked:
                    self._deployment_checked_at[url] = now
                self.loads.setdefault(url, EngineLoad(worker_url=url))
            for url, load in self.loads.items():
                load.healthy = url in healthy

            fingerprints = {
                item.cache_fingerprint
                for item in self.deployments.values()
                if item.worker_url in healthy
            }
            for fingerprint in fingerprints:
                pool_deployments = [
                    item
                    for item in self.deployments.values()
                    if item.cache_fingerprint == fingerprint
                    and item.worker_url in healthy
                ]
                state = self.pools.setdefault(
                    fingerprint,
                    CompatibilityPoolStateMachine(fingerprint, self.config),
                )
                if fingerprint not in self.profiles and self.model_config is not None:
                    template = pool_deployments[0]
                    profile = ModelCacheProfile.from_model_config(
                        self.model_config,
                        deployment={
                            **asdict(template),
                            "fingerprint": fingerprint,
                        },
                    )
                    self.profiles[fingerprint] = replace(
                        profile,
                        fingerprint=fingerprint,
                    )
                profile = self.profiles.get(fingerprint)
                if profile is not None:
                    shared_l3 = any(item.shared_l3 for item in pool_deployments)
                    plan = CalibrationPlan.build(
                        fingerprint=fingerprint,
                        engine_deployments=pool_deployments,
                        shared_l3=shared_l3,
                        host_staging=any(
                            item.host_staging for item in pool_deployments
                        ),
                        gpudirect=any(item.gpudirect for item in pool_deployments),
                        model_cache_profile=profile,
                    )
                    if self.plans.get(fingerprint) != plan:
                        self.plans[fingerprint] = plan
                        new_plans.append((fingerprint, plan))
                    elif (
                        self.calibration_benchmark is not None
                        and not self.calibrator.plan_complete(plan)
                    ):
                        new_plans.append((fingerprint, plan))
                readiness = self._pool_readiness(fingerprint, now=now)
                state.update(readiness)
            for fingerprint, state in self.pools.items():
                if fingerprint not in fingerprints:
                    state.update(self._pool_readiness(fingerprint, now=now))

        for fingerprint, plan in new_plans:
            if self.calibration_benchmark is None:
                continue
            self.calibrator.transition(
                CalibrationState.RUNNING,
                "executing injected calibration benchmark",
                plan_fingerprint=plan.fingerprint,
            )
            await self.calibrator.execute(plan, self.calibration_benchmark)
            self.calibrator.transition(
                (
                    CalibrationState.READY
                    if self.calibrator.plan_complete(plan)
                    else CalibrationState.DEGRADED
                ),
                (
                    "injected calibration benchmark complete"
                    if self.calibrator.plan_complete(plan)
                    else "injected calibration benchmark incomplete"
                ),
                plan_fingerprint=plan.fingerprint,
            )
            async with self._lock:
                state = self.pools.get(fingerprint)
                if state is not None:
                    state.update(
                        self._pool_readiness(fingerprint, now=time.monotonic())
                    )

    @staticmethod
    def _normalize_load(
        url: str,
        payload: Mapping[str, Any],
        *,
        now: float,
    ) -> EngineLoad | None:
        rows = payload.get("loads") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list) or not rows:
            return None
        valid = [row for row in rows if isinstance(row, Mapping)]
        if len(valid) != len(rows):
            return None
        queues = [
            row.get("queues") if isinstance(row.get("queues"), Mapping) else {}
            for row in valid
        ]
        return EngineLoad(
            worker_url=url,
            healthy=True,
            metrics_timestamp=now,
            running=sum(int(row.get("num_running_reqs") or 0) for row in valid),
            queued=sum(
                int(row.get("num_waiting_reqs") or row.get("num_queue_reqs") or 0)
                for row in valid
            ),
            active_tokens=sum(
                int(row.get("num_total_tokens") or row.get("num_used_tokens") or 0)
                for row in valid
            ),
            token_capacity=sum(
                int(row.get("max_total_num_tokens") or 0) for row in valid
            ),
            request_capacity=sum(
                int(row.get("max_running_requests") or 0) for row in valid
            ),
            token_usage=max(float(row.get("token_usage") or 0.0) for row in valid),
            waiting_uncached_tokens=sum(
                int(row.get("num_waiting_uncached_tokens") or 0) for row in valid
            ),
            gen_throughput=sum(
                float(row.get("gen_throughput") or 0.0) for row in valid
            ),
            queue_waiting=sum(int(queue.get("waiting") or 0) for queue in queues),
            queue_paused=sum(int(queue.get("paused") or 0) for queue in queues),
            queue_retracted=sum(int(queue.get("retracted") or 0) for queue in queues),
            queue_grammar=sum(int(queue.get("grammar") or 0) for queue in queues),
            live_queue_metrics_available=all(
                row.get("num_waiting_uncached_tokens") is not None for row in valid
            ),
        )

    def _advance_prefill_reservation_generation(self, url: str) -> None:
        """Retire reservations after one complete observable load generation."""

        generation = self._load_generations[url] + 1
        self._load_generations[url] = generation
        retired = False
        for entry in self._reservations.values():
            if (
                entry.engine_url == url
                and entry.prefill_active
                and entry.prefill_reservation_generation is not None
                and entry.prefill_reservation_generation < generation
            ):
                entry.prefill_active = False
                retired = True
        if retired:
            self._reservation_revision += 1
            self._synchronize_reserved_load(url)

    def _live_reservation_totals(self, url: str) -> tuple[int, int, int]:
        entries = [
            entry for entry in self._reservations.values() if entry.engine_url == url
        ]
        return (
            sum(entry.request_increment for entry in entries),
            sum(entry.token_increment for entry in entries),
            sum(
                entry.prefill_increment for entry in entries if entry.prefill_active
            ),
        )

    def _synchronize_reserved_load(self, url: str) -> None:
        load = self.loads.get(url)
        if load is None:
            return
        (
            load.reserved_requests,
            load.reserved_tokens,
            load.reserved_prefill_tokens,
        ) = self._live_reservation_totals(url)

    def _release_reservation(self, reservation_id: int | None) -> None:
        if reservation_id is None:
            return
        entry = self._reservations.pop(reservation_id, None)
        if entry is None:
            return
        self._reservation_revision += 1
        self._synchronize_reserved_load(entry.engine_url)

    def _pool_readiness(self, fingerprint: str, *, now: float) -> PoolReadiness:
        urls = [
            url
            for url, item in self.deployments.items()
            if item.cache_fingerprint == fingerprint
            and self.loads.get(url) is not None
            and self.loads[url].healthy
        ]
        stale_seconds = self.config.metrics_stale_ms / 1000.0
        healthy_urls = [
            url
            for url in urls
            if self.loads.get(url) is not None
            and self.loads[url].fresh(now=now, stale_seconds=stale_seconds)
        ]
        eligible_paths = 0
        for source in healthy_urls:
            for target in healthy_urls:
                if source == target:
                    continue
                if (
                    self._path_readiness(source, target).cache_source
                    is CacheSource.MOONCAKE
                ):
                    eligible_paths += 1
        return PoolReadiness(
            healthy_engines=len(healthy_urls),
            metrics_fresh=len(healthy_urls) == len(urls) and bool(urls),
            model_cache_profile_ready=(
                fingerprint in self.profiles and self.profiles[fingerprint].available
            ),
            queue_model_ready=self.performance.queue_ready(fingerprint),
            prefill_model_ready=self.performance.prefill_ready(fingerprint),
            eligible_paths=eligible_paths,
        )

    def _path_readiness(self, source: str, target: str) -> ContextPathReadiness:
        source_deployment = self.deployments.get(source)
        target_deployment = self.deployments.get(target)
        if source_deployment is None or target_deployment is None:
            return ContextPathReadiness(source, target, CacheSource.NONE, False, False)
        prefill_ready = self.performance.prefill_throughput(
            fingerprint=target_deployment.cache_fingerprint,
            engine_url=target,
            context_tokens=8 * 1024,
        ) is not None or self.performance.prefill_ready(
            target_deployment.cache_fingerprint
        )
        if not target_deployment.shared_l3:
            return ContextPathReadiness(
                source,
                target,
                CacheSource.NONE,
                prefill_ready,
                prefill_ready,
                skipped_links={"mooncake": "L3 disabled"},
            )
        source_calibration_node = self._calibration_node_for(source_deployment)
        target_calibration_node = self._calibration_node_for(target_deployment)
        if target_deployment.gpudirect:
            link = "mooncake_gpudirect"
        elif (
            source_calibration_node is not None
            and source_calibration_node == target_calibration_node
        ) or source_deployment.node_id == target_deployment.node_id:
            link = "mooncake_local"
        else:
            protocol = re.sub(
                r"[^a-z0-9]+", "_", target_deployment.mooncake_protocol or "remote"
            ).strip("_")
            link = f"mooncake_{protocol or 'remote'}"
        required = [link]
        if target_deployment.host_staging:
            # HiCache write-through persists the completed source prefix before
            # the next turn.  Migration-time restore therefore needs only the
            # target H2D leg; source D2H is retained as calibration telemetry.
            required.append("h2d")
        calibration_compatible = self._deployment_matches_preflight(
            source_deployment
        ) and self._deployment_matches_preflight(target_deployment)
        completed_set = (
            self.calibrator.completed_links(
                source_calibration_node, target_calibration_node
            )
            if calibration_compatible
            and source_calibration_node is not None
            and target_calibration_node is not None
            else set()
        )
        completed = tuple(item for item in required if item in completed_set)
        pending = tuple(item for item in required if item not in completed_set)
        profile = self.profiles.get(target_deployment.cache_fingerprint)
        mooncake_ready = not pending and profile is not None and profile.available
        skipped = {
            "p2p": "native restore path does not use CUDA P2P",
            **(
                {"h2d": "GPUDirect path", "d2h": "GPUDirect path"}
                if target_deployment.gpudirect
                else {}
            ),
        }
        if not mooncake_ready:
            skipped["fallback"] = (
                "full prefill: deployment differs from preflight configuration"
                if not calibration_compatible
                else (
                    "full prefill: model cache profile unavailable"
                    if profile is None or not profile.available
                    else "full prefill: restore path calibration pending"
                )
            )
        return ContextPathReadiness(
            source,
            target,
            CacheSource.MOONCAKE if mooncake_ready else CacheSource.NONE,
            prefill_ready,
            prefill_ready,
            required_links=tuple(required),
            completed_links=completed,
            pending_links=pending,
            skipped_links=skipped,
        )

    def _candidate_path_readiness(
        self,
        session: SessionRoutingState,
        source: str,
        target: str,
    ) -> ContextPathReadiness:
        """Return readiness for the path this session would actually use."""
        if target == session.owner_worker_url:
            deployment = self.deployments[target]
            prefill_ready = self.performance.prefill_throughput(
                fingerprint=deployment.cache_fingerprint,
                engine_url=target,
                context_tokens=8 * 1024,
            ) is not None or self.performance.prefill_ready(
                deployment.cache_fingerprint
            )
            return ContextPathReadiness(
                source,
                target,
                CacheSource.LOCAL,
                prefill_ready,
                prefill_ready,
                skipped_links={"transfer": "target is the current session owner"},
            )
        return self._path_readiness(source, target)

    def _calibration_node_for(self, deployment: EngineDeploymentInfo) -> str | None:
        if self.calibration_benchmark is not None:
            return deployment.node_id
        node_id = deployment.node_id
        target = self._preflight_node_aliases.get(node_id)
        if target is not None:
            return target
        try:
            resolved = socket.gethostbyname(node_id)
        except OSError:
            resolved = node_id
        target = self._preflight_node_aliases.get(resolved)
        if target is not None:
            return target
        if len(self._preflight_node_addresses) == 1 and resolved.startswith("127."):
            # The Router can expose loopback worker URLs on a single-node
            # deployment even though Ray identifies that machine by its
            # routable address.
            return next(iter(self._preflight_node_addresses))
        return None

    def _deployment_matches_preflight(self, deployment: EngineDeploymentInfo) -> bool:
        # Injected benchmarks are intentionally independent of the production
        # deployment-file contract and remain available to unit/embedder tests.
        if self.calibration_benchmark is not None:
            return True
        config = self.machine_calibration_config
        if config is None or self.calibrator.state not in {
            CalibrationState.READY,
            CalibrationState.DEGRADED,
        }:
            return False
        if deployment.shared_l3 != config.shared_l3:
            return False
        if not deployment.shared_l3:
            return True
        if deployment.gpudirect != config.gpudirect:
            return False
        if deployment.mooncake_protocol != config.protocol:
            return False
        mooncake = dict(deployment.mooncake_args)
        if config.device_name and str(mooncake.get("device_name") or "") != str(
            config.device_name
        ):
            return False
        actual_metadata_server = str(
            mooncake.get("metadata_server")
            or mooncake.get("metadata_conn_string")
            or ""
        )
        if config.metadata_server and actual_metadata_server != config.metadata_server:
            return False
        if self._preflight_node_ids and self._calibration_node_for(deployment) is None:
            return False
        hicache = dict(deployment.hicache_args)
        if (
            str(hicache.get("hicache_write_policy") or "").lower()
            != config.write_policy
        ):
            return False
        expected = config.model_deployment
        for key, actual in (
            ("kv_dtype", deployment.kv_dtype),
            ("state_dtype", deployment.state_dtype),
            ("page_size", deployment.page_size),
            ("swa_window_size", deployment.swa_window_size),
            ("mamba_track_interval", deployment.mamba_track_interval),
        ):
            if key in expected and str(expected[key]) != str(actual):
                return False
        return True

    def _resolve_step_budget(
        self,
        *,
        session: SessionRoutingState,
        fingerprint: str,
        step_max_new_tokens: int | None,
        context_remaining_tokens: int | None,
    ) -> StepGenerationBudget:
        caps: list[tuple[str, int]] = []
        for source, raw_value in (
            ("request_or_proxy", step_max_new_tokens),
            ("rollout", session.default_step_max_tokens),
            ("context", context_remaining_tokens),
        ):
            if raw_value is None:
                continue
            value = int(raw_value)
            if value > 0:
                caps.append((source, value))
        effective = min((value for _, value in caps), default=None)
        cap_source = (
            "min(" + ",".join(source for source, _ in caps) + ")"
            if caps
            else "unavailable"
        )
        historical = self.step_lengths.p75(
            fingerprint=fingerprint,
            task_key=session.task_key,
            max_tokens=effective,
        )
        group_remaining = self.group_lengths.remaining(
            group_id=session.group_id,
            task_key=session.task_key,
            generated_tokens=session.generated_tokens,
        )
        estimates = [
            value
            for value in (effective, historical, group_remaining)
            if value is not None
        ]
        estimated = min(estimates) if estimates else None
        return StepGenerationBudget(
            step_max_tokens_source=cap_source,
            effective_step_max_tokens=effective,
            historical_step_tokens_p75=historical,
            group_remaining_tokens=group_remaining,
            estimated_step_output_tokens=estimated,
        )

    async def acquire(
        self,
        *,
        session_id: str,
        input_ids: list[int],
        step_max_new_tokens: int | None = None,
        context_remaining_tokens: int | None = None,
        expected_version: str | None = None,
        require_registered_context: bool = False,
    ) -> RoutingLease:
        if not self.config.enabled:
            budget = self._resolve_step_budget(
                session=SessionRoutingState(),
                fingerprint="",
                step_max_new_tokens=step_max_new_tokens,
                context_remaining_tokens=context_remaining_tokens,
            )
            decision = RoutingDecision(
                session_id=session_id,
                source_worker_url=None,
                target_worker_url=None,
                cache_fingerprint=None,
                state=SchedulerState.OFF,
                reason="engine_rebalancing_disabled",
                **budget.snapshot(),
            )
            return RoutingLease(
                decision=decision,
                worker_url=None,
                reserved_tokens=0,
                base_tokens=0,
                started_monotonic=time.monotonic(),
                context_tokens=len(input_ids),
                expected_output_tokens=budget.estimated_step_output_tokens or 0,
            )
        if not self.deployments:
            calibration_task = self._calibration_task
            if calibration_task is not None and not calibration_task.done():
                await asyncio.shield(calibration_task)
            try:
                await self.refresh()
            except Exception:
                logger.warning("initial engine discovery failed", exc_info=True)

        return await self._enqueue_batch_acquire(
            session_id=session_id,
            input_ids=input_ids,
            step_max_new_tokens=step_max_new_tokens,
            context_remaining_tokens=context_remaining_tokens,
            expected_version=expected_version,
            require_registered_context=require_registered_context,
        )

    async def _enqueue_batch_acquire(
        self,
        *,
        session_id: str,
        input_ids: list[int],
        step_max_new_tokens: int | None,
        context_remaining_tokens: int | None,
        expected_version: str | None,
        require_registered_context: bool,
    ) -> RoutingLease:
        loop = asyncio.get_running_loop()
        async with self._batch_lock:
            if session_id in self._pending_acquires:
                raise RuntimeError(f"session already has a pending acquire: {session_id}")
            step = _PendingBatchStep(
                arrival_id=self._next_arrival_id,
                session_id=session_id,
                input_ids=tuple(input_ids),
                step_max_new_tokens=step_max_new_tokens,
                context_remaining_tokens=context_remaining_tokens,
                expected_version=(
                    None if expected_version is None else str(expected_version)
                ),
                require_registered_context=require_registered_context,
                future=loop.create_future(),
            )
            self._next_arrival_id += 1
            batch = self._open_batch
            if batch is None:
                batch = _OpenBatch(
                    id=self._next_batch_id,
                    started_monotonic=time.monotonic(),
                )
                self._next_batch_id += 1
                self._open_batch = batch
                runner = asyncio.create_task(
                    self._run_batch(batch),
                    name=f"engine-rebalancing-batch-{batch.id}",
                )
                self._batch_runner_tasks.add(runner)
                runner.add_done_callback(self._batch_runner_tasks.discard)
            batch.steps.append(step)
            self._pending_acquires[session_id] = step
        try:
            return await asyncio.shield(step.future)
        except asyncio.CancelledError:
            async with self._batch_lock:
                step.cancelled = True
                lease = step.lease
            if lease is not None:
                await asyncio.shield(self.fail(lease))
            raise
        finally:
            async with self._batch_lock:
                if self._pending_acquires.get(session_id) is step:
                    self._pending_acquires.pop(session_id, None)

    async def _fetch_batch_load(self, url: str) -> _BatchFetchResult:
        started = time.monotonic()
        try:
            payload = await asyncio.wait_for(
                self.client.get_worker_loads(url),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            return _BatchFetchResult(
                url, "timeout", time.monotonic() - started, 0, None
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _BatchFetchResult(
                url, "error", time.monotonic() - started, 0, None
            )
        rows = payload.get("loads") if isinstance(payload, Mapping) else None
        try:
            load = self._normalize_load(url, payload, now=time.monotonic())
        except (TypeError, ValueError, OverflowError):
            load = None
        if (
            load is None
            or load.request_capacity <= 0
            or load.token_capacity <= 0
        ):
            return _BatchFetchResult(
                url,
                "invalid",
                time.monotonic() - started,
                len(rows) if isinstance(rows, list) else 0,
                None,
            )
        return _BatchFetchResult(
            url,
            "ok",
            time.monotonic() - started,
            len(rows),
            load,
        )

    @staticmethod
    def _batch_session_signature(
        session: SessionRoutingState | None,
    ) -> tuple[Any, ...] | None:
        if session is None:
            return None
        return (
            session.owner_worker_url,
            session.pending_owner_worker_url,
            session.fingerprint,
            tuple(session.previous_committed_tokens),
            tuple(sorted(session.seen_engines)),
            session.group_id,
            session.group_size,
            session.task_key,
            session.generated_tokens,
            session.default_step_max_tokens,
        )

    def _batch_topology_signature(self) -> tuple[tuple[str, bool, str, str], ...]:
        return tuple(
            sorted(
                (
                    url,
                    self.loads.get(url) is not None and self.loads[url].healthy,
                    deployment.weight_version,
                    deployment.cache_fingerprint,
                )
                for url, deployment in self.deployments.items()
            )
        )

    async def _freeze_batch(
        self,
        batch: _OpenBatch,
        fetch_results: tuple[_BatchFetchResult, ...],
    ) -> _FrozenBatch:
        async with self._lock:
            successful = {
                result.url: result.load
                for result in fetch_results
                if result.status == "ok" and result.load is not None
            }
            for url, snapshot in successful.items():
                current = self.loads.get(url)
                control_healthy = current is not None and current.healthy
                published = deepcopy(snapshot)
                published.healthy = control_healthy
                self.loads[url] = published
                self._synchronize_reserved_load(url)
                if published.live_queue_metrics_available:
                    self._advance_prefill_reservation_generation(url)

            readiness_now = time.monotonic()
            for fingerprint, state in self.pools.items():
                state.update(
                    self._pool_readiness(fingerprint, now=readiness_now)
                )

            successful_urls = tuple(
                sorted(
                    url
                    for url in successful
                    if url in self.deployments
                    and self.loads[url].healthy
                )
            )
            engines = []
            engine_traces = []
            baselines: dict[str, EngineBaseline] = {}
            for url in successful_urls:
                snapshot = successful[url]
                baseline = EngineBaseline(
                    url=url,
                    base_requests=snapshot.running,
                    base_tokens=snapshot.active_tokens,
                    base_queue=snapshot.queued,
                    request_capacity=snapshot.request_capacity,
                    token_capacity=snapshot.token_capacity,
                    token_usage=snapshot.token_usage,
                )
                engines.append(baseline)
                baselines[url] = baseline

            for result in fetch_results:
                deployment = self.deployments.get(result.url)
                control_load = self.loads.get(result.url)
                snapshot = result.load
                live_requests, live_tokens, live_prefill = (
                    self._live_reservation_totals(result.url)
                )
                baseline = baselines.get(result.url)
                engine_traces.append(
                    _FrozenBatchEngineTrace(
                        url=result.url,
                        fetch_status=result.status,
                        fetch_duration_seconds=result.duration_seconds,
                        health=control_load is not None and control_load.healthy,
                        version=(
                            None if deployment is None else deployment.weight_version
                        ),
                        fingerprint=(
                            None
                            if deployment is None
                            else deployment.cache_fingerprint
                        ),
                        row_count=result.row_count,
                        running=None if snapshot is None else snapshot.running,
                        active_tokens=(
                            None if snapshot is None else snapshot.active_tokens
                        ),
                        request_capacity=(
                            None if snapshot is None else snapshot.request_capacity
                        ),
                        token_capacity=(
                            None if snapshot is None else snapshot.token_capacity
                        ),
                        token_usage=(
                            None if snapshot is None else snapshot.token_usage
                        ),
                        gen_throughput=(
                            None
                            if snapshot is None
                            or not math.isfinite(snapshot.gen_throughput)
                            else snapshot.gen_throughput
                        ),
                        queued=None if snapshot is None else snapshot.queued,
                        queue_pressure=(
                            None
                            if baseline is None
                            else baseline.base_queue / baseline.request_capacity
                        ),
                        waiting_uncached_tokens=(
                            None
                            if snapshot is None
                            else snapshot.waiting_uncached_tokens
                        ),
                        live_ledger_requests=live_requests,
                        live_ledger_tokens=live_tokens,
                        live_ledger_prefill=live_prefill,
                        base_requests=(
                            None if baseline is None else int(baseline.base_requests)
                        ),
                        base_tokens=(
                            None if baseline is None else int(baseline.base_tokens)
                        ),
                        base_queue=(
                            None if baseline is None else int(baseline.base_queue)
                        ),
                    )
                )

            frozen_steps: list[_FrozenBatchStep] = []
            sticky_edges: dict[str, tuple[FeasibleEdge, ...]] = {}
            optimized_edges: dict[str, tuple[FeasibleEdge, ...]] = {}
            for pending in sorted(batch.steps, key=lambda item: item.arrival_id):
                existing = self.sessions.get(pending.session_id)
                signature = self._batch_session_signature(existing)
                if pending.cancelled:
                    frozen_steps.append(
                        _FrozenBatchStep(
                            pending=pending,
                            session_signature=signature,
                            source=(
                                None if existing is None else existing.owner_worker_url
                            ),
                            fingerprint=(
                                None if existing is None else existing.fingerprint
                            ),
                            edges=(),
                            budgets=(),
                            base_tokens=(),
                        )
                    )
                    continue
                if existing is None and pending.require_registered_context:
                    frozen_steps.append(
                        _FrozenBatchStep(
                            pending=pending,
                            session_signature=signature,
                            source=None,
                            fingerprint=None,
                            edges=(),
                            budgets=(),
                            base_tokens=(),
                            failure=(
                                "session context is not registered or was discarded: "
                                f"{pending.session_id}"
                            ),
                        )
                    )
                    continue

                session = existing or SessionRoutingState()
                source = session.owner_worker_url
                source_deployment = (
                    None if source is None else self.deployments.get(source)
                )
                fingerprint = session.fingerprint or (
                    None
                    if source_deployment is None
                    else source_deployment.cache_fingerprint
                )
                version_valid_owner = (
                    source_deployment is not None
                    and (
                        pending.expected_version is None
                        or source_deployment.weight_version
                        == pending.expected_version
                    )
                )
                healthy_owner = (
                    source is not None
                    and version_valid_owner
                    and self.loads.get(source) is not None
                    and self.loads[source].healthy
                )
                owner_snapshot_ok = source in successful_urls
                if healthy_owner and not owner_snapshot_ok:
                    budget = self._resolve_step_budget(
                        session=session,
                        fingerprint=fingerprint or "",
                        step_max_new_tokens=pending.step_max_new_tokens,
                        context_remaining_tokens=pending.context_remaining_tokens,
                    )
                    base = longest_common_prefix_length(
                        session.previous_committed_tokens,
                        pending.input_ids,
                    )
                    frozen_steps.append(
                        _FrozenBatchStep(
                            pending=pending,
                            session_signature=signature,
                            source=source,
                            fingerprint=fingerprint,
                            edges=(),
                            budgets=((source, budget),),
                            base_tokens=((source, base),),
                            fixed_target=source,
                        )
                    )
                    continue

                candidates = [
                    url
                    for url in successful_urls
                    if pending.expected_version is None
                    or self.deployments[url].weight_version
                    == pending.expected_version
                ]
                if source is not None:
                    candidates = [
                        url
                        for url in candidates
                        if fingerprint is not None
                        and self.deployments[url].cache_fingerprint == fingerprint
                    ]

                edges: list[FeasibleEdge] = []
                budgets: list[tuple[str, StepGenerationBudget]] = []
                bases: list[tuple[str, int]] = []
                lcp = (
                    0
                    if source is None
                    else longest_common_prefix_length(
                        session.previous_committed_tokens,
                        pending.input_ids,
                    )
                )
                mandatory_failover = source is not None and not healthy_owner
                budgets_by_fingerprint: dict[str, StepGenerationBudget] = {}
                for target in candidates:
                    voluntary = False
                    if source is not None and not mandatory_failover and target != source:
                        if (
                            self._candidate_path_readiness(
                                session,
                                source,
                                target,
                            ).cache_source
                            is not CacheSource.MOONCAKE
                        ):
                            continue
                        voluntary = True
                    target_fingerprint = self.deployments[target].cache_fingerprint
                    budget = budgets_by_fingerprint.get(target_fingerprint)
                    if budget is None:
                        budget = self._resolve_step_budget(
                            session=session,
                            fingerprint=target_fingerprint,
                            step_max_new_tokens=pending.step_max_new_tokens,
                            context_remaining_tokens=pending.context_remaining_tokens,
                        )
                        budgets_by_fingerprint[target_fingerprint] = budget
                    prompt_tokens = len(pending.input_ids)
                    if healthy_owner and target == source:
                        prefill_increment = prompt_tokens - lcp
                    elif voluntary:
                        page_size = max(1, self.deployments[target].page_size)
                        page_aligned_lcp = lcp - lcp % page_size
                        prefill_increment = prompt_tokens - page_aligned_lcp
                    else:
                        prefill_increment = prompt_tokens
                    edges.append(
                        FeasibleEdge(
                            session_id=pending.session_id,
                            engine_url=target,
                            queue_increment=1,
                            token_increment=(
                                max(0, prompt_tokens - lcp)
                                if healthy_owner and target == source
                                else prompt_tokens
                            ),
                            prefill_increment=prefill_increment,
                            voluntary_migration=voluntary,
                            migration_cost_tokens=lcp if voluntary else 0,
                        )
                    )
                    budgets.append((target, budget))
                    bases.append((target, lcp))

                if not edges:
                    frozen_steps.append(
                        _FrozenBatchStep(
                            pending=pending,
                            session_signature=signature,
                            source=source,
                            fingerprint=fingerprint,
                            edges=(),
                            budgets=(),
                            base_tokens=(),
                            failure=(
                                "engine rebalancing batch found no eligible worker "
                                "with a successful load snapshot"
                            ),
                        )
                    )
                    continue
                frozen_edges = tuple(edges)
                frozen_steps.append(
                    _FrozenBatchStep(
                        pending=pending,
                        session_signature=signature,
                        source=source,
                        fingerprint=fingerprint,
                        edges=frozen_edges,
                        budgets=tuple(budgets),
                        base_tokens=tuple(bases),
                    )
                )
                optimized_edges[pending.session_id] = frozen_edges
                sticky_edges[pending.session_id] = tuple(
                    edge for edge in frozen_edges if not edge.voluntary_migration
                )

            sticky_problem = (
                BatchProblem(engines, sticky_edges)
                if engines and sticky_edges
                else None
            )
            optimized_problem = (
                BatchProblem(engines, optimized_edges)
                if engines and optimized_edges
                else None
            )
            decision_engines = list(engines)
            for url in sorted(
                {
                    step.fixed_target
                    for step in frozen_steps
                    if step.fixed_target is not None
                }
                - baselines.keys()
            ):
                cached = self.loads[url]
                fingerprint = self.deployments[url].cache_fingerprint
                compatible = [
                    engine
                    for engine in engines
                    if self.deployments[engine.url].cache_fingerprint == fingerprint
                ]
                request_capacity = cached.request_capacity or max(
                    [engine.request_capacity for engine in compatible] + [1]
                )
                token_capacity = cached.token_capacity or max(
                    [engine.token_capacity for engine in compatible] + [1]
                )
                decision_engines.append(
                    EngineBaseline(
                        url=url,
                        base_requests=cached.running,
                        base_tokens=cached.active_tokens,
                        base_queue=cached.queued,
                        request_capacity=request_capacity,
                        token_capacity=token_capacity,
                        token_usage=cached.token_usage,
                    )
                )
            return _FrozenBatch(
                reservation_revision=self._reservation_revision,
                topology_signature=self._batch_topology_signature(),
                decision_engines=tuple(decision_engines),
                engine_traces=tuple(engine_traces),
                steps=tuple(frozen_steps),
                sticky_problem=sticky_problem,
                optimized_problem=optimized_problem,
            )

    def _solve_frozen_batch(self, frozen: _FrozenBatch) -> _SolvedBatch:
        started = time.monotonic()
        if frozen.sticky_problem is None:
            return _SolvedBatch(
                assignment=(),
                sticky_greedy=None,
                sticky=None,
                optimized=None,
                adopted="fixed",
                target_maximum_load=None,
                improvement_ratio=None,
                fallback_reason=None,
                elapsed_seconds=time.monotonic() - started,
            )
        sticky_greedy = solve_batch_greedy(frozen.sticky_problem)
        try:
            sticky = solve_batch_milp(
                frozen.sticky_problem,
                deadline_seconds=max(1e-9, 1.0 - (time.monotonic() - started)),
            )
            if sticky.status is not SolverStatus.OPTIMAL:
                raise RuntimeError("sticky batch solution was not optimal")
        except Exception:
            return _SolvedBatch(
                assignment=tuple(sticky_greedy.assignment.items()),
                sticky_greedy=sticky_greedy,
                sticky=sticky_greedy,
                optimized=None,
                adopted="sticky_greedy",
                target_maximum_load=None,
                improvement_ratio=None,
                fallback_reason="sticky_solver_failure",
                elapsed_seconds=time.monotonic() - started,
            )

        target_maximum_load = sticky.maximum_load * (
            1.0 - self.config.min_load_improvement_ratio
        )
        has_voluntary_edge = any(
            edge.voluntary_migration
            for edges in frozen.optimized_problem.edges_by_session.values()
            for edge in edges
        )
        if sticky.maximum_load <= 1e-9 or not has_voluntary_edge:
            return _SolvedBatch(
                assignment=tuple(sticky.assignment.items()),
                sticky_greedy=sticky_greedy,
                sticky=sticky,
                optimized=sticky,
                adopted="sticky",
                target_maximum_load=target_maximum_load,
                improvement_ratio=0.0,
                fallback_reason=None,
                elapsed_seconds=time.monotonic() - started,
            )
        baseline_lower_bound = max(
            self._batch_load_score(engine).total
            for engine in frozen.optimized_problem.engines
        )
        if baseline_lower_bound > target_maximum_load + 1e-7:
            return _SolvedBatch(
                assignment=tuple(sticky.assignment.items()),
                sticky_greedy=sticky_greedy,
                sticky=sticky,
                optimized=None,
                adopted="sticky",
                target_maximum_load=target_maximum_load,
                improvement_ratio=None,
                fallback_reason="target_load_infeasible",
                elapsed_seconds=time.monotonic() - started,
            )
        try:
            remaining = 1.0 - (time.monotonic() - started)
            if remaining <= 0:
                raise BatchSolverError(
                    phase=1,
                    status=None,
                    elapsed_seconds=1.0 - remaining,
                    message="shared batch solve deadline expired",
                )
            optimized = solve_batch_for_target_load(
                frozen.optimized_problem,
                maximum_load_limit=target_maximum_load,
                deadline_seconds=remaining,
            )
            if optimized.status is not SolverStatus.OPTIMAL:
                raise RuntimeError("optimized batch solution was not optimal")
            if optimized.maximum_load > target_maximum_load + 1e-7:
                raise RuntimeError("optimized batch solution exceeded its load target")
        except BatchSolverError as exc:
            if exc.status == 2:
                fallback_reason = "target_load_infeasible"
            elif exc.status in (None, 1):
                fallback_reason = "target_solver_deadline"
            else:
                fallback_reason = "target_solver_failure"
            return _SolvedBatch(
                assignment=tuple(sticky.assignment.items()),
                sticky_greedy=sticky_greedy,
                sticky=sticky,
                optimized=None,
                adopted="sticky",
                target_maximum_load=target_maximum_load,
                improvement_ratio=None,
                fallback_reason=fallback_reason,
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception:
            return _SolvedBatch(
                assignment=tuple(sticky.assignment.items()),
                sticky_greedy=sticky_greedy,
                sticky=sticky,
                optimized=None,
                adopted="sticky",
                target_maximum_load=target_maximum_load,
                improvement_ratio=None,
                fallback_reason="target_solver_failure",
                elapsed_seconds=time.monotonic() - started,
            )
        improvement = (
            0.0
            if sticky.maximum_load <= 1e-9
            else (sticky.maximum_load - optimized.maximum_load)
            / max(sticky.maximum_load, 1e-9)
        )
        return _SolvedBatch(
            assignment=tuple(optimized.assignment.items()),
            sticky_greedy=sticky_greedy,
            sticky=sticky,
            optimized=optimized,
            adopted="optimized",
            target_maximum_load=target_maximum_load,
            improvement_ratio=improvement,
            fallback_reason=None,
            elapsed_seconds=time.monotonic() - started,
        )

    async def _run_batch(self, batch: _OpenBatch) -> None:
        wait_started = time.monotonic()
        coalescing_deadline = batch.started_monotonic + (
            self.config.load_batch_coalescing_window_ms / 1000.0
        )
        fetch_results: tuple[_BatchFetchResult, ...] = ()
        frozen: _FrozenBatch | None = None
        try:
            async with self._batch_run_lock:
                fetch_started = time.monotonic()
                wait_previous = fetch_started - wait_started
                async with self._lock:
                    urls = tuple(
                        sorted(
                            url
                            for url, deployment in self.deployments.items()
                            if deployment.worker_url == url
                            and self.loads.get(url) is not None
                            and self.loads[url].healthy
                        )
                    )
                fetch_results = tuple(
                    await asyncio.gather(
                        *(self._fetch_batch_load(url) for url in urls)
                    )
                )
                fetch_finished = time.monotonic()
                remaining = coalescing_deadline - fetch_finished
                if remaining > 0:
                    await asyncio.sleep(remaining)
                async with self._batch_lock:
                    batch.sealed_monotonic = time.monotonic()
                    if self._open_batch is batch:
                        self._open_batch = None
                frozen = await self._freeze_batch(batch, fetch_results)
                solved = await asyncio.to_thread(self._solve_frozen_batch, frozen)
                await self._commit_batch(
                    batch,
                    fetch_results,
                    frozen=frozen,
                    solved=solved,
                    wait_previous_seconds=wait_previous,
                    fetch_seconds=fetch_finished - fetch_started,
                )
        except asyncio.CancelledError:
            await self._fail_batch(
                batch,
                RuntimeError("engine rebalancer is closed"),
                fetch_results=fetch_results,
                frozen=frozen,
            )
            raise
        except Exception as exc:
            await self._fail_batch(
                batch,
                exc,
                fetch_results=fetch_results,
                frozen=frozen,
            )

    async def _commit_batch(
        self,
        batch: _OpenBatch,
        fetch_results: tuple[_BatchFetchResult, ...],
        *,
        frozen: _FrozenBatch,
        solved: _SolvedBatch,
        wait_previous_seconds: float,
        fetch_seconds: float,
    ) -> None:
        leases: list[tuple[_PendingBatchStep, RoutingLease]] = []
        failures: list[tuple[_PendingBatchStep, Exception]] = []
        committed_at = time.monotonic()
        async with self._batch_lock:
            async with self._lock:
                frozen_state_changed = (
                    self._reservation_revision != frozen.reservation_revision
                    or self._batch_topology_signature()
                    != frozen.topology_signature
                    or any(
                        self._batch_session_signature(
                            self.sessions.get(step.pending.session_id)
                        )
                        != step.session_signature
                        or self._pending_acquires.get(step.pending.session_id)
                        is not step.pending
                        for step in frozen.steps
                        if not step.pending.cancelled
                    )
                )
                if frozen_state_changed and solved.sticky_greedy is not None:
                    solved = replace(
                        solved,
                        assignment=tuple(
                            solved.sticky_greedy.assignment.items()
                        ),
                        sticky=solved.sticky_greedy,
                        optimized=None,
                        adopted="sticky_greedy",
                        improvement_ratio=None,
                        fallback_reason="frozen_state_changed",
                    )
                sessions_before: dict[str, SessionRoutingState | None] = {}
                affected_reservation_urls: set[str] = set()
                next_reservation_before = self._next_reservation_id
                revision_before = self._reservation_revision
                decisions_before = deque(
                    self._decisions,
                    maxlen=self._decisions.maxlen,
                )
                try:
                    assignment = dict(solved.assignment)
                    selected_targets: dict[int, str] = {}
                    for frozen_step in frozen.steps:
                        step = frozen_step.pending
                        if step.cancelled or frozen_step.failure is not None:
                            continue
                        target = frozen_step.fixed_target or assignment.get(
                            step.session_id
                        )
                        if target is None:
                            failures.append(
                                (step, RuntimeError("batch assignment is missing"))
                            )
                            continue
                        deployment = self.deployments.get(target)
                        current_load = self.loads.get(target)
                        current_session = self.sessions.get(step.session_id)
                        if (
                            self._batch_session_signature(current_session)
                            != frozen_step.session_signature
                        ):
                            failures.append(
                                (
                                    step,
                                    RuntimeError(
                                        "batch assignment target is no longer eligible"
                                    ),
                                )
                            )
                            continue
                        fingerprint = (
                            None
                            if current_session is None
                            else current_session.fingerprint
                        ) or frozen_step.fingerprint
                        selected_edge = next(
                            (
                                edge
                                for edge in frozen_step.edges
                                if edge.engine_url == target
                            ),
                            None,
                        )
                        voluntary_path_ready = (
                            selected_edge is None
                            or not selected_edge.voluntary_migration
                            or (
                                current_session is not None
                                and current_session.owner_worker_url is not None
                                and self._candidate_path_readiness(
                                    current_session,
                                    current_session.owner_worker_url,
                                    target,
                                ).cache_source
                                is CacheSource.MOONCAKE
                            )
                        )
                        frozen_target = (
                            target == frozen_step.fixed_target
                            or selected_edge is not None
                        )
                        if not (
                            frozen_target
                            and voluntary_path_ready
                            and deployment is not None
                            and deployment.worker_url == target
                            and current_load is not None
                            and current_load.healthy
                            and (
                                step.expected_version is None
                                or deployment.weight_version
                                == step.expected_version
                            )
                            and (
                                fingerprint is None
                                or deployment.cache_fingerprint == fingerprint
                            )
                        ):
                            failures.append(
                                (
                                    step,
                                    RuntimeError(
                                        "batch assignment target is no longer eligible"
                                    ),
                                )
                            )
                            continue
                        selected_targets[step.arrival_id] = target
                    for frozen_step in frozen.steps:
                        step = frozen_step.pending
                        if step.arrival_id not in selected_targets:
                            continue
                        if step.session_id not in sessions_before:
                            session = self.sessions.get(step.session_id)
                            sessions_before[step.session_id] = (
                                None if session is None else deepcopy(session)
                            )
                        affected_reservation_urls.add(
                            selected_targets[step.arrival_id]
                        )
                    increments = {
                        engine.url: [0.0, 0.0]
                        for engine in frozen.decision_engines
                    }
                    selected_edges_by_arrival: dict[int, FeasibleEdge] = {}
                    for frozen_step in frozen.steps:
                        if frozen_step.pending.cancelled or frozen_step.failure:
                            continue
                        target = selected_targets.get(
                            frozen_step.pending.arrival_id
                        )
                        if target is None:
                            continue
                        edge = next(
                            (
                                item
                                for item in frozen_step.edges
                                if item.engine_url == target
                            ),
                            None,
                        )
                        if edge is None and frozen_step.fixed_target == target:
                            budget = dict(frozen_step.budgets)[target]
                            edge = FeasibleEdge(
                                session_id=frozen_step.pending.session_id,
                                engine_url=target,
                                queue_increment=1,
                                token_increment=max(
                                    0,
                                    len(frozen_step.pending.input_ids)
                                    - dict(frozen_step.base_tokens)[target],
                                ),
                                prefill_increment=max(
                                    0,
                                    len(frozen_step.pending.input_ids)
                                    - dict(frozen_step.base_tokens)[target],
                                ),
                                voluntary_migration=False,
                            )
                        if edge is not None:
                            selected_edges_by_arrival[
                                frozen_step.pending.arrival_id
                            ] = edge
                            totals = increments[edge.engine_url]
                            totals[0] += edge.queue_increment
                            totals[1] += edge.token_increment
                    base_scores: dict[str, LoadScore] = {}
                    projected_scores: dict[str, LoadScore] = {}
                    for engine in frozen.decision_engines:
                        (
                            queue_increment,
                            token_increment,
                        ) = increments[engine.url]
                        base_scores[engine.url] = self._batch_load_score(engine)
                        projected_scores[engine.url] = self._batch_load_score(
                            engine,
                            queue_increment=queue_increment,
                            token_increment=token_increment,
                        )
                    for frozen_step in frozen.steps:
                        step = frozen_step.pending
                        if step.cancelled:
                            continue
                        if frozen_step.failure is not None:
                            failures.append(
                                (step, RuntimeError(frozen_step.failure))
                            )
                            continue
                        target = selected_targets.get(step.arrival_id)
                        if target is None:
                            continue
                        session = self.sessions.get(step.session_id)
                        if session is None:
                            session = SessionRoutingState()
                            self.sessions[step.session_id] = session
                        source = frozen_step.source
                        budget = dict(frozen_step.budgets)[target]
                        base_tokens = dict(frozen_step.base_tokens)[target]
                        moved = source is not None and target != source
                        optimized_migration = (
                            moved
                            and solved.adopted == "optimized"
                            and any(
                                edge.engine_url == target
                                and edge.voluntary_migration
                                for edge in frozen_step.edges
                            )
                        )
                        has_voluntary_edge = any(
                            edge.voluntary_migration
                            for edge in frozen_step.edges
                        )
                        fingerprint = self.deployments[target].cache_fingerprint
                        decision = RoutingDecision(
                            session_id=step.session_id,
                            source_worker_url=source,
                            target_worker_url=target,
                            cache_fingerprint=fingerprint,
                            state=self.pools[fingerprint].state,
                            reason=(
                                "batch_new_session"
                                if source is None
                                else (
                                    "batch_fixed_owner"
                                    if frozen_step.fixed_target is not None
                                    else (
                                        "batch_optimized_migration"
                                        if optimized_migration
                                        else (
                                            "batch_owner_failover"
                                            if moved
                                            else "batch_sticky"
                                        )
                                    )
                                )
                            ),
                            load_improvement_ratio=(
                                solved.improvement_ratio
                                if has_voluntary_edge
                                else None
                            ),
                            required_load_improvement_ratio=(
                                self.config.min_load_improvement_ratio
                                if has_voluntary_edge
                                else None
                            ),
                            source_base_load=base_scores.get(source),
                            target_base_load=base_scores.get(target),
                            source_projected_load=projected_scores.get(source),
                            target_projected_load=projected_scores.get(target),
                            moved=moved,
                            **budget.snapshot(),
                        )
                        if target != source:
                            session.pending_owner_worker_url = target
                        if source is None:
                            session.fingerprint = fingerprint
                        lease = self._reserve(
                            decision,
                            input_ids=list(step.input_ids),
                            base_tokens=base_tokens,
                            budget=budget,
                            batch_id=batch.id,
                            prefill_increment=int(
                                selected_edges_by_arrival[
                                    step.arrival_id
                                ].prefill_increment
                            ),
                            projected_load_score=projected_scores[target].total,
                        )
                        step.lease = lease
                        leases.append((step, lease))
                    trace = self._batch_trace(
                        batch,
                        fetch_results,
                        frozen=frozen,
                        leases=leases,
                        failures=failures,
                        wait_previous_seconds=wait_previous_seconds,
                        fetch_seconds=fetch_seconds,
                        completed_at=committed_at,
                        solved=solved,
                    )
                    self._load_batch_history.record(LoadBatchTrace(trace))
                    batch.terminal_published = True
                except BaseException:
                    new_reservation_ids = [
                        reservation_id
                        for reservation_id in self._reservations
                        if reservation_id >= next_reservation_before
                    ]
                    for reservation_id in new_reservation_ids:
                        entry = self._reservations.pop(reservation_id)
                        affected_reservation_urls.add(entry.engine_url)
                    self._next_reservation_id = next_reservation_before
                    self._reservation_revision = revision_before
                    for session_id, session in sessions_before.items():
                        if session is None:
                            self.sessions.pop(session_id, None)
                        else:
                            self.sessions[session_id] = session
                    self._decisions = decisions_before
                    for url in affected_reservation_urls:
                        self._synchronize_reserved_load(url)
                    for step, _ in leases:
                        step.lease = None
                    raise
        for step, lease in leases:
            if not step.future.done():
                step.future.set_result(lease)
        for step, exc in failures:
            if not step.future.done():
                step.future.set_exception(exc)

    @staticmethod
    def _batch_load_score(
        engine: EngineBaseline,
        *,
        queue_increment: float = 0.0,
        token_increment: float = 0.0,
    ) -> LoadScore:
        request_pressure = engine.base_requests / engine.request_capacity
        token_pressure = max(
            (engine.base_tokens + token_increment) / engine.token_capacity,
            engine.token_usage,
        )
        queue_pressure = (
            engine.base_queue + queue_increment
        ) / engine.request_capacity
        return LoadScore(
            request_pressure=request_pressure,
            token_pressure=token_pressure,
            queue_pressure=queue_pressure,
            total=request_pressure + token_pressure + queue_pressure,
        )

    def _batch_trace(
        self,
        batch: _OpenBatch,
        fetch_results: tuple[_BatchFetchResult, ...],
        *,
        leases: list[tuple[_PendingBatchStep, RoutingLease]],
        failures: list[tuple[_PendingBatchStep, Exception]],
        wait_previous_seconds: float,
        fetch_seconds: float,
        completed_at: float,
        fallback_reason: str | None = None,
        frozen: _FrozenBatch | None = None,
        solved: _SolvedBatch | None = None,
    ) -> dict[str, Any]:
        lease_by_arrival = {step.arrival_id: lease for step, lease in leases}
        failed_arrivals = {step.arrival_id for step, _ in failures}
        frozen_by_arrival = {
            step.pending.arrival_id: step
            for step in (() if frozen is None else frozen.steps)
        }
        solved_assignment = {} if solved is None else dict(solved.assignment)

        def frozen_trace_inputs(
            step: _PendingBatchStep,
        ) -> tuple[
            str | None,
            int | None,
            list[str],
            dict[str, int],
        ]:
            frozen_step = frozen_by_arrival.get(step.arrival_id)
            if frozen_step is None:
                return (
                    None,
                    None,
                    [
                        result.url
                        for result in fetch_results
                        if result.status == "ok"
                    ],
                    {},
                )
            candidate_urls = (
                [frozen_step.fixed_target]
                if frozen_step.fixed_target is not None
                else [edge.engine_url for edge in frozen_step.edges]
            )
            candidate_costs = (
                {frozen_step.fixed_target: 0}
                if frozen_step.fixed_target is not None
                else {
                    edge.engine_url: edge.migration_cost_tokens
                    for edge in frozen_step.edges
                }
            )
            target = frozen_step.fixed_target or solved_assignment.get(
                step.session_id
            )
            budgets = dict(frozen_step.budgets)
            budget = budgets.get(target)
            if budget is None and len(budgets) == 1:
                budget = next(iter(budgets.values()))
            return (
                frozen_step.source,
                None if budget is None else budget.estimated_step_output_tokens,
                candidate_urls,
                candidate_costs,
            )

        def committed_edge(step: _PendingBatchStep) -> FeasibleEdge | None:
            lease = lease_by_arrival.get(step.arrival_id)
            frozen_step = frozen_by_arrival.get(step.arrival_id)
            if lease is None or frozen_step is None:
                return None
            return next(
                (
                    edge
                    for edge in frozen_step.edges
                    if edge.engine_url == lease.worker_url
                ),
                None,
            )

        def committed_token_increment(step: _PendingBatchStep) -> int:
            lease = lease_by_arrival.get(step.arrival_id)
            frozen_step = frozen_by_arrival.get(step.arrival_id)
            if lease is None or frozen_step is None:
                return 0
            edge = committed_edge(step)
            if edge is not None:
                return int(edge.token_increment)
            base_tokens = dict(frozen_step.base_tokens).get(lease.worker_url, 0)
            return max(0, len(step.input_ids) - base_tokens)

        def committed_migration_cost(step: _PendingBatchStep) -> int:
            edge = committed_edge(step)
            return 0 if edge is None else edge.migration_cost_tokens

        trace_inputs = {
            step.arrival_id: frozen_trace_inputs(step) for step in batch.steps
        }
        solved_count = (
            len(leases)
            if frozen is None
            else sum(
                not step.pending.cancelled
                and step.failure is None
                and (step.fixed_target is not None or bool(step.edges))
                for step in frozen.steps
            )
        )
        return {
            "batch": {
                "id": batch.id,
                "completed_at": time.time(),
                "registered_count": len(batch.steps),
                "solved_count": solved_count,
                "committed_count": len(leases),
                "failed_count": len(failures),
                "cancelled_count": sum(step.cancelled for step in batch.steps),
                "wait_for_previous_seconds": wait_previous_seconds,
                "collect_seconds": max(
                    0.0,
                    (batch.sealed_monotonic or completed_at)
                    - batch.started_monotonic,
                ),
                "fetch_seconds": fetch_seconds,
                "solve_seconds": 0.0 if solved is None else solved.elapsed_seconds,
                "total_seconds": max(0.0, completed_at - batch.started_monotonic),
            },
            "steps": [
                {
                    "arrival_id": step.arrival_id,
                    "session_id": step.session_id,
                    "source": (
                        trace_inputs[step.arrival_id][0]
                        if step.arrival_id not in lease_by_arrival
                        else lease_by_arrival[
                            step.arrival_id
                        ].decision.source_worker_url
                    ),
                    "prompt_token_count": len(step.input_ids),
                    "estimated_output": (
                        trace_inputs[step.arrival_id][1]
                        if step.arrival_id not in lease_by_arrival
                        else lease_by_arrival[step.arrival_id].expected_output_tokens
                    ),
                    "candidate_urls": trace_inputs[step.arrival_id][2],
                    "candidate_migration_cost_tokens": trace_inputs[
                        step.arrival_id
                    ][3],
                    "status": (
                        "cancelled"
                        if step.cancelled
                        else (
                            "failed"
                            if step.arrival_id in failed_arrivals
                            else "committed"
                        )
                    ),
                    "target": (
                        None
                        if step.arrival_id not in lease_by_arrival
                        else lease_by_arrival[step.arrival_id].worker_url
                    ),
                    "moved": (
                        False
                        if step.arrival_id not in lease_by_arrival
                        else lease_by_arrival[step.arrival_id].decision.moved
                    ),
                    "queue_increment": (
                        0 if step.arrival_id not in lease_by_arrival else 1
                    ),
                    "token_increment": committed_token_increment(step),
                    "reserved_requests": (
                        0 if step.arrival_id not in lease_by_arrival else 1
                    ),
                    "reserved_tokens": (
                        0
                        if step.arrival_id not in lease_by_arrival
                        else lease_by_arrival[
                            step.arrival_id
                        ].reserved_tokens
                    ),
                    "reserved_prefill_tokens": (
                        0
                        if step.arrival_id not in lease_by_arrival
                        else lease_by_arrival[
                            step.arrival_id
                        ].reserved_prefill_tokens
                    ),
                    "migration_cost_tokens": committed_migration_cost(step),
                }
                for step in sorted(batch.steps, key=lambda item: item.arrival_id)
            ],
            "engines": [
                asdict(engine)
                for engine in (
                    frozen.engine_traces
                    if frozen is not None
                    else tuple(
                        _FrozenBatchEngineTrace(
                            url=result.url,
                            fetch_status=result.status,
                            fetch_duration_seconds=result.duration_seconds,
                            health=False,
                            version=None,
                            fingerprint=None,
                            row_count=result.row_count,
                            running=None,
                            active_tokens=None,
                            request_capacity=None,
                            token_capacity=None,
                            token_usage=None,
                            gen_throughput=None,
                            queued=None,
                            queue_pressure=None,
                            waiting_uncached_tokens=None,
                            live_ledger_requests=0,
                            live_ledger_tokens=0,
                            live_ledger_prefill=0,
                            base_requests=None,
                            base_tokens=None,
                            base_queue=None,
                        )
                        for result in fetch_results
                    )
                )
            ],
            "sticky": (
                None
                if solved is None or solved.sticky is None
                else {
                    "status": solved.sticky.status.value,
                    "maximum_load": solved.sticky.maximum_load,
                    "minimum_load": solved.sticky.minimum_load,
                    "load_range": solved.sticky.load_range,
                    "migration_cost_tokens": (
                        solved.sticky.total_migration_cost_tokens
                    ),
                    "migrations": solved.sticky.voluntary_migrations,
                    "elapsed_seconds": solved.sticky.elapsed_seconds,
                }
            ),
            "optimized": (
                None
                if solved is None or solved.optimized is None
                else {
                    "status": solved.optimized.status.value,
                    "maximum_load": solved.optimized.maximum_load,
                    "minimum_load": solved.optimized.minimum_load,
                    "load_range": solved.optimized.load_range,
                    "migration_cost_tokens": (
                        solved.optimized.total_migration_cost_tokens
                    ),
                    "migrations": solved.optimized.voluntary_migrations,
                    "elapsed_seconds": solved.optimized.elapsed_seconds,
                }
            ),
            "improvement_ratio": (
                None if solved is None else solved.improvement_ratio
            ),
            "target_maximum_load": (
                None if solved is None else solved.target_maximum_load
            ),
            "required_ratio": self.config.min_load_improvement_ratio,
            "adopted_plan": (
                "failure" if solved is None else solved.adopted
            ),
            "fallback_reason": (
                fallback_reason
                if solved is None
                else solved.fallback_reason or fallback_reason
            ),
        }

    async def _fail_batch(
        self,
        batch: _OpenBatch,
        exc: Exception,
        *,
        fetch_results: tuple[_BatchFetchResult, ...] = (),
        frozen: _FrozenBatch | None = None,
    ) -> None:
        async with self._batch_lock:
            async with self._lock:
                if batch.sealed_monotonic is None:
                    batch.sealed_monotonic = time.monotonic()
                if self._open_batch is batch:
                    self._open_batch = None
                failures = [
                    (step, exc) for step in batch.steps if not step.cancelled
                ]
                if not batch.terminal_published:
                    self._load_batch_history.record(
                        LoadBatchTrace(
                            self._batch_trace(
                                batch,
                                fetch_results,
                                leases=[],
                                failures=failures,
                                wait_previous_seconds=0.0,
                                fetch_seconds=0.0,
                                completed_at=time.monotonic(),
                                fallback_reason="runner_failure",
                                frozen=frozen,
                            )
                        )
                    )
                    batch.terminal_published = True
        for step, _ in failures:
            if not step.future.done():
                step.future.set_exception(exc)

    def _projected_load_score(
        self,
        url: str,
        *,
        token_increment: int,
        capacity_fallback: tuple[int, int] | None = None,
    ) -> float:
        return self._load_score(
            url,
            token_increment=token_increment,
            include_pending_request=True,
            capacity_fallback=capacity_fallback,
        ).total

    def _load_score(
        self,
        url: str,
        *,
        token_increment: int,
        include_pending_request: bool,
        capacity_fallback: tuple[int, int] | None = None,
    ) -> LoadScore:
        load = self.loads[url]
        if capacity_fallback is None:
            compatible = [
                candidate
                for candidate, deployment in self.deployments.items()
                if deployment.cache_fingerprint
                == self.deployments[url].cache_fingerprint
                and candidate in self.loads
            ]
            max_running = max(
                [self.loads[item].request_capacity for item in compatible] + [1]
            )
            max_tokens = max(
                [self.loads[item].token_capacity for item in compatible] + [1]
            )
        else:
            max_running, max_tokens = capacity_fallback
        req_capacity = load.request_capacity or max_running
        token_capacity = load.token_capacity or max_tokens
        queue_increment = 1 if include_pending_request else 0
        pending_tokens = (
            max(0, int(token_increment)) if include_pending_request else 0
        )
        request_pressure = load.running / max(1, req_capacity)
        token_pressure = max(
            (load.active_tokens + pending_tokens) / max(1, token_capacity),
            load.token_usage,
        )
        queue_pressure = (load.queued + queue_increment) / max(1, req_capacity)
        return LoadScore(
            request_pressure=request_pressure,
            token_pressure=token_pressure,
            queue_pressure=queue_pressure,
            total=request_pressure + token_pressure + queue_pressure,
        )

    def _reserve(
        self,
        decision: RoutingDecision,
        *,
        input_ids: list[int],
        base_tokens: int,
        budget: StepGenerationBudget,
        batch_id: int | None = None,
        prefill_increment: int | None = None,
        projected_load_score: float | None = None,
    ) -> RoutingLease:
        target = decision.target_worker_url
        expected_output_tokens = budget.estimated_step_output_tokens or 0
        reserved_tokens = len(input_ids) + expected_output_tokens
        reserved_prefill_tokens = 0
        prefill_reservation_generation = None
        reservation_id = None
        if target is not None:
            selected_context = None
            if (
                target == decision.source_worker_url
                and decision.source_context is not None
            ):
                selected_context = decision.source_context
            elif (
                target == decision.target_worker_url
                and decision.target_context is not None
            ):
                selected_context = decision.target_context
            if prefill_increment is not None:
                reserved_prefill_tokens = max(0, int(prefill_increment))
            elif selected_context is not None:
                reserved_prefill_tokens = max(
                    0, int(selected_context.expected_prefill_tokens)
                )
            elif (
                decision.source_worker_url is not None
                and target == decision.source_worker_url
            ):
                reserved_prefill_tokens = max(0, len(input_ids) - base_tokens)
            else:
                reserved_prefill_tokens = len(input_ids)
            if projected_load_score is None:
                token_increment = (
                    max(0, len(input_ids) - base_tokens)
                    if target == decision.source_worker_url
                    else len(input_ids)
                )
                projected_load_score = self._projected_load_score(
                    target,
                    token_increment=token_increment,
                )
            if reserved_prefill_tokens > 0:
                prefill_reservation_generation = self._load_generations[target] + 1
            reservation_id = self._next_reservation_id
            self._next_reservation_id += 1
            self._reservations[reservation_id] = _ReservationEntry(
                engine_url=target,
                request_increment=1,
                token_increment=reserved_tokens,
                prefill_increment=reserved_prefill_tokens,
                prefill_reservation_generation=prefill_reservation_generation,
                prefill_active=reserved_prefill_tokens > 0,
            )
            self._reservation_revision += 1
            self._synchronize_reserved_load(target)
        self._decisions.append(decision.snapshot())
        return RoutingLease(
            decision=decision,
            worker_url=target,
            reserved_tokens=reserved_tokens,
            base_tokens=base_tokens,
            started_monotonic=time.monotonic(),
            projected_load_score=projected_load_score,
            context_tokens=len(input_ids),
            expected_output_tokens=expected_output_tokens,
            reserved_prefill_tokens=reserved_prefill_tokens,
            prefill_reservation_generation=prefill_reservation_generation,
            reservation_id=reservation_id,
            batch_id=batch_id,
        )

    async def _record_completion_observation(
        self, completion: _CompletionObservation
    ) -> None:
        lease = completion.lease
        queue_raw = completion.response_meta.get("queue_time")
        try:
            queue_seconds = (
                None if queue_raw is None else max(0.0, float(queue_raw))
            )
        except (TypeError, ValueError):
            queue_seconds = None
        decode_throughput_raw = completion.response_meta.get("decode_throughput")
        try:
            decode_throughput = float(decode_throughput_raw)
        except (TypeError, ValueError):
            decode_throughput = None
        decode_seconds = (
            max(0, completion.output_tokens - 1) / decode_throughput
            if decode_throughput is not None and decode_throughput > 0
            else 0.0
        )
        e2e = completion.response_meta.get("e2e_latency")
        try:
            e2e_seconds = float(e2e)
        except (TypeError, ValueError):
            e2e_seconds = completion.elapsed_seconds
        context_seconds = (
            None
            if queue_seconds is None
            else max(0.0, e2e_seconds - queue_seconds - decode_seconds)
        )
        cached_raw = completion.response_meta.get("cached_tokens") or 0
        try:
            cached_tokens = int(cached_raw)
        except (TypeError, ValueError):
            cached_tokens = 0
        if completion.engine_url == lease.decision.source_worker_url:
            estimate = lease.decision.source_context
            predicted_queue_seconds = lease.decision.source_queue_seconds
        elif completion.engine_url == lease.decision.target_worker_url:
            estimate = lease.decision.target_context
            predicted_queue_seconds = lease.decision.target_queue_seconds
        else:
            estimate = None
            predicted_queue_seconds = None
        # The native cached-token result determines the observed path even
        # when the Proxy conservatively predicted a full prefill.
        if cached_tokens <= 0:
            source = CacheSource.NONE
        elif (
            completion.old_owner == completion.engine_url
            or completion.target_seen_before
        ):
            source = CacheSource.LOCAL
        elif completion.shared_l3:
            source = CacheSource.MOONCAKE
        else:
            source = CacheSource.LOCAL

        async with self._lock:
            performance_observation = self.performance.observe(
                fingerprint=completion.fingerprint,
                engine_url=completion.engine_url,
                running=completion.running,
                context_tokens=completion.context_tokens,
                queue_seconds=queue_seconds,
                context_seconds=context_seconds,
                cached_tokens=cached_tokens,
                output_tokens=completion.output_tokens,
                decode_throughput=decode_throughput,
                projected_load_score=lease.projected_load_score,
                predicted_queue_seconds=predicted_queue_seconds,
                estimated_context_seconds=(
                    None if estimate is None else estimate.estimated_seconds
                ),
                cache_source=source,
            )
            runtime_key = (
                None
                if completion.old_owner is None
                else (
                    completion.fingerprint,
                    completion.old_owner,
                    completion.engine_url,
                    context_bucket(completion.context_tokens),
                )
            )
            if (
                estimate is not None
                and estimate.cache_source is CacheSource.MOONCAKE
                and performance_observation.context_seconds is not None
                and runtime_key is not None
            ):
                self._runtime_restore_errors[runtime_key].append(
                    abs(
                        estimate.estimated_seconds
                        - performance_observation.context_seconds
                    )
                )
            self.cache_hits.observe(
                fingerprint=completion.fingerprint,
                engine_url=completion.engine_url,
                cache_source=source,
                estimated_base_tokens=lease.base_tokens,
                actual_cached_tokens=performance_observation.cached_tokens,
                context_tokens=completion.context_tokens,
            )
            prefill_throughput = self.performance.prefill_throughput(
                fingerprint=completion.fingerprint,
                engine_url=completion.engine_url,
                context_tokens=completion.context_tokens,
            )
            actual_prefill_tokens = max(
                0,
                completion.context_tokens - performance_observation.cached_tokens,
            )
            restore_seconds_actual = None
            restore_throughput = None
            if (
                source is not CacheSource.NONE
                and runtime_key is not None
                and performance_observation.cached_tokens > 0
                and performance_observation.context_seconds is not None
                and prefill_throughput is not None
            ):
                restore_seconds_actual = max(
                    0.0,
                    performance_observation.context_seconds
                    - actual_prefill_tokens / prefill_throughput,
                )
                self._runtime_restore_seconds[runtime_key].append(
                    restore_seconds_actual
                )
                profile = self.profiles.get(completion.fingerprint)
                if profile is not None and restore_seconds_actual > 0:
                    restore_throughput = (
                        profile.estimate_bytes(performance_observation.cached_tokens)
                        / restore_seconds_actual
                    )
                    self._runtime_restore_throughputs[runtime_key].append(
                        restore_throughput
                    )
            self._observations.append(
                {
                    "session_id": lease.decision.session_id,
                    "engine_url": completion.engine_url,
                    "cache_source": source.value,
                    "expected_cached_tokens": (
                        None if estimate is None else estimate.expected_cached_tokens
                    ),
                    "actual_cached_tokens": performance_observation.cached_tokens,
                    "expected_prefill_tokens": (
                        None if estimate is None else estimate.expected_prefill_tokens
                    ),
                    "actual_prefill_tokens": actual_prefill_tokens,
                    "source_context_seconds": (
                        None
                        if lease.decision.source_context is None
                        else lease.decision.source_context.estimated_seconds
                    ),
                    "target_context_seconds": (
                        None
                        if lease.decision.target_context is None
                        else lease.decision.target_context.estimated_seconds
                    ),
                    "predicted_queue_seconds": (
                        performance_observation.predicted_queue_seconds
                    ),
                    "actual_queue_seconds": performance_observation.queue_seconds,
                    "queue_prediction_error_seconds": (
                        performance_observation.queue_prediction_error_seconds
                    ),
                    "actual_context_seconds": (
                        performance_observation.context_seconds
                    ),
                    "restore_seconds": restore_seconds_actual,
                    "restore_throughput": restore_throughput,
                    "prefill_throughput": prefill_throughput,
                    "hit_probability": (
                        None if estimate is None else estimate.hit_probability
                    ),
                    "queue_risk_seconds": lease.decision.queue_risk_seconds,
                    "context_risk_seconds": lease.decision.context_risk_seconds,
                    "decision_risk": lease.decision.decision_risk_seconds,
                    "step_max_tokens_source": lease.decision.step_max_tokens_source,
                    "effective_step_max_tokens": (
                        lease.decision.effective_step_max_tokens
                    ),
                    "historical_step_tokens_p75": (
                        lease.decision.historical_step_tokens_p75
                    ),
                    "group_remaining_tokens": lease.decision.group_remaining_tokens,
                    "estimated_step_output_tokens": (
                        lease.decision.estimated_step_output_tokens
                    ),
                    "source_decode_seconds": lease.decision.source_decode_seconds,
                    "target_decode_seconds": lease.decision.target_decode_seconds,
                }
            )
            state = self.pools.get(completion.fingerprint)
            if state is not None:
                state.update(
                    self._pool_readiness(
                        completion.fingerprint,
                        now=time.monotonic(),
                    )
                )
            self._record_successful_online_request()

    async def complete(
        self,
        lease: RoutingLease,
        *,
        response_meta: Mapping[str, Any],
        output_tokens: int,
        committed_tokens: list[int],
        success: bool = True,
    ) -> None:
        if not self.config.enabled:
            return
        completion: _CompletionObservation | None = None
        async with self._lock:
            self._release_reservation(lease.reservation_id)
            if lease.worker_url is None:
                return
            context_tokens = (
                lease.context_tokens
                if lease.context_tokens > 0
                else lease.reserved_tokens
            )
            load = self.loads.get(lease.worker_url)
            session = self.sessions.get(lease.decision.session_id)
            if session is None:
                return
            if not success:
                session.pending_owner_worker_url = None
                return
            old_owner = session.owner_worker_url
            new_owner = lease.worker_url
            target_seen_before = new_owner in session.seen_engines
            session.owner_worker_url = new_owner
            session.pending_owner_worker_url = None
            deployment = self.deployments[new_owner]
            session.fingerprint = deployment.cache_fingerprint
            session.seen_engines.add(new_owner)
            session.previous_committed_tokens = list(committed_tokens)
            session.generated_tokens += max(0, int(output_tokens))
            self.step_lengths.observe(
                fingerprint=deployment.cache_fingerprint,
                task_key=session.task_key,
                max_tokens=lease.decision.effective_step_max_tokens,
                output_tokens=output_tokens,
            )
            completion = _CompletionObservation(
                lease=lease,
                engine_url=new_owner,
                fingerprint=deployment.cache_fingerprint,
                shared_l3=deployment.shared_l3,
                old_owner=old_owner,
                target_seen_before=target_seen_before,
                running=load.running if load is not None else 0,
                context_tokens=context_tokens,
                response_meta=dict(response_meta),
                output_tokens=output_tokens,
                elapsed_seconds=max(
                    0.0,
                    time.monotonic() - lease.started_monotonic,
                ),
            )
        self._schedule_completion_observation(completion)

    async def fail(self, lease: RoutingLease) -> None:
        await self.complete(
            lease,
            response_meta={},
            output_tokens=0,
            committed_tokens=[],
            success=False,
        )

    async def register_session_context(
        self,
        *,
        session_id: str,
        group_id: int | str | None,
        group_size: int,
        task_key: str | None,
        default_step_max_tokens: int | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        async with self._lock:
            state = self.sessions.setdefault(session_id, SessionRoutingState())
            state.group_id = group_id
            state.group_size = max(1, int(group_size))
            state.task_key = task_key
            state.default_step_max_tokens = (
                None
                if default_step_max_tokens is None
                else max(1, int(default_step_max_tokens))
            )

    async def discard_session_context(self, session_id: str) -> None:
        async with self._lock:
            self.sessions.pop(session_id, None)

    async def finalize_session(self, session_id: str) -> None:
        async with self._lock:
            state = self.sessions.pop(session_id, None)
            if state is None:
                return
            self.group_lengths.observe(
                group_id=state.group_id,
                task_key=state.task_key,
                final_length=state.generated_tokens,
            )

    def _runtime_calibration_snapshot(self) -> dict[str, Any]:
        keys = sorted(
            set(self._runtime_restore_seconds)
            | set(self._runtime_restore_throughputs)
            | set(self._runtime_restore_errors)
        )
        results: list[dict[str, Any]] = []
        for fingerprint, source_engine, target_engine, bucket in keys:
            key = (fingerprint, source_engine, target_engine, bucket)
            restore_samples = list(self._runtime_restore_seconds.get(key, ()))
            throughput_samples = list(self._runtime_restore_throughputs.get(key, ()))
            error_samples = list(self._runtime_restore_errors.get(key, ()))
            model_ready = len(restore_samples) >= self.config.min_samples
            results.append(
                {
                    "cache_fingerprint": fingerprint,
                    "source_engine": source_engine,
                    "target_engine": target_engine,
                    "context_bucket": bucket,
                    "restore_sample_count": len(restore_samples),
                    "restore_seconds_p75": (
                        None
                        if not restore_samples
                        else percentile(restore_samples, 0.75)
                    ),
                    "restore_throughput_sample_count": len(throughput_samples),
                    "restore_throughput_bytes_per_second_p25": (
                        None
                        if not throughput_samples
                        else percentile(throughput_samples, 0.25)
                    ),
                    "prediction_error_sample_count": len(error_samples),
                    "prediction_error_seconds_p90": (
                        None if not error_samples else percentile(error_samples, 0.90)
                    ),
                    "model_ready": model_ready,
                    "effective_source": "runtime" if model_ready else "offline",
                }
            )
        return {
            "min_samples": self.config.min_samples,
            "results": results,
        }

    def calibration_snapshot(self) -> dict[str, Any]:
        payload = self.calibrator.snapshot()
        plan = self._preflight_plan
        progress = [] if plan is None else self.calibrator.plan_progress(plan)
        payload.update(
            {
                "deployment_configured": (self.machine_calibration_config is not None),
                "preflight_nodes": sorted(self._preflight_node_ids),
                "preflight_node_addresses": sorted(self._preflight_node_addresses),
                "preflight_node_aliases": dict(self._preflight_node_aliases),
                "plan": (
                    None
                    if plan is None
                    else {
                        "fingerprint": plan.fingerprint,
                        "task_count": len(plan.tasks),
                        "tasks": progress,
                        "completed_links": sorted(
                            f"{task['source_node']}->{task['target_node']}:"
                            f"{task['link_type']}"
                            for task in progress
                            if not task["pending_payloads"]
                        ),
                        "pending_links": sorted(
                            f"{task['source_node']}->{task['target_node']}:"
                            f"{task['link_type']}"
                            for task in progress
                            if task["pending_payloads"]
                        ),
                        "skipped_links": dict(plan.skipped_links),
                        "complete": self.calibrator.plan_complete(plan),
                    }
                ),
            }
        )
        return payload

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            now = time.monotonic()
            stale = self.config.metrics_stale_ms / 1000.0
            paths = []
            for source, source_deployment in self.deployments.items():
                for target, target_deployment in self.deployments.items():
                    if (
                        source == target
                        or source_deployment.cache_fingerprint
                        != target_deployment.cache_fingerprint
                    ):
                        continue
                    readiness = self._path_readiness(source, target)
                    payload = asdict(readiness)
                    payload["cache_source"] = readiness.cache_source.value
                    paths.append(payload)
            return {
                "enabled": self.config.enabled,
                "state": (None if self.config.enabled else SchedulerState.OFF.value),
                "effective_config": self.config.snapshot(),
                "compatibility_pools": [
                    state.snapshot() for state in self.pools.values()
                ],
                "engines": [
                    load.snapshot(now=now, stale_seconds=stale)
                    for load in self.loads.values()
                ],
                "deployments": [asdict(item) for item in self.deployments.values()],
                "excluded_engines": dict(self.excluded_engines),
                "path_readiness": paths,
                "model_cache_profiles": [
                    profile.snapshot() for profile in self.profiles.values()
                ],
                "performance_models": self.performance.snapshot(),
                "runtime_restore_model": {
                    "latency_samples": sum(
                        len(values) for values in self._runtime_restore_seconds.values()
                    ),
                    "prediction_error_samples": sum(
                        len(values) for values in self._runtime_restore_errors.values()
                    ),
                    "min_samples": self.config.min_samples,
                },
                "step_length_model": self.step_lengths.snapshot(),
                "cache_hit_model": self.cache_hits.snapshot(),
                "calibration": self.calibration_snapshot(),
                "recent_decisions": list(self._decisions),
                "recent_context_observations": list(self._observations),
                "recent_load_batches": self._load_batch_history.snapshot(),
                "active_sessions": len(self.sessions),
            }
