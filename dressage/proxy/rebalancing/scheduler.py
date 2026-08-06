"""Proxy-side engine placement and turn-boundary rebalancing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import socket
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Deque, Mapping
from urllib.parse import urlsplit

from .cache_hit_estimator import (
    CacheHitEstimator,
    CacheSource,
    ContextRecoveryEstimate,
    context_bucket,
    longest_common_prefix_length,
)
from .context_recovery_model import ContextRecoveryModel, PerformanceHistory, percentile
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
    reserved_requests: int = 0
    reserved_tokens: int = 0

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
    owner_turns: int = 0
    previous_owner_worker_url: str | None = None
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
class EngineStepEstimate:
    engine_url: str
    context: ContextRecoveryEstimate
    queue_seconds: float
    projected_running: int
    projected_load_score: float
    tpot_seconds: float | None
    prefill_throughput: float | None
    decode_seconds: float | None

    @property
    def context_queue_seconds(self) -> float:
        return self.queue_seconds + self.context.estimated_seconds


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
    stay_seconds: float | None = None
    move_seconds: float | None = None
    queue_risk_seconds: float = 0.0
    context_risk_seconds: float = 0.0
    decision_risk_seconds: float = 0.0
    source_decode_seconds: float | None = None
    target_decode_seconds: float | None = None
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


class GroupLengthEstimator:
    def __init__(self, *, history_size: int, min_task_samples: int = 32) -> None:
        self.history_size = history_size
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
        self.history_size = history_size
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
        self.context_model = ContextRecoveryModel(self.performance)
        self.calibrator = TransferCalibrator()
        self.group_lengths = GroupLengthEstimator(history_size=config.history_size)
        self.step_lengths = StepLengthEstimator(
            history_size=config.history_size,
            min_samples=config.min_samples,
        )
        self.deployments: dict[str, EngineDeploymentInfo] = {}
        self.loads: dict[str, EngineLoad] = {}
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
        self._final_snapshot_written = False
        self._lock = asyncio.Lock()
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
        ) -> tuple[str, dict[str, Any], EngineDeploymentInfo | None, bool]:
            try:
                load = await self.client.get_worker_loads(url)
            except Exception:
                logger.debug(
                    "failed to read SGLang load metrics from %s", url, exc_info=True
                )
                load = {}
            last_checked = self._deployment_checked_at.get(url, 0.0)
            refresh_deployment = (
                url not in self.deployments
                or inspection_started - last_checked >= self._deployment_refresh_seconds
            )
            if not refresh_deployment:
                return url, load, None, False
            info, version = await asyncio.gather(
                self.client.get_server_info(url),
                self.client.get_worker_weight_version(url),
                return_exceptions=True,
            )
            if isinstance(info, Exception) or isinstance(version, Exception):
                return url, load, None, False
            deployment = EngineDeploymentInfo.from_worker(
                worker_url=url,
                server_info=info,
                weight_version=version,
                model_id=self.model_id,
            )
            return url, load, deployment, True

        inspected = await asyncio.gather(*(inspect_worker(url) for url in healthy))
        now = time.monotonic()
        new_plans: list[tuple[str, CalibrationPlan]] = []
        async with self._lock:
            for url, load_payload, deployment, deployment_checked in inspected:
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
                self._update_load(url, load_payload, now=now)
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

    def _update_load(self, url: str, payload: Mapping[str, Any], *, now: float) -> None:
        rows = payload.get("loads") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            rows = []
        valid = [row for row in rows if isinstance(row, Mapping)]
        load = self.loads.setdefault(url, EngineLoad(worker_url=url))
        if not valid:
            load.healthy = True
            return
        load.healthy = True
        load.metrics_timestamp = now
        load.running = sum(int(row.get("num_running_reqs") or 0) for row in valid)
        load.queued = sum(
            int(row.get("num_waiting_reqs") or row.get("num_queue_reqs") or 0)
            for row in valid
        )
        load.active_tokens = sum(
            int(row.get("num_total_tokens") or row.get("num_used_tokens") or 0)
            for row in valid
        )
        load.token_capacity = sum(
            int(row.get("max_total_num_tokens") or 0) for row in valid
        )
        load.request_capacity = sum(
            int(row.get("max_running_requests") or 0) for row in valid
        )
        load.token_usage = max(float(row.get("token_usage") or 0.0) for row in valid)

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
                if self._path_readiness(source, target).ready:
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
        if target == session.owner_worker_url or target in session.seen_engines:
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
                skipped_links={"transfer": "target has Engine-local session history"},
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
            try:
                await self.refresh()
            except Exception:
                logger.warning("initial engine discovery failed", exc_info=True)

        async with self._lock:
            now = time.monotonic()
            session = self.sessions.setdefault(session_id, SessionRoutingState())
            healthy_urls = self._healthy_urls(now=now)
            if expected_version is not None:
                healthy_urls = [
                    url
                    for url in healthy_urls
                    if self.deployments[url].weight_version == str(expected_version)
                ]
            if not healthy_urls:
                suffix = (
                    ""
                    if expected_version is None
                    else f" at weight version {expected_version}"
                )
                raise RuntimeError(
                    f"engine rebalancing found no healthy SGLang workers{suffix}"
                )
            source = session.owner_worker_url
            if source is None:
                candidates = healthy_urls
                pool_state = None
            else:
                fingerprint = session.fingerprint
                if fingerprint is None:
                    deployment = self.deployments.get(source)
                    if deployment is None:
                        raise RuntimeError(
                            "session owner is missing deployment compatibility data"
                        )
                    fingerprint = deployment.cache_fingerprint
                pool_state = self.pools[fingerprint].state
                candidates = [
                    url
                    for url in healthy_urls
                    if self.deployments[url].cache_fingerprint == fingerprint
                ]
                if not candidates:
                    raise RuntimeError(
                        "no healthy SGLang worker matches the session cache fingerprint"
                    )

            try:
                decision, base_tokens, budget = self._select_step_engine(
                    session_id=session_id,
                    session=session,
                    candidates=candidates,
                    input_ids=input_ids,
                    step_max_new_tokens=step_max_new_tokens,
                    context_remaining_tokens=context_remaining_tokens,
                    pool_state=pool_state,
                )
            except Exception:
                # A prediction/modeling failure must never change generation
                # semantics. Keep a healthy sticky owner; otherwise place the
                # complete input on the least-loaded eligible Engine.
                logger.warning(
                    "engine rebalancing decision failed for session %s; using safe fallback",
                    session_id,
                    exc_info=True,
                )
                base_tokens = (
                    0
                    if source is None
                    else min(
                        len(session.previous_committed_tokens),
                        longest_common_prefix_length(
                            session.previous_committed_tokens,
                            input_ids,
                        ),
                    )
                )
                if source is not None and source in candidates:
                    target = source
                    fingerprint = self.deployments[source].cache_fingerprint
                    budget = self._resolve_step_budget(
                        session=session,
                        fingerprint=fingerprint,
                        step_max_new_tokens=step_max_new_tokens,
                        context_remaining_tokens=context_remaining_tokens,
                    )
                    decision = RoutingDecision(
                        session_id,
                        source,
                        source,
                        fingerprint,
                        pool_state or self.pools[fingerprint].state,
                        "prediction_error_keep_owner",
                        **budget.snapshot(),
                    )
                else:
                    target, budget = self._least_load(
                        candidates,
                        session_id=session_id,
                        session=session,
                        prompt_tokens=len(input_ids),
                        step_max_new_tokens=step_max_new_tokens,
                        context_remaining_tokens=context_remaining_tokens,
                    )
                    fingerprint = self.deployments[target].cache_fingerprint
                    decision = RoutingDecision(
                        session_id,
                        source,
                        target,
                        fingerprint,
                        pool_state or self.pools[fingerprint].state,
                        (
                            "new_session_projected_load_fallback"
                            if source is None
                            else "owner_unhealthy_failover"
                        ),
                        moved=source is not None and target != source,
                        **budget.snapshot(),
                    )
            if decision.target_worker_url != source:
                session.pending_owner_worker_url = decision.target_worker_url
            if source is None and decision.target_worker_url is not None:
                session.fingerprint = self.deployments[
                    decision.target_worker_url
                ].cache_fingerprint
            return self._reserve(
                decision,
                input_ids=input_ids,
                base_tokens=base_tokens,
                budget=budget,
            )

    def _select_step_engine(
        self,
        *,
        session_id: str,
        session: SessionRoutingState,
        candidates: list[str],
        input_ids: list[int],
        step_max_new_tokens: int | None,
        context_remaining_tokens: int | None,
        pool_state: SchedulerState | None,
    ) -> tuple[RoutingDecision, int, StepGenerationBudget]:
        if not candidates:
            raise RuntimeError("no eligible engine candidates")
        source = session.owner_worker_url
        base_tokens = (
            0
            if source is None
            else min(
                len(session.previous_committed_tokens),
                longest_common_prefix_length(
                    session.previous_committed_tokens,
                    input_ids,
                ),
            )
        )
        budgets = {
            target: self._resolve_step_budget(
                session=session,
                fingerprint=self.deployments[target].cache_fingerprint,
                step_max_new_tokens=step_max_new_tokens,
                context_remaining_tokens=context_remaining_tokens,
            )
            for target in candidates
        }

        if source is None:
            estimates: list[EngineStepEstimate] = []
            for target in candidates:
                estimate = self._estimate_step_candidate(
                    session=session,
                    source_engine=None,
                    target_engine=target,
                    input_ids=input_ids,
                    base_tokens=0,
                    budget=budgets[target],
                    cache_source=CacheSource.NONE,
                )
                if estimate is None:
                    return self._projected_load_selection(
                        session_id=session_id,
                        session=session,
                        source=None,
                        candidates=candidates,
                        input_ids=input_ids,
                        step_max_new_tokens=step_max_new_tokens,
                        context_remaining_tokens=context_remaining_tokens,
                        reason="new_session_projected_load_fallback",
                    )
                estimates.append(estimate)
            homogeneous = self._step_estimates_homogeneous(estimates)
            if not homogeneous and any(
                estimate.decode_seconds is None for estimate in estimates
            ):
                return self._projected_load_selection(
                    session_id=session_id,
                    session=session,
                    source=None,
                    candidates=candidates,
                    input_ids=input_ids,
                    step_max_new_tokens=step_max_new_tokens,
                    context_remaining_tokens=context_remaining_tokens,
                    reason="new_session_projected_load_fallback",
                )

            def new_session_key(
                estimate: EngineStepEstimate,
            ) -> tuple[float, float, str]:
                total = self._step_total_seconds(
                    estimate,
                    include_decode=not homogeneous,
                )
                tie = hashlib.sha256(
                    f"{session_id}\0{estimate.engine_url}".encode()
                ).hexdigest()
                return total, estimate.projected_load_score, tie

            selected = min(estimates, key=new_session_key)
            selected_total = self._step_total_seconds(
                selected,
                include_decode=not homogeneous,
            )
            budget = budgets[selected.engine_url]
            fingerprint = self.deployments[selected.engine_url].cache_fingerprint
            return (
                RoutingDecision(
                    session_id=session_id,
                    source_worker_url=None,
                    target_worker_url=selected.engine_url,
                    cache_fingerprint=fingerprint,
                    state=self.pools[fingerprint].state,
                    reason="new_session_estimated_completion",
                    target_context=selected.context,
                    target_queue_seconds=selected.queue_seconds,
                    move_seconds=selected_total,
                    target_decode_seconds=selected.decode_seconds,
                    **budget.snapshot(),
                ),
                0,
                budget,
            )

        fingerprint = self.deployments[source].cache_fingerprint
        source_healthy = source in candidates
        if not source_healthy:
            estimates = []
            for target in candidates:
                readiness = self._candidate_path_readiness(session, source, target)
                if not readiness.ready:
                    return self._projected_load_selection(
                        session_id=session_id,
                        session=session,
                        source=source,
                        candidates=candidates,
                        input_ids=input_ids,
                        step_max_new_tokens=step_max_new_tokens,
                        context_remaining_tokens=context_remaining_tokens,
                        reason="owner_unhealthy_failover",
                        base_tokens=base_tokens,
                    )
                estimate = self._estimate_step_candidate(
                    session=session,
                    source_engine=source,
                    target_engine=target,
                    input_ids=input_ids,
                    base_tokens=base_tokens,
                    budget=budgets[target],
                    cache_source=readiness.cache_source,
                )
                if estimate is None:
                    return self._projected_load_selection(
                        session_id=session_id,
                        session=session,
                        source=source,
                        candidates=candidates,
                        input_ids=input_ids,
                        step_max_new_tokens=step_max_new_tokens,
                        context_remaining_tokens=context_remaining_tokens,
                        reason="owner_unhealthy_failover",
                        base_tokens=base_tokens,
                    )
                estimates.append(estimate)
            homogeneous = self._step_estimates_homogeneous(estimates)
            if not homogeneous and any(
                estimate.decode_seconds is None for estimate in estimates
            ):
                return self._projected_load_selection(
                    session_id=session_id,
                    session=session,
                    source=source,
                    candidates=candidates,
                    input_ids=input_ids,
                    step_max_new_tokens=step_max_new_tokens,
                    context_remaining_tokens=context_remaining_tokens,
                    reason="owner_unhealthy_failover",
                    base_tokens=base_tokens,
                )
            selected = min(
                estimates,
                key=lambda estimate: (
                    self._step_total_seconds(
                        estimate,
                        include_decode=not homogeneous,
                    ),
                    estimate.projected_load_score,
                    hashlib.sha256(
                        f"{session_id}\0{estimate.engine_url}".encode()
                    ).hexdigest(),
                ),
            )
            selected_total = self._step_total_seconds(
                selected,
                include_decode=not homogeneous,
            )
            budget = budgets[selected.engine_url]
            return (
                RoutingDecision(
                    session_id=session_id,
                    source_worker_url=source,
                    target_worker_url=selected.engine_url,
                    cache_fingerprint=fingerprint,
                    state=pool_state or self.pools[fingerprint].state,
                    reason="owner_unhealthy_failover",
                    target_context=selected.context,
                    target_queue_seconds=selected.queue_seconds,
                    move_seconds=selected_total,
                    target_decode_seconds=selected.decode_seconds,
                    **budget.snapshot(),
                    moved=True,
                ),
                base_tokens,
                budget,
            )

        budget = budgets[source]
        if pool_state is not SchedulerState.ACTIVE:
            return (
                RoutingDecision(
                    session_id,
                    source,
                    source,
                    fingerprint,
                    pool_state or self.pools[fingerprint].state,
                    f"pool_state_{(pool_state or self.pools[fingerprint].state).value.lower()}",
                    **budget.snapshot(),
                ),
                base_tokens,
                budget,
            )

        source_estimate = self._estimate_step_candidate(
            session=session,
            source_engine=source,
            target_engine=source,
            input_ids=input_ids,
            base_tokens=base_tokens,
            budget=budget,
            cache_source=CacheSource.LOCAL,
        )
        if source_estimate is None:
            return (
                RoutingDecision(
                    session_id,
                    source,
                    source,
                    fingerprint,
                    SchedulerState.ACTIVE,
                    "source_model_not_ready",
                    **budget.snapshot(),
                ),
                base_tokens,
                budget,
            )

        best: RoutingDecision | None = None
        best_rejected: RoutingDecision | None = None
        for target in candidates:
            if target == source:
                continue
            readiness = self._candidate_path_readiness(session, source, target)
            if not readiness.ready:
                continue
            target_estimate = self._estimate_step_candidate(
                session=session,
                source_engine=source,
                target_engine=target,
                input_ids=input_ids,
                base_tokens=base_tokens,
                budget=budgets[target],
                cache_source=readiness.cache_source,
            )
            if target_estimate is None:
                continue
            context_risk = self._context_prediction_risk(
                fingerprint=fingerprint,
                source_engine=source,
                target_engine=source,
                estimate=source_estimate.context,
                context_tokens=len(input_ids),
            ) + self._context_prediction_risk(
                fingerprint=fingerprint,
                source_engine=source,
                target_engine=target,
                estimate=target_estimate.context,
                context_tokens=len(input_ids),
            )
            queue_risk = self.performance.queue_risk_seconds(
                fingerprint=fingerprint,
                engine_url=source,
                projected_running=source_estimate.projected_running,
                projected_load_score=source_estimate.projected_load_score,
            ) + self.performance.queue_risk_seconds(
                fingerprint=fingerprint,
                engine_url=target,
                projected_running=target_estimate.projected_running,
                projected_load_score=target_estimate.projected_load_score,
            )
            risk = max(
                self.config.min_risk_ms / 1000.0,
                context_risk + queue_risk,
            )
            homogeneous = self._within_ten_percent(
                source_estimate.tpot_seconds,
                target_estimate.tpot_seconds,
            ) and self._within_ten_percent(
                source_estimate.prefill_throughput,
                target_estimate.prefill_throughput,
            )
            if homogeneous:
                stay = source_estimate.context_queue_seconds
                move = target_estimate.context_queue_seconds + risk
            else:
                if (
                    source_estimate.decode_seconds is None
                    or target_estimate.decode_seconds is None
                ):
                    continue
                stay = self._step_total_seconds(
                    source_estimate,
                    include_decode=True,
                )
                move = (
                    self._step_total_seconds(
                        target_estimate,
                        include_decode=True,
                    )
                    + risk
                )
            beneficial = move < stay

            other_source_work = (
                self.loads[source].running
                + self.loads[source].queued
                + self.loads[source].reserved_requests
                > 0
            )
            if (
                not other_source_work
                or session.owner_turns < self.config.min_hold_turns
            ):
                beneficial = False
            if target == session.previous_owner_worker_url and beneficial:
                beneficial = stay - move > 2.0 * risk
            candidate = RoutingDecision(
                session_id=session_id,
                source_worker_url=source,
                target_worker_url=target if beneficial else source,
                cache_fingerprint=fingerprint,
                state=SchedulerState.ACTIVE,
                reason="context_benefit" if beneficial else "benefit_below_threshold",
                source_context=source_estimate.context,
                target_context=target_estimate.context,
                source_queue_seconds=source_estimate.queue_seconds,
                target_queue_seconds=target_estimate.queue_seconds,
                stay_seconds=stay,
                move_seconds=move,
                queue_risk_seconds=queue_risk,
                context_risk_seconds=context_risk,
                decision_risk_seconds=risk,
                source_decode_seconds=source_estimate.decode_seconds,
                target_decode_seconds=target_estimate.decode_seconds,
                **budgets[target].snapshot(),
                moved=beneficial,
            )
            if beneficial and (
                best is None
                or (candidate.move_seconds or math.inf)
                < (best.move_seconds or math.inf)
            ):
                best = candidate
            elif not beneficial and (
                best_rejected is None
                or (candidate.move_seconds or math.inf)
                < (best_rejected.move_seconds or math.inf)
            ):
                # Keep the most competitive rejected path so diagnostics show
                # which queue/context risk terms prevented the migration.
                best_rejected = candidate
        if best is not None:
            return best, base_tokens, budgets[best.target_worker_url or source]
        if best_rejected is not None:
            return best_rejected, base_tokens, budget
        return (
            RoutingDecision(
                session_id,
                source,
                source,
                fingerprint,
                SchedulerState.ACTIVE,
                "no_beneficial_target",
                source_context=source_estimate.context,
                source_queue_seconds=source_estimate.queue_seconds,
                **budget.snapshot(),
            ),
            base_tokens,
            budget,
        )

    def _estimate_step_candidate(
        self,
        *,
        session: SessionRoutingState,
        source_engine: str | None,
        target_engine: str,
        input_ids: list[int],
        base_tokens: int,
        budget: StepGenerationBudget,
        cache_source: CacheSource,
    ) -> EngineStepEstimate | None:
        fingerprint = self.deployments[target_engine].cache_fingerprint
        context = self._estimate_context(
            session=session,
            source_engine=source_engine,
            target_engine=target_engine,
            context_tokens=len(input_ids),
            base_tokens=base_tokens,
            cache_source=cache_source,
        )
        projected_running = (
            self.loads[target_engine].running
            + self.loads[target_engine].reserved_requests
            + 1
        )
        projected_score = self._projected_load_score(
            target_engine,
            prompt_tokens=len(input_ids),
            expected_output_tokens=budget.estimated_step_output_tokens or 0,
        )
        queue = self.performance.queue_seconds(
            fingerprint=fingerprint,
            engine_url=target_engine,
            projected_running=projected_running,
            projected_load_score=projected_score,
        )
        if context is None or queue is None:
            return None
        tpot = self.performance.tpot_seconds(
            fingerprint=fingerprint,
            engine_url=target_engine,
            projected_running=projected_running,
            projected_load_score=projected_score,
        )
        prefill = self.performance.prefill_throughput(
            fingerprint=fingerprint,
            engine_url=target_engine,
            context_tokens=len(input_ids),
        )
        decode = (
            None
            if tpot is None or budget.estimated_step_output_tokens is None
            else budget.estimated_step_output_tokens * tpot
        )
        return EngineStepEstimate(
            engine_url=target_engine,
            context=context,
            queue_seconds=queue,
            projected_running=projected_running,
            projected_load_score=projected_score,
            tpot_seconds=tpot,
            prefill_throughput=prefill,
            decode_seconds=decode,
        )

    def _projected_load_selection(
        self,
        *,
        session_id: str,
        session: SessionRoutingState,
        source: str | None,
        candidates: list[str],
        input_ids: list[int],
        step_max_new_tokens: int | None,
        context_remaining_tokens: int | None,
        reason: str,
        base_tokens: int = 0,
    ) -> tuple[RoutingDecision, int, StepGenerationBudget]:
        target, budget = self._least_load(
            candidates,
            session_id=session_id,
            session=session,
            prompt_tokens=len(input_ids),
            step_max_new_tokens=step_max_new_tokens,
            context_remaining_tokens=context_remaining_tokens,
        )
        fingerprint = self.deployments[target].cache_fingerprint
        return (
            RoutingDecision(
                session_id=session_id,
                source_worker_url=source,
                target_worker_url=target,
                cache_fingerprint=fingerprint,
                state=self.pools[fingerprint].state,
                reason=reason,
                moved=source is not None and target != source,
                **budget.snapshot(),
            ),
            base_tokens,
            budget,
        )

    @classmethod
    def _step_estimates_homogeneous(
        cls,
        estimates: list[EngineStepEstimate],
    ) -> bool:
        reference = estimates[0]
        return all(
            cls._within_ten_percent(
                reference.tpot_seconds,
                estimate.tpot_seconds,
            )
            and cls._within_ten_percent(
                reference.prefill_throughput,
                estimate.prefill_throughput,
            )
            for estimate in estimates[1:]
        )

    @staticmethod
    def _step_total_seconds(
        estimate: EngineStepEstimate,
        *,
        include_decode: bool,
    ) -> float:
        total = estimate.context_queue_seconds
        if include_decode and estimate.decode_seconds is not None:
            total += estimate.decode_seconds
        return total

    @staticmethod
    def _within_ten_percent(left: float | None, right: float | None) -> bool:
        if left is None or right is None:
            return True
        denominator = max(abs(left), abs(right), 1e-9)
        return abs(left - right) / denominator <= 0.10

    def _estimate_context(
        self,
        *,
        session: SessionRoutingState,
        source_engine: str | None,
        target_engine: str,
        context_tokens: int,
        base_tokens: int,
        cache_source: CacheSource | None = None,
    ) -> ContextRecoveryEstimate | None:
        deployment = self.deployments[target_engine]
        if cache_source is None:
            if (
                target_engine == session.owner_worker_url
                or target_engine in session.seen_engines
            ):
                cache_source = CacheSource.LOCAL
            elif deployment.shared_l3:
                cache_source = CacheSource.MOONCAKE
            else:
                cache_source = CacheSource.NONE
        probability = self.cache_hits.estimate_probability(
            fingerprint=deployment.cache_fingerprint,
            engine_url=target_engine,
            cache_source=cache_source,
            context_tokens=context_tokens,
        )
        restore_seconds: float | None = 0.0
        restore_sample_source = "none"
        if cache_source is CacheSource.MOONCAKE:
            if source_engine is None:
                return None
            readiness = self._path_readiness(source_engine, target_engine)
            profile = self.profiles.get(deployment.cache_fingerprint)
            if profile is None:
                return None
            source_deployment = self.deployments[source_engine]
            source_node = self._calibration_node_for(source_deployment)
            target_node = self._calibration_node_for(deployment)
            if source_node is None or target_node is None:
                return None
            restore_seconds = self.calibrator.estimate(
                source_node=source_node,
                target_node=target_node,
                required_links=readiness.required_links,
                payload_bytes=profile.estimate_bytes(base_tokens),
            )
            restore_sample_source = "offline"
        if cache_source is not CacheSource.NONE:
            runtime_samples = list(
                self._runtime_restore_seconds.get(
                    (
                        deployment.cache_fingerprint,
                        source_engine,
                        target_engine,
                        context_bucket(context_tokens),
                    ),
                    (),
                )
            )
            if len(runtime_samples) >= self.config.min_samples:
                restore_seconds = percentile(runtime_samples, 0.75)
                restore_sample_source = "runtime"
        return self.context_model.estimate(
            fingerprint=deployment.cache_fingerprint,
            engine_url=target_engine,
            cache_source=cache_source,
            context_tokens=context_tokens,
            base_tokens=base_tokens,
            hit_probability=probability,
            restore_seconds=restore_seconds,
            restore_sample_source=restore_sample_source,
        )

    def _context_prediction_risk(
        self,
        *,
        fingerprint: str,
        source_engine: str,
        target_engine: str,
        estimate: ContextRecoveryEstimate,
        context_tokens: int,
    ) -> float:
        if estimate.cache_source is CacheSource.MOONCAKE:
            errors = list(
                self._runtime_restore_errors.get(
                    (
                        fingerprint,
                        source_engine,
                        target_engine,
                        context_bucket(context_tokens),
                    ),
                    (),
                )
            )
            if len(errors) >= self.config.min_samples:
                return percentile(errors, 0.90)
            if estimate.restore_sample_source == "offline":
                return 0.05 * estimate.restore_seconds
        return self.performance.risk_seconds(
            fingerprint=fingerprint,
            source=estimate.cache_source,
            context_tokens=context_tokens,
            minimum_seconds=0.0,
        )

    def _healthy_urls(self, *, now: float) -> list[str]:
        stale = self.config.metrics_stale_ms / 1000.0
        return sorted(
            url
            for url, load in self.loads.items()
            if load.fresh(now=now, stale_seconds=stale) and url in self.deployments
        )

    def _least_load(
        self,
        urls: list[str],
        *,
        session_id: str,
        session: SessionRoutingState,
        prompt_tokens: int,
        step_max_new_tokens: int | None,
        context_remaining_tokens: int | None,
    ) -> tuple[str, StepGenerationBudget]:
        if not urls:
            raise RuntimeError("no compatible healthy engine")
        max_running = max([self.loads[url].request_capacity for url in urls] + [1])
        max_tokens = max([self.loads[url].token_capacity for url in urls] + [1])
        max_queue = max([self.loads[url].queued for url in urls] + [1])

        budgets = {
            url: self._resolve_step_budget(
                session=session,
                fingerprint=self.deployments[url].cache_fingerprint,
                step_max_new_tokens=step_max_new_tokens,
                context_remaining_tokens=context_remaining_tokens,
            )
            for url in urls
        }

        def key(url: str) -> tuple[float, str]:
            score = self._projected_load_score(
                url,
                prompt_tokens=prompt_tokens,
                expected_output_tokens=(budgets[url].estimated_step_output_tokens or 0),
                capacity_fallback=(max_running, max_tokens, max_queue),
            )
            tie = hashlib.sha256(f"{session_id}\0{url}".encode()).hexdigest()
            return score, tie

        target = min(urls, key=key)
        return target, budgets[target]

    def _projected_load_score(
        self,
        url: str,
        *,
        prompt_tokens: int,
        expected_output_tokens: int = 0,
        capacity_fallback: tuple[int, int, int] | None = None,
    ) -> float:
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
            max_queue = max([self.loads[item].queued for item in compatible] + [1])
        else:
            max_running, max_tokens, max_queue = capacity_fallback
        req_capacity = load.request_capacity or max_running
        token_capacity = load.token_capacity or max_tokens
        queue_capacity = max(max_queue, req_capacity)
        token_pressure = max(
            (
                load.active_tokens
                + load.reserved_tokens
                + prompt_tokens
                + max(0, int(expected_output_tokens))
            )
            / max(1, token_capacity),
            load.token_usage,
        )
        return (
            (load.running + load.reserved_requests + 1) / max(1, req_capacity)
            + token_pressure
            + load.queued / max(1, queue_capacity)
        )

    def _reserve(
        self,
        decision: RoutingDecision,
        *,
        input_ids: list[int],
        base_tokens: int,
        budget: StepGenerationBudget,
    ) -> RoutingLease:
        target = decision.target_worker_url
        expected_output_tokens = budget.estimated_step_output_tokens or 0
        reserved_tokens = len(input_ids) + expected_output_tokens
        projected_load_score = None
        if target is not None:
            projected_load_score = self._projected_load_score(
                target,
                prompt_tokens=len(input_ids),
                expected_output_tokens=expected_output_tokens,
            )
            load = self.loads[target]
            load.reserved_requests += 1
            load.reserved_tokens += reserved_tokens
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
        )

    async def complete(
        self,
        lease: RoutingLease,
        *,
        response_meta: Mapping[str, Any],
        output_tokens: int,
        committed_tokens: list[int],
        success: bool = True,
    ) -> None:
        if not self.config.enabled or lease.worker_url is None:
            return
        async with self._lock:
            context_tokens = (
                lease.context_tokens
                if lease.context_tokens > 0
                else lease.reserved_tokens
            )
            load = self.loads.get(lease.worker_url)
            if load is not None:
                load.reserved_requests = max(0, load.reserved_requests - 1)
                load.reserved_tokens = max(
                    0, load.reserved_tokens - lease.reserved_tokens
                )
            session = self.sessions.setdefault(
                lease.decision.session_id, SessionRoutingState()
            )
            if not success:
                session.pending_owner_worker_url = None
                return
            old_owner = session.owner_worker_url
            new_owner = lease.worker_url
            target_seen_before = new_owner in session.seen_engines
            if old_owner != new_owner:
                session.previous_owner_worker_url = old_owner
                session.owner_turns = 1
            else:
                session.owner_turns += 1
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

            elapsed = max(0.0, time.monotonic() - lease.started_monotonic)
            queue_raw = response_meta.get("queue_time")
            try:
                queue_seconds = (
                    None if queue_raw is None else max(0.0, float(queue_raw))
                )
            except (TypeError, ValueError):
                queue_seconds = None
            decode_throughput_raw = response_meta.get("decode_throughput")
            try:
                decode_throughput = float(decode_throughput_raw)
            except (TypeError, ValueError):
                decode_throughput = None
            decode_seconds = (
                max(0, output_tokens - 1) / decode_throughput
                if decode_throughput is not None and decode_throughput > 0
                else 0.0
            )
            e2e = response_meta.get("e2e_latency")
            try:
                e2e_seconds = float(e2e)
            except (TypeError, ValueError):
                e2e_seconds = elapsed
            context_seconds = (
                None
                if queue_seconds is None
                else max(0.0, e2e_seconds - queue_seconds - decode_seconds)
            )
            cached_raw = response_meta.get("cached_tokens") or 0
            try:
                cached_tokens = int(cached_raw)
            except (TypeError, ValueError):
                cached_tokens = 0
            if new_owner == lease.decision.source_worker_url:
                estimate = lease.decision.source_context
                predicted_queue_seconds = lease.decision.source_queue_seconds
            elif new_owner == lease.decision.target_worker_url:
                estimate = lease.decision.target_context
                predicted_queue_seconds = lease.decision.target_queue_seconds
            else:
                estimate = None
                predicted_queue_seconds = None
            # ``cached_tokens`` is the native SGLang outcome and therefore
            # wins over the Proxy prediction.  In particular, a path modeled
            # as full prefill may still hit shared Mooncake; classifying that
            # response as NONE would contaminate the prefill-throughput model.
            if cached_tokens <= 0:
                source = CacheSource.NONE
            elif old_owner == new_owner or target_seen_before:
                source = CacheSource.LOCAL
            elif deployment.shared_l3:
                source = CacheSource.MOONCAKE
            else:
                source = CacheSource.LOCAL
            observation = self.performance.observe(
                fingerprint=deployment.cache_fingerprint,
                engine_url=new_owner,
                running=(load.running if load is not None else 0),
                context_tokens=context_tokens,
                queue_seconds=queue_seconds,
                context_seconds=context_seconds,
                cached_tokens=cached_tokens,
                output_tokens=output_tokens,
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
                if old_owner is None
                else (
                    deployment.cache_fingerprint,
                    old_owner,
                    new_owner,
                    context_bucket(context_tokens),
                )
            )
            if (
                estimate is not None
                and estimate.cache_source is CacheSource.MOONCAKE
                and observation.context_seconds is not None
                and runtime_key is not None
            ):
                self._runtime_restore_errors[runtime_key].append(
                    abs(estimate.estimated_seconds - observation.context_seconds)
                )
            self.cache_hits.observe(
                fingerprint=deployment.cache_fingerprint,
                engine_url=new_owner,
                cache_source=source,
                estimated_base_tokens=lease.base_tokens,
                actual_cached_tokens=observation.cached_tokens,
                context_tokens=context_tokens,
            )
            prefill_throughput = self.performance.prefill_throughput(
                fingerprint=deployment.cache_fingerprint,
                engine_url=new_owner,
                context_tokens=context_tokens,
            )
            actual_prefill_tokens = max(0, context_tokens - observation.cached_tokens)
            restore_seconds_actual = None
            restore_throughput = None
            if (
                source is not CacheSource.NONE
                and runtime_key is not None
                and observation.cached_tokens > 0
                and observation.context_seconds is not None
                and prefill_throughput is not None
            ):
                restore_seconds_actual = max(
                    0.0,
                    observation.context_seconds
                    - actual_prefill_tokens / prefill_throughput,
                )
                self._runtime_restore_seconds[runtime_key].append(
                    restore_seconds_actual
                )
                profile = self.profiles.get(deployment.cache_fingerprint)
                if profile is not None and restore_seconds_actual > 0:
                    restore_throughput = (
                        profile.estimate_bytes(observation.cached_tokens)
                        / restore_seconds_actual
                    )
                    self._runtime_restore_throughputs[runtime_key].append(
                        restore_throughput
                    )
            self._observations.append(
                {
                    "session_id": lease.decision.session_id,
                    "engine_url": new_owner,
                    "cache_source": source.value,
                    "expected_cached_tokens": (
                        None if estimate is None else estimate.expected_cached_tokens
                    ),
                    "actual_cached_tokens": observation.cached_tokens,
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
                    "predicted_queue_seconds": observation.predicted_queue_seconds,
                    "actual_queue_seconds": observation.queue_seconds,
                    "queue_prediction_error_seconds": (
                        observation.queue_prediction_error_seconds
                    ),
                    "actual_context_seconds": observation.context_seconds,
                    "restore_seconds": restore_seconds_actual,
                    "restore_throughput": restore_throughput,
                    "prefill_throughput": prefill_throughput,
                    "hit_probability": (
                        None if estimate is None else estimate.hit_probability
                    ),
                    "queue_risk_seconds": lease.decision.queue_risk_seconds,
                    "context_risk_seconds": lease.decision.context_risk_seconds,
                    "decision_risk": lease.decision.decision_risk_seconds,
                    "step_max_tokens_source": (lease.decision.step_max_tokens_source),
                    "effective_step_max_tokens": (
                        lease.decision.effective_step_max_tokens
                    ),
                    "historical_step_tokens_p75": (
                        lease.decision.historical_step_tokens_p75
                    ),
                    "group_remaining_tokens": (lease.decision.group_remaining_tokens),
                    "estimated_step_output_tokens": (
                        lease.decision.estimated_step_output_tokens
                    ),
                    "source_decode_seconds": (lease.decision.source_decode_seconds),
                    "target_decode_seconds": (lease.decision.target_decode_seconds),
                }
            )
            state = self.pools.get(deployment.cache_fingerprint)
            if state is not None:
                state.update(
                    self._pool_readiness(
                        deployment.cache_fingerprint, now=time.monotonic()
                    )
                )
            self._record_successful_online_request()

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
                "active_sessions": len(self.sessions),
            }
