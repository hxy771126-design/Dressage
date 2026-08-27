"""Proxy-side engine placement and turn-boundary rebalancing."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import socket
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .greedy import (
    EngineBaseline,
    FeasibleEdge,
    GreedyStepDecision,
    GreedyStepKind,
    choose_greedy_step,
    choose_stable_engine,
    projected_pressure,
)
from .load_decision_trace import LoadDecisionHistory, LoadDecisionTrace
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
)
from .snapshot_store import CalibrationSnapshotStore
from .transfer_calibrator import (
    Benchmark,
    CalibrationPlan,
    CalibrationState,
    CacheSource,
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


def longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    size = min(len(left), len(right))
    index = 0
    while index < size and left[index] == right[index]:
        index += 1
    return index


@dataclass(frozen=True)
class EngineDeploymentInfo:
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
    healthy: bool = True
    metrics_timestamp: float = 0.0
    running: int = 0
    queued: int = 0
    active_tokens: int = 0
    token_capacity: int = 0
    request_capacity: int = 0
    token_usage: float = 0.0
    snapshot_generation: int = 0
    snapshot_fetch_status: str = "never"
    snapshot_fetch_duration_seconds: float = 0.0

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
    fingerprint: str | None = None
    previous_committed_tokens: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RoutingDecision:
    session_id: str
    source_worker_url: str | None
    target_worker_url: str | None
    moved: bool = False


@dataclass(frozen=True)
class RoutingLease:
    decision: RoutingDecision
    worker_url: str | None
    reservation_id: int | None = None
    load_decision_id: int | None = None


@dataclass
class _ReservationEntry:
    engine_url: str
    scoring_queue_increment: int
    scoring_token_increment: int
    scoring_revision: int
    scoring_active: bool


@dataclass(frozen=True)
class _LoadFetchResult:
    status: str
    duration_seconds: float
    load: EngineLoad | None


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
        self.calibrator = TransferCalibrator()
        self.deployments: dict[str, EngineDeploymentInfo] = {}
        self.loads: dict[str, EngineLoad] = {}
        self._load_generations: dict[str, int] = defaultdict(int)
        self._snapshot_fetch_status: dict[str, tuple[str, float]] = {}
        self._reservations: dict[int, _ReservationEntry] = {}
        self._next_reservation_id = 1
        self._scoring_revision = 0
        self.profiles: dict[str, ModelCacheProfile] = {}
        self.pools: dict[str, CompatibilityPoolStateMachine] = {}
        self.plans: dict[str, CalibrationPlan] = {}
        self.sessions: dict[str, SessionRoutingState] = {}
        self.excluded_engines: dict[str, str] = {}
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
        self._next_load_decision_id = 1
        self._load_decision_history = LoadDecisionHistory(config.history_size)
        self._refresh_lock = asyncio.Lock()
        self._control_poll_task: asyncio.Task | None = None
        self._snapshot_poll_task: asyncio.Task | None = None
        self._snapshot_fetch_tasks: dict[str, asyncio.Task[None]] = {}
        self._initial_snapshot_event = asyncio.Event()
        self._snapshot_poll_started_monotonic: float | None = None
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
        if self._snapshot_poll_task is not None:
            self._snapshot_poll_task.cancel()
            try:
                await self._snapshot_poll_task
            except asyncio.CancelledError:
                pass
            self._snapshot_poll_task = None
        fetch_tasks = tuple(self._snapshot_fetch_tasks.values())
        for task in fetch_tasks:
            task.cancel()
        if fetch_tasks:
            await asyncio.gather(*fetch_tasks, return_exceptions=True)
        self._snapshot_fetch_tasks.clear()
        if self._control_poll_task is not None:
            self._control_poll_task.cancel()
            try:
                await self._control_poll_task
            except asyncio.CancelledError:
                pass
            self._control_poll_task = None
        await self._drain_snapshot_tasks()
        if self._snapshot_store is not None and not self._final_snapshot_written:
            self._final_snapshot_written = True
            await self._persist_current_snapshot("final", self._online_request_count)

    def _capture_file_snapshot(self, kind: str) -> dict[str, Any] | None:
        try:
            return {"offline_calibration": self.calibration_snapshot()}
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
                if not self._stopping and self._control_poll_task is None:
                    self._snapshot_poll_started_monotonic = time.monotonic()
                    self._control_poll_task = asyncio.create_task(
                        self._control_poll_loop(),
                        name="engine-rebalancing-control-poll",
                    )
                    self._snapshot_poll_task = asyncio.create_task(
                        self._snapshot_poll_loop(),
                        name="engine-rebalancing-snapshot-poll",
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

    async def _control_poll_loop(self) -> None:
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
                        min(startup_failures - 1, len(startup_backoff) - 1)
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
            await asyncio.sleep(delay)

    async def _snapshot_poll_loop(self) -> None:
        """Launch at most one independent load probe per healthy Engine."""

        interval = self.config.load_snapshot_poll_interval_ms / 1000.0
        while not self._stopping:
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
            for url in urls:
                current = self._snapshot_fetch_tasks.get(url)
                if current is not None and not current.done():
                    continue
                task = asyncio.create_task(
                    self._poll_engine_snapshot(url),
                    name=f"engine-rebalancing-snapshot-fetch-{url}",
                )
                self._snapshot_fetch_tasks[url] = task
                task.add_done_callback(
                    lambda completed, worker_url=url: self._snapshot_fetch_done(
                        worker_url, completed
                    )
                )
            await asyncio.sleep(interval)

    def _snapshot_fetch_done(
        self,
        url: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._snapshot_fetch_tasks.get(url) is task:
            self._snapshot_fetch_tasks.pop(url, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.warning(
                "background /v1/loads poll failed for %s",
                url,
                exc_info=True,
            )

    async def _poll_engine_snapshot(self, url: str) -> None:
        async with self._lock:
            deployment = self.deployments.get(url)
            load = self.loads.get(url)
            if deployment is None or load is None or not load.healthy:
                return
            scoring_boundary = self._scoring_revision
            topology_signature = (
                deployment.weight_version,
                deployment.cache_fingerprint,
            )
        result = await self._fetch_load(url)
        async with self._lock:
            self._snapshot_fetch_status[url] = (
                result.status,
                result.duration_seconds,
            )
            current = self.loads.get(url)
            if current is not None:
                current.snapshot_fetch_status = result.status
                current.snapshot_fetch_duration_seconds = result.duration_seconds
            if result.status != "ok" or result.load is None:
                return
            deployment = self.deployments.get(url)
            if (
                deployment is None
                or current is None
                or not current.healthy
                or (
                    deployment.weight_version,
                    deployment.cache_fingerprint,
                )
                != topology_signature
            ):
                return
            published = result.load
            published.healthy = True
            published.snapshot_fetch_status = "ok"
            published.snapshot_fetch_duration_seconds = result.duration_seconds
            self.loads[url] = published
            self._acknowledge_scoring_deltas(url, scoring_boundary)
            self._load_generations[url] += 1
            published.snapshot_generation = self._load_generations[url]
            self._initial_snapshot_event.set()
            readiness_now = time.monotonic()
            for fingerprint, state in self.pools.items():
                state.update(self._pool_readiness(fingerprint, now=readiness_now))

    async def _fetch_load(self, url: str) -> _LoadFetchResult:
        started = time.monotonic()
        try:
            payload = await asyncio.wait_for(
                self.client.get_worker_loads(url),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            return _LoadFetchResult("timeout", time.monotonic() - started, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _LoadFetchResult("error", time.monotonic() - started, None)
        try:
            load = self._normalize_load(url, payload, now=time.monotonic())
        except (TypeError, ValueError, OverflowError):
            load = None
        planning_values_valid = False
        if load is not None:
            planning_values = (
                load.running,
                load.queued,
                load.active_tokens,
                load.request_capacity,
                load.token_capacity,
                load.token_usage,
            )
            try:
                planning_values_valid = all(
                    math.isfinite(value) and value >= 0
                    for value in planning_values
                )
            except (TypeError, ValueError, OverflowError):
                planning_values_valid = False
        if (
            load is None
            or not planning_values_valid
            or load.request_capacity <= 0
            or load.token_capacity <= 0
        ):
            return _LoadFetchResult(
                "invalid",
                time.monotonic() - started,
                None,
            )
        return _LoadFetchResult(
            "ok",
            time.monotonic() - started,
            load,
        )

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
                self.loads.setdefault(url, EngineLoad())
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
        return EngineLoad(
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
        )

    def _live_scoring_totals(self, url: str) -> tuple[int, int]:
        entries = [
            entry
            for entry in self._reservations.values()
            if entry.engine_url == url and entry.scoring_active
        ]
        return (
            sum(entry.scoring_queue_increment for entry in entries),
            sum(entry.scoring_token_increment for entry in entries),
        )

    def _acknowledge_scoring_deltas(self, url: str, revision: int) -> None:
        for entry in self._reservations.values():
            if (
                entry.engine_url == url
                and entry.scoring_active
                and entry.scoring_revision <= revision
            ):
                entry.scoring_active = False

    def _release_reservation(self, reservation_id: int | None) -> None:
        if reservation_id is None:
            return
        self._reservations.pop(reservation_id, None)

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
            eligible_paths=eligible_paths,
        )

    def _path_readiness(self, source: str, target: str) -> ContextPathReadiness:
        source_deployment = self.deployments.get(source)
        target_deployment = self.deployments.get(target)
        if source_deployment is None or target_deployment is None:
            return ContextPathReadiness(source, target, CacheSource.NONE)
        if not target_deployment.shared_l3:
            return ContextPathReadiness(
                source,
                target,
                CacheSource.NONE,
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
            required_links=tuple(required),
            completed_links=completed,
            pending_links=pending,
            skipped_links=skipped,
        )

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

    async def acquire(
        self,
        *,
        session_id: str,
        input_ids: list[int],
        expected_version: str | None = None,
        require_registered_context: bool = False,
    ) -> RoutingLease:
        if not self.config.enabled:
            decision = RoutingDecision(
                session_id=session_id,
                source_worker_url=None,
                target_worker_url=None,
            )
            return RoutingLease(decision=decision, worker_url=None)

        wait_started = time.monotonic()
        await self._wait_for_initial_snapshot()
        snapshot_wait_seconds = time.monotonic() - wait_started
        schedule_started = time.monotonic()
        async with self._lock:
            if self._stopping:
                raise RuntimeError("engine rebalancer is closed")
            session_before = (
                None
                if session_id not in self.sessions
                else deepcopy(self.sessions[session_id])
            )
            next_reservation_before = self._next_reservation_id
            scoring_revision_before = self._scoring_revision
            next_load_decision_before = self._next_load_decision_id
            try:
                lease = self._schedule_step_locked(
                    session_id=session_id,
                    input_ids=input_ids,
                    expected_version=expected_version,
                    require_registered_context=require_registered_context,
                    schedule_started=schedule_started,
                    snapshot_wait_seconds=snapshot_wait_seconds,
                )
            except BaseException as exc:
                for reservation_id in tuple(self._reservations):
                    if reservation_id >= next_reservation_before:
                        self._reservations.pop(reservation_id, None)
                self._next_reservation_id = next_reservation_before
                self._scoring_revision = scoring_revision_before
                self._next_load_decision_id = next_load_decision_before
                if session_before is None:
                    self.sessions.pop(session_id, None)
                else:
                    self.sessions[session_id] = session_before
                self._record_failed_load_decision_locked(
                    session_id=session_id,
                    prompt_token_count=len(input_ids),
                    schedule_started=schedule_started,
                    snapshot_wait_seconds=snapshot_wait_seconds,
                    error=exc,
                )
                raise
            return lease

    async def _wait_for_initial_snapshot(self) -> None:
        if self._initial_snapshot_event.is_set():
            return
        started = self._snapshot_poll_started_monotonic
        if started is None:
            return
        remaining = started + 1.0 - time.monotonic()
        if remaining <= 0:
            return
        try:
            await asyncio.wait_for(
                self._initial_snapshot_event.wait(),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            return

    def _schedule_step_locked(
        self,
        *,
        session_id: str,
        input_ids: list[int],
        expected_version: str | None,
        require_registered_context: bool,
        schedule_started: float,
        snapshot_wait_seconds: float,
    ) -> RoutingLease:
        existing = self.sessions.get(session_id)
        if existing is None and require_registered_context:
            raise RuntimeError(
                "session context is not registered or was discarded: "
                f"{session_id}"
            )
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
        source_load = None if source is None else self.loads.get(source)
        version_valid_owner = (
            source_deployment is not None
            and (
                expected_version is None
                or source_deployment.weight_version == expected_version
            )
        )
        healthy_owner = (
            source is not None
            and version_valid_owner
            and source_load is not None
            and source_load.healthy
        )
        now = time.monotonic()
        stale_seconds = self.config.metrics_stale_ms / 1000.0
        owner_snapshot_fresh = (
            healthy_owner
            and source_load is not None
            and source_load.fresh(now=now, stale_seconds=stale_seconds)
            and self._effective_baseline_locked(source) is not None
        )

        healthy_candidates = [
            url
            for url, deployment in self.deployments.items()
            if deployment.worker_url == url
            and self.loads.get(url) is not None
            and self.loads[url].healthy
            and (
                expected_version is None
                or deployment.weight_version == expected_version
            )
            and (
                source is None
                or (
                    fingerprint is not None
                    and deployment.cache_fingerprint == fingerprint
                )
            )
        ]
        fresh_candidates = [
            url
            for url in healthy_candidates
            if self.loads[url].fresh(now=now, stale_seconds=stale_seconds)
            and self._effective_baseline_locked(url) is not None
        ]

        if healthy_owner and not owner_snapshot_fresh:
            return self._schedule_fallback_locked(
                session_id=session_id,
                session=session,
                source=source,
                target=source,
                kind=GreedyStepKind.FIXED_OWNER,
                input_ids=input_ids,
                fallback_reason="owner_snapshot_unavailable",
                schedule_started=schedule_started,
                snapshot_wait_seconds=snapshot_wait_seconds,
            )

        mandatory_failover = source is not None and not healthy_owner
        kind = (
            GreedyStepKind.NEW_SESSION
            if source is None
            else (
                GreedyStepKind.MANDATORY_FAILOVER
                if mandatory_failover
                else GreedyStepKind.EXISTING_SESSION
            )
        )
        lcp = (
            0
            if source is None
            else longest_common_prefix_length(
                session.previous_committed_tokens,
                input_ids,
            )
        )
        edges: list[FeasibleEdge] = []
        baselines: list[EngineBaseline] = []
        for target in fresh_candidates:
            if source is not None and not mandatory_failover and target != source:
                if (
                    self._path_readiness(source, target).cache_source
                    is not CacheSource.MOONCAKE
                ):
                    continue
            prompt_tokens = len(input_ids)
            if healthy_owner and target == source:
                scoring_token_increment = max(0, prompt_tokens - lcp)
            else:
                scoring_token_increment = prompt_tokens
            edge = FeasibleEdge(
                session_id=session_id,
                engine_url=target,
                queue_increment=1,
                token_increment=scoring_token_increment,
            )
            baseline = self._effective_baseline_locked(target)
            if baseline is None:
                continue
            edges.append(edge)
            baselines.append(baseline)

        if not edges:
            fallback_target: str | None
            fallback_reason: str
            if healthy_owner:
                fallback_target = source
                fallback_reason = "no_fresh_migration_snapshot"
                fallback_kind = GreedyStepKind.FIXED_OWNER
            elif healthy_candidates:
                fallback_target = choose_stable_engine(
                    session_id, sorted(healthy_candidates)
                )
                fallback_reason = (
                    "mandatory_failover_without_fresh_snapshot"
                    if mandatory_failover
                    else "new_session_without_fresh_snapshot"
                )
                fallback_kind = kind
            else:
                fallback_target = None
                fallback_reason = "no_eligible_engine_router_fallback"
                fallback_kind = kind
            return self._schedule_fallback_locked(
                session_id=session_id,
                session=session,
                source=source,
                target=fallback_target,
                kind=fallback_kind,
                input_ids=input_ids,
                fallback_reason=fallback_reason,
                schedule_started=schedule_started,
                snapshot_wait_seconds=snapshot_wait_seconds,
            )

        greedy = choose_greedy_step(
            session_id=session_id,
            kind=kind,
            owner_engine_url=source,
            engines=baselines,
            edges=edges,
            min_load_improvement_ratio=self.config.min_load_improvement_ratio,
        )
        target = greedy.selected_target
        selected_edge = next(edge for edge in edges if edge.engine_url == target)
        deployment = self.deployments[target]
        target_fingerprint = deployment.cache_fingerprint
        moved = source is not None and target != source
        decision = RoutingDecision(
            session_id=session_id,
            source_worker_url=source,
            target_worker_url=target,
            moved=moved,
        )
        if existing is None:
            self.sessions[session_id] = session
        if source is None:
            session.fingerprint = target_fingerprint
        lease = self._reserve(
            decision,
            scoring_queue_increment=int(selected_edge.queue_increment),
            scoring_token_increment=int(selected_edge.token_increment),
        )
        lease = replace(
            lease,
            load_decision_id=self._next_load_decision_id,
        )
        self._next_load_decision_id += 1
        self._record_load_decision_locked(
            lease=lease,
            kind=kind,
            candidate_urls=tuple(sorted(edge.engine_url for edge in edges)),
            greedy=greedy,
            selected_edge=selected_edge,
            fallback_reason=None,
            schedule_started=schedule_started,
            snapshot_wait_seconds=snapshot_wait_seconds,
        )
        return lease

    def _schedule_fallback_locked(
        self,
        *,
        session_id: str,
        session: SessionRoutingState,
        source: str | None,
        target: str | None,
        kind: GreedyStepKind,
        input_ids: list[int],
        fallback_reason: str,
        schedule_started: float,
        snapshot_wait_seconds: float,
    ) -> RoutingLease:
        deployment = None if target is None else self.deployments.get(target)
        target_fingerprint = (
            session.fingerprint
            if deployment is None
            else deployment.cache_fingerprint
        )
        lcp = (
            0
            if source is None
            else longest_common_prefix_length(
                session.previous_committed_tokens,
                input_ids,
            )
        )
        stays_on_healthy_owner = (
            target is not None
            and target == source
            and self.loads.get(target) is not None
            and self.loads[target].healthy
        )
        scoring_token_increment = (
            max(0, len(input_ids) - lcp)
            if stays_on_healthy_owner
            else len(input_ids)
        )
        moved = source is not None and target != source
        decision = RoutingDecision(
            session_id=session_id,
            source_worker_url=source,
            target_worker_url=target,
            moved=moved,
        )
        existing = session_id in self.sessions
        if not existing:
            self.sessions[session_id] = session
        if source is None and target_fingerprint is not None:
            session.fingerprint = target_fingerprint
        lease = self._reserve(
            decision,
            scoring_queue_increment=(0 if target is None else 1),
            scoring_token_increment=(
                0 if target is None else scoring_token_increment
            ),
        )
        lease = replace(
            lease,
            load_decision_id=self._next_load_decision_id,
        )
        self._next_load_decision_id += 1
        selected_edge = (
            None
            if target is None
            else FeasibleEdge(
                session_id=session_id,
                engine_url=target,
                queue_increment=1,
                token_increment=scoring_token_increment,
            )
        )
        self._record_load_decision_locked(
            lease=lease,
            kind=kind,
            candidate_urls=(
                ()
                if target is None
                else (target,)
            ),
            greedy=None,
            selected_edge=selected_edge,
            fallback_reason=fallback_reason,
            schedule_started=schedule_started,
            snapshot_wait_seconds=snapshot_wait_seconds,
        )
        return lease

    def _effective_baseline_locked(self, url: str) -> EngineBaseline | None:
        load = self.loads.get(url)
        if (
            load is None
            or load.request_capacity <= 0
            or load.token_capacity <= 0
        ):
            return None
        queue_delta, token_delta = self._live_scoring_totals(url)
        return EngineBaseline(
            url=url,
            base_requests=load.running,
            base_tokens=load.active_tokens + token_delta,
            base_queue=load.queued + queue_delta,
            request_capacity=load.request_capacity,
            token_capacity=load.token_capacity,
            token_usage=load.token_usage,
        )

    def _record_load_decision_locked(
        self,
        *,
        lease: RoutingLease,
        kind: GreedyStepKind,
        candidate_urls: tuple[str, ...],
        greedy: GreedyStepDecision | None,
        selected_edge: FeasibleEdge | None,
        fallback_reason: str | None,
        schedule_started: float,
        snapshot_wait_seconds: float,
    ) -> None:
        now_monotonic = time.monotonic()
        decision_id = lease.load_decision_id
        if decision_id is None:
            raise RuntimeError("committed lease is missing its load decision id")
        source = lease.decision.source_worker_url
        target = lease.worker_url
        engine_traces = self._load_engine_traces_locked(now=now_monotonic)
        effective_scores = [
            item["effective_pressure"]["total"]
            for item in engine_traces
            if isinstance(item.get("effective_pressure"), Mapping)
        ]
        snapshot_ages = [
            item["snapshot_age_seconds"]
            for item in engine_traces
            if item.get("snapshot_age_seconds") is not None
        ]
        trace = {
            "decision": {
                "id": decision_id,
                "completed_at": time.time(),
                "status": "committed",
                "schedule_seconds": max(
                    0.0, now_monotonic - schedule_started
                ),
                "snapshot_wait_seconds": snapshot_wait_seconds,
                "snapshot_age_seconds": (
                    None if not snapshot_ages else max(snapshot_ages)
                ),
                "fallback_reason": fallback_reason,
            },
            "step": {
                "session_id": lease.decision.session_id,
                "session_kind": kind.value,
                "source": source,
                "target": target,
                "moved": lease.decision.moved,
                "candidate_urls": list(candidate_urls),
                "owner_projected_score": (
                    None if greedy is None else greedy.owner_projected_score
                ),
                "candidate_projected_scores": (
                    {}
                    if greedy is None
                    else dict(greedy.candidate_projected_scores)
                ),
                "best_target": (
                    target if greedy is None else greedy.best_target
                ),
                "best_projected_score": (
                    None if greedy is None else greedy.best_projected_score
                ),
                "selected_projected_score": (
                    None if greedy is None else greedy.selected_projected_score
                ),
                "improvement_ratio": (
                    None if greedy is None else greedy.improvement_ratio
                ),
                "required_improvement_ratio": (
                    None
                    if greedy is None
                    else greedy.required_improvement_ratio
                ),
                "threshold_met": (
                    None if greedy is None else greedy.threshold_met
                ),
                "decision_reason": (
                    fallback_reason
                    if greedy is None
                    else greedy.decision_reason
                ),
                "queue_increment": (
                    0 if selected_edge is None else selected_edge.queue_increment
                ),
                "token_increment": (
                    0 if selected_edge is None else selected_edge.token_increment
                ),
            },
            "engines": engine_traces,
            "scheduler": {
                "strategy": "online_dynamic_greedy",
                "maximum_pressure": (
                    None if not effective_scores else max(effective_scores)
                ),
                "minimum_pressure": (
                    None if not effective_scores else min(effective_scores)
                ),
                "load_range": (
                    None
                    if not effective_scores
                    else max(effective_scores) - min(effective_scores)
                ),
                "migrations": int(
                    lease.decision.moved
                    and kind is GreedyStepKind.EXISTING_SESSION
                ),
                "required_improvement_ratio": (
                    self.config.min_load_improvement_ratio
                ),
                "score_tolerance": 1e-7,
            },
        }
        self._load_decision_history.record(LoadDecisionTrace(trace))

    def _record_failed_load_decision_locked(
        self,
        *,
        session_id: str,
        prompt_token_count: int,
        schedule_started: float,
        snapshot_wait_seconds: float,
        error: BaseException,
    ) -> None:
        now_monotonic = time.monotonic()
        decision_id = self._next_load_decision_id
        self._next_load_decision_id += 1
        self._load_decision_history.record(
            LoadDecisionTrace(
                {
                    "decision": {
                        "id": decision_id,
                        "completed_at": time.time(),
                        "status": "failed",
                        "schedule_seconds": max(
                            0.0, now_monotonic - schedule_started
                        ),
                        "snapshot_wait_seconds": snapshot_wait_seconds,
                        "snapshot_age_seconds": None,
                        "fallback_reason": "scheduling_failure",
                    },
                    "step": {
                        "session_id": session_id,
                        "prompt_token_count": prompt_token_count,
                        "target": None,
                        "moved": False,
                        "decision_reason": type(error).__name__,
                    },
                    "engines": self._load_engine_traces_locked(
                        now=now_monotonic
                    ),
                    "scheduler": {
                        "strategy": "online_dynamic_greedy",
                        "migrations": 0,
                    },
                }
            )
        )

    def _load_engine_traces_locked(self, *, now: float) -> list[dict[str, Any]]:
        stale_seconds = self.config.metrics_stale_ms / 1000.0
        traces: list[dict[str, Any]] = []
        for url in sorted(set(self.loads) | set(self.deployments)):
            load = self.loads.get(url)
            deployment = self.deployments.get(url)
            scoring_queue, scoring_tokens = self._live_scoring_totals(url)
            baseline = self._effective_baseline_locked(url)
            pressure = None if baseline is None else projected_pressure(baseline)
            status, duration = self._snapshot_fetch_status.get(
                url,
                (
                    "never" if load is None else load.snapshot_fetch_status,
                    0.0 if load is None else load.snapshot_fetch_duration_seconds,
                ),
            )
            traces.append(
                {
                    "url": url,
                    "health": load is not None and load.healthy,
                    "version": (
                        None if deployment is None else deployment.weight_version
                    ),
                    "fingerprint": (
                        None
                        if deployment is None
                        else deployment.cache_fingerprint
                    ),
                    "snapshot_generation": (
                        0 if load is None else load.snapshot_generation
                    ),
                    "snapshot_fetch_status": status,
                    "snapshot_fetch_duration_seconds": duration,
                    "snapshot_age_seconds": (
                        None
                        if load is None or load.metrics_timestamp <= 0
                        else max(0.0, now - load.metrics_timestamp)
                    ),
                    "snapshot_fresh": (
                        False
                        if load is None
                        else load.fresh(now=now, stale_seconds=stale_seconds)
                    ),
                    "observed_running": (
                        None if load is None else load.running
                    ),
                    "observed_active_tokens": (
                        None if load is None else load.active_tokens
                    ),
                    "observed_queued": (
                        None if load is None else load.queued
                    ),
                    "request_capacity": (
                        None if load is None else load.request_capacity
                    ),
                    "token_capacity": (
                        None if load is None else load.token_capacity
                    ),
                    "token_usage": (
                        None if load is None else load.token_usage
                    ),
                    "unobserved_queue_delta": scoring_queue,
                    "unobserved_token_delta": scoring_tokens,
                    "effective_requests": (
                        None if baseline is None else baseline.base_requests
                    ),
                    "effective_tokens": (
                        None if baseline is None else baseline.base_tokens
                    ),
                    "effective_queue": (
                        None if baseline is None else baseline.base_queue
                    ),
                    "effective_pressure": (
                        None if pressure is None else asdict(pressure)
                    ),
                }
            )
        return traces

    def _reserve(
        self,
        decision: RoutingDecision,
        *,
        scoring_queue_increment: int = 1,
        scoring_token_increment: int = 0,
    ) -> RoutingLease:
        target = decision.target_worker_url
        reservation_id = None
        if target is not None:
            reservation_id = self._next_reservation_id
            self._next_reservation_id += 1
            self._scoring_revision += 1
            self._reservations[reservation_id] = _ReservationEntry(
                engine_url=target,
                scoring_queue_increment=max(0, int(scoring_queue_increment)),
                scoring_token_increment=max(0, int(scoring_token_increment)),
                scoring_revision=self._scoring_revision,
                scoring_active=True,
            )
        return RoutingLease(
            decision=decision,
            worker_url=target,
            reservation_id=reservation_id,
        )

    async def complete(
        self,
        lease: RoutingLease,
        *,
        committed_tokens: list[int],
        success: bool = True,
    ) -> None:
        if not self.config.enabled:
            return
        async with self._lock:
            self._release_reservation(lease.reservation_id)
            if lease.worker_url is None:
                return
            session = self.sessions.get(lease.decision.session_id)
            if session is None:
                return
            if not success:
                return
            new_owner = lease.worker_url
            session.owner_worker_url = new_owner
            deployment = self.deployments[new_owner]
            session.fingerprint = deployment.cache_fingerprint
            session.previous_committed_tokens = list(committed_tokens)
            self._record_successful_online_request()

    async def fail(self, lease: RoutingLease) -> None:
        await self.complete(
            lease,
            committed_tokens=[],
            success=False,
        )

    async def register_session_context(
        self,
        *,
        session_id: str,
    ) -> None:
        if not self.config.enabled:
            return
        async with self._lock:
            self.sessions.setdefault(session_id, SessionRoutingState())

    async def discard_session_context(self, session_id: str) -> None:
        async with self._lock:
            self.sessions.pop(session_id, None)

    async def finalize_session(self, session_id: str) -> None:
        async with self._lock:
            self.sessions.pop(session_id, None)

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
                "calibration": self.calibration_snapshot(),
                "recent_load_decisions": self._load_decision_history.snapshot(),
                "active_sessions": len(self.sessions),
            }
