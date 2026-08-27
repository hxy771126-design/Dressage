"""Plan-driven transfer calibration for context restoration paths.

The planner contains no SGLang-specific cache logic.  A deployment can attach a
benchmark backend in the node-local process that owns CUDA/Mooncake clients.
When no transport is required (for example full prefill without shared L3), the
path is immediately ready.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable

from .model_cache_profile import ModelCacheProfile, canonical_fingerprint


class CacheSource(str, Enum):
    NONE = "none"
    MOONCAKE = "mooncake"


@dataclass(frozen=True)
class CalibrationTask:
    source_node: str
    target_node: str
    link_type: str
    payload_buckets: tuple[int, ...]


@dataclass(frozen=True)
class CalibrationPlan:
    fingerprint: str
    tasks: tuple[CalibrationTask, ...]
    skipped_links: dict[str, str]

    @classmethod
    def build(
        cls,
        *,
        fingerprint: str,
        engine_deployments: Iterable[Any],
        shared_l3: bool,
        host_staging: bool,
        gpudirect: bool,
        model_cache_profile: ModelCacheProfile,
    ) -> "CalibrationPlan":
        deployments = list(engine_deployments)
        nodes = sorted({str(getattr(item, "node_id", "")) for item in deployments})
        skipped: dict[str, str] = {}
        if len(deployments) < 2:
            skipped["migration"] = "single-engine deployment"
            return cls(fingerprint, (), skipped)
        if not shared_l3:
            skipped["mooncake"] = "L3 disabled"
            skipped["rdma"] = "L3 disabled"
            return cls(fingerprint, (), skipped)

        raw_buckets = [
            model_cache_profile.estimate_bytes(tokens)
            for tokens in (8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024)
        ]
        buckets = tuple(sorted({max(1, int(value)) for value in raw_buckets}))
        tasks: list[CalibrationTask] = []
        protocol_by_node = {
            str(getattr(item, "node_id", "")): str(
                getattr(item, "mooncake_protocol", "") or "remote"
            )
            for item in deployments
        }
        for source in nodes:
            for target in nodes:
                if gpudirect:
                    link = "mooncake_gpudirect"
                elif source == target:
                    link = "mooncake_local"
                else:
                    protocol = "".join(
                        character if character.isalnum() else "_"
                        for character in protocol_by_node[target].lower()
                    ).strip("_")
                    link = f"mooncake_{protocol or 'remote'}"
                tasks.append(CalibrationTask(source, target, link, buckets))
                if host_staging:
                    tasks.append(CalibrationTask(source, target, "d2h", buckets))
                    tasks.append(CalibrationTask(source, target, "h2d", buckets))
        if gpudirect:
            skipped["h2d"] = "GPUDirect restore path"
            skipped["d2h"] = "GPUDirect restore path"
        if len(nodes) == 1:
            skipped["rdma"] = "single-node deployment"
        skipped["p2p"] = "native restore path does not use CUDA P2P"
        return cls(fingerprint, tuple(tasks), skipped)


class CalibrationState(str, Enum):
    DISABLED = "DISABLED"
    WAITING_FOR_RAY = "WAITING_FOR_RAY"
    RUNNING = "RUNNING"
    READY = "READY"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True, init=False)
class CalibrationSample:
    """One payload bucket's complete transfer-time observation.

    ``latency_seconds`` remains an accepted constructor alias for injected
    test backends.  Cost estimation always consumes ``elapsed_seconds_p75``;
    bandwidth is diagnostic and is never added to the measured elapsed time.
    """

    elapsed_seconds_p75: float
    bandwidth_bytes_per_second_p25: float
    payload_bytes: int
    sample_count: int
    measured_at: float

    def __init__(
        self,
        *,
        elapsed_seconds_p75: float | None = None,
        bandwidth_bytes_per_second_p25: float | None = None,
        payload_bytes: int = 0,
        sample_count: int = 1,
        measured_at: float | None = None,
        latency_seconds: float | None = None,
        bandwidth_bytes_per_second: float | None = None,
    ) -> None:
        elapsed = elapsed_seconds_p75
        if elapsed is None:
            elapsed = latency_seconds
        if elapsed is None:
            raise TypeError("elapsed_seconds_p75 is required")
        bandwidth = bandwidth_bytes_per_second_p25
        if bandwidth is None:
            bandwidth = bandwidth_bytes_per_second
        if bandwidth is None:
            raise TypeError("bandwidth_bytes_per_second_p25 is required")
        object.__setattr__(self, "elapsed_seconds_p75", max(0.0, float(elapsed)))
        object.__setattr__(
            self,
            "bandwidth_bytes_per_second_p25",
            max(1.0, float(bandwidth)),
        )
        object.__setattr__(self, "payload_bytes", max(0, int(payload_bytes)))
        object.__setattr__(self, "sample_count", max(1, int(sample_count)))
        object.__setattr__(
            self,
            "measured_at",
            time.time() if measured_at is None else float(measured_at),
        )

    @property
    def latency_seconds(self) -> float:
        return self.elapsed_seconds_p75

    @property
    def bandwidth_bytes_per_second(self) -> float:
        return self.bandwidth_bytes_per_second_p25


@dataclass(frozen=True)
class ContextPathReadiness:
    source_engine: str
    target_engine: str
    cache_source: CacheSource
    required_links: tuple[str, ...] = ()
    completed_links: tuple[str, ...] = ()
    pending_links: tuple[str, ...] = ()
    skipped_links: dict[str, str] = field(default_factory=dict)


Benchmark = Callable[
    [CalibrationTask, int], CalibrationSample | Awaitable[CalibrationSample]
]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class TransferCalibrator:
    def __init__(self) -> None:
        self._results: dict[tuple[str, str, str, int], CalibrationSample] = {}
        self._result_plan_fingerprints: dict[tuple[str, str, str, int], str] = {}
        self._errors: dict[tuple[str, str, str], str] = {}
        self.state = CalibrationState.DISABLED
        self.state_reason = "engine_rebalancing_disabled"
        self.state_since = time.time()
        self.active_plan_fingerprint: str | None = None

    def transition(
        self,
        state: CalibrationState,
        reason: str,
        *,
        plan_fingerprint: str | None = None,
    ) -> None:
        if state is not self.state:
            self.state_since = time.time()
        self.state = state
        self.state_reason = str(reason)
        if plan_fingerprint is not None:
            self.active_plan_fingerprint = plan_fingerprint

    async def execute(
        self,
        plan: CalibrationPlan,
        benchmark: Benchmark | None = None,
    ) -> None:
        if benchmark is None:
            for task in plan.tasks:
                self._errors[(task.source_node, task.target_node, task.link_type)] = (
                    "node-local calibration backend unavailable"
                )
            return
        for task in plan.tasks:
            try:
                for payload in task.payload_buckets:
                    measured: list[CalibrationSample] = []
                    last_error: Exception | None = None
                    try:
                        # One warm-up plus eight measured samples.
                        warmup = benchmark(task, payload)
                        if inspect.isawaitable(warmup):
                            await warmup
                    except Exception as exc:  # pragma: no cover - backend-specific
                        last_error = exc
                    attempts = 0
                    while len(measured) < 8 and attempts < 16:
                        attempts += 1
                        try:
                            value = benchmark(task, payload)
                            if inspect.isawaitable(value):
                                value = await value
                            if not isinstance(value, CalibrationSample):
                                raise TypeError(
                                    "calibration benchmark must return "
                                    "CalibrationSample"
                                )
                            measured.append(value)
                        except Exception as exc:  # pragma: no cover - backend-specific
                            last_error = exc
                    if len(measured) < 8:
                        self._errors[
                            (task.source_node, task.target_node, task.link_type)
                        ] = f"collected {len(measured)}/8 valid samples: {last_error!r}"
                        continue
                    result_key = (
                        task.source_node,
                        task.target_node,
                        task.link_type,
                        payload,
                    )
                    self._results[result_key] = CalibrationSample(
                        elapsed_seconds_p75=_percentile(
                            [item.elapsed_seconds_p75 for item in measured], 0.75
                        ),
                        bandwidth_bytes_per_second_p25=_percentile(
                            [item.bandwidth_bytes_per_second_p25 for item in measured],
                            0.25,
                        ),
                        payload_bytes=payload,
                        sample_count=len(measured),
                    )
                    self._result_plan_fingerprints[result_key] = plan.fingerprint
                    if all(
                        (
                            task.source_node,
                            task.target_node,
                            task.link_type,
                            candidate_payload,
                        )
                        in self._results
                        for candidate_payload in task.payload_buckets
                    ):
                        self._errors.pop(
                            (task.source_node, task.target_node, task.link_type), None
                        )
            finally:
                finish_task = getattr(benchmark, "finish_task", None)
                if callable(finish_task):
                    finished = finish_task(task)
                    if inspect.isawaitable(finished):
                        await finished

    def plan_complete(self, plan: CalibrationPlan) -> bool:
        return all(
            (task.source_node, task.target_node, task.link_type, payload)
            in self._results
            for task in plan.tasks
            for payload in task.payload_buckets
        )

    def completed_links(self, source_node: str, target_node: str) -> set[str]:
        return {
            link
            for source, target, link, _ in self._results
            if source == source_node and target == target_node
        }

    def plan_progress(self, plan: CalibrationPlan) -> list[dict[str, Any]]:
        progress: list[dict[str, Any]] = []
        for task in plan.tasks:
            completed = [
                payload
                for payload in task.payload_buckets
                if (
                    task.source_node,
                    task.target_node,
                    task.link_type,
                    payload,
                )
                in self._results
            ]
            pending = [
                payload for payload in task.payload_buckets if payload not in completed
            ]
            progress.append(
                {
                    **asdict(task),
                    "path_fingerprint": canonical_fingerprint(
                        {
                            "plan_fingerprint": plan.fingerprint,
                            "source_node": task.source_node,
                            "target_node": task.target_node,
                            "link_type": task.link_type,
                        }
                    ),
                    "completed_payloads": completed,
                    "pending_payloads": pending,
                }
            )
        return progress

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "state_reason": self.state_reason,
            "state_since": self.state_since,
            "plan_fingerprint": self.active_plan_fingerprint,
            "samples": len(self._results),
            "completed_links": sorted(
                {
                    f"{source}->{target}:{link}"
                    for source, target, link, _ in self._results
                }
            ),
            "errors": {
                f"{source}->{target}:{link}": error
                for (source, target, link), error in self._errors.items()
            },
            "results": [
                {
                    "source_node": source,
                    "target_node": target,
                    "link_type": link,
                    "path_fingerprint": canonical_fingerprint(
                        {
                            "plan_fingerprint": self._result_plan_fingerprints.get(
                                (source, target, link, payload),
                                self.active_plan_fingerprint,
                            ),
                            "source_node": source,
                            "target_node": target,
                            "link_type": link,
                        }
                    ),
                    "payload_bytes": payload,
                    **asdict(sample),
                }
                for (source, target, link, payload), sample in self._results.items()
            ],
        }
