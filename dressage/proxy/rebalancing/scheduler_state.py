"""Runtime state for Proxy-side SGLang engine rebalancing.

The state machine is deliberately independent from SGLang cache internals.  It
only answers whether a compatible engine pool has enough fresh performance data
to make routing decisions.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class SchedulerState(str, Enum):
    OFF = "OFF"
    BOOTSTRAP = "BOOTSTRAP"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"


@dataclass
class EngineRebalancingConfig:
    enabled: bool = False
    load_poll_interval_ms: int = 250
    load_batch_coalescing_window_ms: int = 125
    metrics_stale_ms: int = 2_000
    history_size: int = 512
    min_samples: int = 16
    min_hold_turns: int = 2
    min_risk_ms: int = 10
    cold_start_hit_probability: float = 1.0
    min_load_improvement_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.load_poll_interval_ms <= 0:
            raise ValueError("load_poll_interval_ms must be positive")
        if self.load_batch_coalescing_window_ms < 0:
            raise ValueError(
                "load_batch_coalescing_window_ms must be non-negative"
            )
        if self.metrics_stale_ms <= 0:
            raise ValueError("metrics_stale_ms must be positive")
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.min_hold_turns < 0:
            raise ValueError("min_hold_turns must be non-negative")
        if self.min_risk_ms < 0:
            raise ValueError("min_risk_ms must be non-negative")
        if not 0.0 <= self.cold_start_hit_probability <= 1.0:
            raise ValueError("cold_start_hit_probability must be between 0 and 1")
        if not 0.0 <= self.min_load_improvement_ratio <= 1.0:
            raise ValueError("min_load_improvement_ratio must be between 0 and 1")

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PoolReadiness:
    healthy_engines: int
    metrics_fresh: bool
    model_cache_profile_ready: bool
    queue_model_ready: bool
    prefill_model_ready: bool
    eligible_paths: int

    @property
    def ready(self) -> bool:
        return (
            self.healthy_engines >= 2
            and self.metrics_fresh
            and self.eligible_paths > 0
        )

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ready"] = self.ready
        return payload


class CompatibilityPoolStateMachine:
    """Four-state load-routing lifecycle for one cache fingerprint."""

    def __init__(
        self,
        fingerprint: str,
        config: EngineRebalancingConfig,
        *,
        now: float | None = None,
    ) -> None:
        self.fingerprint = fingerprint
        self.config = config
        self.state = SchedulerState.BOOTSTRAP if config.enabled else SchedulerState.OFF
        self.state_since = time.time() if now is None else float(now)
        self.transition_reason = (
            "engine_rebalancing_enabled"
            if config.enabled
            else "engine_rebalancing_disabled"
        )
        self.last_readiness: PoolReadiness | None = None
        self._was_active = False

    def update(
        self,
        readiness: PoolReadiness,
        *,
        reason: str | None = None,
        now: float | None = None,
    ) -> SchedulerState:
        self.last_readiness = readiness
        if not self.config.enabled:
            self._transition(
                SchedulerState.OFF,
                reason or "engine_rebalancing_disabled",
                now=now,
            )
            return self.state

        if readiness.ready:
            self._was_active = True
            self._transition(
                SchedulerState.ACTIVE,
                reason or "pool_readiness_satisfied",
                now=now,
            )
        elif self._was_active:
            self._transition(
                SchedulerState.DEGRADED,
                reason or "pool_readiness_lost",
                now=now,
            )
        else:
            self._transition(
                SchedulerState.BOOTSTRAP,
                reason or "pool_readiness_pending",
                now=now,
            )
        return self.state

    def _transition(
        self,
        state: SchedulerState,
        reason: str,
        *,
        now: float | None,
    ) -> None:
        if state == self.state:
            self.transition_reason = reason
            return
        self.state = state
        self.state_since = time.time() if now is None else float(now)
        self.transition_reason = reason

    def snapshot(self) -> dict[str, Any]:
        return {
            "cache_fingerprint": self.fingerprint,
            "state": self.state.value,
            "state_since": self.state_since,
            "transition_reason": self.transition_reason,
            "readiness": (
                None if self.last_readiness is None else self.last_readiness.snapshot()
            ),
        }
