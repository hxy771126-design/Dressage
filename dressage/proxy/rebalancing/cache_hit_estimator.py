"""Proxy-side cache-token estimates derived from sticky-session history."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque


class CacheSource(str, Enum):
    NONE = "none"
    LOCAL = "local"
    MOONCAKE = "mooncake"


@dataclass(frozen=True)
class ContextRecoveryEstimate:
    cache_source: CacheSource
    expected_cached_tokens: int
    expected_prefill_tokens: int
    estimated_seconds: float
    hit_probability: float
    restore_seconds: float = 0.0
    restore_sample_source: str = "none"


@dataclass(frozen=True)
class CacheObservation:
    estimated_base_tokens: int
    actual_cached_tokens: int
    context_tokens: int


def longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    size = min(len(left), len(right))
    index = 0
    while index < size and left[index] == right[index]:
        index += 1
    return index


def context_bucket(tokens: int) -> str:
    value = max(0, int(tokens))
    if value <= 8 * 1024:
        return "0-8k"
    if value <= 16 * 1024:
        return "8-16k"
    if value <= 32 * 1024:
        return "16-32k"
    if value <= 64 * 1024:
        return "32-64k"
    return "64k+"


class CacheHitEstimator:
    def __init__(
        self,
        *,
        history_size: int = 256,
        min_samples: int = 32,
        cold_start_probability: float = 0.1,
    ) -> None:
        self.history_size = history_size
        self.min_samples = min_samples
        self.cold_start_probability = cold_start_probability
        self._history: dict[
            tuple[str, str, CacheSource, str], Deque[CacheObservation]
        ] = defaultdict(lambda: deque(maxlen=self.history_size))
        self._pool_history: dict[
            tuple[str, CacheSource, str], Deque[CacheObservation]
        ] = defaultdict(lambda: deque(maxlen=self.history_size))

    def observe(
        self,
        *,
        fingerprint: str,
        engine_url: str,
        cache_source: CacheSource,
        estimated_base_tokens: int,
        actual_cached_tokens: int,
        context_tokens: int,
    ) -> None:
        observation = CacheObservation(
            estimated_base_tokens=max(0, int(estimated_base_tokens)),
            actual_cached_tokens=max(0, int(actual_cached_tokens)),
            context_tokens=max(0, int(context_tokens)),
        )
        bucket = context_bucket(context_tokens)
        self._history[(fingerprint, engine_url, cache_source, bucket)].append(
            observation
        )
        self._pool_history[(fingerprint, cache_source, bucket)].append(observation)

    def snapshot(self) -> dict[str, Any]:
        return {
            "engine_series": len(self._history),
            "pool_series": len(self._pool_history),
            "samples": sum(len(values) for values in self._history.values()),
            "min_samples": self.min_samples,
            "cold_start_probability": self.cold_start_probability,
        }
