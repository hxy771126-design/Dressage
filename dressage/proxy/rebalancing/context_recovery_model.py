"""Online queue, prefill, decode, and context preparation estimates."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque

from .cache_hit_estimator import CacheSource, ContextRecoveryEstimate, context_bucket


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def load_bucket(running: int) -> str:
    value = max(0, int(running))
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    lower = 2
    upper = 3
    while value > upper and upper < 255:
        lower *= 2
        upper = upper * 2 + 1
    return "256+" if value > 255 else f"{lower}-{upper}"


def projected_load_bucket(score: float) -> str:
    # Quarter-point buckets retain token/queue pressure without creating an
    # impractically sparse multidimensional history table.
    lower = math.floor(max(0.0, float(score)) * 4.0) / 4.0
    return f"{lower:.2f}-{lower + 0.25:.2f}"


@dataclass(frozen=True)
class RequestPerformanceObservation:
    queue_seconds: float | None
    predicted_queue_seconds: float | None
    queue_prediction_error_seconds: float | None
    context_seconds: float | None
    prefill_tokens: int
    cached_tokens: int
    prefill_throughput: float | None
    tpot_seconds: float | None


class PerformanceHistory:
    def __init__(self, *, history_size: int = 256, min_samples: int = 32) -> None:
        self.history_size = history_size
        self.min_samples = min_samples
        self._queue: dict[tuple[str, str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._queue_pool: dict[tuple[str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._queue_error: dict[tuple[str, str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._queue_error_pool: dict[tuple[str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._prefill: dict[tuple[str, str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._prefill_pool: dict[tuple[str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._tpot: dict[tuple[str, str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._context_error: dict[tuple[str, CacheSource, str], Deque[float]] = (
            defaultdict(lambda: deque(maxlen=history_size))
        )

    def observe(
        self,
        *,
        fingerprint: str,
        engine_url: str,
        running: int,
        context_tokens: int,
        queue_seconds: float | None,
        context_seconds: float | None,
        cached_tokens: int,
        output_tokens: int,
        decode_throughput: float | None,
        projected_load_score: float | None = None,
        predicted_queue_seconds: float | None = None,
        estimated_context_seconds: float | None = None,
        cache_source: CacheSource | None = None,
    ) -> RequestPerformanceObservation:
        queue_buckets = {load_bucket(running)}
        if projected_load_score is not None:
            queue_buckets.add(projected_load_bucket(projected_load_score))
        cb = context_bucket(context_tokens)
        queue_value = None if queue_seconds is None else max(0.0, float(queue_seconds))
        predicted_queue_value = (
            None
            if predicted_queue_seconds is None
            else max(0.0, float(predicted_queue_seconds))
        )
        queue_prediction_error = (
            None
            if queue_value is None or predicted_queue_value is None
            else abs(predicted_queue_value - queue_value)
        )
        context_value = (
            None if context_seconds is None else max(0.0, float(context_seconds))
        )
        if queue_value is not None:
            for bucket in queue_buckets:
                self._queue[(fingerprint, engine_url, bucket)].append(queue_value)
                self._queue_pool[(fingerprint, bucket)].append(queue_value)
                if queue_prediction_error is not None:
                    self._queue_error[(fingerprint, engine_url, bucket)].append(
                        queue_prediction_error
                    )
                    self._queue_error_pool[(fingerprint, bucket)].append(
                        queue_prediction_error
                    )

        actual_cached = max(0, min(int(context_tokens), int(cached_tokens)))
        prefill_tokens = max(0, int(context_tokens) - actual_cached)
        prefill_throughput = None
        # A cached request's context phase includes native restore work which
        # the Proxy cannot split reliably.  Learn prefill throughput only from
        # an observed full-prefill path; cached requests still calibrate the
        # end-to-end context-error and cache-hit models below.
        full_prefill_observation = cache_source in (None, CacheSource.NONE)
        if (
            prefill_tokens > 0
            and context_value is not None
            and context_value > 0
            and full_prefill_observation
        ):
            prefill_throughput = prefill_tokens / context_value
            self._prefill[(fingerprint, engine_url, cb)].append(prefill_throughput)
            self._prefill_pool[(fingerprint, cb)].append(prefill_throughput)

        tpot = None
        if decode_throughput is not None and decode_throughput > 0:
            tpot = 1.0 / decode_throughput
            for bucket in queue_buckets:
                self._tpot[(fingerprint, engine_url, bucket)].append(tpot)
        elif output_tokens > 0:
            # The caller already removed queue/context time.  Do not fabricate a
            # TPOT sample unless SGLang reports decode throughput.
            tpot = None

        if (
            estimated_context_seconds is not None
            and cache_source is not None
            and context_value is not None
        ):
            self._context_error[(fingerprint, cache_source, cb)].append(
                abs(float(estimated_context_seconds) - context_value)
            )
        return RequestPerformanceObservation(
            queue_seconds=queue_value,
            predicted_queue_seconds=predicted_queue_value,
            queue_prediction_error_seconds=queue_prediction_error,
            context_seconds=context_value,
            prefill_tokens=prefill_tokens,
            cached_tokens=actual_cached,
            prefill_throughput=prefill_throughput,
            tpot_seconds=tpot,
        )

    def queue_seconds(
        self,
        *,
        fingerprint: str,
        engine_url: str,
        projected_running: int,
        projected_load_score: float | None = None,
    ) -> float | None:
        buckets = []
        if projected_load_score is not None:
            buckets.append(projected_load_bucket(projected_load_score))
        buckets.append(load_bucket(projected_running))
        for bucket in dict.fromkeys(buckets):
            samples = list(self._queue[(fingerprint, engine_url, bucket)])
            if len(samples) < self.min_samples:
                samples = list(self._queue_pool[(fingerprint, bucket)])
            if len(samples) >= self.min_samples:
                return percentile(samples, 0.75)
        return None

    def queue_risk_seconds(
        self,
        *,
        fingerprint: str,
        engine_url: str,
        projected_running: int,
        projected_load_score: float | None = None,
    ) -> float:
        """Return P90 absolute queue-time prediction error for a candidate."""

        buckets = []
        if projected_load_score is not None:
            buckets.append(projected_load_bucket(projected_load_score))
        buckets.append(load_bucket(projected_running))
        for bucket in dict.fromkeys(buckets):
            samples = list(self._queue_error[(fingerprint, engine_url, bucket)])
            if len(samples) < self.min_samples:
                samples = list(self._queue_error_pool[(fingerprint, bucket)])
            if len(samples) >= self.min_samples:
                return percentile(samples, 0.90)
        return 0.0

    def prefill_throughput(
        self,
        *,
        fingerprint: str,
        engine_url: str,
        context_tokens: int,
    ) -> float | None:
        bucket = context_bucket(context_tokens)
        samples = list(self._prefill[(fingerprint, engine_url, bucket)])
        if len(samples) < self.min_samples:
            samples = list(self._prefill_pool[(fingerprint, bucket)])
        if len(samples) < self.min_samples:
            return None
        return max(1e-6, percentile(samples, 0.25))

    def tpot_seconds(
        self,
        *,
        fingerprint: str,
        engine_url: str,
        projected_running: int,
        projected_load_score: float | None = None,
    ) -> float | None:
        buckets = []
        if projected_load_score is not None:
            buckets.append(projected_load_bucket(projected_load_score))
        buckets.append(load_bucket(projected_running))
        for bucket in dict.fromkeys(buckets):
            samples = list(self._tpot[(fingerprint, engine_url, bucket)])
            if len(samples) < self.min_samples:
                samples = [
                    item
                    for (pool_fingerprint, _, pool_bucket), values in self._tpot.items()
                    if pool_fingerprint == fingerprint and pool_bucket == bucket
                    for item in values
                ]
            if len(samples) >= self.min_samples:
                return percentile(samples, 0.75)
        return None

    def risk_seconds(
        self,
        *,
        fingerprint: str,
        source: CacheSource,
        context_tokens: int,
        minimum_seconds: float,
    ) -> float:
        samples = list(
            self._context_error[(fingerprint, source, context_bucket(context_tokens))]
        )
        if len(samples) < self.min_samples:
            return minimum_seconds
        return max(minimum_seconds, percentile(samples, 0.90))

    def queue_ready(self, fingerprint: str) -> bool:
        return any(
            pool_fingerprint == fingerprint and len(values) >= self.min_samples
            for (pool_fingerprint, _), values in self._queue_pool.items()
        )

    def prefill_ready(self, fingerprint: str) -> bool:
        return any(
            pool_fingerprint == fingerprint and len(values) >= self.min_samples
            for (pool_fingerprint, _), values in self._prefill_pool.items()
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "queue_samples": sum(len(values) for values in self._queue.values()),
            "queue_error_samples": sum(
                len(values) for values in self._queue_error.values()
            ),
            "prefill_samples": sum(len(values) for values in self._prefill.values()),
            "tpot_samples": sum(len(values) for values in self._tpot.values()),
            "context_error_samples": sum(
                len(values) for values in self._context_error.values()
            ),
            "min_samples": self.min_samples,
        }


class ContextRecoveryModel:
    def __init__(self, performance: PerformanceHistory) -> None:
        self.performance = performance

    def estimate(
        self,
        *,
        fingerprint: str,
        engine_url: str,
        cache_source: CacheSource,
        context_tokens: int,
        base_tokens: int,
        hit_probability: float,
        restore_seconds: float | None,
        restore_sample_source: str = "none",
    ) -> ContextRecoveryEstimate | None:
        throughput = self.performance.prefill_throughput(
            fingerprint=fingerprint,
            engine_url=engine_url,
            context_tokens=context_tokens,
        )
        if throughput is None:
            return None
        total_tokens = max(0, int(context_tokens))
        reusable = max(0, min(total_tokens, int(base_tokens)))
        probability = max(0.0, min(1.0, float(hit_probability)))
        if cache_source is CacheSource.NONE or reusable == 0:
            probability = 0.0
            source = CacheSource.NONE
            estimated = total_tokens / throughput
        else:
            source = cache_source
            if restore_seconds is None:
                return None
            hit_seconds = (
                max(0.0, restore_seconds) + (total_tokens - reusable) / throughput
            )
            miss_seconds = total_tokens / throughput
            estimated = probability * hit_seconds + (1.0 - probability) * miss_seconds
        expected_cached = int(math.floor(probability * reusable))
        return ContextRecoveryEstimate(
            cache_source=source,
            expected_cached_tokens=expected_cached,
            expected_prefill_tokens=max(0, total_tokens - expected_cached),
            estimated_seconds=max(0.0, estimated),
            hit_probability=probability,
            restore_seconds=(
                0.0 if restore_seconds is None else max(0.0, restore_seconds)
            ),
            restore_sample_source=restore_sample_source,
        )
