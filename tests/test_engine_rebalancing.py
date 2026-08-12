from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from dressage.proxy.rebalancing import EngineRebalancer, EngineRebalancingConfig
from dressage.proxy.rebalancing.cache_hit_estimator import (
    CacheHitEstimator,
    CacheSource,
    ContextRecoveryEstimate,
    context_bucket,
    longest_common_prefix_length,
)
from dressage.proxy.rebalancing.context_recovery_model import (
    ContextRecoveryModel,
    PerformanceHistory,
)
from dressage.proxy.rebalancing.model_cache_profile import ModelCacheProfile
from dressage.proxy.rebalancing.scheduler import (
    EngineDeploymentInfo,
    EngineLoad,
    GroupLengthEstimator,
    RoutingDecision,
    RoutingLease,
    SessionRoutingState,
    StepGenerationBudget,
    StepLengthEstimator,
    sglang_rebalancing_supported,
)
from dressage.proxy.rebalancing.scheduler_state import (
    CompatibilityPoolStateMachine,
    PoolReadiness,
    SchedulerState,
)
from dressage.proxy.rebalancing.snapshot_store import CalibrationSnapshotStore
from dressage.proxy.server import _settle_routing_lease, create_app, parse_args
from dressage.proxy.sglang_client import SGLangResponse, SGLangRouterClient
from dressage.proxy.rebalancing.ray_calibration import MachineCalibrationConfig
from dressage.proxy.rebalancing.transfer_calibrator import (
    CalibrationPlan,
    CalibrationSample,
    CalibrationState,
    CalibrationTask,
    TransferCalibrator,
)
from tests.test_proxy import FakeTokenizer


def run(coro):
    return asyncio.run(coro)


def simple_model_config():
    return {
        "hidden_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": 4,
        "torch_dtype": "bfloat16",
    }


class ControlPlaneClient:
    def __init__(self, *, shared_l3: bool = False):
        self.urls = ["http://node-a:30000", "http://node-b:30000"]
        self.shared_l3 = shared_l3

    async def list_workers(self):
        return [
            {"url": url, "is_healthy": True, "connection_mode": "http"}
            for url in self.urls
        ]

    async def get_worker_loads(self, url):
        del url
        return {
            "loads": [
                {
                    "num_running_reqs": 0,
                    "num_waiting_reqs": 0,
                    "num_total_tokens": 0,
                    "max_total_num_tokens": 100_000,
                    "max_running_requests": 100,
                    "token_usage": 0.0,
                }
            ]
        }

    async def get_server_info(self, url):
        del url
        return {
            "version": "0.5.15.post1",
            "server_args": {
                "tp_size": 1,
                "pp_size": 1,
                "dp_size": 1,
                "dtype": "bfloat16",
                "kv_cache_dtype": "bfloat16",
                "page_size": 1,
                "enable_hierarchical_cache": self.shared_l3,
                "hicache_storage_backend": "mooncake" if self.shared_l3 else None,
            },
        }

    async def get_worker_weight_version(self, url):
        del url
        return "7"


class DirectGenerationClient(ControlPlaneClient):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def generate(
        self,
        input_ids,
        sampling_params,
        *,
        routing_key=None,
        request_id=None,
        logprob_start_len=0,
        worker_url=None,
    ):
        self.calls.append(
            {
                "input_ids": list(input_ids),
                "sampling_params": dict(sampling_params),
                "routing_key": routing_key,
                "request_id": request_id,
                "worker_url": worker_url,
            }
        )
        output = [ord("x")]
        return SGLangResponse(
            input_token_ids=list(input_ids),
            input_token_logprobs_raw=[0.0] * len(input_ids),
            input_token_texts=[""] * len(input_ids),
            output_ids=output,
            output_token_logprobs=[-0.1],
            output_token_texts=["x"],
            output_versions=["7"],
            all_token_ids=list(input_ids) + output,
            all_logprobs=[0.0] * len(input_ids) + [-0.1],
            text="x",
            meta_info={
                "weight_version": "7",
                "cached_tokens": 0,
                "queue_time": 0.0,
                "e2e_latency": 1.0,
                "decode_throughput": 10.0,
            },
            finish_reason="stop",
        )

    async def abort_request(self, request_id, **kwargs):
        return {"success": True, "rid": request_id, **kwargs}

    async def list_models(self):
        return {"object": "list", "data": [{"id": "model"}]}

    async def close(self):
        return None


def test_config_derives_metrics_staleness():
    config = EngineRebalancingConfig(load_poll_interval_ms=750)
    assert config.metrics_stale_ms == 3_000


def test_config_defaults_propagate_to_online_models():
    config = EngineRebalancingConfig(enabled=True)
    assert config.snapshot()["load_poll_interval_ms"] == 125
    assert config.snapshot()["history_size"] == 128
    assert config.snapshot()["min_samples"] == 16
    assert config.snapshot()["min_hold_turns"] == 2
    assert config.snapshot()["min_risk_ms"] == 10
    assert config.snapshot()["cold_start_hit_probability"] == 1.0
    assert config.snapshot()["min_load_improvement_ratio"] == 0.20

    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=config,
        model_id="model",
        model_config=simple_model_config(),
    )
    assert rebalancer.performance.min_samples == 16
    assert rebalancer.cache_hits.min_samples == 16
    assert rebalancer.cache_hits.cold_start_probability == 1.0
    assert rebalancer.step_lengths.min_samples == 16


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_config_rejects_out_of_range_load_improvement_ratio(value):
    with pytest.raises(ValueError, match="min_load_improvement_ratio"):
        EngineRebalancingConfig(min_load_improvement_ratio=value)


def test_load_score_uses_reservation_envelopes_and_adds_pending_step_once():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )

    async def scenario():
        await rebalancer.refresh()
        target = client.urls[0]
        load = rebalancer.loads[target]
        load.running = 2
        load.reserved_requests = 2
        load.queued = 3
        load.active_tokens = 1_000
        load.reserved_tokens = 1_000
        load.token_usage = 0.005
        load.waiting_uncached_tokens = 80
        load.reserved_prefill_tokens = 80

        base = rebalancer._load_score(
            target,
            prompt_tokens=100,
            expected_output_tokens=50,
            include_pending_request=False,
        )
        projected = rebalancer._load_score(
            target,
            prompt_tokens=100,
            expected_output_tokens=50,
            include_pending_request=True,
        )

        assert base.request_pressure == pytest.approx(0.02)
        assert base.token_pressure == pytest.approx(0.01)
        assert base.queue_pressure == pytest.approx(0.03)
        assert base.prefill_pressure == pytest.approx(0.0008)
        assert base.total == pytest.approx(0.0608)
        assert projected.request_pressure == pytest.approx(0.03)
        assert projected.token_pressure == pytest.approx(0.0115)
        assert projected.queue_pressure == base.queue_pressure
        assert projected.prefill_pressure == base.prefill_pressure
        assert projected.total == pytest.approx(0.0723)

        load.running = 1
        load.reserved_requests = 3
        load.active_tokens = 800
        load.reserved_tokens = 1_200
        load.waiting_uncached_tokens = 60
        load.reserved_prefill_tokens = 80
        reservation_dominates = rebalancer._load_score(
            target,
            prompt_tokens=0,
            include_pending_request=False,
        )
        assert reservation_dominates.request_pressure == pytest.approx(0.03)
        assert reservation_dominates.token_pressure == pytest.approx(0.012)
        assert reservation_dominates.prefill_pressure == pytest.approx(0.0008)

        load.running = 4
        load.reserved_requests = 2
        load.active_tokens = 1_400
        load.reserved_tokens = 1_000
        load.waiting_uncached_tokens = 100
        load.reserved_prefill_tokens = 60
        scheduler_load_dominates = rebalancer._load_score(
            target,
            prompt_tokens=0,
            include_pending_request=False,
        )
        assert scheduler_load_dominates.request_pressure == pytest.approx(0.04)
        assert scheduler_load_dominates.token_pressure == pytest.approx(0.014)
        assert scheduler_load_dominates.prefill_pressure == pytest.approx(0.001)

    run(scenario())


def test_state_machine_distinguishes_bootstrap_and_degraded():
    config = EngineRebalancingConfig(enabled=True)
    state = CompatibilityPoolStateMachine("fp", config, now=1.0)
    not_ready = PoolReadiness(2, True, True, False, False, 0)
    ready = PoolReadiness(2, True, True, True, True, 1)

    assert state.update(not_ready, now=2.0) is SchedulerState.BOOTSTRAP
    assert state.update(ready, now=3.0) is SchedulerState.ACTIVE
    assert state.update(not_ready, now=4.0) is SchedulerState.DEGRADED
    assert state.update(ready, now=5.0) is SchedulerState.ACTIVE


def test_pool_readiness_does_not_require_performance_history():
    readiness = PoolReadiness(
        healthy_engines=2,
        metrics_fresh=True,
        model_cache_profile_ready=False,
        queue_model_ready=False,
        prefill_model_ready=False,
        eligible_paths=1,
    )

    assert readiness.ready is True


def test_model_cache_profile_uses_context_and_dtype():
    profile = ModelCacheProfile.from_model_config(
        simple_model_config(),
        deployment={"kv_dtype": "bfloat16", "page_size": 16},
    )
    # 32 tokens * K/V * 4 layers * 2 KV heads * 16 head dim * 2 bytes.
    assert profile.estimate_bytes(32) == 32 * 2 * 4 * 2 * 16 * 2
    assert profile.estimate_bytes(64) == 2 * profile.estimate_bytes(32)


def test_model_cache_profile_limits_swa_to_page_rounded_resident_window():
    profile = ModelCacheProfile.from_model_config(
        {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_hidden_layers": 4,
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            "sliding_window": 50,
            "torch_dtype": "bfloat16",
        },
        deployment={"kv_dtype": "bfloat16", "page_size": 16},
    )
    full = 100 * 2 * 2 * 2 * 16 * 2
    swa = 64 * 2 * 2 * 2 * 16 * 2
    assert profile.full_layers == 2
    assert profile.swa_layers == 2
    assert profile.estimate_bytes(100) == full + swa


def test_model_cache_profile_unwraps_qwen35_text_config_and_counts_gdn_state():
    profile = ModelCacheProfile.from_model_config(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 2560,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "num_hidden_layers": 4,
                "head_dim": 256,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "dtype": "bfloat16",
                "mamba_ssm_dtype": "float32",
            },
        },
        deployment={"kv_dtype": "bfloat16", "mamba_track_interval": 256},
    )
    temporal = 32 * 128 * 128 * 4
    conv = (2 * 16 * 128 + 32 * 128) * 3 * 2
    assert profile.full_layers == 1
    assert profile.state_bytes_per_checkpoint == 3 * (temporal + conv)
    assert profile.confidence == "config"
    assert profile.estimate_bytes(1) - (2 * 1 * 4 * 256 * 2) == (
        profile.state_bytes_per_checkpoint
    )
    assert profile.estimate_bytes(1024) - (1024 * 2 * 1 * 4 * 256 * 2) == (
        profile.state_bytes_per_checkpoint
    )


def test_qwen35_cache_profile_regression_uses_one_tail_state_slot():
    profile = ModelCacheProfile(
        fingerprint="qwen35-4b",
        full_layers=8,
        full_kv_heads=4,
        full_head_dim=256,
        full_dtype_bytes=2,
        state_bytes_per_checkpoint=51_511_296,
        state_checkpoint_interval=256,
    )
    assert profile.estimate_bytes(8 * 1024) == 319_946_752
    assert profile.estimate_bytes(56 * 1024) == 1_930_559_488


def test_cache_hit_estimator_uses_lcp_and_cold_start():
    estimator = CacheHitEstimator(min_samples=2, cold_start_probability=0.1)
    assert longest_common_prefix_length([1, 2, 3], [1, 2, 9]) == 2
    assert (
        estimator.estimate_probability(
            fingerprint="fp",
            engine_url="worker",
            cache_source=CacheSource.MOONCAKE,
            context_tokens=100,
        )
        == 0.1
    )
    assert (
        estimator.estimate_probability(
            fingerprint="fp",
            engine_url="worker",
            cache_source=CacheSource.NONE,
            context_tokens=100,
        )
        == 0.0
    )
    assert (
        estimator.estimate_probability(
            fingerprint="fp",
            engine_url="worker",
            cache_source=CacheSource.LOCAL,
            context_tokens=100,
        )
        == 1.0
    )


def test_default_mooncake_prior_switches_to_observed_p25_at_16_samples():
    config = EngineRebalancingConfig()
    estimator = CacheHitEstimator(
        history_size=config.history_size,
        min_samples=config.min_samples,
        cold_start_probability=config.cold_start_hit_probability,
    )

    def probability(source: CacheSource) -> float:
        return estimator.estimate_probability(
            fingerprint="fp",
            engine_url="worker",
            cache_source=source,
            context_tokens=100,
        )

    assert probability(CacheSource.NONE) == 0.0
    assert probability(CacheSource.LOCAL) == 1.0
    assert probability(CacheSource.MOONCAKE) == 1.0

    for _ in range(15):
        estimator.observe(
            fingerprint="fp",
            engine_url="worker",
            cache_source=CacheSource.MOONCAKE,
            estimated_base_tokens=100,
            actual_cached_tokens=50,
            context_tokens=100,
        )
    assert probability(CacheSource.MOONCAKE) == 1.0

    estimator.observe(
        fingerprint="fp",
        engine_url="worker",
        cache_source=CacheSource.MOONCAKE,
        estimated_base_tokens=100,
        actual_cached_tokens=50,
        context_tokens=100,
    )
    assert probability(CacheSource.MOONCAKE) == 0.5


def test_group_remaining_length_uses_group_then_task_history():
    estimator = GroupLengthEstimator(history_size=256, min_task_samples=3)
    for length in (10, 20, 30):
        estimator.observe(group_id=None, task_key="task", final_length=length)

    # group_size=1 naturally uses task history; no algorithm name is involved.
    assert (
        estimator.remaining(
            group_id="single",
            task_key="task",
            generated_tokens=5,
        )
        == 25
    )
    assert (
        estimator.remaining(
            group_id="new",
            task_key="unknown",
            generated_tokens=5,
        )
        is None
    )

    estimator.observe(group_id="g", task_key="task", final_length=40)
    estimator.observe(group_id="g", task_key="task", final_length=60)
    assert (
        estimator.remaining(
            group_id="g",
            task_key="task",
            generated_tokens=10,
        )
        == 50
    )


def test_step_length_estimator_uses_task_p75_then_pool_fallback():
    estimator = StepLengthEstimator(history_size=8, min_samples=2)
    estimator.observe(
        fingerprint="fp",
        task_key="math",
        max_tokens=8192,
        output_tokens=1000,
    )
    estimator.observe(
        fingerprint="fp",
        task_key="math",
        max_tokens=8192,
        output_tokens=2000,
    )
    assert estimator.p75(fingerprint="fp", task_key="math", max_tokens=8192) == 2000
    assert estimator.p75(fingerprint="fp", task_key="other", max_tokens=8192) == 2000


def test_old_sglang_versions_are_not_rebalancing_compatible():
    assert not sglang_rebalancing_supported("0.5.12")
    assert not sglang_rebalancing_supported("v0.5.15")
    assert sglang_rebalancing_supported("0.5.15.post1")
    assert sglang_rebalancing_supported("0.5.16")


def test_context_model_none_is_full_prefill():
    performance = PerformanceHistory(min_samples=1)
    performance.observe(
        fingerprint="fp",
        engine_url="worker",
        running=1,
        context_tokens=100,
        queue_seconds=0,
        context_seconds=2,
        cached_tokens=0,
        output_tokens=1,
        decode_throughput=10,
    )
    estimate = ContextRecoveryModel(performance).estimate(
        fingerprint="fp",
        engine_url="worker",
        cache_source=CacheSource.NONE,
        context_tokens=100,
        base_tokens=80,
        hit_probability=0.9,
        restore_seconds=None,
    )
    assert estimate is not None
    assert estimate.cache_source is CacheSource.NONE
    assert estimate.expected_cached_tokens == 0
    assert estimate.expected_prefill_tokens == 100
    assert estimate.estimated_seconds == 2.0


def test_context_model_mooncake_is_expected_restore_plus_prefill():
    performance = PerformanceHistory(min_samples=1)
    performance.observe(
        fingerprint="fp",
        engine_url="worker",
        running=1,
        context_tokens=100,
        queue_seconds=0,
        context_seconds=2,
        cached_tokens=0,
        output_tokens=1,
        decode_throughput=10,
        cache_source=CacheSource.NONE,
    )
    estimate = ContextRecoveryModel(performance).estimate(
        fingerprint="fp",
        engine_url="worker",
        cache_source=CacheSource.MOONCAKE,
        context_tokens=100,
        base_tokens=80,
        hit_probability=0.5,
        restore_seconds=1.0,
    )
    assert estimate is not None
    assert estimate.cache_source is CacheSource.MOONCAKE
    assert estimate.expected_cached_tokens == 40
    assert estimate.expected_prefill_tokens == 60
    # hit: 1.0 restore + 20 / 50 prefill; miss: 100 / 50 prefill.
    assert estimate.estimated_seconds == 1.7


def test_missing_sglang_queue_timing_does_not_make_models_ready():
    performance = PerformanceHistory(min_samples=1)
    performance.observe(
        fingerprint="fp",
        engine_url="worker",
        running=1,
        context_tokens=100,
        queue_seconds=None,
        context_seconds=None,
        cached_tokens=0,
        output_tokens=1,
        decode_throughput=10,
        cache_source=CacheSource.NONE,
    )
    assert not performance.queue_ready("fp")
    assert not performance.prefill_ready("fp")


def test_default_queue_and_prefill_models_become_ready_at_16_samples():
    config = EngineRebalancingConfig()
    performance = PerformanceHistory(
        history_size=config.history_size,
        min_samples=config.min_samples,
    )

    def observe() -> None:
        performance.observe(
            fingerprint="fp",
            engine_url="worker",
            running=1,
            context_tokens=100,
            queue_seconds=0.5,
            context_seconds=1.0,
            cached_tokens=0,
            output_tokens=1,
            decode_throughput=10.0,
            cache_source=CacheSource.NONE,
        )

    for _ in range(15):
        observe()
    assert not performance.queue_ready("fp")
    assert not performance.prefill_ready("fp")
    assert (
        performance.queue_seconds(
            fingerprint="fp",
            engine_url="worker",
            projected_running=1,
        )
        is None
    )
    assert (
        performance.prefill_throughput(
            fingerprint="fp",
            engine_url="worker",
            context_tokens=100,
        )
        is None
    )

    observe()
    assert performance.queue_ready("fp")
    assert performance.prefill_ready("fp")
    assert (
        performance.queue_seconds(
            fingerprint="fp",
            engine_url="worker",
            projected_running=1,
        )
        == 0.5
    )
    assert (
        performance.prefill_throughput(
            fingerprint="fp",
            engine_url="worker",
            context_tokens=100,
        )
        == 100.0
    )


def test_queue_prediction_error_uses_p90_and_homogeneous_pool_fallback():
    history = PerformanceHistory(history_size=8, min_samples=2)
    for actual, predicted in ((1.0, 0.0), (5.0, 1.0)):
        history.observe(
            fingerprint="fp",
            engine_url="engine-a",
            running=3,
            projected_load_score=0.6,
            context_tokens=100,
            queue_seconds=actual,
            predicted_queue_seconds=predicted,
            context_seconds=1.0,
            cached_tokens=0,
            output_tokens=1,
            decode_throughput=10,
        )

    # P90 of the two absolute errors (1s and 4s) is 4s. Engine B has no
    # samples of its own, so it uses the compatible-pool history.
    assert (
        history.queue_risk_seconds(
            fingerprint="fp",
            engine_url="engine-a",
            projected_running=3,
            projected_load_score=0.6,
        )
        == 4.0
    )
    assert (
        history.queue_risk_seconds(
            fingerprint="fp",
            engine_url="engine-b",
            projected_running=3,
            projected_load_score=0.6,
        )
        == 4.0
    )

    samples_before = history.snapshot()["queue_error_samples"]
    history.observe(
        fingerprint="fp",
        engine_url="engine-a",
        running=3,
        projected_load_score=0.6,
        context_tokens=100,
        queue_seconds=2.0,
        predicted_queue_seconds=None,
        context_seconds=1.0,
        cached_tokens=0,
        output_tokens=1,
        decode_throughput=10,
    )
    assert history.snapshot()["queue_error_samples"] == samples_before


def test_context_prediction_risk_waits_for_minimum_samples():
    history = PerformanceHistory(history_size=8, min_samples=2)
    history.observe(
        fingerprint="fp",
        engine_url="engine-a",
        running=1,
        context_tokens=100,
        queue_seconds=0.0,
        context_seconds=1.0,
        cached_tokens=0,
        output_tokens=1,
        decode_throughput=10,
        estimated_context_seconds=3.0,
        cache_source=CacheSource.NONE,
    )
    assert (
        history.risk_seconds(
            fingerprint="fp",
            source=CacheSource.NONE,
            context_tokens=100,
            minimum_seconds=0.0,
        )
        == 0.0
    )

    history.observe(
        fingerprint="fp",
        engine_url="engine-a",
        running=1,
        context_tokens=100,
        queue_seconds=0.0,
        context_seconds=1.0,
        cached_tokens=0,
        output_tokens=1,
        decode_throughput=10,
        estimated_context_seconds=4.0,
        cache_source=CacheSource.NONE,
    )
    assert (
        history.risk_seconds(
            fingerprint="fp",
            source=CacheSource.NONE,
            context_tokens=100,
            minimum_seconds=0.0,
        )
        == 3.0
    )


def test_tpot_history_is_partitioned_by_engine_load_bucket():
    history = PerformanceHistory(history_size=8, min_samples=1)
    for engine, running, throughput in (("a", 1, 10.0), ("b", 8, 5.0)):
        history.observe(
            fingerprint="fp",
            engine_url=engine,
            running=running,
            context_tokens=100,
            queue_seconds=0.0,
            context_seconds=1.0,
            cached_tokens=0,
            output_tokens=10,
            decode_throughput=throughput,
        )
    assert (
        history.tpot_seconds(
            fingerprint="fp",
            engine_url="a",
            projected_running=1,
        )
        == 0.1
    )
    assert (
        history.tpot_seconds(
            fingerprint="fp",
            engine_url="b",
            projected_running=8,
        )
        == 0.2
    )


def test_calibration_plan_skips_mooncake_without_l3():
    client = ControlPlaneClient(shared_l3=False)
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )
    run(rebalancer.refresh())
    fingerprint = next(iter(rebalancer.profiles))
    plan = rebalancer.plans[fingerprint]
    assert plan.tasks == ()
    assert plan.skipped_links["mooncake"] == "L3 disabled"


def test_calibration_plan_matches_host_tcp_rdma_and_gpudirect_paths():
    class Slot:
        def __init__(self, node_id, protocol):
            self.node_id = node_id
            self.mooncake_protocol = protocol

    profile = ModelCacheProfile(
        fingerprint="profile",
        full_layers=1,
        full_kv_heads=1,
        full_head_dim=1,
        full_dtype_bytes=2,
    )
    single = CalibrationPlan.build(
        fingerprint="single",
        engine_deployments=[Slot("a", "tcp")],
        shared_l3=True,
        host_staging=True,
        gpudirect=False,
        model_cache_profile=profile,
    )
    assert single.tasks == ()
    assert single.skipped_links["migration"] == "single-engine deployment"

    tcp_slots = [Slot("a", "tcp"), Slot("b", "tcp")]
    host_plan = CalibrationPlan.build(
        fingerprint="host",
        engine_deployments=tcp_slots,
        shared_l3=True,
        host_staging=True,
        gpudirect=False,
        model_cache_profile=profile,
    )
    host_links = {task.link_type for task in host_plan.tasks}
    assert host_links == {"mooncake_local", "mooncake_tcp", "d2h", "h2d"}

    rdma_plan = CalibrationPlan.build(
        fingerprint="rdma",
        engine_deployments=[Slot("a", "rdma"), Slot("b", "rdma")],
        shared_l3=True,
        host_staging=True,
        gpudirect=False,
        model_cache_profile=profile,
    )
    assert "mooncake_rdma" in {task.link_type for task in rdma_plan.tasks}

    gpudirect_plan = CalibrationPlan.build(
        fingerprint="gpudirect",
        engine_deployments=[Slot("a", "rdma"), Slot("a", "rdma")],
        shared_l3=True,
        host_staging=False,
        gpudirect=True,
        model_cache_profile=profile,
    )
    assert {task.link_type for task in gpudirect_plan.tasks} == {"mooncake_gpudirect"}
    assert gpudirect_plan.skipped_links["h2d"] == "GPUDirect restore path"


def test_transfer_estimate_uses_complete_p75_without_bandwidth_double_count():
    calibrator = TransferCalibrator()
    for payload, elapsed in ((100, 1.0), (200, 3.0)):
        calibrator.observe(
            source_node="a",
            target_node="b",
            link_type="mooncake_tcp",
            payload_bytes=payload,
            elapsed_seconds_p75=elapsed,
            bandwidth_bytes_per_second_p25=1.0,
        )
    # 150 bytes uses the 200-byte upper bucket. A nearest lower bucket plus
    # bytes/BW would produce a much larger and incorrect value.
    assert (
        calibrator.estimate(
            source_node="a",
            target_node="b",
            required_links=("mooncake_tcp",),
            payload_bytes=150,
        )
        == 3.0
    )
    assert (
        calibrator.estimate(
            source_node="a",
            target_node="b",
            required_links=("mooncake_tcp",),
            payload_bytes=400,
        )
        == 6.0
    )


def test_calibration_releases_task_buffers_after_sample_failures():
    class FailingBenchmark:
        def __init__(self):
            self.finished = []

        async def __call__(self, task, payload):
            del task, payload
            raise TimeoutError("sample timed out")

        async def finish_task(self, task):
            self.finished.append(task)

    task = CalibrationTask("a", "b", "mooncake_tcp", (100,))
    plan = CalibrationPlan("plan", (task,), {})
    benchmark = FailingBenchmark()
    calibrator = TransferCalibrator()
    run(calibrator.execute(plan, benchmark))
    assert benchmark.finished == [task]
    assert calibrator.plan_complete(plan) is False


def test_machine_calibration_config_rejects_unknown_protocol():
    try:
        MachineCalibrationConfig.from_mapping(
            {
                "schema_version": 1,
                "ray_address": "auto",
                "hicache": {
                    "enabled": True,
                    "storage_backend": "mooncake",
                    "write_policy": "write_through",
                },
                "mooncake": {
                    "protocol": "mystery",
                    "metadata_server": "metadata",
                },
            }
        )
    except ValueError as exc:
        assert "protocol" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown Mooncake protocol was accepted")


def test_machine_path_fingerprint_tracks_topology_but_not_model_version():
    base = {
        "schema_version": 1,
        "ray_address": "auto",
        "nodes": [
            {
                "node_id": "node-a",
                "gpu_count": 2,
                "gpu_ids": [0, 1],
                "numa_node": "0",
                "nic": "eth0",
            }
        ],
        "hicache": {
            "enabled": True,
            "storage_backend": "mooncake",
            "write_policy": "write_through",
        },
        "mooncake": {
            "protocol": "tcp",
            "metadata_server": "metadata",
        },
    }
    first = MachineCalibrationConfig.from_mapping(
        {**base, "model_deployment": {"weight_version": "one"}}
    )
    second = MachineCalibrationConfig.from_mapping(
        {**base, "model_deployment": {"weight_version": "two"}}
    )
    discovered = [
        {
            "node_id": "ray-a",
            "address": "node-a",
            "gpu_count": 2,
            "hardware": [
                {
                    "gpu_uuid": "gpu-0",
                    "numa_node": "0",
                    "cuda_version": "13.0",
                    "driver_version": "999",
                    "mooncake_version": "1.0",
                }
            ],
        }
    ]
    assert first.nodes[0].gpu_ids == (0, 1)
    assert first.buffer_registration_mode == "host_pinned"
    assert first.fingerprint(discovered) == second.fingerprint(discovered)


def test_single_node_loopback_engine_maps_to_routable_calibration_node():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    deployment = EngineDeploymentInfo.from_worker(
        worker_url="http://127.0.0.1:30000",
        server_info={"version": "0.5.15.post1", "server_args": {}},
        weight_version="1",
        model_id="model",
    )
    rebalancer._preflight_node_addresses = {"10.0.0.7"}
    rebalancer._preflight_node_ids = {"ray-node", "10.0.0.7"}
    rebalancer._preflight_node_aliases = {
        "ray-node": "10.0.0.7",
        "10.0.0.7": "10.0.0.7",
    }
    assert rebalancer._calibration_node_for(deployment) == "10.0.0.7"


def test_remote_context_risk_switches_from_transport_margin_to_path_error_p90():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True, min_samples=2),
        model_id="model",
        model_config=simple_model_config(),
    )
    estimate = ContextRecoveryEstimate(
        cache_source=CacheSource.MOONCAKE,
        expected_cached_tokens=100,
        expected_prefill_tokens=0,
        estimated_seconds=4.0,
        hit_probability=1.0,
        restore_seconds=4.0,
        restore_sample_source="offline",
    )
    assert (
        rebalancer._context_prediction_risk(
            fingerprint="fp",
            source_engine="a",
            target_engine="b",
            estimate=estimate,
            context_tokens=100,
        )
        == 0.2
    )
    rebalancer._runtime_restore_errors[("fp", "a", "b", context_bucket(100))].extend(
        [0.2, 0.3]
    )
    assert (
        rebalancer._context_prediction_risk(
            fingerprint="fp",
            source_engine="a",
            target_engine="b",
            estimate=estimate,
            context_tokens=100,
        )
        == 0.3
    )


def test_default_runtime_restore_model_becomes_ready_at_16_samples():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    key = ("fp", "source", "target", context_bucket(100))
    rebalancer._runtime_restore_seconds[key].extend([0.5] * 15)
    result = rebalancer._runtime_calibration_snapshot()["results"][0]
    assert result["restore_sample_count"] == 15
    assert result["model_ready"] is False
    assert result["effective_source"] == "offline"

    rebalancer._runtime_restore_seconds[key].append(0.5)
    result = rebalancer._runtime_calibration_snapshot()["results"][0]
    assert result["restore_sample_count"] == 16
    assert result["model_ready"] is True
    assert result["effective_source"] == "runtime"


def test_shared_l3_calibration_plan_executes_required_restore_links():
    client = ControlPlaneClient(shared_l3=True)

    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )
    run(rebalancer.refresh())
    fingerprint = next(iter(rebalancer.profiles))
    plan = rebalancer.plans[fingerprint]
    assert {task.link_type for task in plan.tasks} == {
        "mooncake_local",
        "mooncake_remote",
        "d2h",
        "h2d",
    }
    assert rebalancer.calibrator.plan_complete(plan)
    readiness = rebalancer._path_readiness(client.urls[0], client.urls[1])
    assert "h2d" in readiness.required_links
    assert "d2h" not in readiness.required_links


def test_missing_l3_calibration_is_not_migration_eligible():
    client = ControlPlaneClient(shared_l3=True)
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        for url in client.urls:
            rebalancer.performance.observe(
                fingerprint=fingerprint,
                engine_url=url,
                running=1,
                context_tokens=8 * 1024,
                queue_seconds=0.0,
                context_seconds=1.0,
                cached_tokens=0,
                output_tokens=1,
                decode_throughput=10.0,
                cache_source=CacheSource.NONE,
            )
        readiness = rebalancer._path_readiness(source, target)
        assert readiness.ready is True
        assert readiness.cache_source is CacheSource.NONE
        assert "full prefill" in readiness.skipped_links["fallback"]
        pool = rebalancer._pool_readiness(fingerprint, now=time.monotonic())
        assert pool.ready is False
        assert pool.eligible_paths == 0

    run(scenario())


def test_machine_calibration_finishes_before_router_discovery(monkeypatch):
    class CountingRouterClient:
        def __init__(self):
            self.list_workers_calls = 0

        async def list_workers(self):
            self.list_workers_calls += 1
            return []

    client = CountingRouterClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    calibration_gate = asyncio.Event()

    async def blocked_calibration():
        await calibration_gate.wait()
        rebalancer.calibrator.transition(
            CalibrationState.DEGRADED,
            "test calibration complete",
        )

    monkeypatch.setattr(
        rebalancer,
        "_run_machine_preflight_impl",
        blocked_calibration,
    )

    async def scenario():
        await rebalancer.start()
        await asyncio.sleep(0)
        assert rebalancer._poll_task is None
        await rebalancer.refresh()
        assert client.list_workers_calls == 0

        calibration_gate.set()
        assert rebalancer._calibration_task is not None
        await rebalancer._calibration_task
        assert rebalancer._poll_task is not None
        assert client.list_workers_calls == 0
        await rebalancer.close()

    run(scenario())


def test_initial_snapshot_finishes_before_router_poll_starts(monkeypatch, tmp_path):
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
        calibration_snapshot_root=tmp_path,
        calibration_snapshot_run_name="startup-order",
    )
    rebalancer.machine_calibration_config = None
    write_started = asyncio.Event()
    allow_write = asyncio.Event()
    original_write = rebalancer._snapshot_store.write

    async def delayed_write(**kwargs):
        if kwargs["kind"] == "initial":
            write_started.set()
            await allow_write.wait()
        return await original_write(**kwargs)

    monkeypatch.setattr(rebalancer._snapshot_store, "write", delayed_write)

    async def scenario():
        await rebalancer.start()
        await write_started.wait()
        assert rebalancer._poll_task is None

        allow_write.set()
        assert rebalancer._calibration_task is not None
        await rebalancer._calibration_task
        assert (rebalancer._snapshot_store.directory / "initial.json").is_file()
        assert rebalancer._poll_task is not None
        await rebalancer.close()

    run(scenario())


def test_router_waiting_backoff_and_runtime_outage_logging(caplog, monkeypatch):
    class FlakyRouterClient:
        def __init__(self):
            self.responses = [
                "startup_failure",
                "startup_failure",
                "startup_failure",
                "startup_failure",
                "success",
                "outage_failure",
                "outage_failure",
                "success",
                "final_outage",
            ]
            self.rebalancer = None

        async def list_workers(self):
            response = self.responses.pop(0)
            if response == "success":
                return []
            if response == "final_outage":
                self.rebalancer._stopping = True
            raise httpx.ConnectError(response)

    client = FlakyRouterClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    client.rebalancer = rebalancer
    delays = []
    real_sleep = asyncio.sleep

    async def capture_sleep(delay):
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", capture_sleep)

    with caplog.at_level(
        logging.DEBUG,
        logger="dressage.proxy.rebalancing.scheduler",
    ):
        run(rebalancer._poll_loop())

    waiting_records = [
        record
        for record in caplog.records
        if "waiting_for_router" in record.getMessage()
    ]
    assert delays[:5] == [0.125, 1.0, 2.0, 5.0, 5.0]
    assert all(delay == 0.125 for delay in delays[5:])
    assert sum(record.levelno == logging.INFO for record in waiting_records) == 1
    assert not any(record.levelno >= logging.WARNING for record in waiting_records)
    assert all(record.exc_info is None for record in waiting_records)

    outage_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Router became unavailable" in record.getMessage()
    ]
    assert len(outage_warnings) == 2
    assert all(record.exc_info is None for record in outage_warnings)
    assert (
        sum(
            "Router connection recovered" in record.getMessage()
            for record in caplog.records
        )
        == 1
    )


def test_runtime_calibration_reports_percentiles_and_source_threshold():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True, min_samples=3),
        model_id="model",
        model_config=simple_model_config(),
    )
    ready_key = ("fp", "source", "target", "8K-16K")
    rebalancer._runtime_restore_seconds[ready_key].extend([1.0, 2.0, 3.0])
    rebalancer._runtime_restore_throughputs[ready_key].extend([100.0, 200.0, 300.0])
    rebalancer._runtime_restore_errors[ready_key].extend([0.1, 0.2, 0.3])
    cold_key = ("fp", "source", "cold-target", "8K-16K")
    rebalancer._runtime_restore_seconds[cold_key].extend([4.0, 5.0])

    results = {
        (item["source_engine"], item["target_engine"]): item
        for item in rebalancer._runtime_calibration_snapshot()["results"]
    }
    ready = results[("source", "target")]
    assert ready["restore_sample_count"] == 3
    assert ready["restore_seconds_p75"] == 3.0
    assert ready["restore_throughput_bytes_per_second_p25"] == 100.0
    assert ready["prediction_error_seconds_p90"] == 0.3
    assert ready["model_ready"] is True
    assert ready["effective_source"] == "runtime"
    assert results[("source", "cold-target")]["effective_source"] == "offline"


def test_calibration_snapshots_are_atomic_periodic_and_final(tmp_path):
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True, min_samples=3),
        model_id="model",
        model_config=simple_model_config(),
        calibration_snapshot_root=tmp_path,
        calibration_snapshot_run_name="snapshot-test",
    )
    rebalancer.machine_calibration_config = None

    async def scenario():
        await rebalancer.start()
        assert rebalancer._calibration_task is not None
        await rebalancer._calibration_task
        await rebalancer._drain_snapshot_tasks()
        directory = rebalancer._snapshot_store.directory
        assert (directory / "initial.json").is_file()
        runtime_key = ("fp", "source", "target", "8K-16K")
        rebalancer._runtime_restore_seconds[runtime_key].extend([1.0, 2.0, 3.0])
        rebalancer._runtime_restore_throughputs[runtime_key].extend(
            [100.0, 200.0, 300.0]
        )
        rebalancer._runtime_restore_errors[runtime_key].extend([0.1, 0.2, 0.3])

        for _ in range(127):
            rebalancer._record_successful_online_request()
        await rebalancer._drain_snapshot_tasks()
        assert not (directory / "request-000000127.json").exists()
        assert not (directory / "request-000000128.json").exists()

        failed = RoutingLease(
            decision=RoutingDecision(
                session_id="failed",
                source_worker_url=None,
                target_worker_url="worker",
                cache_fingerprint=None,
                state=SchedulerState.BOOTSTRAP,
                reason="test",
            ),
            worker_url="worker",
            reserved_tokens=1,
            base_tokens=0,
            started_monotonic=time.monotonic(),
        )
        await rebalancer.fail(failed)
        assert rebalancer._online_request_count == 127

        rebalancer._record_successful_online_request()
        await rebalancer._drain_snapshot_tasks()
        periodic = directory / "request-000000128.json"
        assert periodic.is_file()
        runtime_result = json.loads(periodic.read_text(encoding="utf-8"))[
            "runtime_calibration"
        ]["results"][0]
        assert runtime_result["restore_seconds_p75"] == 3.0
        assert runtime_result["restore_throughput_bytes_per_second_p25"] == 100.0
        assert runtime_result["prediction_error_seconds_p90"] == 0.3
        assert runtime_result["effective_source"] == "runtime"
        for _ in range(128):
            rebalancer._record_successful_online_request()
        await rebalancer._drain_snapshot_tasks()
        second_periodic = directory / "request-000000256.json"
        assert second_periodic.is_file()
        assert periodic.is_file()

        await rebalancer.close()
        final = directory / "final.json"
        assert final.is_file()
        for path, expected_kind in (
            (directory / "initial.json", "initial"),
            (periodic, "periodic"),
            (second_periodic, "periodic"),
            (final, "final"),
        ):
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert set(payload) == {
                "snapshot_type",
                "snapshot_time",
                "online_request_count",
                "offline_calibration",
                "runtime_calibration",
            }
            assert payload["snapshot_type"] == expected_kind
            assert "results" in payload["offline_calibration"]
            assert "results" in payload["runtime_calibration"]
        assert json.loads(final.read_text())["online_request_count"] == 256
        assert not list(directory.glob(".*.tmp"))

    run(scenario())

    first = CalibrationSnapshotStore(
        root=tmp_path,
        run_name="snapshot-test",
        started_at=1.0,
        pid=1,
    )
    second = CalibrationSnapshotStore(
        root=tmp_path,
        run_name="snapshot-test",
        started_at=2.0,
        pid=1,
    )
    assert first.directory != second.directory


def test_ray_preflight_state_is_independent_and_releases_backend(monkeypatch, tmp_path):
    config_path = tmp_path / "deployment.json"
    config_path.write_text(
        __import__("json").dumps(
            {
                "schema_version": 1,
                "ray_address": "auto",
                "nodes": [{"node_id": "node-a", "gpu_count": 2}],
                "hicache": {
                    "enabled": True,
                    "storage_backend": "mooncake",
                    "write_policy": "write_through",
                },
                "mooncake": {
                    "protocol": "tcp",
                    "metadata_server": "metadata",
                },
                "model_deployment": {"kv_dtype": "bfloat16"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "DRESSAGE_ENGINE_REBALANCING_DEPLOYMENT_CONFIG", str(config_path)
    )

    class Slot:
        node_id = "node-a"
        mooncake_protocol = "tcp"

    holder = {}

    class FakeRayBenchmark:
        instances = []
        resources_recovered = True

        def __init__(self, config):
            self.config = config
            self.closed = False
            self.state_seen_on_close = None
            self.instances.append(self)

        async def connect(self):
            return [
                {
                    "node_id": "ray-node-a",
                    "address": "node-a",
                    "gpu_count": 2,
                    "resources": {"GPU": 2},
                    "hardware": {"gpu_name": "fake"},
                }
            ]

        def planned_engine_slots(self):
            return [Slot(), Slot()]

        async def __call__(self, task, payload):
            del task
            return CalibrationSample(
                elapsed_seconds_p75=0.01,
                bandwidth_bytes_per_second_p25=payload / 0.01,
                payload_bytes=payload,
            )

        async def close(self):
            self.state_seen_on_close = holder["rebalancer"].calibrator.state
            self.closed = True
            return self.resources_recovered

    monkeypatch.setattr(
        "dressage.proxy.rebalancing.scheduler.RayTransferBenchmark",
        FakeRayBenchmark,
    )
    rebalancer = EngineRebalancer(
        ControlPlaneClient(shared_l3=True),
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )
    holder["rebalancer"] = rebalancer

    async def scenario():
        await rebalancer.start()
        assert rebalancer._calibration_task is not None
        await rebalancer._calibration_task
        assert rebalancer.calibrator.state is CalibrationState.READY
        plan_snapshot = rebalancer.calibration_snapshot()["plan"]
        assert plan_snapshot["complete"] is True
        assert plan_snapshot["pending_links"] == []
        assert plan_snapshot["completed_links"]
        assert all(task["path_fingerprint"] for task in plan_snapshot["tasks"])
        assert FakeRayBenchmark.instances[0].closed is True
        assert (
            FakeRayBenchmark.instances[0].state_seen_on_close
            is CalibrationState.RUNNING
        )
        await rebalancer.close()

    run(scenario())

    FakeRayBenchmark.resources_recovered = False
    degraded = EngineRebalancer(
        ControlPlaneClient(shared_l3=True),
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )
    holder["rebalancer"] = degraded

    async def degraded_scenario():
        await degraded.start()
        assert degraded._calibration_task is not None
        await degraded._calibration_task
        assert degraded.calibrator.state is CalibrationState.DEGRADED
        assert "GPU resources" in degraded.calibrator.state_reason
        await degraded.close()

    run(degraded_scenario())


def test_reservations_spread_simultaneous_new_sessions():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        first, second = await asyncio.gather(
            rebalancer.acquire(session_id="a", input_ids=[1] * 100),
            rebalancer.acquire(session_id="b", input_ids=[1] * 100),
        )
        try:
            assert first.worker_url != second.worker_url
            assert first.decision.reason == "new_session_min_load"
            assert second.decision.reason == "new_session_min_load"
            assert first.decision.source_base_load is None
            assert first.decision.target_projected_load is not None
            assert first.decision.target_projected_load.total > 0
        finally:
            await rebalancer.fail(first)
            await rebalancer.fail(second)

    run(scenario())


@pytest.mark.parametrize(
    (
        "source_running",
        "target_running",
        "source_queued",
        "target_queued",
        "source_prefill",
        "target_prefill",
        "expected_worker",
        "expected_reason",
    ),
    [
        (2, 1, 0, 0, 0, 0, "source", "backlog_advantage_not_met"),
        (1, 2, 0, 0, 0, 0, "source", "backlog_advantage_not_met"),
        (2, 0, 1, 1, 0, 0, "source", "backlog_advantage_not_met"),
        (2, 0, 0, 1, 0, 0, "source", "backlog_advantage_not_met"),
        (2, 1, 1, 0, 0, 0, "target", "load_improvement_threshold_met"),
        (2, 1, 0, 0, 1_000, 0, "target", "load_improvement_threshold_met"),
    ],
)
def test_existing_session_requires_strict_backlog_advantage(
    source_running,
    target_running,
    source_queued,
    target_queued,
    source_prefill,
    target_prefill,
    expected_worker,
    expected_reason,
):
    client = ControlPlaneClient(shared_l3=True)

    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_hold_turns=1),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        rebalancer.loads[source].running = source_running
        rebalancer.loads[source].queued = source_queued
        rebalancer.loads[source].waiting_uncached_tokens = source_prefill
        rebalancer.loads[target].running = target_running
        rebalancer.loads[target].queued = target_queued
        rebalancer.loads[target].waiting_uncached_tokens = target_prefill
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["backlog"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source},
            owner_turns=2,
        )

        lease = await rebalancer.acquire(
            session_id="backlog",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == (
                source if expected_worker == "source" else target
            )
            assert lease.decision.reason == expected_reason
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_existing_session_selects_lowest_load_target_with_backlog_advantage():
    client = ControlPlaneClient(shared_l3=True)
    client.urls.append("http://node-c:30000")

    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_hold_turns=1),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, lowest_load_without_advantage, eligible_target = client.urls
        rebalancer.loads[source].running = 4
        rebalancer.loads[source].queued = 1
        rebalancer.loads[lowest_load_without_advantage].queued = 1
        rebalancer.loads[eligible_target].running = 2
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["backlog-target"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source},
            owner_turns=2,
        )

        lease = await rebalancer.acquire(
            session_id="backlog-target",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == eligible_target
            assert lease.decision.reason == "load_improvement_threshold_met"
            assert lease.decision.target_base_load is not None
            assert lease.decision.target_base_load.queue_pressure == 0.0
        finally:
            await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize(
    (
        "target_running",
        "minimum_ratio",
        "expected_worker",
        "expected_reason",
    ),
    [
        (71, 0.30, "source", "load_improvement_below_threshold"),
        (70, 0.30, "target", "load_improvement_threshold_met"),
        (69, 0.30, "target", "load_improvement_threshold_met"),
        (65, 0.40, "source", "load_improvement_below_threshold"),
    ],
)
def test_existing_session_uses_base_load_improvement_threshold(
    target_running, minimum_ratio, expected_worker, expected_reason
):
    client = ControlPlaneClient(shared_l3=True)

    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_hold_turns=1,
            min_load_improvement_ratio=minimum_ratio,
        ),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        rebalancer.loads[source].running = 100
        rebalancer.loads[source].waiting_uncached_tokens = 1
        rebalancer.loads[target].running = target_running
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["threshold"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source, target},
            owner_turns=2,
        )

        lease = await rebalancer.acquire(
            session_id="threshold",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == (
                source if expected_worker == "source" else target
            )
            assert lease.decision.reason == expected_reason
            assert lease.decision.source_base_load is not None
            assert lease.decision.target_base_load is not None
            assert lease.decision.load_improvement_ratio == pytest.approx(
                (
                    lease.decision.source_base_load.total
                    - lease.decision.target_base_load.total
                )
                / lease.decision.source_base_load.total
            )
            assert lease.decision.required_load_improvement_ratio == minimum_ratio
            assert lease.decision.source_context is None
            assert lease.decision.target_context is None
        finally:
            await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize(
    (
        "min_hold_turns",
        "target_running",
        "owner_turns",
        "expected_worker",
        "expected_reason",
        "required",
    ),
    [
        (1, 50, 1, "source", "return_hysteresis_below_threshold", 0.60),
        (1, 40, 1, "target", "load_improvement_threshold_met", 0.60),
        (2, 50, 2, "source", "return_hysteresis_below_threshold", 0.60),
        (2, 40, 2, "target", "load_improvement_threshold_met", 0.60),
        (2, 50, 3, "target", "load_improvement_threshold_met", 0.30),
    ],
)
def test_previous_owner_uses_double_threshold_only_for_first_step(
    min_hold_turns,
    target_running,
    owner_turns,
    expected_worker,
    expected_reason,
    required,
):
    client = ControlPlaneClient(shared_l3=True)

    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_hold_turns=min_hold_turns,
            min_load_improvement_ratio=0.30,
        ),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        rebalancer.loads[source].running = 100
        rebalancer.loads[source].waiting_uncached_tokens = 1
        rebalancer.loads[target].running = target_running
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["return"] = SessionRoutingState(
            owner_worker_url=source,
            previous_owner_worker_url=target,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source, target},
            owner_turns=owner_turns,
        )

        lease = await rebalancer.acquire(
            session_id="return",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == (
                source if expected_worker == "source" else target
            )
            assert lease.decision.reason == expected_reason
            assert lease.decision.required_load_improvement_ratio == required
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_min_hold_turns_blocks_otherwise_beneficial_migration():
    client = ControlPlaneClient(shared_l3=True)

    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_hold_turns=2),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        rebalancer.loads[source].running = 100
        rebalancer.loads[target].running = 50
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["held"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source},
            owner_turns=1,
        )

        lease = await rebalancer.acquire(
            session_id="held",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == source
            assert lease.decision.reason == "min_hold_turns_not_met"
            assert lease.decision.load_improvement_ratio == pytest.approx(0.5)
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_default_hold_blocks_next_hop_until_second_owner_turn():
    client = ControlPlaneClient(shared_l3=True)
    client.urls.append("http://node-c:30000")

    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        previous_owner, source, target = client.urls
        rebalancer.loads[previous_owner].running = 100
        rebalancer.loads[source].running = 4
        rebalancer.loads[source].queued = 1
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["multi-hop"] = SessionRoutingState(
            owner_worker_url=source,
            previous_owner_worker_url=previous_owner,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={previous_owner, source},
            owner_turns=1,
        )

        sticky = await rebalancer.acquire(
            session_id="multi-hop",
            input_ids=[1] * 100,
        )
        assert sticky.worker_url == source
        assert sticky.decision.reason == "min_hold_turns_not_met"
        await rebalancer.complete(
            sticky,
            response_meta={
                "cached_tokens": 100,
                "queue_time": 0.0,
                "e2e_latency": 1.0,
                "decode_throughput": 10.0,
            },
            output_tokens=1,
            committed_tokens=[1] * 100,
        )
        assert rebalancer.sessions["multi-hop"].owner_turns == 2

        movable = await rebalancer.acquire(
            session_id="multi-hop",
            input_ids=[1] * 100,
        )
        try:
            assert movable.worker_url == target
            assert movable.decision.reason == "load_improvement_threshold_met"
        finally:
            await rebalancer.fail(movable)

    run(scenario())


def test_seen_engine_without_shared_l3_is_not_a_migration_target():
    client = ControlPlaneClient(shared_l3=False)
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        rebalancer.loads[source].running = 100
        rebalancer.loads[source].waiting_uncached_tokens = 1
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["seen-no-l3"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source, target},
            owner_turns=2,
        )
        session = rebalancer.sessions["seen-no-l3"]
        assert (
            rebalancer._candidate_path_readiness(session, source, source).cache_source
            is CacheSource.LOCAL
        )
        assert (
            rebalancer._candidate_path_readiness(session, source, target).cache_source
            is CacheSource.NONE
        )

        lease = await rebalancer.acquire(
            session_id="seen-no-l3",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == source
            assert lease.decision.reason == "no_eligible_migration_target"
            assert lease.decision.target_base_load is None
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_existing_session_rejects_move_when_projected_target_is_busier():
    client = ControlPlaneClient(shared_l3=True)

    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_hold_turns=1),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        rebalancer.loads[source].running = 100
        rebalancer.loads[source].request_capacity = 100
        rebalancer.loads[source].token_capacity = 1_000
        rebalancer.loads[source].waiting_uncached_tokens = 1
        rebalancer.loads[target].running = 6
        rebalancer.loads[target].request_capacity = 10
        rebalancer.loads[target].token_capacity = 100
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["projected-safety"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 150,
            seen_engines={source},
            owner_turns=2,
        )

        lease = await rebalancer.acquire(
            session_id="projected-safety",
            input_ids=[1] * 150,
        )
        try:
            assert lease.worker_url == source
            assert lease.decision.reason == "projected_load_safety_check_failed"
            assert lease.decision.load_improvement_ratio == pytest.approx(
                (
                    lease.decision.source_base_load.total
                    - lease.decision.target_base_load.total
                )
                / lease.decision.source_base_load.total
            )
            assert (
                lease.decision.target_projected_load.total
                > lease.decision.source_projected_load.total
            )
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_new_session_uses_projected_load_instead_of_prediction_history():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        slower_queue, faster_queue = client.urls
        fingerprint = rebalancer.deployments[slower_queue].cache_fingerprint
        for url, queue_seconds in ((slower_queue, 3.0), (faster_queue, 0.0)):
            rebalancer.performance.observe(
                fingerprint=fingerprint,
                engine_url=url,
                running=1,
                context_tokens=100,
                queue_seconds=queue_seconds,
                context_seconds=2.0,
                cached_tokens=0,
                output_tokens=10,
                decode_throughput=10.0,
                cache_source=CacheSource.NONE,
            )
        rebalancer.loads[slower_queue].running = 10

        lease = await rebalancer.acquire(
            session_id="new-full-prefill",
            input_ids=[1] * 100,
            step_max_new_tokens=10,
        )
        try:
            assert lease.worker_url == faster_queue
            assert lease.base_tokens == 0
            assert lease.decision.reason == "new_session_min_load"
            assert lease.decision.source_worker_url is None
            assert lease.decision.moved is False
            assert lease.decision.target_context is None
            assert lease.decision.target_projected_load is not None
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_new_session_does_not_use_engine_specific_prefill_history():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        slow_prefill, fast_prefill = client.urls
        fingerprint = rebalancer.deployments[slow_prefill].cache_fingerprint
        for url, queue_seconds, context_seconds in (
            (slow_prefill, 0.0, 4.0),
            (fast_prefill, 0.4, 1.0),
        ):
            rebalancer.performance.observe(
                fingerprint=fingerprint,
                engine_url=url,
                running=1,
                context_tokens=100,
                queue_seconds=queue_seconds,
                context_seconds=context_seconds,
                cached_tokens=0,
                output_tokens=10,
                decode_throughput=10.0,
                cache_source=CacheSource.NONE,
            )
        # Historical prefill favors this Engine, but load-only placement must
        # select the other Engine because it has no live queue.
        rebalancer.loads[fast_prefill].queued = 1

        lease = await rebalancer.acquire(
            session_id="new-prefill-throughput",
            input_ids=[1] * 100,
            step_max_new_tokens=10,
        )
        try:
            assert lease.worker_url == slow_prefill
            assert lease.decision.reason == "new_session_min_load"
            assert lease.decision.target_context is None
            assert lease.decision.target_projected_load.queue_pressure == 0.0
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_step_budget_prefers_request_and_rollout_caps_before_context():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=32),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        await rebalancer.register_session_context(
            session_id="rollout-cap",
            group_id=None,
            group_size=1,
            task_key="task",
            default_step_max_tokens=8192,
        )
        rollout_limited = await rebalancer.acquire(
            session_id="rollout-cap",
            input_ids=[1] * 100,
            context_remaining_tokens=56 * 1024,
        )
        try:
            assert rollout_limited.decision.effective_step_max_tokens == 8192
            assert rollout_limited.decision.estimated_step_output_tokens == 8192
            assert rollout_limited.expected_output_tokens == 8192
            assert rollout_limited.reserved_tokens == 8292
            assert "rollout" in rollout_limited.decision.step_max_tokens_source
        finally:
            await rebalancer.fail(rollout_limited)

        await rebalancer.register_session_context(
            session_id="request-cap",
            group_id=None,
            group_size=1,
            task_key="task",
            default_step_max_tokens=8192,
        )
        request_limited = await rebalancer.acquire(
            session_id="request-cap",
            input_ids=[1] * 100,
            step_max_new_tokens=2048,
            context_remaining_tokens=56 * 1024,
        )
        try:
            assert request_limited.decision.effective_step_max_tokens == 2048
            assert request_limited.expected_output_tokens == 2048
        finally:
            await rebalancer.fail(request_limited)

        context_only = await rebalancer.acquire(
            session_id="context-only",
            input_ids=[1] * 100,
            context_remaining_tokens=4096,
        )
        try:
            assert context_only.decision.effective_step_max_tokens == 4096
            assert context_only.decision.step_max_tokens_source == "min(context)"
        finally:
            await rebalancer.fail(context_only)

        rebalancer.group_lengths.observe(
            group_id="g", task_key="task", final_length=5000
        )
        rebalancer.group_lengths.observe(
            group_id="g", task_key="task", final_length=5000
        )
        await rebalancer.register_session_context(
            session_id="group-cap",
            group_id="g",
            group_size=2,
            task_key="task",
            default_step_max_tokens=8192,
        )
        rebalancer.sessions["group-cap"].generated_tokens = 4000
        group_limited = await rebalancer.acquire(
            session_id="group-cap",
            input_ids=[1] * 100,
            context_remaining_tokens=56 * 1024,
        )
        try:
            assert group_limited.decision.group_remaining_tokens == 1000
            assert group_limited.decision.estimated_step_output_tokens == 1000
        finally:
            await rebalancer.fail(group_limited)

    run(scenario())


def test_bootstrap_sticky_turn_keeps_committed_prefix_for_hit_learning():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        owner = client.urls[0]
        fingerprint = rebalancer.deployments[owner].cache_fingerprint
        rebalancer.sessions["session"] = SessionRoutingState(
            owner_worker_url=owner,
            fingerprint=fingerprint,
            previous_committed_tokens=[1, 2, 3, 4],
            seen_engines={owner},
        )
        lease = await rebalancer.acquire(
            session_id="session",
            input_ids=[1, 2, 3, 9, 10],
        )
        try:
            assert lease.decision.state is SchedulerState.BOOTSTRAP
            assert lease.base_tokens == 3
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_load_snapshot_accepts_public_and_internal_queue_field_names():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )
    rebalancer._update_load(
        "worker",
        {
            "loads": [
                {"num_waiting_reqs": 2},
                {"num_queue_reqs": 3},
            ]
        },
        now=1.0,
    )
    assert rebalancer.loads["worker"].queued == 5
    assert rebalancer.loads["worker"].waiting_uncached_tokens == 0
    assert rebalancer.loads["worker"].gen_throughput == 0.0
    assert rebalancer.loads["worker"].live_queue_metrics_available is False


def test_load_snapshot_aggregates_live_queue_fields_across_dp_ranks():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )
    rebalancer._update_load(
        "worker",
        {
            "loads": [
                {
                    "num_waiting_reqs": 2,
                    "num_waiting_uncached_tokens": 3_000,
                    "gen_throughput": 120.5,
                    "queues": {
                        "waiting": 2,
                        "paused": 1,
                        "retracted": 3,
                        "grammar": 4,
                    },
                },
                {
                    "num_waiting_reqs": 5,
                    "num_waiting_uncached_tokens": 5_000,
                    "gen_throughput": 79.5,
                    "queues": {
                        "waiting": 5,
                        "paused": 2,
                        "retracted": 4,
                        "grammar": 1,
                    },
                },
            ]
        },
        now=1.0,
    )

    load = rebalancer.loads["worker"]
    assert load.queued == 7
    assert load.waiting_uncached_tokens == 8_000
    assert load.gen_throughput == 200.0
    assert load.queue_waiting == 7
    assert load.queue_paused == 3
    assert load.queue_retracted == 7
    assert load.queue_grammar == 5
    assert load.live_queue_metrics_available is True


def test_live_queue_seconds_uses_prefill_p25_and_falls_back_when_unavailable():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
    )
    fingerprint = "fp"
    engine = "worker"
    rebalancer.performance.observe(
        fingerprint=fingerprint,
        engine_url=engine,
        running=1,
        context_tokens=8_000,
        queue_seconds=0.1,
        context_seconds=2.0,
        cached_tokens=0,
        output_tokens=1,
        decode_throughput=10.0,
        cache_source=CacheSource.NONE,
    )
    rebalancer.loads[engine] = EngineLoad(
        worker_url=engine,
        metrics_timestamp=10.0,
        waiting_uncached_tokens=8_000,
        live_queue_metrics_available=True,
    )

    assert (
        rebalancer._live_queue_seconds(
            fingerprint=fingerprint,
            engine_url=engine,
            context_tokens=8_000,
            now=10.0,
        )
        == 2.0
    )
    rebalancer.loads[engine].reserved_prefill_tokens = 2_000
    assert (
        rebalancer._live_queue_seconds(
            fingerprint=fingerprint,
            engine_url=engine,
            context_tokens=8_000,
            now=10.0,
        )
        == 2.5
    )

    rebalancer.loads[engine].metrics_timestamp = 1.0
    assert (
        rebalancer._live_queue_seconds(
            fingerprint=fingerprint,
            engine_url=engine,
            context_tokens=8_000,
            now=10.0,
        )
        is None
    )
    rebalancer.loads[engine].metrics_timestamp = 10.0
    rebalancer.loads[engine].live_queue_metrics_available = False
    assert (
        rebalancer._live_queue_seconds(
            fingerprint=fingerprint,
            engine_url=engine,
            context_tokens=8_000,
            now=10.0,
        )
        is None
    )

    empty_history = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
    )
    empty_history.loads[engine] = EngineLoad(
        worker_url=engine,
        metrics_timestamp=10.0,
        waiting_uncached_tokens=8_000,
        live_queue_metrics_available=True,
    )
    assert (
        empty_history._live_queue_seconds(
            fingerprint=fingerprint,
            engine_url=engine,
            context_tokens=8_000,
            now=10.0,
        )
        is None
    )


def test_prefill_reservations_expire_by_load_generation_and_release_on_failure():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )

    async def scenario():
        await rebalancer.refresh()
        target = client.urls[0]
        live_payload = {
            "loads": [
                {
                    "num_waiting_uncached_tokens": 0,
                    "num_waiting_reqs": 0,
                }
            ]
        }
        rebalancer._update_load(target, live_payload, now=time.monotonic())
        fingerprint = rebalancer.deployments[target].cache_fingerprint
        estimate = ContextRecoveryEstimate(
            cache_source=CacheSource.MOONCAKE,
            expected_cached_tokens=20,
            expected_prefill_tokens=80,
            estimated_seconds=1.0,
            hit_probability=0.2,
        )
        decision = RoutingDecision(
            session_id="generation-reservation",
            source_worker_url=None,
            target_worker_url=target,
            cache_fingerprint=fingerprint,
            state=SchedulerState.ACTIVE,
            reason="test",
            target_context=estimate,
        )
        budget = StepGenerationBudget("unavailable", None, None, None, None)
        lease = rebalancer._reserve(
            decision,
            input_ids=[1] * 100,
            base_tokens=0,
            budget=budget,
        )
        assert lease.reserved_prefill_tokens == 80
        assert rebalancer.loads[target].reserved_prefill_tokens == 80

        generation = rebalancer._load_generations[target]
        rebalancer._update_load(
            target,
            {"loads": [{"num_waiting_reqs": 0}]},
            now=time.monotonic(),
        )
        assert rebalancer._load_generations[target] == generation
        assert rebalancer.loads[target].reserved_prefill_tokens == 80

        rebalancer._update_load(target, live_payload, now=time.monotonic())
        assert rebalancer.loads[target].reserved_prefill_tokens == 80
        rebalancer._update_load(target, live_payload, now=time.monotonic())
        assert rebalancer.loads[target].reserved_prefill_tokens == 0
        await rebalancer.fail(lease)
        assert rebalancer.loads[target].reserved_prefill_tokens == 0

        second = rebalancer._reserve(
            decision,
            input_ids=[1] * 100,
            base_tokens=0,
            budget=budget,
        )
        assert rebalancer.loads[target].reserved_prefill_tokens == 80
        await rebalancer.fail(second)
        assert rebalancer.loads[target].reserved_prefill_tokens == 0

        sticky_decision = RoutingDecision(
            session_id="sticky-reservation",
            source_worker_url=target,
            target_worker_url=target,
            cache_fingerprint=fingerprint,
            state=SchedulerState.BOOTSTRAP,
            reason="test",
        )
        sticky = rebalancer._reserve(
            sticky_decision,
            input_ids=[1] * 100,
            base_tokens=80,
            budget=budget,
        )
        assert sticky.reserved_prefill_tokens == 20
        await rebalancer.fail(sticky)

        full_prefill_decision = RoutingDecision(
            session_id="new-session-reservation",
            source_worker_url=None,
            target_worker_url=target,
            cache_fingerprint=fingerprint,
            state=SchedulerState.BOOTSTRAP,
            reason="test",
        )
        full_prefill = rebalancer._reserve(
            full_prefill_decision,
            input_ids=[1] * 100,
            base_tokens=0,
            budget=budget,
        )
        assert full_prefill.reserved_prefill_tokens == 100
        await rebalancer.fail(full_prefill)
        assert rebalancer.loads[target].reserved_prefill_tokens == 0

    run(scenario())


def test_active_scheduler_routes_from_load_without_prediction_history():
    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    client = ControlPlaneClient(shared_l3=True)
    config = EngineRebalancingConfig(
        enabled=True,
        min_samples=1,
        min_hold_turns=0,
        min_risk_ms=100,
    )
    rebalancer = EngineRebalancer(
        client,
        config=config,
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        for url, queue in ((source, 5.0), (target, 0.0)):
            rebalancer.performance.observe(
                fingerprint=fingerprint,
                engine_url=url,
                running=1,
                context_tokens=100,
                queue_seconds=queue,
                context_seconds=1.0,
                cached_tokens=0,
                output_tokens=1,
                decode_throughput=10,
            )
        rebalancer.loads[source].running = 100
        rebalancer.loads[source].waiting_uncached_tokens = 1
        rebalancer.loads[target].running = 50
        rebalancer.sessions["session"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
            owner_turns=2,
        )
        lease = await rebalancer.acquire(
            session_id="session",
            input_ids=[1] * 100,
        )
        try:
            assert lease.decision.state is SchedulerState.ACTIVE
            assert lease.decision.moved is True
            assert lease.worker_url == target
            assert lease.decision.reason == "load_improvement_threshold_met"
            assert lease.decision.source_context is None
            assert lease.decision.target_context is None
            assert lease.decision.queue_risk_seconds == 0.0
            assert lease.decision.context_risk_seconds == 0.0
            assert lease.decision.decision_risk_seconds == 0.0
            assert lease.decision.source_base_load is not None
            assert lease.decision.target_base_load is not None
            assert (
                lease.decision.source_base_load.total
                > lease.decision.target_base_load.total
            )
            snapshot = lease.decision.snapshot()
            assert snapshot["source_base_load"]["total"] == pytest.approx(1.00001)
            assert snapshot["target_base_load"]["total"] == pytest.approx(0.5)
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_prefill_pressure_can_prevent_load_ratio_migration():
    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    client = ControlPlaneClient(shared_l3=True)
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_samples=1,
            min_hold_turns=0,
            min_risk_ms=100,
        ),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        for url, queue in ((source, 2.0), (target, 0.1)):
            rebalancer.performance.observe(
                fingerprint=fingerprint,
                engine_url=url,
                running=1,
                context_tokens=100,
                queue_seconds=queue,
                context_seconds=1.0,
                cached_tokens=0,
                output_tokens=1,
                decode_throughput=10.0,
                cache_source=CacheSource.NONE,
            )
        rebalancer.loads[source].running = 50
        rebalancer.loads[target].running = 20
        rebalancer.loads[source].waiting_uncached_tokens = 0
        rebalancer.loads[target].waiting_uncached_tokens = 40_000
        rebalancer.sessions["live-backlog"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
            owner_turns=2,
        )

        lease = await rebalancer.acquire(
            session_id="live-backlog",
            input_ids=[1] * 100,
        )
        try:
            assert lease.decision.state is SchedulerState.ACTIVE
            assert lease.decision.moved is False
            assert lease.worker_url == source
            assert lease.decision.reason == "backlog_advantage_not_met"
            assert lease.decision.target_base_load is not None
            assert lease.decision.target_base_load.prefill_pressure == pytest.approx(
                0.4
            )
            assert lease.reserved_prefill_tokens == 20
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_mooncake_prior_does_not_affect_load_routing():
    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    async def prepare(cold_start_probability: float) -> EngineRebalancer:
        client = ControlPlaneClient(shared_l3=True)
        rebalancer = EngineRebalancer(
            client,
            config=EngineRebalancingConfig(
                enabled=True,
                cold_start_hit_probability=cold_start_probability,
                min_hold_turns=1,
            ),
            model_id="model",
            model_config=simple_model_config(),
            calibration_benchmark=benchmark,
        )
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        for _ in range(rebalancer.config.min_samples):
            for engine_url, queue_seconds in ((source, 0.5), (target, 0.0)):
                rebalancer.performance.observe(
                    fingerprint=fingerprint,
                    engine_url=engine_url,
                    running=1,
                    context_tokens=100,
                    queue_seconds=queue_seconds,
                    context_seconds=1.0,
                    cached_tokens=0,
                    output_tokens=1,
                    decode_throughput=10.0,
                    cache_source=CacheSource.NONE,
                )
        rebalancer.loads[source].queued = 1
        rebalancer.pools[fingerprint].update(
            rebalancer._pool_readiness(fingerprint, now=time.monotonic())
        )
        return rebalancer

    async def decide(
        rebalancer: EngineRebalancer,
        *,
        session_id: str,
        owner_turns: int,
    ) -> RoutingDecision:
        source = rebalancer.client.urls[0]
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions[session_id] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
            owner_turns=owner_turns,
        )
        lease = await rebalancer.acquire(
            session_id=session_id,
            input_ids=[1] * 100,
        )
        try:
            return lease.decision
        finally:
            await rebalancer.fail(lease)

    async def scenario():
        conservative = await prepare(0.1)
        default = await prepare(EngineRebalancingConfig().cold_start_hit_probability)

        conservative_decision = await decide(
            conservative,
            session_id="conservative",
            owner_turns=2,
        )
        held_decision = await decide(
            default,
            session_id="held",
            owner_turns=1,
        )
        default_decision = await decide(
            default,
            session_id="default",
            owner_turns=2,
        )

        for decision in (
            conservative_decision,
            held_decision,
            default_decision,
        ):
            assert decision.moved is True
            assert decision.reason == "load_improvement_threshold_met"
            assert decision.source_context is None
            assert decision.target_context is None
        assert conservative_decision.load_improvement_ratio == pytest.approx(
            default_decision.load_improvement_ratio
        )
        assert default.config.min_hold_turns == 1
        assert default.config.min_risk_ms == 10

    run(scenario())


def test_prediction_history_cannot_bypass_missing_l3_path():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_samples=1,
            min_hold_turns=0,
            min_risk_ms=100,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        for url, queue_seconds, context_seconds in (
            (source, 0.0, 4.0),
            (target, 0.5, 1.0),
        ):
            rebalancer.performance.observe(
                fingerprint=fingerprint,
                engine_url=url,
                running=1,
                context_tokens=100,
                queue_seconds=queue_seconds,
                context_seconds=context_seconds,
                cached_tokens=0,
                output_tokens=10,
                decode_throughput=10.0,
                cache_source=CacheSource.NONE,
            )
        rebalancer.loads[source].queued = 1
        rebalancer.pools[fingerprint].update(
            rebalancer._pool_readiness(fingerprint, now=time.monotonic())
        )
        rebalancer.sessions["total-step"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[],
            seen_engines={source},
            owner_turns=2,
        )

        lease = await rebalancer.acquire(
            session_id="total-step",
            input_ids=[1] * 100,
            step_max_new_tokens=10,
        )
        try:
            assert lease.worker_url == source
            assert lease.decision.moved is False
            assert lease.decision.reason == "no_eligible_migration_target"
            assert lease.decision.stay_seconds is None
            assert lease.decision.move_seconds is None
            assert lease.decision.source_context is None
            assert lease.decision.target_context is None
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_owner_failure_uses_projected_load_without_threshold():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.performance.observe(
            fingerprint=fingerprint,
            engine_url=target,
            running=1,
            context_tokens=100,
            queue_seconds=0.5,
            context_seconds=1.0,
            cached_tokens=0,
            output_tokens=10,
            decode_throughput=10.0,
            cache_source=CacheSource.NONE,
        )
        rebalancer.loads[source].healthy = False
        rebalancer.sessions["failed-owner"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
            owner_turns=2,
        )

        lease = await rebalancer.acquire(
            session_id="failed-owner",
            input_ids=[1] * 100,
            step_max_new_tokens=10,
        )
        try:
            assert lease.worker_url == target
            assert lease.decision.reason == "owner_unhealthy_failover"
            assert lease.decision.moved is True
            assert lease.decision.decision_risk_seconds == 0.0
            assert lease.decision.target_context is None
            assert lease.decision.target_projected_load is not None
            assert lease.decision.required_load_improvement_ratio is None
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_load_routing_keeps_single_step_budget_for_reservation():
    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    client = ControlPlaneClient(shared_l3=True)
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_samples=1,
            min_hold_turns=0,
            min_risk_ms=100,
        ),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        for url, decode_throughput in ((source, 10.0), (target, 20.0)):
            rebalancer.performance.observe(
                fingerprint=fingerprint,
                engine_url=url,
                running=1,
                context_tokens=100,
                queue_seconds=1.0,
                context_seconds=1.0,
                cached_tokens=0,
                output_tokens=10,
                decode_throughput=decode_throughput,
                cache_source=CacheSource.NONE,
            )
        rebalancer.loads[source].queued = 1
        rebalancer.loads[source].running = 100
        rebalancer.loads[target].running = 50
        rebalancer.sessions["heterogeneous"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
            owner_turns=2,
        )
        lease = await rebalancer.acquire(
            session_id="heterogeneous",
            input_ids=[1] * 100,
            step_max_new_tokens=8192,
            context_remaining_tokens=56 * 1024,
        )
        try:
            assert lease.decision.moved is True
            assert lease.decision.estimated_step_output_tokens == 8192
            assert lease.decision.source_decode_seconds is None
            assert lease.decision.target_decode_seconds is None
            assert lease.decision.source_projected_load is not None
            assert lease.decision.target_projected_load is not None
            assert lease.reserved_tokens == 100 + 8192
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_prediction_risks_do_not_block_load_ratio_migration():
    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    client = ControlPlaneClient(shared_l3=True)
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_samples=1,
            min_hold_turns=0,
            min_risk_ms=100,
        ),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        # Queue errors are 1s on the source and 2s on the target. Context
        # errors are also 1s and 2s, respectively.
        rebalancer.performance.observe(
            fingerprint=fingerprint,
            engine_url=source,
            running=1,
            context_tokens=100,
            queue_seconds=5.0,
            predicted_queue_seconds=4.0,
            context_seconds=1.0,
            cached_tokens=80,
            output_tokens=1,
            decode_throughput=10,
            estimated_context_seconds=2.0,
            cache_source=CacheSource.LOCAL,
        )
        rebalancer.performance.observe(
            fingerprint=fingerprint,
            engine_url=target,
            running=1,
            context_tokens=100,
            queue_seconds=0.0,
            predicted_queue_seconds=2.0,
            context_seconds=1.0,
            cached_tokens=0,
            output_tokens=1,
            decode_throughput=10,
            estimated_context_seconds=3.0,
            cache_source=CacheSource.NONE,
        )
        rebalancer.loads[source].queued = 1
        rebalancer.loads[source].running = 100
        rebalancer.loads[target].running = 50
        rebalancer.sessions["risk-blocked"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
            owner_turns=2,
        )

        lease = await rebalancer.acquire(
            session_id="risk-blocked",
            input_ids=[1] * 100,
        )
        try:
            assert lease.decision.state is SchedulerState.ACTIVE
            assert lease.decision.moved is True
            assert lease.worker_url == target
            assert lease.decision.reason == "load_improvement_threshold_met"
            assert lease.decision.queue_risk_seconds == 0.0
            assert lease.decision.context_risk_seconds == 0.0
            assert lease.decision.decision_risk_seconds == 0.0
            assert lease.decision.source_context is None
            assert lease.decision.target_context is None
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_completion_pairs_actual_queue_with_the_selected_path_prediction():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint

        for session_id, selected, actual, expected, moved in (
            ("stay", source, 2.0, 3.0, False),
            ("move", target, 4.0, 1.0, True),
        ):
            rebalancer.sessions[session_id] = SessionRoutingState(
                owner_worker_url=source,
                fingerprint=fingerprint,
                seen_engines={source},
            )
            decision = RoutingDecision(
                session_id=session_id,
                source_worker_url=source,
                target_worker_url=target if moved else source,
                cache_fingerprint=fingerprint,
                state=SchedulerState.ACTIVE,
                reason="test",
                source_queue_seconds=3.0,
                target_queue_seconds=1.0,
                moved=moved,
            )
            lease = RoutingLease(
                decision=decision,
                worker_url=selected,
                reserved_tokens=100,
                base_tokens=0,
                started_monotonic=time.monotonic(),
            )
            await rebalancer.complete(
                lease,
                response_meta={
                    "queue_time": actual,
                    "e2e_latency": actual + 1.0,
                    "cached_tokens": 0,
                    "decode_throughput": 10.0,
                },
                output_tokens=1,
                committed_tokens=[1] * 100,
            )
            observation = rebalancer._observations[-1]
            assert observation["predicted_queue_seconds"] == expected
            assert observation["actual_queue_seconds"] == actual
            assert observation["queue_prediction_error_seconds"] == abs(
                expected - actual
            )

    run(scenario())


def test_actual_mooncake_hit_overrides_full_prefill_prediction_classification():
    client = ControlPlaneClient(shared_l3=True)
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["s"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            seen_engines={source},
        )
        predicted_full_prefill = ContextRecoveryEstimate(
            cache_source=CacheSource.NONE,
            expected_cached_tokens=0,
            expected_prefill_tokens=100,
            estimated_seconds=1.0,
            hit_probability=0.0,
        )
        lease = RoutingLease(
            decision=RoutingDecision(
                session_id="s",
                source_worker_url=source,
                target_worker_url=target,
                cache_fingerprint=fingerprint,
                state=SchedulerState.ACTIVE,
                reason="test",
                target_context=predicted_full_prefill,
                moved=True,
            ),
            worker_url=target,
            reserved_tokens=100,
            base_tokens=80,
            started_monotonic=time.monotonic(),
            context_tokens=100,
        )
        await rebalancer.complete(
            lease,
            response_meta={
                "queue_time": 0.0,
                "e2e_latency": 1.0,
                "cached_tokens": 80,
                "decode_throughput": 10.0,
            },
            output_tokens=1,
            committed_tokens=[1] * 101,
        )
        assert rebalancer._observations[-1]["cache_source"] == "mooncake"
        assert rebalancer.performance.snapshot()["prefill_samples"] == 0

    run(scenario())


def test_sglang_client_can_target_worker_directly():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "text": "x",
                "output_ids": [120],
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 120, "x"]],
                    "finish_reason": {"type": "stop"},
                },
            },
        )

    async def scenario():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = SGLangRouterClient("http://router", client=http_client)
            response = await client.generate(
                [1, 2],
                {"max_new_tokens": 1},
                worker_url="http://worker-a:30000",
            )
            assert response.output_ids == [120]

    run(scenario())
    assert seen == ["http://worker-a:30000/generate"]


def test_sglang_client_weight_version_uses_model_info_with_legacy_fallback():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/model_info":
            return httpx.Response(404)
        return httpx.Response(200, json={"weight_version": "9"})

    async def scenario():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = SGLangRouterClient("http://router", client=http_client)
            assert (
                await client.get_worker_weight_version("http://worker-a:30000") == "9"
            )

    run(scenario())
    assert seen == ["/model_info", "/get_weight_version"]


def test_cli_exposes_single_rebalancing_switch(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["dressage-proxy", "--tokenizer-path", "model", "--enable-engine-rebalancing"],
    )
    args = parse_args()
    assert args.enable_engine_rebalancing is True
    assert args.engine_rebalancing_min_load_improvement_ratio == 0.20


def test_cli_accepts_load_improvement_ratio(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dressage-proxy",
            "--tokenizer-path",
            "model",
            "--enable-engine-rebalancing",
            "--engine-rebalancing-min-load-improvement-ratio",
            "0.45",
        ],
    )

    assert parse_args().engine_rebalancing_min_load_improvement_ratio == 0.45


@pytest.mark.parametrize("value", ["-0.01", "1.01"])
def test_cli_rejects_out_of_range_load_improvement_ratio(
    monkeypatch, capsys, value
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dressage-proxy",
            "--tokenizer-path",
            "model",
            "--engine-rebalancing-min-load-improvement-ratio",
            value,
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()
    assert "must be between 0 and 1" in capsys.readouterr().err


def test_enabled_proxy_places_first_request_directly_and_reports_state():
    client = DirectGenerationClient()
    app = create_app(
        tokenizer=FakeTokenizer(),
        tokenizer_path="model",
        token_build_mode="snapshot",
        sglang_client=client,
        enable_engine_rebalancing=True,
        engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
        engine_rebalancing_model_config=simple_model_config(),
        tool_call_parse_backend="local",
        reasoning_parse_backend="local",
    )
    with TestClient(app) as http_client:
        context_response = http_client.post(
            "/v1/session/context",
            json={
                "session_id": "s1",
                "group_id": 3,
                "group_size": 4,
                "task_key": "math",
                "default_step_max_tokens": 8192,
            },
        )
        assert context_response.status_code == 200
        response = http_client.post(
            "/v1/chat/completions",
            headers={"X-Session-ID": "s1"},
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200
        assert client.calls[0]["worker_url"] in client.urls
        assert client.calls[0]["input_ids"]

        loads = http_client.get("/v1/engines/load").json()
        assert loads["enabled"] is True
        assert loads["effective_config"]["metrics_stale_ms"] == 2_000
        assert loads["effective_config"]["load_poll_interval_ms"] == 125
        assert loads["effective_config"]["history_size"] == 128
        assert loads["effective_config"]["min_samples"] == 16
        assert loads["effective_config"]["min_hold_turns"] == 2
        assert loads["effective_config"]["min_risk_ms"] == 10
        assert loads["effective_config"]["cold_start_hit_probability"] == 1.0
        assert loads["effective_config"]["min_load_improvement_ratio"] == 0.20
        assert loads["compatibility_pools"][0]["state"] in {
            "BOOTSTRAP",
            "ACTIVE",
        }
        engine_load = loads["engines"][0]
        assert "waiting_uncached_tokens" in engine_load
        assert "gen_throughput" in engine_load
        assert "queue_waiting" in engine_load
        assert "queue_paused" in engine_load
        assert "queue_retracted" in engine_load
        assert "queue_grammar" in engine_load
        assert "reserved_prefill_tokens" in engine_load
        assert "live_queue_metrics_available" in engine_load
        observation = loads["recent_context_observations"][0]
        assert observation["cache_source"] == "none"
        assert observation["actual_cached_tokens"] == 0
        assert observation["actual_prefill_tokens"] > 0
        assert "predicted_queue_seconds" in observation
        assert "actual_queue_seconds" in observation
        assert "queue_prediction_error_seconds" in observation
        assert "queue_risk_seconds" in observation
        assert "context_risk_seconds" in observation
        assert "decision_risk" in observation
        assert "queue_error_samples" in loads["performance_models"]
        assert loads["recent_decisions"][0]["effective_step_max_tokens"] == 8192
        assert loads["recent_decisions"][0]["estimated_step_output_tokens"] == 8192
        assert loads["recent_decisions"][0]["target_projected_load"] is not None
        assert "target_queue_history_seconds" in loads["recent_decisions"][0]
        assert "target_queue_live_seconds" in loads["recent_decisions"][0]

        calibration = http_client.get("/v1/engines/calibration").json()
        assert calibration["state"] == "DEGRADED"
        assert "full-prefill fallback" in calibration["state_reason"]
        assert "online_request_count" not in calibration
        assert "runtime_calibration" not in calibration
        assert "snapshot_persistence" not in calibration
        assert "effective_model_sources" not in calibration
        assert "router_discovery" not in loads


def test_disabled_proxy_reports_off_without_discovery():
    client = DirectGenerationClient()
    app = create_app(
        tokenizer=FakeTokenizer(),
        token_build_mode="snapshot",
        sglang_client=client,
        tool_call_parse_backend="local",
        reasoning_parse_backend="local",
    )
    with TestClient(app) as http_client:
        payload = http_client.get("/v1/engines/load").json()
        assert payload["enabled"] is False
        assert payload["state"] == "OFF"


def test_single_node_l3_hicache_script_owns_mooncake_lifecycle():
    path = Path("examples/scripts/run_blackbox_qwen3.5_4b_sync_local_l3_hicache.sh")
    source = path.read_text()

    assert '"qwen3.5-4B-sync-local-l3-hicache"' in source
    assert "--debug-rollout-only" in source
    assert "--enable-engine-rebalancing" in source
    assert "MOONCAKE_MASTER_PORT=50051" in source
    assert "MOONCAKE_METADATA_PORT=8080" in source
    assert (
        'MOONCAKE_MASTER_ADDRESS="${MOONCAKE_MASTER_HOST}:${MOONCAKE_MASTER_PORT}"'
        in source
    )
    assert (
        'MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-4gb}"' in source
    )
    assert '"protocol": "tcp"' in source
    assert '"metadata_server": metadata_server' in source
    assert "mooncake_master \\\n" in source
    assert "--enable_http_metadata_server=true" in source
    assert "mooncake_store_service" not in source
    cleanup = source[source.index("cleanup() {") : source.index("trap cleanup EXIT")]
    assert cleanup.index("_stop_proxy_on_exit") < cleanup.index(
        "_stop_ray_cluster_on_exit"
    )
    assert cleanup.index("_stop_proxy_on_exit") < cleanup.index(
        "_stop_mooncake_master_on_exit"
    )
    assert '[[ "${wait_count}" -lt 100 ]]' in source
    for argument in (
        "--sglang-enable-hierarchical-cache",
        "--sglang-hicache-ratio 2.0",
        "--sglang-hicache-write-policy write_through",
        "--sglang-hicache-mem-layout page_first",
        "--sglang-hicache-storage-backend mooncake",
        "--sglang-hicache-storage-backend-extra-config",
    ):
        assert argument in source
    assert cleanup.index("_stop_ray_cluster_on_exit") < cleanup.index(
        "_stop_mooncake_master_on_exit"
    )
    assert 'rm -f "${MOONCAKE_MASTER_PID_FILE}"' in source


def test_engine_rebalancing_benchmark_defaults_to_one_off_on_pair(tmp_path):
    path = Path(
        "examples/scripts/benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    source = path.read_text()
    result = subprocess.run(
        ["bash", str(path)],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "BENCHMARK_DRY_RUN": "1",
            "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
            "BENCHMARK_SEED": "20260806",
            "PROMPT_DATA": str(
                Path("examples/data/dressage_dapo_prompts_dynamic_multi.jsonl")
            ),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "seed20260806-off-r1" in result.stdout
    assert "seed20260806-on-r1" in result.stdout
    assert "rollout batch: 256" in result.stdout
    assert "global batch:  256" in result.stdout
    assert "response max:  12288" in result.stdout
    assert "sandbox slots: 16" in result.stdout
    assert "slot timeout:  3600" in result.stdout
    assert "dressage_dapo_prompts_long_tail.jsonl" in result.stdout
    assert "warm-up" not in result.stdout
    assert "off-r2" not in result.stdout
    assert "on-r2" not in result.stdout
    assert "Valid measured pairs: `{len(valid_rows)}/1`" in source
    assert "Median rollout speedup" not in source
    assert "Warm-up" not in source
    assert not (tmp_path / "benchmark").exists()


def test_engine_rebalancing_benchmark_samples_long_tail_dataset_once():
    source = Path(
        "examples/scripts/benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
    ).read_text(encoding="utf-8")
    prepare_call = (
        'prepare_long_tail_prompts "${PROMPT_SOURCE}" "${PROMPT_EFFECTIVE}" '
        '"${BENCHMARK_SEED}"'
    )
    run_one = source[source.index("run_one() {") : source.index("write_summary() {")]

    assert source.count(prepare_call) == 1
    assert 'python3 "${LONG_TAIL_TOOL}" sample' in source
    assert "prepare_long_tail_prompts" not in run_one
    assert "ROLLOUT_BATCH_SIZE=256" in source
    assert "GLOBAL_BATCH_SIZE=256" in source
    assert "ROLLOUT_MAX_RESPONSE_LEN=12288" in source
    assert "DRESSAGE_BLACKBOX_SLOTS_PER_NODE=16" in source
    assert "DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC=3600" in source


def test_disabled_rebalancer_does_not_create_session_context_state():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=False),
        model_id="model",
    )

    async def scenario():
        for index in range(256):
            await rebalancer.register_session_context(
                session_id=f"disabled-{index}",
                group_id=index,
                group_size=4,
                task_key="task",
            )
        assert (await rebalancer.snapshot())["active_sessions"] == 0

    run(scenario())


def test_discard_session_context_is_idempotent_without_group_observation():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )

    async def scenario():
        await rebalancer.register_session_context(
            session_id="discarded",
            group_id="group",
            group_size=2,
            task_key="task",
        )
        rebalancer.sessions["discarded"].generated_tokens = 17

        await rebalancer.discard_session_context("discarded")
        await rebalancer.discard_session_context("discarded")

        assert "discarded" not in rebalancer.sessions
        assert not rebalancer.group_lengths._group
        assert not rebalancer.group_lengths._task

    run(scenario())


@pytest.mark.parametrize("settle_method", ["complete", "fail"])
def test_late_lease_settle_releases_reservation_without_recreating_discarded_session(
    settle_method,
):
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        await rebalancer.register_session_context(
            session_id="discard-before-settle",
            group_id="group",
            group_size=1,
            task_key="task",
        )
        lease = await rebalancer.acquire(
            session_id="discard-before-settle",
            input_ids=[1] * 100,
        )
        load = rebalancer.loads[lease.worker_url]
        assert load.reserved_requests == 1
        assert load.reserved_tokens > 0
        assert load.reserved_prefill_tokens > 0

        await rebalancer.discard_session_context("discard-before-settle")
        if settle_method == "complete":
            await rebalancer.complete(
                lease,
                response_meta={
                    "cached_tokens": 0,
                    "queue_time": 0.0,
                    "e2e_latency": 1.0,
                    "decode_throughput": 10.0,
                },
                output_tokens=1,
                committed_tokens=[1] * 101,
            )
        else:
            await rebalancer.fail(lease)

        assert "discard-before-settle" not in rebalancer.sessions
        assert (await rebalancer.snapshot())["active_sessions"] == 0
        assert load.reserved_requests == 0
        assert load.reserved_tokens == 0
        assert load.reserved_prefill_tokens == 0

    run(scenario())


def test_registered_context_acquire_rejects_session_discarded_before_acquire():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        await rebalancer.register_session_context(
            session_id="discard-before-acquire",
            group_id="group",
            group_size=1,
            task_key="task",
        )
        await rebalancer.discard_session_context("discard-before-acquire")

        with pytest.raises(RuntimeError, match="context.*registered|discarded"):
            await rebalancer.acquire(
                session_id="discard-before-acquire",
                input_ids=[1] * 100,
                require_registered_context=True,
            )

        assert "discard-before-acquire" not in rebalancer.sessions
        assert (await rebalancer.snapshot())["active_sessions"] == 0

    run(scenario())


def test_partial_rollout_request_rejects_context_discarded_before_acquire():
    app = create_app(
        tokenizer=FakeTokenizer(),
        tokenizer_path="model",
        token_build_mode="snapshot",
        sglang_client=DirectGenerationClient(),
        enable_engine_rebalancing=True,
        engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
        engine_rebalancing_model_config=simple_model_config(),
        tool_call_parse_backend="local",
        reasoning_parse_backend="local",
        stream_heartbeat_interval_seconds=0,
    )

    with TestClient(app) as http_client:
        assert http_client.post(
            "/v1/session/context",
            json={"session_id": "discarded-request", "group_size": 1},
        ).status_code == 200
        assert http_client.delete(
            "/v1/session/context/discarded-request"
        ).status_code == 200

        response = http_client.post(
            "/v1/chat/completions",
            headers={
                "X-Session-ID": "discarded-request",
                "X-Dressage-Partial-Rollout": "1",
            },
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        assert response.status_code == 503
        assert "context" in str(response.json()["detail"]["message"])
        assert http_client.get("/v1/engines/load").json()["active_sessions"] == 0


def test_general_request_without_partial_rollout_header_creates_routing_session():
    app = create_app(
        tokenizer=FakeTokenizer(),
        tokenizer_path="model",
        token_build_mode="snapshot",
        sglang_client=DirectGenerationClient(),
        enable_engine_rebalancing=True,
        engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
        engine_rebalancing_model_config=simple_model_config(),
        tool_call_parse_backend="local",
        reasoning_parse_backend="local",
        stream_heartbeat_interval_seconds=0,
    )

    with TestClient(app) as http_client:
        response = http_client.post(
            "/v1/chat/completions",
            headers={"X-Session-ID": "general-request"},
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        assert response.status_code == 200
        assert http_client.get("/v1/engines/load").json()["active_sessions"] == 1


def test_registered_partial_rollout_request_acquires_normally():
    app = create_app(
        tokenizer=FakeTokenizer(),
        tokenizer_path="model",
        token_build_mode="snapshot",
        sglang_client=DirectGenerationClient(),
        enable_engine_rebalancing=True,
        engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
        engine_rebalancing_model_config=simple_model_config(),
        tool_call_parse_backend="local",
        reasoning_parse_backend="local",
        stream_heartbeat_interval_seconds=0,
    )

    with TestClient(app) as http_client:
        assert http_client.post(
            "/v1/session/context",
            json={"session_id": "registered-request", "group_size": 1},
        ).status_code == 200

        response = http_client.post(
            "/v1/chat/completions",
            headers={
                "X-Session-ID": "registered-request",
                "X-Dressage-Partial-Rollout": "1",
            },
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        assert response.status_code == 200
        assert http_client.get("/v1/engines/load").json()["active_sessions"] == 1


def test_session_context_delete_endpoint_requires_auth_and_is_idempotent():
    app = create_app(
        tokenizer=FakeTokenizer(),
        tokenizer_path="model",
        token_build_mode="snapshot",
        sglang_client=DirectGenerationClient(),
        api_key="proxy-secret",
        enable_engine_rebalancing=True,
        engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
        engine_rebalancing_model_config=simple_model_config(),
        tool_call_parse_backend="local",
        reasoning_parse_backend="local",
    )
    headers = {"Authorization": "Bearer proxy-secret"}

    with TestClient(app) as http_client:
        registered = http_client.post(
            "/v1/session/context",
            headers=headers,
            json={"session_id": "discard-me", "group_size": 1},
        )
        assert registered.status_code == 200
        assert http_client.get("/v1/engines/load", headers=headers).json()[
            "active_sessions"
        ] == 1

        unauthorized = http_client.delete("/v1/session/context/discard-me")
        assert unauthorized.status_code == 401

        first = http_client.delete(
            "/v1/session/context/discard-me", headers=headers
        )
        second = http_client.delete(
            "/v1/session/context/discard-me", headers=headers
        )

        assert first.status_code == second.status_code == 200
        assert first.json() == second.json() == {
            "success": True,
            "session_id": "discard-me",
        }
        assert http_client.get("/v1/engines/load", headers=headers).json()[
            "active_sessions"
        ] == 0


@pytest.mark.parametrize("settle_method", ["complete", "fail"])
def test_request_cancellation_waits_for_routing_lease_settle(
    monkeypatch, settle_method
):
    class ControlledGenerationClient(DirectGenerationClient):
        def __init__(self):
            super().__init__()
            self.generation_started = asyncio.Event()
            self.generation_release = asyncio.Event()

        async def generate(self, *args, **kwargs):
            self.generation_started.set()
            await self.generation_release.wait()
            if settle_method == "fail":
                raise RuntimeError("generation boom")
            return await super().generate(*args, **kwargs)

    async def scenario():
        generation_client = ControlledGenerationClient()
        app = create_app(
            tokenizer=FakeTokenizer(),
            tokenizer_path="model",
            token_build_mode="snapshot",
            sglang_client=generation_client,
            enable_engine_rebalancing=True,
            engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
            engine_rebalancing_model_config=simple_model_config(),
            tool_call_parse_backend="local",
            reasoning_parse_backend="local",
            stream_heartbeat_interval_seconds=0,
        )
        rebalancer = app.state.engine_rebalancer
        await rebalancer.refresh()
        original_settle = getattr(rebalancer, settle_method)
        settle_started = asyncio.Event()
        settle_tasks: list[asyncio.Task] = []
        settle_leases: list[RoutingLease] = []

        async def tracked_settle(*args, **kwargs):
            settle_tasks.append(asyncio.current_task())
            settle_leases.append(args[0])
            settle_started.set()
            return await original_settle(*args, **kwargs)

        monkeypatch.setattr(rebalancer, settle_method, tracked_settle)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
        ) as http_client:
            request_task = asyncio.create_task(
                http_client.post(
                    "/v1/chat/completions",
                    headers={"X-Session-ID": "cancel-settle"},
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )
            )
            await generation_client.generation_started.wait()
            await rebalancer._lock.acquire()
            try:
                generation_client.generation_release.set()
                await settle_started.wait()
                load = rebalancer.loads[settle_leases[0].worker_url]
                assert load.reserved_requests == 1
                assert load.reserved_tokens > 0
                assert load.reserved_prefill_tokens > 0

                request_task.cancel()
                await asyncio.sleep(0)
                request_task.cancel()
                await asyncio.sleep(0)
                assert not request_task.done()
            finally:
                rebalancer._lock.release()

            with pytest.raises(asyncio.CancelledError):
                await request_task

        assert len(settle_tasks) == 1
        assert settle_tasks[0].done()
        assert load.reserved_requests == 0
        assert load.reserved_tokens == 0
        assert load.reserved_prefill_tokens == 0
        assert rebalancer.sessions["cancel-settle"].pending_owner_worker_url is None

    run(scenario())


def test_generation_failure_is_not_masked_when_lease_fail_settle_raises(monkeypatch):
    class FailingGenerationClient(DirectGenerationClient):
        async def generate(self, *args, **kwargs):
            raise RuntimeError("generation boom")

    async def scenario():
        app = create_app(
            tokenizer=FakeTokenizer(),
            tokenizer_path="model",
            token_build_mode="snapshot",
            sglang_client=FailingGenerationClient(),
            enable_engine_rebalancing=True,
            engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
            engine_rebalancing_model_config=simple_model_config(),
            tool_call_parse_backend="local",
            reasoning_parse_backend="local",
            stream_heartbeat_interval_seconds=0,
        )

        async def fail_settle(_lease):
            raise RuntimeError("settle boom")

        monkeypatch.setattr(app.state.engine_rebalancer, "fail", fail_settle)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
        ) as http_client:
            with pytest.raises(RuntimeError, match="generation boom"):
                await http_client.post(
                    "/v1/chat/completions",
                    headers={"X-Session-ID": "generation-fail"},
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )

    run(scenario())


def test_cancelled_lease_settle_logs_late_failure_without_masking_cancel(caplog):
    async def scenario():
        settle_started = asyncio.Event()
        settle_release = asyncio.Event()

        async def settle():
            settle_started.set()
            await settle_release.wait()
            raise RuntimeError("late settle boom")

        task = asyncio.create_task(_settle_routing_lease(settle()))
        await settle_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        settle_release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.WARNING, logger="dressage.proxy.server"):
        run(scenario())

    assert "routing lease settle failed after caller cancellation" in caplog.text
    assert "late settle boom" in caplog.text


def test_self_cancelled_fail_settle_does_not_mask_generation_failure(
    monkeypatch, caplog
):
    class FailingGenerationClient(DirectGenerationClient):
        async def generate(self, *args, **kwargs):
            raise RuntimeError("generation boom")

    async def scenario():
        app = create_app(
            tokenizer=FakeTokenizer(),
            tokenizer_path="model",
            token_build_mode="snapshot",
            sglang_client=FailingGenerationClient(),
            enable_engine_rebalancing=True,
            engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
            engine_rebalancing_model_config=simple_model_config(),
            tool_call_parse_backend="local",
            reasoning_parse_backend="local",
            stream_heartbeat_interval_seconds=0,
        )

        async def self_cancelled_fail(_lease):
            raise asyncio.CancelledError()

        monkeypatch.setattr(
            app.state.engine_rebalancer, "fail", self_cancelled_fail
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
        ) as http_client:
            with pytest.raises(RuntimeError, match="generation boom"):
                await http_client.post(
                    "/v1/chat/completions",
                    headers={"X-Session-ID": "self-cancelled-fail"},
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )

    with caplog.at_level(logging.WARNING, logger="dressage.proxy.server"):
        run(scenario())

    assert "engine rebalancing failure settle failed" in caplog.text


def test_self_cancelled_complete_settle_does_not_change_generation_response(
    monkeypatch, caplog
):
    async def scenario():
        app = create_app(
            tokenizer=FakeTokenizer(),
            tokenizer_path="model",
            token_build_mode="snapshot",
            sglang_client=DirectGenerationClient(),
            enable_engine_rebalancing=True,
            engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
            engine_rebalancing_model_config=simple_model_config(),
            tool_call_parse_backend="local",
            reasoning_parse_backend="local",
            stream_heartbeat_interval_seconds=0,
        )

        async def self_cancelled_complete(*args, **kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(
            app.state.engine_rebalancer, "complete", self_cancelled_complete
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
        ) as http_client:
            response = await http_client.post(
                "/v1/chat/completions",
                headers={"X-Session-ID": "self-cancelled-complete"},
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "x"

    with caplog.at_level(logging.WARNING, logger="dressage.proxy.server"):
        run(scenario())

    assert "engine rebalancing observation failed" in caplog.text


def test_complete_observation_failure_does_not_change_generation_response(monkeypatch):
    async def scenario():
        app = create_app(
            tokenizer=FakeTokenizer(),
            tokenizer_path="model",
            token_build_mode="snapshot",
            sglang_client=DirectGenerationClient(),
            enable_engine_rebalancing=True,
            engine_rebalancing_config=EngineRebalancingConfig(enabled=True),
            engine_rebalancing_model_config=simple_model_config(),
            tool_call_parse_backend="local",
            reasoning_parse_backend="local",
            stream_heartbeat_interval_seconds=0,
        )

        async def complete_settle(*args, **kwargs):
            raise RuntimeError("observation boom")

        monkeypatch.setattr(app.state.engine_rebalancer, "complete", complete_settle)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
        ) as http_client:
            response = await http_client.post(
                "/v1/chat/completions",
                headers={"X-Session-ID": "observation-fail"},
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "x"

    run(scenario())


def _benchmark_heredoc(function_name: str) -> str:
    path = Path(
        "examples/scripts/benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    source = path.read_text(encoding="utf-8")
    marker = f"{function_name}() {{"
    assert marker in source, f"{function_name} is missing"
    function_start = source.index(marker)
    heredoc_start = source.index("<<'PY'\n", function_start) + len("<<'PY'\n")
    heredoc_end = source.index("\nPY\n", heredoc_start)
    return source[heredoc_start:heredoc_end]


def _run_benchmark_heredoc(
    function_name: str,
    *args: str,
    env: dict[str, str] | None = None,
    file_size_limit: int | None = None,
) -> subprocess.CompletedProcess[str]:
    def limit_file_size() -> None:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_limit, file_size_limit))

    return subprocess.run(
        [sys.executable, "-c", _benchmark_heredoc(function_name), *args],
        cwd=Path.cwd(),
        env=env,
        preexec_fn=limit_file_size if file_size_limit is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )


def test_engine_rebalancing_benchmark_prepares_once_before_both_runs():
    source = Path(
        "examples/scripts/benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
    ).read_text(encoding="utf-8")
    prepare_call = (
        'prepare_long_tail_prompts "${PROMPT_SOURCE}" '
        '"${PROMPT_EFFECTIVE}" "${BENCHMARK_SEED}"'
    )
    run_loop = 'for index in "${!RUN_NAMES[@]}"; do\n  run_one'
    run_one = source[source.index("run_one() {") : source.index("write_summary() {")]

    assert source.count(prepare_call) == 1
    assert source.index(prepare_call) < source.index(run_loop)
    assert run_one.count('export PROMPT_DATA="${PROMPT_EFFECTIVE}"') == 1
    assert "prepare_long_tail_prompts" not in run_one


def test_engine_rebalancing_benchmark_environment_records_prompt_fingerprints(
    tmp_path,
):
    repo = Path.cwd()
    benchmark_script = repo / (
        "examples/scripts/"
        "benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    source_recipe = repo / (
        "examples/scripts/run_blackbox_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    prompt_source = repo / "examples/data/dressage_dapo_prompts_long_tail.jsonl"
    prompt_effective = tmp_path / "prompts.deterministic.jsonl"
    prompt_effective.write_text("effective\n", encoding="utf-8")
    output = tmp_path / "environment.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '0, Test GPU, uuid, driver'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)

    result = _run_benchmark_heredoc(
        "record_environment",
        str(repo),
        str(source_recipe),
        str(benchmark_script),
        str(prompt_source),
        str(prompt_effective),
        str(output),
        "seed20260806-off-r1",
        "off",
        "20260806",
        "0",
        "256",
        "1",
        "256",
        "12288",
        "16",
        "3600",
        "20",
        "16gb",
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0, result.stderr
    environment = dict(
        line.split("=", 1)
        for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert environment["prompt_source"] == str(prompt_source)
    assert environment["prompt_source_sha256"] == hashlib.sha256(
        prompt_source.read_bytes()
    ).hexdigest()
    assert environment["prompt_effective"] == str(prompt_effective)
    assert environment["prompt_effective_sha256"] == hashlib.sha256(
        prompt_effective.read_bytes()
    ).hexdigest()
    assert environment["rollout_batch_size"] == "256"
    assert environment["n_samples_per_prompt"] == "1"
    assert environment["global_batch_size"] == "256"
    assert environment["rollout_max_response_len"] == "12288"
    assert environment["sandbox_slots_per_node"] == "16"
    assert environment["sandbox_acquire_timeout_sec"] == "3600"
    assert "prompt_source_workload_distribution_json" in environment
    assert "prompt_effective_workload_distribution_json" in environment


def test_engine_rebalancing_benchmark_collector_uses_sampling_seed_identity(tmp_path):
    run_dir = tmp_path / "run"
    samples_dir = run_dir / "runtime" / "traj_payload" / "run" / "samples"
    samples_dir.mkdir(parents=True)
    samples = [
        ("z.json", "alpha", 29, 1),
        ("a.json", "alpha", 11, 0),
        ("y.json", "beta", 41, 0),
        ("b.json", "beta", 17, 1),
    ]
    for filename, instance_id, sampling_seed, segment_index in samples:
        (samples_dir / filename).write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "segment_index": segment_index,
                    "tokens": [1, 2],
                    "status": "complete",
                    "reward": 1.0,
                    "metadata": {"rollout_sampling_seed": sampling_seed},
                    "loss_mask": [1, 1],
                }
            ),
            encoding="utf-8",
        )

    result = _run_benchmark_heredoc(
        "collect_run", str(run_dir), "run", "off", "0", "1", "2", "2"
    )

    assert result.returncode == 0, result.stderr
    hash_lines = (run_dir / "trajectory_hashes.txt").read_text(encoding="utf-8")
    assert "instance_id=alpha sampling_seed=11 segment_index=0" in hash_lines
    assert "instance_id=alpha sampling_seed=29 segment_index=1" in hash_lines
    assert "instance_id=beta sampling_seed=17 segment_index=1" in hash_lines
    assert hash_lines.index("instance_id=alpha sampling_seed=11") < hash_lines.index(
        "instance_id=alpha sampling_seed=29"
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert not any("sampling seed" in error for error in metrics["acceptance_errors"])

    duplicate = json.loads((samples_dir / "b.json").read_text(encoding="utf-8"))
    duplicate["metadata"]["rollout_sampling_seed"] = 41
    (samples_dir / "b.json").write_text(json.dumps(duplicate), encoding="utf-8")
    duplicate_result = _run_benchmark_heredoc(
        "collect_run", str(run_dir), "run", "off", "0", "1", "2", "2"
    )

    assert duplicate_result.returncode == 0, duplicate_result.stderr
    duplicate_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert any("sampling seed" in error for error in duplicate_metrics["acceptance_errors"])


def test_engine_rebalancing_benchmark_collector_accepts_one_sample_per_prompt(tmp_path):
    run_dir = tmp_path / "run"
    sample_path = run_dir / "runtime" / "traj_payload" / "run" / "samples" / "a.json"
    _write_benchmark_sample(
        sample_path,
        instance_id="alpha",
        sampling_seed=11,
        segment_index=0,
    )

    result = _run_benchmark_heredoc(
        "collect_run", str(run_dir), "run", "off", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert not any("sampling seed" in error for error in metrics["acceptance_errors"])


def _write_benchmark_sample(
    path: Path,
    *,
    instance_id: str,
    sampling_seed: int | None,
    segment_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if sampling_seed is not None:
        metadata["rollout_sampling_seed"] = sampling_seed
    path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "segment_index": segment_index,
                "tokens": [1, 2],
                "status": "complete",
                "reward": 1.0,
                "metadata": metadata,
                "loss_mask": [1, 1],
            }
        ),
        encoding="utf-8",
    )


def _collect_benchmark_run(run_dir: Path) -> dict:
    result = _run_benchmark_heredoc(
        "collect_run", str(run_dir), "run", "off", "0", "1", "2", "2"
    )
    assert result.returncode == 0, result.stderr
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def test_engine_rebalancing_benchmark_collector_rejects_missing_sampling_seed(
    tmp_path,
):
    samples = tmp_path / "runtime" / "traj_payload" / "run" / "samples"
    _write_benchmark_sample(
        samples / "a.json",
        instance_id="alpha",
        sampling_seed=11,
        segment_index=0,
    )
    _write_benchmark_sample(
        samples / "b.json",
        instance_id="alpha",
        sampling_seed=None,
        segment_index=0,
    )

    metrics = _collect_benchmark_run(tmp_path)

    assert any(
        "instance alpha is missing a rollout sampling seed" in error
        for error in metrics["acceptance_errors"]
    )


def test_engine_rebalancing_benchmark_collector_allows_segments_per_base_seed(
    tmp_path,
):
    samples = tmp_path / "runtime" / "traj_payload" / "run" / "samples"
    for sampling_seed in (11, 29):
        for segment_index in (0, 1):
            _write_benchmark_sample(
                samples / f"{sampling_seed}-{segment_index}.json",
                instance_id="alpha",
                sampling_seed=sampling_seed,
                segment_index=segment_index,
            )

    metrics = _collect_benchmark_run(tmp_path)

    assert not any(
        "sampling seed" in error for error in metrics["acceptance_errors"]
    )


def test_engine_rebalancing_benchmark_trajectory_hash_uses_seed_not_file_order(
    tmp_path,
):
    records = [
        ("alpha", 11, 0),
        ("alpha", 29, 1),
        ("beta", 17, 1),
        ("beta", 41, 0),
    ]
    first_run = tmp_path / "first"
    second_run = tmp_path / "second"
    changed_seed_run = tmp_path / "changed-seed"
    for index, (instance_id, sampling_seed, segment_index) in enumerate(records):
        _write_benchmark_sample(
            first_run
            / "runtime"
            / "traj_payload"
            / "run"
            / f"batch-{index}"
            / "samples"
            / f"sample-{index}.json",
            instance_id=instance_id,
            sampling_seed=sampling_seed,
            segment_index=segment_index,
        )
        reverse_index = len(records) - index
        _write_benchmark_sample(
            second_run
            / "runtime"
            / "traj_payload"
            / "run"
            / f"batch-{reverse_index}"
            / "samples"
            / f"sample-{reverse_index}.json",
            instance_id=instance_id,
            sampling_seed=sampling_seed,
            segment_index=segment_index,
        )
        _write_benchmark_sample(
            changed_seed_run
            / "runtime"
            / "traj_payload"
            / "run"
            / f"batch-{index}"
            / "samples"
            / f"sample-{index}.json",
            instance_id=instance_id,
            sampling_seed=30 if sampling_seed == 29 else sampling_seed,
            segment_index=segment_index,
        )

    first_hash = _collect_benchmark_run(first_run)["trajectory_hash"]
    second_hash = _collect_benchmark_run(second_run)["trajectory_hash"]
    changed_seed_hash = _collect_benchmark_run(changed_seed_run)["trajectory_hash"]

    assert first_hash == second_hash
    assert changed_seed_hash != first_hash


def test_sync_local_script_gracefully_stops_proxy_before_ray():
    path = Path("examples/scripts/run_blackbox_qwen3.5_4b_sync_local.sh")
    source = path.read_text()
    cleanup = source[source.index("cleanup() {") : source.index("trap cleanup EXIT")]

    assert cleanup.index("_stop_proxy_on_exit") < cleanup.index(
        "_stop_ray_cluster_on_exit"
    )
    assert '[[ "${wait_count}" -lt 100 ]]' in source
    assert "Dressage proxy did not stop gracefully; sending SIGKILL" in source
