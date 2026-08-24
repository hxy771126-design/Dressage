from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import resource
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from dressage.proxy.rebalancing import EngineRebalancer, EngineRebalancingConfig
from dressage.proxy.rebalancing._batch_milp import (
    BatchSolution,
    BatchSolverError,
    SolverStatus,
)
from dressage.proxy.rebalancing.cache_hit_estimator import (
    CacheSource,
    ContextRecoveryEstimate,
    context_bucket,
    longest_common_prefix_length,
)
from dressage.proxy.rebalancing.context_recovery_model import PerformanceHistory
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
    ContextPathReadiness,
)
from tests.test_proxy import FakeTokenizer


def run(coro):
    return asyncio.run(coro)


def serve_current_loads_for_batch(client, rebalancer):
    async def get_worker_loads(url):
        load = rebalancer.loads[url]
        return {
            "loads": [
                {
                    "num_running_reqs": load.running,
                    "num_waiting_reqs": load.queued,
                    "num_total_tokens": load.active_tokens,
                    "max_total_num_tokens": load.token_capacity or 100_000,
                    "max_running_requests": load.request_capacity or 100,
                    "token_usage": load.token_usage,
                    "num_waiting_uncached_tokens": load.waiting_uncached_tokens,
                }
            ]
        }

    client.get_worker_loads = get_worker_loads


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


class ControlledBatchLoadClient(ControlPlaneClient):
    def __init__(self):
        super().__init__()
        self.batch_load_calls = {url: 0 for url in self.urls}
        self.batch_load_futures = {url: [] for url in self.urls}
        self.batch_load_started = None

    def control_batch_loads(self):
        self.batch_load_started = asyncio.Queue()

    async def get_worker_loads(self, url):
        if self.batch_load_started is None:
            return await super().get_worker_loads(url)
        index = self.batch_load_calls[url]
        self.batch_load_calls[url] += 1
        future = asyncio.get_running_loop().create_future()
        self.batch_load_futures[url].append(future)
        self.batch_load_started.put_nowait((url, index))
        return await future

    def resolve_batch(
        self,
        index,
        *,
        running=0,
        queued=0,
        waiting_uncached=None,
        gen_throughput=0.0,
    ):
        payload = {
            "loads": [
                {
                    "num_running_reqs": running,
                    "num_waiting_reqs": queued,
                    "num_total_tokens": 0,
                    "max_total_num_tokens": 100_000,
                    "max_running_requests": 100,
                    "token_usage": 0.0,
                    "gen_throughput": gen_throughput,
                }
            ]
        }
        if waiting_uncached is not None:
            payload["loads"][0]["num_waiting_uncached_tokens"] = waiting_uncached
        for url in self.urls:
            self.batch_load_futures[url][index].set_result(payload)

    def resolve_url(
        self,
        url,
        index,
        *,
        running=0,
        queued=0,
        gen_throughput=0.0,
    ):
        self.batch_load_futures[url][index].set_result(
            {
                "loads": [
                    {
                        "num_running_reqs": running,
                        "num_waiting_reqs": queued,
                        "num_total_tokens": 0,
                        "max_total_num_tokens": 100_000,
                        "max_running_requests": 100,
                        "token_usage": 0.0,
                        "gen_throughput": gen_throughput,
                    }
                ]
            }
        )

    def fail_url(self, url, index):
        self.batch_load_futures[url][index].set_exception(RuntimeError("load failed"))


async def wait_for_condition(condition, *, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not satisfied before timeout")
        await asyncio.sleep(0)


def test_config_metrics_staleness_is_independent_of_control_poll_interval():
    config = EngineRebalancingConfig(load_poll_interval_ms=750)
    assert config.metrics_stale_ms == 2_000
    explicit = EngineRebalancingConfig(
        load_poll_interval_ms=750,
        metrics_stale_ms=3_000,
    )
    assert explicit.metrics_stale_ms == 3_000


@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_non_positive_metrics_staleness(value):
    with pytest.raises(ValueError, match="metrics_stale_ms"):
        EngineRebalancingConfig(metrics_stale_ms=value)


def test_config_defaults_propagate_to_online_models():
    config = EngineRebalancingConfig(enabled=True)
    assert config.snapshot()["load_poll_interval_ms"] == 250
    assert config.snapshot()["load_batch_coalescing_window_ms"] == 125
    assert config.snapshot()["history_size"] == 512
    assert config.snapshot()["min_samples"] == 16
    assert "min_hold_turns" not in config.snapshot()
    assert config.snapshot()["min_risk_ms"] == 10
    assert config.snapshot()["cold_start_hit_probability"] == 1.0
    assert config.snapshot()["min_load_improvement_ratio"] == 0.10

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


def test_config_accepts_zero_and_rejects_negative_load_batch_window():
    assert EngineRebalancingConfig(
        load_batch_coalescing_window_ms=0
    ).load_batch_coalescing_window_ms == 0
    with pytest.raises(ValueError, match="load_batch_coalescing_window_ms"):
        EngineRebalancingConfig(load_batch_coalescing_window_ms=-1)


def test_load_score_adds_only_unseen_request_ledger_to_queue():
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
        load.reserved_requests = 5
        load.queued = 3
        load.active_tokens = 1_000
        load.reserved_tokens = 90_000
        load.request_capacity = 100
        load.token_capacity = 100_000
        load.token_usage = 0.005
        load.waiting_uncached_tokens = 80
        load.reserved_prefill_tokens = 90_000
        rebalancer.loads[client.urls[1]].queued = 1_000

        observed = rebalancer._load_score(
            target,
            token_increment=0,
            include_pending_request=False,
        )
        assert observed.request_pressure == pytest.approx(0.02)
        assert observed.token_pressure == pytest.approx(0.01)
        assert observed.queue_pressure == pytest.approx(0.03)
        assert observed.total == pytest.approx(0.06)

        load.reserved_requests = 7
        lagged = rebalancer._load_score(
            target,
            token_increment=0,
            include_pending_request=False,
        )
        projected = rebalancer._load_score(
            target,
            token_increment=100,
            include_pending_request=True,
        )

        assert lagged.request_pressure == pytest.approx(0.02)
        assert lagged.token_pressure == pytest.approx(0.01)
        assert lagged.queue_pressure == pytest.approx(0.05)
        assert lagged.total == pytest.approx(0.08)
        assert projected.request_pressure == lagged.request_pressure
        assert projected.token_pressure == pytest.approx(0.011)
        assert projected.queue_pressure == pytest.approx(0.06)
        assert projected.total == pytest.approx(0.091)
        assert set(projected.__dict__) == {
            "request_pressure",
            "token_pressure",
            "queue_pressure",
            "total",
        }

        load.running = 4
        handed_off = rebalancer._load_score(
            target,
            token_increment=0,
            include_pending_request=False,
        )
        assert handed_off.request_pressure == pytest.approx(0.04)
        assert handed_off.queue_pressure == pytest.approx(0.03)
        assert handed_off.total == lagged.total

        load.running = 2
        load.token_usage = 0.02
        usage_floor = rebalancer._load_score(
            target,
            token_increment=0,
            include_pending_request=False,
        )
        assert usage_floor.token_pressure == pytest.approx(0.02)
        assert usage_floor.total == pytest.approx(0.09)

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


def test_longest_common_prefix_length():
    assert longest_common_prefix_length([1, 2, 3], [1, 2, 9]) == 2


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
        performance.prefill_throughput(
            fingerprint="fp",
            engine_url="worker",
            context_tokens=100,
        )
        == 100.0
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


def test_first_acquire_waits_for_calibration_before_engine_discovery():
    class CountingControlPlaneClient(ControlPlaneClient):
        def __init__(self):
            super().__init__()
            self.list_workers_calls = 0

        async def list_workers(self):
            self.list_workers_calls += 1
            return await super().list_workers()

    client = CountingControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            load_batch_coalescing_window_ms=0,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        calibration_gate = asyncio.Event()

        async def blocked_calibration():
            await calibration_gate.wait()

        rebalancer._calibration_task = asyncio.create_task(blocked_calibration())
        acquire_task = asyncio.create_task(
            rebalancer.acquire(session_id="first", input_ids=[1, 2, 3])
        )
        await asyncio.sleep(0)
        assert client.list_workers_calls == 0

        calibration_gate.set()
        lease = await acquire_task
        try:
            assert client.list_workers_calls == 1
            assert lease.worker_url in client.urls
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_control_refresh_tracks_topology_without_fetching_worker_loads():
    class MutableControlPlaneClient(ControlPlaneClient):
        def __init__(self):
            super().__init__()
            self.health = {url: True for url in self.urls}
            self.versions = {url: "7" for url in self.urls}
            self.load_calls = 0

        async def list_workers(self):
            return [
                {
                    "url": url,
                    "is_healthy": self.health[url],
                    "connection_mode": "http",
                }
                for url in self.urls
            ]

        async def get_worker_loads(self, url):
            self.load_calls += 1
            return await super().get_worker_loads(url)

        async def get_worker_weight_version(self, url):
            return self.versions[url]

    client = MutableControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()

        assert client.load_calls == 0
        assert set(rebalancer.deployments) == set(client.urls)
        assert set(rebalancer.loads) == set(client.urls)
        assert all(load.healthy for load in rebalancer.loads.values())
        assert all(load.metrics_timestamp == 0 for load in rebalancer.loads.values())

        unhealthy = client.urls[0]
        existing = client.urls[1]
        added = "http://node-c:30000"
        client.health[unhealthy] = False
        client.urls.append(added)
        client.health[added] = True
        client.versions[added] = "8"
        client.versions[existing] = "8"
        rebalancer._deployment_refresh_seconds = 0

        await rebalancer.refresh()

        assert client.load_calls == 0
        assert rebalancer.loads[unhealthy].healthy is False
        assert rebalancer.loads[existing].healthy is True
        assert rebalancer.loads[added].healthy is True
        assert rebalancer.deployments[existing].weight_version == "8"
        assert rebalancer.deployments[added].weight_version == "8"

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
    assert delays[:5] == [0.25, 1.0, 2.0, 5.0, 5.0]
    assert all(delay == 0.25 for delay in delays[5:])
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
            assert first.decision.reason == "batch_new_session"
            assert second.decision.reason == "batch_new_session"
            assert first.decision.source_base_load is None
            assert first.decision.target_projected_load is not None
            assert first.decision.target_projected_load.total > 0
        finally:
            await rebalancer.fail(first)
            await rebalancer.fail(second)

    run(scenario())


def test_batch_acquires_fetch_each_engine_once_and_publish_one_trace():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        tasks = [
            asyncio.create_task(
                rebalancer.acquire(session_id=f"batch-{index}", input_ids=[index] * 10)
            )
            for index in range(4)
        ]
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        assert client.batch_load_calls == {url: 1 for url in client.urls}

        client.resolve_batch(0)
        leases = await asyncio.gather(*tasks)
        snapshot = await rebalancer.snapshot()

        assert len({lease.batch_id for lease in leases}) == 1
        assert len(snapshot["recent_load_batches"]) == 1
        trace = snapshot["recent_load_batches"][0]
        assert trace["batch"]["registered_count"] == 4
        assert [step["session_id"] for step in trace["steps"]] == [
            "batch-0",
            "batch-1",
            "batch-2",
            "batch-3",
        ]
        for lease in leases:
            await rebalancer.fail(lease)

    run(scenario())


def test_batch_freeze_resolves_step_budget_once_per_fingerprint(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    original = rebalancer._resolve_step_budget
    fingerprints = []

    def record_budget_resolution(**kwargs):
        fingerprints.append(kwargs["fingerprint"])
        return original(**kwargs)

    monkeypatch.setattr(rebalancer, "_resolve_step_budget", record_budget_resolution)

    async def scenario():
        await rebalancer.refresh()
        fingerprint = rebalancer.deployments[client.urls[0]].cache_fingerprint
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="budget-cache", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        lease = await task
        try:
            assert fingerprints == [fingerprint]
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_batch_snapshot_refreshes_pool_readiness_after_control_only_polling():
    async def benchmark(task, payload):
        del task
        return CalibrationSample(
            latency_seconds=0.01,
            bandwidth_bytes_per_second=max(1, payload * 10),
        )

    client = ControlPlaneClient(shared_l3=True)
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        fingerprint = rebalancer.deployments[client.urls[0]].cache_fingerprint
        assert rebalancer.pools[fingerprint].state is SchedulerState.BOOTSTRAP

        first = await rebalancer.acquire(
            session_id="readiness-first",
            input_ids=[1] * 10,
        )
        assert first.decision.state is SchedulerState.ACTIVE
        await rebalancer.fail(first)

        stale_timestamp = time.monotonic() - (
            rebalancer.config.metrics_stale_ms / 1000.0
        ) - 1.0
        for load in rebalancer.loads.values():
            load.metrics_timestamp = stale_timestamp
        await rebalancer.refresh()
        assert rebalancer.pools[fingerprint].state is SchedulerState.DEGRADED

        second = await rebalancer.acquire(
            session_id="readiness-second",
            input_ids=[2] * 10,
        )
        try:
            assert second.decision.state is SchedulerState.ACTIVE
            snapshot = await rebalancer.snapshot()
            assert len(snapshot["recent_load_batches"]) == 2
            assert all(
                engine["fetch_status"] == "ok"
                for engine in snapshot["recent_load_batches"][-1]["engines"]
            )
        finally:
            await rebalancer.fail(second)

    run(scenario())


def test_configured_load_batch_window_collects_step_after_fast_fetch():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_samples=1,
            load_batch_coalescing_window_ms=50,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        first_task = asyncio.create_task(
            rebalancer.acquire(session_id="window-first", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        await asyncio.sleep(0.01)
        second_task = asyncio.create_task(
            rebalancer.acquire(session_id="window-second", input_ids=[2] * 10)
        )

        async def resolve_second_batch_if_started():
            try:
                await wait_for_condition(
                    lambda: sum(client.batch_load_calls.values())
                    == 2 * len(client.urls),
                    timeout=0.2,
                )
            except AssertionError:
                return
            client.resolve_batch(1)

        resolver = asyncio.create_task(resolve_second_batch_if_started())
        first, second = await asyncio.gather(first_task, second_task)
        resolver.cancel()
        await asyncio.gather(resolver, return_exceptions=True)
        trace = (await rebalancer.snapshot())["recent_load_batches"][0]

        assert first.batch_id == second.batch_id
        assert client.batch_load_calls == {url: 1 for url in client.urls}
        assert trace["batch"]["registered_count"] == 2
        assert trace["batch"]["collect_seconds"] >= 0.045
        await rebalancer.fail(first)
        await rebalancer.fail(second)

    run(scenario())


def test_slow_load_fetch_dominates_configured_batch_window():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_samples=1,
            load_batch_coalescing_window_ms=20,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="slow-fetch", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        await asyncio.sleep(0.04)
        client.resolve_batch(0)
        lease = await task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]["batch"]

        assert trace["fetch_seconds"] >= 0.035
        assert trace["collect_seconds"] >= trace["fetch_seconds"]
        assert trace["collect_seconds"] - trace["fetch_seconds"] < 0.02
        await rebalancer.fail(lease)

    run(scenario())


def test_queued_batch_window_starts_when_first_step_arrives(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_samples=1,
            load_batch_coalescing_window_ms=40,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        first_commit_started = asyncio.Event()
        allow_first_commit = asyncio.Event()
        original_commit = rebalancer._commit_batch
        commit_count = 0

        async def controlled_commit(*args, **kwargs):
            nonlocal commit_count
            commit_count += 1
            if commit_count == 1:
                first_commit_started.set()
                await allow_first_commit.wait()
            return await original_commit(*args, **kwargs)

        monkeypatch.setattr(rebalancer, "_commit_batch", controlled_commit)
        first_task = asyncio.create_task(
            rebalancer.acquire(session_id="queued-first", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        await first_commit_started.wait()

        second_task = asyncio.create_task(
            rebalancer.acquire(session_id="queued-second", input_ids=[2] * 10)
        )
        await asyncio.sleep(0.06)
        assert client.batch_load_calls == {url: 1 for url in client.urls}
        allow_first_commit.set()
        first = await first_task
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == 2 * len(client.urls)
        )
        client.resolve_batch(1)
        second = await second_task
        trace = next(
            item
            for item in (await rebalancer.snapshot())["recent_load_batches"]
            if item["batch"]["id"] == second.batch_id
        )["batch"]

        assert second.batch_id == first.batch_id + 1
        assert trace["wait_for_previous_seconds"] >= 0.055
        assert (
            trace["collect_seconds"]
            - trace["wait_for_previous_seconds"]
            - trace["fetch_seconds"]
            < 0.02
        )
        await rebalancer.fail(first)
        await rebalancer.fail(second)

    run(scenario())


def test_close_during_load_batch_window_publishes_one_failure_trace():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            load_batch_coalescing_window_ms=200,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="close-during-window", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        await asyncio.sleep(0.01)
        await rebalancer.close()

        with pytest.raises(RuntimeError, match="engine rebalancer is closed"):
            await task
        snapshot = await rebalancer.snapshot()
        assert len(snapshot["recent_load_batches"]) == 1
        assert snapshot["recent_load_batches"][0]["batch"]["committed_count"] == 0
        assert snapshot["recent_load_batches"][0]["batch"]["failed_count"] == 1
        assert rebalancer._reservations == {}

    run(scenario())


def test_step_after_fetch_completion_waits_for_previous_batch_commit(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_samples=1,
            load_batch_coalescing_window_ms=0,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        commit_started = asyncio.Event()
        allow_commit = asyncio.Event()
        original_commit = rebalancer._commit_batch

        async def controlled_commit(*args, **kwargs):
            commit_started.set()
            await allow_commit.wait()
            return await original_commit(*args, **kwargs)

        monkeypatch.setattr(rebalancer, "_commit_batch", controlled_commit)
        first_task = asyncio.create_task(
            rebalancer.acquire(session_id="first-batch", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        await commit_started.wait()

        second_task = asyncio.create_task(
            rebalancer.acquire(session_id="second-batch", input_ids=[2] * 10)
        )
        await asyncio.sleep(0)
        assert client.batch_load_calls == {url: 1 for url in client.urls}

        allow_commit.set()
        first = await first_task
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == 2 * len(client.urls)
        )
        client.resolve_batch(1)
        second = await second_task
        traces = (await rebalancer.snapshot())["recent_load_batches"]
        second_trace = next(
            trace for trace in traces if trace["batch"]["id"] == second.batch_id
        )

        assert second.batch_id == first.batch_id + 1
        assert second_trace["batch"]["collect_seconds"] >= (
            second_trace["batch"]["wait_for_previous_seconds"]
            + second_trace["batch"]["fetch_seconds"]
            - 1e-6
        )
        await rebalancer.fail(first)
        await rebalancer.fail(second)

    run(scenario())


def test_duplicate_pending_batch_acquire_fails_explicitly():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        first = asyncio.create_task(
            rebalancer.acquire(session_id="duplicate", input_ids=[1])
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        with pytest.raises(RuntimeError, match="pending acquire"):
            await rebalancer.acquire(session_id="duplicate", input_ids=[2])
        client.resolve_batch(0)
        lease = await first
        await rebalancer.fail(lease)

    run(scenario())


def test_joint_batch_assignment_spreads_equal_new_sessions():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, min_samples=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        tasks = [
            asyncio.create_task(
                rebalancer.acquire(
                    session_id=f"joint-{index}",
                    input_ids=[index] * 10,
                    step_max_new_tokens=10,
                )
            )
            for index in range(4)
        ]
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        leases = await asyncio.gather(*tasks)

        assert sorted(lease.worker_url for lease in leases) == sorted(client.urls * 2)
        assert all(lease.batch_id == leases[0].batch_id for lease in leases)
        for lease in leases:
            assert len(rebalancer._reservations) == 4
            assert rebalancer.sessions[lease.decision.session_id].fingerprint
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert trace["adopted_plan"] in {"sticky", "optimized"}
        for lease in leases:
            await rebalancer.fail(lease)

    run(scenario())


def test_partial_and_all_batch_fetch_failure_exclude_untrusted_engines():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        partial = asyncio.create_task(
            rebalancer.acquire(session_id="partial", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_url(client.urls[0], 0)
        client.fail_url(client.urls[1], 0)
        lease = await partial
        assert lease.worker_url == client.urls[0]
        partial_trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert {engine["fetch_status"] for engine in partial_trace["engines"]} == {
            "ok",
            "error",
        }
        await rebalancer.fail(lease)

        failed = asyncio.create_task(
            rebalancer.acquire(session_id="all-failed", input_ids=[2] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == 2 * len(client.urls)
        )
        for url in client.urls:
            client.fail_url(url, 1)
        with pytest.raises(RuntimeError, match="successful load snapshot"):
            await failed
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert trace["batch"]["failed_count"] == 1

    run(scenario())


def test_batch_malformed_numeric_load_is_invalid_without_failing_other_engine():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        invalid, valid = client.urls
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="malformed-load", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.batch_load_futures[invalid][0].set_result(
            {
                "loads": [
                    {
                        "num_running_reqs": "not-a-number",
                        "max_total_num_tokens": 100_000,
                        "max_running_requests": 100,
                    }
                ]
            }
        )
        client.resolve_url(valid, 0)
        lease = await task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert lease.worker_url == valid
        assert {
            engine["url"]: engine["fetch_status"] for engine in trace["engines"]
        } == {invalid: "invalid", valid: "ok"}
        await rebalancer.fail(lease)

    run(scenario())


def test_batch_commit_revalidation_fails_only_step_with_newly_unhealthy_target(
    monkeypatch,
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["safe-sticky"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        original_solve = rebalancer._solve_frozen_batch

        def make_selected_target_unhealthy(frozen):
            solved = original_solve(frozen)
            assert dict(solved.assignment)["unsafe-new"] == target
            rebalancer.loads[target].healthy = False
            return solved

        monkeypatch.setattr(
            rebalancer,
            "_solve_frozen_batch",
            make_selected_target_unhealthy,
        )
        client.control_batch_loads()
        unsafe = asyncio.create_task(
            rebalancer.acquire(session_id="unsafe-new", input_ids=[2] * 10)
        )
        safe = asyncio.create_task(
            rebalancer.acquire(session_id="safe-sticky", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_url(source, 0, running=80)
        client.resolve_url(target, 0, running=0)
        unsafe_result, safe_result = await asyncio.gather(
            unsafe,
            safe,
            return_exceptions=True,
        )
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert isinstance(unsafe_result, RuntimeError)
        assert isinstance(safe_result, RoutingLease)
        assert safe_result.worker_url == source
        assert trace["batch"]["solved_count"] == 2
        assert trace["batch"]["committed_count"] == 1
        assert trace["batch"]["failed_count"] == 1
        assert all(
            entry.engine_url != target
            for entry in rebalancer._reservations.values()
        )
        await rebalancer.fail(safe_result)

    run(scenario())


def test_batch_commit_revalidation_rejects_fixed_owner_that_became_unhealthy(
    monkeypatch,
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["fixed-became-unhealthy"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        original_solve = rebalancer._solve_frozen_batch

        def make_fixed_owner_unhealthy(frozen):
            solved = original_solve(frozen)
            rebalancer.loads[source].healthy = False
            return solved

        monkeypatch.setattr(
            rebalancer,
            "_solve_frozen_batch",
            make_fixed_owner_unhealthy,
        )
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(
                session_id="fixed-became-unhealthy",
                input_ids=[1] * 10,
                step_max_new_tokens=7,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.fail_url(source, 0)
        client.resolve_url(target, 0)

        with pytest.raises(RuntimeError, match="eligible"):
            await task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        step = trace["steps"][0]
        assert step["source"] == source
        assert step["candidate_urls"] == [source]
        assert step["estimated_output"] == 7
        assert step["status"] == "failed"
        assert rebalancer._reservations == {}

    run(scenario())


@pytest.mark.parametrize(
    "changed_field",
    ["owner", "pending_owner", "fingerprint", "previous_committed"],
)
def test_batch_commit_revalidation_rejects_session_signature_drift_only(
    monkeypatch,
    changed_field,
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["signature-drift"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        original_solve = rebalancer._solve_frozen_batch

        def change_session_during_solve(frozen):
            solved = original_solve(frozen)
            session = rebalancer.sessions["signature-drift"]
            if changed_field == "owner":
                session.owner_worker_url = target
            elif changed_field == "pending_owner":
                session.pending_owner_worker_url = target
            elif changed_field == "fingerprint":
                session.fingerprint = None
            else:
                session.previous_committed_tokens.append(9)
            return solved

        monkeypatch.setattr(
            rebalancer,
            "_solve_frozen_batch",
            change_session_during_solve,
        )
        client.control_batch_loads()
        drifted = asyncio.create_task(
            rebalancer.acquire(
                session_id="signature-drift",
                input_ids=[1] * 10,
            )
        )
        safe = asyncio.create_task(
            rebalancer.acquire(
                session_id="signature-safe",
                input_ids=[2] * 10,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.fail_url(source, 0)
        client.resolve_url(target, 0)
        drifted_result, safe_result = await asyncio.gather(
            drifted,
            safe,
            return_exceptions=True,
        )
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert isinstance(drifted_result, RuntimeError)
        assert isinstance(safe_result, RoutingLease)
        assert safe_result.worker_url == target
        assert trace["batch"]["solved_count"] == 2
        assert trace["batch"]["committed_count"] == 1
        assert trace["batch"]["failed_count"] == 1
        assert set(rebalancer._reservations) == {safe_result.reservation_id}
        await rebalancer.fail(safe_result)

    run(scenario())


def test_batch_commit_revalidation_rejects_lost_mooncake_migration_only(
    monkeypatch,
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_load_improvement_ratio=0.10,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["readiness-lost"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        mooncake_ready = True

        def current_readiness(*args):
            return ContextPathReadiness(
                source,
                target,
                CacheSource.MOONCAKE if mooncake_ready else CacheSource.NONE,
                True,
                True,
            )

        monkeypatch.setattr(
            rebalancer,
            "_candidate_path_readiness",
            current_readiness,
        )
        original_solve = rebalancer._solve_frozen_batch

        def lose_readiness_during_solve(frozen):
            nonlocal mooncake_ready
            solved = original_solve(frozen)
            mooncake_ready = False
            return replace(
                solved,
                assignment=(
                    ("readiness-lost", target),
                    ("readiness-safe", target),
                ),
                adopted="optimized",
            )

        monkeypatch.setattr(
            rebalancer,
            "_solve_frozen_batch",
            lose_readiness_during_solve,
        )
        client.control_batch_loads()
        migration = asyncio.create_task(
            rebalancer.acquire(
                session_id="readiness-lost",
                input_ids=[1] * 10,
            )
        )
        safe = asyncio.create_task(
            rebalancer.acquire(
                session_id="readiness-safe",
                input_ids=[2] * 10,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_url(source, 0, queued=1)
        client.resolve_url(target, 0, running=0)
        migration_result, safe_result = await asyncio.gather(
            migration,
            safe,
            return_exceptions=True,
        )
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert isinstance(migration_result, RuntimeError)
        assert isinstance(safe_result, RoutingLease)
        assert safe_result.worker_url == target
        assert trace["batch"]["solved_count"] == 2
        assert trace["batch"]["committed_count"] == 1
        assert trace["batch"]["failed_count"] == 1
        assert set(rebalancer._reservations) == {safe_result.reservation_id}
        await rebalancer.fail(safe_result)

    run(scenario())


def test_batch_collect_time_stops_at_atomic_seal(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            load_batch_coalescing_window_ms=0,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )
    original_solve = rebalancer._solve_frozen_batch
    original_commit = rebalancer._commit_batch

    def slow_solve(frozen):
        time.sleep(0.05)
        return original_solve(frozen)

    async def slow_commit(*args, **kwargs):
        await asyncio.sleep(0.05)
        return await original_commit(*args, **kwargs)

    monkeypatch.setattr(rebalancer, "_solve_frozen_batch", slow_solve)
    monkeypatch.setattr(rebalancer, "_commit_batch", slow_commit)

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="seal-timing", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        lease = await task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert trace["batch"]["collect_seconds"] >= trace["batch"]["fetch_seconds"]
        assert trace["batch"]["collect_seconds"] < 0.04
        assert trace["batch"]["total_seconds"] >= 0.09
        await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize("terminal", ["error", "cancel"])
def test_batch_success_trace_is_not_republished_by_late_runner_failure(
    monkeypatch,
    terminal,
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        original_commit = rebalancer._commit_batch

        async def late_terminal(*args, **kwargs):
            await original_commit(*args, **kwargs)
            if terminal == "cancel":
                raise asyncio.CancelledError
            raise RuntimeError("after successful commit")

        monkeypatch.setattr(rebalancer, "_commit_batch", late_terminal)
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id=f"late-{terminal}", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        lease = await task
        await wait_for_condition(lambda: not rebalancer._batch_runner_tasks)
        snapshot = await rebalancer.snapshot()

        assert len(snapshot["recent_load_batches"]) == 1
        assert snapshot["recent_load_batches"][0]["batch"]["committed_count"] == 1
        assert snapshot["recent_load_batches"][0]["batch"]["failed_count"] == 0
        assert json.loads(json.dumps(snapshot["recent_load_batches"])) == snapshot[
            "recent_load_batches"
        ]
        assert lease.reservation_id in rebalancer._reservations
        await rebalancer.fail(lease)

    run(scenario())


def test_batch_target_solver_receives_hard_limit_and_lcp_cost(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_load_improvement_ratio=0.20,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )
    captured = {}

    def sticky_solver(problem, deadline_seconds=1.0):
        del deadline_seconds
        return BatchSolution(
            status=SolverStatus.OPTIMAL,
            assignment={
                session_id: edges[0].engine_url
                for session_id, edges in problem.edges_by_session.items()
            },
            maximum_load=1.0,
            minimum_load=0.2,
            load_range=0.8,
            total_migration_cost_tokens=0,
            voluntary_migrations=0,
            elapsed_seconds=0.01,
        )

    def target_solver(
        problem,
        *,
        maximum_load_limit,
        deadline_seconds,
    ):
        del deadline_seconds
        captured["limit"] = maximum_load_limit
        captured["costs"] = {
            edge.engine_url: edge.migration_cost_tokens
            for edge in problem.edges_by_session["target"]
        }
        captured["prefills"] = {
            edge.engine_url: edge.prefill_increment
            for edge in problem.edges_by_session["target"]
        }
        captured["queues"] = {
            edge.engine_url: edge.queue_increment
            for edge in problem.edges_by_session["target"]
        }
        captured["tokens"] = {
            edge.engine_url: edge.token_increment
            for edge in problem.edges_by_session["target"]
        }
        source, target = client.urls
        return BatchSolution(
            status=SolverStatus.OPTIMAL,
            assignment={"target": target},
            maximum_load=0.8,
            minimum_load=0.4,
            load_range=0.4,
            total_migration_cost_tokens=7,
            voluntary_migrations=1,
            elapsed_seconds=0.01,
        )

    import dressage.proxy.rebalancing.scheduler as scheduler_module

    monkeypatch.setattr(
        scheduler_module,
        "solve_batch_milp",
        sticky_solver,
    )
    monkeypatch.setattr(
        scheduler_module,
        "solve_batch_for_target_load",
        target_solver,
        raising=False,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        for url in client.urls:
            rebalancer.deployments[url] = replace(
                rebalancer.deployments[url],
                page_size=4,
            )
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["target"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 7,
            seen_engines={source},
        )
        monkeypatch.setattr(
            rebalancer,
            "_candidate_path_readiness",
            lambda *args: ContextPathReadiness(
                source,
                target,
                CacheSource.MOONCAKE,
                True,
                True,
            ),
        )
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(
                session_id="target",
                input_ids=[1] * 7 + [2] * 3,
                step_max_new_tokens=10_000,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_url(source, 0, gen_throughput=240.0)
        client.resolve_url(target, 0, gen_throughput=120.0)
        lease = await task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert lease.worker_url == target
        assert captured == {
            "limit": pytest.approx(0.8),
            "costs": {source: 0, target: 7},
            "prefills": {source: 3, target: 6},
            "queues": {source: 1, target: 1},
            "tokens": {source: 3, target: 10},
        }
        assert lease.reserved_prefill_tokens == 6
        assert rebalancer.loads[target].reserved_prefill_tokens == 6
        assert trace["adopted_plan"] == "optimized"
        assert trace["improvement_ratio"] == pytest.approx(0.2)
        assert trace["target_maximum_load"] == pytest.approx(0.8)
        assert trace["sticky"] == {
            "status": "optimal",
            "maximum_load": 1.0,
            "minimum_load": 0.2,
            "load_range": 0.8,
            "migration_cost_tokens": 0,
            "migrations": 0,
            "elapsed_seconds": 0.01,
        }
        assert trace["optimized"] == {
            "status": "optimal",
            "maximum_load": 0.8,
            "minimum_load": 0.4,
            "load_range": 0.4,
            "migration_cost_tokens": 7,
            "migrations": 1,
            "elapsed_seconds": 0.01,
        }
        assert trace["steps"][0]["candidate_migration_cost_tokens"] == {
            source: 0,
            target: 7,
        }
        assert trace["steps"][0]["migration_cost_tokens"] == 7
        assert trace["steps"][0]["queue_increment"] == 1
        assert trace["steps"][0]["token_increment"] == 10
        assert trace["steps"][0]["reserved_requests"] == 1
        assert trace["steps"][0]["reserved_tokens"] == 10_010
        assert trace["steps"][0]["reserved_prefill_tokens"] == 6
        assert lease.decision.target_projected_load is not None
        assert set(lease.decision.target_projected_load.__dict__) == {
            "request_pressure",
            "token_pressure",
            "queue_pressure",
            "total",
        }
        assert lease.reserved_tokens == 10_010
        assert rebalancer._reservations[lease.reservation_id].token_increment == 10_010
        engines = {engine["url"]: engine for engine in trace["engines"]}
        assert engines[source]["gen_throughput"] == pytest.approx(240.0)
        assert engines[target]["gen_throughput"] == pytest.approx(120.0)
        assert "decode_rate_ratio" not in engines[source]
        await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize(
    ("rates", "expected_trace_rates"),
    [
        ((0.0, 0.0), (0.0, 0.0)),
        ((0.0, 240.0), (0.0, 240.0)),
        ((-1.0, 240.0), (-1.0, 240.0)),
        ((math.nan, 240.0), (None, 240.0)),
    ],
)
def test_batch_gen_throughput_is_diagnostic_only(
    rates,
    expected_trace_rates,
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(
                session_id="throughput-diagnostic",
                input_ids=[1] * 10,
                step_max_new_tokens=10_000,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        for url, rate in zip(client.urls, rates):
            client.resolve_url(url, 0, gen_throughput=rate)
        lease = await task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert "candidate_decode_pressure" not in trace["steps"][0]
        assert "decode_pressure_increment" not in trace["steps"][0]
        engines = {engine["url"]: engine for engine in trace["engines"]}
        assert all("decode_rate_ratio" not in engine for engine in engines.values())
        assert tuple(engines[url]["gen_throughput"] for url in client.urls) == (
            expected_trace_rates
        )
        json.dumps(trace, allow_nan=False)
        assert math.isfinite(lease.decision.target_projected_load.total)
        await rebalancer.fail(lease)

    run(scenario())


def test_estimated_output_does_not_change_batch_token_increment():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        tasks = [
            asyncio.create_task(
                rebalancer.acquire(
                    session_id="output-short",
                    input_ids=[1] * 10,
                    step_max_new_tokens=5_000,
                )
            ),
            asyncio.create_task(
                rebalancer.acquire(
                    session_id="output-long",
                    input_ids=[1] * 10,
                    step_max_new_tokens=10_000,
                )
            ),
            asyncio.create_task(
                rebalancer.acquire(
                    session_id="output-unknown",
                    input_ids=[1] * 10,
                )
            ),
        ]
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0, gen_throughput=200.0)
        leases = await asyncio.gather(*tasks)
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        steps = {step["session_id"]: step for step in trace["steps"]}

        assert steps["output-short"]["token_increment"] == 10
        assert steps["output-long"]["token_increment"] == 10
        assert steps["output-unknown"]["token_increment"] == 10
        assert steps["output-short"]["reserved_tokens"] == 5_010
        assert steps["output-long"]["reserved_tokens"] == 10_010
        assert steps["output-unknown"]["reserved_tokens"] == 10
        for lease in leases:
            await rebalancer.fail(lease)

    run(scenario())


def test_batch_skips_target_solver_when_baseline_already_exceeds_target(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_load_improvement_ratio=0.10,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )
    target_solver_called = False

    def unexpected_target_solver(*args, **kwargs):
        nonlocal target_solver_called
        target_solver_called = True
        raise RuntimeError("target solver should have been skipped")

    monkeypatch.setattr(
        "dressage.proxy.rebalancing.scheduler.solve_batch_for_target_load",
        unexpected_target_solver,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["baseline-bound"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        monkeypatch.setattr(
            rebalancer,
            "_candidate_path_readiness",
            lambda *args: ContextPathReadiness(
                source,
                target,
                CacheSource.MOONCAKE,
                True,
                True,
            ),
        )
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="baseline-bound", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_url(source, 0, running=100)
        client.resolve_url(target, 0, running=0)
        lease = await task
        try:
            trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
            assert target_solver_called is False
            assert trace["adopted_plan"] == "sticky"
            assert trace["fallback_reason"] == "target_load_infeasible"
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_batch_reservation_reuses_final_batch_projected_load(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    def unexpected_recalculation(*args, **kwargs):
        raise RuntimeError("batch projected load must not be recalculated")

    monkeypatch.setattr(
        rebalancer,
        "_projected_load_score",
        unexpected_recalculation,
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="projected-reuse", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        lease = await task
        try:
            assert lease.decision.target_projected_load is not None
            assert lease.projected_load_score == pytest.approx(
                lease.decision.target_projected_load.total
            )
        finally:
            await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize(
    ("mooncake_ready", "expected_worker", "expected_reason"),
    [
        (False, "source", "batch_sticky"),
        (True, "target", "batch_optimized_migration"),
    ],
)
def test_batch_healthy_owner_only_voluntarily_migrates_over_mooncake(
    monkeypatch, mooncake_ready, expected_worker, expected_reason
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_load_improvement_ratio=0.10,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["candidate-owner"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        monkeypatch.setattr(
            rebalancer,
            "_candidate_path_readiness",
            lambda *args: ContextPathReadiness(
                source,
                target,
                CacheSource.MOONCAKE if mooncake_ready else CacheSource.NONE,
                True,
                True,
            ),
        )
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="candidate-owner", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_url(source, 0, queued=1)
        client.resolve_url(target, 0, running=0)
        lease = await task

        assert lease.worker_url == {"source": source, "target": target}[
            expected_worker
        ]
        assert lease.decision.reason == expected_reason
        await rebalancer.fail(lease)

    run(scenario())


def test_batch_zero_lcp_migration_has_zero_kv_cost(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_load_improvement_ratio=0.10,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["zero-lcp"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[9] * 10,
            seen_engines={source},
        )
        monkeypatch.setattr(
            rebalancer,
            "_candidate_path_readiness",
            lambda *args: ContextPathReadiness(
                source,
                target,
                CacheSource.MOONCAKE,
                True,
                True,
            ),
        )
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="zero-lcp", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_url(source, 0, queued=1)
        client.resolve_url(target, 0, running=0)
        lease = await task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert lease.worker_url == target
        assert trace["steps"][0]["candidate_migration_cost_tokens"] == {
            source: 0,
            target: 0,
        }
        assert trace["steps"][0]["migration_cost_tokens"] == 0
        assert trace["steps"][0]["token_increment"] == 10
        assert trace["steps"][0]["reserved_prefill_tokens"] == 10
        assert trace["optimized"]["migration_cost_tokens"] == 0
        assert lease.reserved_prefill_tokens == 10
        await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize("owner_invalid_by", ["health", "version"])
def test_batch_owner_unhealthy_or_version_invalid_uses_mandatory_failover(
    owner_invalid_by,
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["candidate-failover"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        expected_version = None
        expected_fetches = 2
        if owner_invalid_by == "health":
            rebalancer.loads[source].healthy = False
            expected_fetches = 1
        else:
            rebalancer.deployments[target] = replace(
                rebalancer.deployments[target],
                weight_version="8",
            )
            expected_version = "8"
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(
                session_id="candidate-failover",
                input_ids=[1] * 10,
                expected_version=expected_version,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == expected_fetches
        )
        for url in client.urls:
            if client.batch_load_calls[url]:
                client.resolve_url(url, 0)
        lease = await task

        assert lease.worker_url == target
        assert lease.decision.reason == "batch_owner_failover"
        assert lease.reserved_prefill_tokens == 10
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert trace["steps"][0]["token_increment"] == 10
        assert trace["steps"][0]["reserved_prefill_tokens"] == 10
        await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize("mismatch", ["version", "fingerprint"])
def test_batch_mandatory_failover_rejects_version_or_fingerprint_mismatch(mismatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["candidate-mismatch"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        rebalancer.loads[source].healthy = False
        expected_version = "8" if mismatch == "version" else None
        if mismatch == "fingerprint":
            rebalancer.deployments[target] = replace(
                rebalancer.deployments[target],
                cache_fingerprint="different-fingerprint",
            )
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(
                session_id="candidate-mismatch",
                input_ids=[1] * 10,
                expected_version=expected_version,
            )
        )
        await wait_for_condition(
            lambda: client.batch_load_calls[target] == 1
        )
        client.resolve_url(target, 0)

        with pytest.raises(RuntimeError, match="successful load snapshot"):
            await task

    run(scenario())


def test_batch_failed_healthy_owner_snapshot_fixes_step_to_owner():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["candidate-fixed"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="candidate-fixed", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.fail_url(source, 0)
        client.resolve_url(target, 0)
        lease = await task

        assert lease.worker_url == source
        assert lease.decision.reason == "batch_fixed_owner"
        assert lease.decision.target_projected_load is not None
        assert lease.decision.target_projected_load.request_pressure == 0
        assert lease.decision.target_projected_load.token_pressure == pytest.approx(
            0.0
        )
        assert lease.decision.target_projected_load.queue_pressure == pytest.approx(
            0.01
        )
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert trace["steps"][0]["queue_increment"] == 1
        assert trace["steps"][0]["token_increment"] == 0
        await rebalancer.fail(lease)

    run(scenario())


def test_batch_sticky_solver_exception_uses_stable_greedy_assignment(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    def fail_solver(*args, **kwargs):
        raise RuntimeError("sticky solver boom")

    monkeypatch.setattr(
        "dressage.proxy.rebalancing.scheduler.solve_batch_milp",
        fail_solver,
    )

    async def acquire_batch(index):
        tasks = [
            asyncio.create_task(
                rebalancer.acquire(
                    session_id=f"greedy-{session_index}",
                    input_ids=[session_index] * 10,
                )
            )
            for session_index in range(4)
        ]
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values())
            == (index + 1) * len(client.urls)
        )
        client.resolve_batch(index)
        leases = await asyncio.gather(*tasks)
        assignment = {
            lease.decision.session_id: lease.worker_url for lease in leases
        }
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert trace["adopted_plan"] == "sticky_greedy"
        assert trace["fallback_reason"] == "sticky_solver_failure"
        assert trace["sticky"]["status"] == "greedy"
        assert sorted(assignment.values()) == sorted(client.urls * 2)
        for lease in leases:
            await rebalancer.fail(lease)
        return assignment

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        first = await acquire_batch(0)
        second = await acquire_batch(1)
        assert second == first

    run(scenario())


@pytest.mark.parametrize(
    ("failure_kind", "expected_reason"),
    [
        ("runtime", "target_solver_failure"),
        (2, "target_load_infeasible"),
        (1, "target_solver_deadline"),
        (None, "target_solver_deadline"),
    ],
)
def test_batch_target_solver_failure_adopts_sticky_assignment(
    monkeypatch,
    failure_kind,
    expected_reason,
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_load_improvement_ratio=0.0,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )
    def fail_target(*args, **kwargs):
        if failure_kind == "runtime":
            raise RuntimeError("target solver boom")
        raise BatchSolverError(
            phase=1,
            status=failure_kind,
            elapsed_seconds=0.01,
            message="target solver boom",
        )

    monkeypatch.setattr(
        "dressage.proxy.rebalancing.scheduler.solve_batch_for_target_load",
        fail_target,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["optimized-failure"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        monkeypatch.setattr(
            rebalancer,
            "_candidate_path_readiness",
            lambda *args: ContextPathReadiness(
                source,
                target,
                CacheSource.MOONCAKE,
                True,
                True,
            ),
        )
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="optimized-failure", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_url(source, 0, running=80)
        client.resolve_url(target, 0, running=0)
        lease = await task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert lease.worker_url == source
        assert lease.decision.reason == "batch_sticky"
        assert trace["adopted_plan"] == "sticky"
        assert trace["fallback_reason"] == expected_reason
        assert trace["sticky"]["status"] == "optimal"
        assert trace["optimized"] is None
        await rebalancer.fail(lease)

    run(scenario())


def test_batch_shared_deadline_expires_before_target_solver(monkeypatch):
    import dressage.proxy.rebalancing.scheduler as scheduler_module
    from dressage.proxy.rebalancing._batch_milp import (
        BatchProblem,
        EngineBaseline,
        FeasibleEdge,
    )

    engine = EngineBaseline("source", 0, 0, 0, 100, 100_000, 0)
    owner = FeasibleEdge("session", "source", 1, 1, 0, False)
    migration = FeasibleEdge(
        "session",
        "target",
        1,
        1,
        1,
        True,
        migration_cost_tokens=1,
    )
    target_engine = EngineBaseline("target", 0, 0, 0, 100, 100_000, 0)
    sticky_problem = BatchProblem((engine, target_engine), {"session": (owner,)})
    optimized_problem = BatchProblem(
        (engine, target_engine),
        {"session": (owner, migration)},
    )
    sticky = BatchSolution(
        status=SolverStatus.OPTIMAL,
        assignment={"session": "source"},
        maximum_load=1.0,
        minimum_load=0.0,
        load_range=1.0,
        total_migration_cost_tokens=0,
        voluntary_migrations=0,
        elapsed_seconds=0.01,
    )
    times = iter((0.0, 0.0, 2.0, 2.0))
    monkeypatch.setattr(
        scheduler_module,
        "time",
        type(
            "FakeTime",
            (),
            {
                "monotonic": staticmethod(lambda: next(times)),
                "time": staticmethod(time.time),
            },
        ),
    )
    monkeypatch.setattr(
        scheduler_module,
        "solve_batch_milp",
        lambda *args, **kwargs: sticky,
    )
    rebalancer = EngineRebalancer(
        ControlledBatchLoadClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    frozen = scheduler_module._FrozenBatch(
        reservation_revision=0,
        topology_signature=(),
        decision_engines=(),
        engine_traces=(),
        steps=(),
        sticky_problem=sticky_problem,
        optimized_problem=optimized_problem,
    )

    solved = rebalancer._solve_frozen_batch(frozen)

    assert solved.adopted == "sticky"
    assert solved.fallback_reason == "target_solver_deadline"


def test_batch_frozen_revision_change_uses_sticky_greedy(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_load_improvement_ratio=0.0,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["frozen-change"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 10,
            seen_engines={source},
        )
        monkeypatch.setattr(
            rebalancer,
            "_candidate_path_readiness",
            lambda *args: ContextPathReadiness(
                source,
                target,
                CacheSource.MOONCAKE,
                True,
                True,
            ),
        )
        anchor = rebalancer._reserve(
            RoutingDecision(
                session_id="revision-anchor",
                source_worker_url=None,
                target_worker_url=source,
                cache_fingerprint=fingerprint,
                state=SchedulerState.ACTIVE,
                reason="test",
            ),
            input_ids=[9],
            base_tokens=0,
            budget=StepGenerationBudget("test", 1, None, None, 1),
        )
        original_commit = rebalancer._commit_batch

        async def change_revision_before_commit(*args, **kwargs):
            await rebalancer.fail(anchor)
            return await original_commit(*args, **kwargs)

        monkeypatch.setattr(
            rebalancer,
            "_commit_batch",
            change_revision_before_commit,
        )
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(session_id="frozen-change", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_url(source, 0, running=80)
        client.resolve_url(target, 0, running=0)
        lease = await task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert lease.worker_url == source
        assert lease.decision.reason == "batch_sticky"
        assert trace["adopted_plan"] == "sticky_greedy"
        assert trace["fallback_reason"] == "frozen_state_changed"
        assert trace["sticky"]["status"] == "greedy"
        await rebalancer.fail(lease)

    run(scenario())


def test_batch_cancel_before_seal_is_excluded_from_solver_and_ledger(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    from dressage.proxy.rebalancing.scheduler import solve_batch_milp as real_solver

    solved_sessions = []

    def capture_problem(problem, deadline_seconds=1.0):
        solved_sessions.append(tuple(sorted(problem.edges_by_session)))
        return real_solver(problem, deadline_seconds=deadline_seconds)

    monkeypatch.setattr(
        "dressage.proxy.rebalancing.scheduler.solve_batch_milp",
        capture_problem,
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        cancelled = asyncio.create_task(
            rebalancer.acquire(session_id="cancel-before-seal", input_ids=[1] * 10)
        )
        kept = asyncio.create_task(
            rebalancer.acquire(session_id="kept-before-seal", input_ids=[2] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        client.resolve_batch(0)
        lease = await kept
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]

        assert solved_sessions == [("kept-before-seal",)]
        assert "cancel-before-seal" not in rebalancer.sessions
        assert len(rebalancer._reservations) == 1
        assert trace["batch"]["cancelled_count"] == 1
        assert [step["status"] for step in trace["steps"]] == [
            "cancelled",
            "committed",
        ]
        assert trace["steps"][0]["candidate_migration_cost_tokens"] == {}
        assert trace["steps"][0]["migration_cost_tokens"] == 0
        await rebalancer.fail(lease)

    run(scenario())


def test_batch_cancel_after_commit_releases_lease_and_pending_owner(monkeypatch):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        original_commit = rebalancer._commit_batch
        acquire_task = None
        committed_lease = None

        async def cancel_after_commit(*args, **kwargs):
            nonlocal committed_lease
            await original_commit(*args, **kwargs)
            pending = rebalancer._pending_acquires["cancel-after-commit"]
            committed_lease = pending.lease
            acquire_task.cancel()

        monkeypatch.setattr(rebalancer, "_commit_batch", cancel_after_commit)
        client.control_batch_loads()
        acquire_task = asyncio.create_task(
            rebalancer.acquire(session_id="cancel-after-commit", input_ids=[1] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        with pytest.raises(asyncio.CancelledError):
            await acquire_task

        assert committed_lease is not None
        assert committed_lease.reservation_id not in rebalancer._reservations
        state = rebalancer.sessions["cancel-after-commit"]
        assert state.pending_owner_worker_url is None
        assert all(load.reserved_requests == 0 for load in rebalancer.loads.values())
        revision = rebalancer._reservation_revision
        await rebalancer.fail(committed_lease)
        assert rebalancer._reservation_revision == revision

    run(scenario())


def test_batch_commit_failure_rolls_back_all_state_and_publishes_one_failure_trace(
    monkeypatch,
):
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()

        class NoDeepcopyTokens(list):
            def __deepcopy__(self, memo):
                del memo
                raise AssertionError("unrelated session must not be copied")

        unrelated = SessionRoutingState(
            previous_committed_tokens=NoDeepcopyTokens([9] * 10_000),
        )
        existing = SessionRoutingState(task_key="existing")
        rebalancer.sessions["unrelated"] = unrelated
        rebalancer.sessions["rollback-0"] = existing
        initial_decisions = list(rebalancer._decisions)
        initial_next_reservation_id = rebalancer._next_reservation_id
        initial_revision = rebalancer._reservation_revision
        original_reserve = rebalancer._reserve
        reserve_calls = 0

        def fail_second_reserve(*args, **kwargs):
            nonlocal reserve_calls
            reserve_calls += 1
            if reserve_calls == 2:
                raise RuntimeError("commit injection")
            return original_reserve(*args, **kwargs)

        monkeypatch.setattr(rebalancer, "_reserve", fail_second_reserve)
        client.control_batch_loads()
        tasks = [
            asyncio.create_task(
                rebalancer.acquire(
                    session_id=f"rollback-{index}",
                    input_ids=[index] * 10,
                )
            )
            for index in range(2)
        ]
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        trace = (await rebalancer.snapshot())["recent_load_batches"]

        assert all(
            isinstance(result, RuntimeError) and "commit injection" in str(result)
            for result in results
        )
        assert rebalancer.sessions["unrelated"] is unrelated
        restored = rebalancer.sessions["rollback-0"]
        assert restored.owner_worker_url is None
        assert restored.pending_owner_worker_url is None
        assert restored.fingerprint is None
        assert restored.task_key == "existing"
        assert set(rebalancer.sessions) == {"unrelated", "rollback-0"}
        assert rebalancer._reservations == {}
        assert rebalancer._pending_acquires == {}
        assert rebalancer._next_reservation_id == initial_next_reservation_id
        assert rebalancer._reservation_revision == initial_revision
        assert list(rebalancer._decisions) == initial_decisions
        assert all(
            (
                load.reserved_requests,
                load.reserved_tokens,
                load.reserved_prefill_tokens,
            )
            == (0, 0, 0)
            for load in rebalancer.loads.values()
        )
        assert len(trace) == 1
        assert trace[0]["batch"]["committed_count"] == 0
        assert trace[0]["batch"]["failed_count"] == 2
        assert trace[0]["fallback_reason"] == "runner_failure"
        assert all(engine["fetch_status"] == "ok" for engine in trace[0]["engines"])
        assert all(engine["request_capacity"] == 100 for engine in trace[0]["engines"])
        assert all(engine["base_requests"] == 0 for engine in trace[0]["engines"])

    run(scenario())


def test_batch_future_wakes_after_every_reservation_and_session_is_visible():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    session_ids = [f"visible-{index}" for index in range(4)]

    async def acquire_and_observe(session_id, token):
        lease = await rebalancer.acquire(session_id=session_id, input_ids=[token] * 10)
        return (
            lease,
            len(rebalancer._reservations),
            all(
                rebalancer.sessions[item].fingerprint is not None
                for item in session_ids
            ),
        )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        tasks = [
            asyncio.create_task(acquire_and_observe(session_id, index))
            for index, session_id in enumerate(session_ids)
        ]
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        observed = await asyncio.gather(*tasks)

        assert [(count, all_visible) for _, count, all_visible in observed] == [
            (4, True),
            (4, True),
            (4, True),
            (4, True),
        ]
        for lease, _, _ in observed:
            await rebalancer.fail(lease)

    run(scenario())


def test_next_batch_trace_effective_base_includes_previous_live_ledger():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        first_task = asyncio.create_task(
            rebalancer.acquire(
                session_id="baseline-first",
                input_ids=[1] * 40,
                step_max_new_tokens=10,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0)
        first = await first_task
        live_requests, live_tokens, live_prefill = (
            rebalancer._live_reservation_totals(first.worker_url)
        )
        assert (live_requests, live_tokens, live_prefill) == (1, 50, 40)

        second_task = asyncio.create_task(
            rebalancer.acquire(
                session_id="baseline-second",
                input_ids=[2] * 10,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == 2 * len(client.urls)
        )
        client.resolve_batch(1)
        second = await second_task
        trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        engine = next(
            item for item in trace["engines"] if item["url"] == first.worker_url
        )

        assert engine["live_ledger_requests"] == 1
        assert engine["live_ledger_tokens"] == 50
        assert engine["live_ledger_prefill"] == 40
        assert engine["base_requests"] == 0
        assert engine["base_tokens"] == 0
        assert engine["base_queue"] == 1
        assert engine["queue_pressure"] == pytest.approx(0.01)
        assert second.worker_url != first.worker_url
        await rebalancer.fail(first)
        await rebalancer.fail(second)

        third_task = asyncio.create_task(
            rebalancer.acquire(
                session_id="baseline-third",
                input_ids=[3] * 10,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == 3 * len(client.urls)
        )
        client.resolve_batch(2)
        third = await third_task
        released_trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert all(
            item["live_ledger_requests"] == 0
            and item["base_queue"] == 0
            for item in released_trace["engines"]
        )
        await rebalancer.fail(third)

    run(scenario())


def test_next_batch_trace_effective_base_retires_observed_prefill_generation():
    client = ControlledBatchLoadClient()
    client.urls = client.urls[:1]
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def acquire_batch(index, session_id, prompt_tokens):
        task = asyncio.create_task(
            rebalancer.acquire(
                session_id=session_id,
                input_ids=[index + 1] * prompt_tokens,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == index + 1
        )
        client.resolve_batch(index, waiting_uncached=0)
        return await task

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        first = await acquire_batch(0, "prefill-first", 40)
        second = await acquire_batch(1, "prefill-second", 10)
        second_trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert second_trace["engines"][0]["live_ledger_prefill"] == 40

        await rebalancer.fail(second)
        third = await acquire_batch(2, "prefill-third", 5)
        third_trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert third_trace["engines"][0]["live_ledger_prefill"] == 0

        await rebalancer.fail(first)
        await rebalancer.fail(third)

    run(scenario())


def test_batch_trace_is_normalized_complete_immutable_and_contains_no_token_content():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    secret_tokens = [918273, 817263, 716253]

    async def scenario():
        await rebalancer.refresh()
        old_snapshot = await rebalancer.snapshot()
        client.control_batch_loads()
        task = asyncio.create_task(
            rebalancer.acquire(
                session_id="trace-normalized",
                input_ids=secret_tokens,
                step_max_new_tokens=7,
            )
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values()) == len(client.urls)
        )
        client.resolve_batch(0, running=3)
        lease = await task
        snapshot = await rebalancer.snapshot()
        trace = snapshot["recent_load_batches"][-1]
        encoded = json.dumps(trace, sort_keys=True)

        assert set(old_snapshot).issubset(snapshot)
        assert set(trace["batch"]) == {
            "id",
            "completed_at",
            "registered_count",
            "solved_count",
            "committed_count",
            "failed_count",
            "cancelled_count",
            "wait_for_previous_seconds",
            "collect_seconds",
            "fetch_seconds",
            "solve_seconds",
            "total_seconds",
        }
        assert {
            "arrival_id",
            "session_id",
            "source",
            "prompt_token_count",
            "estimated_output",
            "candidate_urls",
            "status",
            "target",
            "moved",
            "queue_increment",
            "token_increment",
            "reserved_requests",
            "reserved_tokens",
            "reserved_prefill_tokens",
        }.issubset(trace["steps"][0])
        for engine in trace["engines"]:
            assert {
                "url",
                "fetch_status",
                "fetch_duration_seconds",
                "health",
                "version",
                "fingerprint",
                "row_count",
                "running",
                "active_tokens",
                "request_capacity",
                "token_capacity",
                "token_usage",
                "gen_throughput",
                "queued",
                "queue_pressure",
                "waiting_uncached_tokens",
                "live_ledger_requests",
                "live_ledger_tokens",
                "live_ledger_prefill",
                "base_requests",
                "base_tokens",
                "base_queue",
            }.issubset(engine)
        assert trace["sticky"]["status"] in {"optimal", "greedy"}
        assert "maximum_load" in trace["sticky"]
        assert "input_ids" not in encoded
        assert "server_args" not in encoded
        assert "loads" not in encoded
        assert all(str(token) not in encoded for token in secret_tokens)

        trace["batch"]["id"] = -1
        trace["steps"][0]["candidate_urls"].append("mutated")
        fresh = (await rebalancer.snapshot())["recent_load_batches"][-1]
        assert fresh["batch"]["id"] == lease.batch_id
        assert "mutated" not in fresh["steps"][0]["candidate_urls"]
        await rebalancer.fail(lease)

    run(scenario())


def test_batch_trace_history_records_each_terminal_batch_once_and_evicts_oldest():
    client = ControlledBatchLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, history_size=2),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def run_batch(index, session_id, outcome):
        task = asyncio.create_task(
            rebalancer.acquire(session_id=session_id, input_ids=[index] * 10)
        )
        await wait_for_condition(
            lambda: sum(client.batch_load_calls.values())
            == (index + 1) * len(client.urls)
        )
        if outcome == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            client.resolve_batch(index)
            await wait_for_condition(
                lambda: len(rebalancer._load_batch_history.snapshot()) == index + 1
            )
        elif outcome == "failed":
            for url in client.urls:
                client.fail_url(url, index)
            with pytest.raises(RuntimeError, match="successful load snapshot"):
                await task
        else:
            client.resolve_batch(index)
            return await task
        return None

    async def scenario():
        await rebalancer.refresh()
        client.control_batch_loads()
        cancelled = await run_batch(0, "history-cancelled", "cancelled")
        assert cancelled is None
        assert len(rebalancer._load_batch_history.snapshot()) == 1
        failed = await run_batch(1, "history-failed", "failed")
        assert failed is None
        assert len(rebalancer._load_batch_history.snapshot()) == 2
        success = await run_batch(2, "history-success", "success")
        snapshot = await rebalancer.snapshot()

        assert [trace["batch"]["id"] for trace in snapshot["recent_load_batches"]] == [
            2,
            3,
        ]
        assert [
            trace["batch"]["registered_count"]
            for trace in snapshot["recent_load_batches"]
        ] == [1, 1]
        assert json.loads(json.dumps(snapshot["recent_load_batches"])) == snapshot[
            "recent_load_batches"
        ]
        await rebalancer.fail(success)

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
        "expected_required_ratio",
    ),
    [
        (
            100,
            61,
            0,
            0,
            0,
            0,
            "source",
            "no_backlog_load_improvement_below_threshold",
            0.40,
        ),
        (
            100,
            60,
            0,
            0,
            0,
            0,
            "target",
            "no_backlog_load_improvement_threshold_met",
            0.40,
        ),
        (
            2,
            1,
            0,
            0,
            0,
            0,
            "target",
            "no_backlog_load_improvement_threshold_met",
            0.40,
        ),
        (
            2,
            0,
            1,
            1,
            0,
            0,
            "target",
            "no_backlog_load_improvement_threshold_met",
            0.40,
        ),
        (2, 1, 1, 0, 0, 0, "target", "load_improvement_threshold_met", 0.20),
        (
            2,
            1,
            0,
            0,
            1_000,
            0,
            "target",
            "load_improvement_threshold_met",
            0.20,
        ),
    ],
)
def test_existing_session_uses_two_level_backlog_threshold(
    source_running,
    target_running,
    source_queued,
    target_queued,
    source_prefill,
    target_prefill,
    expected_worker,
    expected_reason,
    expected_required_ratio,
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
        config=EngineRebalancingConfig(enabled=True),
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
        serve_current_loads_for_batch(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["backlog"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source},
        )

        lease = await rebalancer.acquire(
            session_id="backlog",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url in {source, target}
            assert lease.decision.reason in {
                "batch_sticky",
                "batch_optimized_migration",
            }
            assert lease.decision.required_load_improvement_ratio == 0.10
            trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
            assert trace["adopted_plan"] == (
                "optimized" if lease.decision.moved else "sticky"
            )
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
        config=EngineRebalancingConfig(enabled=True),
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
        serve_current_loads_for_batch(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["backlog-target"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source},
        )

        lease = await rebalancer.acquire(
            session_id="backlog-target",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url in client.urls
            assert lease.decision.reason in {
                "batch_sticky",
                "batch_optimized_migration",
            }
            assert lease.decision.target_base_load is not None
            assert lease.decision.required_load_improvement_ratio == 0.10
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_existing_session_selects_lowest_load_target_without_backlog_advantage():
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
        source, lowest_load, higher_load = client.urls
        rebalancer.loads[source].running = 100
        rebalancer.loads[lowest_load].running = 20
        rebalancer.loads[higher_load].running = 50
        serve_current_loads_for_batch(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["no-backlog-target"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source},
        )

        lease = await rebalancer.acquire(
            session_id="no-backlog-target",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url in client.urls
            assert lease.decision.reason in {
                "batch_sticky",
                "batch_optimized_migration",
            }
            assert lease.decision.required_load_improvement_ratio == 0.10
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
        serve_current_loads_for_batch(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["threshold"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source, target},
        )

        lease = await rebalancer.acquire(
            session_id="threshold",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url in {source, target}
            assert lease.decision.reason in {
                "batch_sticky",
                "batch_optimized_migration",
            }
            assert lease.decision.source_base_load is not None
            assert lease.decision.target_base_load is not None
            assert lease.decision.required_load_improvement_ratio == minimum_ratio
            assert lease.decision.source_context is None
            assert lease.decision.target_context is None
        finally:
            await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize("target_running", [50, 40])
def test_batch_gate_uses_configured_ratio_across_target_loads(target_running):
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
        serve_current_loads_for_batch(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["return"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source, target},
        )

        lease = await rebalancer.acquire(
            session_id="return",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url in {source, target}
            assert lease.decision.reason in {
                "batch_sticky",
                "batch_optimized_migration",
            }
            assert lease.decision.required_load_improvement_ratio == 0.30
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_batch_gate_uses_configured_ratio_without_previous_owner_state():
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
            min_load_improvement_ratio=0.20,
        ),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, previous_owner = client.urls
        rebalancer.loads[source].running = 100
        rebalancer.loads[previous_owner].running = 50
        serve_current_loads_for_batch(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["no-backlog-return"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={source, previous_owner},
        )

        lease = await rebalancer.acquire(
            session_id="no-backlog-return",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url in {source, previous_owner}
            assert lease.decision.reason in {
                "batch_sticky",
                "batch_optimized_migration",
            }
            assert lease.decision.required_load_improvement_ratio == 0.20
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_consecutive_batch_steps_keep_session_state_consistent():
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
        serve_current_loads_for_batch(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["multi-hop"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
            seen_engines={previous_owner, source},
        )

        sticky = await rebalancer.acquire(
            session_id="multi-hop",
            input_ids=[1] * 100,
        )
        assert sticky.worker_url in client.urls
        assert sticky.decision.reason in {
            "batch_sticky",
            "batch_optimized_migration",
        }
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
        state = rebalancer.sessions["multi-hop"]
        assert state.owner_worker_url == sticky.worker_url
        assert state.pending_owner_worker_url is None
        assert state.previous_committed_tokens == [1] * 100

        movable = await rebalancer.acquire(
            session_id="multi-hop",
            input_ids=[1] * 100,
        )
        try:
            assert movable.worker_url in client.urls
            assert movable.decision.reason in {
                "batch_sticky",
                "batch_optimized_migration",
            }
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
            assert lease.decision.reason == "batch_sticky"
            assert lease.decision.target_base_load is not None
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
        config=EngineRebalancingConfig(enabled=True),
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
        rebalancer.loads[target].running = 6
        rebalancer.loads[target].request_capacity = 10
        rebalancer.loads[target].token_capacity = 100
        serve_current_loads_for_batch(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["projected-safety"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 150,
            seen_engines={source},
        )

        lease = await rebalancer.acquire(
            session_id="projected-safety",
            input_ids=[1] * 150,
        )
        try:
            assert lease.worker_url == source
            assert lease.decision.reason == "batch_sticky"
            assert lease.decision.required_load_improvement_ratio == 0.10
            assert lease.decision.source_projected_load is not None
            assert lease.decision.target_projected_load is not None
            trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
            assert trace["adopted_plan"] == "sticky"
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
            assert lease.decision.reason == "batch_new_session"
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
            assert lease.worker_url in client.urls
            assert lease.decision.reason == "batch_new_session"
            assert lease.decision.target_context is None
            assert lease.decision.target_projected_load is not None
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
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )
    load = rebalancer._normalize_load(
        "worker",
        {
            "loads": [
                {"num_waiting_reqs": 2},
                {"num_queue_reqs": 3},
            ]
        },
        now=1.0,
    )
    assert load is not None
    assert load.queued == 5
    assert load.waiting_uncached_tokens == 0
    assert load.gen_throughput == 0.0
    assert load.live_queue_metrics_available is False


def test_load_snapshot_aggregates_live_queue_fields_across_dp_ranks():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )
    load = rebalancer._normalize_load(
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

    assert load is not None
    assert load.queued == 7
    assert load.waiting_uncached_tokens == 8_000
    assert load.gen_throughput == 200.0
    assert load.queue_waiting == 7
    assert load.queue_paused == 3
    assert load.queue_retracted == 7
    assert load.queue_grammar == 5
    assert load.live_queue_metrics_available is True


def test_live_reservation_ledger_uses_distinct_ids_and_aggregates_engine_load():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )

    async def scenario():
        await rebalancer.refresh()
        target = client.urls[0]
        fingerprint = rebalancer.deployments[target].cache_fingerprint
        estimate = ContextRecoveryEstimate(
            cache_source=CacheSource.MOONCAKE,
            expected_cached_tokens=20,
            expected_prefill_tokens=80,
            estimated_seconds=1.0,
            hit_probability=0.2,
        )
        budget = StepGenerationBudget("test", 10, None, None, 10)
        leases = [
            rebalancer._reserve(
                RoutingDecision(
                    session_id=f"ledger-{index}",
                    source_worker_url=None,
                    target_worker_url=target,
                    cache_fingerprint=fingerprint,
                    state=SchedulerState.ACTIVE,
                    reason="test",
                    target_context=estimate,
                ),
                input_ids=[1] * 100,
                base_tokens=0,
                budget=budget,
                batch_id=7,
            )
            for index in range(2)
        ]

        assert leases[0].reservation_id is not None
        assert leases[0].reservation_id != leases[1].reservation_id
        assert leases[0].batch_id == 7
        assert set(rebalancer._reservations) == {
            leases[0].reservation_id,
            leases[1].reservation_id,
        }
        assert rebalancer._live_reservation_totals(target) == (2, 220, 160)
        load = rebalancer.loads[target]
        assert (
            load.reserved_requests,
            load.reserved_tokens,
            load.reserved_prefill_tokens,
        ) == rebalancer._live_reservation_totals(target)

    run(scenario())


@pytest.mark.parametrize("settle_method", ["complete", "fail"])
def test_reservation_settle_releases_exact_entry_once(settle_method):
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )

    async def scenario():
        await rebalancer.refresh()
        target = client.urls[0]
        decision = RoutingDecision(
            session_id="exact-release",
            source_worker_url=None,
            target_worker_url=target,
            cache_fingerprint=rebalancer.deployments[target].cache_fingerprint,
            state=SchedulerState.ACTIVE,
            reason="test",
        )
        budget = StepGenerationBudget("unavailable", None, None, None, None)
        first = rebalancer._reserve(
            decision,
            input_ids=[1] * 100,
            base_tokens=0,
            budget=budget,
        )
        second = rebalancer._reserve(
            decision,
            input_ids=[1] * 100,
            base_tokens=0,
            budget=budget,
        )

        async def settle():
            if settle_method == "complete":
                await rebalancer.complete(
                    first,
                    response_meta={},
                    output_tokens=0,
                    committed_tokens=[],
                )
            else:
                await rebalancer.fail(first)

        await settle()
        assert set(rebalancer._reservations) == {second.reservation_id}
        assert rebalancer._live_reservation_totals(target) == (1, 100, 100)
        await settle()
        assert set(rebalancer._reservations) == {second.reservation_id}
        assert rebalancer._live_reservation_totals(target) == (1, 100, 100)
        load = rebalancer.loads[target]
        assert (
            load.reserved_requests,
            load.reserved_tokens,
            load.reserved_prefill_tokens,
        ) == (1, 100, 100)

    run(scenario())


def test_stale_and_legacy_settle_cannot_release_equal_newer_reservation():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )

    async def scenario():
        await rebalancer.refresh()
        target = client.urls[0]
        decision = RoutingDecision(
            session_id="stale-release",
            source_worker_url=None,
            target_worker_url=target,
            cache_fingerprint=rebalancer.deployments[target].cache_fingerprint,
            state=SchedulerState.ACTIVE,
            reason="test",
        )
        budget = StepGenerationBudget("unavailable", None, None, None, None)
        stale = rebalancer._reserve(
            decision,
            input_ids=[1] * 100,
            base_tokens=0,
            budget=budget,
        )
        await rebalancer.fail(stale)
        newer = rebalancer._reserve(
            decision,
            input_ids=[1] * 100,
            base_tokens=0,
            budget=budget,
        )
        legacy = RoutingLease(
            decision=decision,
            worker_url=target,
            reserved_tokens=100,
            base_tokens=0,
            started_monotonic=time.monotonic(),
            reserved_prefill_tokens=100,
        )

        await rebalancer.fail(stale)
        await rebalancer.fail(legacy)

        assert set(rebalancer._reservations) == {newer.reservation_id}
        assert rebalancer._live_reservation_totals(target) == (1, 100, 100)
        load = rebalancer.loads[target]
        assert (
            load.reserved_requests,
            load.reserved_tokens,
            load.reserved_prefill_tokens,
        ) == (1, 100, 100)

    run(scenario())


def test_prefill_retirement_keeps_request_and_token_reservation_until_settle():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )

    async def scenario():
        await rebalancer.refresh()
        target = client.urls[0]
        decision = RoutingDecision(
            session_id="prefill-retirement",
            source_worker_url=None,
            target_worker_url=target,
            cache_fingerprint=rebalancer.deployments[target].cache_fingerprint,
            state=SchedulerState.ACTIVE,
            reason="test",
        )
        budget = StepGenerationBudget("unavailable", None, None, None, None)
        lease = rebalancer._reserve(
            decision,
            input_ids=[1] * 100,
            base_tokens=0,
            budget=budget,
        )
        revision = rebalancer._reservation_revision

        rebalancer._advance_prefill_reservation_generation(target)
        assert rebalancer._live_reservation_totals(target) == (1, 100, 100)
        assert rebalancer._reservation_revision == revision
        rebalancer._advance_prefill_reservation_generation(target)

        assert rebalancer._live_reservation_totals(target) == (1, 100, 0)
        assert rebalancer._reservation_revision == revision + 1
        assert rebalancer._reservations[lease.reservation_id].prefill_active is False
        load = rebalancer.loads[target]
        assert (
            load.reserved_requests,
            load.reserved_tokens,
            load.reserved_prefill_tokens,
        ) == (1, 100, 0)

        await rebalancer.fail(lease)
        assert rebalancer._live_reservation_totals(target) == (0, 0, 0)
        assert (
            load.reserved_requests,
            load.reserved_tokens,
            load.reserved_prefill_tokens,
        ) == (0, 0, 0)

    run(scenario())


def test_live_prefill_ledger_retires_by_load_generation_and_releases_on_failure():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )

    async def scenario():
        await rebalancer.refresh()
        target = client.urls[0]
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

        rebalancer._advance_prefill_reservation_generation(target)
        assert rebalancer.loads[target].reserved_prefill_tokens == 80
        rebalancer._advance_prefill_reservation_generation(target)
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
        serve_current_loads_for_batch(client, rebalancer)
        rebalancer.sessions["session"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
        )
        lease = await rebalancer.acquire(
            session_id="session",
            input_ids=[1] * 100,
        )
        try:
            assert lease.decision.state is SchedulerState.ACTIVE
            assert lease.decision.moved is False
            assert lease.worker_url == source
            assert lease.decision.reason == "batch_sticky"
            assert lease.decision.source_context is None
            assert lease.decision.target_context is None
            assert lease.decision.queue_risk_seconds == 0.0
            assert lease.decision.context_risk_seconds == 0.0
            assert lease.decision.decision_risk_seconds == 0.0
            assert lease.decision.source_base_load is not None
            assert lease.decision.target_base_load is not None
            trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
            assert trace["adopted_plan"] == "sticky"
            assert trace["optimized"] is None
            assert trace["improvement_ratio"] is None
            assert trace["fallback_reason"] == "target_load_infeasible"
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_waiting_uncached_tokens_is_diagnostic_only():
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
        serve_current_loads_for_batch(client, rebalancer)
        rebalancer.sessions["live-backlog"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
        )

        lease = await rebalancer.acquire(
            session_id="live-backlog",
            input_ids=[1] * 100,
        )
        try:
            assert lease.decision.state is SchedulerState.ACTIVE
            assert lease.decision.moved is False
            assert lease.worker_url == source
            assert lease.decision.reason == "batch_sticky"
            assert lease.decision.target_base_load is not None
            trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
            target_input = next(
                engine for engine in trace["engines"] if engine["url"] == target
            )
            assert target_input["waiting_uncached_tokens"] == 40_000
            assert target_input["base_tokens"] == 0
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
        serve_current_loads_for_batch(client, rebalancer)
        rebalancer.pools[fingerprint].update(
            rebalancer._pool_readiness(fingerprint, now=time.monotonic())
        )
        return rebalancer

    async def decide(
        rebalancer: EngineRebalancer,
        *,
        session_id: str,
    ) -> RoutingDecision:
        source = rebalancer.client.urls[0]
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions[session_id] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
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
        )
        default_decision = await decide(
            default,
            session_id="default",
        )

        for decision in (
            conservative_decision,
            default_decision,
        ):
            assert decision.moved is True
            assert decision.reason == "batch_optimized_migration"
            assert decision.source_context is None
            assert decision.target_context is None
        assert conservative_decision.load_improvement_ratio == pytest.approx(
            default_decision.load_improvement_ratio
        )
        assert default.config.min_risk_ms == 10

    run(scenario())


def test_prediction_history_cannot_bypass_missing_l3_path():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_samples=1,
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
        )

        lease = await rebalancer.acquire(
            session_id="total-step",
            input_ids=[1] * 100,
            step_max_new_tokens=10,
        )
        try:
            assert lease.worker_url == source
            assert lease.decision.moved is False
            assert lease.decision.reason == "batch_sticky"
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
        )

        lease = await rebalancer.acquire(
            session_id="failed-owner",
            input_ids=[1] * 100,
            step_max_new_tokens=10,
        )
        try:
            assert lease.worker_url == target
            assert lease.decision.reason == "batch_owner_failover"
            assert lease.decision.moved is True
            assert lease.decision.decision_risk_seconds == 0.0
            assert lease.decision.target_context is None
            assert lease.decision.target_projected_load is not None
            assert lease.decision.required_load_improvement_ratio is None
            trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
            step = trace["steps"][0]
            assert step["candidate_migration_cost_tokens"] == {target: 0}
            assert step["migration_cost_tokens"] == 0
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
        serve_current_loads_for_batch(client, rebalancer)
        rebalancer.sessions["heterogeneous"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
        )
        lease = await rebalancer.acquire(
            session_id="heterogeneous",
            input_ids=[1] * 100,
            step_max_new_tokens=8192,
            context_remaining_tokens=56 * 1024,
        )
        try:
            assert lease.decision.estimated_step_output_tokens == 8192
            assert lease.decision.source_decode_seconds is None
            assert lease.decision.target_decode_seconds is None
            assert lease.decision.source_projected_load is not None
            assert lease.decision.target_projected_load is not None
            assert lease.reserved_tokens == 100 + 8192
            assert lease.decision.reason in {
                "batch_sticky",
                "batch_optimized_migration",
            }
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
        serve_current_loads_for_batch(client, rebalancer)
        rebalancer.sessions["risk-blocked"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
            seen_engines={source},
        )

        lease = await rebalancer.acquire(
            session_id="risk-blocked",
            input_ids=[1] * 100,
        )
        try:
            assert lease.decision.state is SchedulerState.ACTIVE
            assert lease.decision.moved is False
            assert lease.worker_url == source
            assert lease.decision.reason == "batch_sticky"
            assert lease.decision.queue_risk_seconds == 0.0
            assert lease.decision.context_risk_seconds == 0.0
            assert lease.decision.decision_risk_seconds == 0.0
            assert lease.decision.source_context is None
            assert lease.decision.target_context is None
            trace = (await rebalancer.snapshot())["recent_load_batches"][-1]
            assert trace["adopted_plan"] == "sticky"
            assert trace["optimized"] is None
            assert trace["improvement_ratio"] is None
            assert trace["fallback_reason"] == "target_load_infeasible"
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
            await rebalancer._drain_observation_tasks()
            observation = rebalancer._observations[-1]
            assert observation["predicted_queue_seconds"] == expected
            assert observation["actual_queue_seconds"] == actual
            assert observation["queue_prediction_error_seconds"] == abs(
                expected - actual
            )

    run(scenario())


def test_complete_commits_session_before_background_observation(monkeypatch):
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            load_batch_coalescing_window_ms=0,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )
    observation_started = asyncio.Event()
    release_observation = asyncio.Event()
    captured = []

    async def blocked_observation(observation):
        captured.append(observation)
        observation_started.set()
        await release_observation.wait()

    monkeypatch.setattr(
        rebalancer,
        "_record_completion_observation",
        blocked_observation,
        raising=False,
    )

    async def scenario():
        await rebalancer.refresh()
        lease = await rebalancer.acquire(
            session_id="background-observation",
            input_ids=[1] * 10,
        )
        assert lease.reservation_id in rebalancer._reservations

        await rebalancer.complete(
            lease,
            response_meta={
                "queue_time": 0.0,
                "e2e_latency": 1.0,
                "cached_tokens": 0,
                "decode_throughput": 10.0,
            },
            output_tokens=1,
            committed_tokens=[1] * 11,
        )
        await wait_for_condition(observation_started.is_set)

        session = rebalancer.sessions["background-observation"]
        assert lease.reservation_id not in rebalancer._reservations
        assert session.owner_worker_url == lease.worker_url
        assert session.pending_owner_worker_url is None
        assert session.previous_committed_tokens == [1] * 11
        assert list(rebalancer._observations) == []
        assert len(captured) == 1

        release_observation.set()
        await rebalancer._drain_observation_tasks()

    run(scenario())


def test_close_waits_for_background_completion_observation(monkeypatch):
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )
    observation_started = asyncio.Event()
    release_observation = asyncio.Event()

    async def blocked_observation(observation):
        del observation
        observation_started.set()
        await release_observation.wait()

    monkeypatch.setattr(
        rebalancer,
        "_record_completion_observation",
        blocked_observation,
        raising=False,
    )

    async def scenario():
        await rebalancer.refresh()
        worker = client.urls[0]
        fingerprint = rebalancer.deployments[worker].cache_fingerprint
        rebalancer.sessions["close-observation"] = SessionRoutingState(
            owner_worker_url=worker,
            fingerprint=fingerprint,
            seen_engines={worker},
        )
        lease = RoutingLease(
            decision=RoutingDecision(
                session_id="close-observation",
                source_worker_url=worker,
                target_worker_url=worker,
                cache_fingerprint=fingerprint,
                state=SchedulerState.ACTIVE,
                reason="test",
            ),
            worker_url=worker,
            reserved_tokens=10,
            base_tokens=0,
            started_monotonic=time.monotonic(),
            context_tokens=10,
        )
        await rebalancer.complete(
            lease,
            response_meta={},
            output_tokens=1,
            committed_tokens=[1] * 11,
        )
        await wait_for_condition(observation_started.is_set)

        close_task = asyncio.create_task(rebalancer.close())
        await asyncio.sleep(0)
        assert close_task.done() is False
        release_observation.set()
        await close_task

    run(scenario())


def test_background_completion_observation_failure_is_logged(monkeypatch, caplog):
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def fail_observation(observation):
        del observation
        raise RuntimeError("observation injection")

    monkeypatch.setattr(
        rebalancer,
        "_record_completion_observation",
        fail_observation,
    )

    async def scenario():
        await rebalancer.refresh()
        worker = client.urls[0]
        fingerprint = rebalancer.deployments[worker].cache_fingerprint
        rebalancer.sessions["failed-observation"] = SessionRoutingState(
            owner_worker_url=worker,
            fingerprint=fingerprint,
            seen_engines={worker},
        )
        lease = RoutingLease(
            decision=RoutingDecision(
                session_id="failed-observation",
                source_worker_url=worker,
                target_worker_url=worker,
                cache_fingerprint=fingerprint,
                state=SchedulerState.ACTIVE,
                reason="test",
            ),
            worker_url=worker,
            reserved_tokens=10,
            base_tokens=0,
            started_monotonic=time.monotonic(),
            context_tokens=10,
        )
        await rebalancer.complete(
            lease,
            response_meta={},
            output_tokens=1,
            committed_tokens=[1] * 11,
        )
        await rebalancer._drain_observation_tasks()
        assert rebalancer.sessions["failed-observation"].owner_worker_url == worker

    with caplog.at_level(
        logging.WARNING,
        logger="dressage.proxy.rebalancing.scheduler",
    ):
        run(scenario())
    assert "completion observation failed" in caplog.text


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
        await rebalancer._drain_observation_tasks()
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
    assert args.engine_rebalancing_min_load_improvement_ratio == 0.10
    assert args.engine_rebalancing_load_batch_coalescing_window_ms == 125


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


def test_cli_accepts_zero_load_batch_coalescing_window(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dressage-proxy",
            "--tokenizer-path",
            "model",
            "--engine-rebalancing-load-batch-coalescing-window-ms",
            "0",
        ],
    )

    assert parse_args().engine_rebalancing_load_batch_coalescing_window_ms == 0


def test_cli_rejects_negative_load_batch_coalescing_window(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dressage-proxy",
            "--tokenizer-path",
            "model",
            "--engine-rebalancing-load-batch-coalescing-window-ms",
            "-1",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()
    assert "must be greater than or equal to 0" in capsys.readouterr().err


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
        assert loads["effective_config"]["load_poll_interval_ms"] == 250
        assert (
            loads["effective_config"]["load_batch_coalescing_window_ms"] == 125
        )
        assert loads["effective_config"]["history_size"] == 512
        assert loads["effective_config"]["min_samples"] == 16
        assert "min_hold_turns" not in loads["effective_config"]
        assert loads["effective_config"]["min_risk_ms"] == 10
        assert loads["effective_config"]["cold_start_hit_probability"] == 1.0
        assert loads["effective_config"]["min_load_improvement_ratio"] == 0.10
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
    env = {
        **os.environ,
        "BENCHMARK_DRY_RUN": "1",
        "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
        "BENCHMARK_SEED": "20260806",
        "PROMPT_DATA": str(
            Path("examples/data/dressage_dapo_prompts_dynamic_multi.jsonl")
        ),
    }
    env.pop("DRESSAGE_PROXY_REQUEST_TIMEOUT_SEC", None)
    result = subprocess.run(
        ["bash", str(path)],
        cwd=Path.cwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "seed20260806-off-r1" in result.stdout
    assert "seed20260806-on-r1" in result.stdout
    assert "rollout batch: 64" in result.stdout
    assert "global batch:  64" in result.stdout
    assert "response max:  12288" in result.stdout
    assert "sandbox slots: 24" in result.stdout
    assert "slot timeout:  3600" in result.stdout
    assert "request timeout: 300 seconds" in result.stdout
    assert "Mooncake size: 24gb" in result.stdout
    assert "load batch window: 60 ms" in result.stdout
    assert "min load improvement ratio: 0.10" in result.stdout
    assert "dressage_dapo_prompts_step_balanced_64.jsonl" in result.stdout
    assert "warm-up" not in result.stdout
    assert "off-r2" not in result.stdout
    assert "on-r2" not in result.stdout
    assert "Valid measured pairs: `{len(valid_rows)}/1`" in source
    assert "Median rollout speedup" not in source
    assert "Warm-up" not in source
    assert not (tmp_path / "benchmark").exists()


def test_engine_rebalancing_benchmark_accepts_load_batch_window_override(tmp_path):
    path = Path(
        "examples/scripts/benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    result = subprocess.run(
        ["bash", str(path)],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "BENCHMARK_DRY_RUN": "1",
            "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
            "ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS": "0",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "load batch window: 0 ms" in result.stdout
    assert not (tmp_path / "benchmark").exists()


def test_engine_rebalancing_benchmark_accepts_load_improvement_ratio_override(
    tmp_path,
):
    path = Path(
        "examples/scripts/benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    result = subprocess.run(
        ["bash", str(path)],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "BENCHMARK_DRY_RUN": "1",
            "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
            "ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO": "0.05",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "min load improvement ratio: 0.05" in result.stdout
    assert not (tmp_path / "benchmark").exists()


@pytest.mark.parametrize("value", ["-0.1", "1.1", "invalid"])
def test_engine_rebalancing_benchmark_rejects_invalid_load_improvement_ratio(
    tmp_path,
    value,
):
    path = Path(
        "examples/scripts/benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    result = subprocess.run(
        ["bash", str(path)],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "BENCHMARK_DRY_RUN": "1",
            "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
            "ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO": value,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be a number between 0 and 1" in result.stderr
    assert not (tmp_path / "benchmark").exists()


def test_engine_rebalancing_benchmark_rejects_negative_load_batch_window(tmp_path):
    path = Path(
        "examples/scripts/benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    result = subprocess.run(
        ["bash", str(path)],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "BENCHMARK_DRY_RUN": "1",
            "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
            "ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS": "-1",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be a non-negative integer" in result.stderr
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
    assert 'BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE:-64}"' in source
    assert 'ROLLOUT_BATCH_SIZE="${BENCHMARK_BATCH_SIZE}"' in source
    assert 'GLOBAL_BATCH_SIZE="${BENCHMARK_BATCH_SIZE}"' in source
    assert '--sample-size "${ROLLOUT_BATCH_SIZE}"' in source
    assert "ROLLOUT_MAX_RESPONSE_LEN=12288" in source
    assert "DRESSAGE_BLACKBOX_SLOTS_PER_NODE=24" in source
    assert "DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC=3600" in source
    assert "MOONCAKE_GLOBAL_SEGMENT_SIZE=24gb" in source


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
        assert lease.reservation_id in rebalancer._reservations
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
        assert lease.reservation_id not in rebalancer._reservations
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
        "dapo_long_tail",
        "20260806",
        "0",
        "256",
        "1",
        "256",
        "12288",
        "16",
        "3600",
        "20",
        "0",
        "dressage.rollout.generate.blackbox_dispatch.generate",
        "",
        "blackbox",
        "background",
        "0",
        "300",
        "65536",
        "default",
        "16gb",
        "125",
        "0.05",
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
    assert environment["benchmark_workload"] == "dapo_long_tail"
    assert environment["generate_function_path"] == (
        "dressage.rollout.generate.blackbox_dispatch.generate"
    )
    assert environment["context_window"] == "65536"
    assert environment["proxy_request_timeout_sec"] == "300"
    assert environment["load_batch_coalescing_window_ms"] == "125"
    assert environment["min_load_improvement_ratio"] == "0.05"
    assert environment["engine_load_snapshot_interval_seconds"] == "5"
    assert environment["sglang_worker_load_snapshot_interval_seconds"] == "1"
    assert "prompt_source_workload_distribution_json" in environment
    assert "prompt_effective_workload_distribution_json" in environment


def test_engine_rebalancing_benchmark_injects_rebalancing_settings_only_for_on(
    tmp_path,
):
    source_recipe = Path(
        "examples/scripts/run_blackbox_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    flags = (
        "--engine-rebalancing-load-batch-coalescing-window-ms",
        "--engine-rebalancing-min-load-improvement-ratio",
    )

    for mode in ("off", "on"):
        output = tmp_path / f"{mode}.sh"
        result = _run_benchmark_heredoc(
            "build_temporary_recipe",
            str(source_recipe),
            str(output),
            mode,
        )
        assert result.returncode == 0, result.stderr
        generated = output.read_text(encoding="utf-8")
        for flag in flags:
            assert (flag in generated) is (mode == "on")
        assert ("engine_load_snapshots.jsonl" in generated) is (mode == "on")
        assert "sglang_worker_load_snapshots.jsonl" in generated
        assert "_start_benchmark_sglang_worker_load_sampler\n" in generated
        assert ("_start_benchmark_engine_load_sampler\n" in generated) is (
            mode == "on"
        )
        assert generated.index("_start_benchmark_sglang_worker_load_sampler\n") < (
            generated.index("ray job submit")
        )
        if mode == "on":
            assert generated.index("_start_benchmark_engine_load_sampler\n") < generated.index(
                "ray job submit"
            )
            cleanup = generated[
                generated.index("cleanup() {") : generated.index("trap cleanup EXIT")
            ]
            assert cleanup.index("_stop_benchmark_engine_load_sampler") < cleanup.index(
                "_capture_benchmark_snapshots"
            )
        syntax = subprocess.run(
            ["bash", "-n", str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr


def test_engine_rebalancing_benchmark_sampler_stops_without_leaking_processes(
    tmp_path,
):
    source_recipe = Path(
        "examples/scripts/run_blackbox_qwen3.5_4b_sync_local_l3_hicache.sh"
    )
    generated_path = tmp_path / "on.sh"
    generated_result = _run_benchmark_heredoc(
        "build_temporary_recipe",
        str(source_recipe),
        str(generated_path),
        "on",
    )
    assert generated_result.returncode == 0, generated_result.stderr
    generated = generated_path.read_text(encoding="utf-8")
    sampler = generated[
        generated.index("_capture_benchmark_engine_load_history() {") : generated.index(
            "cleanup() {"
        )
    ]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '{\"recent_load_batches\":[]}'\n",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    output_dir = tmp_path / "output"
    shell = f"""
set -Eeuo pipefail
{sampler}
export DRESSAGE_BENCHMARK_OUTPUT_DIR={shlex.quote(str(output_dir))}
export DRESSAGE_PROXY_URL=http://proxy.test
_start_benchmark_engine_load_sampler
sampler_pid="${{BENCHMARK_ENGINE_LOAD_SAMPLER_PID}}"
sleep 0.05
child_pid="$(pgrep -P "${{sampler_pid}}" | head -n 1 || true)"
_stop_benchmark_engine_load_sampler
if kill -0 "${{sampler_pid}}" 2>/dev/null; then
  exit 10
fi
if [[ -n "${{child_pid}}" ]] && kill -0 "${{child_pid}}" 2>/dev/null; then
  exit 11
fi
"""
    result = subprocess.run(
        ["bash", "-c", shell],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    records = [
        json.loads(line)
        for line in output_dir.joinpath("engine_load_snapshots.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["phase"] for record in records] == ["baseline", "final"]


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


def _write_tail_metric_fixture(
    run_dir: Path,
    *,
    batch_ids: tuple[int, ...] = (41, 42, 43),
) -> None:
    _write_benchmark_sample(
        run_dir / "runtime" / "traj_payload" / "run" / "samples" / "a.json",
        instance_id="alpha",
        sampling_seed=11,
        segment_index=0,
    )
    session_dir = run_dir / "runtime" / "traj_payload" / "run" / "alpha"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_dir.joinpath("session.json").write_text(
        json.dumps(
            {
                "trajectory_id": "session-alpha",
                "data": [
                    {
                        "extra_info": {
                            "segment_view": "lineage",
                            "request_metrics": [
                                {
                                    "step_id": f"step-{index}",
                                    "request_e2e_latency_seconds": float(index),
                                    "request_queue_seconds": index / 10.0,
                                    "rebalancing_batch_id": 41 + index,
                                    "rebalancing_moved": index % 2 == 1,
                                }
                                for index in range(5)
                            ],
                        }
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def trace(batch_id: int) -> dict:
        value = float(batch_id - 40)
        return {
            "batch": {
                "id": batch_id,
                "registered_count": 1 if batch_id != 42 else 2,
                "solved_count": 1,
                "total_seconds": value,
                "collect_seconds": value / 10.0,
                "wait_for_previous_seconds": value / 100.0,
                "fetch_seconds": value / 20.0,
                "solve_seconds": value / 50.0,
            },
            "engines": [
                {
                    "fetch_status": "timeout" if batch_id == 43 else "ok",
                    "fetch_duration_seconds": value / 25.0,
                }
            ],
            "sticky": {"elapsed_seconds": value / 100.0},
            "optimized": (
                {"elapsed_seconds": value / 200.0} if batch_id == 42 else None
            ),
            "fallback_reason": (
                "target_load_infeasible" if batch_id == 43 else None
            ),
            "adopted_plan": "optimized" if batch_id == 42 else "sticky",
        }

    snapshots = [
        {
            "captured_at": 1.0,
            "phase": "baseline",
            "payload": {"recent_load_batches": [trace(40)]},
        },
        {
            "captured_at": 2.0,
            "phase": "sample",
            "payload": {
                "recent_load_batches": [trace(batch_id) for batch_id in batch_ids[:2]]
            },
        },
        {
            "captured_at": 3.0,
            "phase": "final",
            "payload": {
                "recent_load_batches": [trace(batch_id) for batch_id in batch_ids[1:]]
            },
        },
    ]
    run_dir.joinpath("engine_load_snapshots.jsonl").write_text(
        "".join(json.dumps(snapshot) + "\n" for snapshot in snapshots),
        encoding="utf-8",
    )
    run_dir.joinpath("engine_load.json").write_text(
        json.dumps(snapshots[-1]["payload"]),
        encoding="utf-8",
    )
    run_dir.joinpath("calibration.json").write_text(
        json.dumps({"state": "READY"}),
        encoding="utf-8",
    )
    run_dir.joinpath("environment.txt").write_text(
        "gpu_count=8\nhostname=test\ngpu_inventory_sha256=gpu\ncode_fingerprint=code\n",
        encoding="utf-8",
    )


def test_engine_rebalancing_benchmark_collects_type7_tail_metrics(tmp_path):
    _write_tail_metric_fixture(tmp_path)

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "on", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    tail = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))[
        "tail_metrics"
    ]
    assert tail["request"]["e2e_latency_seconds"] == {
        "sample_count": 5,
        "p50": 2.0,
        "p90": 3.6,
        "p95": 3.8,
        "p99": 3.96,
        "max": 4.0,
    }
    assert tail["batch"]["total_seconds"]["p95"] == pytest.approx(2.9)
    assert tail["batch"]["registered_count"]["sample_count"] == 3
    assert tail["batch"]["singleton_ratio"] == pytest.approx(2 / 3)
    assert tail["load_fetch"]["status_counts"] == {
        "ok": 2,
        "timeout": 1,
        "error": 0,
        "invalid": 0,
    }
    assert tail["milp"]["optimized_elapsed_seconds"]["sample_count"] == 1
    assert tail["milp"]["fallback_counts"]["target_load_infeasible"] == 1
    assert tail["coverage"]["batch_ids"] == [41, 42, 43]
    assert tail["coverage"]["missing_batch_ids"] == []
    assert tail["coverage"]["complete"] is True


def test_engine_rebalancing_benchmark_collects_worker_load_skew(tmp_path):
    _write_tail_metric_fixture(tmp_path)
    snapshots = []
    for captured_at, outstanding in (
        (1.0, [10] * 8),
        (2.0, [26, 18, 14, 10, 9, 8, 8, 7]),
    ):
        snapshots.append(
            {
                "captured_at": captured_at,
                "phase": "sample",
                "topology_sha256": "topology",
                "workers": [
                    {
                        "url": f"http://worker-{index}:30000",
                        "load": {
                            "num_running_reqs": value,
                            "num_waiting_reqs": 0,
                            "num_total_tokens": value * 1000,
                            "token_usage": value / 100,
                        },
                    }
                    for index, value in enumerate(outstanding)
                ],
            }
        )
    tmp_path.joinpath("sglang_worker_load_snapshots.jsonl").write_text(
        "".join(json.dumps(snapshot) + "\n" for snapshot in snapshots),
        encoding="utf-8",
    )

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "on", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    worker_load = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )["tail_metrics"]["sglang_worker_load"]
    assert worker_load["snapshot_count"] == 2
    assert worker_load["topology_sha256"] == ["topology"]
    assert worker_load["outstanding_max_to_mean"]["max"] == pytest.approx(2.08)
    assert worker_load["token_max_to_mean"]["max"] == pytest.approx(2.08)
    assert worker_load["token_usage_max_to_mean"]["max"] == pytest.approx(2.08)


def test_engine_rebalancing_benchmark_marks_batch_history_gaps_incomplete(tmp_path):
    _write_tail_metric_fixture(tmp_path, batch_ids=(41, 43))

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "on", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    coverage = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )["tail_metrics"]["coverage"]
    assert coverage["batch_ids"] == [41, 43]
    assert coverage["missing_batch_ids"] == [42]
    assert coverage["complete"] is False
    assert "missing load batch IDs: 42" in coverage["incomplete_reasons"]


def test_engine_rebalancing_benchmark_requires_final_load_snapshot(tmp_path):
    _write_tail_metric_fixture(tmp_path)
    snapshot_path = tmp_path / "engine_load_snapshots.jsonl"
    records = [
        json.loads(line)
        for line in snapshot_path.read_text(encoding="utf-8").splitlines()
    ]
    snapshot_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in records
            if record.get("phase") != "final"
        ),
        encoding="utf-8",
    )

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "on", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    coverage = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )["tail_metrics"]["coverage"]
    assert coverage["complete"] is False
    assert "missing final load snapshot" in coverage["incomplete_reasons"]


def test_engine_rebalancing_benchmark_counts_all_executed_solver_timings(tmp_path):
    _write_tail_metric_fixture(tmp_path)
    snapshot_path = tmp_path / "engine_load_snapshots.jsonl"
    records = [
        json.loads(line)
        for line in snapshot_path.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        for trace in record.get("payload", {}).get("recent_load_batches", []):
            batch_id = trace.get("batch", {}).get("id")
            if batch_id == 42:
                trace["adopted_plan"] = "sticky"
            elif batch_id == 43:
                trace["sticky"] = None
                trace["fallback_reason"] = "sticky_solver_failure"
                trace["batch"]["solve_seconds"] = 0.25
    snapshot_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "on", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    milp = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )["tail_metrics"]["milp"]
    assert milp["solve_seconds"]["sample_count"] == 3
    assert milp["optimized_elapsed_seconds"]["sample_count"] == 1


def test_engine_rebalancing_benchmark_tail_metrics_filter_invalid_values(tmp_path):
    _write_benchmark_sample(
        tmp_path / "runtime" / "traj_payload" / "run" / "samples" / "a.json",
        instance_id="alpha",
        sampling_seed=11,
        segment_index=0,
    )
    session_dir = tmp_path / "runtime" / "traj_payload" / "run" / "alpha"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_dir.joinpath("session.json").write_text(
        json.dumps(
            {
                "trajectory_id": "session-alpha",
                "data": [
                    {
                        "extra_info": {
                            "segment_view": "timeline",
                            "step_id": "valid",
                            "request_e2e_latency_seconds": 1.0,
                            "request_queue_seconds": "invalid",
                            "rebalancing_moved": False,
                        }
                    },
                    {
                        "extra_info": {
                            "segment_view": "timeline",
                            "step_id": "negative",
                            "request_e2e_latency_seconds": -1.0,
                            "request_queue_seconds": True,
                            "rebalancing_moved": True,
                        }
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "off", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    tail = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))[
        "tail_metrics"
    ]
    assert tail["request"]["e2e_latency_seconds"] == {
        "sample_count": 1,
        "p50": 1.0,
        "p90": 1.0,
        "p95": 1.0,
        "p99": 1.0,
        "max": 1.0,
    }
    assert tail["request"]["queue_seconds"]["sample_count"] == 0
    assert tail["request"]["queue_seconds"]["p99"] is None
    assert tail["coverage"]["complete"] is False


def _write_benchmark_sample(
    path: Path,
    *,
    instance_id: str,
    sampling_seed: int | None,
    segment_index: int,
    metadata_overrides: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if sampling_seed is not None:
        metadata["rollout_sampling_seed"] = sampling_seed
    metadata.update(metadata_overrides or {})
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


def test_repeat_benchmark_collector_records_complete_trajectory_health(tmp_path):
    step_distribution = [1] * 216 + [8] + [52] * 39
    samples = tmp_path / "runtime" / "traj_payload" / "run" / "samples"
    for index, planned_steps in enumerate(step_distribution):
        _write_benchmark_sample(
            samples / f"{index:03d}.json",
            instance_id=f"repeat-{index:03d}",
            sampling_seed=11,
            segment_index=0,
            metadata_overrides={
                "planned_model_steps": planned_steps,
                "attempted_model_steps": planned_steps,
                "actual_model_steps": planned_steps,
                "failed_step_count": 0,
                "truncated_step_count": 0,
                "protocol_success": True,
                "repeat_tool_delay_ms": 0,
            },
        )
    tmp_path.joinpath("run.log").write_text(
        "[2026-08-24 12:00:00] perf 0: "
        "{'perf/rollout_time': 10.0, "
        "'perf/effective_tokens_per_gpu_per_sec': 1.0}\n",
        encoding="utf-8",
    )
    tmp_path.joinpath("environment.txt").write_text(
        "benchmark_workload=repeat_multistep\n"
        "rollout_batch_size=256\n"
        "prompt_effective_planned_model_steps_total=2252\n"
        "prompt_effective_sha256=dataset-sha\n"
        "prompt_effective_workload_distribution_json={\"steps:1\":216,\"steps:8\":1,\"steps:52\":39}\n"
        "generate_function_path=dressage.recipes.repeat_multistep.agent_whitebox.generate\n"
        "repeat_tool_delay_ms=0\n"
        "context_window=262144\n"
        "sglang_context_length=262144\n"
        "gpu_count=8\n"
        "hostname=test\n"
        "gpu_inventory_sha256=gpu\n"
        "code_fingerprint=code\n",
        encoding="utf-8",
    )

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "off", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["repeat_workload"] == {
        "actual_model_steps_total": 2252,
        "attempted_model_steps_total": 2252,
        "failed_step_count": 0,
        "planned_model_steps_total": 2252,
        "protocol_failure_count": 0,
        "rebalancing_batch_id_count": 0,
        "trajectory_health": {
            f"repeat-{index:03d}": {
                "actual_model_steps": planned_steps,
                "attempted_model_steps": planned_steps,
                "failed_step_count": 0,
                "planned_model_steps": planned_steps,
                "protocol_success": True,
                "repeat_tool_delay_ms": 0,
                "truncated_step_count": 0,
            }
            for index, planned_steps in enumerate(step_distribution)
        },
        "trajectory_health_count": 256,
        "truncated_step_count": 0,
    }
    assert not any(
        "repeat workload" in error for error in metrics["acceptance_errors"]
    )
    assert metrics["workload"] == {
        "benchmark_workload": "repeat_multistep",
        "context_window": "262144",
        "generate_function_path": (
            "dressage.recipes.repeat_multistep.agent_whitebox.generate"
        ),
        "planned_model_steps_total": "2252",
        "prompt_effective_sha256": "dataset-sha",
        "repeat_tool_delay_ms": "0",
        "sglang_context_length": "262144",
        "step_distribution_json": (
            '{"steps:1":216,"steps:8":1,"steps:52":39}'
        ),
    }


def test_repeat_benchmark_collector_requires_batch_ids_and_natural_batch(tmp_path):
    _write_tail_metric_fixture(tmp_path)
    tmp_path.joinpath("environment.txt").write_text(
        "benchmark_workload=repeat_multistep\n"
        "rollout_batch_size=256\n"
        "prompt_effective_planned_model_steps_total=2252\n"
        "gpu_count=8\n",
        encoding="utf-8",
    )
    session_path = (
        tmp_path / "runtime" / "traj_payload" / "run" / "alpha" / "session.json"
    )
    session = json.loads(session_path.read_text(encoding="utf-8"))
    for metric in session["data"][0]["extra_info"]["request_metrics"]:
        metric.pop("rebalancing_batch_id")
    session_path.write_text(json.dumps(session), encoding="utf-8")
    snapshot_path = tmp_path / "engine_load_snapshots.jsonl"
    snapshots = [
        json.loads(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines()
    ]
    for snapshot in snapshots:
        for trace in snapshot["payload"]["recent_load_batches"]:
            trace["batch"]["registered_count"] = 1
    snapshot_path.write_text(
        "".join(json.dumps(snapshot) + "\n" for snapshot in snapshots),
        encoding="utf-8",
    )

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "on", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    errors = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))[
        "acceptance_errors"
    ]
    assert "repeat workload ON requests missing rebalancing batch IDs: 5/5" in errors
    assert "repeat workload observed no natural multi-step load batch" in errors


def test_benchmark_collector_rejects_context_overflow(tmp_path):
    _write_benchmark_sample(
        tmp_path / "runtime" / "traj_payload" / "run" / "samples" / "a.json",
        instance_id="alpha",
        sampling_seed=11,
        segment_index=0,
    )
    tmp_path.joinpath("run.log").write_text(
        "request rejected: input exceeds maximum context length\n",
        encoding="utf-8",
    )

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "off", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    errors = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))[
        "acceptance_errors"
    ]
    assert any("detected context_overflow" in error for error in errors)


@pytest.mark.parametrize(
    ("log_line", "error_key"),
    [
        ("httpx.ReadTimeout while reading model response", "read_timeout"),
        ("KV cache pool is full. Retract requests", "kv_cache_pool_full"),
    ],
)
def test_repeat_benchmark_collector_rejects_runtime_capacity_failures(
    tmp_path,
    log_line,
    error_key,
):
    _write_benchmark_sample(
        tmp_path / "runtime" / "traj_payload" / "run" / "samples" / "a.json",
        instance_id="alpha",
        sampling_seed=11,
        segment_index=0,
    )
    tmp_path.joinpath("run.log").write_text(log_line + "\n", encoding="utf-8")

    result = _run_benchmark_heredoc(
        "collect_run", str(tmp_path), "run", "off", "0", "1", "2", "1"
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["error_matches"][error_key]
    assert any(
        error.startswith(f"detected {error_key}:")
        for error in metrics["acceptance_errors"]
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


def test_engine_rebalancing_benchmark_summary_reports_tail_metrics(tmp_path):
    def metric(p50, p95, p99):
        return {
            "sample_count": 10,
            "p50": p50,
            "p90": None if p95 is None else p95 - 0.1,
            "p95": p95,
            "p99": p99,
            "max": None if p99 is None else p99 + 0.1,
        }

    for mode in ("off", "on"):
        run_name = f"seed42-{mode}-r1"
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        tail = {
            "request": {
                "e2e_latency_seconds": metric(
                    1.0,
                    2.0 if mode == "off" else 1.8,
                    3.0,
                ),
                "queue_seconds": metric(0.1, 0.2, 0.3),
                "moved_e2e_latency_seconds": (
                    metric(1.1, 2.1, 3.1)
                    if mode == "on"
                    else metric(None, None, None)
                ),
                "sticky_e2e_latency_seconds": metric(0.9, 1.9, 2.9),
            },
            "batch": {
                "total_seconds": metric(0.06, 0.08, 0.1),
                "registered_count": metric(2.0, 4.0, 5.0),
                "singleton_ratio": 0.25,
            },
            "load_fetch": {"batch_fetch_seconds": metric(0.01, 0.02, 0.03)},
            "milp": {"solve_seconds": metric(0.005, 0.01, 0.02)},
            "coverage": {"complete": True},
        }
        payload = {
            "run_name": run_name,
            "valid_run": True,
            "acceptance_errors": [],
            "hostname": "host",
            "gpu_inventory_sha256": "gpu",
            "code_fingerprint": "code",
            "effective_token_total": 100,
            "trajectory_hash": "trajectory",
            "calibration_state": "READY" if mode == "on" else "OFF",
            "rollout_time_seconds": 10.0,
            "effective_tokens_per_gpu_per_sec": 20.0,
            "tail_metrics": tail,
            "kv_migration_evidence": mode == "on",
        }
        run_dir.joinpath("metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _run_benchmark_heredoc("write_summary", str(tmp_path), "42")

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    row = summary["pairs"][0]
    assert summary["tail_metrics_by_run"]["seed42-on-r1"]["request"][
        "e2e_latency_seconds"
    ]["p90"] == 1.7
    assert row["off_request_e2e_p95"] == 2.0
    assert row["on_request_e2e_p95"] == 1.8
    assert row["on_moved_e2e_p99"] == 3.1
    assert row["on_moved_e2e_count"] == 10
    assert row["on_batch_total_p95"] == 0.08
    assert row["on_batch_count"] == 10
    assert row["on_batch_fetch_p99"] == 0.03
    assert row["on_solve_p99"] == 0.02
    assert row["on_batch_size_p50"] == 2.0
    assert row["on_batch_singleton_ratio"] == 0.25
    assert row["on_tail_metrics_complete"] is True
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Tail latency" in markdown
    assert "Request E2E" in markdown
    assert "Batch total" in markdown
    assert "| Metric | OFF N | OFF P95 | OFF P99 | ON N | ON P95 | ON P99 |" in markdown

    on_metrics_path = tmp_path / "seed42-on-r1" / "metrics.json"
    on_metrics = json.loads(on_metrics_path.read_text(encoding="utf-8"))
    on_metrics["tail_metrics"]["coverage"]["complete"] = False
    on_metrics_path.write_text(json.dumps(on_metrics), encoding="utf-8")
    incomplete_result = _run_benchmark_heredoc("write_summary", str(tmp_path), "42")
    assert incomplete_result.returncode == 0, incomplete_result.stderr
    incomplete_row = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )["pairs"][0]
    assert incomplete_row["on_request_e2e_p95"] is None
    assert incomplete_row["on_batch_total_p95"] is None
    assert incomplete_row["on_tail_metrics_complete"] is False


def test_sync_local_script_gracefully_stops_proxy_before_ray():
    path = Path("examples/scripts/run_blackbox_qwen3.5_4b_sync_local.sh")
    source = path.read_text()
    cleanup = source[source.index("cleanup() {") : source.index("trap cleanup EXIT")]

    assert cleanup.index("_stop_proxy_on_exit") < cleanup.index(
        "_stop_ray_cluster_on_exit"
    )
    assert '[[ "${wait_count}" -lt 100 ]]' in source
    assert "Dressage proxy did not stop gracefully; sending SIGKILL" in source
