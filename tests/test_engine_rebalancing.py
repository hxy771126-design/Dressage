from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from dressage.proxy.rebalancing import EngineRebalancer, EngineRebalancingConfig
from dressage.proxy.rebalancing.greedy import projected_pressure
from dressage.proxy.rebalancing.model_cache_profile import ModelCacheProfile
from dressage.proxy.rebalancing.scheduler import (
    EngineDeploymentInfo,
    EngineLoad,
    RoutingDecision,
    RoutingLease,
    SessionRoutingState,
    longest_common_prefix_length,
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
    CacheSource,
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


async def publish_current_loads(client, rebalancer):
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
                }
            ]
        }

    client.get_worker_loads = get_worker_loads
    await asyncio.gather(
        *(rebalancer._poll_engine_snapshot(url) for url in client.urls)
    )


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


class ControlledLoadClient(ControlPlaneClient):
    def __init__(self):
        super().__init__()
        self.load_calls = {url: 0 for url in self.urls}
        self.load_futures = {url: [] for url in self.urls}
        self.load_started = None

    def control_loads(self):
        self.load_started = asyncio.Queue()

    async def get_worker_loads(self, url):
        if self.load_started is None:
            return await super().get_worker_loads(url)
        index = self.load_calls[url]
        self.load_calls[url] += 1
        future = asyncio.get_running_loop().create_future()
        self.load_futures[url].append(future)
        self.load_started.put_nowait((url, index))
        return await future

    def resolve_all(
        self,
        index,
        *,
        running=0,
        active_tokens=0,
        queued=0,
        token_usage=0.0,
    ):
        payload = {
            "loads": [
                {
                    "num_running_reqs": running,
                    "num_waiting_reqs": queued,
                    "num_total_tokens": active_tokens,
                    "max_total_num_tokens": 100_000,
                    "max_running_requests": 100,
                    "token_usage": token_usage,
                }
            ]
        }
        for url in self.urls:
            self.load_futures[url][index].set_result(payload)

    def resolve_url(
        self,
        url,
        index,
        *,
        running=0,
        active_tokens=0,
        queued=0,
        token_usage=0.0,
    ):
        self.load_futures[url][index].set_result(
            {
                "loads": [
                    {
                        "num_running_reqs": running,
                        "num_waiting_reqs": queued,
                        "num_total_tokens": active_tokens,
                        "max_total_num_tokens": 100_000,
                        "max_running_requests": 100,
                        "token_usage": token_usage,
                    }
                ]
            }
        )

    def fail_url(self, url, index):
        self.load_futures[url][index].set_exception(RuntimeError("load failed"))


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


def test_config_defaults_cover_greedy_scheduler():
    config = EngineRebalancingConfig(enabled=True)
    assert config.snapshot()["load_poll_interval_ms"] == 250
    assert config.snapshot()["load_snapshot_poll_interval_ms"] == 60
    assert config.snapshot()["history_size"] == 512
    assert config.snapshot()["min_load_improvement_ratio"] == 0.10


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_config_rejects_out_of_range_load_improvement_ratio(value):
    with pytest.raises(ValueError, match="min_load_improvement_ratio"):
        EngineRebalancingConfig(min_load_improvement_ratio=value)


def test_config_rejects_non_positive_snapshot_poll_interval():
    for value in (0, -1):
        with pytest.raises(ValueError, match="load_snapshot_poll_interval_ms"):
            EngineRebalancingConfig(load_snapshot_poll_interval_ms=value)


def test_effective_load_adds_request_token_and_queue_pressure():
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
        load.queued = 3
        load.active_tokens = 1_000
        load.request_capacity = 100
        load.token_capacity = 100_000
        load.token_usage = 0.005
        rebalancer.loads[client.urls[1]].queued = 1_000

        baseline = rebalancer._effective_baseline_locked(target)
        assert baseline is not None
        base = projected_pressure(baseline)
        projected = projected_pressure(
            baseline,
            token_increment=100,
            queue_increment=1,
        )

        assert base.request == pytest.approx(0.02)
        assert base.token == pytest.approx(0.01)
        assert base.queue == pytest.approx(0.03)
        assert base.total == pytest.approx(0.06)
        assert projected.request == base.request
        assert projected.token == pytest.approx(0.011)
        assert projected.queue == pytest.approx(0.04)
        assert projected.total == pytest.approx(0.071)
        load.token_usage = 0.02
        baseline = rebalancer._effective_baseline_locked(target)
        assert baseline is not None
        assert projected_pressure(baseline).token == pytest.approx(0.02)

    run(scenario())


def test_state_machine_distinguishes_bootstrap_and_degraded():
    config = EngineRebalancingConfig(enabled=True)
    state = CompatibilityPoolStateMachine("fp", config, now=1.0)
    not_ready = PoolReadiness(2, True, 0)
    ready = PoolReadiness(2, True, 1)

    assert state.update(not_ready, now=2.0) is SchedulerState.BOOTSTRAP
    assert state.update(ready, now=3.0) is SchedulerState.ACTIVE
    assert state.update(not_ready, now=4.0) is SchedulerState.DEGRADED
    assert state.update(ready, now=5.0) is SchedulerState.ACTIVE


def test_pool_readiness_requires_fresh_metrics_and_eligible_path():
    readiness = PoolReadiness(
        healthy_engines=2,
        metrics_fresh=True,
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
    )
    assert profile.estimate_bytes(8 * 1024) == 319_946_752
    assert profile.estimate_bytes(56 * 1024) == 1_930_559_488


def test_longest_common_prefix_length():
    assert longest_common_prefix_length([1, 2, 3], [1, 2, 9]) == 2


def test_old_sglang_versions_are_not_rebalancing_compatible():
    assert not sglang_rebalancing_supported("0.5.12")
    assert not sglang_rebalancing_supported("v0.5.15")
    assert sglang_rebalancing_supported("0.5.15.post1")
    assert sglang_rebalancing_supported("0.5.16")


def test_calibration_plan_skips_mooncake_without_l3():
    client = ControlPlaneClient(shared_l3=False)
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
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
        config=EngineRebalancingConfig(enabled=True),
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
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        readiness = rebalancer._path_readiness(source, target)
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
        assert rebalancer._control_poll_task is None
        assert rebalancer._snapshot_poll_task is None
        await rebalancer.refresh()
        assert client.list_workers_calls == 0

        calibration_gate.set()
        assert rebalancer._calibration_task is not None
        await rebalancer._calibration_task
        assert rebalancer._control_poll_task is not None
        assert rebalancer._snapshot_poll_task is not None
        await wait_for_condition(lambda: client.list_workers_calls == 1)
        await rebalancer.close()

    run(scenario())


def test_acquire_does_not_wait_for_calibration_or_perform_discovery():
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
        config=EngineRebalancingConfig(enabled=True),
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

        lease = await acquire_task
        try:
            assert client.list_workers_calls == 0
            assert lease.worker_url is None
        finally:
            await rebalancer.fail(lease)
            calibration_gate.set()
            await rebalancer._calibration_task

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
        assert rebalancer._control_poll_task is None
        assert rebalancer._snapshot_poll_task is None

        allow_write.set()
        assert rebalancer._calibration_task is not None
        await rebalancer._calibration_task
        assert (rebalancer._snapshot_store.directory / "initial.json").is_file()
        assert rebalancer._control_poll_task is not None
        assert rebalancer._snapshot_poll_task is not None
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
        run(rebalancer._control_poll_loop())

    waiting_records = [
        record
        for record in caplog.records
        if "waiting_for_router" in record.getMessage()
    ]
    assert delays[:5] == [0.25, 1.0, 2.0, 5.0, 0.25]
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


def test_calibration_snapshots_are_atomic_periodic_and_final(tmp_path):
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
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
            ),
            worker_url="worker",
        )
        await rebalancer.fail(failed)
        assert rebalancer._online_request_count == 127

        rebalancer._record_successful_online_request()
        await rebalancer._drain_snapshot_tasks()
        periodic = directory / "request-000000128.json"
        assert periodic.is_file()
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
            }
            assert payload["snapshot_type"] == expected_kind
            assert "results" in payload["offline_calibration"]
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
        config=EngineRebalancingConfig(enabled=True),
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
        config=EngineRebalancingConfig(enabled=True),
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


def test_unobserved_deltas_spread_simultaneous_new_sessions():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        await asyncio.gather(
            *(rebalancer._poll_engine_snapshot(url) for url in client.urls)
        )
        first, second = await asyncio.gather(
            rebalancer.acquire(session_id="a", input_ids=[1] * 10_000),
            rebalancer.acquire(session_id="b", input_ids=[1] * 10_000),
        )
        try:
            assert first.worker_url != second.worker_url
            assert sum(
                rebalancer._live_scoring_totals(url)[0]
                for url in client.urls
            ) == 2
        finally:
            await rebalancer.fail(first)
            await rebalancer.fail(second)

    run(scenario())


def test_acquire_uses_cached_snapshot_without_fetching_loads():
    class CountingClient(ControlPlaneClient):
        def __init__(self):
            super().__init__()
            self.load_calls = 0

        async def get_worker_loads(self, url):
            self.load_calls += 1
            return await super().get_worker_loads(url)

    client = CountingClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        await asyncio.gather(
            *(rebalancer._poll_engine_snapshot(url) for url in client.urls)
        )
        calls_before = client.load_calls
        lease = await rebalancer.acquire(session_id="cached", input_ids=[1, 2])
        try:
            assert client.load_calls == calls_before
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_successful_poll_acknowledges_only_preexisting_deltas():
    client = ControlledLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        for load in rebalancer.loads.values():
            load.metrics_timestamp = time.monotonic()
            load.request_capacity = 100
            load.token_capacity = 100_000

        first = await rebalancer.acquire(
            session_id="before-poll",
            input_ids=[1] * 100,
        )
        target = first.worker_url
        assert target is not None
        client.control_loads()
        poll = asyncio.create_task(rebalancer._poll_engine_snapshot(target))
        assert await client.load_started.get() == (target, 0)

        second = await rebalancer.acquire(
            session_id="during-poll",
            input_ids=[1] * 200,
        )
        client.resolve_url(target, 0)
        await poll

        active = [
            entry
            for entry in rebalancer._reservations.values()
            if entry.engine_url == target and entry.scoring_active
        ]
        assert all(entry.scoring_revision > 1 for entry in active)
        assert rebalancer._reservations[first.reservation_id].scoring_active is False
        if second.worker_url == target:
            assert rebalancer._reservations[second.reservation_id].scoring_active is True
        await rebalancer.fail(first)
        await rebalancer.fail(second)

    run(scenario())


def test_failed_poll_keeps_unobserved_delta():
    client = ControlledLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        for load in rebalancer.loads.values():
            load.metrics_timestamp = time.monotonic()
            load.request_capacity = 100
            load.token_capacity = 100_000
        lease = await rebalancer.acquire(session_id="failed-poll", input_ids=[1] * 10)
        assert lease.worker_url is not None
        previous_timestamp = rebalancer.loads[lease.worker_url].metrics_timestamp
        client.control_loads()
        poll = asyncio.create_task(
            rebalancer._poll_engine_snapshot(lease.worker_url)
        )
        assert await client.load_started.get() == (lease.worker_url, 0)
        client.fail_url(lease.worker_url, 0)
        await poll
        assert rebalancer._reservations[lease.reservation_id].scoring_active is True
        assert rebalancer.loads[lease.worker_url].metrics_timestamp == previous_timestamp
        assert rebalancer.loads[lease.worker_url].snapshot_fetch_status == "error"
        await rebalancer.fail(lease)

    run(scenario())


def test_complete_and_fail_release_reservations_idempotently():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        await asyncio.gather(
            *(rebalancer._poll_engine_snapshot(url) for url in client.urls)
        )
        lease = await rebalancer.acquire(session_id="release", input_ids=[1] * 10)
        reservation_id = lease.reservation_id
        assert reservation_id in rebalancer._reservations
        await rebalancer.fail(lease)
        await rebalancer.fail(lease)
        assert reservation_id not in rebalancer._reservations

    run(scenario())


def test_stale_snapshot_uses_sticky_and_stable_hash_fallback():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, metrics_stale_ms=1),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        for load in rebalancer.loads.values():
            load.metrics_timestamp = time.monotonic() - 1
            load.request_capacity = 100
            load.token_capacity = 100_000
        source = client.urls[0]
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["old"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1],
        )
        sticky = await rebalancer.acquire(session_id="old", input_ids=[1, 2])
        fresh = await rebalancer.acquire(session_id="new", input_ids=[3])
        try:
            assert sticky.worker_url == source
            assert fresh.worker_url in client.urls
        finally:
            await rebalancer.fail(sticky)
            await rebalancer.fail(fresh)

    run(scenario())


def test_no_discovered_engine_falls_back_to_router():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        lease = await rebalancer.acquire(session_id="router", input_ids=[1])
        assert lease.worker_url is None
        assert lease.reservation_id is None

    run(scenario())


def test_cold_start_waits_only_while_initial_snapshot_window_is_open():
    rebalancer = EngineRebalancer(
        ControlPlaneClient(),
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        rebalancer._snapshot_poll_started_monotonic = time.monotonic()
        waiting = asyncio.create_task(
            rebalancer.acquire(session_id="cold-wait", input_ids=[1])
        )
        await asyncio.sleep(0)
        assert not waiting.done()
        rebalancer._initial_snapshot_event.set()
        lease = await waiting
        assert lease.worker_url is None

        rebalancer._initial_snapshot_event.clear()
        rebalancer._snapshot_poll_started_monotonic = time.monotonic() - 2.0
        immediate = await asyncio.wait_for(
            rebalancer.acquire(session_id="cold-expired", input_ids=[2]),
            timeout=0.05,
        )
        assert immediate.worker_url is None

    run(scenario())


def test_mandatory_failover_bypasses_threshold():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            min_load_improvement_ratio=1.0,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        await asyncio.gather(
            *(rebalancer._poll_engine_snapshot(url) for url in client.urls)
        )
        source, target = client.urls
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.loads[source].healthy = False
        rebalancer.sessions["failover"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1],
        )
        lease = await rebalancer.acquire(
            session_id="failover",
            input_ids=[1, 2],
        )
        try:
            assert lease.worker_url == target
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_online_decision_trace_is_serializable_and_bounded():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True, history_size=2),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        await asyncio.gather(
            *(rebalancer._poll_engine_snapshot(url) for url in client.urls)
        )
        leases = []
        for index in range(3):
            leases.append(
                await rebalancer.acquire(
                    session_id=f"trace-{index}",
                    input_ids=[index],
                )
            )
        snapshot = await rebalancer.snapshot()
        traces = snapshot["recent_load_decisions"]
        assert [trace["decision"]["id"] for trace in traces] == [2, 3]
        assert all(
            trace["scheduler"]["strategy"] == "online_dynamic_greedy"
            for trace in traces
        )
        assert json.loads(json.dumps(traces)) == traces
        for lease in leases:
            await rebalancer.fail(lease)

    run(scenario())


def test_background_poll_supervisor_limits_one_fetch_per_engine():
    client = ControlledLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            load_snapshot_poll_interval_ms=1,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_loads()
        rebalancer._snapshot_poll_task = asyncio.create_task(
            rebalancer._snapshot_poll_loop()
        )
        started = {
            await client.load_started.get(),
            await client.load_started.get(),
        }
        assert {url for url, index in started if index == 0} == set(client.urls)
        await asyncio.sleep(0.01)
        assert all(count == 1 for count in client.load_calls.values())
        for url in client.urls:
            client.resolve_url(url, 0)
        await wait_for_condition(
            lambda: all(
                rebalancer.loads[url].metrics_timestamp > 0
                for url in client.urls
            )
        )
        await rebalancer.close()

    run(scenario())


def test_slow_snapshot_fetch_does_not_block_other_engine_publication():
    client = ControlledLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_loads()
        slow_url, fast_url = client.urls
        slow = asyncio.create_task(rebalancer._poll_engine_snapshot(slow_url))
        fast = asyncio.create_task(rebalancer._poll_engine_snapshot(fast_url))
        started = {
            await client.load_started.get(),
            await client.load_started.get(),
        }
        assert {url for url, _ in started} == set(client.urls)

        client.resolve_url(fast_url, 0, running=7)
        await fast
        assert rebalancer.loads[fast_url].running == 7
        assert rebalancer.loads[fast_url].snapshot_generation == 1
        assert not slow.done()

        slow.cancel()
        await asyncio.gather(slow, return_exceptions=True)

    run(scenario())


def test_close_cancels_snapshot_supervisor_and_inflight_fetches():
    client = ControlledLoadClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(
            enabled=True,
            load_snapshot_poll_interval_ms=1,
        ),
        model_id="model",
        model_config=simple_model_config(),
    )

    async def scenario():
        await rebalancer.refresh()
        client.control_loads()
        rebalancer._snapshot_poll_task = asyncio.create_task(
            rebalancer._snapshot_poll_loop()
        )
        await client.load_started.get()
        await client.load_started.get()
        assert len(rebalancer._snapshot_fetch_tasks) == 2

        await rebalancer.close()

        assert rebalancer._snapshot_poll_task is None
        assert rebalancer._snapshot_fetch_tasks == {}
        assert all(
            future.cancelled()
            for futures in client.load_futures.values()
            for future in futures
        )

    run(scenario())


@pytest.mark.parametrize(
    (
        "source_tokens",
        "target_tokens",
        "source_running",
        "target_running",
        "source_queued",
        "target_queued",
        "expected_worker",
        "expected_reason",
        "expected_required_ratio",
    ),
    [
        (
            100_000,
            60_000,
            0,
            0,
            0,
            0,
            "target",
            "load_improvement_threshold_met",
            0.10,
        ),
        (
            100_000,
            60_000,
            0,
            1_000,
            0,
            1_000,
            "source",
            "load_improvement_below_threshold",
            0.10,
        ),
        (
            10_000,
            10_000,
            1_000,
            0,
            1_000,
            0,
            "target",
            "load_improvement_threshold_met",
            0.10,
        ),
    ],
)
def test_existing_session_uses_additive_pressure(
    source_tokens,
    target_tokens,
    source_running,
    target_running,
    source_queued,
    target_queued,
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
        rebalancer.loads[source].active_tokens = source_tokens
        rebalancer.loads[source].running = source_running
        rebalancer.loads[source].queued = source_queued
        rebalancer.loads[target].active_tokens = target_tokens
        rebalancer.loads[target].running = target_running
        rebalancer.loads[target].queued = target_queued
        await publish_current_loads(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["backlog"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
        )

        lease = await rebalancer.acquire(
            session_id="backlog",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == {
                "source": source,
                "target": target,
            }[expected_worker]
            trace = (await rebalancer.snapshot())["recent_load_decisions"][-1]
            assert trace["scheduler"]["strategy"] == "online_dynamic_greedy"
            assert trace["step"]["decision_reason"] == expected_reason
            assert trace["step"]["threshold_met"] is (
                expected_worker == "target"
            )
            assert (
                trace["step"]["required_improvement_ratio"]
                == expected_required_ratio
            )
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_existing_session_selects_lowest_additive_pressure_target():
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
        source, lowest_token_target, higher_token_target = client.urls
        rebalancer.loads[source].active_tokens = 400
        rebalancer.loads[source].queued = 1
        rebalancer.loads[lowest_token_target].queued = 1
        rebalancer.loads[higher_token_target].active_tokens = 300
        await publish_current_loads(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["backlog-target"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
        )

        lease = await rebalancer.acquire(
            session_id="backlog-target",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == higher_token_target
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_existing_session_selects_lowest_token_pressure_target():
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
        rebalancer.loads[source].active_tokens = 100_000
        rebalancer.loads[lowest_load].active_tokens = 20_000
        rebalancer.loads[higher_load].active_tokens = 50_000
        await publish_current_loads(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["no-backlog-target"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
        )

        lease = await rebalancer.acquire(
            session_id="no-backlog-target",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == lowest_load
        finally:
            await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize(
    (
        "target_projected_percent",
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
    target_projected_percent, minimum_ratio, expected_worker, expected_reason
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
        rebalancer.loads[source].active_tokens = 100_000
        rebalancer.loads[target].active_tokens = (
            target_projected_percent * 1_010 - 1_100
        )
        await publish_current_loads(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["threshold"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
        )

        lease = await rebalancer.acquire(
            session_id="threshold",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == {
                "source": source,
                "target": target,
            }[expected_worker]
            trace = (await rebalancer.snapshot())["recent_load_decisions"][-1]
            assert trace["step"]["decision_reason"] == expected_reason
            assert trace["step"]["threshold_met"] is (
                expected_worker == "target"
            )
            assert trace["step"]["required_improvement_ratio"] == minimum_ratio
        finally:
            await rebalancer.fail(lease)

    run(scenario())


@pytest.mark.parametrize("target_projected_percent", [50, 40])
def test_batch_gate_uses_configured_ratio_across_target_loads(
    target_projected_percent,
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
            min_load_improvement_ratio=0.30,
        ),
        model_id="model",
        model_config=simple_model_config(),
        calibration_benchmark=benchmark,
    )

    async def scenario():
        await rebalancer.refresh()
        source, target = client.urls
        rebalancer.loads[source].active_tokens = 100_000
        rebalancer.loads[target].active_tokens = (
            target_projected_percent * 1_000 - 100
        )
        await publish_current_loads(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["return"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
        )

        lease = await rebalancer.acquire(
            session_id="return",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == target
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
        rebalancer.loads[source].active_tokens = 100_000
        rebalancer.loads[previous_owner].active_tokens = 49_900
        await publish_current_loads(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["no-backlog-return"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
        )

        lease = await rebalancer.acquire(
            session_id="no-backlog-return",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == previous_owner
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
        await publish_current_loads(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["multi-hop"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
        )

        sticky = await rebalancer.acquire(
            session_id="multi-hop",
            input_ids=[1] * 100,
        )
        assert sticky.worker_url in client.urls
        await rebalancer.complete(
            sticky,
            committed_tokens=[1] * 100,
        )
        state = rebalancer.sessions["multi-hop"]
        assert state.owner_worker_url == sticky.worker_url
        assert state.previous_committed_tokens == [1] * 100

        movable = await rebalancer.acquire(
            session_id="multi-hop",
            input_ids=[1] * 100,
        )
        try:
            assert movable.worker_url in client.urls
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
        await publish_current_loads(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["seen-no-l3"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 100,
        )
        assert (
            rebalancer._path_readiness(source, target).cache_source
            is CacheSource.NONE
        )

        lease = await rebalancer.acquire(
            session_id="seen-no-l3",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == source
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
        await publish_current_loads(client, rebalancer)
        fingerprint = rebalancer.deployments[source].cache_fingerprint
        rebalancer.sessions["projected-safety"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 150,
        )

        lease = await rebalancer.acquire(
            session_id="projected-safety",
            input_ids=[1] * 150,
        )
        try:
            assert lease.worker_url == source
            trace = (await rebalancer.snapshot())["recent_load_decisions"][-1]
            assert trace["scheduler"]["strategy"] == "online_dynamic_greedy"
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


def test_load_snapshot_aggregates_scoring_fields_across_dp_ranks():
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
                    "num_running_reqs": 1,
                    "num_waiting_reqs": 2,
                    "num_total_tokens": 3_000,
                    "max_total_num_tokens": 10_000,
                    "max_running_requests": 10,
                    "token_usage": 0.3,
                },
                {
                    "num_running_reqs": 2,
                    "num_waiting_reqs": 5,
                    "num_total_tokens": 5_000,
                    "max_total_num_tokens": 20_000,
                    "max_running_requests": 20,
                    "token_usage": 0.25,
                },
            ]
        },
        now=1.0,
    )

    assert load is not None
    assert load.running == 3
    assert load.queued == 7
    assert load.active_tokens == 8_000
    assert load.token_capacity == 30_000
    assert load.request_capacity == 30
    assert load.token_usage == 0.3


def test_scoring_reservations_use_distinct_ids_and_aggregate_deltas():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
    )

    async def scenario():
        await rebalancer.refresh()
        target = client.urls[0]
        leases = [
            rebalancer._reserve(
                RoutingDecision(
                    session_id=f"ledger-{index}",
                    source_worker_url=None,
                    target_worker_url=target,
                ),
                scoring_queue_increment=1,
                scoring_token_increment=100,
            )
            for index in range(2)
        ]

        assert leases[0].reservation_id is not None
        assert leases[0].reservation_id != leases[1].reservation_id
        assert set(rebalancer._reservations) == {
            leases[0].reservation_id,
            leases[1].reservation_id,
        }
        assert rebalancer._live_scoring_totals(target) == (2, 200)

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
        )
        first = rebalancer._reserve(
            decision,
            scoring_queue_increment=1,
            scoring_token_increment=100,
        )
        second = rebalancer._reserve(
            decision,
            scoring_queue_increment=1,
            scoring_token_increment=100,
        )

        async def settle():
            if settle_method == "complete":
                await rebalancer.complete(
                    first,
                    committed_tokens=[],
                )
            else:
                await rebalancer.fail(first)

        await settle()
        assert set(rebalancer._reservations) == {second.reservation_id}
        assert rebalancer._live_scoring_totals(target) == (1, 100)
        await settle()
        assert set(rebalancer._reservations) == {second.reservation_id}
        assert rebalancer._live_scoring_totals(target) == (1, 100)

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
        )
        stale = rebalancer._reserve(
            decision,
            scoring_queue_increment=1,
            scoring_token_increment=100,
        )
        await rebalancer.fail(stale)
        newer = rebalancer._reserve(
            decision,
            scoring_queue_increment=1,
            scoring_token_increment=100,
        )
        legacy = RoutingLease(
            decision=decision,
            worker_url=target,
        )

        await rebalancer.fail(stale)
        await rebalancer.fail(legacy)

        assert set(rebalancer._reservations) == {newer.reservation_id}
        assert rebalancer._live_scoring_totals(target) == (1, 100)

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
        rebalancer.loads[source].active_tokens = 100_000
        rebalancer.loads[target].active_tokens = 49_900
        await publish_current_loads(client, rebalancer)
        rebalancer.sessions["session"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
        )
        lease = await rebalancer.acquire(
            session_id="session",
            input_ids=[1] * 100,
        )
        try:
            assert lease.decision.moved is True
            assert lease.worker_url == target
            trace = (await rebalancer.snapshot())["recent_load_decisions"][-1]
            assert trace["scheduler"]["strategy"] == "online_dynamic_greedy"
            assert trace["scheduler"]["migrations"] == 1
            assert trace["step"]["threshold_met"] is True
            assert trace["decision"]["fallback_reason"] is None
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_owner_failure_uses_projected_load_without_threshold():
    client = ControlPlaneClient()
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
        await asyncio.gather(
            *(rebalancer._poll_engine_snapshot(url) for url in client.urls)
        )
        rebalancer.loads[source].healthy = False
        rebalancer.sessions["failed-owner"] = SessionRoutingState(
            owner_worker_url=source,
            fingerprint=fingerprint,
            previous_committed_tokens=[1] * 80,
        )

        lease = await rebalancer.acquire(
            session_id="failed-owner",
            input_ids=[1] * 100,
        )
        try:
            assert lease.worker_url == target
            assert lease.decision.moved is True
            trace = (await rebalancer.snapshot())["recent_load_decisions"][-1]
            step = trace["step"]
            assert step["decision_reason"] == "mandatory_failover"
        finally:
            await rebalancer.fail(lease)

    run(scenario())


def test_complete_commits_session_and_releases_reservation():
    client = ControlPlaneClient()
    rebalancer = EngineRebalancer(
        client,
        config=EngineRebalancingConfig(enabled=True),
        model_id="model",
        model_config=simple_model_config(),
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
            committed_tokens=[1] * 11,
        )

        session = rebalancer.sessions["background-observation"]
        assert lease.reservation_id not in rebalancer._reservations
        assert session.owner_worker_url == lease.worker_url
        assert session.previous_committed_tokens == [1] * 11
        assert rebalancer._online_request_count == 1

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
    assert args.engine_rebalancing_load_snapshot_poll_interval_ms == 60


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


def test_cli_accepts_snapshot_poll_interval(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dressage-proxy",
            "--tokenizer-path",
            "model",
            "--engine-rebalancing-load-snapshot-poll-interval-ms",
            "75",
        ],
    )

    assert parse_args().engine_rebalancing_load_snapshot_poll_interval_ms == 75


def test_cli_rejects_non_positive_snapshot_poll_interval(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dressage-proxy",
            "--tokenizer-path",
            "model",
            "--engine-rebalancing-load-snapshot-poll-interval-ms",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()
    assert "must be greater than 0" in capsys.readouterr().err


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
            json={"session_id": "s1"},
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
        assert loads["effective_config"]["load_snapshot_poll_interval_ms"] == 60
        assert loads["effective_config"]["history_size"] == 512
        assert loads["effective_config"]["min_load_improvement_ratio"] == 0.10
        assert loads["compatibility_pools"][0]["state"] in {
            "BOOTSTRAP",
            "ACTIVE",
        }
        engine_load = loads["engines"][0]
        assert {"running", "queued", "active_tokens"} <= set(engine_load)
        trace = loads["recent_load_decisions"][0]
        assert trace["step"]["target"] == client.calls[0]["worker_url"]
        assert trace["scheduler"]["strategy"] == "online_dynamic_greedy"
        assert "effective_pressure" in trace["engines"][0]
        assert "performance_models" not in loads
        assert "recent_context_observations" not in loads
        assert "recent_decisions" not in loads

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
        assert "state" not in payload


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


def test_single_node_l3_hicache_script_gates_engine_rebalancing():
    path = Path("examples/scripts/run_blackbox_qwen3.5_4b_sync_local_l3_hicache.sh")
    source = path.read_text()

    assert 'ENABLE_ENGINE_REBALANCING="${ENABLE_ENGINE_REBALANCING:-0}"' in source
    assert 'ENABLE_ENGINE_REBALANCING must be 0 or 1' in source
    assert (
        'if [[ "${ENABLE_ENGINE_REBALANCING}" == "1" ]]; then\n'
        '    PROXY_ARGS+=(--enable-engine-rebalancing)\n'
        "fi"
    ) in source
    calibration = source.index('CALIBRATION_STATE=""')
    gate = source.rindex(
        'if [[ "${ENABLE_ENGINE_REBALANCING}" == "1" ]]; then',
        0,
        calibration,
    )
    assert gate < calibration


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
        )

        await rebalancer.discard_session_context("discarded")
        await rebalancer.discard_session_context("discarded")

        assert "discarded" not in rebalancer.sessions

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
        )
        lease = await rebalancer.acquire(
            session_id="discard-before-settle",
            input_ids=[1] * 100,
        )
        assert lease.reservation_id in rebalancer._reservations
        queue_delta, token_delta = rebalancer._live_scoring_totals(lease.worker_url)
        assert queue_delta > 0
        assert token_delta > 0

        await rebalancer.discard_session_context("discard-before-settle")
        if settle_method == "complete":
            await rebalancer.complete(
                lease,
                committed_tokens=[1] * 101,
            )
        else:
            await rebalancer.fail(lease)

        assert "discard-before-settle" not in rebalancer.sessions
        assert (await rebalancer.snapshot())["active_sessions"] == 0
        assert lease.reservation_id not in rebalancer._reservations
        assert rebalancer._live_scoring_totals(lease.worker_url) == (0, 0)

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
            json={"session_id": "discarded-request"},
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
            json={"session_id": "registered-request"},
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
            json={"session_id": "discard-me"},
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
                worker_url = settle_leases[0].worker_url
                queue_delta, token_delta = rebalancer._live_scoring_totals(worker_url)
                assert queue_delta > 0
                assert token_delta > 0

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
        assert rebalancer._live_scoring_totals(worker_url) == (0, 0)

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

    assert "engine rebalancing lease settlement failed" in caplog.text


def test_complete_settlement_failure_does_not_change_generation_response(monkeypatch):
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
            raise RuntimeError("settlement boom")

        monkeypatch.setattr(app.state.engine_rebalancer, "complete", complete_settle)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
        ) as http_client:
            response = await http_client.post(
                "/v1/chat/completions",
                headers={"X-Session-ID": "settlement-fail"},
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "x"

    run(scenario())
