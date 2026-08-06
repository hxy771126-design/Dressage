"""Ray preflight backend for machine-level CUDA and Mooncake calibration.

This module deliberately imports Ray, Torch and Mooncake only inside the
runtime path.  A Proxy without those optional runtime pieces remains healthy
and reports a degraded calibration state instead of changing generation
semantics.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .model_cache_profile import canonical_fingerprint
from .transfer_calibrator import CalibrationSample, CalibrationTask


DEPLOYMENT_CONFIG_ENV = "DRESSAGE_ENGINE_REBALANCING_DEPLOYMENT_CONFIG"
_SUPPORTED_PROTOCOLS = {"tcp", "rdma"}


@dataclass(frozen=True)
class CalibrationNode:
    node_id: str
    gpu_count: int | None = None
    gpu_ids: tuple[int, ...] = ()
    numa_node: str = ""
    nic: str = ""


@dataclass(frozen=True)
class MachineCalibrationConfig:
    schema_version: int
    ray_address: str
    nodes: tuple[CalibrationNode, ...]
    shared_l3: bool
    write_policy: str
    protocol: str
    device_name: str
    metadata_server: str
    gpudirect: bool
    model_config_path: str
    model_deployment: dict[str, Any]
    connect_timeout_seconds: float = 120.0
    task_timeout_seconds: float = 120.0

    @property
    def host_staging(self) -> bool:
        return self.shared_l3 and not self.gpudirect

    @property
    def buffer_registration_mode(self) -> str:
        return "cuda" if self.gpudirect else "host_pinned"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MachineCalibrationConfig":
        schema_version = int(payload.get("schema_version") or 0)
        if schema_version != 1:
            raise ValueError("engine rebalancing deployment schema_version must be 1")
        ray_address = str(payload.get("ray_address") or "auto").strip()
        if not ray_address:
            raise ValueError("ray_address must not be empty")

        raw_nodes = payload.get("nodes") or []
        if not isinstance(raw_nodes, list):
            raise ValueError("nodes must be a list")
        nodes: list[CalibrationNode] = []
        for item in raw_nodes:
            if not isinstance(item, Mapping):
                raise ValueError("each calibration node must be an object")
            node_id = str(item.get("node_id") or item.get("address") or "").strip()
            if not node_id:
                raise ValueError("each calibration node requires node_id")
            raw_gpu_count = item.get("gpu_count")
            gpu_count = None if raw_gpu_count is None else int(raw_gpu_count)
            if gpu_count is not None and gpu_count <= 0:
                raise ValueError("node gpu_count must be positive")
            raw_gpu_ids = item.get("gpu_ids") or []
            if not isinstance(raw_gpu_ids, list):
                raise ValueError("node gpu_ids must be a list")
            gpu_ids = tuple(int(value) for value in raw_gpu_ids)
            if any(value < 0 for value in gpu_ids):
                raise ValueError("node gpu_ids must be non-negative")
            if gpu_count is not None and gpu_ids and len(gpu_ids) != gpu_count:
                raise ValueError("node gpu_ids length must equal gpu_count")
            nodes.append(
                CalibrationNode(
                    node_id,
                    gpu_count,
                    gpu_ids,
                    str(item.get("numa_node") or ""),
                    str(item.get("nic") or ""),
                )
            )

        hicache = payload.get("hicache") or {}
        mooncake = payload.get("mooncake") or {}
        if not isinstance(hicache, Mapping) or not isinstance(mooncake, Mapping):
            raise ValueError("hicache and mooncake must be objects")
        shared_l3 = (
            bool(hicache.get("enabled"))
            and str(hicache.get("storage_backend") or "").lower() == "mooncake"
        )
        write_policy = str(hicache.get("write_policy") or "").lower()
        gpudirect = bool(hicache.get("gpudirect") or mooncake.get("gpudirect"))
        protocol = str(mooncake.get("protocol") or "").lower()
        device_name = str(mooncake.get("device_name") or "").strip()
        if shared_l3:
            if write_policy != "write_through":
                raise ValueError(
                    "preflight restore calibration requires HiCache write_through"
                )
            if protocol not in _SUPPORTED_PROTOCOLS:
                raise ValueError(
                    "Mooncake protocol must be one of "
                    f"{sorted(_SUPPORTED_PROTOCOLS)}, got {protocol!r}"
                )
            if gpudirect and protocol != "rdma":
                raise ValueError("Mooncake GPUDirect calibration requires RDMA")
            if protocol == "rdma" and not device_name:
                raise ValueError("Mooncake RDMA calibration requires device_name")
        metadata_server = str(
            mooncake.get("metadata_server")
            or mooncake.get("metadata_conn_string")
            or ""
        ).strip()
        if shared_l3 and not metadata_server:
            raise ValueError("Mooncake metadata_server is required when L3 is enabled")
        model_deployment = payload.get("model_deployment") or {}
        if not isinstance(model_deployment, Mapping):
            raise ValueError("model_deployment must be an object")
        return cls(
            schema_version=schema_version,
            ray_address=ray_address,
            nodes=tuple(nodes),
            shared_l3=shared_l3,
            write_policy=write_policy,
            protocol=protocol,
            device_name=device_name,
            metadata_server=metadata_server,
            gpudirect=gpudirect,
            model_config_path=str(payload.get("model_config_path") or ""),
            model_deployment=dict(model_deployment),
            connect_timeout_seconds=max(
                1.0, float(payload.get("connect_timeout_seconds") or 120.0)
            ),
            task_timeout_seconds=max(
                1.0, float(payload.get("task_timeout_seconds") or 120.0)
            ),
        )

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "MachineCalibrationConfig":
        config_path = Path(path).expanduser()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid engine rebalancing deployment config {config_path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError("engine rebalancing deployment config must be an object")
        return cls.from_mapping(payload)

    def fingerprint(self, discovered_nodes: list[dict[str, Any]]) -> str:
        payload = asdict(self)
        payload.pop("connect_timeout_seconds", None)
        payload.pop("task_timeout_seconds", None)
        # Results are indexed by the actual payload byte count. Keep the
        # reusable machine-path fingerprint independent of model/weight data.
        payload.pop("model_deployment", None)
        payload.pop("model_config_path", None)
        payload["buffer_registration_mode"] = self.buffer_registration_mode
        payload["configured_nodes"] = payload.pop("nodes", [])
        payload["nodes"] = sorted(
            discovered_nodes,
            key=lambda item: (str(item.get("node_id")), str(item.get("address"))),
        )
        return canonical_fingerprint(payload)


def load_machine_calibration_config_from_env() -> MachineCalibrationConfig | None:
    path = os.environ.get(DEPLOYMENT_CONFIG_ENV)
    if path is None or not path.strip():
        return None
    return MachineCalibrationConfig.from_file(path.strip())


@dataclass(frozen=True)
class PlannedEngineSlot:
    node_id: str
    mooncake_protocol: str


class _CalibrationActor:
    """Node-local worker. Instantiated through ``ray.remote`` at runtime."""

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, int], tuple[Any, Any]] = {}
        self._engines: dict[tuple[str, str, str], Any] = {}
        self._registered: list[tuple[Any, int]] = []

    @staticmethod
    def _torch() -> Any:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in calibration actor")
        return torch

    @staticmethod
    def _node_address() -> str:
        try:
            from ray.util import get_node_ip_address

            return str(get_node_ip_address())
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    @classmethod
    def _server_name(cls) -> str:
        host = cls._node_address()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind((host, 0))
            port = int(handle.getsockname()[1])
        return f"{host}:{port}"

    def identity(self) -> dict[str, Any]:
        torch = self._torch()
        properties = torch.cuda.get_device_properties(0)
        driver_version = getattr(
            getattr(torch, "_C", object()),
            "_cuda_getDriverVersion",
            lambda: "unknown",
        )()
        pci_bus_id = str(getattr(properties, "pci_bus_id", "") or "")
        numa_node = ""
        if pci_bus_id:
            try:
                numa_node = (
                    Path(f"/sys/bus/pci/devices/{pci_bus_id}/numa_node")
                    .read_text(encoding="utf-8")
                    .strip()
                )
            except OSError:
                pass
        mooncake_version = "unknown"
        for distribution in (
            "mooncake-transfer-engine",
            "mooncake_transfer_engine",
            "mooncake",
        ):
            try:
                mooncake_version = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        return {
            "hostname": socket.gethostname(),
            "address": self._node_address(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_uuid": str(getattr(properties, "uuid", "") or ""),
            "pci_bus_id": pci_bus_id,
            "numa_node": numa_node,
            "cuda_version": str(torch.version.cuda or ""),
            "driver_version": str(driver_version),
            "mooncake_version": mooncake_version,
        }

    def _buffers_for_copy(self, payload: int) -> tuple[Any, Any]:
        key = ("copy", 0)
        existing = self._buffers.get(key)
        if existing is None or int(existing[0].nbytes) < payload:
            torch = self._torch()
            host = torch.empty(payload, dtype=torch.uint8, pin_memory=True)
            gpu = torch.empty(payload, dtype=torch.uint8, device="cuda")
            self._buffers[key] = (host, gpu)
        return self._buffers[key]

    def measure_copy(self, link_type: str, payload: int) -> CalibrationSample:
        torch = self._torch()
        host, gpu = self._buffers_for_copy(payload)
        torch.cuda.synchronize()
        started = time.perf_counter()
        if link_type == "h2d":
            gpu[:payload].copy_(host[:payload], non_blocking=True)
        elif link_type == "d2h":
            host[:payload].copy_(gpu[:payload], non_blocking=True)
        else:
            raise ValueError(f"unsupported CUDA copy link {link_type!r}")
        torch.cuda.synchronize()
        elapsed = max(1e-9, time.perf_counter() - started)
        return CalibrationSample(
            elapsed_seconds_p75=elapsed,
            bandwidth_bytes_per_second_p25=payload / elapsed,
            payload_bytes=payload,
        )

    def _engine(
        self,
        *,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ) -> tuple[Any, str]:
        key = (metadata_server, protocol, device_name)
        cached = self._engines.get(key)
        if cached is not None:
            return cached
        from mooncake.engine import TransferEngine

        engine = TransferEngine()
        local_server_name = self._server_name()
        result = engine.initialize(
            local_server_name,
            metadata_server,
            protocol,
            device_name,
        )
        if result != 0:
            raise RuntimeError(f"Mooncake TransferEngine initialize failed: {result}")
        if metadata_server == "P2PHANDSHAKE":
            host = local_server_name.rpartition(":")[0]
            local_server_name = f"{host}:{engine.get_rpc_port()}"
        value = (engine, local_server_name)
        self._engines[key] = value
        return value

    def _registered_buffer(
        self,
        *,
        payload: int,
        use_cuda: bool,
        metadata_server: str,
        protocol: str,
        device_name: str,
        role: str,
    ) -> tuple[Any, int, str, Any]:
        del role
        key = (f"mooncake:{int(use_cuda)}", 0)
        engine, server_name = self._engine(
            metadata_server=metadata_server,
            protocol=protocol,
            device_name=device_name,
        )
        existing = self._buffers.get(key)
        if existing is None or int(existing[0].nbytes) < payload:
            if existing is not None:
                old_tensor, old_engine = existing
                old_address = old_tensor.data_ptr()
                try:
                    old_engine.unregister_memory(old_address)
                except Exception:
                    pass
                finally:
                    self._registered = [
                        item for item in self._registered if item[1] != old_address
                    ]
            torch = self._torch()
            tensor = (
                torch.empty(payload, dtype=torch.uint8, device="cuda")
                if use_cuda
                else torch.empty(payload, dtype=torch.uint8, pin_memory=True)
            )
            result = engine.register_memory(tensor.data_ptr(), tensor.nbytes)
            if result != 0:
                raise RuntimeError(f"Mooncake register_memory failed: {result}")
            self._buffers[key] = (tensor, engine)
            self._registered.append((engine, tensor.data_ptr()))
        tensor, _ = self._buffers[key]
        return tensor, tensor.data_ptr(), server_name, engine

    def prepare_mooncake_source(
        self,
        payload: int,
        config: Mapping[str, Any],
        use_cuda: bool,
    ) -> dict[str, Any]:
        tensor, address, server_name, _ = self._registered_buffer(
            payload=payload,
            use_cuda=use_cuda,
            metadata_server=str(config["metadata_server"]),
            protocol=str(config["protocol"]),
            device_name=str(config.get("device_name") or ""),
            role="source",
        )
        tensor.zero_()
        if use_cuda:
            self._torch().cuda.synchronize()
        return {"server_name": server_name, "address": address}

    def measure_mooncake_read(
        self,
        payload: int,
        config: Mapping[str, Any],
        remote_source: Mapping[str, Any],
        use_cuda: bool,
    ) -> CalibrationSample:
        torch = self._torch()
        _, local_address, _, engine = self._registered_buffer(
            payload=payload,
            use_cuda=use_cuda,
            metadata_server=str(config["metadata_server"]),
            protocol=str(config["protocol"]),
            device_name=str(config.get("device_name") or ""),
            role="target",
        )
        if use_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = engine.transfer_sync_read(
            str(remote_source["server_name"]),
            local_address,
            int(remote_source["address"]),
            payload,
        )
        if result != 0:
            raise RuntimeError(f"Mooncake transfer_sync_read failed: {result}")
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = max(1e-9, time.perf_counter() - started)
        return CalibrationSample(
            elapsed_seconds_p75=elapsed,
            bandwidth_bytes_per_second_p25=payload / elapsed,
            payload_bytes=payload,
        )

    def release_link(self, link_type: str) -> None:
        if link_type in {"h2d", "d2h"}:
            for key in [key for key in self._buffers if key[0] == "copy"]:
                self._buffers.pop(key, None)
        else:
            for engine, address in reversed(self._registered):
                try:
                    engine.unregister_memory(address)
                except Exception:
                    pass
            self._registered.clear()
            for key in [
                key for key in self._buffers if str(key[0]).startswith("mooncake:")
            ]:
                self._buffers.pop(key, None)
        try:
            self._torch().cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        for engine, address in reversed(self._registered):
            try:
                engine.unregister_memory(address)
            except Exception:
                pass
        self._registered.clear()
        self._buffers.clear()
        for engine, _ in self._engines.values():
            close = getattr(engine, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self._engines.clear()


class RayTransferBenchmark:
    """Callable benchmark used by :class:`TransferCalibrator`."""

    def __init__(self, config: MachineCalibrationConfig) -> None:
        self.config = config
        self.ray: Any | None = None
        self._owns_ray = False
        self._nodes: dict[str, dict[str, Any]] = {}
        self._actors: list[Any] = []
        self._actor_pools: dict[str, list[Any]] = {}
        self._sample_indices: dict[tuple[CalibrationTask, int], int] = {}
        self._baseline_available_gpus = 0.0
        self.hardware: list[dict[str, Any]] = []

    async def connect(self) -> list[dict[str, Any]]:
        import ray

        self.ray = ray
        self._owns_ray = not ray.is_initialized()

        deadline = time.monotonic() + self.config.connect_timeout_seconds
        last_error: Exception | None = None
        while not ray.is_initialized() and time.monotonic() < deadline:
            try:
                await asyncio.to_thread(
                    ray.init,
                    address=self.config.ray_address,
                    ignore_reinit_error=True,
                    namespace="dressage-engine-rebalancing-calibration",
                    logging_level="ERROR",
                )
            except Exception as exc:  # Ray may not have started yet.
                last_error = exc
                await asyncio.sleep(0.5)
        if not ray.is_initialized():
            raise RuntimeError(
                "timed out waiting for Ray calibration cluster"
            ) from last_error
        rows = [
            row
            for row in ray.nodes()
            if row.get("Alive") and float(row.get("Resources", {}).get("GPU", 0)) > 0
        ]
        requested = {node.node_id for node in self.config.nodes}
        if requested:
            rows = [
                row
                for row in rows
                if str(row.get("NodeID")) in requested
                or str(row.get("NodeManagerAddress")) in requested
            ]
        if not rows:
            raise RuntimeError("Ray has no matching live GPU nodes for calibration")
        for requested_node in self.config.nodes:
            matches = [
                row
                for row in rows
                if str(row.get("NodeID")) == requested_node.node_id
                or str(row.get("NodeManagerAddress")) == requested_node.node_id
            ]
            if not matches:
                raise RuntimeError(
                    f"configured calibration node {requested_node.node_id!r} "
                    "is not live in Ray"
                )
            available = int(float(matches[0].get("Resources", {}).get("GPU", 0)))
            if (
                requested_node.gpu_count is not None
                and available < requested_node.gpu_count
            ):
                raise RuntimeError(
                    f"calibration node {requested_node.node_id!r} reports "
                    f"{available} GPUs, expected at least "
                    f"{requested_node.gpu_count}"
                )
        self._nodes = {str(row["NodeManagerAddress"]): row for row in rows}
        self._baseline_available_gpus = float(ray.available_resources().get("GPU", 0.0))
        discovered = [
            {
                "node_id": str(row["NodeID"]),
                "address": str(row["NodeManagerAddress"]),
                "gpu_count": int(float(row.get("Resources", {}).get("GPU", 0))),
                "resources": dict(row.get("Resources", {})),
            }
            for row in rows
        ]
        hardware = await asyncio.to_thread(self._create_actor_pools_sync)
        for item in discovered:
            item["hardware"] = hardware.get(item["address"], [])
        return discovered

    def planned_engine_slots(self) -> list[PlannedEngineSlot]:
        slots: list[PlannedEngineSlot] = []
        for address, actors in self._actor_pools.items():
            for _ in actors:
                slots.append(PlannedEngineSlot(address, self.config.protocol))
        return slots

    def _new_actor(self, node: str) -> Any:
        assert self.ray is not None
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        row = self._nodes[node]
        actor_class = self.ray.remote(num_gpus=1)(_CalibrationActor)
        actor = actor_class.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=str(row["NodeID"]), soft=False
            )
        ).remote()
        self._actors.append(actor)
        return actor

    def _create_actor_pools_sync(self) -> dict[str, list[dict[str, Any]]]:
        hardware: dict[str, list[dict[str, Any]]] = {}
        for address, row in self._nodes.items():
            count = int(float(row.get("Resources", {}).get("GPU", 0)))
            self._actor_pools[address] = [
                self._new_actor(address) for _ in range(max(1, count))
            ]
        for address, actors in self._actor_pools.items():
            identities = self._ray_get([actor.identity.remote() for actor in actors])
            hardware[address] = [dict(item) for item in identities]
        self.hardware = [dict(item) for values in hardware.values() for item in values]
        return hardware

    def _ray_get(self, value: Any) -> Any:
        assert self.ray is not None
        return self.ray.get(value, timeout=self.config.task_timeout_seconds)

    def _close_actors_sync(self) -> None:
        if self.ray is None:
            return
        for actor in self._actors:
            try:
                self._ray_get(actor.close.remote())
            except Exception:
                pass
            try:
                self.ray.kill(actor, no_restart=True)
            except Exception:
                pass
        self._actors.clear()
        self._actor_pools.clear()
        self._sample_indices.clear()

    async def __call__(self, task: CalibrationTask, payload: int) -> CalibrationSample:
        def measure() -> CalibrationSample:
            sample_key = (task, int(payload))
            sample_index = self._sample_indices.get(sample_key, 0)
            self._sample_indices[sample_key] = sample_index + 1
            if task.link_type in {"h2d", "d2h"}:
                node = task.target_node if task.link_type == "h2d" else task.source_node
                pool = self._actor_pools[node]
                actor = pool[sample_index % len(pool)]
                return self._ray_get(actor.measure_copy.remote(task.link_type, payload))
            use_cuda = task.link_type == "mooncake_gpudirect"
            source_pool = self._actor_pools[task.source_node]
            target_pool = self._actor_pools[task.target_node]
            source_index = sample_index % len(source_pool)
            target_index = sample_index % len(target_pool)
            if task.source_node == task.target_node:
                if len(target_pool) < 2:
                    raise RuntimeError(
                        "node-local Mooncake migration requires at least two GPUs"
                    )
                target_index = (source_index + 1) % len(target_pool)
            source_actor = source_pool[source_index]
            target_actor = target_pool[target_index]
            remote_source = self._ray_get(
                source_actor.prepare_mooncake_source.remote(
                    payload,
                    {
                        # Preflight intentionally measures TransferEngine data
                        # movement without Mooncake Store metadata. Runtime
                        # Store/HiCache overhead is learned by the online
                        # context residual model.
                        "metadata_server": "P2PHANDSHAKE",
                        "protocol": self.config.protocol,
                        "device_name": self.config.device_name,
                    },
                    use_cuda,
                )
            )
            return self._ray_get(
                target_actor.measure_mooncake_read.remote(
                    payload,
                    {
                        "metadata_server": "P2PHANDSHAKE",
                        "protocol": self.config.protocol,
                        "device_name": self.config.device_name,
                    },
                    remote_source,
                    use_cuda,
                )
            )

        return await asyncio.to_thread(measure)

    async def finish_task(self, task: CalibrationTask) -> None:
        def release() -> None:
            self._ray_get(
                [actor.release_link.remote(task.link_type) for actor in self._actors]
            )

        await asyncio.to_thread(release)

    async def close(self) -> bool:
        await asyncio.to_thread(self._close_actors_sync)
        resources_recovered = True
        if self.ray is not None:
            deadline = time.monotonic() + min(30.0, self.config.task_timeout_seconds)
            while time.monotonic() < deadline:
                available = float(self.ray.available_resources().get("GPU", 0.0))
                if available + 0.01 >= self._baseline_available_gpus:
                    break
                await asyncio.sleep(0.25)
            else:
                resources_recovered = False
            if self._owns_ray:
                await asyncio.to_thread(self.ray.shutdown)
        return resources_recovered
