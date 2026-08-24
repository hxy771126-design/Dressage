from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

import httpx


def worker_urls_from_payload(payload: Mapping[str, Any]) -> list[str]:
    workers = payload.get("workers")
    if not isinstance(workers, list):
        workers = payload.get("engines")
    if not isinstance(workers, list):
        workers = payload.get("deployments")
    if not isinstance(workers, list):
        raise ValueError("topology payload must contain workers, engines, or deployments")

    urls: set[str] = set()
    for worker in workers:
        if not isinstance(worker, Mapping):
            continue
        if worker.get("is_healthy", worker.get("healthy", True)) is not True:
            continue
        if str(worker.get("connection_mode", "http")).lower() != "http":
            continue
        url = worker.get("url", worker.get("worker_url"))
        if isinstance(url, str) and url.strip():
            urls.add(url.rstrip("/"))
    if not urls:
        raise ValueError("topology payload contains no healthy HTTP workers")
    return sorted(urls)


def topology_sha256(worker_urls: list[str]) -> str:
    normalized = sorted({url.rstrip("/") for url in worker_urls})
    payload = json.dumps(normalized, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


async def validate_live_topology(
    metadata: Mapping[str, Any],
    *,
    client: httpx.AsyncClient | None = None,
) -> None:
    expected = metadata.get("target_topology_sha256")
    if expected is None:
        return
    if not isinstance(expected, str) or not expected:
        raise ValueError("metadata.target_topology_sha256 must be a non-empty string")

    router_url = os.environ.get("SGLANG_ROUTER_URL")
    if not router_url:
        raise ValueError(
            "SGLANG_ROUTER_URL is required for topology-bound repeat data"
        )
    owns_client = client is None
    effective_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        trust_env=False,
    )
    try:
        response = await effective_client.get(f"{router_url.rstrip('/')}/workers")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("SGLang /workers returned an invalid payload")
        actual = topology_sha256(worker_urls_from_payload(payload))
    finally:
        if owns_client:
            await effective_client.aclose()
    if actual != expected:
        raise ValueError(
            "topology fingerprint mismatch: "
            f"dataset={expected} current={actual}"
        )
