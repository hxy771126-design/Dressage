from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "data" / "prepare_repeat_multistep_64.py"
SOURCE = (
    REPO_ROOT
    / "examples"
    / "data"
    / "dressage_repeat_multistep_4k_52_256.jsonl"
)
NATURAL_64 = (
    REPO_ROOT
    / "examples"
    / "data"
    / "dressage_repeat_multistep_4k_52_64.jsonl"
)
BENCHMARK = (
    REPO_ROOT
    / "examples"
    / "scripts"
    / "benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh"
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_tool():
    spec = importlib.util.spec_from_file_location("prepare_repeat_multistep_64", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workers_payload(count: int = 8) -> dict:
    return {
        "workers": [
            {
                "url": f"http://worker-{index}:30000",
                "is_healthy": True,
                "connection_mode": "http",
            }
            for index in range(count)
        ]
    }


def test_natural_dataset_has_exact_64_trajectory_long_tail(tmp_path):
    output = tmp_path / "natural.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "natural",
            "--input",
            str(SOURCE),
            "--output",
            str(output),
            "--seed",
            "20260806",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = _read_jsonl(output)
    metadata = [row["metadata"] for row in rows]
    assert len(rows) == 64
    assert len({item["instance_id"] for item in metadata}) == 64
    assert Counter(item["planned_model_steps"] for item in metadata) == {
        1: 54,
        52: 10,
    }
    assert sum(item["planned_model_steps"] for item in metadata) == 574
    assert {item["payload_token_count"] for item in metadata} == {4096}

    repeated = tmp_path / "natural-repeated.jsonl"
    repeated_result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "natural",
            "--input",
            str(SOURCE),
            "--output",
            str(repeated),
            "--seed",
            "20260806",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated_result.returncode == 0, repeated_result.stderr
    assert repeated.read_bytes() == output.read_bytes()


def test_committed_natural_dataset_matches_generator(tmp_path):
    regenerated = tmp_path / "natural.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "natural",
            "--input",
            str(SOURCE),
            "--output",
            str(regenerated),
            "--seed",
            "20260806",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert regenerated.read_bytes() == NATURAL_64.read_bytes()
    assert hashlib.sha256(NATURAL_64.read_bytes()).hexdigest() == (
        "2c95d99fcfd5c28eaa017e317922870e6dc2e1af448c80c3bf6da5adaac6cf5f"
    )


def test_repeat_benchmark_dry_run_accepts_natural_batch_64(tmp_path):
    result = subprocess.run(
        ["bash", str(BENCHMARK)],
        cwd=REPO_ROOT,
        env={
            "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
            "BENCHMARK_DRY_RUN": "1",
            "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
            "BENCHMARK_WORKLOAD": "repeat_multistep",
            "BENCHMARK_PROMPT_DATA": str(NATURAL_64),
            "BENCHMARK_BATCH_SIZE": "64",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "rollout batch: 64" in result.stdout
    assert "global batch:  64" in result.stdout
    assert f"prompt source: {NATURAL_64}" in result.stdout
    assert not (tmp_path / "benchmark").exists()


def test_consistent_hash_ring_matches_sglang_blake3_vector():
    tool = _load_tool()
    workers = [f"http://worker-{index}:30000" for index in range(3)]
    ring = tool.ConsistentHashRing(workers)

    assert ring.worker_for("repeat-ms-test-0") == "http://worker-2:30000"
    assert ring.worker_for("user-123") == "http://worker-2:30000"
    assert ring.worker_for("session-abc-123") == "http://worker-0:30000"


def test_skew_dataset_has_exact_per_engine_workload(tmp_path):
    topology = tmp_path / "workers.json"
    topology.write_text(json.dumps(_workers_payload()), encoding="utf-8")
    output = tmp_path / "skew.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "skew",
            "--input",
            str(SOURCE),
            "--engine-topology-json",
            str(topology),
            "--output",
            str(output),
            "--seed",
            "20260806",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = _read_jsonl(output)
    metadata = [row["metadata"] for row in rows]
    assert len(rows) == 64
    assert len({item["instance_id"] for item in metadata}) == 64
    assert len({item["dressage_deterministic_session_id"] for item in metadata}) == 64
    assert Counter(item["expected_off_engine_index"] for item in metadata) == {
        index: 8 for index in range(8)
    }
    assert [
        sum(
            item["planned_model_steps"]
            for item in metadata
            if item["expected_off_engine_index"] == index
        )
        for index in range(8)
    ] == [163, 125, 112, 100, 88, 80, 70, 62]
    assert sum(item["planned_model_steps"] for item in metadata) == 800
    assert max(item["estimated_max_context_tokens"] for item in metadata) < 262144
    assert len({item["target_topology_sha256"] for item in metadata}) == 1

    tool = _load_tool()
    worker_urls = tool.worker_urls_from_payload(_workers_payload())
    ring = tool.ConsistentHashRing(worker_urls)
    for item in metadata:
        expected_url = worker_urls[item["expected_off_engine_index"]]
        assert ring.worker_for(item["dressage_deterministic_session_id"]) == expected_url


def test_verify_rejects_topology_mismatch(tmp_path):
    topology = tmp_path / "workers.json"
    topology.write_text(json.dumps(_workers_payload()), encoding="utf-8")
    dataset = tmp_path / "skew.jsonl"
    generated = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "skew",
            "--input",
            str(SOURCE),
            "--engine-topology-json",
            str(topology),
            "--output",
            str(dataset),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr

    changed_topology = tmp_path / "changed-workers.json"
    changed_topology.write_text(json.dumps(_workers_payload(7)), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            "--input",
            str(dataset),
            "--engine-topology-json",
            str(changed_topology),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "topology fingerprint mismatch" in result.stderr
