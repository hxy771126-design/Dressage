#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from dressage.recipes.repeat_multistep.topology import (
    topology_sha256,
    worker_urls_from_payload,
)


NATURAL_COUNTS = {1: 54, 52: 10}
SKEW_STEP_GROUPS = (
    (32, 28, 24, 22, 20, 16, 12, 9),
    (24, 20, 18, 16, 14, 12, 11, 10),
    (20, 18, 16, 14, 12, 11, 11, 10),
    (18, 16, 14, 12, 11, 10, 10, 9),
    (16, 14, 12, 11, 10, 9, 8, 8),
    (14, 12, 11, 10, 9, 8, 8, 8),
    (12, 10, 9, 9, 8, 8, 7, 7),
    (10, 9, 8, 8, 7, 7, 7, 6),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("input JSONL rows must be objects")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _instance_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("instance_id") is None:
        raise ValueError("every source row must have metadata.instance_id")
    return str(metadata["instance_id"])


def _stable_order(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{_instance_id(row)}".encode()
        ).digest(),
    )


class ConsistentHashRing:
    def __init__(self, worker_urls: list[str]) -> None:
        try:
            import blake3
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "skew generation requires the data extra: pip install 'dressage[data]'"
            ) from exc
        self._blake3 = blake3
        self._entries: list[tuple[int, str]] = []
        for worker_url in worker_urls:
            for vnode in range(150):
                digest = blake3.blake3(
                    worker_url.encode() + b"#" + vnode.to_bytes(8, "little")
                ).digest()
                self._entries.append((int.from_bytes(digest[:8], "little"), worker_url))
        self._entries.sort()
        self._positions = [position for position, _ in self._entries]

    def worker_for(self, key: str) -> str:
        digest = self._blake3.blake3(key.encode()).digest()
        position = int.from_bytes(digest[:8], "little")
        index = bisect.bisect_left(self._positions, position)
        return self._entries[index % len(self._entries)][1]


def _natural_rows(source: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for planned_steps, count in NATURAL_COUNTS.items():
        candidates = [
            row
            for row in source
            if row.get("metadata", {}).get("planned_model_steps") == planned_steps
        ]
        if len(candidates) < count:
            raise ValueError(
                f"not enough {planned_steps}-step rows: need {count}, found {len(candidates)}"
            )
        selected.extend(_stable_order(candidates, seed)[:count])
    return _stable_order(selected, f"output:{seed}")


def _context_profile(source: list[dict[str, Any]]) -> tuple[int, int]:
    short_overheads: list[int] = []
    long_overheads: list[int] = []
    for row in source:
        metadata = row.get("metadata", {})
        steps = metadata.get("planned_model_steps")
        estimate = metadata.get("estimated_max_context_tokens")
        user_tokens = metadata.get("user_content_token_count")
        if not isinstance(estimate, int) or not isinstance(user_tokens, int):
            continue
        overhead = estimate - user_tokens
        if steps == 1:
            short_overheads.append(overhead)
        elif steps == 52:
            long_overheads.append(overhead)
    if not short_overheads or not long_overheads:
        raise ValueError("source data must contain 1-step and 52-step context estimates")
    base_overhead = round(statistics.median(short_overheads))
    step_increment = round(
        (statistics.median(long_overheads) - statistics.median(short_overheads)) / 51
    )
    return base_overhead, step_increment


def _session_ids_by_worker(
    ring: ConsistentHashRing,
    worker_urls: list[str],
    seed: str,
) -> list[list[str]]:
    result = [[] for _ in worker_urls]
    worker_indexes = {url: index for index, url in enumerate(worker_urls)}
    counter = 0
    while any(len(session_ids) < 8 for session_ids in result):
        session_id = f"repeat-ms-tail-{seed}-{counter:08d}"
        worker_index = worker_indexes[ring.worker_for(session_id)]
        if len(result[worker_index]) < 8:
            result[worker_index].append(session_id)
        counter += 1
    return result


def _skew_rows(
    source: list[dict[str, Any]],
    worker_urls: list[str],
    seed: str,
) -> list[dict[str, Any]]:
    if len(worker_urls) != 8:
        raise ValueError(f"skew generation requires exactly 8 workers, found {len(worker_urls)}")
    if len(source) < 64:
        raise ValueError(f"skew generation requires at least 64 source rows, found {len(source)}")
    ring = ConsistentHashRing(worker_urls)
    session_ids = _session_ids_by_worker(ring, worker_urls, seed)
    base_overhead, step_increment = _context_profile(source)
    selected = _stable_order(source, seed)[:64]
    fingerprint = topology_sha256(worker_urls)
    rows: list[dict[str, Any]] = []
    source_index = 0
    for engine_index, planned_steps in enumerate(SKEW_STEP_GROUPS):
        for slot, steps in enumerate(planned_steps):
            row = copy.deepcopy(selected[source_index])
            source_index += 1
            metadata = row["metadata"]
            instance_id = f"repeat_4k_tail_skew_e{engine_index}_{slot:02d}"
            metadata.update(
                {
                    "instance_id": instance_id,
                    "planned_model_steps": steps,
                    "workload_profile": "repeat_4k_tail_skew_64",
                    "workload_profile_version": "repeat-tail-skew-v1",
                    "workload_assignment_seed": seed,
                    "dressage_deterministic_session_id": session_ids[engine_index][slot],
                    "expected_off_engine_index": engine_index,
                    "target_topology_sha256": fingerprint,
                    "estimated_max_context_tokens": (
                        int(metadata["user_content_token_count"])
                        + base_overhead
                        + (steps - 1) * step_increment
                    ),
                }
            )
            rows.append(row)
    return _stable_order(rows, f"output:{seed}")


def _verify(rows: list[dict[str, Any]], worker_urls: list[str]) -> None:
    actual_fingerprint = topology_sha256(worker_urls)
    fingerprints = {
        row.get("metadata", {}).get("target_topology_sha256") for row in rows
    }
    if fingerprints != {actual_fingerprint}:
        raise ValueError(
            "topology fingerprint mismatch: "
            f"dataset={sorted(map(str, fingerprints))} current={actual_fingerprint}"
        )
    ring = ConsistentHashRing(worker_urls)
    for row in rows:
        metadata = row["metadata"]
        expected_url = worker_urls[int(metadata["expected_off_engine_index"])]
        session_id = str(metadata["dressage_deterministic_session_id"])
        if ring.worker_for(session_id) != expected_url:
            raise ValueError(f"session {session_id} does not map to {expected_url}")


def _topology(path: Path) -> list[str]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("engine topology JSON must be an object")
    return worker_urls_from_payload(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    natural = subparsers.add_parser("natural")
    natural.add_argument("--input", type=Path, required=True)
    natural.add_argument("--output", type=Path, required=True)
    natural.add_argument("--seed", default="20260806")

    skew = subparsers.add_parser("skew")
    skew.add_argument("--input", type=Path, required=True)
    skew.add_argument("--engine-topology-json", type=Path, required=True)
    skew.add_argument("--output", type=Path, required=True)
    skew.add_argument("--seed", default="20260806")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--engine-topology-json", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "natural":
        _write_jsonl(args.output, _natural_rows(_read_jsonl(args.input), args.seed))
    elif args.command == "skew":
        worker_urls = _topology(args.engine_topology_json)
        rows = _skew_rows(_read_jsonl(args.input), worker_urls, args.seed)
        _verify(rows, worker_urls)
        _write_jsonl(args.output, rows)
    else:
        _verify(_read_jsonl(args.input), _topology(args.engine_topology_json))


if __name__ == "__main__":
    main()
