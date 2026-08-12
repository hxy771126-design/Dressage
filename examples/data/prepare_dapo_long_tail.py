#!/usr/bin/env python3
"""Build and deterministically sample the DAPO long-tail benchmark dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import tempfile
from collections import Counter


BUILD_COUNTS = {"short": 2100, "medium": 600, "long": 300}
SAMPLE_COUNTS = {"short": 179, "medium": 51, "long": 26}
TOOL_CALLS = {"short": 1, "medium": 5, "long": 15}
PROFILE_VERSION = "dapo-long-tail-v1"


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = pathlib.Path(handle.name)
    os.replace(temporary_path, path)


def _instance_id(row: dict) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("instance_id"), str):
        raise ValueError("row is missing metadata.instance_id")
    return metadata["instance_id"]


def _rank(seed: str, instance_id: str, purpose: str) -> str:
    return hashlib.sha256(f"{purpose}:{seed}:{instance_id}".encode()).hexdigest()


def _workflow(tool_calls: int) -> str:
    commands = ["`mktemp /tmp/dressage-step.XXXXXX`"]
    if tool_calls >= 2:
        commands.append("`date +%s%N > <PATH>`")
    commands.extend("`cat <PATH>`" for _ in range(tool_calls - 2))
    steps = []
    for index, command in enumerate(commands, start=1):
        if index == 1:
            suffix = " Remember the returned filename as <PATH>."
            steps.append(f"{index}. In one bash tool call, run exactly {command}.{suffix}")
        else:
            steps.append(
                f"{index}. Only after the previous result is available, make one new "
                f"bash tool call and run exactly {command}, replacing <PATH> with the "
                "filename returned by the first call."
            )
    return (
        "\n\nBefore solving the original math problem, complete the following mandatory "
        f"tool workflow. This session requires exactly {tool_calls} sequential bash "
        f"tool call(s), producing an expected total of {tool_calls + 1} model steps "
        "including the final answer.\n\nTool workflow:\n"
        + "\n".join(steps)
        + "\n\nWorkflow rules:\n"
        "- During the tool workflow, make exactly one tool call in each assistant response.\n"
        "- Wait for each tool result before deciding and issuing the next tool call.\n"
        "- Do not issue parallel tool calls.\n"
        "- Do not combine workflow commands with `;`, `&&`, pipelines, or scripts.\n"
        "- Do not skip, reorder, or add tool calls.\n"
        "- After all required tool calls finish, make no more tool calls. Solve the "
        "original math problem and provide the required boxed answer."
    )


def build(input_path: pathlib.Path, output_path: pathlib.Path, seed: str) -> None:
    rows = _read_jsonl(input_path)
    if len(rows) != sum(BUILD_COUNTS.values()):
        raise ValueError(f"expected 3000 source rows, found {len(rows)}")
    ids = [_instance_id(row) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("source instance_id values must be unique")

    ranked_ids = sorted(ids, key=lambda value: _rank(seed, value, "class"))
    assignments: dict[str, str] = {}
    offset = 0
    for workload_class, count in BUILD_COUNTS.items():
        for instance_id in ranked_ids[offset : offset + count]:
            assignments[instance_id] = workload_class
        offset += count

    generated = []
    for source in rows:
        row = copy.deepcopy(source)
        instance_id = _instance_id(row)
        workload_class = assignments[instance_id]
        tool_calls = TOOL_CALLS[workload_class]
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt or not isinstance(prompt[0].get("content"), str):
            raise ValueError(f"instance {instance_id} has an invalid prompt")
        prompt[0]["content"] += _workflow(tool_calls)
        row["blackbox_type"] = "opencode"
        row["metadata"]["workload_class"] = workload_class
        row["metadata"]["planned_tool_calls"] = tool_calls
        row["metadata"]["workload_profile_version"] = PROFILE_VERSION
        row["metadata"]["workload_assignment_seed"] = seed
        generated.append(row)

    _write_jsonl(output_path, generated)


def _determinize(row: dict, seed: str) -> None:
    instance_id = _instance_id(row)
    digest = hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()
    path = f"/tmp/dressage-step-{digest[:16]}"
    timestamp = 1_700_000_000_000_000_000 + int(digest[16:28], 16) % 1_000_000_000_000
    for message in row["prompt"]:
        content = message["content"]
        content = content.replace(
            "mktemp /tmp/dressage-step.XXXXXX",
            f"LC_ALL=C install -v -m 600 /dev/null {path}",
        )
        content = content.replace("date +%s%N > <PATH>", f"printf '%s\\n' '{timestamp}' > {path}")
        content = content.replace("cat <PATH>", f"cat {path}")
        content = content.replace(
            "Remember the returned filename as <PATH>.",
            f"Use the deterministic filename `{path}` as <PATH>.",
        )
        content = content.replace(
            "replacing <PATH> with the filename returned by the first call.",
            f"replacing <PATH> with the deterministic filename `{path}`.",
        )
        message["content"] = content


def sample(input_path: pathlib.Path, output_path: pathlib.Path, seed: str) -> None:
    rows = _read_jsonl(input_path)
    by_class: dict[str, list[dict]] = {name: [] for name in SAMPLE_COUNTS}
    for row in rows:
        workload_class = row.get("metadata", {}).get("workload_class")
        if workload_class not in by_class:
            raise ValueError(f"invalid workload_class {workload_class!r}")
        by_class[workload_class].append(row)

    selected = []
    for workload_class, count in SAMPLE_COUNTS.items():
        candidates = sorted(
            by_class[workload_class],
            key=lambda row: _rank(seed, _instance_id(row), "sample"),
        )
        if len(candidates) < count:
            raise ValueError(f"not enough {workload_class} rows")
        selected.extend(copy.deepcopy(candidates[:count]))
    selected.sort(key=lambda row: _rank(seed, _instance_id(row), "order"))
    for row in selected:
        _determinize(row, seed)
    _write_jsonl(output_path, selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "sample"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--input", type=pathlib.Path, required=True)
        subparser.add_argument("--output", type=pathlib.Path, required=True)
        subparser.add_argument("--seed", default="20260806")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if args.command == "build":
        build(args.input, args.output, args.seed)
    else:
        sample(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
