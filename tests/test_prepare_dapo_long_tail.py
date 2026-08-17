import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


SCRIPT = Path("examples/data/prepare_dapo_long_tail.py")
SOURCE = Path("examples/data/dressage_dapo_prompts.jsonl")
STEP_BALANCED_300 = Path(
    "examples/data/dressage_dapo_prompts_step_balanced_300.jsonl"
)
STEP_BALANCED_256 = Path(
    "examples/data/dressage_dapo_prompts_step_balanced_256.jsonl"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_preserves_source_and_creates_exact_workload_classes(tmp_path):
    output = tmp_path / "long-tail.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "build",
            "--input",
            str(SOURCE),
            "--output",
            str(output),
            "--seed",
            "20260806",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    source_rows = read_jsonl(SOURCE)
    output_rows = read_jsonl(output)
    assert len(source_rows) == len(output_rows) == 3000
    assert Counter(row["metadata"]["workload_class"] for row in output_rows) == {
        "short": 2100,
        "medium": 600,
        "long": 300,
    }
    assert Counter(row["metadata"]["planned_tool_calls"] for row in output_rows) == {
        1: 2100,
        5: 600,
        15: 300,
    }
    for source, generated in zip(source_rows, output_rows, strict=True):
        for key in ("label", "reward_fn", "agent_mode", "task_type"):
            assert generated[key] == source[key]
        assert generated["blackbox_type"] == "opencode"
        assert generated["metadata"]["instance_id"] == source["metadata"]["instance_id"]
        original = source["prompt"][0]["content"]
        generated_content = generated["prompt"][0]["content"]
        assert generated_content.startswith(original)
        calls = generated["metadata"]["planned_tool_calls"]
        assert f"exactly {calls} sequential bash tool call(s)" in generated_content
        assert generated_content.count("make one new bash tool call") == calls - 1


def test_sample_selects_exact_strata_and_determinizes_commands(tmp_path):
    dataset = tmp_path / "long-tail.jsonl"
    sample = tmp_path / "sample.jsonl"
    build = subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--input", str(SOURCE), "--output", str(dataset)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "sample",
            "--input",
            str(dataset),
            "--output",
            str(sample),
            "--seed",
            "20260806",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = read_jsonl(sample)
    assert len(rows) == 256
    assert Counter(row["metadata"]["workload_class"] for row in rows) == {
        "short": 179,
        "medium": 51,
        "long": 26,
    }
    assert len({row["metadata"]["instance_id"] for row in rows}) == 256
    assert {row["blackbox_type"] for row in rows} == {"opencode"}
    all_content = "\n".join(row["prompt"][0]["content"] for row in rows)
    assert "mktemp /tmp/dressage-step.XXXXXX" not in all_content
    assert "date +%s%N > <PATH>" not in all_content
    assert "cat <PATH>" not in all_content


def test_committed_step_balanced_dataset_has_exact_workload(tmp_path):
    assert len(read_jsonl(STEP_BALANCED_300)) == 300
    assert hashlib.sha256(STEP_BALANCED_300.read_bytes()).hexdigest() == (
        "9e53e5d4619b8ff33fdd1ce1a4ccc2caa59f3ada0b8195c20b7112d127598686"
    )

    rows = read_jsonl(STEP_BALANCED_256)
    assert len(rows) == 256
    assert hashlib.sha256(STEP_BALANCED_256.read_bytes()).hexdigest() == (
        "2c1094ab95473120c5f5706a2a399d1d2e4f496f198af1f1601e9b3c8aef5ef9"
    )
    assert len({row["metadata"]["instance_id"] for row in rows}) == 256
    assert Counter(row["metadata"]["workload_class"] for row in rows) == {
        "short": 179,
        "medium": 51,
        "long": 26,
    }
    expected = {
        "short": (4, 5),
        "medium": (9, 10),
        "long": (19, 20),
    }
    assert Counter(
        (
            row["metadata"]["workload_class"],
            row["metadata"]["planned_tool_calls"],
            row["metadata"]["expected_model_steps"],
        )
        for row in rows
    ) == {
        ("short", 4, 5): 179,
        ("medium", 9, 10): 51,
        ("long", 19, 20): 26,
    }
    assert sum(row["metadata"]["planned_tool_calls"] for row in rows) == 1669
    assert sum(row["metadata"]["expected_model_steps"] for row in rows) == 1925
    assert {row["metadata"]["workload_profile_version"] for row in rows} == {
        "dapo-step-balanced-v1"
    }
    assert {str(row["metadata"]["workload_assignment_seed"]) for row in rows} == {
        "20260814"
    }
    for row in rows:
        metadata = row["metadata"]
        tool_calls, model_steps = expected[metadata["workload_class"]]
        content = row["prompt"][0]["content"]
        assert (
            f"requires exactly {tool_calls} sequential bash tool call(s), "
            f"producing an expected total of {model_steps} model steps"
        ) in content
        workflow = content.split("Tool workflow:\n", 1)[1].split(
            "\n\nWorkflow rules:", 1
        )[0]
        numbered_steps = re.findall(r"(?m)^([0-9]+)\. ", workflow)
        assert numbered_steps == [str(index) for index in range(1, tool_calls + 1)]

    resampled = tmp_path / "step-balanced-resampled.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "sample",
            "--input",
            str(STEP_BALANCED_256),
            "--output",
            str(resampled),
            "--seed",
            "20260806",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert resampled.read_bytes() == STEP_BALANCED_256.read_bytes()
