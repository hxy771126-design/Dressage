from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from dressage.recipes.repeat_multistep.agent_whitebox import (
    RepeatMultiStepWhiteboxAgent,
)
from dressage.recipes.repeat_multistep.reward import repeat_multistep
from dressage.recipes.repeat_multistep.tools import REPEAT_INSTRUCTION


REPO_ROOT = Path(__file__).resolve().parents[1]


def _response(
    content: str,
    *,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
) -> dict:
    message = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": message,
            }
        ]
    }


def _sample(planned_steps: int, payload: str = "UNIQUE PAYLOAD") -> SimpleNamespace:
    return SimpleNamespace(
        prompt=[
            {
                "role": "user",
                "content": (
                    "Repeat the payload exactly.\n"
                    "BEGIN_REPEAT_PAYLOAD\n"
                    f"{payload}\n"
                    "END_REPEAT_PAYLOAD"
                ),
            }
        ],
        label=None,
        metadata={"planned_model_steps": planned_steps},
    )


class FakeRepeatAgent(RepeatMultiStepWhiteboxAgent):
    def __init__(
        self,
        responses: list[dict],
        *,
        exception_at: int | None = None,
    ) -> None:
        self.responses = responses
        self.exception_at = exception_at
        self.calls: list[tuple[str | None, dict]] = []
        self.args = SimpleNamespace(rollout_max_response_len=128)
        self.session_id = "repeat-session"
        self.instance_id = "repeat-instance"

    async def chat(self, body, *, turn_id=None):
        call_index = len(self.calls)
        self.calls.append((turn_id, copy.deepcopy(body)))
        if call_index == self.exception_at:
            raise RuntimeError("proxy request failed")
        return self.responses[call_index]


@pytest.fixture(autouse=True)
def _clear_repeat_delay(monkeypatch):
    monkeypatch.delenv("DRESSAGE_REPEAT_MULTISTEP_TOOL_DELAY_MS", raising=False)


@pytest.mark.parametrize("planned_steps", [1, 8, 36, 52])
def test_repeat_loop_uses_exactly_planned_model_steps(planned_steps):
    payload = "UNIQUE PAYLOAD"
    sample = _sample(planned_steps, payload)
    agent = FakeRepeatAgent([_response(payload)] * planned_steps)

    result = asyncio.run(
        agent.rollout(sample, {"temperature": 0.9, "max_new_tokens": 256})
    )

    assert result == payload
    assert len(agent.calls) == planned_steps
    assert [turn_id for turn_id, _ in agent.calls] == [
        f"repeat-step-{index:04d}" for index in range(planned_steps)
    ]
    assert [len(body["messages"]) for _, body in agent.calls] == [
        2 + 2 * index for index in range(planned_steps)
    ]
    assert all(
        "tools" not in body and "tool_choice" not in body
        for _, body in agent.calls
    )
    assert all(body["temperature"] == 0.0 for _, body in agent.calls)
    assert all(body["max_tokens"] == 256 for _, body in agent.calls)
    assert sample.metadata["attempted_model_steps"] == planned_steps
    assert sample.metadata["actual_model_steps"] == planned_steps
    assert sample.metadata["protocol_success"] is True
    assert sample.metadata["repeat_tool_delay_ms"] == 0
    assert repeat_multistep(sample) == 1.0


def test_repeat_loop_appends_assistant_then_host_tool_message():
    payloads = ["first", "second", "third"]
    sample = _sample(3, payloads[0])
    agent = FakeRepeatAgent(
        [
            _response(payloads[0], reasoning_content="copy exactly"),
            _response(payloads[1]),
            _response(payloads[2]),
        ]
    )

    asyncio.run(agent.rollout(sample, {}))

    first_messages = agent.calls[0][1]["messages"]
    second_messages = agent.calls[1][1]["messages"]
    third_messages = agent.calls[2][1]["messages"]
    assert [message["role"] for message in first_messages] == ["system", "user"]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_messages[2] == {
        "role": "assistant",
        "content": "first",
        "reasoning_content": "copy exactly",
    }
    assert second_messages[3] == {
        "role": "tool",
        "tool_call_id": "repeat-0000",
        "content": REPEAT_INSTRUCTION,
    }
    assert [message["role"] for message in third_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert third_messages[4]["content"] == "second"
    assert third_messages[5]["tool_call_id"] == "repeat-0001"
    assert len(agent.calls) == 3
    assert sample.metadata["protocol_success"] is True
    assert repeat_multistep(sample) == 1.0


def test_zero_delay_does_not_sleep(monkeypatch):
    sample = _sample(3)
    agent = FakeRepeatAgent([_response("payload")] * 3)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setenv("DRESSAGE_REPEAT_MULTISTEP_TOOL_DELAY_MS", "0")
    monkeypatch.setattr(
        "dressage.recipes.repeat_multistep.agent_whitebox.asyncio.sleep",
        fake_sleep,
    )

    asyncio.run(agent.rollout(sample, {}))

    assert sleeps == []
    assert sample.metadata["repeat_tool_delay_ms"] == 0


def test_delay_waits_between_steps_including_after_failure(monkeypatch):
    sample = _sample(3)
    agent = FakeRepeatAgent(
        [_response("first"), _response("third"), _response("unused")],
        exception_at=1,
    )
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setenv("DRESSAGE_REPEAT_MULTISTEP_TOOL_DELAY_MS", "200")
    monkeypatch.setattr(
        "dressage.recipes.repeat_multistep.agent_whitebox.asyncio.sleep",
        fake_sleep,
    )

    asyncio.run(agent.rollout(sample, {}))

    assert sleeps == [0.2, 0.2]
    assert sample.metadata["repeat_tool_delay_ms"] == 200
    assert len(agent.calls) == 3


@pytest.mark.parametrize("delay", ["-1", "1.5", "invalid", ""])
def test_invalid_delay_fails_before_first_request(monkeypatch, delay):
    sample = _sample(1)
    agent = FakeRepeatAgent([_response("payload")])
    monkeypatch.setenv("DRESSAGE_REPEAT_MULTISTEP_TOOL_DELAY_MS", delay)

    with pytest.raises(ValueError, match="non-negative integer"):
        asyncio.run(agent.rollout(sample, {}))

    assert agent.calls == []


@pytest.mark.parametrize(
    ("bad_response", "expected_truncated"),
    [
        (_response("partial", finish_reason="length"), [1]),
        (_response(""), []),
    ],
)
def test_bad_response_scores_zero_without_stopping(
    bad_response,
    expected_truncated,
):
    payload = "expected"
    sample = _sample(3, payload)
    agent = FakeRepeatAgent(
        [_response(payload), bad_response, _response(payload)]
    )

    asyncio.run(agent.rollout(sample, {}))

    assert len(agent.calls) == 3
    assert sample.metadata["attempted_model_steps"] == 3
    assert sample.metadata["actual_model_steps"] == 3
    assert sample.metadata["protocol_success"] is False
    assert sample.metadata["repeat_truncated_step_indices"] == expected_truncated
    assert repeat_multistep(sample) == 0.0
    assert agent.calls[2][1]["messages"][-2]["content"] == (
        bad_response["choices"][0]["message"]["content"]
    )
    assert agent.calls[2][1]["messages"][-1]["role"] == "tool"


def test_response_content_is_not_scored():
    sample = _sample(3, "expected")
    agent = FakeRepeatAgent(
        [_response("first"), _response("different"), _response("still normal")]
    )

    asyncio.run(agent.rollout(sample, {}))

    assert len(agent.calls) == 3
    assert sample.metadata["protocol_success"] is True
    assert repeat_multistep(sample) == 1.0


def test_proxy_exception_is_recorded_without_stopping():
    payload = "expected"
    sample = _sample(3, payload)
    agent = FakeRepeatAgent(
        [_response(payload), _response(payload), _response(payload)],
        exception_at=1,
    )

    asyncio.run(agent.rollout(sample, {}))

    assert len(agent.calls) == 3
    assert sample.metadata["attempted_model_steps"] == 3
    assert sample.metadata["actual_model_steps"] == 2
    assert sample.metadata["failed_step_count"] == 1
    assert json.loads(sample.metadata["repeat_step_errors_json"]) == [
        {"step_index": 1, "error_type": "RuntimeError"}
    ]
    assert [message["role"] for message in agent.calls[2][1]["messages"][-2:]] == [
        "tool",
        "tool",
    ]
    assert agent.calls[2][1]["messages"][-1]["tool_call_id"] == "repeat-0001"
    assert repeat_multistep(sample) == 0.0


def test_reward_only_checks_execution_health():
    sample = SimpleNamespace(
        label=None,
        metadata={
            "planned_model_steps": 2,
            "attempted_model_steps": 2,
            "actual_model_steps": 2,
            "protocol_success": True,
            "failed_step_count": 0,
            "truncated_step_count": 0,
        },
    )
    assert repeat_multistep(sample) == 1.0

    sample.metadata["actual_model_steps"] = 1
    assert repeat_multistep(sample) == 0.0
    sample.metadata["actual_model_steps"] = 2
    sample.metadata["truncated_step_count"] = 1
    assert repeat_multistep(sample) == 0.0
    sample.metadata["truncated_step_count"] = 0
    sample.label = "not json"
    assert repeat_multistep(sample) == 1.0


@pytest.mark.parametrize("planned_steps", [0, -1, True, None])
def test_repeat_loop_rejects_invalid_planned_steps(planned_steps):
    sample = _sample(1)
    sample.metadata["planned_model_steps"] = planned_steps
    agent = FakeRepeatAgent([])

    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(agent.rollout(sample, {}))


@pytest.mark.parametrize(
    (
        "filename",
        "sha256",
        "step_distribution",
        "payload_tokens",
        "min_context",
        "max_context",
    ),
    [
        (
            "dressage_repeat_multistep_4k_52_256.jsonl",
            "3812e48476893e913980c622322a1ef19d3135a7fed2d8a2c8db12d4a5178c81",
            {1: 216, 8: 1, 52: 39},
            4096,
            8574,
            219673,
        ),
        (
            "dressage_repeat_multistep_6k_36_256.jsonl",
            "0345db94b1a8757f58de30bdfc7dca0419c341ba499928dd93804a85cadd098a",
            {1: 198, 2: 1, 36: 57},
            6144,
            12670,
            229225,
        ),
    ],
)
def test_repeat_dataset_is_complete_and_deterministic(
    filename,
    sha256,
    step_distribution,
    payload_tokens,
    min_context,
    max_context,
):
    path = REPO_ROOT / "examples" / "data" / filename
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    metadata = [row["metadata"] for row in rows]

    assert hashlib.sha256(data).hexdigest() == sha256
    assert len(rows) == 256
    assert len({item["instance_id"] for item in metadata}) == 256
    assert Counter(item["planned_model_steps"] for item in metadata) == Counter(
        step_distribution
    )
    assert sum(item["planned_model_steps"] for item in metadata) == 2252
    assert {item["payload_token_count"] for item in metadata} == {payload_tokens}
    contexts = [item["estimated_max_context_tokens"] for item in metadata]
    assert (min(contexts), max(contexts)) == (min_context, max_context)
    assert {row["agent_mode"] for row in rows} == {"whitebox"}
    assert {row["reward_fn"] for row in rows} == {"repeat_multistep"}
    assert {row["generate_function_path"] for row in rows} == {
        "dressage.recipes.repeat_multistep.agent_whitebox.generate"
    }


@pytest.mark.parametrize(
    ("model", "mooncake_size"),
    [
        ("qwen3.5_4b", "64gb"),
        ("qwen3.5_35b_a3b", "128gb"),
    ],
)
def test_repeat_benchmark_dry_run_uses_whitebox_limits(
    tmp_path,
    model,
    mooncake_size,
):
    script = REPO_ROOT / "examples" / "scripts" / (
        f"benchmark_engine_rebalancing_{model}_sync_local_l3_hicache.sh"
    )
    prompt = REPO_ROOT / "examples" / "data" / (
        "dressage_repeat_multistep_4k_52_256.jsonl"
    )
    result = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "BENCHMARK_DRY_RUN": "1",
            "BENCHMARK_ROOT": str(tmp_path / model),
            "BENCHMARK_WORKLOAD": "repeat_multistep",
            "BENCHMARK_PROMPT_DATA": str(prompt),
            "DRESSAGE_REPEAT_MULTISTEP_TOOL_DELAY_MS": "200",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "workload:      repeat_multistep" in result.stdout
    assert "rollout batch: 256" in result.stdout
    assert "global batch:  256" in result.stdout
    assert "response max:  6400" in result.stdout
    assert "sandbox slots: 0" in result.stdout
    assert "Proxy max session steps: 100" in result.stdout
    assert (
        "generate function: "
        "dressage.recipes.repeat_multistep.agent_whitebox.generate"
    ) in result.stdout
    assert "Paddock mode:  whitebox" in result.stdout
    assert "log write mode:await" in result.stdout
    assert "tool delay:    200 ms" in result.stdout
    assert "context window: 262144" in result.stdout
    assert "SGLang context: 262144" in result.stdout
    assert f"Mooncake size: {mooncake_size}" in result.stdout
    assert not (tmp_path / model).exists()
