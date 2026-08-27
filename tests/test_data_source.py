"""Tests for DressageDataSource."""

from __future__ import annotations

import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dressage.rollout.data_source import DressageDataSource


def _make_jsonl(data: list[dict]) -> str:
    """Write test data to a temp JSONL file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture
def basic_data():
    data = [
        {"prompt": "What is 2+2?", "label": "4", "reward_fn": "math_score"},
        {"prompt": "Write hello world", "label": "hello world", "reward_fn": "exact_match"},
        {"prompt": "Tell me a joke", "agent_mode": "blackbox", "agent_id": "comedy_agent"},
    ]
    path = _make_jsonl(data)
    yield path
    os.unlink(path)


@pytest.fixture
def args(basic_data):
    return Namespace(
        prompt_data=basic_data,
        input_key="prompt",
        label_key="label",
        metadata_key="metadata",
        multimodal_keys=None,
        apply_chat_template=False,
        rollout_shuffle=False,
        rollout_seed=42,
        n_samples_per_prompt=2,
        rollout_global_dataset=True,
        rollout_max_prompt_len=None,
        hf_checkpoint=None,
    )


class TestDressageDataSource:
    def test_load_samples(self, args):
        ds = DressageDataSource(args)
        assert len(ds) == 3

    def test_get_samples_groups(self, args):
        ds = DressageDataSource(args)
        groups = ds.get_samples(2)
        assert len(groups) == 2
        assert len(groups[0]) == 2  # n_samples_per_prompt
        assert len(groups[1]) == 2

    def test_metadata_passthrough(self, args):
        ds = DressageDataSource(args)
        groups = ds.get_samples(3)

        first_sample = groups[0][0]
        assert first_sample.metadata.get("reward_fn") == "math_score"

        third_sample = groups[2][0]
        assert third_sample.metadata.get("agent_mode") == "blackbox"
        assert third_sample.metadata.get("agent_id") == "comedy_agent"

    def test_stripped_blackbox_metadata_passthrough(self):
        path = _make_jsonl(
            [
                {
                    "prompt": "run a blackbox task",
                    "label": "done",
                    "metadata": {"instance_id": "inst-1"},
                    "agent_mode": "blackbox",
                    "blackbox_type": "openclaw",
                    "reward_fn": "contains_label",
                    "task_type": "smoke",
                }
            ]
        )
        try:
            ds = DressageDataSource(
                Namespace(
                    prompt_data=path,
                    input_key="prompt",
                    label_key="label",
                    metadata_key="metadata",
                    multimodal_keys=None,
                    apply_chat_template=False,
                    rollout_shuffle=False,
                    rollout_seed=42,
                    n_samples_per_prompt=1,
                    rollout_global_dataset=True,
                    rollout_max_prompt_len=None,
                    hf_checkpoint=None,
                )
            )
            sample = ds.get_samples(1)[0][0]
        finally:
            os.unlink(path)

        assert sample.metadata == {
            "instance_id": "inst-1",
            "agent_mode": "blackbox",
            "blackbox_type": "openclaw",
            "reward_fn": "contains_label",
            "task_type": "smoke",
        }
        assert "env_type" not in sample.metadata
        assert "sandbox_env_key" not in sample.metadata
        assert "backend_options" not in sample.metadata

    def test_group_index_unique(self, args):
        ds = DressageDataSource(args)
        groups = ds.get_samples(3)

        group_indices = set()
        for group in groups:
            gi = group[0].group_index
            assert gi not in group_indices
            group_indices.add(gi)
            for s in group:
                assert s.group_index == gi

    def test_sample_index_unique(self, args):
        ds = DressageDataSource(args)
        groups = ds.get_samples(3)

        indices = set()
        for group in groups:
            for s in group:
                assert s.index not in indices
                indices.add(s.index)

    def test_deterministic_blackbox_samples_get_stable_session_ids(self, args):
        path = _make_jsonl(
            [
                {
                    "prompt": "first",
                    "agent_mode": "blackbox",
                    "metadata": {"instance_id": "first"},
                },
                {
                    "prompt": "second",
                    "agent_mode": "blackbox",
                    "metadata": {"instance_id": "second"},
                },
            ]
        )
        args.prompt_data = path
        args.sglang_enable_deterministic_inference = True
        try:
            first, second = [group[0] for group in DressageDataSource(args).get_samples(2)]
            repeated_first = DressageDataSource(args).get_samples(1)[0][0]
        finally:
            os.unlink(path)

        assert getattr(first, "session_id", None) == "bbs-det-42-0"
        assert first.metadata["dressage_deterministic_session_id"] == "bbs-det-42-0"
        assert repeated_first.session_id == first.session_id
        assert getattr(second, "session_id", None) == "bbs-det-42-2"
        assert second.metadata["dressage_deterministic_session_id"] == "bbs-det-42-2"

    def test_stable_session_ids_do_not_apply_without_deterministic_blackbox_inference(
        self, args
    ):
        path = _make_jsonl(
            [
                {"prompt": "blackbox", "agent_mode": "blackbox"},
                {"prompt": "whitebox", "agent_mode": "whitebox"},
            ]
        )
        args.prompt_data = path
        try:
            nondeterministic = DressageDataSource(args).get_samples(1)[0][0]
            args.sglang_enable_deterministic_inference = True
            non_blackbox = DressageDataSource(args).get_samples(2)[1][0]
        finally:
            os.unlink(path)

        assert getattr(nondeterministic, "session_id", None) is None
        assert "dressage_deterministic_session_id" not in nondeterministic.metadata
        assert getattr(non_blackbox, "session_id", None) is None
        assert "dressage_deterministic_session_id" not in non_blackbox.metadata

    def test_buffer_add_and_get(self, args):
        ds = DressageDataSource(args)
        groups = ds.get_samples(2)
        ds.add_samples(groups)
        assert len(ds.buffer) == 2
        recovered = ds.get_samples(2)
        assert len(recovered) == 2
        assert len(ds.buffer) == 0

    def test_epoch_wraparound(self, args):
        ds = DressageDataSource(args)
        groups1 = ds.get_samples(3)
        groups2 = ds.get_samples(3)  # should wrap to next epoch
        assert len(groups2) == 3

    def test_label_preserved(self, args):
        ds = DressageDataSource(args)
        groups = ds.get_samples(2)
        assert groups[0][0].label == "4"
        assert groups[1][0].label == "hello world"

    def test_e2b_dapo_sample_dataset_schema(self):
        path = Path(__file__).resolve().parents[1] / "examples/data/dressage_dapo_prompts_e2b.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        sandbox_images = {
            "opencode": "huangs-default-team-3fa4/dressage-blackbox-opencode",
            "openclaw": "huangs-default-team-3fa4/dressage-blackbox-claw",
        }
        sandbox_cmd = (
            "nohup /usr/local/bin/start-blackbox-server "
            ">/tmp/blackbox-server.log 2>&1 & "
            "for i in $(seq 1 60); do "
            "curl -fsS http://127.0.0.1:31000/health && exit 0; "
            "sleep 1; "
            "done; "
            "cat /tmp/blackbox-server.log; "
            "exit 1"
        )

        assert len(rows) == 10
        assert [row["blackbox_type"] for row in rows].count("opencode") == 5
        assert [row["blackbox_type"] for row in rows].count("openclaw") == 5
        for row in rows:
            metadata = row["metadata"]
            blackbox_type = row["blackbox_type"]
            assert row["prompt"][0]["content"].startswith(
                "Solve the following math problem step by step."
            )
            assert metadata["sandbox_image"] == sandbox_images[blackbox_type]
            assert metadata["sandbox_timeout_sec"] == 3600
            assert metadata["sandbox_cmd"] == sandbox_cmd
            assert metadata["blackbox_execute_cmds"] == {
                "before_agent": [
                    {
                        "name": "env_check",
                        "cmd": "python -V && ls -la",
                        "timeout": 30,
                        "required": False,
                    }
                ],
                "after_agent": [
                    {
                        "name": "inspect_files",
                        "cmd": "find . -maxdepth 2 -type f",
                        "timeout": 30,
                        "required": False,
                    }
                ],
            }
            for key in (
                "e2b_template",
                "image",
                "env_key",
                "sandbox_env_key",
                "cmd",
                "container_backend",
                "sandbox_extra_params",
                "sandbox_container_backend",
            ):
                assert key not in metadata
