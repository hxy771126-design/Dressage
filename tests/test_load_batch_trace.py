from __future__ import annotations

import json

import pytest

from dressage.proxy.rebalancing.load_batch_trace import (
    LoadBatchHistory,
    LoadBatchTrace,
)


def test_history_evicts_oldest_trace_at_configured_capacity():
    history = LoadBatchHistory(history_size=2)

    history.record(LoadBatchTrace({"batch": {"id": "batch-1"}}))
    history.record(LoadBatchTrace({"batch": {"id": "batch-2"}}))
    history.record(LoadBatchTrace({"batch": {"id": "batch-3"}}))

    assert history.snapshot() == [
        {"batch": {"id": "batch-2"}},
        {"batch": {"id": "batch-3"}},
    ]


def test_history_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="history_size must be positive"):
        LoadBatchHistory(history_size=0)


def test_record_copies_nested_batch_step_engine_and_solver_fields():
    payload = {
        "batch": {"id": "batch-7", "steps": [{"id": "step-1", "engine": "a"}]},
        "engines": [{"id": "a", "available": True}],
        "solver": {"status": "optimal", "objective": 1.25},
    }
    history = LoadBatchHistory(history_size=1)
    history.record(LoadBatchTrace(payload))
    payload["batch"]["steps"][0]["engine"] = "changed"
    payload["engines"][0]["available"] = False
    payload["solver"]["status"] = "changed"

    assert history.snapshot() == [
        {
            "batch": {
                "id": "batch-7",
                "steps": [{"id": "step-1", "engine": "a"}],
            },
            "engines": [{"id": "a", "available": True}],
            "solver": {"status": "optimal", "objective": 1.25},
        }
    ]


def test_snapshot_returns_independent_json_serializable_copies():
    history = LoadBatchHistory(history_size=1)
    history.record(LoadBatchTrace({"batch": {"id": "batch-8", "steps": []}}))

    snapshot = history.snapshot()
    snapshot[0]["batch"]["id"] = "mutated"

    assert history.snapshot() == [{"batch": {"id": "batch-8", "steps": []}}]
    assert json.loads(json.dumps(history.snapshot())) == [
        {"batch": {"id": "batch-8", "steps": []}}
    ]
