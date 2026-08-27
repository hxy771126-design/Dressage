from __future__ import annotations

import pytest

from dressage.proxy.rebalancing.load_decision_trace import (
    LoadDecisionHistory,
    LoadDecisionTrace,
)


def test_history_is_bounded_and_copies_payloads() -> None:
    history = LoadDecisionHistory(history_size=2)
    for decision_id in (1, 2, 3):
        history.record(
            LoadDecisionTrace(
                {
                    "decision": {"id": decision_id},
                    "engines": [{"url": "a"}],
                }
            )
        )
    snapshot = history.snapshot()
    assert [item["decision"]["id"] for item in snapshot] == [2, 3]
    snapshot[0]["engines"][0]["url"] = "mutated"
    assert history.snapshot()[0]["engines"][0]["url"] == "a"


def test_history_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        LoadDecisionHistory(history_size=0)
