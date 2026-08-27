from __future__ import annotations

import pytest

from dressage.proxy.rebalancing.greedy import (
    EngineBaseline,
    FeasibleEdge,
    GreedyStepKind,
    choose_greedy_step,
    choose_stable_engine,
    projected_pressure,
)


def engine(
    url: str,
    *,
    requests: float = 0,
    tokens: float = 0,
    queue: float = 0,
    token_usage: float = 0,
) -> EngineBaseline:
    return EngineBaseline(
        url=url,
        base_requests=requests,
        base_tokens=tokens,
        base_queue=queue,
        request_capacity=100,
        token_capacity=1_000,
        token_usage=token_usage,
    )


def edge(
    session_id: str,
    url: str,
    *,
    tokens: float = 100,
) -> FeasibleEdge:
    return FeasibleEdge(
        session_id=session_id,
        engine_url=url,
        queue_increment=1,
        token_increment=tokens,
    )


def test_pressure_adds_request_token_and_queue() -> None:
    pressure = projected_pressure(
        engine("a", requests=20, tokens=300, queue=10),
        queue_increment=2,
        token_increment=100,
    )
    assert pressure.request == pytest.approx(0.2)
    assert pressure.token == pytest.approx(0.4)
    assert pressure.queue == pytest.approx(0.12)
    assert pressure.total == pytest.approx(0.72)


def test_new_session_selects_lowest_projected_pressure() -> None:
    decision = choose_greedy_step(
        session_id="new",
        kind=GreedyStepKind.NEW_SESSION,
        owner_engine_url=None,
        engines=(engine("a", tokens=800), engine("b", tokens=100)),
        edges=(edge("new", "a"), edge("new", "b")),
        min_load_improvement_ratio=0.1,
    )
    assert decision.selected_target == "b"
    assert decision.decision_reason == "lowest_projected_pressure"


def test_existing_session_stays_below_threshold() -> None:
    decision = choose_greedy_step(
        session_id="old",
        kind=GreedyStepKind.EXISTING_SESSION,
        owner_engine_url="a",
        engines=(engine("a", tokens=500), engine("b", tokens=490)),
        edges=(edge("old", "a"), edge("old", "b")),
        min_load_improvement_ratio=0.1,
    )
    assert decision.selected_target == "a"
    assert decision.threshold_met is False


def test_existing_session_moves_on_threshold_boundary() -> None:
    decision = choose_greedy_step(
        session_id="old",
        kind=GreedyStepKind.EXISTING_SESSION,
        owner_engine_url="a",
        engines=(engine("a", tokens=900), engine("b", tokens=800)),
        edges=(edge("old", "a", tokens=0), edge("old", "b", tokens=0)),
        min_load_improvement_ratio=0.1,
    )
    assert decision.improvement_ratio == pytest.approx((0.91 - 0.81) / 0.91)
    assert decision.selected_target == "b"
    assert decision.threshold_met is True


def test_mandatory_failover_bypasses_threshold() -> None:
    decision = choose_greedy_step(
        session_id="failover",
        kind=GreedyStepKind.MANDATORY_FAILOVER,
        owner_engine_url="dead",
        engines=(engine("a", tokens=400), engine("b", tokens=100)),
        edges=(edge("failover", "a"), edge("failover", "b")),
        min_load_improvement_ratio=1.0,
    )
    assert decision.selected_target == "b"
    assert decision.threshold_met is None


def test_owner_wins_score_tie() -> None:
    decision = choose_greedy_step(
        session_id="old",
        kind=GreedyStepKind.EXISTING_SESSION,
        owner_engine_url="a",
        engines=(engine("a"), engine("b")),
        edges=(edge("old", "a"), edge("old", "b")),
        min_load_improvement_ratio=0,
    )
    assert decision.best_target == "a"
    assert decision.selected_target == "a"


def test_non_owner_tie_is_stable() -> None:
    urls = ("a", "b", "c")
    assert choose_stable_engine("session", urls) == choose_stable_engine(
        "session", tuple(reversed(urls))
    )


def test_invalid_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="capacities"):
        EngineBaseline(
            url="a",
            base_requests=0,
            base_tokens=0,
            base_queue=0,
            request_capacity=0,
            token_capacity=1,
            token_usage=0,
        )
