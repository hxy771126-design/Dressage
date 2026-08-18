from __future__ import annotations

import hashlib
import importlib
import itertools
from dataclasses import FrozenInstanceError

import pytest


def solver_module():
    return importlib.import_module("dressage.proxy.rebalancing._batch_milp")


def engine(
    url: str,
    *,
    base_requests: float = 0.0,
    base_tokens: float = 0.0,
    base_prefill: float = 0.0,
    request_capacity: float = 10.0,
    token_capacity: float = 10.0,
    token_usage: float = 0.0,
    queue_pressure: float = 0.0,
):
    return solver_module().EngineBaseline(
        url=url,
        base_requests=base_requests,
        base_tokens=base_tokens,
        base_prefill=base_prefill,
        request_capacity=request_capacity,
        token_capacity=token_capacity,
        token_usage=token_usage,
        queue_pressure=queue_pressure,
    )


def edge(
    session_id: str,
    engine_url: str,
    *,
    requests: float,
    tokens: float = 0.0,
    prefill: float = 0.0,
    decode_pressure: float = 0.0,
    migration: bool = False,
    migration_cost_tokens: int = 0,
):
    return solver_module().FeasibleEdge(
        session_id=session_id,
        engine_url=engine_url,
        request_increment=requests,
        token_increment=tokens,
        prefill_increment=prefill,
        decode_pressure_increment=decode_pressure,
        voluntary_migration=migration,
        migration_cost_tokens=migration_cost_tokens,
    )


def problem(engines, edges_by_session):
    return solver_module().BatchProblem(
        engines=tuple(engines),
        edges_by_session=edges_by_session,
    )


def stable_coefficient(session_id: str, engine_url: str) -> float:
    digest = hashlib.sha256(
        session_id.encode("utf-8") + b"\0" + engine_url.encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") / (1 << 256)


def assignment_objectives(batch, selected_edges):
    loads = []
    for baseline in batch.engines:
        assigned = [edge for edge in selected_edges if edge.engine_url == baseline.url]
        request = (
            baseline.base_requests + sum(edge.request_increment for edge in assigned)
        ) / baseline.request_capacity
        token = max(
            (baseline.base_tokens + sum(edge.token_increment for edge in assigned))
            / baseline.token_capacity,
            baseline.token_usage,
        )
        prefill = (
            baseline.base_prefill + sum(edge.prefill_increment for edge in assigned)
        ) / baseline.token_capacity
        decode = sum(edge.decode_pressure_increment for edge in assigned)
        loads.append(request + token + baseline.queue_pressure + prefill + decode)
    return (
        max(loads),
        sum(
            stable_coefficient(edge.session_id, edge.engine_url)
            for edge in selected_edges
        ),
    )


def target_assignment_objectives(batch, selected_edges):
    maximum_load, tie = assignment_objectives(batch, selected_edges)
    return (
        sum(edge.migration_cost_tokens for edge in selected_edges),
        maximum_load,
        tie,
    )


def test_milp_balances_hand_derived_request_load():
    batch = problem(
        [engine("a"), engine("b")],
        {
            "s1": (edge("s1", "a", requests=6), edge("s1", "b", requests=6)),
            "s2": (edge("s2", "a", requests=6), edge("s2", "b", requests=6)),
        },
    )

    result = solver_module().solve_batch_milp(batch)

    assert result.status is solver_module().SolverStatus.OPTIMAL
    assert set(result.assignment.values()) == {"a", "b"}
    assert result.maximum_load == pytest.approx(0.6)
    assert result.voluntary_migrations == 0
    assert result.elapsed_seconds >= 0.0


def test_milp_recomputes_load_with_token_usage_floor():
    batch = problem(
        [
            engine(
                "a",
                base_requests=1,
                base_tokens=1,
                base_prefill=2,
                token_usage=0.6,
                queue_pressure=0.05,
            ),
            engine("b", base_requests=1),
        ],
        {
            "s": (
                edge("s", "a", requests=1, tokens=2, prefill=1),
            )
        },
    )

    result = solver_module().solve_batch_milp(batch)

    assert result.maximum_load == pytest.approx(1.15)
    assert result.minimum_load == pytest.approx(0.1)
    assert result.load_range == pytest.approx(1.05)


def test_milp_recomputes_load_with_decode_pressure():
    batch = problem(
        [engine("a"), engine("b")],
        {
            "s": (
                edge("s", "a", requests=1, decode_pressure=0.25),
                edge("s", "b", requests=1, decode_pressure=0.05),
            )
        },
    )

    result = solver_module().solve_batch_milp(batch)

    assert result.assignment == {"s": "b"}
    assert result.maximum_load == pytest.approx(0.15)
    assert result.minimum_load == pytest.approx(0.0)
    assert result.load_range == pytest.approx(0.15)


def test_milp_does_not_impose_a_unit_load_limit():
    batch = problem(
        [engine("a", base_requests=15)],
        {"s": (edge("s", "a", requests=5),)},
    )

    result = solver_module().solve_batch_milp(batch)

    assert result.status is solver_module().SolverStatus.OPTIMAL
    assert result.maximum_load == pytest.approx(2.0)


def test_milp_uses_stable_hash_after_load_even_if_tie_migrates():
    batch = problem(
        [engine("a"), engine("b")],
        {
            "session": (
                edge("session", "a", requests=1),
                edge("session", "b", requests=1, migration=True),
            )
        },
    )

    result = solver_module().solve_batch_milp(batch)

    assert result.assignment == {"session": "b"}
    assert result.maximum_load == pytest.approx(0.1)
    assert result.voluntary_migrations == 1


def test_milp_uses_stable_sha256_tie_breaking():
    candidates = (
        edge("s", "a", requests=1),
        edge("s", "b", requests=1),
    )
    expected = min(
        candidates,
        key=lambda candidate: stable_coefficient(
            candidate.session_id, candidate.engine_url
        ),
    ).engine_url
    batch = problem([engine("a"), engine("b")], {"s": candidates})

    first = solver_module().solve_batch_milp(batch)
    second = solver_module().solve_batch_milp(batch)

    assert first.assignment == second.assignment == {"s": expected}


def test_target_milp_minimizes_kv_cost_before_maximum_load():
    batch = problem(
        [engine("a", base_requests=6), engine("b")],
        {
            "large": (
                edge("large", "a", requests=2),
                edge(
                    "large",
                    "b",
                    requests=2,
                    migration=True,
                    migration_cost_tokens=5,
                ),
            ),
            "small": (
                edge("small", "a", requests=1),
                edge(
                    "small",
                    "b",
                    requests=1,
                    migration=True,
                    migration_cost_tokens=1,
                ),
            ),
        },
    )

    result = solver_module().solve_batch_for_target_load(
        batch,
        maximum_load_limit=0.8,
        deadline_seconds=1.0,
    )

    assert result.assignment == {"large": "a", "small": "b"}
    assert result.maximum_load == pytest.approx(0.8)
    assert result.minimum_load == pytest.approx(0.1)
    assert result.load_range == pytest.approx(0.7)
    assert result.total_migration_cost_tokens == 1
    assert result.voluntary_migrations == 1


def test_target_milp_reports_an_infeasible_load_limit():
    batch = problem(
        [engine("a", base_requests=9)],
        {"s": (edge("s", "a", requests=2),)},
    )

    with pytest.raises(solver_module().BatchSolverError) as raised:
        solver_module().solve_batch_for_target_load(
            batch,
            maximum_load_limit=1.0,
            deadline_seconds=1.0,
        )

    assert raised.value.status == 2


def test_target_milp_keeps_zero_cost_owner_when_sticky_load_meets_target():
    batch = problem(
        [engine("a", base_requests=8), engine("b")],
        {
            "s": (
                edge("s", "a", requests=1),
                edge(
                    "s",
                    "b",
                    requests=1,
                    migration=True,
                    migration_cost_tokens=5,
                ),
            )
        },
    )

    result = solver_module().solve_batch_for_target_load(
        batch,
        maximum_load_limit=0.9,
        deadline_seconds=1.0,
    )

    assert result.assignment == {"s": "a"}
    assert result.total_migration_cost_tokens == 0
    assert result.voluntary_migrations == 0


def test_target_milp_keeps_kv_cost_priority_over_decode_pressure():
    batch = problem(
        [engine("a"), engine("b")],
        {
            "s": (
                edge("s", "a", requests=1, decode_pressure=0.4),
                edge(
                    "s",
                    "b",
                    requests=1,
                    decode_pressure=0.0,
                    migration=True,
                    migration_cost_tokens=1,
                ),
            )
        },
    )

    result = solver_module().solve_batch_for_target_load(
        batch,
        maximum_load_limit=0.5,
        deadline_seconds=1.0,
    )

    assert result.assignment == {"s": "a"}
    assert result.maximum_load == pytest.approx(0.5)
    assert result.total_migration_cost_tokens == 0


def test_target_milp_matches_exhaustive_cost_load_and_hash_ordering():
    batch = problem(
        [engine("a", base_requests=4), engine("b")],
        {
            "s1": (
                edge("s1", "a", requests=3),
                edge(
                    "s1",
                    "b",
                    requests=3,
                    migration=True,
                    migration_cost_tokens=5,
                ),
            ),
            "s2": (
                edge("s2", "a", requests=2),
                edge(
                    "s2",
                    "b",
                    requests=2,
                    migration=True,
                    migration_cost_tokens=2,
                ),
            ),
            "new": (
                edge("new", "a", requests=1),
                edge("new", "b", requests=1),
            ),
        },
    )
    limit = 0.8
    feasible = [
        selected
        for selected in itertools.product(*batch.edges_by_session.values())
        if assignment_objectives(batch, selected)[0] <= limit + 1e-7
    ]
    expected_edges = min(
        feasible,
        key=lambda selected: target_assignment_objectives(batch, selected),
    )

    result = solver_module().solve_batch_for_target_load(
        batch,
        maximum_load_limit=limit,
        deadline_seconds=1.0,
    )

    assert result.assignment == {
        edge.session_id: edge.engine_url for edge in expected_edges
    }
    assert result.total_migration_cost_tokens == sum(
        edge.migration_cost_tokens for edge in expected_edges
    )
    assert result.maximum_load == pytest.approx(
        assignment_objectives(batch, expected_edges)[0]
    )


def test_problem_rejects_a_session_without_an_edge():
    with pytest.raises(ValueError, match="session 'missing' has no feasible edge"):
        problem([engine("a")], {"missing": ()})


def test_problem_rejects_an_edge_for_an_unknown_engine():
    with pytest.raises(ValueError, match="unknown Engine 'missing'"):
        problem(
            [engine("a")],
            {"s": (edge("s", "missing", requests=1),)},
        )


def test_problem_rejects_non_positive_engine_capacity():
    with pytest.raises(ValueError, match="capacities must be positive"):
        problem([engine("a", request_capacity=0)], {})


def test_problem_rejects_negative_migration_cost_tokens():
    with pytest.raises(ValueError, match="migration cost tokens must be non-negative"):
        problem(
            [engine("a")],
            {
                "s": (
                    edge(
                        "s",
                        "a",
                        requests=1,
                        migration=True,
                        migration_cost_tokens=-1,
                    ),
                )
            },
        )


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_problem_rejects_invalid_decode_pressure(value):
    with pytest.raises(ValueError, match="decode pressure increment"):
        problem(
            [engine("a")],
            {"s": (edge("s", "a", requests=1, decode_pressure=value),)},
        )


def test_input_values_are_frozen_and_problem_copies_edge_mapping():
    baseline = engine("a")
    candidate = edge("s", "a", requests=1)
    edges_by_session = {"s": [candidate]}
    batch = problem([baseline], edges_by_session)
    edges_by_session["s"].clear()

    with pytest.raises(FrozenInstanceError):
        baseline.url = "changed"
    with pytest.raises(FrozenInstanceError):
        candidate.engine_url = "changed"
    with pytest.raises(TypeError):
        batch.edges_by_session["other"] = ()
    assert batch.edges_by_session == {"s": (candidate,)}


def test_greedy_never_migrates_when_an_owner_edge_exists():
    batch = problem(
        [engine("a", base_requests=9), engine("b")],
        {
            "s": (
                edge("s", "a", requests=1),
                edge("s", "b", requests=1, migration=True),
            )
        },
    )

    result = solver_module().solve_batch_greedy(batch)

    assert result.status is solver_module().SolverStatus.GREEDY
    assert result.assignment == {"s": "a"}
    assert result.maximum_load == pytest.approx(1.0)
    assert result.voluntary_migrations == 0


def test_greedy_recomputes_selected_decode_pressure():
    batch = problem(
        [engine("a"), engine("b")],
        {
            "s": (
                edge("s", "a", requests=1, decode_pressure=0.2),
                edge("s", "b", requests=1, migration=True),
            )
        },
    )

    result = solver_module().solve_batch_greedy(batch)

    assert result.assignment == {"s": "a"}
    assert result.maximum_load == pytest.approx(0.3)


def test_greedy_is_stable_across_session_and_edge_input_order():
    edges = {
        "s2": (
            edge("s2", "b", requests=3, migration=True),
            edge("s2", "a", requests=3, migration=True),
        ),
        "s1": (
            edge("s1", "b", requests=3, migration=True),
            edge("s1", "a", requests=3, migration=True),
        ),
    }
    reversed_edges = {
        session_id: tuple(reversed(candidates))
        for session_id, candidates in reversed(tuple(edges.items()))
    }

    first = solver_module().solve_batch_greedy(
        problem([engine("a"), engine("b")], edges)
    )
    second = solver_module().solve_batch_greedy(
        problem([engine("b"), engine("a")], reversed_edges)
    )

    assert first.assignment == second.assignment
    assert first.maximum_load == second.maximum_load == pytest.approx(0.3)
    assert first.voluntary_migrations == second.voluntary_migrations == 2


def test_milp_matches_exhaustive_load_and_tie_ordering():
    batch = problem(
        [engine("a"), engine("b")],
        {
            "s1": (
                edge("s1", "a", requests=4),
                edge("s1", "b", requests=4, migration=True),
            ),
            "s2": (
                edge("s2", "a", requests=4, migration=True),
                edge("s2", "b", requests=4),
            ),
            "s3": (
                edge("s3", "a", requests=0),
                edge("s3", "b", requests=0),
            ),
        },
    )
    assignments = itertools.product(*batch.edges_by_session.values())
    expected_edges = min(assignments, key=lambda selected: assignment_objectives(batch, selected))
    expected_assignment = {
        candidate.session_id: candidate.engine_url for candidate in expected_edges
    }
    expected_load, _ = assignment_objectives(batch, expected_edges)
    expected_migrations = sum(
        candidate.voluntary_migration for candidate in expected_edges
    )

    result = solver_module().solve_batch_milp(batch)

    assert result.assignment == expected_assignment
    assert result.maximum_load == pytest.approx(expected_load)
    assert result.voluntary_migrations == expected_migrations
