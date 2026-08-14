"""Synchronous batch assignment using SciPy's CPU HiGHS MILP solver."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


_OPTIMUM_TOLERANCE = 1e-7


@dataclass(frozen=True)
class EngineBaseline:
    url: str
    base_requests: float
    base_tokens: float
    base_prefill: float
    request_capacity: float
    token_capacity: float
    token_usage: float
    queue_pressure: float

    def __post_init__(self) -> None:
        if self.request_capacity <= 0 or self.token_capacity <= 0:
            raise ValueError(f"Engine '{self.url}' capacities must be positive")


@dataclass(frozen=True)
class FeasibleEdge:
    session_id: str
    engine_url: str
    request_increment: float
    token_increment: float
    prefill_increment: float
    voluntary_migration: bool


@dataclass(frozen=True)
class BatchProblem:
    engines: tuple[EngineBaseline, ...]
    edges_by_session: Mapping[str, tuple[FeasibleEdge, ...]]

    def __init__(
        self,
        engines: Sequence[EngineBaseline],
        edges_by_session: Mapping[str, Sequence[FeasibleEdge]],
    ) -> None:
        frozen_engines = tuple(engines)
        frozen_edges = {
            session_id: tuple(edges)
            for session_id, edges in edges_by_session.items()
        }
        if not frozen_engines:
            raise ValueError("problem must contain at least one Engine")
        engine_urls = {engine.url for engine in frozen_engines}
        if len(engine_urls) != len(frozen_engines):
            raise ValueError("Engine URLs must be unique")
        for session_id, edges in frozen_edges.items():
            if not edges:
                raise ValueError(f"session '{session_id}' has no feasible edge")
            seen_engines: set[str] = set()
            for edge in edges:
                if edge.session_id != session_id:
                    raise ValueError(
                        f"edge session '{edge.session_id}' does not match '{session_id}'"
                    )
                if edge.engine_url not in engine_urls:
                    raise ValueError(
                        f"edge for session '{session_id}' references unknown Engine "
                        f"'{edge.engine_url}'"
                    )
                if edge.engine_url in seen_engines:
                    raise ValueError(
                        f"session '{session_id}' has duplicate edge for Engine "
                        f"'{edge.engine_url}'"
                    )
                seen_engines.add(edge.engine_url)
        object.__setattr__(self, "engines", frozen_engines)
        object.__setattr__(
            self,
            "edges_by_session",
            MappingProxyType(frozen_edges),
        )


class SolverStatus(str, Enum):
    OPTIMAL = "optimal"
    GREEDY = "greedy"


@dataclass(frozen=True)
class BatchSolution:
    status: SolverStatus
    assignment: Mapping[str, str]
    maximum_load: float
    voluntary_migrations: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignment", MappingProxyType(dict(self.assignment)))


class BatchSolverError(RuntimeError):
    def __init__(
        self,
        phase: int,
        status: int | None,
        elapsed_seconds: float,
        message: str,
    ) -> None:
        self.phase = phase
        self.status = status
        self.elapsed_seconds = elapsed_seconds
        super().__init__(f"batch MILP phase {phase} was not optimal: {message}")


@dataclass(frozen=True)
class _Model:
    edges: tuple[FeasibleEdge, ...]
    edge_sessions: tuple[str, ...]
    constraints: tuple[LinearConstraint, ...]
    bounds: Bounds
    integrality: np.ndarray
    maximum_load_index: int


def _stable_coefficient(session_id: str, engine_url: str) -> float:
    digest = hashlib.sha256(
        session_id.encode("utf-8") + b"\0" + engine_url.encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") / (1 << 256)


def _build_model(problem: BatchProblem) -> _Model:
    edge_sessions = tuple(sorted(problem.edges_by_session))
    edges = tuple(
        edge
        for session_id in edge_sessions
        for edge in problem.edges_by_session[session_id]
    )
    edge_count = len(edges)
    engine_count = len(problem.engines)
    token_offset = edge_count
    maximum_load_index = edge_count + engine_count
    variable_count = maximum_load_index + 1
    engine_indexes = {
        engine.url: index for index, engine in enumerate(problem.engines)
    }

    assignment_matrix = np.zeros((len(edge_sessions), variable_count))
    edge_indexes_by_session: dict[str, list[int]] = {
        session_id: [] for session_id in edge_sessions
    }
    for edge_index, edge in enumerate(edges):
        edge_indexes_by_session[edge.session_id].append(edge_index)
    for row, session_id in enumerate(edge_sessions):
        assignment_matrix[row, edge_indexes_by_session[session_id]] = 1.0

    token_matrix = np.zeros((engine_count, variable_count))
    load_matrix = np.zeros((engine_count, variable_count))
    token_upper = np.empty(engine_count)
    load_upper = np.empty(engine_count)
    for engine_index, engine in enumerate(problem.engines):
        token_matrix[engine_index, token_offset + engine_index] = -1.0
        load_matrix[engine_index, token_offset + engine_index] = 1.0
        load_matrix[engine_index, maximum_load_index] = -1.0
        token_upper[engine_index] = -engine.base_tokens / engine.token_capacity
        load_upper[engine_index] = -(
            engine.base_requests / engine.request_capacity
            + engine.queue_pressure
            + engine.base_prefill / engine.token_capacity
        )
    for edge_index, edge in enumerate(edges):
        engine_index = engine_indexes[edge.engine_url]
        engine = problem.engines[engine_index]
        token_matrix[engine_index, edge_index] = (
            edge.token_increment / engine.token_capacity
        )
        load_matrix[engine_index, edge_index] = (
            edge.request_increment / engine.request_capacity
            + edge.prefill_increment / engine.token_capacity
        )

    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.full(variable_count, np.inf)
    upper_bounds[:edge_count] = 1.0
    for engine_index, engine in enumerate(problem.engines):
        lower_bounds[token_offset + engine_index] = engine.token_usage
    lower_bounds[maximum_load_index] = -np.inf

    return _Model(
        edges=edges,
        edge_sessions=edge_sessions,
        constraints=(
            LinearConstraint(assignment_matrix, 1.0, 1.0),
            LinearConstraint(token_matrix, -np.inf, token_upper),
            LinearConstraint(load_matrix, -np.inf, load_upper),
        ),
        bounds=Bounds(lower_bounds, upper_bounds),
        integrality=np.concatenate(
            (np.ones(edge_count, dtype=int), np.zeros(engine_count + 1, dtype=int))
        ),
        maximum_load_index=maximum_load_index,
    )


def _solve_phase(
    model: _Model,
    objective: np.ndarray,
    constraints: Sequence[LinearConstraint],
    phase: int,
    started_at: float,
    deadline_seconds: float,
):
    elapsed = time.monotonic() - started_at
    remaining = deadline_seconds - elapsed
    if remaining <= 0:
        raise BatchSolverError(phase, None, elapsed, "shared deadline expired")
    result = milp(
        objective,
        integrality=model.integrality,
        bounds=model.bounds,
        constraints=constraints,
        options={"mip_rel_gap": 0.0, "time_limit": remaining},
    )
    if result.status != 0 or result.x is None:
        raise BatchSolverError(
            phase,
            result.status,
            time.monotonic() - started_at,
            result.message,
        )
    return result


def _fixed_objective_constraint(
    objective: np.ndarray,
    optimum: float,
) -> LinearConstraint:
    return LinearConstraint(
        objective,
        optimum - _OPTIMUM_TOLERANCE,
        optimum + _OPTIMUM_TOLERANCE,
    )


def _selected_edges(model: _Model, values: np.ndarray) -> tuple[FeasibleEdge, ...]:
    selected = tuple(
        edge for edge, value in zip(model.edges, values) if value > 0.5
    )
    if len(selected) != len(model.edge_sessions):
        raise RuntimeError("optimal batch MILP result has an incomplete assignment")
    return selected


def _maximum_load(
    problem: BatchProblem,
    selected_edges: Sequence[FeasibleEdge],
) -> float:
    loads = []
    for engine in problem.engines:
        assigned = [edge for edge in selected_edges if edge.engine_url == engine.url]
        request = (
            engine.base_requests
            + sum(edge.request_increment for edge in assigned)
        ) / engine.request_capacity
        token = max(
            (
                engine.base_tokens
                + sum(edge.token_increment for edge in assigned)
            )
            / engine.token_capacity,
            engine.token_usage,
        )
        prefill = (
            engine.base_prefill
            + sum(edge.prefill_increment for edge in assigned)
        ) / engine.token_capacity
        loads.append(request + token + engine.queue_pressure + prefill)
    return max(loads)


def _solution(
    status: SolverStatus,
    problem: BatchProblem,
    selected_edges: Sequence[FeasibleEdge],
    elapsed_seconds: float,
) -> BatchSolution:
    return BatchSolution(
        status=status,
        assignment={edge.session_id: edge.engine_url for edge in selected_edges},
        maximum_load=_maximum_load(problem, selected_edges),
        voluntary_migrations=sum(
            edge.voluntary_migration for edge in selected_edges
        ),
        elapsed_seconds=elapsed_seconds,
    )


def solve_batch_milp(
    problem: BatchProblem,
    deadline_seconds: float = 1.0,
) -> BatchSolution:
    """Solve one batch lexicographically by load, then stable hash."""
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    started_at = time.monotonic()
    model = _build_model(problem)
    variable_count = model.maximum_load_index + 1

    load_objective = np.zeros(variable_count)
    load_objective[model.maximum_load_index] = 1.0
    phase_one = _solve_phase(
        model,
        load_objective,
        model.constraints,
        1,
        started_at,
        deadline_seconds,
    )
    load_constraint = _fixed_objective_constraint(
        load_objective,
        phase_one.fun,
    )

    tie_objective = np.zeros(variable_count)
    for edge_index, edge in enumerate(model.edges):
        tie_objective[edge_index] = _stable_coefficient(
            edge.session_id,
            edge.engine_url,
        )
    phase_two = _solve_phase(
        model,
        tie_objective,
        (*model.constraints, load_constraint),
        2,
        started_at,
        deadline_seconds,
    )
    selected = _selected_edges(model, phase_two.x)
    return _solution(
        SolverStatus.OPTIMAL,
        problem,
        selected,
        time.monotonic() - started_at,
    )


def solve_batch_greedy(problem: BatchProblem) -> BatchSolution:
    """Assign sessions stably, preserving any available non-migration owner."""
    started_at = time.monotonic()
    selected: list[FeasibleEdge] = []
    for session_id in sorted(problem.edges_by_session):
        candidates = problem.edges_by_session[session_id]
        owners = tuple(edge for edge in candidates if not edge.voluntary_migration)
        if owners:
            candidates = owners
        chosen = min(
            candidates,
            key=lambda edge: (
                _maximum_load(problem, (*selected, edge)),
                edge.voluntary_migration,
                _stable_coefficient(edge.session_id, edge.engine_url),
            ),
        )
        selected.append(chosen)
    return _solution(
        SolverStatus.GREEDY,
        problem,
        selected,
        time.monotonic() - started_at,
    )
