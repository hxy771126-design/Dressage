"""Deterministic sticky-aware greedy placement for one online step."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence


SCORE_TOLERANCE = 1e-7
_RATIO_EPSILON = 1e-9


@dataclass(frozen=True)
class EngineBaseline:
    """Observed Engine load plus unobserved local scheduling deltas."""

    url: str
    base_requests: float
    base_tokens: float
    base_queue: float
    request_capacity: float
    token_capacity: float
    token_usage: float

    def __post_init__(self) -> None:
        if self.request_capacity <= 0 or self.token_capacity <= 0:
            raise ValueError(f"Engine '{self.url}' capacities must be positive")
        values = (
            self.base_requests,
            self.base_tokens,
            self.base_queue,
            self.request_capacity,
            self.token_capacity,
            self.token_usage,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Engine '{self.url}' pressure inputs must be finite")
        if any(value < 0 for value in values):
            raise ValueError(
                f"Engine '{self.url}' pressure inputs must be non-negative"
            )


@dataclass(frozen=True)
class FeasibleEdge:
    """Projected pressure increments for routing one step to one Engine."""

    session_id: str
    engine_url: str
    queue_increment: float
    token_increment: float

    def __post_init__(self) -> None:
        increments = (
            self.queue_increment,
            self.token_increment,
        )
        if not all(math.isfinite(value) and value >= 0 for value in increments):
            raise ValueError("edge pressure increments must be finite and non-negative")


class GreedyStepKind(str, Enum):
    FIXED_OWNER = "fixed_owner"
    MANDATORY_FAILOVER = "mandatory_failover"
    NEW_SESSION = "new_session"
    EXISTING_SESSION = "existing_session"


@dataclass(frozen=True)
class EnginePressure:
    request: float
    token: float
    queue: float
    total: float


@dataclass(frozen=True)
class GreedyStepDecision:
    decision_reason: str
    owner_projected_score: float | None
    candidate_projected_scores: Mapping[str, float]
    best_target: str
    best_projected_score: float
    selected_target: str
    selected_projected_score: float
    improvement_ratio: float | None
    required_improvement_ratio: float | None
    threshold_met: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_projected_scores",
            MappingProxyType(dict(self.candidate_projected_scores)),
        )


def projected_pressure(
    engine: EngineBaseline,
    *,
    queue_increment: float = 0.0,
    token_increment: float = 0.0,
) -> EnginePressure:
    """Return additive request + token + queue Pressure."""

    request = engine.base_requests / engine.request_capacity
    token = max(
        (engine.base_tokens + token_increment) / engine.token_capacity,
        engine.token_usage,
    )
    queue = (engine.base_queue + queue_increment) / engine.request_capacity
    return EnginePressure(
        request=request,
        token=token,
        queue=queue,
        total=request + token + queue,
    )


def stable_engine_rank(session_id: str, engine_url: str) -> int:
    digest = hashlib.sha256(
        session_id.encode("utf-8") + b"\0" + engine_url.encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big")


def choose_stable_engine(session_id: str, engine_urls: Sequence[str]) -> str:
    if not engine_urls:
        raise ValueError("at least one Engine is required")
    return min(
        engine_urls,
        key=lambda url: (stable_engine_rank(session_id, url), url),
    )


def choose_greedy_step(
    *,
    session_id: str,
    kind: GreedyStepKind,
    owner_engine_url: str | None,
    engines: Sequence[EngineBaseline],
    edges: Sequence[FeasibleEdge],
    min_load_improvement_ratio: float,
) -> GreedyStepDecision:
    """Choose one step using the latest effective Engine loads."""

    if not math.isfinite(min_load_improvement_ratio) or not (
        0.0 <= min_load_improvement_ratio <= 1.0
    ):
        raise ValueError("min load improvement ratio must be within [0, 1]")
    baselines = {engine.url: engine for engine in engines}
    frozen_edges = tuple(edges)
    if not frozen_edges:
        raise ValueError("step must contain at least one feasible edge")
    if len({edge.engine_url for edge in frozen_edges}) != len(frozen_edges):
        raise ValueError("step contains duplicate Engine edges")
    if any(edge.session_id != session_id for edge in frozen_edges):
        raise ValueError("edge session does not match step session")
    if any(edge.engine_url not in baselines for edge in frozen_edges):
        raise ValueError("edge references an unknown Engine")

    owner_edge = next(
        (edge for edge in frozen_edges if edge.engine_url == owner_engine_url),
        None,
    )
    if kind is GreedyStepKind.NEW_SESSION and owner_engine_url is not None:
        raise ValueError("new session must not have an owner")
    if kind in {GreedyStepKind.FIXED_OWNER, GreedyStepKind.EXISTING_SESSION}:
        if owner_engine_url is None or owner_edge is None:
            raise ValueError("sticky step must contain its owner edge")
    if kind is GreedyStepKind.MANDATORY_FAILOVER and owner_edge is not None:
        raise ValueError("mandatory failover cannot retain its owner")

    scores = {
        edge.engine_url: projected_pressure(
            baselines[edge.engine_url],
            queue_increment=edge.queue_increment,
            token_increment=edge.token_increment,
        ).total
        for edge in frozen_edges
    }
    minimum = min(scores.values())
    tied = [
        edge
        for edge in frozen_edges
        if scores[edge.engine_url] <= minimum + SCORE_TOLERANCE
    ]
    best = next(
        (edge for edge in tied if edge.engine_url == owner_engine_url),
        None,
    ) or min(
        tied,
        key=lambda edge: (stable_engine_rank(session_id, edge.engine_url), edge.engine_url),
    )

    if kind is GreedyStepKind.FIXED_OWNER:
        selected = owner_edge
        reason = "fixed_owner"
        owner_score = scores[owner_engine_url]
        improvement = None
        threshold_met = None
        required = None
    elif kind in {GreedyStepKind.NEW_SESSION, GreedyStepKind.MANDATORY_FAILOVER}:
        selected = best
        reason = (
            "mandatory_failover"
            if kind is GreedyStepKind.MANDATORY_FAILOVER
            else "lowest_projected_pressure"
        )
        owner_score = None
        improvement = None
        threshold_met = None
        required = None
    else:
        if owner_edge is None or owner_engine_url is None:
            raise RuntimeError("existing session is missing its owner")
        owner_score = scores[owner_engine_url]
        improvement = max(
            0.0,
            (owner_score - scores[best.engine_url])
            / max(owner_score, _RATIO_EPSILON),
        )
        threshold_met = (
            best.engine_url != owner_engine_url
            and scores[best.engine_url] < owner_score - SCORE_TOLERANCE
            and improvement >= min_load_improvement_ratio
        )
        selected = best if threshold_met else owner_edge
        reason = (
            "load_improvement_threshold_met"
            if threshold_met
            else "load_improvement_below_threshold"
        )
        required = min_load_improvement_ratio

    if selected is None:  # Only possible if validation above regresses.
        raise RuntimeError("greedy step did not select an Engine")
    return GreedyStepDecision(
        decision_reason=reason,
        owner_projected_score=owner_score,
        candidate_projected_scores=scores,
        best_target=best.engine_url,
        best_projected_score=scores[best.engine_url],
        selected_target=selected.engine_url,
        selected_projected_score=scores[selected.engine_url],
        improvement_ratio=improvement,
        required_improvement_ratio=required,
        threshold_met=threshold_met,
    )
