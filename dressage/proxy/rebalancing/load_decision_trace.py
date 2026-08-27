"""In-memory observation records for online load-scheduling decisions."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, init=False)
class LoadDecisionTrace:
    """Immutable JSON-compatible payload captured for one scheduling attempt."""

    _payload: Mapping[str, Any]

    def __init__(self, payload: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_payload", deepcopy(dict(payload)))

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(dict(self._payload))


class LoadDecisionHistory:
    """Keep a bounded history of online scheduling attempts."""

    def __init__(self, history_size: int) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self._traces: deque[dict[str, Any]] = deque(maxlen=history_size)

    def record(self, trace: LoadDecisionTrace) -> None:
        self._traces.append(trace.snapshot())

    def snapshot(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._traces))
