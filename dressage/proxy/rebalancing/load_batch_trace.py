"""In-memory observation records for completed load-scheduling batches."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, init=False)
class LoadBatchTrace:
    """Immutable JSON-compatible payload captured for one completed batch."""

    _payload: Mapping[str, Any]

    def __init__(self, payload: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_payload", deepcopy(dict(payload)))

    def snapshot(self) -> dict[str, Any]:
        """Return an independent copy of this trace payload."""
        return deepcopy(dict(self._payload))


class LoadBatchHistory:
    """Keep a bounded, in-memory history of completed batch traces."""

    def __init__(self, history_size: int) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self._traces: deque[dict[str, Any]] = deque(maxlen=history_size)

    def record(self, trace: LoadBatchTrace) -> None:
        """Store an independent copy of a completed batch trace."""
        self._traces.append(trace.snapshot())

    def snapshot(self) -> list[dict[str, Any]]:
        """Return independent copies of the retained trace payloads."""
        return deepcopy(list(self._traces))
