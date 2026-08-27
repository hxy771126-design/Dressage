"""Atomic persistence for engine-rebalancing calibration snapshots."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return normalized or "dressage"


class CalibrationSnapshotStore:
    """Write immutable, run-scoped JSON snapshots without partial files."""

    def __init__(
        self,
        *,
        root: str | os.PathLike[str],
        run_name: str,
        interval_requests: int = 128,
        started_at: float | None = None,
        pid: int | None = None,
    ) -> None:
        if interval_requests <= 0:
            raise ValueError("interval_requests must be positive")
        started = time.time() if started_at is None else float(started_at)
        timestamp = datetime.fromtimestamp(started, timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        self.interval_requests = int(interval_requests)
        self.run_id = f"{timestamp}-{os.getpid() if pid is None else int(pid)}"
        self.directory = (
            Path(root).expanduser() / _safe_component(run_name) / self.run_id
        )
        self._write_lock = asyncio.Lock()

    async def write(
        self,
        *,
        kind: str,
        online_request_count: int,
        payload: Mapping[str, Any],
    ) -> Path:
        if kind == "initial":
            filename = "initial.json"
        elif kind == "final":
            filename = "final.json"
        elif kind == "periodic":
            filename = f"request-{max(0, int(online_request_count)):09d}.json"
        else:
            raise ValueError(f"unsupported calibration snapshot kind: {kind!r}")

        captured_at = time.time()
        count = max(0, int(online_request_count))
        document = dict(payload)
        document["snapshot_type"] = kind
        document["snapshot_time"] = captured_at
        document["online_request_count"] = count
        async with self._write_lock:
            return await asyncio.to_thread(
                self._write_atomic,
                self.directory / filename,
                document,
            )

    @staticmethod
    def _write_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return path
