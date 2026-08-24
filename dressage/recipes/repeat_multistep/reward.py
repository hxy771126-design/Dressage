"""Binary trajectory reward for deterministic repeat rollouts."""

from __future__ import annotations

from typing import Any

from dressage.reward import register_reward


@register_reward("repeat_multistep")
def repeat_multistep(sample: Any, *, args: Any = None, **kwargs: Any) -> float:
    del args, kwargs
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return 0.0

    planned = metadata.get("planned_model_steps")
    if not isinstance(planned, int) or isinstance(planned, bool) or planned <= 0:
        return 0.0
    if metadata.get("attempted_model_steps") != planned:
        return 0.0
    if metadata.get("actual_model_steps") != planned:
        return 0.0
    if metadata.get("protocol_success") is not True:
        return 0.0
    if metadata.get("failed_step_count") != 0:
        return 0.0
    if metadata.get("truncated_step_count") != 0:
        return 0.0
    return 1.0
