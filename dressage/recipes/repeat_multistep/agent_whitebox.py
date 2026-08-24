"""Host-driven append-only loop for deterministic repeat rollouts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from dressage.rollout import multi_segment
from dressage.rollout.artifacts.writer import DEFAULT_WRITER
from dressage.rollout.generate.whitebox_agent import (
    WhiteboxAgent,
    extract_assistant_content,
    extract_finish_reason,
    make_generate,
)

from .tools import (
    REPEAT_INSTRUCTION,
    SYSTEM_PROMPT,
    extract_assistant_message,
)
from .topology import validate_live_topology


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import httpx


def _initial_messages(sample: Any) -> list[dict[str, Any]]:
    prompt = getattr(sample, "prompt", None)
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise ValueError("repeat sample.prompt must contain exactly one message")
    message = prompt[0]
    if (
        not isinstance(message, dict)
        or message.get("role") != "user"
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
    ):
        raise ValueError("repeat sample.prompt must contain one non-empty user message")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        dict(message),
    ]


def _tool_delay_ms() -> int:
    value = os.environ.get("DRESSAGE_REPEAT_MULTISTEP_TOOL_DELAY_MS", "0")
    if not value.isascii() or not value.isdigit():
        raise ValueError(
            "DRESSAGE_REPEAT_MULTISTEP_TOOL_DELAY_MS must be a non-negative integer"
        )
    return int(value)


class RepeatMultiStepWhiteboxAgent(WhiteboxAgent):
    name = "repeat_multistep_whitebox_agent"
    session_prefix = "repeat-ms"

    async def validate_topology(
        self,
        sample: Any,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        metadata = getattr(sample, "metadata", None)
        await validate_live_topology(
            metadata if isinstance(metadata, dict) else {},
            client=client,
        )

    async def setup(self, sample: Any) -> None:
        await self.validate_topology(sample)

    async def rollout(
        self,
        sample: Any,
        sampling_params: dict[str, Any],
    ) -> str:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata

        planned_steps = metadata.get("planned_model_steps")
        if (
            not isinstance(planned_steps, int)
            or isinstance(planned_steps, bool)
            or planned_steps <= 0
        ):
            raise ValueError("metadata.planned_model_steps must be a positive integer")

        tool_delay_ms = _tool_delay_ms()
        metadata["repeat_tool_delay_ms"] = tool_delay_ms
        messages = _initial_messages(sample)
        max_tokens = int(
            sampling_params.get("max_new_tokens")
            or getattr(self.args, "rollout_max_response_len", 1024)
        )
        step_errors: list[dict[str, Any]] = []
        truncated_steps: list[int] = []
        actual_steps = 0
        protocol_success = True
        last_assistant_content = ""

        for step_index in range(planned_steps):
            assistant_message: dict[str, Any] | None = None
            try:
                response = await self.chat(
                    {
                        "model": "proxy-model",
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "stream": False,
                    },
                    turn_id=f"repeat-step-{step_index:04d}",
                )
            except Exception as exc:
                logger.warning(
                    "Repeat step request failed session=%s step_index=%d",
                    self.session_id,
                    step_index,
                    exc_info=True,
                )
                step_errors.append(
                    {
                        "step_index": step_index,
                        "error_type": type(exc).__name__,
                    }
                )
                protocol_success = False
            else:
                actual_steps += 1
                assistant_message = extract_assistant_message(response)
                last_assistant_content = extract_assistant_content(response)
                finish_reason = extract_finish_reason(response)
                if finish_reason == "length":
                    truncated_steps.append(step_index)
                if (
                    assistant_message.get("role") != "assistant"
                    or assistant_message.get("tool_calls")
                    or not last_assistant_content
                    or finish_reason != "stop"
                ):
                    protocol_success = False

            if assistant_message is not None:
                messages.append(assistant_message)
            if step_index + 1 < planned_steps:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"repeat-{step_index:04d}",
                        "content": REPEAT_INSTRUCTION,
                    }
                )
                if tool_delay_ms:
                    await asyncio.sleep(tool_delay_ms / 1000)

        metadata["attempted_model_steps"] = planned_steps
        metadata["actual_model_steps"] = actual_steps
        metadata["completed_step_count"] = actual_steps
        metadata["failed_step_count"] = len(step_errors)
        metadata["repeat_step_errors_json"] = json.dumps(
            step_errors,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        metadata["truncated_step_count"] = len(truncated_steps)
        metadata["repeat_truncated_step_indices"] = truncated_steps
        metadata["protocol_success"] = protocol_success
        return last_assistant_content

    async def _finalize_trajectory(
        self,
        sample: Any,
        agent_response: str,
    ) -> Any:
        try:
            await self.proxy.finalize_session(
                self.session_id,
                instance_id=self.instance_id,
                label=getattr(sample, "label", None),
            )
            payload = await self.proxy.read_trajectory(
                trajectory_id=self.session_id,
                instance_id=self.instance_id,
                drain=True,
            )
            segments = payload.get("data") or []
        except Exception:
            logger.exception(
                "Repeat trajectory drain failed session=%s instance=%s",
                self.session_id,
                self.instance_id,
            )
            return self._abort(sample)

        if not segments:
            logger.warning(
                "No repeat segments returned session=%s instance=%s",
                self.session_id,
                self.instance_id,
            )
            return self._abort(sample)

        try:
            await DEFAULT_WRITER.write_session_payload(
                payload,
                session_id=self.session_id,
                instance_id=self.instance_id,
            )
        except Exception:
            logger.warning(
                "Failed to write repeat session payload session=%s",
                self.session_id,
                exc_info=True,
            )

        base_metadata = dict(getattr(sample, "metadata", None) or {})
        result = multi_segment.expand_segments_to_samples(
            sample,
            segments,
            args=self.args,
            agent_response=agent_response,
            session_id=self.session_id,
            instance_id=self.instance_id,
        )

        try:
            await DEFAULT_WRITER.write_segment_samples(
                sample,
                args=self.args,
                segments=segments,
                base_metadata=base_metadata,
                session_id=self.session_id,
                instance_id=self.instance_id,
                agent_response=agent_response,
            )
        except Exception:
            logger.warning(
                "Failed to write repeat sample artifacts session=%s",
                self.session_id,
                exc_info=True,
            )

        return result


generate = make_generate(RepeatMultiStepWhiteboxAgent)
