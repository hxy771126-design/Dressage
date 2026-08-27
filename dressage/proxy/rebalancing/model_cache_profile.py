"""Model-side byte estimates used only for context recovery timing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


_DTYPE_BYTES = {
    "float64": 8,
    "fp64": 8,
    "float32": 4,
    "fp32": 4,
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "float8": 1,
    "fp8": 1,
    "int8": 1,
}


def canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dtype_nbytes(value: Any, *, default: int = 2) -> int:
    if value is None:
        return default
    text = str(value).lower().replace("torch.", "")
    return _DTYPE_BYTES.get(text, default)


@dataclass(frozen=True)
class ModelCacheProfile:
    fingerprint: str
    full_layers: int
    full_kv_heads: int
    full_head_dim: int
    full_dtype_bytes: int
    swa_layers: int = 0
    swa_kv_heads: int = 0
    swa_head_dim: int = 0
    swa_dtype_bytes: int = 2
    page_size: int = 1
    swa_window_size: int | None = None
    state_bytes_per_checkpoint: int = 0
    metadata_bytes_estimate: int = 0
    confidence: str = "exact"

    @property
    def available(self) -> bool:
        return (
            self.full_layers > 0
            or self.swa_layers > 0
            or self.state_bytes_per_checkpoint > 0
        )

    def page_round_up(self, tokens: int) -> int:
        size = max(1, int(self.page_size))
        return int(math.ceil(max(0, tokens) / size) * size)

    def estimate_bytes(self, context_tokens: int) -> int:
        length = max(0, int(context_tokens))
        full = (
            length
            * 2
            * max(0, self.full_layers)
            * max(0, self.full_kv_heads)
            * max(0, self.full_head_dim)
            * max(1, self.full_dtype_bytes)
        )
        swa_resident = 0
        if self.swa_layers > 0 and self.swa_window_size:
            swa_resident = self.page_round_up(min(length, self.swa_window_size))
        swa = (
            swa_resident
            * 2
            * max(0, self.swa_layers)
            * max(0, self.swa_kv_heads)
            * max(0, self.swa_head_dim)
            * max(1, self.swa_dtype_bytes)
        )
        # HiCache keeps one recoverable tail state for a non-empty hybrid
        # prefix.  ``mamba_track_interval`` controls where that state is
        # refreshed; it does not turn every historical checkpoint into a
        # simultaneously resident cache object.
        state = self.state_bytes_per_checkpoint if length > 0 else 0
        return int(full + swa + state + max(0, self.metadata_bytes_estimate))

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["available"] = self.available
        return payload

    @classmethod
    def from_model_config(
        cls,
        model_config: Mapping[str, Any] | Any,
        *,
        deployment: Mapping[str, Any],
    ) -> "ModelCacheProfile":
        config = (
            dict(model_config)
            if isinstance(model_config, Mapping)
            else dict(getattr(model_config, "to_dict")())
        )
        text_config = config.get("text_config")
        if isinstance(text_config, Mapping):
            # Multimodal Hugging Face configs (including Qwen3.5) keep the
            # language-model cache geometry under text_config.
            config = dict(text_config)
        hidden_size = int(config.get("hidden_size") or 0)
        attention_heads = int(config.get("num_attention_heads") or 1)
        head_dim = int(config.get("head_dim") or hidden_size // max(1, attention_heads))
        kv_heads = int(config.get("num_key_value_heads") or attention_heads)
        layer_count = int(
            config.get("num_hidden_layers") or config.get("num_layers") or 0
        )
        layer_types = config.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != layer_count:
            layer_types = []

        swa_markers = {"sliding_attention", "swa", "sliding_window"}
        state_markers = {"linear_attention", "mamba", "gdn"}
        if layer_types:
            swa_layers = sum(str(item).lower() in swa_markers for item in layer_types)
            state_layers = sum(
                str(item).lower() in state_markers for item in layer_types
            )
            full_layers = max(0, layer_count - swa_layers - state_layers)
        else:
            swa_layers = 0
            state_layers = 0
            full_layers = layer_count

        dtype = (
            deployment.get("kv_dtype")
            or deployment.get("dtype")
            or config.get("torch_dtype")
            or config.get("dtype")
        )
        state_dtype = deployment.get("state_dtype") or dtype
        page_size = int(deployment.get("page_size") or config.get("page_size") or 1)
        swa_window = deployment.get("swa_window_size") or config.get("sliding_window")

        state_bytes = 0
        confidence = "exact"
        if state_layers:
            linear_key_heads = int(config.get("linear_num_key_heads") or 0)
            linear_value_heads = int(config.get("linear_num_value_heads") or 0)
            linear_key_dim = int(config.get("linear_key_head_dim") or 0)
            linear_value_dim = int(config.get("linear_value_head_dim") or 0)
            linear_conv_kernel = int(config.get("linear_conv_kernel_dim") or 0)
            state_size = int(
                config.get("mamba_d_state") or config.get("state_size") or 0
            )
            conv_size = int(
                config.get("mamba_d_conv") or config.get("conv_kernel") or 0
            )
            expand = float(config.get("mamba_expand") or config.get("expand") or 1.0)
            inner_dim = int(config.get("mamba_inner_dim") or hidden_size * expand)
            if all(
                value > 0
                for value in (
                    linear_key_heads,
                    linear_value_heads,
                    linear_key_dim,
                    linear_value_dim,
                    linear_conv_kernel,
                )
            ):
                temporal_elements = (
                    linear_value_heads * linear_value_dim * linear_key_dim
                )
                conv_elements = (
                    2 * linear_key_heads * linear_key_dim
                    + linear_value_heads * linear_value_dim
                ) * max(1, linear_conv_kernel - 1)
                temporal_dtype = config.get("mamba_ssm_dtype") or state_dtype
                state_bytes = state_layers * (
                    temporal_elements * dtype_nbytes(temporal_dtype)
                    + conv_elements * dtype_nbytes(dtype)
                )
                confidence = "config"
            elif inner_dim > 0 and (state_size > 0 or conv_size > 0):
                per_layer_elements = inner_dim * (state_size + conv_size)
                state_bytes = (
                    state_layers * per_layer_elements * dtype_nbytes(state_dtype)
                )
            else:
                # A conservative family-level bound.  Unknown layouts must not
                # make context recovery look artificially cheap.
                state_bytes = (
                    state_layers * max(1, hidden_size) * 64 * dtype_nbytes(state_dtype)
                )
                confidence = "upper_bound"

        fingerprint = canonical_fingerprint(dict(deployment))
        return cls(
            fingerprint=fingerprint,
            full_layers=full_layers,
            full_kv_heads=kv_heads,
            full_head_dim=head_dim,
            full_dtype_bytes=dtype_nbytes(dtype),
            swa_layers=swa_layers,
            swa_kv_heads=kv_heads,
            swa_head_dim=head_dim,
            swa_dtype_bytes=dtype_nbytes(dtype),
            page_size=page_size,
            swa_window_size=None if swa_window is None else int(swa_window),
            state_bytes_per_checkpoint=state_bytes,
            metadata_bytes_estimate=int(deployment.get("metadata_bytes_estimate") or 0),
            confidence=confidence,
        )
