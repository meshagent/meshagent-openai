from __future__ import annotations

from typing import Any
import logging

from opentelemetry import metrics
from pydantic import BaseModel

logger = logging.getLogger("openai_agent")
_usage_meters: dict[str, object] = {}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _flatten_usage(usage: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}

    for key, value in usage.items():
        try:
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        continue
                    nested_number = _to_float(nested_value)
                    if nested_number is not None:
                        flattened[nested_key] = nested_number
                continue

            number = _to_float(value)
            if number is not None:
                flattened[key] = number
        except Exception as error:
            logger.warning("unexpected usage key %s:%s", key, value, exc_info=error)

    return flattened


def normalize_openai_usage(usage: object) -> dict[str, Any] | None:
    if usage is None:
        return None

    if isinstance(usage, BaseModel):
        try:
            return usage.model_dump(mode="json")
        except Exception:
            return None

    if isinstance(usage, dict):
        return usage

    return None


def preprocess_openai_usage(
    *, model: str, usage: dict[str, Any]
) -> dict[str, float] | None:
    del model

    if not isinstance(usage, dict):
        return None

    flattened = _flatten_usage(usage)

    input_details = usage.get("input_token_details")
    if not isinstance(input_details, dict):
        input_details = usage.get("input_tokens_details")
        if not isinstance(input_details, dict):
            input_details = None

    output_details = usage.get("output_token_details")
    if not isinstance(output_details, dict):
        output_details = usage.get("output_tokens_details")
        if not isinstance(output_details, dict):
            output_details = None

    out: dict[str, float] = {}
    saw_modal_details = False

    if input_details is not None:
        cached_details = input_details.get("cached_tokens_details")
        if not isinstance(cached_details, dict):
            cached_details = {}

        cached_text = _to_float(cached_details.get("text_tokens"))
        cached_audio = _to_float(cached_details.get("audio_tokens"))
        cached_image = _to_float(cached_details.get("image_tokens"))

        cached_total = _to_float(input_details.get("cached_tokens"))
        if cached_text is None and cached_total is not None:
            cached_text = cached_total

        if cached_text is not None and cached_text > 0:
            out["cached_tokens"] = cached_text
        if cached_audio is not None and cached_audio > 0:
            out["audio_cached_tokens"] = cached_audio
        if cached_image is not None and cached_image > 0:
            out["image_cached_tokens"] = cached_image

        text_total = _to_float(input_details.get("text_tokens"))
        audio_total = _to_float(input_details.get("audio_tokens"))
        image_total = _to_float(input_details.get("image_tokens"))

        if text_total is not None:
            saw_modal_details = True
            uncached_text = max(0.0, text_total - (cached_text or 0.0))
            if uncached_text > 0:
                out["input_tokens"] = uncached_text

        if audio_total is not None:
            saw_modal_details = True
            uncached_audio = max(0.0, audio_total - (cached_audio or 0.0))
            if uncached_audio > 0:
                out["audio_input_tokens"] = uncached_audio

        if image_total is not None:
            saw_modal_details = True
            uncached_image = max(0.0, image_total - (cached_image or 0.0))
            if uncached_image > 0:
                out["image_input_tokens"] = uncached_image

    if output_details is not None:
        text_output = _to_float(output_details.get("text_tokens"))
        audio_output = _to_float(output_details.get("audio_tokens"))
        image_output = _to_float(output_details.get("image_tokens"))

        if text_output is not None:
            saw_modal_details = True
            if text_output > 0:
                out["output_tokens"] = text_output
        if audio_output is not None:
            saw_modal_details = True
            if audio_output > 0:
                out["audio_output_tokens"] = audio_output
        if image_output is not None:
            saw_modal_details = True
            if image_output > 0:
                out["image_output_tokens"] = image_output

    if saw_modal_details:
        if "output_tokens" not in out:
            aggregate_output = _to_float(usage.get("output_tokens"))
            if aggregate_output is not None and aggregate_output > 0:
                out["output_tokens"] = aggregate_output

        if "reasoning_tokens" in flattened:
            out["reasoning_tokens"] = float(flattened["reasoning_tokens"])

        return out

    for key, value in flattened.items():
        normalized_key = key
        if normalized_key == "prompt_tokens":
            normalized_key = "input_tokens"
        if normalized_key == "completion_tokens":
            normalized_key = "output_tokens"
        out[normalized_key] = float(value)

    return out


def add_usage_metrics(*, totals: dict[str, float], usage: dict[str, float]) -> None:
    for key, value in usage.items():
        totals[key] = float(totals.get(key, 0.0)) + float(value)


def _get_counter_meter(name: str):
    meter = _usage_meters.get(name)
    if meter is None:
        usage_meter = metrics.get_meter("meshagent.usage")
        meter = usage_meter.create_counter(name, "tokens")
        _usage_meters[name] = meter
    return meter


def track_otel_usage_metrics(
    *, model: str, provider: str, tokens: dict[str, float]
) -> None:
    for token_name, total in tokens.items():
        meter = _get_counter_meter(token_name)
        meter.add(total, {"model": model, "provider": provider})


# Backwards compatibility for older imports.
_preprocess_openai = preprocess_openai_usage
