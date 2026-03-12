#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

from openai import OpenAI
from openai.types.responses import Response

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_ITERATIONS = 20
DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_PROMPT = (
    "Use the computer tool exactly once to take a screenshot, then reply with DONE."
)
COMPUTER_TOOL = {"type": "computer"}
DUMMY_TOOL = {
    "type": "function",
    "name": "dummy_helper",
    "description": (
        "DO NOT CALL THIS TOOL unless the user explicitly asks for dummy_helper by "
        "name. This tool is unrelated to browser, screenshot, and computer tasks. "
        "Never use it for computer use."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Arbitrary text. Do not call this tool unless the user "
                    "explicitly requested dummy_helper."
                ),
            }
        },
    },
    "strict": True,
}


@dataclass
class TrialResult:
    trial: int
    dummy_call_count: int
    computer_call_count: int
    follow_up_requests: int
    finished_with: str
    first_output_types: list[str]
    step_output_types: list[list[str]]
    error: str | None = None
    terminal_text: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Direct OpenAI Responses API repro for computer-tool selection when an "
            "unrelated function tool is also present."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between trials to reduce burstiness.",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print the raw output items for every step.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="Per-request timeout passed to the OpenAI client.",
    )
    return parser.parse_args()


def _require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key == "":
        raise SystemExit("OPENAI_API_KEY must be set")
    return api_key


def _output_items_to_json(response: Response) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in response.output:
        try:
            dumped = item.model_dump(mode="json")
        except Exception:
            continue
        if isinstance(dumped, dict):
            items.append(dumped)
    return items


def _extract_message_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text != "":
            parts.append(text)
    return "".join(parts)


def _make_function_outputs(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for item in items:
        if item.get("type") != "function_call":
            continue
        if item.get("name") != "dummy_helper":
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or call_id == "":
            continue
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": "ok",
            }
        )
    return outputs


def run_trial(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    max_steps: int,
    max_output_tokens: int,
    dump_json: bool,
    trial: int,
) -> TrialResult:
    dummy_call_count = 0
    computer_call_count = 0
    follow_up_requests = 0
    first_output_types: list[str] = []
    step_output_types: list[list[str]] = []
    terminal_text: str | None = None

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            tools=[COMPUTER_TOOL, DUMMY_TOOL],
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        return TrialResult(
            trial=trial,
            dummy_call_count=0,
            computer_call_count=0,
            follow_up_requests=0,
            finished_with="request_error",
            first_output_types=[],
            step_output_types=[],
            error=f"{type(exc).__name__}: {exc}",
        )

    for step_index in range(max_steps):
        items = _output_items_to_json(response)
        output_types = [
            item_type
            for item in items
            if isinstance((item_type := item.get("type")), str)
        ]
        if step_index == 0:
            first_output_types = output_types.copy()
        step_output_types.append(output_types)

        if dump_json:
            print(f"\ntrial={trial} step={step_index}")
            print(json.dumps(items, ensure_ascii=False, indent=2))

        current_dummy_calls = sum(
            1
            for item in items
            if item.get("type") == "function_call"
            and item.get("name") == "dummy_helper"
        )
        current_computer_calls = sum(
            1 for item in items if item.get("type") == "computer_call"
        )
        dummy_call_count += current_dummy_calls
        computer_call_count += current_computer_calls

        if current_computer_calls > 0:
            return TrialResult(
                trial=trial,
                dummy_call_count=dummy_call_count,
                computer_call_count=computer_call_count,
                follow_up_requests=follow_up_requests,
                finished_with="computer_call",
                first_output_types=first_output_types,
                step_output_types=step_output_types,
            )

        final_messages = [
            item
            for item in items
            if item.get("type") == "message" and item.get("phase") == "final_answer"
        ]
        if len(final_messages) > 0:
            terminal_text = _extract_message_text(final_messages[0])
            return TrialResult(
                trial=trial,
                dummy_call_count=dummy_call_count,
                computer_call_count=computer_call_count,
                follow_up_requests=follow_up_requests,
                finished_with="final_answer",
                first_output_types=first_output_types,
                step_output_types=step_output_types,
                terminal_text=terminal_text,
            )

        function_outputs = _make_function_outputs(items)
        if len(function_outputs) == 0:
            return TrialResult(
                trial=trial,
                dummy_call_count=dummy_call_count,
                computer_call_count=computer_call_count,
                follow_up_requests=follow_up_requests,
                finished_with="stopped_without_tool",
                first_output_types=first_output_types,
                step_output_types=step_output_types,
            )

        follow_up_requests += 1
        try:
            response = client.responses.create(
                model=model,
                previous_response_id=response.id,
                input=function_outputs,
                tools=[COMPUTER_TOOL, DUMMY_TOOL],
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            return TrialResult(
                trial=trial,
                dummy_call_count=dummy_call_count,
                computer_call_count=computer_call_count,
                follow_up_requests=follow_up_requests,
                finished_with="follow_up_error",
                first_output_types=first_output_types,
                step_output_types=step_output_types,
                error=f"{type(exc).__name__}: {exc}",
            )

    return TrialResult(
        trial=trial,
        dummy_call_count=dummy_call_count,
        computer_call_count=computer_call_count,
        follow_up_requests=follow_up_requests,
        finished_with="max_steps",
        first_output_types=first_output_types,
        step_output_types=step_output_types,
        terminal_text=terminal_text,
    )


def _print_trial(result: TrialResult) -> None:
    detail = (
        f"trial={result.trial:02d} "
        f"dummy_calls={result.dummy_call_count} "
        f"computer_calls={result.computer_call_count} "
        f"follow_ups={result.follow_up_requests} "
        f"finished_with={result.finished_with} "
        f"first_output_types={result.first_output_types} "
        f"step_output_types={result.step_output_types}"
    )
    if result.terminal_text:
        detail += f" terminal_text={result.terminal_text!r}"
    if result.error:
        detail += f" error={result.error!r}"
    print(detail, flush=True)


def _print_summary(results: list[TrialResult]) -> None:
    dummy_counts = [result.dummy_call_count for result in results]
    computer_trials = sum(
        1 for result in results if result.finished_with == "computer_call"
    )
    final_without_computer = sum(
        1
        for result in results
        if result.finished_with == "final_answer" and result.computer_call_count == 0
    )
    any_dummy_trials = sum(1 for result in results if result.dummy_call_count > 0)
    errors = [result for result in results if result.error is not None]

    print("\nSummary")
    print(f"trials={len(results)}", flush=True)
    print(f"trials_with_dummy_calls={any_dummy_trials}", flush=True)
    print(f"total_dummy_calls={sum(dummy_counts)}", flush=True)
    print(f"average_dummy_calls={mean(dummy_counts):.2f}", flush=True)
    print(f"max_dummy_calls={max(dummy_counts) if dummy_counts else 0}", flush=True)
    print(f"trials_reaching_computer_call={computer_trials}", flush=True)
    print(
        f"trials_finishing_without_computer_call={final_without_computer}", flush=True
    )
    print(f"trials_with_errors={len(errors)}", flush=True)
    if len(errors) > 0:
        print(
            "error_trials="
            + json.dumps(
                [
                    {
                        "trial": result.trial,
                        "finished_with": result.finished_with,
                        "error": result.error,
                    }
                    for result in errors
                ],
                ensure_ascii=False,
            ),
            flush=True,
        )


def main() -> int:
    args = _parse_args()
    api_key = _require_api_key()
    client = OpenAI(api_key=api_key, timeout=args.request_timeout_seconds)

    results: list[TrialResult] = []
    for trial in range(1, args.iterations + 1):
        result = run_trial(
            client=client,
            model=args.model,
            prompt=args.prompt,
            max_steps=args.max_steps,
            max_output_tokens=args.max_output_tokens,
            dump_json=args.dump_json,
            trial=trial,
        )
        results.append(result)
        _print_trial(result)
        if args.sleep_seconds > 0 and trial < args.iterations:
            time.sleep(args.sleep_seconds)

    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
