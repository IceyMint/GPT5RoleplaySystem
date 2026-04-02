#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openai import OpenAI
from openai.lib._parsing._completions import type_to_response_format_param

from gpt5_roleplay_system.config import load_config
from gpt5_roleplay_system.llm import (
    StructuredBundle,
    _extract_prompt_cache_usage,
    _field_as_dict,
    _field_get,
    _json_safe,
    _message_content_text,
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256_prefix(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _json_dump(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=True, indent=2, sort_keys=True)


def _structured_response_format() -> dict[str, Any]:
    response_format = type_to_response_format_param(StructuredBundle)
    if isinstance(response_format, dict):
        return response_format
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "StructuredBundle",
            "schema": StructuredBundle.model_json_schema(),
            "strict": True,
        },
    }


def _provider_payload(config: Any) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    if config.llm.provider_order:
        payload["order"] = [str(item).strip() for item in config.llm.provider_order if str(item).strip()]
    if config.llm.provider_allow_fallbacks is not None:
        payload["allow_fallbacks"] = bool(config.llm.provider_allow_fallbacks)
    return payload or None


def _extra_body(config: Any) -> dict[str, Any] | None:
    extra_body: dict[str, Any] = {}
    if config.llm.reasoning:
        extra_body["include_reasoning"] = True
        extra_body["reasoning_effort"] = str(config.llm.reasoning)
    provider = _provider_payload(config)
    if provider:
        extra_body["provider"] = provider
    return extra_body or None


def _completion_usage_payload(completion: Any) -> dict[str, Any]:
    usage = _field_get(completion, "usage")
    return _field_as_dict(usage)


def _print_header(
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    extra_body: dict[str, Any] | None,
    response_format: dict[str, Any] | None,
) -> None:
    print(f"model={model}")
    print(f"base_url={base_url}")
    print(f"mode={args.mode}")
    print(f"repeats={args.repeats}")
    print(f"delay_seconds={args.delay}")
    print(f"max_tokens={max_tokens}")
    print(f"temperature={temperature}")
    print(f"system_chars={len(system_prompt)} system_sha256={_sha256_prefix(system_prompt)}")
    print(f"user_chars={len(user_prompt)} user_sha256={_sha256_prefix(user_prompt)}")
    if extra_body:
        print("extra_body=")
        print(_json_dump(extra_body))
    else:
        print("extra_body=null")
    if response_format:
        print(f"response_format_sha256={_sha256_prefix(json.dumps(response_format, ensure_ascii=True, sort_keys=True))}")
        print("response_format=")
        print(_json_dump(response_format))
    else:
        print("response_format=null")
    print("---")


def _request_once(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    extra_body: dict[str, Any] | None,
    mode: str,
) -> tuple[Any | None, Exception | None]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        if mode == "parse":
            completion = client.chat.completions.parse(response_format=StructuredBundle, **kwargs)
        elif mode == "create_schema":
            completion = client.chat.completions.create(response_format=_structured_response_format(), **kwargs)
        else:
            completion = client.chat.completions.create(**kwargs)
        return completion, None
    except Exception as exc:  # pragma: no cover - live provider behavior
        completion = getattr(exc, "completion", None)
        return completion, exc


def _validate_structured_completion(completion: Any) -> tuple[StructuredBundle | None, Exception | None]:
    message = _field_get(_field_get(completion, "choices", [{}])[0], "message")
    raw_text = _message_content_text(message)
    if not raw_text:
        return None, ValueError("No response text available for StructuredBundle.model_validate_json().")
    try:
        return StructuredBundle.model_validate_json(raw_text), None
    except Exception as exc:
        return None, exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay an exact prompt pair against the live structured LLM path.")
    parser.add_argument("--system-file", required=True, help="Path to the exact system prompt text file.")
    parser.add_argument("--user-file", required=True, help="Path to the exact user prompt text file.")
    parser.add_argument("--config", default="", help="Optional config path. Defaults to the repo config loader behavior.")
    parser.add_argument("--model", default="", help="Optional model override. Defaults to llm.model from config.")
    parser.add_argument("--base-url", default="", help="Optional base URL override. Defaults to llm.base_url from config.")
    parser.add_argument("--api-key", default="", help="Optional API key override. Defaults to llm.api_key from config/env.")
    parser.add_argument("--repeats", type=int, default=2, help="Number of identical requests to send.")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between identical requests.")
    parser.add_argument(
        "--mode",
        choices=("parse", "create", "create_schema"),
        default="parse",
        help="Request path to replay.",
    )
    parser.add_argument("--max-tokens", type=int, default=-1, help="Optional max_tokens override.")
    parser.add_argument("--temperature", type=float, default=float("nan"), help="Optional temperature override.")
    parser.add_argument("--print-response", action="store_true", help="Print the raw response content for each request.")
    args = parser.parse_args()

    config = load_config(args.config or None)
    model = args.model or config.llm.model
    base_url = args.base_url or config.llm.base_url
    api_key = args.api_key or config.llm.api_key
    max_tokens = int(args.max_tokens if args.max_tokens >= 0 else config.llm.max_tokens)
    temperature = float(args.temperature if args.temperature == args.temperature else config.llm.temperature)

    if not api_key:
        print("No API key available from config or --api-key.", file=sys.stderr)
        return 2

    system_prompt = _read_text(Path(args.system_file))
    user_prompt = _read_text(Path(args.user_file))
    extra_body = _extra_body(config)
    response_format = _structured_response_format() if args.mode == "create_schema" else None

    actual_base_url = base_url or None
    client = OpenAI(api_key=api_key, base_url=actual_base_url, timeout=float(config.llm.timeout_seconds))

    _print_header(
        args,
        system_prompt,
        user_prompt,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
        response_format=response_format,
    )

    for index in range(args.repeats):
        if index > 0 and args.delay > 0:
            time.sleep(args.delay)

        started_at = time.time()
        completion, error = _request_once(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
            mode=args.mode,
        )
        elapsed = time.time() - started_at

        print(f"request={index + 1}")
        print(f"elapsed_seconds={elapsed:.3f}")
        if error is not None:
            print(f"error_class={error.__class__.__name__}")
            print(f"error_text={error}")

        if completion is None:
            print("completion=null")
            print("---")
            continue

        sample = _extract_prompt_cache_usage(completion)
        usage_payload = _completion_usage_payload(completion)
        print(f"completion_id={_field_get(completion, 'id', '')}")
        print(f"completion_model={_field_get(completion, 'model', '')}")
        print(f"finish_reason={_field_get(_field_get(completion, 'choices', [{}])[0], 'finish_reason', '')}")
        print("usage=")
        print(_json_dump(usage_payload))
        if sample is None:
            print("cache_sample=null")
        else:
            print(
                "cache_sample="
                + _json_dump(
                    {
                        "prompt_tokens": sample.prompt_tokens,
                        "completion_tokens": sample.completion_tokens,
                        "total_tokens": sample.total_tokens,
                        "cached_read_tokens": sample.cached_read_tokens,
                        "cache_write_tokens": sample.cache_write_tokens,
                        "uncached_prompt_tokens": sample.uncached_prompt_tokens,
                        "cache_discount": sample.cache_discount,
                    }
                )
            )
        if args.mode == "create_schema":
            parsed_bundle, validation_error = _validate_structured_completion(completion)
            if validation_error is None and parsed_bundle is not None:
                print("schema_validation=ok")
                print(f"parsed_actions={len(parsed_bundle.actions)}")
                print(f"parsed_facts={len(parsed_bundle.facts)}")
                print(f"parsed_participant_hints={len(parsed_bundle.participant_hints)}")
                print(f"parsed_autonomy_decision={parsed_bundle.autonomy_decision}")
                print(f"parsed_next_delay_seconds={parsed_bundle.next_delay_seconds}")
            else:
                print("schema_validation=failed")
                print(f"schema_validation_error_class={validation_error.__class__.__name__ if validation_error else ''}")
                print(f"schema_validation_error_text={validation_error}")
        if args.print_response:
            message = _field_get(_field_get(completion, "choices", [{}])[0], "message")
            print("response_text=")
            print(_message_content_text(message))
        print("---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
