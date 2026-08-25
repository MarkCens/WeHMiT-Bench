# Import standard utilities for durable checkpoints and reproducible retries.
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import random
import re
import threading
import time
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

# Use one OpenAI-compatible client and one metric suite for every model.
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score


# Use explicit exit codes so a scheduled runner can distinguish stop conditions.
DEFAULT_OUTPUT_DIR = "experiment_outputs"
DEFAULT_MODEL_CONFIG = "model.json"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_FREQUENCY_PENALTY = 0.5
DEFAULT_MAX_TOKENS = 4096
CHECKPOINT_MAX_TOKENS = 1024
DEFAULT_TOKEN_STAGES = "1024,2048,4096,8192"
DEFAULT_TOKEN_STAGE_PLAN = "4096,8192,16384"
TOKEN_STAGE_POLICY = "repeat_until_success"
DEFAULT_MAX_ESCALATION_TOKENS = 262144
DEFAULT_PARALLEL_WORKERS = 128
DEFAULT_WORKERS_PER_MODEL = 128
DEFAULT_THINKING_BUDGET = 1024
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_TIMEOUT = 300.0
DEFAULT_RETRIES = 3
DEFAULT_STREAM_CHECKPOINT_SECONDS = 30.0
EXIT_COMPLETE = 0
EXIT_INCOMPLETE = 3
EXIT_BALANCE = 42
EXIT_TRANSIENT = 75
EXIT_FATAL = 2

# Serialize individual JSONL lines while API workers run concurrently.
JSONL_WRITE_LOCK = threading.Lock()

# Fix one title-only rubric and one request policy across all compared models.
SYSTEM_PROMPT = """You evaluate whether a Chinese WeChat article title itself reveals or strongly suggests health misinformation. Use only the title. Do not retrieve information, infer the article body, or assume unstated facts.

Label Yes when the title promotes, endorses, or sensationalizes a likely false, unsafe, unsupported, or exaggerated health claim. Typical signals include miracle cures, absolute efficacy, folk remedies presented as treatment, food or supplements claimed to cure disease, replacement of standard care, detoxification claims, hidden-cure conspiracies, and product claims with unsupported medical benefits.

Label No when the title is neutral medical education, ordinary wellness advice, a news or personal story, a question or debunking statement without endorsement, an out-of-domain use of a trigger word, or too ambiguous to infer misinformation. Be conservative when evidence is insufficient.

Use one concise reasoning pass, do not repeatedly reconsider the decision, and always reserve output tokens for the final JSON object.

Return one JSON object with exactly these fields: label, explanation, confidence. The label must be Yes or No. The explanation must be plain Chinese, grounded only in the title, and no longer than 200 Chinese characters. Confidence must be a number from 0 to 1."""


def parse_args():
    # Evaluate every Core title and OOD title by default.
    parser = argparse.ArgumentParser(description="Detect potentially misleading health claims with configured LLMs.")
    parser.add_argument("--core-file", "--test-file", dest="core_file", default=f"{DEFAULT_OUTPUT_DIR}/01_core_health_set.csv", help="Full Core Health Set CSV.")
    parser.add_argument("--ood-file", default=f"{DEFAULT_OUTPUT_DIR}/01_out_of_domain_set.csv", help="Out-of-Domain Set CSV.")
    parser.add_argument("--include-ood", dest="include_ood", action="store_true", help="Include the OOD set.")
    parser.add_argument("--exclude-ood", dest="include_ood", action="store_false", help="Exclude the OOD set.")
    parser.set_defaults(include_ood=True)

    # Read all nested provider-model entries unless an explicit model filter is used.
    parser.add_argument("--models-json", default=DEFAULT_MODEL_CONFIG, help="Nested or flat model configuration JSON.")
    parser.add_argument("--model-names", default="", help="Optional comma-separated display names to run.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSONL checkpoints and result tables.")

    # Apply the same semantic inference policy to every provider and model.
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Shared sampling temperature.")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Shared nucleus-sampling probability.")
    parser.add_argument("--frequency-penalty", type=float, default=DEFAULT_FREQUENCY_PENALTY, help="Shared repetition penalty.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Initial shared output-token budget.")
    parser.add_argument("--token-stages", default=DEFAULT_TOKEN_STAGES, help="Checkpoint-compatible unique escalation budgets.")
    parser.add_argument("--token-stage-plan", default=DEFAULT_TOKEN_STAGE_PLAN, help="Nondecreasing comma-separated runtime stages; repeated budgets are allowed.")
    parser.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS, help="Maximum concurrent model workers.")
    parser.add_argument("--workers-per-model", type=int, default=DEFAULT_WORKERS_PER_MODEL, help="Maximum concurrent requests per model.")
    parser.add_argument("--ready-stage-first", action="store_true", help="Advance fresh token-limit responses before retrying unrelated transient failures.")
    parser.add_argument("--escalate-after-max", action="store_true", help="Keep doubling the final stage for token-limited responses instead of cycling back.")
    parser.add_argument("--max-escalation-tokens", type=int, default=DEFAULT_MAX_ESCALATION_TOKENS, help="Hard ceiling for geometric token escalation.")
    parser.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET, help="Shared thinking-token budget when supported.")
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default=DEFAULT_REASONING_EFFORT, help="Shared reasoning effort.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Shared API timeout in seconds.")
    parser.add_argument("--stream-checkpoint-seconds", type=float, default=DEFAULT_STREAM_CHECKPOINT_SECONDS, help="Seconds between durable stream progress records.")

    # Retry transient failures while stopping immediately on an exhausted balance.
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Attempts per request and thinking profile.")
    parser.add_argument("--retry-base-seconds", type=float, default=2.0, help="Initial retry delay.")
    parser.add_argument("--retry-max-seconds", type=float, default=60.0, help="Maximum retry delay.")
    parser.add_argument("--request-delay", type=float, default=0.2, help="Delay after each successful API call.")

    # Support confidence intervals, bounded scheduled runs, and inexpensive checks.
    parser.add_argument("--bootstrap-rounds", type=int, default=1000, help="Bootstrap rounds for confidence intervals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap per subset for testing.")
    parser.add_argument("--time-budget-minutes", type=float, default=0.0, help="Stop cleanly after this many minutes; zero means unlimited.")
    parser.add_argument("--resume-after-balance", action="store_true", help="Clear a prior balance stop after funds are restored.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data, models, and thinking profiles without API calls.")
    return parser.parse_args()


def utc_now():
    # Store all checkpoint times in one timezone-independent representation.
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value):
    # Hash canonical JSON so changed prompts or request settings trigger new calls.
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def parse_checkpoint_token_stages(args):
    # Preserve the original unique schedule inside scientific condition signatures.
    stages = [int(value.strip()) for value in str(args.token_stages).split(",") if value.strip()]
    initial_max_tokens = CHECKPOINT_MAX_TOKENS
    if not stages:
        raise ValueError("--token-stages must contain at least one positive integer.")
    if initial_max_tokens not in stages:
        stages.insert(0, initial_max_tokens)
    if stages[0] != initial_max_tokens:
        raise ValueError("--max-tokens must equal the first value in --token-stages.")
    if any(value <= 0 for value in stages):
        raise ValueError("Every token stage must be positive.")
    if stages != sorted(set(stages)):
        raise ValueError("--token-stages must be strictly increasing without duplicates.")
    return stages


def parse_token_stages(args):
    # Validate the runtime plan while allowing repeated low-cost token budgets.
    stage_text = getattr(args, "token_stage_plan", "") or args.token_stages
    stages = [int(value.strip()) for value in str(stage_text).split(",") if value.strip()]
    initial_max_tokens = int(getattr(args, "initial_max_tokens", args.max_tokens))
    if not stages:
        raise ValueError("--token-stage-plan must contain at least one positive integer.")
    if stages[0] != initial_max_tokens:
        raise ValueError("--max-tokens must equal the first value in --token-stage-plan.")
    if any(value <= 0 for value in stages):
        raise ValueError("Every runtime token stage must be positive.")
    if stages != sorted(stages):
        raise ValueError("--token-stage-plan must be nondecreasing.")
    return stages


def make_token_plan_signature(token_stages):
    # Distinguish stage checkpoints when a repeated-budget schedule changes.
    return stable_hash({"token_stage_plan": token_stages, "token_stage_policy": TOKEN_STAGE_POLICY})


def get_token_budget_for_stage(token_stages, token_stage_index, escalate_after_max=False, max_escalation_tokens=DEFAULT_MAX_ESCALATION_TOKENS):
    # Keep the signed base plan stable while allowing monotonic overflow stages.
    if not token_stages:
        raise ValueError("At least one token stage is required.")
    if token_stage_index < 0:
        raise ValueError("Token stage index cannot be negative.")
    if not escalate_after_max:
        return token_stages[token_stage_index % len(token_stages)]
    if max_escalation_tokens < token_stages[-1]:
        raise ValueError("--max-escalation-tokens cannot be below the final base stage.")
    if token_stage_index < len(token_stages):
        return token_stages[token_stage_index]
    overflow_steps = token_stage_index - len(token_stages) + 1
    return min(token_stages[-1] * (2 ** overflow_steps), int(max_escalation_tokens))


def make_stage_args(args, token_budget):
    # Clone parsed arguments so concurrent workers never mutate shared settings.
    stage_args = copy(args)
    stage_args.initial_max_tokens = int(getattr(args, "initial_max_tokens", args.max_tokens))
    stage_args.max_tokens = int(token_budget)
    return stage_args


def get_shared_request_policy(args):
    # Keep one auditable semantic request policy for every provider and experiment.
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "max_tokens": args.max_tokens,
        "token_stages": parse_token_stages(args),
        "response_format": "json_object",
        "thinking_budget": args.thinking_budget,
        "reasoning_effort": args.reasoning_effort,
        "timeout": args.timeout,
        "stream": True,
        "stream_checkpoint_seconds": args.stream_checkpoint_seconds,
        "retries": args.retries,
        "retry_base_seconds": args.retry_base_seconds,
        "retry_max_seconds": args.retry_max_seconds
    }


def get_checkpoint_request_policy(args):
    # Reconstruct the prior policy solely to recognize existing successful rows.
    policy = get_shared_request_policy(args)
    policy["max_tokens"] = CHECKPOINT_MAX_TOKENS
    policy["token_stages"] = parse_checkpoint_token_stages(args)
    return policy


def infer_thinking_profiles(provider, model_id, base_url):
    # Translate one shared thinking policy into provider-specific request syntax.
    provider_text = str(provider).lower()
    model_text = str(model_id).lower()
    base_text = str(base_url).lower()
    if provider_text == "deepseek" or "api.deepseek.com" in base_text:
        return ["deepseek_thinking"]
    if provider_text == "siliconflow" or "siliconflow" in base_text:
        if "glm-5.2" in model_text:
            return ["thinking_object", "siliconflow_enable"]
        if "kimi-k2.6" in model_text:
            return ["siliconflow_enable", "thinking_object"]
        return ["siliconflow_enable"]
    return ["thinking_object"]


def normalize_flat_config(config):
    # Preserve legacy flat configurations while adding inferred thinking metadata.
    normalized = dict(config)
    normalized["provider"] = str(config.get("provider", "openai_compatible"))
    normalized["name"] = str(config.get("name", config["model"]))
    normalized["model"] = str(config["model"])
    normalized["base_url"] = str(config.get("base_url", "https://api.openai.com/v1"))
    normalized["thinking_profiles"] = config.get("thinking_profiles", infer_thinking_profiles(normalized["provider"], normalized["model"], normalized["base_url"]))
    return normalized


def expand_provider_config(provider_config):
    # Expand one provider block into one independent configuration per model.
    provider_name = str(provider_config.get("provider", "openai_compatible"))
    base_url = str(provider_config["base_url"])
    model_entries = provider_config["model"]
    expanded = []
    for model_entry in model_entries:
        model_id = str(model_entry.get("full_name", model_entry.get("model", model_entry["name"])))
        display_name = str(model_entry.get("name", model_id))
        config = {
            "provider": provider_name,
            "protocol": provider_config.get("protocol", "openai"),
            "base_url": base_url,
            "api_key": provider_config.get("api_key", ""),
            "api_key_env": provider_config.get("api_key_env", ""),
            "name": display_name,
            "model": model_id,
            "token_parameter": provider_config.get("token_parameter", "max_tokens"),
            "supports_temperature": provider_config.get("supports_temperature", True),
            "thinking_profiles": model_entry.get("thinking_profiles", infer_thinking_profiles(provider_name, model_id, base_url))
        }
        expanded.append(config)
    return expanded


def load_model_configs(config_path, model_names, dry_run):
    # Accept the provided nested model.json schema and the earlier flat schema.
    if config_path and Path(config_path).exists():
        config_data = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
        raw_configs = config_data.get("models", config_data) if isinstance(config_data, dict) else config_data
        raw_configs = raw_configs if isinstance(raw_configs, list) else [raw_configs]
        model_configs = []
        for raw_config in raw_configs:
            if isinstance(raw_config.get("model"), list):
                model_configs.extend(expand_provider_config(raw_config))
            else:
                model_configs.append(normalize_flat_config(raw_config))
    elif os.environ.get("LLM_MODELS_JSON"):
        raw_configs = json.loads(os.environ["LLM_MODELS_JSON"])
        model_configs = [normalize_flat_config(config) for config in raw_configs]
    elif dry_run:
        model_configs = [normalize_flat_config({"name": "example-model", "model": "example-model", "base_url": "https://api.example.com/v1", "api_key_env": "EXAMPLE_API_KEY"})]
    else:
        model_name = os.environ["LLM_MODEL_NAME"]
        model_configs = [normalize_flat_config({"name": model_name, "model": model_name, "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"), "api_key_env": os.environ.get("LLM_API_KEY_ENV", "OPENAI_API_KEY")})]

    # Apply an optional display-name filter without changing model request settings.
    selected_names = {name.strip() for name in model_names.split(",") if name.strip()}
    if selected_names:
        model_configs = [config for config in model_configs if config["name"] in selected_names]
    if not model_configs:
        raise ValueError("No model configuration was selected.")

    # Require stable unique names because they form durable checkpoint keys.
    seen_names = set()
    for config in model_configs:
        required_keys = {"name", "model", "base_url", "thinking_profiles"}
        missing_keys = required_keys - set(config)
        if missing_keys:
            raise ValueError(f"Model config is missing keys: {sorted(missing_keys)}")
        if config["name"] in seen_names:
            raise ValueError(f"Duplicate model display name: {config['name']}")
        seen_names.add(config["name"])
    return model_configs


def resolve_api_key(model_config):
    # Prefer the direct key in model.json and otherwise read the named environment variable.
    direct_key = str(model_config.get("api_key", "")).strip()
    if direct_key:
        return direct_key
    environment_name = str(model_config.get("api_key_env", "")).strip()
    if not environment_name:
        raise ValueError(f"No API key or api_key_env is configured for {model_config['name']}.")
    return os.environ[environment_name]


def load_evaluation_data(core_file, ood_file, include_ood, max_records):
    # Label the complete Core set separately from the dedicated OOD stress test.
    core_data = pd.read_csv(core_file, encoding="utf-8-sig")
    core_data["subset"] = "core_all"
    frames = [core_data]
    if include_ood:
        ood_data = pd.read_csv(ood_file, encoding="utf-8-sig")
        ood_data["subset"] = "out_of_domain"
        frames.append(ood_data)
    evaluation_data = pd.concat(frames, ignore_index=True)

    # Apply a test cap per subset and normalize ids before constructing keys.
    if max_records > 0:
        evaluation_data = evaluation_data.groupby("subset", group_keys=False).head(max_records).reset_index(drop=True)
    required_columns = {"id", "title", "gold_label", "subset"}
    missing_columns = required_columns - set(evaluation_data.columns)
    if missing_columns:
        raise ValueError(f"Evaluation data is missing columns: {sorted(missing_columns)}")
    evaluation_data["id"] = evaluation_data["id"].astype(str)
    return evaluation_data.reset_index(drop=True)


def sanitize_unicode_text(value):
    # Replace isolated UTF-16 surrogate code points before UTF-8 serialization.
    return re.sub(r"[\ud800-\udfff]", "\ufffd", str(value))


def build_user_prompt(title):
    # Insert only the observed title so no article-body evidence can leak into input.
    safe_title = sanitize_unicode_text(title)
    return f"Article title: {safe_title}\nJudge the title now and return JSON only."


def parse_model_response(raw_response):
    # Remove optional fenced-code wrappers before locating the response object.
    cleaned_response = raw_response.strip()
    cleaned_response = re.sub(r"^```(?:json)?\s*", "", cleaned_response, flags=re.IGNORECASE)
    cleaned_response = re.sub(r"\s*```$", "", cleaned_response)
    match = re.search(r"\{.*\}", cleaned_response, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            label_text = str(parsed.get("label", "")).strip().lower()
            label = "Yes" if label_text == "yes" else "No" if label_text == "no" else ""
            explanation = str(parsed.get("explanation", "")).strip()
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
            if label:
                return label, explanation, confidence, True
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Keep format compliance measurable while rescuing a clearly stated legal label.
    label_match = re.search(r"\b(Yes|No)\b", cleaned_response, flags=re.IGNORECASE)
    label = label_match.group(1).title() if label_match else ""
    return label, cleaned_response[:200], 0.5, False


def build_request_data(model_config, messages, args, thinking_profile):
    # Begin with the exact same model-independent request fields.
    request_data = {"model": model_config["model"], "messages": messages, "stream": True, "stream_options": {"include_usage": True}, "temperature": args.temperature, "top_p": args.top_p, "frequency_penalty": args.frequency_penalty, "reasoning_effort": args.reasoning_effort, "response_format": {"type": "json_object"}}
    token_parameter = model_config.get("token_parameter", "max_tokens")
    request_data[token_parameter] = args.max_tokens
    extra_body = dict(model_config.get("extra_body", {}))

    # Enable thinking with equivalent settings using each provider's required syntax.
    if thinking_profile == "deepseek_thinking":
        extra_body["thinking"] = {"type": "enabled"}
    elif thinking_profile == "siliconflow_enable":
        extra_body["enable_thinking"] = True
        extra_body["thinking_budget"] = args.thinking_budget
    elif thinking_profile == "thinking_object":
        extra_body["thinking"] = {"type": "enabled"}
    else:
        raise ValueError(f"Unknown thinking profile: {thinking_profile}")

    # Provider-specific keys express the same requested thinking behavior.
    if extra_body:
        request_data["extra_body"] = extra_body
    return request_data


def extract_status_code(error):
    # OpenAI-compatible exceptions expose status on either the error or response.
    status_code = getattr(error, "status_code", None)
    if status_code is None and getattr(error, "response", None) is not None:
        status_code = getattr(error.response, "status_code", None)
    return int(status_code) if status_code is not None else None


def classify_api_error(error):
    # Distinguish balance exhaustion from retryable and permanent failures.
    status_code = extract_status_code(error)
    error_text = f"{type(error).__name__}: {error}"
    lower_text = error_text.lower()
    balance_terms = ["insufficient balance", "insufficient funds", "balance not enough", "insufficient quota", "quota is not enough", "account balance", "recharge required", "余额不足", "账户余额", "额度不足", "请充值", "欠费"]
    if status_code == 402 or any(term in lower_text for term in balance_terms):
        return "balance_exhausted", status_code, error_text
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return "transient", status_code, error_text
    transient_terms = ["timeout", "timed out", "connection", "network", "temporarily unavailable", "remote protocol", "server disconnected"]
    if status_code is None and any(term in lower_text for term in transient_terms):
        return "transient", status_code, error_text
    if status_code in {400, 422}:
        return "invalid_parameters", status_code, error_text
    return "permanent", status_code, error_text


def get_retry_delay(error, attempt_index, args):
    # Honor Retry-After when present and otherwise use capped exponential jitter.
    response = getattr(error, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(args.retry_max_seconds, max(0.0, float(retry_after)))
            except ValueError:
                pass
    exponential = args.retry_base_seconds * (2.0 ** attempt_index)
    return min(args.retry_max_seconds, exponential + random.uniform(0.0, args.retry_base_seconds))


def extract_usage_values(usage):
    # Normalize token accounting across ordinary and streamed response objects.
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    return prompt_tokens, completion_tokens, reasoning_tokens, total_tokens


def extract_usage(response):
    # Preserve the ordinary-response helper for provider diagnostics and reuse.
    return extract_usage_values(getattr(response, "usage", None))


def make_stream_progress_record(thinking_profile, api_attempt, attempt_status, raw_response, reasoning_chars, start_time, finish_reason=""):
    # Record compact stream state without duplicating the full hidden reasoning.
    return {
        "timestamp": utc_now(),
        "attempt_status": attempt_status,
        "thinking_profile": thinking_profile,
        "api_attempt": api_attempt,
        "finish_reason": finish_reason,
        "raw_response": raw_response,
        "reasoning_content": "",
        "thinking_observed": reasoning_chars > 0,
        "reasoning_chars": reasoning_chars,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "http_status": None,
        "error_kind": "",
        "error": ""
    }


def collect_streamed_completion(client, request_data, args, thinking_profile, api_attempt, start_time, attempt_callback):
    # Accumulate SSE deltas while periodically fsyncing compact progress records.
    content_parts = []
    reasoning_parts = []
    reasoning_chars = 0
    finish_reason = ""
    usage = None
    chunk_count = 0
    last_periodic_checkpoint = time.monotonic()
    last_content_checkpoint_length = 0
    if attempt_callback is not None:
        started_record = make_stream_progress_record(thinking_profile, api_attempt, "request_started", "", 0, start_time)
        attempt_callback(started_record)

    # Stream every provider with identical settings and collect its final usage chunk.
    response_stream = client.chat.completions.create(**request_data)
    for chunk in response_stream:
        chunk_count += 1
        if time.perf_counter() - start_time >= args.timeout:
            try:
                response_stream.close()
            except Exception:
                pass
            raise TimeoutError(f"Stream exceeded the shared {args.timeout}-second wall-clock limit.")
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if choices:
            choice = choices[0]
            finish_reason = str(getattr(choice, "finish_reason", "") or finish_reason)
            delta = getattr(choice, "delta", None)
            content_delta = sanitize_unicode_text(getattr(delta, "content", "") or "")
            reasoning_delta = sanitize_unicode_text(getattr(delta, "reasoning_content", "") or "")
            if content_delta:
                content_parts.append(content_delta)
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
                reasoning_chars += len(reasoning_delta)

        # Save accumulated final-answer text in small durable increments.
        raw_response = "".join(content_parts)
        if attempt_callback is not None and len(raw_response) - last_content_checkpoint_length >= 64:
            checkpoint = make_stream_progress_record(thinking_profile, api_attempt, "content_checkpoint", raw_response, reasoning_chars, start_time, finish_reason)
            attempt_callback(checkpoint)
            last_content_checkpoint_length = len(raw_response)

        # Save liveness and reasoning length during long thinking responses.
        if attempt_callback is not None and time.monotonic() - last_periodic_checkpoint >= args.stream_checkpoint_seconds:
            checkpoint = make_stream_progress_record(thinking_profile, api_attempt, "stream_checkpoint", raw_response, reasoning_chars, start_time, finish_reason)
            attempt_callback(checkpoint)
            last_periodic_checkpoint = time.monotonic()

    prompt_tokens, completion_tokens, reasoning_tokens, total_tokens = extract_usage_values(usage)
    return {
        "raw_response": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "stream_chunk_count": chunk_count
    }


def stream_hit_token_limit(stream_result, token_budget):
    # Detect explicit truncation and providers that omit a finish reason at the cap.
    finish_reason = str(stream_result.get("finish_reason", "")).strip().lower()
    if finish_reason in {"length", "max_tokens", "max_output_tokens", "token_limit"}:
        return True
    completion_tokens = int(stream_result.get("completion_tokens", 0) or 0)
    return not finish_reason and completion_tokens >= int(token_budget)


def make_token_limit_result(stream_result, thinking_profile, api_attempts, latency_ms):
    # Preserve truncated output and usage so the next token stage is auditable.
    reasoning_content = stream_result["reasoning_content"]
    return {
        "status": "token_limit",
        "pred_label": "",
        "explanation": "",
        "confidence": 0.0,
        "format_valid": False,
        "raw_response": stream_result["raw_response"],
        "finish_reason": stream_result["finish_reason"],
        "thinking_profile": thinking_profile,
        "thinking_requested": True,
        "thinking_observed": bool(reasoning_content or stream_result["reasoning_tokens"]),
        "reasoning_chars": len(reasoning_content),
        "prompt_tokens": stream_result["prompt_tokens"],
        "completion_tokens": stream_result["completion_tokens"],
        "reasoning_tokens": stream_result["reasoning_tokens"],
        "total_tokens": stream_result["total_tokens"],
        "latency_ms": latency_ms,
        "api_attempts": api_attempts,
        "error_kind": "token_limit",
        "http_status": None,
        "error": "The streamed completion reached the current output-token limit."
    }


def request_chat_completion(client, model_config, messages, args, response_parser, attempt_callback=None):
    # Try only documented thinking syntaxes and retain a full status for checkpointing.
    last_error = ""
    last_error_kind = "permanent"
    last_status_code = None
    total_attempts = 0
    profiles = list(model_config["thinking_profiles"])
    for profile_index, thinking_profile in enumerate(profiles):
        for attempt_index in range(args.retries):
            total_attempts += 1
            start_time = time.perf_counter()
            try:
                request_data = build_request_data(model_config, messages, args, thinking_profile)
                stream_result = collect_streamed_completion(client, request_data, args, thinking_profile, total_attempts, start_time, attempt_callback)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                raw_response = stream_result["raw_response"]
                reasoning_content = stream_result["reasoning_content"]
                label, explanation, confidence, format_valid = response_parser(raw_response)
                prompt_tokens = stream_result["prompt_tokens"]
                completion_tokens = stream_result["completion_tokens"]
                reasoning_tokens = stream_result["reasoning_tokens"]
                total_tokens = stream_result["total_tokens"]
                token_limit = stream_hit_token_limit(stream_result, args.max_tokens)
                attempt_record = {
                    "timestamp": utc_now(),
                    "attempt_status": "token_limit" if token_limit else "valid_response" if label in {"Yes", "No"} else "invalid_response",
                    "thinking_profile": thinking_profile,
                    "api_attempt": total_attempts,
                    "finish_reason": stream_result["finish_reason"],
                    "stream_chunk_count": stream_result["stream_chunk_count"],
                    "pred_label": label,
                    "explanation": explanation,
                    "confidence": confidence,
                    "format_valid": format_valid,
                    "raw_response": raw_response,
                    "reasoning_content": reasoning_content,
                    "thinking_observed": bool(reasoning_content or reasoning_tokens),
                    "reasoning_chars": len(reasoning_content),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": latency_ms,
                    "http_status": None,
                    "error_kind": "",
                    "error": ""
                }
                if attempt_callback is not None:
                    attempt_callback(attempt_record)

                # Defer a truncated response until every item finishes this stage.
                if token_limit:
                    return make_token_limit_result(stream_result, thinking_profile, total_attempts, latency_ms)

                # Retry an unusable response because it cannot complete the benchmark item.
                if label not in {"Yes", "No"}:
                    last_error = "Model response did not contain a legal Yes or No label."
                    last_error_kind = "invalid_response"
                    if attempt_index + 1 < args.retries:
                        time.sleep(get_retry_delay(RuntimeError(last_error), attempt_index, args))
                        continue
                    break
                return {
                    "status": "success",
                    "pred_label": label,
                    "explanation": explanation,
                    "confidence": confidence,
                    "format_valid": format_valid,
                    "raw_response": raw_response,
                    "finish_reason": stream_result["finish_reason"],
                    "thinking_profile": thinking_profile,
                    "thinking_requested": True,
                    "thinking_observed": bool(reasoning_content or reasoning_tokens),
                    "reasoning_chars": len(reasoning_content),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": latency_ms,
                    "api_attempts": total_attempts,
                    "error_kind": "",
                    "http_status": None,
                    "error": ""
                }
            except Exception as error:
                error_kind, status_code, error_text = classify_api_error(error)
                last_error = error_text
                last_error_kind = error_kind
                last_status_code = status_code
                attempt_record = {
                    "timestamp": utc_now(),
                    "attempt_status": "request_error",
                    "thinking_profile": thinking_profile,
                    "api_attempt": total_attempts,
                    "finish_reason": "",
                    "stream_chunk_count": 0,
                    "pred_label": "",
                    "explanation": "",
                    "confidence": 0.0,
                    "format_valid": False,
                    "raw_response": "",
                    "reasoning_content": "",
                    "thinking_observed": False,
                    "reasoning_chars": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "http_status": status_code,
                    "error_kind": error_kind,
                    "error": error_text
                }
                if attempt_callback is not None:
                    attempt_callback(attempt_record)
                if error_kind == "balance_exhausted":
                    return make_failure_result("balance_exhausted", thinking_profile, total_attempts, error_kind, status_code, error_text)

                # A rejected thinking syntax moves to the documented fallback profile.
                has_profile_fallback = profile_index + 1 < len(profiles)
                if error_kind == "invalid_parameters" and has_profile_fallback:
                    break
                if error_kind != "transient":
                    return make_failure_result("permanent_failure", thinking_profile, total_attempts, error_kind, status_code, error_text)
                if attempt_index + 1 < args.retries:
                    time.sleep(get_retry_delay(error, attempt_index, args))
                    continue
                return make_failure_result("transient_failure", thinking_profile, total_attempts, error_kind, status_code, error_text)

    # Invalid model output or exhausted parameter profiles remain eligible for resumption.
    final_status = "transient_failure" if last_error_kind == "invalid_response" else "permanent_failure"
    final_profile = profiles[-1] if profiles else ""
    return make_failure_result(final_status, final_profile, total_attempts, last_error_kind, last_status_code, last_error)


def make_failure_result(status, thinking_profile, api_attempts, error_kind, status_code, error_text):
    # Return the same schema for every failure so JSONL remains easy to audit.
    return {
        "status": status,
        "pred_label": "",
        "explanation": "",
        "confidence": 0.0,
        "format_valid": False,
        "raw_response": "",
        "finish_reason": "",
        "thinking_profile": thinking_profile,
        "thinking_requested": True,
        "thinking_observed": False,
        "reasoning_chars": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
        "api_attempts": api_attempts,
        "error_kind": error_kind,
        "http_status": status_code,
        "error": error_text
    }


def append_jsonl(output_path, record):
    # Append and force every request result to disk before starting another call.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_record = sanitize_unicode_text(json.dumps(record, ensure_ascii=False)) + "\n"
    with JSONL_WRITE_LOCK:
        with output_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(serialized_record)
            file.flush()
            os.fsync(file.fileno())


def load_jsonl_records(output_path):
    # Repair only a truncated final line left by a hard process interruption.
    if not output_path.exists():
        return []
    raw_text = output_path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    records = []
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if line_index != len(lines) - 1:
                raise
            corrupt_path = output_path.with_name(output_path.name + ".corrupt_tail")
            with corrupt_path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(line + "\n")
            valid_text = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
            atomic_write_text(output_path, valid_text + ("\n" if valid_text else ""))
            return records

    # Ensure the next append starts on a new line after a manually edited file.
    if raw_text and not raw_text.endswith("\n"):
        with output_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
    return records


def atomic_write_text(output_path, text):
    # Replace completed summaries atomically so readers never see partial files.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, output_path)


def atomic_write_json(output_path, value):
    # Serialize status and completion markers with stable readable formatting.
    atomic_write_text(output_path, json.dumps(value, ensure_ascii=False, indent=2))


def atomic_write_csv(data_frame, output_path):
    # Write CSV through a temporary sibling before replacing the prior artifact.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    data_frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    os.replace(temporary_path, output_path)


def make_condition_signatures(model_configs, args):
    # Bind checkpoints to the exact prompt and shared inference configuration.
    signatures = {}
    for config in model_configs:
        signature_data = {
            "experiment": "llm_title_detection",
            "prompt": SYSTEM_PROMPT,
            "model": config["model"],
            "provider": config["provider"],
            "thinking_profiles": config["thinking_profiles"],
            "shared_request_policy": get_checkpoint_request_policy(args)
        }
        signatures[config["name"]] = stable_hash(signature_data)
    return signatures


def record_key(record):
    # Normalize key fields before comparing current and historical JSONL entries.
    return str(record["model_name"]), str(record["subset"]), str(record["id"])


def load_success_state(records, condition_signatures, expected_metadata=None):
    # A failed request never becomes a completed checkpoint key.
    latest_records = {}
    for record in records:
        model_name = str(record.get("model_name", ""))
        if model_name not in condition_signatures:
            continue
        if record.get("condition_signature") != condition_signatures[model_name]:
            continue
        if record.get("status") != "success" or record.get("pred_label") not in {"Yes", "No"}:
            continue
        key = record_key(record)
        if expected_metadata is not None:
            current = expected_metadata.get(key)
            if current is None or str(record.get("title", "")) != current["title"]:
                continue
            record = dict(record)
            record.update(current)
        latest_records[key] = record
    return latest_records


def load_stage_outcomes(records, condition_signatures, token_plan_signature, token_stages=None, escalate_after_max=False, max_escalation_tokens=DEFAULT_MAX_ESCALATION_TOKENS, expected_metadata=None):
    # Use the global stage index so later cycles can revisit the same budget.
    stage_outcomes = {}
    for record in records:
        model_name = str(record.get("model_name", ""))
        if model_name not in condition_signatures:
            continue
        if record.get("condition_signature") != condition_signatures[model_name]:
            continue
        if record.get("token_plan_signature") != token_plan_signature:
            continue
        token_stage_index = int(record.get("token_stage_index", -1) or 0)
        if token_stage_index < 0:
            continue
        if escalate_after_max:
            expected_budget = get_token_budget_for_stage(token_stages, token_stage_index, True, max_escalation_tokens)
            try:
                recorded_budget = int(record.get("token_budget"))
            except (TypeError, ValueError):
                continue
            if recorded_budget != expected_budget:
                continue
        key = record_key(record)
        if expected_metadata is not None:
            current = expected_metadata.get(key)
            if current is None or str(record.get("title", "")) != current["title"]:
                continue
        stage_outcomes[(key, token_stage_index)] = str(record.get("status", ""))
    return stage_outcomes


def get_stage_candidate_keys(expected_keys, success_records, stage_outcomes, token_stages, stage_index):
    # Stage zero covers all items; every later global stage receives prior truncations.
    incomplete_keys = expected_keys - set(success_records)
    if stage_index == 0:
        return incomplete_keys
    return {key for key in incomplete_keys if stage_outcomes.get((key, stage_index - 1)) == "token_limit"}


def get_stage_pending_keys(candidate_keys, success_records, stage_outcomes, token_stage_index):
    # Do not repeat a completed success or a recorded truncation in this exact cycle.
    pending_keys = set()
    for key in candidate_keys:
        if key in success_records:
            continue
        if stage_outcomes.get((key, token_stage_index)) == "token_limit":
            continue
        pending_keys.add(key)
    return pending_keys


def get_item_token_stage_index(key, stage_outcomes):
    # Advance one item only through stages that explicitly ended at the token limit.
    token_stage_index = 0
    while stage_outcomes.get((key, token_stage_index)) == "token_limit":
        token_stage_index += 1
    return token_stage_index


def select_ready_stage_work(expected_keys, success_records, stage_outcomes, token_stages):
    # Prefer never-attempted escalations while preserving each item's stage order.
    recorded_indices = [stage_index for (key, stage_index) in stage_outcomes if key in expected_keys]
    max_stage_index = max(recorded_indices, default=-1) + 1
    ready_work = []
    for stage_index in range(max_stage_index + 1):
        candidate_keys = get_stage_candidate_keys(expected_keys, success_records, stage_outcomes, token_stages, stage_index)
        pending_keys = get_stage_pending_keys(candidate_keys, success_records, stage_outcomes, stage_index)
        if not pending_keys:
            continue
        fresh_keys = {key for key in pending_keys if (key, stage_index) not in stage_outcomes}
        ready_work.append((stage_index, pending_keys, fresh_keys))

    fresh_escalations = [work for work in ready_work if work[0] > 0 and work[2]]
    if fresh_escalations:
        stage_index, _, fresh_keys = max(fresh_escalations, key=lambda work: (len(work[2]), -work[0]))
        return stage_index, fresh_keys

    fresh_work = [work for work in ready_work if work[2]]
    if fresh_work:
        stage_index, _, fresh_keys = max(fresh_work, key=lambda work: (len(work[2]), -work[0]))
        return stage_index, fresh_keys

    if ready_work:
        stage_index, pending_keys, _ = max(ready_work, key=lambda work: (len(work[1]), -work[0]))
        return stage_index, pending_keys
    return None


def summarize_token_limit_outcomes(expected_keys, stage_outcomes, token_stages, escalate_after_max=False, max_escalation_tokens=DEFAULT_MAX_ESCALATION_TOKENS):
    # Report aggregate budgets while retaining cycle-level audit counts.
    by_budget = {str(token_budget): 0 for token_budget in token_stages}
    by_cycle = {}
    by_global_stage = {}
    stage_count = len(token_stages)
    for (key, token_stage_index), outcome in stage_outcomes.items():
        if key not in expected_keys or outcome != "token_limit":
            continue
        token_budget = get_token_budget_for_stage(token_stages, token_stage_index, escalate_after_max, max_escalation_tokens)
        token_cycle = 0 if escalate_after_max else token_stage_index // stage_count
        stage_in_cycle = token_stage_index if escalate_after_max else token_stage_index % stage_count
        by_budget.setdefault(str(token_budget), 0)
        by_budget[str(token_budget)] += 1
        cycle_counts = by_cycle.setdefault(str(token_cycle), {str(value): 0 for value in token_stages})
        cycle_counts.setdefault(str(token_budget), 0)
        cycle_counts[str(token_budget)] += 1
        stage_counts = by_global_stage.setdefault(str(token_stage_index), {"token_cycle": token_cycle, "stage_in_cycle": stage_in_cycle, "token_budget": token_budget, "token_limits": 0})
        stage_counts["token_limits"] += 1
    return by_budget, by_cycle, by_global_stage


def make_model_worker_specs(model_configs, jobs_by_model, workers_per_model):
    # Split each model queue evenly, then interleave shards for fair scheduling.
    shards_by_model = {}
    for model_config in model_configs:
        model_name = model_config["name"]
        jobs = jobs_by_model.get(model_name, [])
        shard_count = min(int(workers_per_model), len(jobs))
        shards = [[] for shard_index in range(shard_count)]
        for job_index, job in enumerate(jobs):
            shards[job_index % shard_count].append(job)
        shards_by_model[model_name] = shards

    # Interleave the same shard index across models before moving to the next.
    worker_specs = []
    max_shards = max((len(shards) for shards in shards_by_model.values()), default=0)
    for shard_index in range(max_shards):
        for model_config in model_configs:
            model_shards = shards_by_model[model_config["name"]]
            if shard_index < len(model_shards):
                worker_specs.append((model_config, shard_index, model_shards[shard_index]))
    return worker_specs


def confidence_to_probability_yes(prediction_label, confidence):
    # Convert confidence in the chosen class into a positive-class probability.
    if prediction_label == "Yes":
        return confidence
    if prediction_label == "No":
        return 1.0 - confidence
    return 0.5


def expected_calibration_error(true_values, probabilities, bin_count=10):
    # Compare confidence with empirical prevalence in equal-width probability bins.
    bins = np.linspace(0.0, 1.0, bin_count + 1)
    error_value = 0.0
    for bin_index in range(bin_count):
        lower = bins[bin_index]
        upper = bins[bin_index + 1]
        in_bin = (probabilities >= lower) & (probabilities < upper if bin_index < bin_count - 1 else probabilities <= upper)
        if np.any(in_bin):
            error_value += np.mean(in_bin) * abs(np.mean(probabilities[in_bin]) - np.mean(true_values[in_bin]))
    return float(error_value)


def compute_point_metrics(data_frame):
    # Compute the complete binary classification and calibration metric suite.
    true_values = (data_frame["gold_label"] == "Yes").astype(int).to_numpy()
    predicted_values = (data_frame["pred_label"] == "Yes").astype(int).to_numpy()
    confidence_values = data_frame["confidence"].astype(float).to_numpy()
    probability_yes = np.array([confidence_to_probability_yes(label, confidence) for label, confidence in zip(data_frame["pred_label"], confidence_values)])
    tn_value, fp_value, fn_value, tp_value = confusion_matrix(true_values, predicted_values, labels=[0, 1]).ravel()
    metrics = {
        "n": int(len(data_frame)),
        "format_valid_rate": float(data_frame["format_valid"].mean()),
        "thinking_observed_rate": float(data_frame["thinking_observed"].mean()),
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "balanced_accuracy": float(recall_score(true_values, predicted_values, labels=[0, 1], average="macro", zero_division=0)),
        "precision_yes": float(precision_score(true_values, predicted_values, zero_division=0)),
        "recall_yes": float(recall_score(true_values, predicted_values, zero_division=0)),
        "f1_yes": float(f1_score(true_values, predicted_values, zero_division=0)),
        "macro_f1": float(f1_score(true_values, predicted_values, labels=[0, 1], average="macro", zero_division=0)),
        "specificity": float(tn_value / (tn_value + fp_value)) if tn_value + fp_value else 0.0,
        "mcc": float(matthews_corrcoef(true_values, predicted_values)) if len(np.unique(np.concatenate([true_values, predicted_values]))) == 2 else 0.0,
        "ece": expected_calibration_error(true_values, probability_yes),
        "mean_latency_ms": float(data_frame["latency_ms"].mean()),
        "mean_reasoning_tokens": float(data_frame["reasoning_tokens"].mean()),
        "tp": int(tp_value),
        "fp": int(fp_value),
        "tn": int(tn_value),
        "fn": int(fn_value)
    }
    metrics["auroc"] = float(roc_auc_score(true_values, probability_yes)) if len(np.unique(true_values)) == 2 else float("nan")
    return metrics


def bootstrap_confidence_intervals(data_frame, rounds, seed):
    # Resample paired gold and model outputs to preserve title dependence.
    if len(data_frame) < 2 or rounds <= 0:
        return {}
    random_generator = np.random.default_rng(seed)
    bootstrap_values = {"accuracy": [], "macro_f1": [], "f1_yes": []}
    for round_index in range(rounds):
        sampled_indices = random_generator.integers(0, len(data_frame), len(data_frame))
        sampled_data = data_frame.iloc[sampled_indices]
        true_values = sampled_data["gold_label"] == "Yes"
        predicted_values = sampled_data["pred_label"] == "Yes"
        bootstrap_values["accuracy"].append(accuracy_score(true_values, predicted_values))
        bootstrap_values["macro_f1"].append(f1_score(true_values, predicted_values, labels=[False, True], average="macro", zero_division=0))
        bootstrap_values["f1_yes"].append(f1_score(true_values, predicted_values, zero_division=0))
    intervals = {}
    for metric_name, values in bootstrap_values.items():
        intervals[f"{metric_name}_ci_low"] = float(np.percentile(values, 2.5))
        intervals[f"{metric_name}_ci_high"] = float(np.percentile(values, 97.5))
    return intervals


def evaluate_predictions(predictions, bootstrap_rounds, seed):
    # Summarize every model and dataset subset independently.
    metric_rows = []
    for (model_name, subset), group_data in predictions.groupby(["model_name", "subset"], sort=True):
        metrics = compute_point_metrics(group_data)
        metrics.update(bootstrap_confidence_intervals(group_data.reset_index(drop=True), bootstrap_rounds, seed))
        metrics["model_name"] = model_name
        metrics["subset"] = subset
        metric_rows.append(metrics)
    return pd.DataFrame(metric_rows)


def compute_pairwise_mcnemar(predictions):
    # Compare models only on shared successfully completed titles.
    result_rows = []
    for subset, subset_data in predictions.groupby("subset", sort=True):
        model_names = sorted(subset_data["model_name"].unique())
        for model_a, model_b in itertools.combinations(model_names, 2):
            left = subset_data[subset_data["model_name"] == model_a][["id", "gold_label", "pred_label"]]
            right = subset_data[subset_data["model_name"] == model_b][["id", "pred_label"]]
            paired = left.merge(right, on="id", suffixes=("_a", "_b"))
            correct_a = paired["pred_label_a"] == paired["gold_label"]
            correct_b = paired["pred_label_b"] == paired["gold_label"]
            a_only = int((correct_a & ~correct_b).sum())
            b_only = int((~correct_a & correct_b).sum())
            p_value = float(binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue) if a_only + b_only else 1.0
            result_rows.append({"subset": subset, "model_a": model_a, "model_b": model_b, "shared_n": int(len(paired)), "model_a_only_correct": a_only, "model_b_only_correct": b_only, "exact_mcnemar_p": p_value})

    result_frame = pd.DataFrame(result_rows)
    if result_frame.empty:
        return result_frame

    # Apply the Holm procedure within each benchmark subset.
    result_frame["holm_adjusted_p"] = np.nan
    for _, subset_indices in result_frame.groupby("subset").groups.items():
        index_list = list(subset_indices)
        raw_values = result_frame.loc[index_list, "exact_mcnemar_p"].to_numpy(dtype=float)
        order = np.argsort(raw_values)
        adjusted_values = np.empty(len(raw_values), dtype=float)
        running_maximum = 0.0
        for rank, ordered_index in enumerate(order):
            candidate = min(1.0, (len(raw_values) - rank) * raw_values[ordered_index])
            running_maximum = max(running_maximum, candidate)
            adjusted_values[ordered_index] = running_maximum
        result_frame.loc[index_list, "holm_adjusted_p"] = adjusted_values
    return result_frame


def refresh_result_tables(success_records, output_dir, bootstrap_rounds, seed):
    # Rebuild result tables from durable success checkpoints after every run segment.
    if not success_records:
        return
    predictions = pd.DataFrame(list(success_records.values()))
    predictions = predictions.sort_values(["model_name", "subset", "id"]).reset_index(drop=True)
    metrics = evaluate_predictions(predictions, bootstrap_rounds, seed)
    pairwise_tests = compute_pairwise_mcnemar(predictions)
    atomic_write_csv(predictions, output_dir / "llm_predictions.csv")
    atomic_write_csv(metrics, output_dir / "llm_metrics.csv")
    atomic_write_csv(pairwise_tests, output_dir / "02_pairwise_mcnemar.csv")


def make_expected_keys(evaluation_data, model_configs):
    # Cross every configured model with every requested benchmark title.
    expected_keys = set()
    for row_index, row in evaluation_data.iterrows():
        for config in model_configs:
            expected_keys.add((config["name"], str(row["subset"]), str(row["id"])))
    return expected_keys


def make_expected_metadata(evaluation_data, model_configs):
    # Refresh mutable benchmark annotations without changing title-only request identity.
    metadata = {}
    transferable_columns = ["title", "gold_label", "split"]
    for row_index, row in evaluation_data.iterrows():
        current = {column: row[column] for column in transferable_columns if column in evaluation_data.columns}
        current = {key: ("" if pd.isna(value) else value) for key, value in current.items()}
        current["title"] = str(current["title"])
        current["gold_label"] = str(current["gold_label"])
        for config in model_configs:
            metadata[(config["name"], str(row["subset"]), str(row["id"]))] = dict(current)
    return metadata


def write_progress(output_dir, expected_keys, success_records, model_configs, status, token_stages=None, stage_outcomes=None, active_token_budget=None, active_token_stage_index=None, escalate_after_max=False, max_escalation_tokens=DEFAULT_MAX_ESCALATION_TOKENS):
    # Expose per-model completion counts for monitoring scheduled continuation.
    completed_keys = set(success_records)
    token_stages = token_stages or []
    stage_outcomes = stage_outcomes or {}
    model_progress = {}
    for config in model_configs:
        model_name = config["name"]
        expected_count = sum(1 for key in expected_keys if key[0] == model_name)
        completed_count = sum(1 for key in completed_keys if key[0] == model_name)
        model_progress[model_name] = {"completed": completed_count, "expected": expected_count}
    progress = {
        "status": status,
        "updated_at": utc_now(),
        "completed_requests": len(completed_keys & expected_keys),
        "expected_requests": len(expected_keys),
        "remaining_requests": len(expected_keys - completed_keys),
        "models": model_progress
    }
    if token_stages:
        token_limit_by_stage, token_limit_by_cycle, token_limit_by_global_stage = summarize_token_limit_outcomes(expected_keys, stage_outcomes, token_stages, escalate_after_max, max_escalation_tokens)
        progress["token_stages"] = token_stages
        progress["token_stage_policy"] = TOKEN_STAGE_POLICY
        progress["escalate_after_max"] = bool(escalate_after_max)
        progress["max_escalation_tokens"] = int(max_escalation_tokens)
        progress["active_token_budget"] = active_token_budget
        progress["active_token_stage_index"] = active_token_stage_index
        progress["active_token_cycle"] = (0 if escalate_after_max else int(active_token_stage_index // len(token_stages))) if active_token_stage_index is not None else None
        progress["token_limit_by_stage"] = token_limit_by_stage
        progress["token_limit_by_cycle"] = token_limit_by_cycle
        progress["token_limit_by_global_stage"] = token_limit_by_global_stage
    atomic_write_json(output_dir / "llm_progress.json", progress)
    return progress


def time_budget_reached(start_time, minutes):
    # Check the budget only between requests so every JSONL record remains complete.
    if minutes <= 0:
        return False
    return time.monotonic() - start_time >= minutes * 60.0


def make_request_record(model_config, row, condition_signature, result, token_budget, token_stage_index, token_plan_signature):
    # Combine stable benchmark metadata with one API attempt result.
    record = {
        "timestamp": utc_now(),
        "experiment": "llm_title_detection",
        "condition_signature": condition_signature,
        "provider": model_config["provider"],
        "model_name": model_config["name"],
        "model_id": model_config["model"],
        "subset": str(row["subset"]),
        "id": str(row["id"]),
        "title": str(row["title"]),
        "split": str(row.get("split", "")),
        "gold_label": str(row["gold_label"]),
        "token_budget": int(token_budget),
        "token_stage_index": int(token_stage_index),
        "token_plan_signature": token_plan_signature
    }
    record.update(result)
    return record


def make_attempt_logger(attempt_path, model_config, row, condition_signature, args, token_budget, token_stage_index, token_plan_signature):
    # Attach stable item metadata and fsync every individual paid API attempt.
    base_record = {
        "experiment": "llm_title_detection",
        "condition_signature": condition_signature,
        "provider": model_config["provider"],
        "model_name": model_config["name"],
        "model_id": model_config["model"],
        "subset": str(row["subset"]),
        "id": str(row["id"]),
        "title": str(row["title"]),
        "gold_label": str(row["gold_label"]),
        "token_budget": int(token_budget),
        "token_stage_index": int(token_stage_index),
        "token_plan_signature": token_plan_signature,
        "shared_request_policy": get_shared_request_policy(args)
    }

    def log_attempt(attempt_record):
        full_record = dict(base_record)
        full_record.update(attempt_record)
        append_jsonl(attempt_path, full_record)

    return log_attempt


def run_main_model_stage(model_config, model_shard_index, jobs, stage_args, token_budget, token_stage_index, token_plan_signature, condition_signature, prediction_path, attempt_path, success_records, stage_outcomes, state_lock, stop_event, time_event, start_time):
    # Give each model shard one client while preserving identical request settings.
    client = OpenAI(api_key=resolve_api_key(model_config), base_url=model_config["base_url"], timeout=stage_args.timeout, max_retries=0)
    worker_result = {"model_name": model_config["name"], "model_shard_index": model_shard_index, "successes": 0, "token_limits": 0, "transient_failures": 0, "permanent_failures": 0, "balance_record": None}
    for row, key in jobs:
        if stop_event.is_set():
            break
        if time_budget_reached(start_time, stage_args.time_budget_minutes):
            time_event.set()
            stop_event.set()
            break

        # Send one title with the active stage budget and append its result immediately.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user_prompt(row["title"])}]
        attempt_logger = make_attempt_logger(attempt_path, model_config, row, condition_signature, stage_args, token_budget, token_stage_index, token_plan_signature)
        result = request_chat_completion(client, model_config, messages, stage_args, parse_model_response, attempt_logger)
        record = make_request_record(model_config, row, condition_signature, result, token_budget, token_stage_index, token_plan_signature)
        append_jsonl(prediction_path, record)
        with state_lock:
            stage_outcomes[(key, token_stage_index)] = result["status"]
            if result["status"] == "success":
                success_records[key] = record

        # Continue past token truncation so the complete stage reaches its barrier.
        if result["status"] == "success":
            worker_result["successes"] += 1
        elif result["status"] == "token_limit":
            worker_result["token_limits"] += 1
        elif result["status"] == "balance_exhausted":
            worker_result["balance_record"] = {"status": "balance_exhausted", "stopped_at": utc_now(), "provider": model_config["provider"], "model_name": model_config["name"], "model_id": model_config["model"], "token_budget": token_budget, "error": result["error"]}
            stop_event.set()
            break
        elif result["status"] == "transient_failure":
            worker_result["transient_failures"] += 1
            # Drain the shard so deterministic restarts do not starve later jobs.
        else:
            worker_result["permanent_failures"] += 1
            break
        if stage_args.request_delay > 0:
            time.sleep(stage_args.request_delay)
    return worker_result


def run_main_model_item_local(model_config, model_shard_index, jobs, args, token_stages, token_plan_signature, condition_signature, prediction_path, attempt_path, success_records, stage_outcomes, state_lock, stop_event, time_event, start_time):
    # Let each item retry malformed output and escalate truncation without a batch barrier.
    client = OpenAI(api_key=resolve_api_key(model_config), base_url=model_config["base_url"], timeout=args.timeout, max_retries=0)
    worker_result = {"model_name": model_config["name"], "model_shard_index": model_shard_index, "successes": 0, "token_limits": 0, "transient_failures": 0, "permanent_failures": 0, "immediate_invalid_retries": 0, "balance_record": None}
    for row, key in jobs:
        while not stop_event.is_set():
            if time_budget_reached(start_time, args.time_budget_minutes):
                time_event.set()
                stop_event.set()
                break

            with state_lock:
                if key in success_records:
                    break
                token_stage_index = get_item_token_stage_index(key, stage_outcomes)
            token_budget = get_token_budget_for_stage(token_stages, token_stage_index, args.escalate_after_max, args.max_escalation_tokens)
            stage_args = make_stage_args(args, token_budget)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user_prompt(row["title"])}]
            attempt_logger = make_attempt_logger(attempt_path, model_config, row, condition_signature, stage_args, token_budget, token_stage_index, token_plan_signature)
            result = request_chat_completion(client, model_config, messages, stage_args, parse_model_response, attempt_logger)
            record = make_request_record(model_config, row, condition_signature, result, token_budget, token_stage_index, token_plan_signature)
            append_jsonl(prediction_path, record)
            with state_lock:
                stage_outcomes[(key, token_stage_index)] = result["status"]
                if result["status"] == "success":
                    success_records[key] = record

            if result["status"] == "success":
                worker_result["successes"] += 1
                break
            if result["status"] == "token_limit":
                worker_result["token_limits"] += 1
            elif result["status"] == "balance_exhausted":
                worker_result["balance_record"] = {"status": "balance_exhausted", "stopped_at": utc_now(), "provider": model_config["provider"], "model_name": model_config["name"], "model_id": model_config["model"], "token_budget": token_budget, "token_stage_index": token_stage_index, "error": result["error"]}
                stop_event.set()
                break
            elif result["status"] == "transient_failure":
                worker_result["transient_failures"] += 1
                if result.get("error_kind") != "invalid_response":
                    break
                worker_result["immediate_invalid_retries"] += 1
            else:
                worker_result["permanent_failures"] += 1
                break

            if stage_args.request_delay > 0:
                time.sleep(stage_args.request_delay)
    return worker_result


def run_experiment(args):
    # Resolve all inputs and condition signatures before making a paid request.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "llm_predictions.jsonl"
    attempt_path = output_dir / "llm_request_attempts.jsonl"
    balance_path = output_dir / "02_api_balance_stop.json"
    completion_path = output_dir / "llm_title_detection_complete.json"
    if balance_path.exists() and not args.resume_after_balance:
        print(balance_path.read_text(encoding="utf-8"))
        return EXIT_BALANCE
    if balance_path.exists() and args.resume_after_balance:
        balance_path.unlink()

    # Load every configured model under one shared staged request policy.
    evaluation_data = load_evaluation_data(args.core_file, args.ood_file, args.include_ood, args.max_records)
    model_configs = load_model_configs(args.models_json, args.model_names, args.dry_run)
    token_stages = parse_token_stages(args)
    token_plan_signature = make_token_plan_signature(token_stages)
    if args.parallel_workers <= 0:
        raise ValueError("--parallel-workers must be positive.")
    if args.workers_per_model <= 0:
        raise ValueError("--workers-per-model must be positive.")
    if args.max_escalation_tokens <= 0:
        raise ValueError("--max-escalation-tokens must be positive.")
    if args.escalate_after_max and args.max_escalation_tokens < token_stages[-1]:
        raise ValueError("--max-escalation-tokens cannot be below the final base stage.")
    condition_signatures = make_condition_signatures(model_configs, args)
    expected_keys = make_expected_keys(evaluation_data, model_configs)
    expected_metadata = make_expected_metadata(evaluation_data, model_configs)
    evaluation_signature = stable_hash(evaluation_data[["subset", "id", "title", "gold_label"]].sort_values(["subset", "id"]).to_dict("records"))
    history_records = load_jsonl_records(prediction_path)
    success_records = load_success_state(history_records, condition_signatures, expected_metadata)
    stage_outcomes = load_stage_outcomes(history_records, condition_signatures, token_plan_signature, token_stages, args.escalate_after_max, args.max_escalation_tokens, expected_metadata)
    policy_signature = stable_hash({"condition_signatures": condition_signatures, "token_stages": token_stages, "token_stage_policy": TOKEN_STAGE_POLICY, "token_plan_signature": token_plan_signature, "escalate_after_max": args.escalate_after_max, "max_escalation_tokens": args.max_escalation_tokens})

    # A dry run reveals only nonsecret configuration and checkpoint scope.
    if args.dry_run:
        preview = {
            "records": int(len(evaluation_data)),
            "models": [{"name": config["name"], "model": config["model"], "provider": config["provider"], "thinking_profiles": config["thinking_profiles"]} for config in model_configs],
            "expected_requests": len(expected_keys),
            "already_completed": len(set(success_records) & expected_keys),
            "evaluation_signature": evaluation_signature,
            "token_stages": token_stages,
            "token_stage_policy": TOKEN_STAGE_POLICY,
            "token_plan_signature": token_plan_signature,
            "parallel_workers": min(args.parallel_workers, len(model_configs) * args.workers_per_model, len(expected_keys)),
            "workers_per_model": args.workers_per_model,
            "ready_stage_first": args.ready_stage_first,
            "escalate_after_max": args.escalate_after_max,
            "max_escalation_tokens": args.max_escalation_tokens,
            "policy_signature": policy_signature,
            "shared_request": get_shared_request_policy(args)
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return EXIT_COMPLETE

    # Repeat the schedule until every item succeeds; optionally prioritize ready escalations.
    start_time = time.monotonic()
    state_lock = threading.Lock()
    token_stage_index = 0
    active_token_budget = token_stages[0]
    while expected_keys - set(success_records):
        if args.ready_stage_first:
            # Run every incomplete item at its own current stage so unrelated work cannot block it.
            pending_keys = expected_keys - set(success_records)
            jobs_by_model = {config["name"]: [] for config in model_configs}
            for row_index, row in evaluation_data.iterrows():
                for model_config in model_configs:
                    key = (model_config["name"], str(row["subset"]), str(row["id"]))
                    if key in pending_keys:
                        jobs_by_model[model_config["name"]].append((row, key))
            worker_specs = make_model_worker_specs(model_configs, jobs_by_model, args.workers_per_model)
            stop_event = threading.Event()
            time_event = threading.Event()
            worker_results = []
            worker_count = min(args.parallel_workers, len(worker_specs))
            if worker_count <= 0:
                raise RuntimeError("Incomplete requests remain but no item-local worker is ready.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = []
                for model_config, model_shard_index, jobs in worker_specs:
                    future = executor.submit(run_main_model_item_local, model_config, model_shard_index, jobs, args, token_stages, token_plan_signature, condition_signatures[model_config["name"]], prediction_path, attempt_path, success_records, stage_outcomes, state_lock, stop_event, time_event, start_time)
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    worker_results.append(future.result())

            balance_records = [result["balance_record"] for result in worker_results if result["balance_record"]]
            if balance_records:
                stop_record = balance_records[0]
                atomic_write_json(balance_path, stop_record)
                refresh_result_tables(success_records, output_dir, args.bootstrap_rounds, args.seed)
                write_progress(output_dir, expected_keys, success_records, model_configs, "balance_exhausted", token_stages, stage_outcomes, None, None, args.escalate_after_max, args.max_escalation_tokens)
                print(json.dumps(stop_record, ensure_ascii=False, indent=2))
                return EXIT_BALANCE
            if time_event.is_set():
                refresh_result_tables(success_records, output_dir, args.bootstrap_rounds, args.seed)
                progress = write_progress(output_dir, expected_keys, success_records, model_configs, "time_budget_reached", token_stages, stage_outcomes, None, None, args.escalate_after_max, args.max_escalation_tokens)
                progress["item_local_scheduling"] = True
                atomic_write_json(output_dir / "llm_progress.json", progress)
                print(json.dumps(progress, ensure_ascii=False, indent=2))
                return EXIT_INCOMPLETE

            unresolved_keys = expected_keys - set(success_records)
            if unresolved_keys:
                transient_failures = sum(result["transient_failures"] for result in worker_results)
                permanent_failures = sum(result["permanent_failures"] for result in worker_results)
                immediate_invalid_retries = sum(result["immediate_invalid_retries"] for result in worker_results)
                failed_models = sorted({result["model_name"] for result in worker_results if result["transient_failures"] or result["permanent_failures"]})
                status = "transient_failures" if transient_failures else "permanent_failures"
                progress = write_progress(output_dir, expected_keys, success_records, model_configs, status, token_stages, stage_outcomes, None, None, args.escalate_after_max, args.max_escalation_tokens)
                progress["item_local_scheduling"] = True
                progress["immediate_invalid_retries_this_run"] = immediate_invalid_retries
                progress["transient_failures_this_run"] = transient_failures
                progress["permanent_failures_this_run"] = permanent_failures
                progress["suspended_models_this_run"] = failed_models
                progress["unresolved_items"] = len(unresolved_keys)
                atomic_write_json(output_dir / "llm_progress.json", progress)
                if permanent_failures:
                    refresh_result_tables(success_records, output_dir, args.bootstrap_rounds, args.seed)
                    print(json.dumps(progress, ensure_ascii=False, indent=2))
                    return EXIT_FATAL
                continue
            break

        candidate_keys = get_stage_candidate_keys(expected_keys, success_records, stage_outcomes, token_stages, token_stage_index)
        pending_keys = get_stage_pending_keys(candidate_keys, success_records, stage_outcomes, token_stage_index)
        if not pending_keys:
            token_stage_index += 1
            continue

        token_budget = get_token_budget_for_stage(token_stages, token_stage_index, args.escalate_after_max, args.max_escalation_tokens)
        active_token_budget = token_budget

        # Build one stable queue per model to avoid concurrent calls to one model.
        jobs_by_model = {config["name"]: [] for config in model_configs}
        for row_index, row in evaluation_data.iterrows():
            for model_config in model_configs:
                key = (model_config["name"], str(row["subset"]), str(row["id"]))
                if key in pending_keys:
                    jobs_by_model[model_config["name"]].append((row, key))
        worker_specs = make_model_worker_specs(model_configs, jobs_by_model, args.workers_per_model)
        stage_args = make_stage_args(args, token_budget)
        stop_event = threading.Event()
        time_event = threading.Event()
        worker_results = []

        # Parallelism is across model queues; each queue remains deterministic.
        worker_count = min(args.parallel_workers, len(worker_specs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for model_config, model_shard_index, jobs in worker_specs:
                future = executor.submit(run_main_model_stage, model_config, model_shard_index, jobs, stage_args, token_budget, token_stage_index, token_plan_signature, condition_signatures[model_config["name"]], prediction_path, attempt_path, success_records, stage_outcomes, state_lock, stop_event, time_event, start_time)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                worker_results.append(future.result())

        # Stop the complete run immediately after any provider reports no balance.
        balance_records = [result["balance_record"] for result in worker_results if result["balance_record"]]
        if balance_records:
            stop_record = balance_records[0]
            atomic_write_json(balance_path, stop_record)
            refresh_result_tables(success_records, output_dir, args.bootstrap_rounds, args.seed)
            write_progress(output_dir, expected_keys, success_records, model_configs, "balance_exhausted", token_stages, stage_outcomes, token_budget, token_stage_index, args.escalate_after_max, args.max_escalation_tokens)
            print(json.dumps(stop_record, ensure_ascii=False, indent=2))
            return EXIT_BALANCE
        if time_event.is_set():
            refresh_result_tables(success_records, output_dir, args.bootstrap_rounds, args.seed)
            progress = write_progress(output_dir, expected_keys, success_records, model_configs, "time_budget_reached", token_stages, stage_outcomes, token_budget, token_stage_index, args.escalate_after_max, args.max_escalation_tokens)
            print(json.dumps(progress, ensure_ascii=False, indent=2))
            return EXIT_INCOMPLETE

        # A stage advances only when every candidate succeeded or hit its token cap.
        unresolved_keys = get_stage_pending_keys(candidate_keys, success_records, stage_outcomes, token_stage_index)
        if unresolved_keys:
            transient_failures = sum(result["transient_failures"] for result in worker_results)
            permanent_failures = sum(result["permanent_failures"] for result in worker_results)
            failed_models = sorted({result["model_name"] for result in worker_results if result["transient_failures"] or result["permanent_failures"]})
            status = "transient_failures" if transient_failures else "permanent_failures"
            progress = write_progress(output_dir, expected_keys, success_records, model_configs, status, token_stages, stage_outcomes, token_budget, token_stage_index, args.escalate_after_max, args.max_escalation_tokens)
            progress["transient_failures_this_run"] = transient_failures
            progress["permanent_failures_this_run"] = permanent_failures
            progress["suspended_models_this_run"] = failed_models
            progress["unresolved_in_active_stage"] = len(unresolved_keys)
            atomic_write_json(output_dir / "llm_progress.json", progress)
            if args.ready_stage_first and transient_failures and not permanent_failures:
                continue
            refresh_result_tables(success_records, output_dir, args.bootstrap_rounds, args.seed)
            print(json.dumps(progress, ensure_ascii=False, indent=2))
            return EXIT_TRANSIENT if transient_failures else EXIT_FATAL
        if not args.ready_stage_first:
            token_stage_index += 1

    # Refresh publication tables from successful JSONL records after this run.
    refresh_result_tables(success_records, output_dir, args.bootstrap_rounds, args.seed)

    # Mark completion only when every configured model-title pair succeeded.
    if args.ready_stage_first:
        token_stage_index = max((get_item_token_stage_index(key, stage_outcomes) for key in expected_keys), default=0)
        active_token_budget = get_token_budget_for_stage(token_stages, token_stage_index, args.escalate_after_max, args.max_escalation_tokens)
        completed_token_stage_index = token_stage_index
    else:
        completed_token_stage_index = max(token_stage_index - 1, 0)
    progress = write_progress(output_dir, expected_keys, success_records, model_configs, "complete", token_stages, stage_outcomes, active_token_budget, completed_token_stage_index, args.escalate_after_max, args.max_escalation_tokens)
    completion = {"status": "complete", "completed_at": utc_now(), "expected_requests": len(expected_keys), "models": [config["name"] for config in model_configs], "core_rows": int((evaluation_data["subset"] == "core_all").sum()), "ood_rows": int((evaluation_data["subset"] == "out_of_domain").sum()), "evaluation_signature": evaluation_signature, "token_stages": token_stages, "token_stage_policy": TOKEN_STAGE_POLICY, "token_plan_signature": token_plan_signature, "escalate_after_max": args.escalate_after_max, "max_escalation_tokens": args.max_escalation_tokens, "parallel_workers": min(args.parallel_workers, len(model_configs) * args.workers_per_model), "workers_per_model": args.workers_per_model, "policy_signature": policy_signature}
    atomic_write_json(completion_path, completion)
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    return EXIT_COMPLETE


def main():
    # Return a machine-readable exit condition to the scheduled continuation task.
    args = parse_args()
    random.seed(args.seed)
    raise SystemExit(run_experiment(args))


if __name__ == "__main__":
    main()
