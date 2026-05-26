#!/usr/bin/env python3
"""Fetch model pricing from LiteLLM's community dataset and generate cloud.toml."""

import json
import sys
from urllib.request import urlopen

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
OUTPUT = "models/cloud.toml"

# Provider prefixes we care about. LiteLLM keys are "provider/model" or just "model".
PROVIDERS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google",
    "vertex_ai": "Google",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "groq": "Groq",
    "together_ai": "Together",
}

# Models we skip (fine-tuned, deprecated, duplicates).
SKIP_PATTERNS = [
    "ft:",
    "sample_spec",
    "audio",
    "realtime",
    "search",
    "tts",
    "dall-e",
    "whisper",
    "davinci",
    "babbage",
    "ada",  # old completions models, not ada-002 embedding
    "moderation",
]


def should_skip(key: str) -> bool:
    lower = key.lower()
    return any(p in lower for p in SKIP_PATTERNS)


def extract_model_key(key: str) -> tuple[str, str] | None:
    """Extract (provider_label, model_key) from a LiteLLM key.

    Provider-prefixed keys like "openai/gpt-4o" become ("OpenAI", "openai/gpt-4o").
    Bare keys like "gpt-4o" are kept as-is with provider from litellm_provider.
    Returns None for providers we don't track.
    """
    parts = key.split("/", 1)
    if len(parts) == 2:
        provider_prefix = parts[0]
        if provider_prefix not in PROVIDERS:
            return None
        return (PROVIDERS[provider_prefix], key)
    # Bare key — will get provider label from litellm_provider field later.
    return ("", key)


def to_toml_key(name: str) -> str:
    """Always quote the key — model names can contain dots and slashes."""
    return f'["{name}"]'


def to_toml_pricing_key(name: str) -> str:
    return f'["{name}".pricing]'


def to_per_million(per_token):
    """Convert LiteLLM's per-token cost to per-million, rounded to 4 places."""
    if per_token is None:
        return None
    return round(per_token * 1_000_000, 4)


def extract_search_per_call(info):
    """LiteLLM's `search_context_cost_per_query` is a sub-object keyed by
    context size. Anthropic and OpenAI typically charge the same regardless of
    size, so we pick the medium tier as the canonical per-call rate."""
    bucket = info.get("search_context_cost_per_query")
    if not isinstance(bucket, dict):
        return None
    for size in ("search_context_size_medium", "search_context_size_high", "search_context_size_low"):
        if size in bucket and bucket[size] is not None and bucket[size] > 0:
            return float(bucket[size])
    return None


def main():
    print(f"Fetching {LITELLM_URL} ...")
    with urlopen(LITELLM_URL) as resp:
        data = json.loads(resp.read())

    # Remove the "sample_spec" entry if present.
    data.pop("sample_spec", None)

    # Collect models grouped by provider.
    models: dict[str, dict] = {}  # model_name -> info
    seen_providers: dict[str, str] = {}  # model_name -> provider_label

    for key, info in data.items():
        if should_skip(key):
            continue

        result = extract_model_key(key)
        if not result:
            continue
        provider_label, model_key = result

        # Need pricing and context length.
        input_cost = info.get("input_cost_per_token")
        output_cost = info.get("output_cost_per_token")
        context = info.get("max_input_tokens") or info.get("max_tokens")

        if input_cost is None or context is None:
            continue

        # Skip duplicates.
        if model_key in models:
            continue

        # Resolve provider label from litellm_provider for bare keys.
        if not provider_label:
            provider = info.get("litellm_provider", "")
            provider_label = PROVIDERS.get(provider.split("/")[0], provider)

        pricing = {
            "input_cost_per_million": to_per_million(input_cost),
            "output_cost_per_million": to_per_million(output_cost or 0),
        }
        # Each of these is dropped on the canonical floor if missing — meaning
        # billing falls back to the coarser bucket. Only emit when LiteLLM has
        # the data.
        for src_field, dst_field in [
            ("cache_read_input_token_cost", "cache_read_cost_per_million"),
            ("cache_creation_input_token_cost", "cache_write_cost_per_million"),
            ("output_cost_per_reasoning_token", "reasoning_cost_per_million"),
            ("input_cost_per_audio_token", "audio_input_cost_per_million"),
            ("output_cost_per_audio_token", "audio_output_cost_per_million"),
        ]:
            v = info.get(src_field)
            if v is not None and v > 0:
                pricing[dst_field] = to_per_million(v)

        search_per_call = extract_search_per_call(info)

        entry = {
            "context_length": int(context),
            "pricing": pricing,
        }
        if search_per_call is not None:
            entry["server_tool_cost_per_call"] = {"web_search": search_per_call}
        if info.get("supports_vision"):
            entry["vision"] = True
        models[model_key] = entry
        seen_providers[model_key] = provider_label

    # Group by provider for organized output.
    by_provider: dict[str, list[str]] = {}
    for name, label in seen_providers.items():
        by_provider.setdefault(label, []).append(name)

    # Sort providers and models within each provider.
    lines = [
        "# Cloud model metadata — auto-generated from LiteLLM's dataset.",
        f"# Source: {LITELLM_URL}",
        "#",
        "# Regenerate: python3 scripts/update_cloud_models.py",
        "",
    ]

    # Field order for [<model>.pricing]. Required fields first, then optional
    # ones — matches the struct definition in PricingConfig.
    PRICING_FIELDS = [
        "input_cost_per_million",
        "output_cost_per_million",
        "cache_read_cost_per_million",
        "cache_write_cost_per_million",
        "reasoning_cost_per_million",
        "audio_input_cost_per_million",
        "audio_output_cost_per_million",
    ]

    count = 0
    for provider in sorted(by_provider):
        lines.append(f"# {provider}")
        for name in sorted(by_provider[provider]):
            info = models[name]
            ctx = f"{info['context_length']:_}"
            lines.append(f"{to_toml_key(name)}")
            lines.append(f"context_length = {ctx}")
            if info.get("vision"):
                lines.append("vision = true")
            pricing = info["pricing"]
            lines.append(f"{to_toml_pricing_key(name)}")
            for field in PRICING_FIELDS:
                if field in pricing:
                    lines.append(f"{field} = {pricing[field]}")
            tool_costs = info.get("server_tool_cost_per_call")
            if tool_costs:
                pairs = ", ".join(f'"{k}" = {v}' for k, v in sorted(tool_costs.items()))
                lines.append(f"server_tool_cost_per_call = {{ {pairs} }}")
            lines.append("")
            count += 1

    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {count} models to {OUTPUT}")


if __name__ == "__main__":
    main()
