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


def extract_model_name(key: str) -> str | None:
    """Strip provider prefix from LiteLLM key to get the model name."""
    # Some keys have provider prefix: "openai/gpt-4o", "anthropic/claude-..."
    # Others are bare: "gpt-4o", "claude-3-5-sonnet-20241022"
    parts = key.split("/", 1)
    if len(parts) == 2:
        provider_prefix, model = parts
        # Skip providers we don't track
        if provider_prefix not in PROVIDERS:
            return None
        return model
    # Bare key — include if it looks like a known model
    return key


def to_toml_key(name: str) -> str:
    """Always quote the key — model names can contain dots and slashes."""
    return f'["{name}"]'


def to_toml_pricing_key(name: str) -> str:
    return f'["{name}".pricing]'


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

        model_name = extract_model_name(key)
        if not model_name:
            continue

        # Need pricing and context length.
        input_cost = info.get("input_cost_per_token")
        output_cost = info.get("output_cost_per_token")
        context = info.get("max_input_tokens") or info.get("max_tokens")

        if input_cost is None or context is None:
            continue

        # Skip if we already have this model (first occurrence wins,
        # which is usually the canonical entry).
        if model_name in models:
            continue

        provider = info.get("litellm_provider", "")
        provider_label = PROVIDERS.get(provider.split("/")[0], provider)

        entry = {
            "context_length": int(context),
            "prompt_cost_per_million": round(input_cost * 1_000_000, 4),
            "completion_cost_per_million": round((output_cost or 0) * 1_000_000, 4),
        }
        if info.get("supports_vision"):
            entry["vision"] = True
        models[model_name] = entry
        seen_providers[model_name] = provider_label

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
            lines.append(f"{to_toml_pricing_key(name)}")
            lines.append(f"prompt_cost_per_million = {info['prompt_cost_per_million']}")
            lines.append(
                f"completion_cost_per_million = {info['completion_cost_per_million']}"
            )
            lines.append("")
            count += 1

    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {count} models to {OUTPUT}")


if __name__ == "__main__":
    main()
