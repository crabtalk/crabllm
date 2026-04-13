#!/usr/bin/env python3
"""Fetch mlx-community models from HuggingFace and generate local.toml.

Requires: pip install huggingface_hub
"""

from huggingface_hub import list_models

OUTPUT = "models/local.toml"
ORG = "mlx-community"

# Only include quantized MLX models (the ones you'd actually run locally).
QUANT_SUFFIXES = ("-4bit", "-8bit", "-3bit", "-6bit")


def make_alias(repo_id: str) -> str:
    """Convert 'mlx-community/Qwen3.5-2B-MLX-4bit' to 'qwen3.5-2b-mlx-4bit'."""
    name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
    return name.lower()


def main():
    print(f"Fetching models from {ORG} on HuggingFace ...")
    all_models = list(list_models(author=ORG, sort="downloads", direction=-1))
    print(f"  Found {len(all_models)} total repos")

    entries: list[tuple[str, str]] = []
    for m in all_models:
        repo_id = m.id
        if not repo_id.lower().endswith(QUANT_SUFFIXES):
            continue
        alias = make_alias(repo_id)
        entries.append((alias, repo_id))

    entries.sort()

    lines = [
        "# Local model registry — auto-generated from mlx-community on HuggingFace.",
        f"# Source: https://huggingface.co/{ORG}",
        "#",
        "# Regenerate: python3 scripts/update_local_models.py",
        "#",
        "# Each entry maps a short alias to a HuggingFace repo ID.",
        "# Comment out models you don't need.",
        "",
        "[models]",
    ]

    for alias, repo_id in entries:
        lines.append(f'{alias} = "{repo_id}"')

    lines.append("")

    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(entries)} quantized models to {OUTPUT}")


if __name__ == "__main__":
    main()
