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


def get_size_mb(model) -> int | None:
    """Extract total model size in MB from safetensors metadata."""
    st = getattr(model, "safetensors", None)
    if not st:
        return None
    # safetensors has 'total' (total params) and 'parameters' (per dtype).
    # We want disk size. Each param takes dtype bits / 8 bytes.
    # But the API doesn't give disk size directly — compute from param counts.
    params = st.get("parameters", {})
    if not params:
        # Fall back to total param count with rough estimate.
        total = st.get("total", 0)
        if total == 0:
            return None
        # Assume average 4 bits per param for quantized models.
        return max(1, int(total * 0.5 / 1_000_000))

    total_bytes = 0
    dtype_bytes = {
        "F32": 4, "F16": 2, "BF16": 2,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1,
        "U8": 1, "Q8_0": 1, "Q6_K": 0.75,
        "Q5_K": 0.625, "Q5_1": 0.625, "Q5_0": 0.625,
        "Q4_K": 0.5, "Q4_1": 0.5, "Q4_0": 0.5,
        "Q3_K": 0.375, "Q2_K": 0.25,
    }
    for dtype, count in params.items():
        bpp = dtype_bytes.get(dtype, 2)  # default 2 bytes (FP16)
        total_bytes += count * bpp

    return max(1, int(total_bytes / (1024 * 1024)))


def main():
    print(f"Fetching models from {ORG} on HuggingFace ...")
    all_models = list(
        list_models(author=ORG, sort="downloads", direction=-1, expand=["safetensors"])
    )
    print(f"  Found {len(all_models)} total repos")

    entries: list[tuple[str, str, int | None]] = []
    for m in all_models:
        repo_id = m.id
        if not repo_id.lower().endswith(QUANT_SUFFIXES):
            continue
        alias = make_alias(repo_id)
        size_mb = get_size_mb(m)
        entries.append((alias, repo_id, size_mb))

    entries.sort()

    lines = [
        "# Local model registry — auto-generated from mlx-community on HuggingFace.",
        f"# Source: https://huggingface.co/{ORG}",
        "#",
        "# Regenerate: python3 scripts/update_local_models.py",
        "#",
        "# Each entry maps a short alias to a HuggingFace repo ID with disk size.",
        "# Comment out models you don't need.",
        "",
    ]

    for alias, repo_id, size_mb in entries:
        # Quote alias if it contains dots (TOML treats dots as table separators).
        key = f'"{alias}"' if "." in alias else alias
        lines.append(f"[models.{key}]")
        lines.append(f'repo_id = "{repo_id}"')
        if size_mb is not None:
            lines.append(f"size_mb = {size_mb}")
        lines.append("")

    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))

    with_size = sum(1 for _, _, s in entries if s is not None)
    print(f"Wrote {len(entries)} quantized models to {OUTPUT} ({with_size} with size info)")


if __name__ == "__main__":
    main()
