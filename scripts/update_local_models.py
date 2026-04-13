#!/usr/bin/env python3
"""Fetch mlx-community models from HuggingFace and generate local.toml.

Only includes models whose architecture (model_type in config.json) is
supported by mlx-swift-lm. The supported types are extracted from the
Swift factory source files at build time.

Requires: pip install huggingface_hub
"""

import re
from pathlib import Path

from huggingface_hub import list_models

OUTPUT = "models/local.toml"
ORG = "mlx-community"

# Only include quantized MLX models (the ones you'd actually run locally).
QUANT_SUFFIXES = ("-4bit", "-8bit", "-3bit", "-6bit")

# Swift factory source files that register supported model_type strings.
SWIFT_FACTORIES = [
    "mlx/.build/checkouts/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift",
    "mlx/.build/checkouts/mlx-swift-lm/Libraries/MLXVLM/VLMModelFactory.swift",
]


def load_supported_model_types() -> tuple[set[str], set[str]]:
    """Extract supported model_type strings from mlx-swift-lm factory source.

    Returns (all_types, vlm_types) — vlm_types is the subset that supports vision.
    """
    all_types: set[str] = set()
    vlm_types: set[str] = set()
    pattern = re.compile(r'"([^"]+)":\s*create\(')

    for path in SWIFT_FACTORIES:
        is_vlm = "VLM" in path
        try:
            text = Path(path).read_text()
        except FileNotFoundError:
            print(f"  warning: {path} not found, skipping")
            continue
        for m in pattern.finditer(text):
            model_type = m.group(1)
            all_types.add(model_type)
            if is_vlm:
                vlm_types.add(model_type)

    return all_types, vlm_types


def parse_model(repo_id: str) -> tuple[str, str, str] | None:
    """Parse 'mlx-community/Qwen3.5-2B-MLX-4bit' into ('qwen3.5', '2b', '4bit').

    Returns (family, size, quant) or None if the name can't be parsed.
    """
    import re

    name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
    name = name.lower()

    # Strip MLX marker.
    name = re.sub(r"-?mlx-?", "-", name).strip("-")
    name = re.sub(r"-{2,}", "-", name)

    # Extract quant suffix (e.g., "4bit", "8bit").
    quant_match = re.search(r"-(\d+bit)$", name)
    # Extract param size (e.g., "2b", "70b", "0.5b", "135m").
    size_match = re.search(r"-(\d+(?:\.\d+)?[bm])(?=-|$)", name)

    if not (size_match and quant_match):
        return None

    quant = quant_match.group(1)  # e.g. "4bit"
    size = size_match.group(1)  # e.g. "8b", "0.6b", "135m"
    # Family = everything before the size. Qualifiers between size and
    # quant (e.g., "-instruct") are appended to keep aliases unique.
    prefix = name[: size_match.start()]
    suffix = name[size_match.end() : quant_match.start()].strip("-")
    family = f"{prefix}-{suffix}" if suffix else prefix
    return (family, size, quant)


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
        return None

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


def get_model_type(model) -> str | None:
    """Extract model_type from HuggingFace model config."""
    cfg = getattr(model, "config", None)
    if not cfg:
        return None
    return cfg.get("model_type")


def main():
    supported, vlm_types = load_supported_model_types()
    print(f"Supported model types ({len(supported)}), VLM types ({len(vlm_types)}): {', '.join(sorted(vlm_types))}")

    print(f"Fetching models from {ORG} on HuggingFace ...")
    all_models = list(
        list_models(
            author=ORG,
            sort="downloads",
            direction=-1,
            expand=["config", "safetensors"],
        )
    )
    print(f"  Found {len(all_models)} total repos")

    # Parse into (family, size, quant, repo_id, size_mb).
    # Models are sorted by downloads (descending), so first occurrence wins.
    seen: set[tuple[str, str, str]] = set()
    entries: list[tuple[str, str, str, str, int, bool]] = []
    skipped = 0
    unsupported = 0
    dupes = 0
    for m in all_models:
        repo_id = m.id
        if not repo_id.lower().endswith(QUANT_SUFFIXES):
            continue
        # Filter by supported architecture.
        model_type = get_model_type(m)
        if model_type and model_type not in supported:
            unsupported += 1
            continue
        is_vlm = model_type in vlm_types if model_type else False
        parsed = parse_model(repo_id)
        if not parsed:
            skipped += 1
            continue
        family, size, quant = parsed
        key = (family, size, quant)
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        size_mb = get_size_mb(m)
        if size_mb is None:
            skipped += 1
            continue
        entries.append((family, size, quant, repo_id, size_mb, is_vlm))

    entries.sort()

    lines = [
        "# Local model registry — auto-generated from mlx-community on HuggingFace.",
        f"# Source: https://huggingface.co/{ORG}",
        "#",
        "# Regenerate: python3 scripts/update_local_models.py",
        "#",
        "# Format: [models.{family}.{size}.{quant}]",
        "# Family names with dots are quoted: [models.\"qwen3.5\".2b.4bit]",
        "",
    ]

    def toml_key(s: str) -> str:
        """Quote a TOML key if it contains dots."""
        return f'"{s}"' if "." in s else s

    for family, size, quant, repo_id, size_mb, is_vlm in entries:
        lines.append(f"[models.{toml_key(family)}.{toml_key(size)}.{toml_key(quant)}]")
        lines.append(f'repo_id = "{repo_id}"')
        lines.append(f"size_mb = {size_mb}")
        if is_vlm:
            lines.append("vision = true")
        lines.append("")

    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))

    print(
        f"Wrote {len(entries)} models to {OUTPUT} "
        f"({unsupported} unsupported arch, {skipped} skipped, {dupes} dupes)"
    )


if __name__ == "__main__":
    main()
