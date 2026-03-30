#!/usr/bin/env python3
"""Generate benchmark comparison charts from oha JSON results.

Usage:
  python3 chart.py [results_dir]          # terminal output (default)
  python3 chart.py [results_dir] --png    # save PNGs to charts/
"""

import json
import os
import re
import sys
from collections import defaultdict

RESULTS_DIR = next((a for a in sys.argv[1:] if not a.startswith("-")), "results")
PNG_MODE = "--png" in sys.argv

# gateway -> scenario -> rps -> metrics
data = defaultdict(lambda: defaultdict(dict))

FILE_RE = re.compile(r"^(.+?)-(.+?)-(\d+)(rps|)\.json$")

for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith(".json"):
        continue
    m = FILE_RE.match(fname)
    if not m:
        continue
    gw, scenario, level = m.group(1), m.group(2), int(m.group(3))

    path = os.path.join(RESULTS_DIR, fname)
    try:
        with open(path) as f:
            j = json.load(f)
    except (json.JSONDecodeError, OSError):
        continue

    p = j.get("latencyPercentiles", {})
    s = j.get("summary", {})
    if not p.get("p50"):
        continue

    data[gw][scenario][level] = {
        "p50": p["p50"] * 1000,
        "p90": p["p90"] * 1000,
        "p99": p["p99"] * 1000,
        "rps": s.get("requestsPerSec", 0),
        "success": s.get("successRate", 0),
    }

if not data:
    print(f"No results found in {RESULTS_DIR}/", file=sys.stderr)
    sys.exit(1)

gateways = sorted(data.keys())
all_scenarios = sorted({s for gw in data.values() for s in gw})

COLORS_ANSI = {
    "direct": "\033[90m",
    "crabllm": "\033[91m",
    "bifrost": "\033[94m",
    "litellm": "\033[92m",
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

BAR_CHAR = "█"
BAR_HALF = "▌"
BAR_WIDTH = 50


def ansi(gw):
    return COLORS_ANSI.get(gw, "\033[37m")


def bar(value, max_val):
    if max_val <= 0:
        return ""
    ratio = value / max_val
    full = int(ratio * BAR_WIDTH)
    half = 1 if (ratio * BAR_WIDTH - full) >= 0.5 else 0
    return BAR_CHAR * full + (BAR_HALF if half else "")


def print_header(title):
    print(f"\n{BOLD}{'─' * 70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}")


def chart_latency_scenario(scenario):
    """Show P50/P99 latency bars per gateway at each RPS level."""
    all_levels = sorted({
        l for gw in gateways for l in data[gw].get(scenario, {})
    })
    if not all_levels:
        return

    print_header(f"Latency — {scenario}")

    for level in all_levels:
        entries = []
        for gw in gateways:
            e = data[gw].get(scenario, {}).get(level)
            if e:
                entries.append((gw, e))
        if not entries:
            continue

        max_p99 = max(e["p99"] for _, e in entries)
        print(f"\n  {DIM}{level} RPS{RESET}")
        for gw, e in entries:
            c = ansi(gw)
            p50_bar = bar(e["p50"], max_p99)
            p99_bar = bar(e["p99"], max_p99)
            print(f"    {c}{gw:>8}{RESET} P50 {c}{p50_bar}{RESET} {e['p50']:.2f}ms")
            print(f"    {' ':>8} P99 {c}{p99_bar}{RESET} {e['p99']:.2f}ms")


def chart_overhead_summary():
    """Compare P50 across gateways at max common RPS per scenario."""
    print_header("Gateway Overhead Summary (P50)")

    for scenario in all_scenarios:
        common = None
        for gw in gateways:
            levels = set(data[gw].get(scenario, {}).keys())
            if levels:
                common = levels if common is None else common & levels
        if not common:
            continue

        level = max(common)
        entries = [(gw, data[gw][scenario][level]) for gw in gateways if level in data[gw].get(scenario, {})]
        if not entries:
            continue

        max_val = max(e["p50"] for _, e in entries)
        print(f"\n  {DIM}{scenario} @ {level} RPS{RESET}")
        for gw, e in entries:
            c = ansi(gw)
            b = bar(e["p50"], max_val)
            print(f"    {c}{gw:>8}{RESET} {c}{b}{RESET} {e['p50']:.2f}ms")


def chart_success_summary():
    """Show success rates where they're below 100%."""
    print_header("Success Rates (showing < 100% only)")

    found = False
    for scenario in all_scenarios:
        for gw in gateways:
            for level, e in sorted(data[gw].get(scenario, {}).items()):
                rate = e["success"]
                if rate < 1.0:
                    found = True
                    c = ansi(gw)
                    pct = rate * 100
                    b = bar(rate, 1.0)
                    print(f"    {c}{gw:>8}{RESET} {scenario}@{level} {c}{b}{RESET} {pct:.1f}%")

    if not found:
        print(f"    {DIM}All scenarios at 100% success{RESET}")


# ── Terminal output ──

if not PNG_MODE:
    for scenario in all_scenarios:
        chart_latency_scenario(scenario)
    chart_overhead_summary()
    chart_success_summary()
    print()
    sys.exit(0)

# ── PNG output (requires matplotlib) ──

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("error: --png requires matplotlib — pip install matplotlib", file=sys.stderr)
    sys.exit(1)

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

COLORS_HEX = {
    "direct": "#999999",
    "crabllm": "#e74c3c",
    "bifrost": "#3498db",
    "litellm": "#2ecc71",
}


def gw_color(gw):
    return COLORS_HEX.get(gw, "#95a5a6")


for scenario in all_scenarios:
    fig, ax = plt.subplots(figsize=(10, 5))
    for gw in gateways:
        points = data[gw].get(scenario, {})
        if not points:
            continue
        levels = sorted(points.keys())
        p50 = [points[l]["p50"] for l in levels]
        p99 = [points[l]["p99"] for l in levels]
        color = gw_color(gw)
        ax.plot(levels, p50, "o-", color=color, label=f"{gw} P50")
        ax.plot(levels, p99, "s--", color=color, alpha=0.5, label=f"{gw} P99")
    ax.set_xlabel("Target RPS")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency vs RPS — {scenario}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"latency-{scenario}.png"), dpi=150)
    plt.close(fig)

print(f"Charts saved to {CHARTS_DIR}/")
for f in sorted(os.listdir(CHARTS_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")
