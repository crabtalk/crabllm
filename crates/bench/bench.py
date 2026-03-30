#!/usr/bin/env python3
"""Benchmark orchestrator for crabllm gateway comparison.

Replaces compare.sh (orchestration) and chart.py (visualization) in a single
stdlib-only Python script. Runs inside the Docker runner container.

Usage:
  python3 bench.py                                  # run all groups
  python3 bench.py --group overhead --duration 5    # quick smoke test
  python3 bench.py --rps "100 500"                  # custom RPS levels
  python3 bench.py --chart                          # render terminal charts from existing results
  python3 bench.py --chart --png                    # also export PNGs (requires matplotlib)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError
from urllib.request import Request, urlopen

# ── Gateway definitions ──────────────────────────────────────────────────────

GATEWAYS = [
    {"name": "direct",  "url": "http://localhost:9999", "health": "http://localhost:9999/v1/models",           "proc": "crabllm-bench"},
    {"name": "crabllm", "url": "http://localhost:6666", "health": "http://localhost:6666/health",              "proc": "crabllm"},
    {"name": "bifrost", "url": "http://localhost:6668", "health": "http://localhost:6668/",                     "proc": "bifrost"},
    {"name": "litellm", "url": "http://localhost:4000", "health": "http://localhost:4000/health/liveliness",   "proc": "litellm"},
]

# ── Request bodies ───────────────────────────────────────────────────────────

_J = {"separators": (",", ":")}  # compact JSON, no spaces
CHAT_BODY = json.dumps({"model": "bench-chat", "messages": [{"role": "user", "content": "hi"}]}, **_J)
STREAM_BODY = json.dumps({"model": "bench-chat", "messages": [{"role": "user", "content": "hi"}], "stream": True}, **_J)
EMBED_BODY = json.dumps({"model": "bench-embed", "input": "benchmark text"}, **_J)


def build_large_context(n=50):
    msgs = [{"role": "user", "content": f"This is message number {i} in a long conversation to test how well the gateway handles large request bodies with many messages."} for i in range(1, n + 1)]
    return json.dumps({"model": "bench-chat", "messages": msgs}, **_J)


def build_tools_body(n=10):
    tools = [{"type": "function", "function": {"name": f"tool_{i}", "description": f"Tool number {i} for benchmarking", "parameters": {"type": "object", "properties": {"arg1": {"type": "string"}, "arg2": {"type": "integer"}, "arg3": {"type": "boolean"}}, "required": ["arg1"]}}} for i in range(1, n + 1)]
    return json.dumps({"model": "bench-chat", "messages": [{"role": "user", "content": "hi"}], "tools": tools}, **_J)


LARGE_CONTEXT = build_large_context()
TOOLS_BODY = build_tools_body()

# ── Helpers ──────────────────────────────────────────────────────────────────

DEVNULL = subprocess.DEVNULL


def adapt_body(gw_name, body):
    if gw_name == "bifrost":
        return body.replace('"model":"', '"model":"openai/')
    return body


def sample_memory_kb(proc_name):
    if not proc_name:
        return 0
    try:
        out = subprocess.run(
            ["ps", "-eo", "comm,rss", "--no-headers"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and proc_name in parts[0]:
                return int(parts[1])
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return 0


def fmt_mb(kb):
    return f"{kb / 1024:.1f}MB"


def run_oha(url, endpoint, body, rps, duration, outfile):
    with open(outfile, "w") as f:
        subprocess.run(
            ["oha", "-z", f"{duration}s", "-q", str(rps),
             "-m", "POST", "-H", "Content-Type: application/json",
             "-d", body, "--output-format", "json", "--no-tui",
             f"{url}{endpoint}"],
            stdout=f, stderr=DEVNULL,
        )


def run_oha_concurrent(url, endpoint, body, conc, duration, outfile):
    with open(outfile, "w") as f:
        subprocess.run(
            ["oha", "-z", f"{duration}s", "-c", str(conc),
             "-m", "POST", "-H", "Content-Type: application/json",
             "-d", body, "--output-format", "json", "--no-tui",
             f"{url}{endpoint}"],
            stdout=f, stderr=DEVNULL,
        )


def warmup(url, endpoint, body):
    subprocess.run(
        ["oha", "-n", "100", "-m", "POST",
         "-H", "Content-Type: application/json",
         "-d", body, "--no-tui", f"{url}{endpoint}"],
        stdout=DEVNULL, stderr=DEVNULL,
    )


# ── Health checks (async) ───────────────────────────────────────────────────

def _check_health(url, timeout=5):
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout):
            return True
    except (URLError, OSError, TimeoutError):
        return False


def _poll_gateway(gw, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _check_health(gw["health"]):
            return gw
        time.sleep(0.1)
    return None


def wait_for_gateways(gateways, timeout=60):
    print("==> Waiting for gateways...")
    active = []
    with ThreadPoolExecutor(max_workers=len(gateways)) as pool:
        futures = {pool.submit(_poll_gateway, gw, timeout): gw for gw in gateways}
        for future in as_completed(futures):
            gw = futures[future]
            if future.result():
                print(f"  {gw['name']}: ready")
                active.append(gw)
            else:
                print(f"  SKIP: {gw['name']} not available")
    # Preserve gateway order
    order = {gw["name"]: i for i, gw in enumerate(gateways)}
    active.sort(key=lambda gw: order[gw["name"]])
    print()
    return active


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_scenario(name, endpoint, body, rps_levels, active_gws, duration, outdir):
    print(f"== Scenario: {name} ==")
    for gw in active_gws:
        gw_body = adapt_body(gw["name"], body)
        mem = fmt_mb(sample_memory_kb(gw["proc"]))
        print(f"  [{gw['name']}] warming up... (mem: {mem})")
        warmup(gw["url"], endpoint, gw_body)
        for rps in rps_levels:
            outfile = os.path.join(outdir, f"{gw['name']}-{name}-{rps}rps.json")
            print(f"  [{gw['name']}] {rps} RPS for {duration}s...")
            run_oha(gw["url"], endpoint, gw_body, rps, duration, outfile)
            mem_kb = sample_memory_kb(gw["proc"])
            print(f"  [{gw['name']}] done (mem: {fmt_mb(mem_kb)})")
            mem_file = os.path.join(outdir, f"{gw['name']}-{name}-{rps}rps.mem.json")
            with open(mem_file, "w") as f:
                json.dump({"memory_kb": mem_kb}, f)
    print()


def run_overhead(active_gws, duration, rps_levels, outdir):
    print("==> Group 1: Gateway Overhead (instant responses)")
    print()
    run_scenario("chat-minimal", "/v1/chat/completions", CHAT_BODY, rps_levels, active_gws, duration, outdir)
    run_scenario("chat-stream", "/v1/chat/completions", STREAM_BODY, rps_levels, active_gws, duration, outdir)
    run_scenario("embeddings", "/v1/embeddings", EMBED_BODY, rps_levels, active_gws, duration, outdir)


def run_payload(active_gws, duration, outdir):
    rps = [100, 500, 1000]
    print("==> Group 2: Payload Stress")
    print()
    run_scenario("chat-large-context", "/v1/chat/completions", LARGE_CONTEXT, rps, active_gws, duration, outdir)
    run_scenario("chat-long-stream", "/v1/chat/completions", STREAM_BODY, rps, active_gws, duration, outdir)
    run_scenario("chat-tools", "/v1/chat/completions", TOOLS_BODY, rps, active_gws, duration, outdir)


def run_concurrent(active_gws, duration, outdir):
    print("==> Group 3: Concurrent Streams")
    print()
    for conc in [100, 500, 1000]:
        print(f"== concurrent-streams-{conc} ==")
        for gw in active_gws:
            gw_body = adapt_body(gw["name"], STREAM_BODY)
            warmup(gw["url"], "/v1/chat/completions", gw_body)
            outfile = os.path.join(outdir, f"{gw['name']}-concurrent-{conc}.json")
            print(f"  [{gw['name']}] {conc} concurrent streams for {duration}s...")
            run_oha_concurrent(gw["url"], "/v1/chat/completions", gw_body, conc, duration, outfile)
            mem_kb = sample_memory_kb(gw["proc"])
            print(f"  [{gw['name']}] done (mem: {fmt_mb(mem_kb)})")
            mem_file = os.path.join(outdir, f"{gw['name']}-concurrent-{conc}.mem.json")
            with open(mem_file, "w") as f:
                json.dump({"memory_kb": mem_kb}, f)
        print()


# ── Validation & summary ────────────────────────────────────────────────────

def validate_results(outdir):
    print("==> Validating results...")
    failed = False
    for fname in sorted(os.listdir(outdir)):
        if not fname.endswith(".json") or fname.endswith(".mem.json"):
            continue
        path = os.path.join(outdir, fname)
        try:
            with open(path) as f:
                json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  FAIL: {fname} is not valid JSON")
            failed = True
    if failed:
        print("Some results are invalid. Check gateway connectivity.")
        return False
    print("  OK")
    print()
    return True


def print_summary(outdir):
    print("==> Results")
    print()
    header = f"{'scenario':<40} {'RPS':>10} {'P50':>10} {'P90':>10} {'P99':>10} {'success':>8}"
    print(header)
    print("-" * len(header))

    for fname in sorted(os.listdir(outdir)):
        if not fname.endswith(".json") or fname.endswith(".mem.json"):
            continue
        path = os.path.join(outdir, fname)
        try:
            with open(path) as f:
                j = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        p = j.get("latencyPercentiles", {})
        s = j.get("summary", {})
        p50 = p.get("p50")
        if p50 is None:
            continue

        name = fname.removesuffix(".json")
        p50_ms = f"{p50 * 1000:.2f}ms"
        p90_ms = f"{p.get('p90', 0) * 1000:.2f}ms"
        p99_ms = f"{p.get('p99', 0) * 1000:.2f}ms"
        rps = f"{s.get('requestsPerSec', 0):.0f}"
        success = f"{s.get('successRate', 0)}"
        print(f"{name:<40} {rps:>10} {p50_ms:>10} {p90_ms:>10} {p99_ms:>10} {success:>8}")

    print()
    print(f"Results saved to {outdir}/")


# ── Chart rendering ──────────────────────────────────────────────────────────

COLORS_ANSI = {
    "direct": "\033[90m",
    "crabllm": "\033[38;5;208m",
    "bifrost": "\033[38;5;39m",
    "litellm": "\033[38;5;25m",
}
COLORS_HEX = {
    "direct": "#999999",
    "crabllm": "#dea584",
    "bifrost": "#00ADD8",
    "litellm": "#3178C6",
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BAR_CHAR = "\u2588"
BAR_HALF = "\u258c"
BAR_WIDTH = 50


def _ansi(gw):
    return COLORS_ANSI.get(gw, "\033[37m")


def _bar(value, max_val):
    if max_val <= 0:
        return ""
    ratio = value / max_val
    full = int(ratio * BAR_WIDTH)
    half = 1 if (ratio * BAR_WIDTH - full) >= 0.5 else 0
    return BAR_CHAR * full + (BAR_HALF if half else "")


def _header(title):
    print(f"\n{BOLD}{'─' * 70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}")


def _parse_result_filename(fname, known_gws):
    """Parse {gw}-{scenario}-{level}rps.json, handling hyphens in scenario names."""
    if not fname.endswith(".json") or fname.endswith(".mem.json"):
        return None
    for gw in known_gws:
        prefix = gw + "-"
        if not fname.startswith(prefix):
            continue
        rest = fname[len(prefix):-len(".json")]  # e.g. "chat-large-context-1000rps"
        m = re.match(r"^(.+)-(\d+)(rps)?$", rest)
        if m:
            return gw, m.group(1), int(m.group(2))
    return None


def load_results(outdir):
    data = defaultdict(lambda: defaultdict(dict))
    known_gws = {gw["name"] for gw in GATEWAYS}

    for fname in sorted(os.listdir(outdir)):
        parsed = _parse_result_filename(fname, known_gws)
        if not parsed:
            continue
        gw, scenario, level = parsed

        path = os.path.join(outdir, fname)
        try:
            with open(path) as f:
                j = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        p = j.get("latencyPercentiles", {})
        s = j.get("summary", {})
        if not p.get("p50"):
            continue

        mem_kb = 0
        mem_path = path.replace(".json", ".mem.json")
        try:
            with open(mem_path) as mf:
                mem_kb = json.load(mf).get("memory_kb", 0)
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            pass

        data[gw][scenario][level] = {
            "p50": p["p50"] * 1000,
            "p90": p["p90"] * 1000,
            "p99": p["p99"] * 1000,
            "rps": s.get("requestsPerSec", 0),
            "success": s.get("successRate", 0),
            "memory_mb": mem_kb / 1024,
        }
    return data


def chart_latency(data, gateways, scenario):
    all_levels = sorted({l for gw in gateways for l in data[gw].get(scenario, {})})
    if not all_levels:
        return
    _header(f"Latency — {scenario}")
    for level in all_levels:
        entries = [(gw, data[gw][scenario][level]) for gw in gateways if level in data[gw].get(scenario, {})]
        if not entries:
            continue
        max_p99 = max(e["p99"] for _, e in entries)
        print(f"\n  {DIM}{level} RPS{RESET}")
        for gw, e in entries:
            c = _ansi(gw)
            print(f"    {c}{gw:>8}{RESET} P50 {c}{_bar(e['p50'], max_p99)}{RESET} {e['p50']:.2f}ms")
            print(f"    {' ':>8} P99 {c}{_bar(e['p99'], max_p99)}{RESET} {e['p99']:.2f}ms")


def chart_overhead(data, gateways, scenarios):
    _header("Gateway Overhead Summary (P50)")
    for scenario in scenarios:
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
            c = _ansi(gw)
            print(f"    {c}{gw:>8}{RESET} {c}{_bar(e['p50'], max_val)}{RESET} {e['p50']:.2f}ms")


def chart_memory(data, gateways, scenarios):
    _header("Memory Usage (RSS)")
    peak = {}
    for gw in gateways:
        for sc in data[gw]:
            for e in data[gw][sc].values():
                mem = e.get("memory_mb", 0)
                if mem > peak.get(gw, 0):
                    peak[gw] = mem
    if not any(peak.values()):
        print(f"    {DIM}No memory data available{RESET}")
        return
    max_mem = max(peak.values())
    print(f"\n  {DIM}Peak RSS across all scenarios{RESET}")
    for gw in gateways:
        mem = peak.get(gw, 0)
        if mem <= 0:
            continue
        c = _ansi(gw)
        print(f"    {c}{gw:>8}{RESET} {c}{_bar(mem, max_mem)}{RESET} {mem:.1f}MB")


def chart_success(data, gateways, scenarios):
    _header("Success Rates (showing < 100% only)")
    found = False
    for sc in scenarios:
        for gw in gateways:
            for level, e in sorted(data[gw].get(sc, {}).items()):
                if e["success"] < 1.0:
                    found = True
                    c = _ansi(gw)
                    pct = e["success"] * 100
                    print(f"    {c}{gw:>8}{RESET} {sc}@{level} {c}{_bar(e['success'], 1.0)}{RESET} {pct:.1f}%")
    if not found:
        print(f"    {DIM}All scenarios at 100% success{RESET}")


def render_terminal_charts(outdir):
    data = load_results(outdir)
    if not data:
        print(f"No results found in {outdir}/", file=sys.stderr)
        return
    gateways = sorted(data.keys())
    scenarios = sorted({s for gw in data.values() for s in gw})
    for sc in scenarios:
        chart_latency(data, gateways, sc)
    chart_overhead(data, gateways, scenarios)
    chart_memory(data, gateways, scenarios)
    chart_success(data, gateways, scenarios)
    print()


def render_png_charts(outdir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("error: --png requires matplotlib — pip install matplotlib", file=sys.stderr)
        return

    data = load_results(outdir)
    if not data:
        return
    gateways = sorted(data.keys())
    scenarios = sorted({s for gw in data.values() for s in gw})

    charts_dir = os.path.join(outdir, "..", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    for scenario in scenarios:
        fig, ax = plt.subplots(figsize=(10, 5))
        for gw in gateways:
            points = data[gw].get(scenario, {})
            if not points:
                continue
            levels = sorted(points.keys())
            p50 = [points[l]["p50"] for l in levels]
            p99 = [points[l]["p99"] for l in levels]
            color = COLORS_HEX.get(gw, "#95a5a6")
            ax.plot(levels, p50, "o-", color=color, label=f"{gw} P50")
            ax.plot(levels, p99, "s--", color=color, alpha=0.5, label=f"{gw} P99")
        ax.set_xlabel("Target RPS")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Latency vs RPS — {scenario}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, f"latency-{scenario}.png"), dpi=150)
        plt.close(fig)

    print(f"Charts saved to {charts_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="crabllm benchmark orchestrator")
    parser.add_argument("--group", default="overhead", choices=["overhead", "payload", "concurrent", "all"])
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--rps", default="100 500 1000 2000 5000")
    parser.add_argument("--output", default="/results")
    parser.add_argument("--chart", action="store_true", help="render terminal charts after run")
    parser.add_argument("--chart-only", action="store_true", help="render charts from existing results (no benchmark)")
    parser.add_argument("--png", action="store_true", help="export PNG charts (requires matplotlib)")

    args = parser.parse_args()
    outdir = args.output
    rps_levels = [int(x) for x in args.rps.split()]

    # Chart-only mode: just render from existing results
    if args.chart_only:
        render_terminal_charts(outdir)
        if args.png:
            render_png_charts(outdir)
        return

    # Full benchmark run
    os.makedirs(outdir, exist_ok=True)

    active = wait_for_gateways(GATEWAYS)
    if not active:
        print("No gateways available, exiting.", file=sys.stderr)
        sys.exit(1)

    names = " ".join(gw["name"] for gw in active)
    print(f"==> Active gateways: {names}")
    print(f"==> Duration: {args.duration}s per scenario, RPS levels: {args.rps}")
    print()

    group = args.group
    if group in ("overhead", "all"):
        run_overhead(active, args.duration, rps_levels, outdir)
    if group in ("payload", "all"):
        run_payload(active, args.duration, outdir)
    if group in ("concurrent", "all"):
        run_concurrent(active, args.duration, outdir)

    validate_results(outdir)
    print_summary(outdir)

    if args.chart:
        render_terminal_charts(outdir)
    if args.png:
        render_png_charts(outdir)


if __name__ == "__main__":
    main()
