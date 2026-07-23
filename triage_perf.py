"""Latency + memory instrumentation for the on-device triage demo.

Measures wall-clock per-query serve latency (device-synced) and peak process /
accelerator memory on a fixed held-out batch after warmup. Shared by
run_baselines.py and run_sdft.py; numbers land in outputs/perf.json (and are
merged into baselines.json / results.json).
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import psutil
import torch

from triage_common import (
    FIG_DIR, MAX_NEW, OUT_DIR, STREAM_LEN, build_msgs, make_retriever,
    recent_demos, to_model_device,
)

PERF_JSON = OUT_DIR / "perf.json"
PERF_WARMUP = 2          # discarded timed calls before the measured batch
PERF_FIG = FIG_DIR / "online_sdft_perf.png"


def sync_device(device: str) -> None:
    """Block until pending device kernels finish (honest wall-clock)."""
    kind = device.split(":")[0]
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def device_kind(model_or_device) -> str:
    if isinstance(model_or_device, str):
        return model_or_device.split(":")[0]
    return str(next(model_or_device.parameters()).device).split(":")[0]


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def peak_rss_mb() -> float:
    """Process peak RSS (ru_maxrss). Darwin reports bytes; Linux reports KiB."""
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def device_allocated_mb(device: str) -> float | None:
    kind = device.split(":")[0]
    if kind == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if kind == "mps":
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return None


def reset_device_peak(device: str) -> None:
    kind = device.split(":")[0]
    if kind == "cuda":
        torch.cuda.reset_peak_memory_stats()


@torch.inference_mode()
def timed_generate_one(model, tok, messages: list[dict], *,
                       max_new: int = MAX_NEW) -> tuple[float, int]:
    """Greedy decode one chat; return (wall-clock ms, new-token count) with sync."""
    device = device_kind(model)
    model.eval()
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    encoding = to_model_device(
        tok(text, return_tensors="pt", add_special_tokens=False), model)
    prompt_len = encoding["input_ids"].shape[1]
    sync_device(device)
    t0 = time.perf_counter()
    output = model.generate(**encoding, max_new_tokens=max_new, do_sample=False,
                            pad_token_id=tok.pad_token_id)
    sync_device(device)
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, int(output.shape[1] - prompt_len)


def _percentile(ordered: list[float], p: float) -> float:
    if not ordered:
        return float("nan")
    idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[idx]


def summarize_latencies(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    return {
        "latency_ms_median": statistics.median(ordered),
        "latency_ms_mean": statistics.mean(ordered),
        "latency_ms_p90": _percentile(ordered, 0.90),
        "latency_ms_min": ordered[0],
        "latency_ms_max": ordered[-1],
        "n": len(ordered),
    }


def benchmark_serve(model, tok, msgs_list: list[list[dict]], *,
                    warmup: int = PERF_WARMUP, label: str = "serve") -> dict:
    """Median per-query latency after warmup; peak RSS / device mem over measure."""
    device = device_kind(model)
    if len(msgs_list) <= warmup:
        raise ValueError(f"need >{warmup} items to benchmark; got {len(msgs_list)}")

    for i, msgs in enumerate(msgs_list[:warmup]):
        timed_generate_one(model, tok, msgs)
        print(f"  [{label}/warmup] {i + 1}/{warmup}", flush=True)

    reset_device_peak(device)
    rss_peak = current_rss_mb()
    device_peak = device_allocated_mb(device)
    samples: list[float] = []
    new_tokens: list[int] = []
    measure = msgs_list[warmup:]
    for i, msgs in enumerate(measure):
        ms, n_new = timed_generate_one(model, tok, msgs)
        samples.append(ms)
        new_tokens.append(n_new)
        rss_peak = max(rss_peak, current_rss_mb())
        allocated = device_allocated_mb(device)
        if allocated is not None:
            device_peak = allocated if device_peak is None else max(device_peak, allocated)
        if (i + 1) % 4 == 0 or i + 1 == len(measure):
            print(f"  [{label}] {i + 1}/{len(measure)}  "
                  f"last={ms:.0f}ms  median_so_far={statistics.median(samples):.0f}ms",
                  flush=True)

    summary = summarize_latencies(samples)
    summary.update({
        "new_tokens_median": statistics.median(new_tokens),
        "new_tokens_mean": statistics.mean(new_tokens),
        "peak_rss_mb": rss_peak,
        "peak_device_mb": device_peak,
        "device": device,
        "warmup": warmup,
    })
    return summary


def timed_callable(fn: Callable[[], None], *, device: str, repeats: int = 3,
                   warmup: int = 1) -> dict:
    """Time an arbitrary side-effecting step (e.g. one SDFT update) with sync."""
    for _ in range(warmup):
        sync_device(device)
        fn()
        sync_device(device)
    samples: list[float] = []
    rss_peak = current_rss_mb()
    reset_device_peak(device)
    device_peak = device_allocated_mb(device)
    for _ in range(repeats):
        sync_device(device)
        t0 = time.perf_counter()
        fn()
        sync_device(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
        rss_peak = max(rss_peak, current_rss_mb())
        allocated = device_allocated_mb(device)
        if allocated is not None:
            device_peak = allocated if device_peak is None else max(device_peak, allocated)
    summary = summarize_latencies(samples)
    summary.update({"peak_rss_mb": rss_peak, "peak_device_mb": device_peak,
                    "device": device, "warmup": warmup})
    return summary


def heldout_msgs(evals: dict[int, list[dict]], demos_for) -> list[list[dict]]:
    """All held-out chats for one arm, flattened across regimes (fixed batch)."""
    msgs: list[list[dict]] = []
    for phase in (1, 2, 3):
        for item in evals[phase]:
            msgs.append(build_msgs(item, demos_for(item)))
    return msgs


def baseline_demos_fn(method: str | None, k: int, stream: list[dict],
                      upto: int = STREAM_LEN):
    """Causal end-of-week context builders for ZS / ICL / RAG serve timing."""
    retrieve = make_retriever(stream)

    def demos_for(item: dict):
        if method is None:
            return None
        if method == "ICL":
            return recent_demos(stream[:upto], k)
        return retrieve(item, k, upto=upto)

    return demos_for


def write_perf(payload: dict, path: Path = PERF_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def make_perf_figure(results: dict, perf: dict,
                     out: Path = PERF_FIG) -> Path:
    """Two-panel on-device cost figure: latency bars + latency vs live accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = results["arms"]
    colors = {"ZS": "#9aa0a6", "ICL": "#e8710a", "RAG": "#d93025",
              "Online-SDFT": "#1a73e8"}

    def arm_color(name: str) -> str:
        for prefix, color in colors.items():
            if name.startswith(prefix):
                return color
        return "#5f6368"

    # Prefer the showcase arm order from results; fall back to perf keys.
    names = [n for n in arms if n in perf.get("arms", {})]
    if not names:
        names = list(perf.get("arms", {}))

    fig, (ax_lat, ax_scatter) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    # --- left: median serve latency --- #
    xs = list(range(len(names)))
    latencies = [perf["arms"][n]["latency_ms_median"] for n in names]
    bar_colors = [arm_color(n) for n in names]
    ax_lat.bar(xs, latencies, color=bar_colors, edgecolor="white",
               linewidth=1.2, width=0.72, zorder=3)
    for x, name, ms in zip(xs, names, latencies):
        n_tok = perf["arms"][name].get("new_tokens_median")
        label = f"{ms:.0f} ms" + (f"\n(~{n_tok:.0f} tok)" if n_tok is not None else "")
        ax_lat.text(x, ms + max(latencies) * 0.02, label,
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=arm_color(name))
    ax_lat.set_xticks(xs)
    ax_lat.set_xticklabels(names, fontsize=10)
    ax_lat.set_ylabel("Median per-query serve latency  (ms)")
    ax_lat.set_title("D.  On-device serve latency", fontsize=12, fontweight="bold")
    ax_lat.set_ylim(0, max(latencies) * 1.28)
    ax_lat.grid(True, axis="y", alpha=0.25, zorder=0)

    update = perf.get("sdft_update")
    if update and "Online-SDFT" in names:
        note = (f"Online-SDFT update step (median): "
                f"{update['latency_ms_median']:.0f} ms "
                f"· peak RSS {perf['arms']['Online-SDFT']['peak_rss_mb']:.0f} MB")
        ax_lat.text(0.5, -0.18, note, transform=ax_lat.transAxes, ha="center",
                    fontsize=8.5, color=colors["Online-SDFT"], style="italic")

    # --- right: latency vs whole-week live accuracy (mirrors panel A) --- #
    for name in names:
        x = perf["arms"][name]["latency_ms_median"]
        y = arms[name]["acc_mean_live"] * 100
        ax_scatter.scatter(x, y, s=170, color=arm_color(name), zorder=3,
                           edgecolor="white", linewidth=1.5)
        dy = 3.2 if not name.startswith("RAG") else -10.0
        ax_scatter.annotate(name, (x, y), textcoords="offset points",
                            xytext=(8, dy), fontsize=10.5, fontweight="bold",
                            color=arm_color(name))

    # Memory footnotes: shared base dominates RSS; call out accelerator mem + LoRA.
    mem_bits = []
    for name in names:
        entry = perf["arms"][name]
        device_mb = entry.get("peak_device_mb")
        if device_mb is not None:
            mem_bits.append(f"{name} {device_mb:.0f} MB")
        else:
            mem_bits.append(f"{name} RSS {entry['peak_rss_mb']:.0f} MB")
    adapter_mb = results.get("adapter_bytes", perf.get("adapter_bytes", 0)) / 1e6
    mem_note = "Peak accelerator mem during timed serve  ·  " + "  ·  ".join(mem_bits)
    if adapter_mb:
        mem_note += f"  ·  LoRA on disk {adapter_mb:.1f} MB"
    ax_scatter.text(0.5, -0.20, mem_note,
                    transform=ax_scatter.transAxes, ha="center", fontsize=7.5,
                    color="#5f6368")

    ax_scatter.set_xlabel("Median per-query serve latency  (ms)")
    ax_scatter.set_ylabel("Whole-week accuracy: mean over regimes,\n"
                          "each scored while live  (%)")
    ax_scatter.set_title("E.  Latency vs whole-week accuracy",
                         fontsize=12, fontweight="bold")
    ax_scatter.set_ylim(0, 105)
    ax_scatter.grid(True, alpha=0.25)

    device = perf.get("device", "?")
    fig.suptitle(
        f"On-device cost · {device} · LFM2.5-230M"
        + (f" · {adapter_mb:.1f} MB LoRA" if adapter_mb else ""),
        fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, flush=True)

    # Standalone panel crops for the blog, matching A/B/C export.
    renderer = fig.canvas.get_renderer()
    for ax, suffix in ((ax_lat, "d"), (ax_scatter, "e")):
        extent = (ax.get_tightbbox(renderer)
                  .transformed(fig.dpi_scale_trans.inverted()))
        panel_out = out.with_name(f"{out.stem}_{suffix}{out.suffix}")
        fig.savefig(panel_out, dpi=150, bbox_inches=extent.expanded(1.02, 1.08))
        print("wrote", panel_out, flush=True)
    return out
