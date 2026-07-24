"""Measure on-device serve latency / memory and draw the perf figure.

Uses the already-trained adapter in outputs/adapter-online-sft and the
accuracy numbers in outputs/results.json — no full SFT retrain required.

Run after a normal `python run.py` (or run_baselines + run_sft) once:

  python run_perf.py

Full pipelines also record these numbers themselves: run_baselines.py writes
baseline-arm timing into outputs/perf.json; run_sft.py merges Online-SFT
serve + update-step timing and draws figures/online_sft_perf.png.
"""

from __future__ import annotations

import json
import random

from peft import PeftModel

from triage_common import (
    ACTIONS, BASELINES_JSON, EVAL_N, MODEL_NAME, OUT_DIR, SEED,
    build_eval, build_stream, build_teacher_msgs, load_base_model,
    load_tokenizer, pick_device, recent_demos, render_prompt,
)
from triage_perf import (
    PERF_WARMUP, baseline_demos_fn, benchmark_serve, heldout_msgs,
    make_perf_figure, timed_callable, write_perf,
)
from run_sft import (ADAPTER_DIR, DISTILL_BETA, DISTILL_T, LR, STEPS_PER_ITEM,
                      TEACHER_SHOTS, make_updater)

RESULTS_JSON = OUT_DIR / "results.json"


def main() -> None:
    if not RESULTS_JSON.exists():
        raise SystemExit(f"{RESULTS_JSON} not found — run `python run.py` first.")
    if not ADAPTER_DIR.exists():
        raise SystemExit(f"{ADAPTER_DIR} not found — run `python run_sft.py` first.")

    results = json.loads(RESULTS_JSON.read_text())
    baselines = (json.loads(BASELINES_JSON.read_text())
                 if BASELINES_JSON.exists() else {})
    config = results["config"]
    icl_k, rag_k = config["icl_k"], config["rag_k"]

    device = pick_device()
    print(f"device={device}  model={MODEL_NAME}", flush=True)
    stream = build_stream(random.Random(SEED))
    evals = {phase: build_eval(random.Random(SEED + phase), phase)
             for phase in (1, 2, 3)}

    tok = load_tokenizer()
    base = load_base_model(device)

    print("\n== baseline serve latency / memory ==", flush=True)
    zs_name, icl_name, rag_name = "ZS", f"ICL k={icl_k}", f"RAG k={rag_k}"
    perf_arms = {
        zs_name: benchmark_serve(
            base, tok, heldout_msgs(evals, baseline_demos_fn(None, 0, stream)),
            warmup=PERF_WARMUP, label="perf/zs"),
        icl_name: benchmark_serve(
            base, tok, heldout_msgs(evals, baseline_demos_fn("ICL", icl_k, stream)),
            warmup=PERF_WARMUP, label="perf/icl"),
        rag_name: benchmark_serve(
            base, tok, heldout_msgs(evals, baseline_demos_fn("RAG", rag_k, stream)),
            warmup=PERF_WARMUP, label="perf/rag"),
    }

    print("\n== Online-SFT serve latency / memory ==", flush=True)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=True)
    sft_serve = benchmark_serve(
        model, tok, heldout_msgs(evals, lambda _item: None),
        warmup=PERF_WARMUP, label="perf/sft")
    perf_arms["Online-SFT"] = sft_serve

    # One update step on a synthetic mini-batch matching the live loop shape.
    print("\n== SFT update-step latency ==", flush=True)
    item = stream[-1]
    expert = item["action"]
    history = recent_demos(stream[:-1], TEACHER_SHOTS) if TEACHER_SHOTS else None
    row = {"prompt": render_prompt(item),
           "teacher_msgs": build_teacher_msgs(item, expert, history),
           "action": expert}
    update_batch = [row]
    for action in ACTIONS:
        if action == expert:
            continue
        prior = next((s for s in reversed(stream[:-1]) if s["action"] == action), None)
        if prior is None:
            continue
        update_batch.append({
            "prompt": render_prompt(prior),
            "teacher_msgs": build_teacher_msgs(prior, prior["action"], None),
            "action": prior["action"],
        })
    update = make_updater(model, tok, lr=LR,
                          distill_t=DISTILL_T, distill_beta=DISTILL_BETA)
    sft_update = timed_callable(
        lambda: update(update_batch, STEPS_PER_ITEM),
        device=device, repeats=3, warmup=1)

    for name, entry in perf_arms.items():
        device_mb = entry.get("peak_device_mb")
        device_bit = (f"  device={device_mb:.0f}MB" if device_mb is not None else "")
        print(f"  {name:14s} median={entry['latency_ms_median']:.0f}ms  "
              f"p90={entry['latency_ms_p90']:.0f}ms  "
              f"new_tok~{entry['new_tokens_median']:.0f}  "
              f"RSS={entry['peak_rss_mb']:.0f}MB{device_bit}", flush=True)
    print(f"  update step     median={sft_update['latency_ms_median']:.0f}ms  "
          f"(steps_per_item={STEPS_PER_ITEM}, batch={len(update_batch)})",
          flush=True)

    adapter_bytes = results.get("adapter_bytes") or (
        ADAPTER_DIR / "adapter_model.safetensors").stat().st_size
    perf = {
        "device": device,
        "warmup": PERF_WARMUP,
        "n_queries": EVAL_N * 3 - PERF_WARMUP,
        "arms": perf_arms,
        "sft_update": {**sft_update,
                        "steps_per_item": STEPS_PER_ITEM,
                        "batch_size": len(update_batch)},
        "adapter_bytes": adapter_bytes,
    }
    write_perf(perf)

    results["perf"] = perf
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS_JSON}", flush=True)

    if baselines:
        baselines["perf"] = {
            "device": device,
            "warmup": PERF_WARMUP,
            "n_queries": EVAL_N * 3 - PERF_WARMUP,
            "arms": {k: v for k, v in perf_arms.items() if k != "Online-SFT"},
        }
        BASELINES_JSON.write_text(json.dumps(baselines, indent=2))
        print(f"wrote {BASELINES_JSON}", flush=True)

    make_perf_figure(results, perf)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
