"""Baseline arms for the continual-triage demo: zero-shot, ICL, and RAG — online.

Every baseline is *causal*, exactly like the SDFT learner: when the stream is
at position t, ICL's context window and RAG's decision store hold only the t
decisions observed so far — nothing from the future. None of them update any
weights:

  ZS       bare prompt, base priors
  ICL k    the k most recent observed decisions pinned in context (token tax)
  RAG k    k nearest neighbours retrieved from the decisions observed so far

Three views are computed, matching the blog figure's three panels:

  arms + k sweep   end-of-week snapshot: mean accuracy across all three regime
                   policies vs the per-query prompt-token bill        (Panel A)
  curves           accuracy on the *current* regime's held-out policy at each
                   checkpoint, with only the history available then; position
                   0 (nothing streamed) is the zero-shot value        (Panel B)
  regret           cumulative mistakes on the streamed items themselves, each
                   predicted BEFORE its label is revealed             (Panel C)

Writes outputs/baselines.json. Run before run_sdft.py,
which adds the Online-SDFT arm and draws the figure.

Run:  python run_baselines.py
"""

from __future__ import annotations

import json
import random

from triage_common import (
    BASELINES_JSON, DATA_OUT, DRIFTS, EVAL_N, MODEL_NAME, OUT_DIR, REGIMES, SEED,
    STREAM_LEN, accuracy, build_eval, build_msgs, build_stream, export_dataset,
    generate, load_base_model, load_tokenizer, make_retriever, parse_action,
    phase_of, pick_device, prompt_tokens, recent_demos, render_prompt,
)

# --- baseline knobs --------------------------------------------------------- #
K_SWEEP = (3, 6, 9, 12)   # context sizes tried for BOTH baselines
CHECKPOINTS = tuple(range(6, STREAM_LEN + 1, 6))   # the grid run_sdft.py probes too


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}  model={MODEL_NAME}", flush=True)

    # The same seeded drifting stream and held-out sets run_sdft.py uses.
    stream = build_stream(random.Random(SEED))
    evals = {phase: build_eval(random.Random(SEED + phase), phase) for phase in (1, 2, 3)}

    export_dataset(stream, evals)   # the committed copy the Colab notebook fetches
    print(f"wrote dataset -> {DATA_OUT} ({len(stream)} stream + 3x{EVAL_N} eval items; "
          f"regimes {REGIMES}; drifts@{DRIFTS})", flush=True)

    tok = load_tokenizer()
    base = load_base_model(device)
    retrieve = make_retriever(stream)   # causal via upto=: store[:t] only

    def demos_for(method: str, item: dict, k: int, upto: int) -> list[tuple[dict, str]]:
        """The context a frozen baseline gets at stream position `upto`."""
        if method == "ICL":
            return recent_demos(stream[:upto], k)
        return retrieve(item, k, upto=upto)

    def eval_arm(method: str | None, k: int, upto: int, label: str) -> dict:
        """Accuracy per regime policy (mean across them) + the mean prompt-token
        bill, with only the history available at position `upto` in context."""
        accs, token_counts = {}, []
        for phase, regime in zip((1, 2, 3), REGIMES):
            msgs = [build_msgs(item,
                               None if method is None else demos_for(method, item, k, upto))
                    for item in evals[phase]]
            accs[regime] = accuracy(evals[phase],
                                    generate(base, tok, msgs, label=f"{label}/{regime}"))
            token_counts += [prompt_tokens(tok, m) for m in msgs]
        return {"acc_by_regime": accs, "acc_mean": sum(accs.values()) / len(accs),
                "tok_per_query": sum(token_counts) / len(token_counts)}

    # -- end-of-week arms + the k sweep (Panel A) ------------------------------ #
    print("\n== k sweep at end of week: mean accuracy across the three policies ==",
          flush=True)
    sweeps: dict[str, dict] = {"ICL": {}, "RAG": {}}
    for method in ("ICL", "RAG"):
        for k in K_SWEEP:
            sweeps[method][k] = eval_arm(method, k, STREAM_LEN, f"{method.lower()} k={k}")
            entry = sweeps[method][k]
            regime_report = "  ".join(f"{regime}={entry['acc_by_regime'][regime]:.2f}"
                                      for regime in REGIMES)
            print(f"  {method} k={k:2d}: mean={entry['acc_mean']:.2f}  ({regime_report})  "
                  f"tok/query={entry['tok_per_query']:.0f}", flush=True)
    icl_k = max(K_SWEEP, key=lambda k: (sweeps["ICL"][k]["acc_mean"], -k))  # ties -> cheaper
    rag_k = max(K_SWEEP, key=lambda k: (sweeps["RAG"][k]["acc_mean"], -k))
    print(f"  best: ICL k={icl_k}, RAG k={rag_k}", flush=True)

    print("\n== zero-shot arm ==", flush=True)
    zs_arm = eval_arm(None, 0, 0, "zs")

    # -- checkpoint curves on the current regime's policy (Panel B) ------------ #
    # At position 0 nothing has streamed: every method IS zero-shot, so all
    # curves share the zero-shot value on the weekday (first-regime) eval.
    print("\n== checkpoint curves: current-regime accuracy with causal history ==",
          flush=True)
    zs_start = zs_arm["acc_by_regime"][REGIMES[0]]
    curves: dict[str, list] = {"pos": [0, *CHECKPOINTS],
                               "ICL": [zs_start], "RAG": [zs_start],
                               "zs_by_phase": zs_arm["acc_by_regime"]}
    for pos in CHECKPOINTS:
        phase = phase_of(pos)
        for method, k in (("ICL", icl_k), ("RAG", rag_k)):
            msgs = [build_msgs(item, demos_for(method, item, k, pos))
                    for item in evals[phase]]
            curves[method].append(accuracy(
                evals[phase], generate(base, tok, msgs, label=f"{method.lower()}@{pos}")))
        print(f"  pos {pos:2d} ({REGIMES[phase - 1]}): ICL={curves['ICL'][-1]:.2f}  "
              f"RAG={curves['RAG'][-1]:.2f}", flush=True)

    # -- accumulated regret on the stream itself (Panel C) --------------------- #
    # Prequential: predict item t with the history 1..t-1, THEN reveal the label.
    # k is swept HERE too — Panel C shows each baseline at its LEAST-REGRET k,
    # which may differ from its whole-week-accuracy k (most flattering per metric).
    print("\n== accumulated regret: predict each streamed item before its label ==",
          flush=True)

    def regret_curve(method: str, k: int, label: str) -> list[int]:
        msgs = [build_msgs(item,
                           None if method == "ZS" else demos_for(method, item, k, i))
                for i, item in enumerate(stream)]
        predictions = [parse_action(reply) for reply in generate(base, tok, msgs, label=label)]
        cumulative = [0]
        for prediction, item in zip(predictions, stream):
            cumulative.append(cumulative[-1] + int(prediction != item["action"]))
        return cumulative

    regret: dict[str, list] = {"pos": list(range(STREAM_LEN + 1)),
                               "ZS": regret_curve("ZS", 0, "regret/zs")}
    print(f"  ZS: {regret['ZS'][-1]}/{STREAM_LEN} mistakes", flush=True)
    regret_sweep: dict[str, dict[int, int]] = {}
    regret_k: dict[str, int] = {}
    for method in ("ICL", "RAG"):
        curves_by_k = {k: regret_curve(method, k, f"regret/{method.lower()} k={k}")
                       for k in K_SWEEP}
        regret_sweep[method] = {k: curve[-1] for k, curve in curves_by_k.items()}
        regret_k[method] = min(K_SWEEP, key=lambda k: (regret_sweep[method][k], k))
        regret[method] = curves_by_k[regret_k[method]]
        table = "  ".join(f"k={k}:{final}" for k, final in regret_sweep[method].items())
        print(f"  {method}: {table}  -> least-regret k={regret_k[method]} "
              f"({regret[method][-1]}/{STREAM_LEN} mistakes)", flush=True)

    # The qualitative drifted items — off-hours `social` pushes that should now
    # INTERRUPT. Capture every candidate's baseline replies (end-of-week
    # history); run_sdft.py picks the one the served adapter gets right.
    social_items = [item for item in evals[3] if item["category"] == "social"]
    qualitative = [{
        "item": item,
        "prompt": render_prompt(item),
        "gold": item["action"],
        "zs": generate(base, tok, [build_msgs(item)], label="q/zs", batch_size=1)[0],
        "icl": generate(base, tok,
                        [build_msgs(item, demos_for("ICL", item, icl_k, STREAM_LEN))],
                        label="q/icl", batch_size=1)[0],
        "rag": generate(base, tok,
                        [build_msgs(item, demos_for("RAG", item, rag_k, STREAM_LEN))],
                        label="q/rag", batch_size=1)[0],
    } for item in social_items]

    baselines = {
        "config": {"model": MODEL_NAME, "seed": SEED, "stream_len": STREAM_LEN,
                   "drifts": list(DRIFTS), "regimes": list(REGIMES), "eval_n": EVAL_N,
                   "k_sweep": list(K_SWEEP), "icl_k": icl_k, "rag_k": rag_k,
                   "icl_k_regret": regret_k["ICL"], "rag_k_regret": regret_k["RAG"],
                   "checkpoints": list(CHECKPOINTS)},
        "sweeps": sweeps,
        "regret_sweep": regret_sweep,
        "arms": {
            "ZS": {**zs_arm, "labels_needed": 0},
            f"ICL k={icl_k}": {**sweeps["ICL"][icl_k], "labels_needed": icl_k},
            f"RAG k={rag_k}": {**sweeps["RAG"][rag_k], "labels_needed": STREAM_LEN},
        },
        "curves": curves,
        "regret": regret,
        "qualitative_base": qualitative,
    }
    BASELINES_JSON.write_text(json.dumps(baselines, indent=2))
    print(f"\nwrote {BASELINES_JSON}", flush=True)
    for name, arm in baselines["arms"].items():
        regime_report = "  ".join(f"{regime}={arm['acc_by_regime'][regime]:.2f}"
                                  for regime in REGIMES)
        print(f"  {name:10s} mean={arm['acc_mean']:.2f}  ({regime_report})  "
              f"tok/query={arm['tok_per_query']:.0f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
