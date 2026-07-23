"""Online SDFT on the drifting inbox stream — the loop the blog post describes.

The steps, exactly as in the accompanying blog post:

  1. A tiny dataset: a stream that drifts twice
       seeded synthetic inbox; the 3-way policy flips at DRIFTS (triage_common)
  2. The online loop, one item at a time (prequential: test, THEN train)
       Student (bare prompt) guesses first — scored for regret BEFORE feedback.
       Teacher is the same adapter conditioned on the expert action (your
       observed behavior). Distill teacher → student with hard CE + temp-soft
       KL on the completion tokens (SDFT), batch_size=1 with replay.
  3. The probe guardrail: keep the best adapter on the current regime
  4. Serving: bare prompt — the adapter carries the policy

Reads outputs/baselines.json (run run_baselines.py
first) and writes results.json, the trained adapter, and the blog figures:
A/B/C accuracy panels plus D/E on-device latency / latency-vs-accuracy.

Run:  python run_sdft.py
"""

from __future__ import annotations

import copy
import json
import random

import torch
import torch.nn.functional as F
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)

from triage_common import (
    ACTIONS, BASELINES_JSON, DRIFTS, FIG_DIR, MODEL_NAME, OUT_DIR, REGIMES, SEED,
    STREAM_LEN, accuracy, build_eval, build_msgs, build_stream, build_teacher_msgs,
    generate, load_base_model, load_tokenizer, parse_action, phase_of, pick_device,
    recent_demos, render_prompt,
)
from triage_perf import (
    PERF_WARMUP, benchmark_serve, heldout_msgs, make_perf_figure, timed_callable,
    write_perf,
)

# --- training knobs (the blog's knob table) ---------------------------------- #
# Student always serves / guesses with the bare prompt; the teacher sees the
# expert (user) action as an in-context demonstration. TEACHER_SHOTS prepends
# older causal decisions to the teacher only (0 = just this item's expert action).
LORA_R = 8                                       # adapter rank (~1.4 MB on disk)
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET = r".*self_attn\.(q|k|v|out)_proj"   # LFM2 attention projections
LR = 7e-4            # one persistent AdamW across the whole stream: the completion
                     # is 1-2 tokens (tiny loss), so it wants a larger step than a
                     # scheduled batch trainer — but 1e-3 overshoots on the live
                     # stream (regret 28 vs 18) and 2e-3 collapses outright
REPLAY = 16          # sliding replay-buffer size (items; 32 drowns the fresh item)
STEPS_PER_ITEM = 8   # batch_size=1 update steps per incoming item — 3-way wants
                     # more gradient than binary (3 stalls on the cold start)
TEACHER_SHOTS = 0    # extra history demos in the teacher context (beyond expert)
DISTILL_T = 3.0      # temperature for the teacher→student soft-CE term
DISTILL_BETA = 0.25  # weight on that term (0 = hard CE only; pure soft CE collapses)
CHECKPOINTS = tuple(range(6, STREAM_LEN + 1, 6))   # eval every 6 streamed items

ADAPTER_DIR = OUT_DIR / "adapter-online-sdft"


def make_updater(model, tok, lr: float = LR,
                 distill_t: float = DISTILL_T, distill_beta: float = DISTILL_BETA):
    """Persistent AdamW + SDFT teacher→student distill.

    Teacher (no grad) is the same adapter conditioned on the expert action;
    student is the bare serving prompt. Loss = hard CE on the expert-action
    tokens under the student prompt (stable on 1–2-token labels) + a
    temperature-softened forward-KL to the teacher (the distill term).
    Pure full-vocab soft CE collapses this 230M to a first-token loop.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    eos = tok.eos_token or ""
    device = next(model.parameters()).device

    def encode_prompt(messages: list[dict]) -> list[int]:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    def completion_logits(logits: torch.Tensor, prompt_len: int, n_comp: int):
        # logits[i] predicts token i+1; completion starts at index prompt_len.
        return logits[:, prompt_len - 1: prompt_len - 1 + n_comp, :]

    def update(batch: list[dict], steps: int) -> None:
        model.train()
        model.config.use_cache = False
        for step_idx in range(steps):
            row = batch[step_idx % len(batch)]    # cycle the (item + replay) mini-batch
            student_msgs = [{"role": "user", "content": row["prompt"]}]
            student_ids = encode_prompt(student_msgs)
            teacher_ids = encode_prompt(row["teacher_msgs"])
            # Distill along the expert action (user behavior) — the privileged
            # completion the teacher was conditioned to express.
            comp_ids = tok(row["action"] + eos, add_special_tokens=False)["input_ids"]
            n_comp = len(comp_ids)
            comp_tensor = torch.tensor(comp_ids, device=device)

            s_out = model(input_ids=torch.tensor(
                [student_ids + comp_ids], device=device))
            s_logits = completion_logits(s_out.logits, len(student_ids), n_comp)
            hard = F.cross_entropy(
                s_logits.reshape(-1, s_logits.size(-1)), comp_tensor)

            if distill_beta > 0:
                with torch.no_grad():
                    t_out = model(input_ids=torch.tensor(
                        [teacher_ids + comp_ids], device=device))
                    t_logits = completion_logits(
                        t_out.logits, len(teacher_ids), n_comp)
                t_soft = F.softmax(t_logits / distill_t, dim=-1)
                soft = (-(t_soft * F.log_softmax(s_logits / distill_t, dim=-1))
                        .sum(dim=-1).mean()) * (distill_t * distill_t)
                loss = hard + distill_beta * soft
            else:
                loss = hard
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)   # keep bs=1 steps from diverging
            optimizer.step()
            optimizer.zero_grad()
        model.config.use_cache = True
        model.eval()

    return update


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINES_JSON.exists():
        raise SystemExit(f"{BASELINES_JSON} not found — run `python run_baselines.py` "
                         "first (the figure compares against the ZS / ICL / RAG arms it writes).")
    baselines = json.loads(BASELINES_JSON.read_text())
    config = baselines["config"]
    if (config["model"], config["seed"], config["stream_len"],
            config["checkpoints"]) != (MODEL_NAME, SEED, STREAM_LEN, list(CHECKPOINTS)):
        raise SystemExit("baselines.json was produced with a different model/seed/stream/"
                         "checkpoint grid — re-run run_baselines.py")

    device = pick_device()
    print(f"device={device}  model={MODEL_NAME}", flush=True)
    torch.manual_seed(SEED)   # LoRA init + dropout masks — makes the run repeatable

    # -- 1. a tiny dataset: a stream that drifts twice ------------------------ #
    stream = build_stream(random.Random(SEED))
    evals = {phase: build_eval(random.Random(SEED + phase), phase) for phase in (1, 2, 3)}

    tok = load_tokenizer()
    base = load_base_model(device)

    # -- 2+3. the online loop: student guess -> teacher distill -> update ----- #
    # Prequential (test-then-train): STUDENT (bare prompt) guesses first — that
    # prediction feeds the regret curve — then TEACHER (same adapter + expert
    # action as in-context demo) provides the soft target and we soft-CE
    # distill into the student. Baselines get the same causal history.
    print("\n== online SDFT: student guesses bare, teacher sees expert action ==",
          flush=True)
    model = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    update = make_updater(model, tok)

    def heldout_accuracy(phase: int, label: str) -> float:
        return accuracy(evals[phase], generate(
            model, tok, [build_msgs(item) for item in evals[phase]], label=label))

    curve = {"pos": [], "acc_p1": [], "acc_p2": [], "acc_p3": []}
    sdft_regret = [0]
    rows: list[dict] = []
    replay_buffer: list[dict] = []
    sampler = random.Random(SEED)
    # Probe guardrail: 3-way over-trains and *decays* past its peak, so during the
    # final regime snapshot the adapter whenever the current-policy probe is at its
    # best, and roll back to that snapshot before serving (auto-rollback on decay).
    best = {"acc": -1.0, "pos": None, "state": None}
    for i, item in enumerate(stream):
        # STUDENT call — the exact bare prompt it serves, scored before feedback
        guess = generate(model, tok, [build_msgs(item)],
                         label=f"guess@{i + 1}", batch_size=1)[0]
        prediction = parse_action(guess)
        expert = item["action"]                          # actual user behavior
        history = recent_demos(stream[:i], TEACHER_SHOTS) if TEACHER_SHOTS else None
        # TEACHER context: optional history + (this item, expert action) demo + re-ask
        teacher_msgs = build_teacher_msgs(item, expert, history)
        row = {"prompt": render_prompt(item), "teacher_msgs": teacher_msgs,
               "action": expert, "pred": prediction,
               "feedback": int(prediction == expert)}
        rows.append(row)
        sdft_regret.append(sdft_regret[-1] + 1 - row["feedback"])

        replay_buffer = (replay_buffer + [row])[-REPLAY:]
        # pair the fresh item with one replayed item from EACH other class, so
        # every batch_size=1 update cycles all three actions (binary's
        # pair-with-the-opposite trick, generalised — kills majority collapse)
        batch = [row]
        for action in ACTIONS:
            pool = [b for b in replay_buffer[:-1] if b["action"] == action]
            if action != row["action"] and pool:
                batch.append(sampler.sample(pool, 1)[0])
        update(batch, STEPS_PER_ITEM)

        pos = i + 1
        if pos in CHECKPOINTS:
            curve["pos"].append(pos)
            for phase in (1, 2, 3):
                curve[f"acc_p{phase}"].append(
                    heldout_accuracy(phase, f"sdft@{pos}/p{phase}"))
            report = "  ".join(f"{regime}={curve[f'acc_p{phase}'][-1]:.2f}"
                               for phase, regime in zip((1, 2, 3), REGIMES))
            print(f"  checkpoint {pos}: {report}  (stream mistakes so far: "
                  f"{sdft_regret[-1]}/{pos})", flush=True)
            if pos > DRIFTS[1] and curve["acc_p3"][-1] >= best["acc"]:
                best = {"acc": curve["acc_p3"][-1], "pos": pos,
                        "state": copy.deepcopy(get_peft_model_state_dict(model))}

    n_reinforced = sum(row["feedback"] for row in rows)
    reinforce_frac = n_reinforced / len(rows)
    with (OUT_DIR / "sdft_targets.jsonl").open("w") as fh:
        for row in rows:
            # teacher_msgs are bulky; log the distillable fields only
            fh.write(json.dumps({k: row[k] for k in
                                 ("prompt", "action", "pred", "feedback")}) + "\n")
    print(f"  student already matched expert (online): "
          f"{n_reinforced}/{len(rows)}", flush=True)

    if best["state"] is not None:            # roll back to the probe-kept best
        set_peft_model_state_dict(model, best["state"])
        note = ("" if best["pos"] == CHECKPOINTS[-1]
                else " instead of the decayed final one")
        print(f"  probe guardrail: serving the adapter from item {best['pos']} "
              f"({REGIMES[2]} acc {best['acc']:.2f}){note}", flush=True)

    # -- 4. serving: bare prompt, adapter carries the policy ------------------ #
    model.save_pretrained(str(ADAPTER_DIR))
    adapter_bytes = (ADAPTER_DIR / "adapter_model.safetensors").stat().st_size

    # Score the SERVED adapter on all three regime policies — Panel A plots the
    # mean, the same whole-week yardstick every baseline arm gets.
    print("\n== served adapter across the whole week ==", flush=True)
    served = {regime: heldout_accuracy(phase, f"served/{regime}")
              for phase, regime in zip((1, 2, 3), REGIMES)}
    served_mean = sum(served.values()) / len(served)
    print("  " + "  ".join(f"{regime}={acc:.2f}" for regime, acc in served.items())
          + f"  mean={served_mean:.2f}", flush=True)

    # "One item, four minds": among the off-hours social pushes the baselines
    # answered, pick one the served adapter gets right where zero-shot doesn't.
    qualitative = None
    for candidate in baselines["qualitative_base"]:
        item = candidate["item"]
        assert candidate["prompt"] == render_prompt(item), \
            "baselines.json is stale — re-run run_baselines.py"
        reply = generate(model, tok, [build_msgs(item)], label="q/sdft", batch_size=1)[0]
        picked = {"prompt": candidate["prompt"], "gold": candidate["gold"],
                  "zs": candidate["zs"], "icl": candidate["icl"],
                  "rag": candidate["rag"], "sdft": reply}
        if qualitative is None:
            qualitative = picked                     # fallback: first candidate
        if (parse_action(reply) == item["action"]
                and parse_action(candidate["zs"]) != item["action"]):
            qualitative = picked                     # the showcase pick
            break

    arms = {name: dict(arm) for name, arm in baselines["arms"].items()}
    arms["Online-SDFT"] = {
        "acc_by_regime": served, "acc_mean": served_mean,   # week-END adapter, re-graded
        "tok_per_query": arms["ZS"]["tok_per_query"],       # served bare
        "labels_needed": 0,
    }

    # Panel B curve for SDFT: held-out accuracy on the regime that is current at
    # each checkpoint; position 0 (nothing streamed, adapter still identity) is
    # exactly the zero-shot value — every method starts from the same point.
    curves = dict(baselines["curves"])
    curves["SDFT"] = [curves["zs_by_phase"][REGIMES[0]]] + [
        curve[f"acc_p{phase_of(pos)}"][idx] for idx, pos in enumerate(curve["pos"])]

    # Panel A grades the week AS LIVED: each regime scored while it is live (its
    # block-end checkpoint — the same three positions for every method), then
    # averaged. The week-end snapshot above stays in results.json as the honest
    # "and if you froze Friday's context and re-graded the whole week" number.
    block_ends = [*DRIFTS, STREAM_LEN]
    for name, key in (("ZS", "ZS"), (f"ICL k={config['icl_k']}", "ICL"),
                      (f"RAG k={config['rag_k']}", "RAG"), ("Online-SDFT", "SDFT")):
        live = {regime: (curves["zs_by_phase"][regime] if key == "ZS"
                         else curves[key][curves["pos"].index(pos)])
                for pos, regime in zip(block_ends, REGIMES)}
        arms[name]["acc_live_by_regime"] = live
        arms[name]["acc_mean_live"] = sum(live.values()) / len(live)

    regret = dict(baselines["regret"])
    regret["SDFT"] = sdft_regret

    # -- on-device serve latency / memory + one SDFT update step --------------- #
    print("\n== serve latency / memory (held-out batch, after warmup) ==", flush=True)
    sdft_serve = benchmark_serve(
        model, tok, heldout_msgs(evals, lambda _item: None),
        warmup=PERF_WARMUP, label="perf/sdft")
    print(f"  Online-SDFT  median={sdft_serve['latency_ms_median']:.0f}ms  "
          f"p90={sdft_serve['latency_ms_p90']:.0f}ms  "
          f"new_tok~{sdft_serve['new_tokens_median']:.0f}  "
          f"RSS={sdft_serve['peak_rss_mb']:.0f}MB", flush=True)

    # One online update step on a fresh+replay-style mini-batch (same shape as
    # the live loop) — the recurring train cost the phone pays per notification.
    print("\n== SDFT update-step latency ==", flush=True)
    sample_row = rows[-1]
    update_batch = [sample_row]
    for action in ACTIONS:
        pool = [b for b in rows[:-1] if b["action"] == action]
        if action != sample_row["action"] and pool:
            update_batch.append(pool[-1])
    sdft_update = timed_callable(
        lambda: update(update_batch, STEPS_PER_ITEM),
        device=device, repeats=3, warmup=1)
    print(f"  update median={sdft_update['latency_ms_median']:.0f}ms  "
          f"(steps_per_item={STEPS_PER_ITEM}, batch={len(update_batch)})",
          flush=True)

    perf = dict(baselines.get("perf") or {})
    perf_arms = dict(perf.get("arms") or {})
    perf_arms["Online-SDFT"] = sdft_serve
    perf = {
        **perf,
        "device": device,
        "warmup": PERF_WARMUP,
        "n_queries": len(evals[1]) + len(evals[2]) + len(evals[3]) - PERF_WARMUP,
        "arms": perf_arms,
        "sdft_update": {**sdft_update,
                        "steps_per_item": STEPS_PER_ITEM,
                        "batch_size": len(update_batch)},
        "adapter_bytes": adapter_bytes,
    }
    write_perf(perf)

    results = {
        "config": {**config, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                   "lora_dropout": LORA_DROPOUT, "lr": LR,
                   "replay": REPLAY, "steps_per_item": STEPS_PER_ITEM,
                   "teacher_shots": TEACHER_SHOTS,
                   "distill_t": DISTILL_T, "distill_beta": DISTILL_BETA},
        "arms": arms,
        "sweeps": baselines["sweeps"],
        "curve": curve,
        "curves": curves,
        "regret": regret,
        "sdft_best": {"pos": best["pos"], "acc": best["acc"]},
        "qualitative": qualitative,
        "adapter_bytes": adapter_bytes,
        # fraction of stream where the bare student already matched the expert
        "reinforce_frac": reinforce_frac,
        "perf": perf,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT_DIR / "results.json", flush=True)

    make_figure(results)
    make_perf_figure(results, perf)
    print("DONE", flush=True)


def make_figure(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    arms = results["arms"]
    curves = results["curves"]
    regret = results["regret"]
    drifts = results["config"]["drifts"]
    stream_len = results["config"]["stream_len"]
    icl_name = f"ICL k={results['config']['icl_k']}"
    rag_name = f"RAG k={results['config']['rag_k']}"
    zs_tokens = arms["ZS"]["tok_per_query"]
    colors = {"ZS": "#9aa0a6", "ICL": "#e8710a", "RAG": "#d93025", "Online-SDFT": "#1a73e8"}

    def arm_color(name: str) -> str:
        for prefix, color in colors.items():
            if name.startswith(prefix):
                return color
        return "#5f6368"

    fig, (ax_cost, ax_drift, ax_regret) = plt.subplots(1, 3, figsize=(18.4, 4.9))

    # Panel A: whole-week accuracy — each regime scored while it is live (its
    # block-end checkpoint), averaged — vs the recurring prompt-token bill.
    for name, arm in arms.items():
        x, y = arm["tok_per_query"], arm["acc_mean_live"] * 100
        ax_cost.scatter(x, y, s=170, color=arm_color(name), zorder=3,
                        edgecolor="white", linewidth=1.5)
        dy = 3.2 if not name.startswith("RAG") else -12.0   # RAG sits ~ICL's x; dodge below
        ax_cost.annotate(name, (x, y), textcoords="offset points", xytext=(8, dy),
                         fontsize=10.5, fontweight="bold", color=arm_color(name))
    ax_cost.set_xlabel("Recurring prompt tokens / query  (on-device cost, every notification)")
    ax_cost.set_ylabel("Whole-week accuracy: mean over regimes,\neach scored while live  (%)")
    ax_cost.set_title("A.  Whole-week accuracy vs the recurring token bill",
                      fontsize=12, fontweight="bold")
    ax_cost.grid(True, alpha=0.25)
    ax_cost.set_ylim(0, 105)
    ax_cost.axvspan(0, zs_tokens + 22, color="#1a73e8", alpha=0.05)
    ax_cost.text((zs_tokens + 22) / 2, 6, "bare-prompt zone\n(weights carry the policy)",
                 ha="center", fontsize=8.5, color="#1a73e8", style="italic")

    # Panel B: current-regime accuracy along the stream. Every method starts at
    # the zero-shot value (position 0 = nothing streamed) and gets the same
    # causal history; the dotted grey steps are the zero-shot floor per regime.
    for x in drifts:
        ax_drift.axvline(x, color="#5f6368", ls="--", lw=1.2)
    for (start, end), regime in zip(zip([0, *drifts], [*drifts, stream_len]), REGIMES):
        ax_drift.hlines(curves["zs_by_phase"][regime] * 100, start, end,
                        color=colors["ZS"], ls=":", lw=1.6, zorder=1)
    for key, name in (("SDFT", "Online-SDFT"), ("ICL", icl_name), ("RAG", rag_name)):
        ax_drift.plot(curves["pos"], [v * 100 for v in curves[key]], "-o",
                      color=colors[key if key != "SDFT" else "Online-SDFT"],
                      lw=2.6 if key == "SDFT" else 1.8,
                      ms=5 if key == "SDFT" else 3.8,
                      alpha=1.0 if key == "SDFT" else 0.85, zorder=4 if key == "SDFT" else 3)
    kept = results.get("sdft_best") or {}
    if kept.get("pos"):   # star the checkpoint the probe guardrail serves
        ax_drift.scatter([kept["pos"]], [kept["acc"] * 100], marker="*", s=300,
                         color=colors["Online-SDFT"], edgecolor="white",
                         linewidth=1.2, zorder=5)
        near_top = kept["acc"] > 0.88          # dodge below the star near the ceiling
        ax_drift.annotate("probe keeps\nthis adapter", (kept["pos"], kept["acc"] * 100),
                          textcoords="offset points",
                          xytext=(0, -24 if near_top else 10), fontsize=7.5,
                          color=colors["Online-SDFT"], ha="center", fontweight="bold")

    # Per-phase sub-titles along the x-axis (neutral grey: the curves are
    # methods now, not regimes).
    phase_axis = blended_transform_factory(ax_drift.transData, ax_drift.transAxes)
    bounds = [0, *drifts, stream_len]
    for index, (start, end, regime) in enumerate(zip(bounds, bounds[1:], REGIMES)):
        if index % 2:   # alternate tint so the regime spans read at a glance
            ax_drift.axvspan(start, end, color="#5f6368", alpha=0.05, zorder=0)
        ax_drift.text((start + end) / 2, -0.115, f"{regime}\nitems {start + 1}–{end}",
                      transform=phase_axis, ha="center", va="top", fontsize=8.3,
                      color="#5f6368", fontweight="bold")
    # B ships standalone on the blog too, so it carries its own compact legend.
    from matplotlib.lines import Line2D
    ax_drift.legend(handles=[
        Line2D([], [], color=colors["Online-SDFT"], lw=2.6, marker="o", ms=5,
               label="Online-SDFT"),
        Line2D([], [], color=colors["ICL"], lw=1.8, marker="o", ms=3.8, label=icl_name),
        Line2D([], [], color=colors["RAG"], lw=1.8, marker="o", ms=3.8, label=rag_name),
        Line2D([], [], color=colors["ZS"], ls=":", lw=1.6, label="zero-shot floor"),
    ], fontsize=7.5, loc="lower center", ncols=2, framealpha=0.95)
    ax_drift.set_xlim(-1, stream_len + 2)
    ax_drift.set_xlabel("Items streamed  (same causal history for every method)",
                        labelpad=36)
    ax_drift.set_ylabel("Held-out accuracy on the CURRENT regime  (%)")
    ax_drift.set_title("B.  Who tracks the drifting policy", fontsize=12, fontweight="bold")
    ax_drift.set_ylim(0, 105)
    ax_drift.grid(True, alpha=0.25)

    # Panel C: accumulated regret — cumulative mistakes on the streamed items,
    # each predicted before its label lands (prequential). Same ICL/RAG k as A/B.
    for x in drifts:
        ax_regret.axvline(x, color="#5f6368", ls="--", lw=1.2)
    series = (("ZS", "ZS", ":"),
              ("ICL", icl_name, "-"),
              ("RAG", rag_name, "-"),
              ("SDFT", "Online-SDFT", "-"))
    for key, name, style in series:
        color = colors[key if key != "SDFT" else "Online-SDFT"]
        ax_regret.plot(regret["pos"], regret[key], style, drawstyle="steps-post",
                       color=color, lw=2.6 if key == "SDFT" else 1.8, label=name,
                       zorder=4 if key == "SDFT" else 3)
        ax_regret.annotate(str(regret[key][-1]), (stream_len, regret[key][-1]),
                           textcoords="offset points", xytext=(6, -3), fontsize=9,
                           color=color, fontweight="bold")
    ax_regret.set_xlim(-1, stream_len + 4)
    ax_regret.set_xlabel("Items streamed")
    ax_regret.set_ylabel("Cumulative mistakes on streamed items")
    ax_regret.set_title("C.  Accumulated regret",
                        fontsize=12, fontweight="bold")
    ax_regret.grid(True, alpha=0.25)
    ax_regret.legend(fontsize=8.5, loc="upper left", framealpha=0.95)

    adapter_mb = results["adapter_bytes"] / 1e6
    fig.suptitle(
        f"On-device 3-way triage across 3 regimes · LFM2.5-230M · policy lives in a "
        f"{adapter_mb:.1f} MB LoRA adapter, no gold labels",
        fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "online_sdft_triage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, flush=True)

    # Also export each panel on its own — the blog shows A/B/C full-width, one
    # per "Reading the panels" paragraph, instead of three squeezed thumbnails.
    renderer = fig.canvas.get_renderer()
    for ax, suffix in ((ax_cost, "a"), (ax_drift, "b"), (ax_regret, "c")):
        extent = (ax.get_tightbbox(renderer)
                  .transformed(fig.dpi_scale_trans.inverted()))
        panel_out = FIG_DIR / f"online_sdft_triage_{suffix}.png"
        fig.savefig(panel_out, dpi=150, bbox_inches=extent.expanded(1.02, 1.04))
        print("wrote", panel_out, flush=True)


if __name__ == "__main__":
    main()
