"""Hyper-parameter sweep for the online-SDFT arm of the triage showcase.

Each config replays the exact online loop of run_sdft.py — prequential student
guess (bare prompt), teacher conditioned on the expert (user) action, soft-CE
distill with all-class replay, probe guardrail over the final regime — but
probes only what the headline metric needs, so one config costs ~4 min instead
of a full run:

  metric   set PRIMARY below. "live_mean" ranks by the whole-week live mean
           (held-out accuracy per regime at its block-end checkpoint, averaged
           — Panel A's yardstick) with regret as tiebreak; "regret" ranks by
           accumulated stream mistakes (Panel C's yardstick) with the live
           mean as tiebreak. TEACHER_SHOTS prepends older causal decisions to
           the teacher chat only (student always guesses bare).

Three stages, all automatic:

  A  grid over LR x STEPS_PER_ITEM (the two axes that bite: too-small LRs
     stall on 1-2-token completions, too many steps over-train past the peak)
  B  one-at-a-time around the stage-A winner: LORA_R, TEACHER_SHOTS, REPLAY
  C  seed check: winner vs the current default on two extra torch seeds
     (the dataset is frozen — the seed moves only LoRA init / dropout / replay
     sampling), so the crown isn't a single-seed lottery ticket

Writes outputs/sdft_sweep.json and prints a ranked table.

Run:  python sweep_sdft.py
"""

from __future__ import annotations

import json
import random
import time

import torch
from peft import LoraConfig, get_peft_model

from run_sdft import make_updater
from triage_common import (
    ACTIONS, DRIFTS, MODEL_NAME, OUT_DIR, REGIMES, SEED, STREAM_LEN, accuracy,
    build_eval, build_msgs, build_stream, build_teacher_msgs, generate,
    load_base_model, load_tokenizer, parse_action, pick_device, recent_demos,
    render_prompt,
)

# --- the sweep space --------------------------------------------------------- #
PRIMARY = "regret"        # "live_mean" (Panel A) or "regret" (Panel C) first
DEFAULT = {"lr": 7e-4, "steps": 8, "r": 8, "shots": 0, "replay": 16}   # adopted winner
LR_GRID = (7e-4, 1e-3)    # 5e-4 underperforms, 2e-3 collapses (earlier sweep)
STEPS_GRID = (5, 6, 8)    # steps=5 scored regret 24 in the earlier sweep
R_GRID = (16,)            # stage B, around the winner (8 is the default centre)
SHOTS_GRID = (0,)         # 0 = teacher sees only this item's expert action
REPLAY_GRID = (8,)        # 32 drowns the fresh item (earlier sweep)
EXTRA_SEEDS = (8, 9)      # stage C robustness check (data stays seed-7)

BLOCK_ENDS = tuple(zip([*DRIFTS, STREAM_LEN], (1, 2, 3)))   # (pos, phase)
GUARD_PROBES = tuple(pos for pos in range(6, STREAM_LEN + 1, 6) if pos > DRIFTS[1])
SWEEP_JSON = OUT_DIR / "sdft_sweep.json"


def run_config(base, tok, stream, evals, cfg: dict, torch_seed: int = SEED) -> dict:
    """One abbreviated online-SDFT run; returns the headline metrics."""
    torch.manual_seed(torch_seed)
    model = get_peft_model(base, LoraConfig(
        r=cfg["r"], lora_alpha=2 * cfg["r"], lora_dropout=0.05,
        target_modules=r".*self_attn\.(q|k|v|out)_proj", task_type="CAUSAL_LM"))
    update = make_updater(model, tok, lr=cfg["lr"])

    def probe(phase: int) -> float:
        return accuracy(evals[phase], generate(
            model, tok, [build_msgs(item) for item in evals[phase]],
            label=f"probe/p{phase}"))

    live: dict[int, float] = {}
    guard_best = -1.0
    mistakes = 0
    replay_buffer: list[dict] = []
    sampler = random.Random(torch_seed)
    for i, item in enumerate(stream):
        # STUDENT: bare serving call (regret measurement)
        guess = generate(model, tok, [build_msgs(item)],
                         label=f"guess@{i + 1}", batch_size=1)[0]
        mistakes += parse_action(guess) != item["action"]

        history = recent_demos(stream[:i], cfg["shots"]) if cfg["shots"] else None
        row = {"prompt": item["prompt"], "action": item["action"],
               "teacher_msgs": build_teacher_msgs(item, item["action"], history)}
        replay_buffer = (replay_buffer + [row])[-cfg["replay"]:]
        batch = [row]
        for action in ACTIONS:
            pool = [b for b in replay_buffer[:-1] if b["action"] == action]
            if action != item["action"] and pool:
                batch.append(sampler.sample(pool, 1)[0])
        update(batch, cfg["steps"])

        pos = i + 1
        for end, phase in BLOCK_ENDS:
            if pos == end:
                live[phase] = probe(phase)
        if pos in GUARD_PROBES:
            acc = live[3] if pos == STREAM_LEN else probe(3)   # reuse the pos-60 probe
            guard_best = max(guard_best, acc)

    # detach the adapter so the shared base is pristine for the next config
    model.unload()
    return {"live_by": {REGIMES[p - 1]: live[p] for p in (1, 2, 3)},
            "live_mean": sum(live.values()) / 3,
            "regret": mistakes, "guard_best": guard_best}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device_note = "cuda" if torch.cuda.is_available() else "mps/cpu"
    print(f"model={MODEL_NAME} ({device_note})  PRIMARY metric={PRIMARY}", flush=True)

    stream = build_stream(random.Random(SEED))
    for item in stream:   # the loop trains on prompts, same as the exported dataset
        item["prompt"] = render_prompt(item)
    evals = {phase: build_eval(random.Random(SEED + phase), phase) for phase in (1, 2, 3)}

    tok = load_tokenizer()
    base = load_base_model(pick_device())

    results: list[dict] = []

    def trial(cfg: dict, tag: str, torch_seed: int = SEED) -> dict:
        t0 = time.time()
        out = run_config(base, tok, stream, evals, cfg, torch_seed)
        entry = {"tag": tag, "seed": torch_seed, **cfg, **out,
                 "minutes": round((time.time() - t0) / 60, 1)}
        results.append(entry)
        SWEEP_JSON.write_text(json.dumps(results, indent=2))   # checkpoint as we go
        by = "  ".join(f"{k}={v:.2f}" for k, v in out["live_by"].items())
        print(f"  [{tag}] {cfg} seed={torch_seed}: mean={out['live_mean']:.3f} "
              f"({by})  regret={out['regret']}/60  ({entry['minutes']}m)", flush=True)
        return entry

    def rank(entry: dict) -> tuple:
        if PRIMARY == "regret":
            return (-entry["regret"], entry["live_mean"])
        return (entry["live_mean"], -entry["regret"])

    print("\n== stage A: LR x STEPS_PER_ITEM ==", flush=True)
    for lr in LR_GRID:
        for steps in STEPS_GRID:
            trial({**DEFAULT, "lr": lr, "steps": steps}, "A")
    best = max(results, key=rank)
    print(f"  stage-A winner: lr={best['lr']}, steps={best['steps']}", flush=True)

    print("\n== stage B: one-at-a-time around the winner ==", flush=True)
    centre = {k: best[k] for k in DEFAULT}
    for key, grid in (("r", R_GRID), ("shots", SHOTS_GRID), ("replay", REPLAY_GRID)):
        for value in grid:
            trial({**centre, key: value}, f"B/{key}")
    best = max(results, key=rank)
    winner = {k: best[k] for k in DEFAULT}
    print(f"  overall winner: {winner}", flush=True)

    print("\n== stage C: winner vs current default on extra torch seeds ==", flush=True)
    for torch_seed in EXTRA_SEEDS:
        trial(winner, "C/winner", torch_seed)
        if winner != DEFAULT:
            trial(DEFAULT, "C/default", torch_seed)

    print("\n== ranked (seed 7) ==", flush=True)
    for entry in sorted((e for e in results if e["seed"] == SEED), key=rank, reverse=True):
        print(f"  {entry['live_mean']:.3f}  regret={entry['regret']:2d}  "
              f"lr={entry['lr']:<6} steps={entry['steps']} r={entry['r']:<2} "
              f"shots={entry['shots']} replay={entry['replay']}  [{entry['tag']}]", flush=True)
    print(f"\nwrote {SWEEP_JSON}", flush=True)
    print("WINNER:", json.dumps(winner), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
