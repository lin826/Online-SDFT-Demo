"""Run one regime schedule across SEEDS for SFT winner + best ICL/RAG."""
from __future__ import annotations

import json
import sys

from sweep_sft import (
    OUT_JSON, SFT_FLAT_JSON, SCHEDULES, SEEDS, _cfg_key, _prepare_stream,
    is_corrupt, load_base_model, load_tokenizer, pick_device, rank_key,
    run_baseline_live, run_baseline_regret, run_sft_fresh, save, summarize,
)


def main(sched_name: str) -> None:
    if sched_name not in SCHEDULES:
        raise SystemExit(f"unknown schedule {sched_name}; have {list(SCHEDULES)}")
    p = json.loads(OUT_JSON.read_text())
    flat = json.loads(SFT_FLAT_JSON.read_text()) if SFT_FLAT_JSON.exists() else []
    complete = {k: v for k, v in p["sft"]["multi_seed"].items()
                if v.get("n", 0) >= len(SEEDS)}
    best_sft_key = min(complete, key=lambda k: rank_key(complete[k]))
    best_sft_cfg = complete[best_sft_key]["cfg"]
    best_icl = min(p["icl"], key=lambda k: rank_key(p["icl"][k]))
    best_rag = min(p["rag"], key=lambda k: rank_key(p["rag"][k]))
    icl_k, rag_k = int(best_icl), int(best_rag)

    sched = SCHEDULES[sched_name]
    sp = {"order": list(sched["order"]), "lengths": list(sched["lengths"])}
    existing = p.get("schedules", {}).get(sched_name, {})
    rows_s = list(existing.get("sft", {}).get("rows", []))
    rows_i = list(existing.get("icl", {}).get("rows", []))
    rows_r = list(existing.get("rag", {}).get("rows", []))
    done = {r["seed"] for r in rows_s if not is_corrupt(r)}

    tok = load_tokenizer()
    base = load_base_model(pick_device())
    print(f"schedule={sched_name} sft={best_sft_key} icl={icl_k} rag={rag_k} "
          f"done={sorted(done)}", flush=True)

    for seed in SEEDS:
        if seed in done:
            print(f"skip seed={seed}", flush=True)
            continue
        out = run_sft_fresh(best_sft_cfg, seed, schedule=sp)
        stream, evals, be, _ = _prepare_stream(seed, sp)
        ir = run_baseline_regret(base, tok, stream, "ICL", icl_k)
        rr = run_baseline_regret(base, tok, stream, "RAG", rag_k)
        il = run_baseline_live(base, tok, stream, evals, "ICL", icl_k, be)
        rl = run_baseline_live(base, tok, stream, evals, "RAG", rag_k, be)
        if not out.get("corrupt"):
            rows_s.append({"seed": seed, "regret": out["regret"],
                           "live_mean": out["live_mean"]})
        # replace baseline rows for this seed if re-run
        rows_i = [r for r in rows_i if r["seed"] != seed] + [
            {"seed": seed, "regret": ir, "live_mean": il}]
        rows_r = [r for r in rows_r if r["seed"] != seed] + [
            {"seed": seed, "regret": rr, "live_mean": rl}]
        print(f"  seed={seed}: SFT={out.get('regret')} ICL={ir} RAG={rr} "
              f"corrupt={out.get('corrupt')}", flush=True)
        p.setdefault("schedules", {})[sched_name] = {
            "schedule": sp,
            "sft": {"cfg_key": best_sft_key, "cfg": best_sft_cfg,
                     **summarize(rows_s), "rows": rows_s} if rows_s else {},
            "icl": {"k": icl_k, **summarize(rows_i), "rows": rows_i},
            "rag": {"k": rag_k, **summarize(rows_r), "rows": rows_r},
        }
        save(p, flat)

    s = p["schedules"][sched_name]
    print(f"DONE {sched_name}: SFT {s['sft'].get('regret_mean')}  "
          f"ICL {s['icl']['regret_mean']:.1f}  RAG {s['rag']['regret_mean']:.1f}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
