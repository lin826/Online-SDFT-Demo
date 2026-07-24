"""Finish remaining multi-seed regret eval without touching sweep_sft.py.

Completes:
  - DEFAULT SFT on 5 seeds (comparison baseline)
  - ICL / RAG k grids on 5 seeds
  - Regime schedule variants on 5 seeds using SFT/ICL/RAG winners
  - winners block in outputs/regret_sweep.json

Uses in-process model reload (stable on MPS). Resumes partial JSON.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from sweep_sft import (
    DEFAULT, DEFAULT_ICL_K, DEFAULT_RAG_K, ICL_KS, RAG_KS, SCHEDULES, SEEDS,
    OUT_JSON, SFT_FLAT_JSON, _cfg_key, _cleanup_mps, _prepare_stream,
    is_corrupt, load_base_model, load_tokenizer, pick_device, rank_key,
    run_baseline_live, run_baseline_regret, run_sft_fresh, save, sft_once,
    summarize, MODEL_NAME,
)

LOCK = Path("/tmp/online_sft_finish.lock")


def main() -> None:
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            import os
            os.kill(old, 0)
            raise SystemExit(f"finish already running pid={old}")
        except (ValueError, OSError):
            pass
    import os
    LOCK.write_text(str(os.getpid()))
    try:
        _run()
    finally:
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink(missing_ok=True)


def _run() -> None:
    print(f"Finish regret eval  model={MODEL_NAME}  seeds={SEEDS}", flush=True)
    payload, flat = json.loads(OUT_JSON.read_text()), (
        json.loads(SFT_FLAT_JSON.read_text()) if SFT_FLAT_JSON.exists() else [])
    payload.setdefault("sft", {}).setdefault("multi_seed", {})
    payload.setdefault("icl", {})
    payload.setdefault("rag", {})
    payload.setdefault("schedules", {})

    # --- DEFAULT SFT on all seeds ------------------------------------------ #
    dkey = _cfg_key(DEFAULT)
    print(f"\n== DEFAULT SFT multi-seed ({dkey}) ==", flush=True)
    existing = payload["sft"]["multi_seed"].get(dkey, {})
    rows = [r for r in existing.get("rows", []) if not is_corrupt(r)]
    done = {r["seed"] for r in rows}
    for seed in SEEDS:
        if seed in done:
            print(f"  [skip] seed={seed}", flush=True)
            continue
        out = run_sft_fresh(DEFAULT, seed)
        if out.get("corrupt"):
            print(f"  [FAIL] seed={seed} corrupt — retry", flush=True)
            time.sleep(3)
            _cleanup_mps()
            out = run_sft_fresh(DEFAULT, seed)
        if out.get("corrupt"):
            print(f"  [FAIL] seed={seed} still corrupt", flush=True)
            continue
        rows.append({"seed": seed, "regret": out["regret"],
                     "live_mean": out["live_mean"],
                     "live_by": out.get("live_by", {}),
                     "guard_best": out.get("guard_best", 0.0),
                     "minutes": out.get("minutes", 0.0)})
        flat.append({"seed": seed, "tag": "B/default", **DEFAULT,
                     "regret": out["regret"], "live_mean": out["live_mean"],
                     "minutes": out.get("minutes", 0.0)})
        print(f"  seed={seed}: regret={out['regret']}/60  live={out['live_mean']:.3f}  "
              f"({out.get('minutes', 0):.2f}m)", flush=True)
        payload["sft"]["multi_seed"][dkey] = {"cfg": DEFAULT, **summarize(rows),
                                              "rows": rows}
        save(payload, flat)
    if dkey in payload["sft"]["multi_seed"]:
        s = payload["sft"]["multi_seed"][dkey]
        print(f"  >> DEFAULT: regret {s['regret_mean']:.1f}±{s['regret_std']:.1f}",
              flush=True)

    complete = {k: v for k, v in payload["sft"]["multi_seed"].items()
                if v.get("n", 0) >= len(SEEDS)}
    if not complete:
        raise SystemExit("no complete SFT multi-seed results")
    best_sft_key = min(complete, key=lambda k: rank_key(complete[k]))
    best_sft_cfg = complete[best_sft_key]["cfg"]
    print(f"  SFT winner: {best_sft_key}", flush=True)

    # --- ICL / RAG ---------------------------------------------------------- #
    device = pick_device()
    tok = load_tokenizer()
    base = load_base_model(device)
    print(f"\n== ICL / RAG k on seeds {SEEDS} ==", flush=True)
    for method, ks, bucket, default_k in (
            ("ICL", ICL_KS, "icl", DEFAULT_ICL_K),
            ("RAG", RAG_KS, "rag", DEFAULT_RAG_K)):
        for k in ks:
            if str(k) in payload[bucket] and payload[bucket][str(k)].get("n", 0) >= len(SEEDS):
                s = payload[bucket][str(k)]
                print(f"  [{method} k={k} skip] regret "
                      f"{s['regret_mean']:.1f}±{s['regret_std']:.1f}", flush=True)
                continue
            rows = []
            for seed in SEEDS:
                stream, evals, block_ends, _ = _prepare_stream(seed)
                regret = run_baseline_regret(base, tok, stream, method, k)
                live = run_baseline_live(base, tok, stream, evals, method, k, block_ends)
                rows.append({"seed": seed, "regret": regret, "live_mean": live})
                print(f"  [{method} k={k}] seed={seed}: regret={regret}/60  "
                      f"live={live:.3f}", flush=True)
            payload[bucket][str(k)] = {
                "k": k, "is_default": k == default_k,
                **summarize(rows), "rows": rows,
            }
            save(payload, flat)
            s = payload[bucket][str(k)]
            print(f"  >> {method} k={k}: regret {s['regret_mean']:.1f}±{s['regret_std']:.1f}",
                  flush=True)

    del base
    _cleanup_mps()
    best_icl_k = min(payload["icl"], key=lambda k: rank_key(payload["icl"][k]))
    best_rag_k = min(payload["rag"], key=lambda k: rank_key(payload["rag"][k]))
    print(f"  ICL winner k={best_icl_k}  RAG winner k={best_rag_k}", flush=True)

    # --- Schedules ---------------------------------------------------------- #
    print(f"\n== schedules on seeds {SEEDS} ==", flush=True)
    tok = load_tokenizer()
    base = load_base_model(device)
    icl_k, rag_k = int(best_icl_k), int(best_rag_k)
    for sched_name, sched in SCHEDULES.items():
        sched_payload = {"order": list(sched["order"]),
                         "lengths": list(sched["lengths"])}
        existing = payload["schedules"].get(sched_name)
        if existing and existing.get("sft", {}).get("n", 0) >= len(SEEDS):
            print(f"  [{sched_name} skip]", flush=True)
            continue
        rows_sft, rows_icl, rows_rag = [], [], []
        for seed in SEEDS:
            out = run_sft_fresh(best_sft_cfg, seed, schedule=sched_payload)
            stream, evals, block_ends, _ = _prepare_stream(seed, sched_payload)
            icl_r = run_baseline_regret(base, tok, stream, "ICL", icl_k)
            rag_r = run_baseline_regret(base, tok, stream, "RAG", rag_k)
            icl_l = run_baseline_live(base, tok, stream, evals, "ICL", icl_k, block_ends)
            rag_l = run_baseline_live(base, tok, stream, evals, "RAG", rag_k, block_ends)
            if not out.get("corrupt"):
                rows_sft.append({"seed": seed, "regret": out["regret"],
                                  "live_mean": out["live_mean"]})
            rows_icl.append({"seed": seed, "regret": icl_r, "live_mean": icl_l})
            rows_rag.append({"seed": seed, "regret": rag_r, "live_mean": rag_l})
            print(f"  [{sched_name}] seed={seed}: SFT={out.get('regret','?')}  "
                  f"ICL={icl_r}  RAG={rag_r}", flush=True)
            payload["schedules"][sched_name] = {
                "schedule": sched_payload,
                "sft": {"cfg_key": best_sft_key, "cfg": best_sft_cfg,
                         **summarize(rows_sft), "rows": rows_sft} if rows_sft else {},
                "icl": {"k": icl_k, **summarize(rows_icl), "rows": rows_icl},
                "rag": {"k": rag_k, **summarize(rows_rag), "rows": rows_rag},
            }
            save(payload, flat)

    # --- Winners ------------------------------------------------------------ #
    def_ref = payload["sft"]["multi_seed"].get(dkey)
    winners = {
        "sft": {
            "key": best_sft_key,
            "cfg": best_sft_cfg,
            **{k: complete[best_sft_key][k]
               for k in ("regret_mean", "regret_std", "live_mean", "live_std", "n")},
            "beats_default": (
                def_ref is not None and def_ref.get("n", 0) >= len(SEEDS)
                and rank_key(complete[best_sft_key]) < rank_key(def_ref)
            ),
            "default_ref": None if not def_ref else {
                "regret_mean": def_ref["regret_mean"],
                "regret_std": def_ref["regret_std"],
                "live_mean": def_ref["live_mean"],
            },
        },
        "icl": {
            "k": int(best_icl_k),
            **{k: payload["icl"][best_icl_k][k]
               for k in ("regret_mean", "regret_std", "live_mean", "live_std", "n")},
            "default_k": DEFAULT_ICL_K,
            "beats_default": rank_key(payload["icl"][best_icl_k])
            < rank_key(payload["icl"][str(DEFAULT_ICL_K)]),
            "default_ref": {
                "regret_mean": payload["icl"][str(DEFAULT_ICL_K)]["regret_mean"],
                "regret_std": payload["icl"][str(DEFAULT_ICL_K)]["regret_std"],
            },
        },
        "rag": {
            "k": int(best_rag_k),
            **{k: payload["rag"][best_rag_k][k]
               for k in ("regret_mean", "regret_std", "live_mean", "live_std", "n")},
            "default_k": DEFAULT_RAG_K,
            "beats_default": rank_key(payload["rag"][best_rag_k])
            < rank_key(payload["rag"][str(DEFAULT_RAG_K)]),
            "default_ref": {
                "regret_mean": payload["rag"][str(DEFAULT_RAG_K)]["regret_mean"],
                "regret_std": payload["rag"][str(DEFAULT_RAG_K)]["regret_std"],
            },
        },
    }
    if payload["schedules"]:
        winners["schedule"] = min(
            (n for n, s in payload["schedules"].items() if s.get("sft")),
            key=lambda n: (
                payload["schedules"][n]["sft"]["regret_mean"]
                + payload["schedules"][n]["icl"]["regret_mean"]
                + payload["schedules"][n]["rag"]["regret_mean"]
            ),
            default=None,
        )
    payload["winners"] = winners
    save(payload, flat)

    print("\n== winners ==", flush=True)
    print(json.dumps(winners, indent=2), flush=True)
    print(f"wrote {OUT_JSON}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
