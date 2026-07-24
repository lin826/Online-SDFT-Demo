"""Run one baseline (ICL|RAG, k) across SEEDS; save after each seed."""
from __future__ import annotations

import json
import sys

from sweep_sft import (
    DEFAULT_ICL_K, DEFAULT_RAG_K, OUT_JSON, SFT_FLAT_JSON, SEEDS,
    _prepare_stream, load_base_model, load_tokenizer, pick_device,
    run_baseline_live, run_baseline_regret, save, summarize,
)


def main(method: str, k: int) -> None:
    bucket = method.lower()
    p = json.loads(OUT_JSON.read_text())
    flat = json.loads(SFT_FLAT_JSON.read_text()) if SFT_FLAT_JSON.exists() else []
    p.setdefault(bucket, {})
    rows = list(p[bucket].get(str(k), {}).get("rows", []))
    done = {r["seed"] for r in rows}
    default_k = DEFAULT_ICL_K if method == "ICL" else DEFAULT_RAG_K
    tok = load_tokenizer()
    base = load_base_model(pick_device())
    print(f"start {method} k={k} done={sorted(done)}", flush=True)
    for seed in SEEDS:
        if seed in done:
            print(f"skip seed={seed}", flush=True)
            continue
        stream, evals, be, _ = _prepare_stream(seed)
        regret = run_baseline_regret(base, tok, stream, method, k)
        live = run_baseline_live(base, tok, stream, evals, method, k, be)
        rows.append({"seed": seed, "regret": regret, "live_mean": live})
        print(f"  seed={seed}: regret={regret} live={live:.3f}", flush=True)
        p[bucket][str(k)] = {
            "k": k, "is_default": k == default_k,
            **summarize(rows), "rows": rows,
        }
        save(p, flat)
    s = p[bucket][str(k)]
    print(f"DONE {method} k={k}: {s['regret_mean']:.1f}±{s['regret_std']:.1f} "
          f"n={s['n']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
