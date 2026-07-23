# Online SDFT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lin826/Online-SDFT-Demo/blob/main/online_sdft_colab.ipynb)

A phone-class 230M model ([LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)) learns your **drifting** notification-triage policy **online** — one `batch_size=1` LoRA update per item via **self-distillation**: the student guesses with a bare prompt, the teacher is the same adapter conditioned on your observed action (open / wait / never), and reverse KL distills teacher → student — **no hand-written gold labels** — beating causal ICL and RAG on whole-week accuracy, accumulated regret, and per-query cost. The learned policy is a **~1.4 MB adapter** served with a bare ~90-token prompt.

![Whole-week accuracy vs token bill, current-regime tracking, accumulated regret](figures/online_sdft_triage.png)

Companion code for the blog post *"Your phone should learn your attention, not just borrow it"*, using **self-distillation fine-tuning (SDFT)** ([Shenfeld et al., 2026](https://arxiv.org/abs/2601.19897)) run online.



## Reproduce

```bash
pip install -r requirements.txt
```

```bash
python run.py
```

That runs the causal baselines (with their k sweeps), the online SDFT loop, and draws every figure — `outputs/results.json` + `figures/*.png`, seeded end to end (same command, same numbers on the same device). About 15 minutes on an M-series Mac (MPS) or any CUDA GPU.

Prefer a Colab Jupyter Notebook finished in 3 minutes? Open [online_sdft_colab.ipynb](https://colab.research.google.com/github/lin826/Online-SDFT-Demo/blob/main/online_sdft_colab.ipynb) on a free Colab T4 — standalone, it fetches the seeded dataset straight from this repo. Optional local alternative:

```bash
python sweep_sdft.py     # the hyper-parameter sweep that picked the shipped setup
```

## What's here

| File                     | Role                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| `triage_common.py`       | the drifting inbox stream, the 3-way policy, model helpers                                   |
| `run_baselines.py`       | causal ZS / ICL / RAG arms + both k sweeps → `outputs/baselines.json`                        |
| `run_sdft.py`            | the online SDFT loop (student guess → teacher+expert → reverse-KL → guardrail) → results + figures |
| `sweep_sdft.py`          | the sweep behind the shipped hyper-parameters (regret-primary)                               |
| `draw_loop_diagram.py`   | the TEACH / CHECK / LEARN loop diagram                                                       |
| `data/inbox_triage.json` | the seeded dataset (re-exported and verified on every baselines run)                         |
