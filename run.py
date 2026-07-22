"""Reproduce the whole showcase in one command:  python run.py

Runs the causal ZS / ICL / RAG baselines (with both k sweeps), then the online
SDFT loop, then draws every figure. Roughly 15 minutes on an M-series Mac (MPS)
or any CUDA GPU; CPU works but is slow. Offline after the first model download.

Outputs: outputs/results.json, the trained adapter, figures/*.png.
"""

import draw_loop_diagram
import run_baselines
import run_sdft

run_baselines.main()
run_sdft.main()
draw_loop_diagram.main()
