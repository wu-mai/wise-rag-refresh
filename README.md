# Web-RAG Refresh: Cost-Aware Cache Refresh Under Drift Regimes

Companion code for the WISE 2026 paper:

**"When Should Web-RAG Refresh? Adapting Refresh Policies Across Drift Regimes"**

## What this repo contains

- `src/fresh_rag/` — the RAG pipeline: router, retriever (BGE), refresh policy, reranker, generator (GPT-5.4).
- `src/fresh_rag/backends/` — corpus backends: FreshQA (Wayback) and TimeVarying (our curated high-drift benchmark).
- `src/fresh_rag/drift_predictor.py` — Length+Source MLP / BGE+MLP drift predictors.
- `experiments/label_efficiency/` — the main paper experiment: 5 k-values x 3 seeds, fixed entertainment hold-out.
- `experiments/target_only_baseline/` — from-scratch (generic-init) control arm for the adaptation story: same hold-out, same k-grid, same seeds as label_efficiency; two lr variants.
- `experiments/blockE_nobonus/` — reranker-bonus ablation.
- `experiments/regime_classifier/` — self-contained corpus-level regime classifier: 6-feature MLP, 5-seed out-of-fold evaluation, τ_regime midpoint. Bundles a 20-query low-drift synthetic corpus.
- `experiments/decomp_freshqa/` — self-contained decomposition harness; supports `--mode synthetic` (bundled, default) and `--mode freshqa` (requires the FreshQA Wayback corpus).
- `data/time_varying_benchmark.json` — our 60-query 5-category TimeVarying benchmark (released).
- `data/synthetic_lowdrift/queries.json` — 20 stable-knowledge queries used as the low-drift validation corpus for the regime classifier.
- `data/drift_scores.json` — per-query retrospective drift δ for both corpora (used by the regime classifier).
- `results/label_efficiency.json` — raw per-(k, seed) F1 numbers for the label-efficiency curve.
- `results/target_only_baseline.json` — from-scratch control arm: same protocol, generic DeBERTa-v3-small init instead of the proxy-trained router.pt.
- `results/regime_classifier.json` — per-seed τ_regime, δ̂̄, and classification decision.
- `results/decomp_freshqa.json` — three-condition decomposition in `synthetic` mode.
- `docs/` — design spec + plan for the paper.

## Reproducing the main result

The headline number in the paper (adapted router F1 = 0.685 on
TimeVarying) is produced by a two-step pipeline. The pre-trained
source router checkpoint (`models/router.pt`, ~540 MB) is not
included in this repo --- train it from scratch in Step 1, or skip
to Step 2 if you already have one.

**Step 1. Train the source router on FreshQA** (~30-60 min on 1x
RTX 4090; held-out MSE should land near 0.032):

```bash
python3 experiments/train_router.py
# → writes models/router.pt and models/router_metrics.json
```

Inputs: `data/freshqa_test.parquet` (already in the repo). Hyper-params
hard-coded in the script: 6 epochs, batch 16, lr 2e-5, AdamW with
linear warmup, MSE loss, max_len 64.

**Step 2. Few-shot adapt to TimeVarying and run the label-efficiency
curve** (reproduces Table 5 and Fig. 4 of the paper):

```bash
python3 experiments/label_efficiency/run_label_efficiency.py
# → writes results/label_efficiency.json
```

For the full pipeline (cost-accuracy frontier, LOCO cross-validation,
Drift Predictor ablation, Meta-Router), see the per-experiment scripts
under `experiments/`. Outputs go to `outputs/<experiment>.log` and
summaries to `results/`.

**New self-contained experiments** (revision):

```bash
python3 experiments/regime_classifier/run_regime_classifier.py
# → results/regime_classifier.json + figures/fig_regime_classifier.pdf
python3 experiments/decomp_freshqa/run_decomp.py            # synthetic mode
python3 experiments/decomp_freshqa/run_decomp.py --mode freshqa   # needs FreshQA corpus
```

The regime classifier runs out of the box (uses only bundled data); the
decomposition runs in `synthetic` mode by default and switches to the
real FreshQA corpus when `models/router.pt` + `data/freshqa_*.parquet` are
present.

## Setup

- Python 3.13
- PyTorch (CUDA optional; CPU works for the small Drift Predictor)
- `transformers`, `numpy`, `pandas`, `scipy`, `matplotlib`
- See `pyproject.toml` for the pinned versions.

The BGE-base retriever and DeBERTa-v3-small router checkpoints are
auto-downloaded on first use; the TimeVarying benchmark is already
in `data/`.

## Layout

```
src/fresh_rag/        — pipeline + backends + drift predictor
experiments/          — per-block experiment scripts
data/                 — released benchmark (TimeVarying)
results/              — numeric outputs (label_efficiency.json etc.)
outputs/              — raw per-experiment logs
docs/                 — design spec and implementation plan
tests/                — unit tests (pytest)
literature/           — literature survey notes
```

## Citation

```bibtex
@inproceedings{anonymous2026when,
  title  = {When Should Web-RAG Refresh? Adapting Refresh Policies Across Drift Regimes},
  author = {Anonymous},
  year   = {2026},
  booktitle = {Proceedings of the International Conference on Web Information Systems Engineering (WISE)}
}
```

## License

TBD — paper is currently under double-blind review; the license will
be added on the camera-ready.
