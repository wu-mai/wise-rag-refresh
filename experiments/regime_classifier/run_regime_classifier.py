#!/usr/bin/env python3
"""Run the corpus-level regime classifier experiment.

Self-contained: uses only files in this repo —
    - data/drift_scores.json (per-query retrospective drift δ)
    - data/synthetic_lowdrift/queries.json (low-drift stand-in corpus)
    - data/time_varying_benchmark.json (high-drift corpus)

Protocol (mirrors the deployment story in the paper):
    1. Per corpus, hold out a batch of N_BATCH=20 queries (never trained on).
    2. Train the 6-feature MLP on the remaining queries from both corpora.
    3. For each held-out batch, average the predicted drift δ̂ over the 20
       queries and threshold at the midpoint between the two corpora's
       training-fold mean predictions (τ_regime).
    4. The predicted regime selects a policy family:
         low  -> Never Refresh / Keyword Router
         high -> Always Refresh / adapted Fact-Type Router
    5. Repeat over 5 random seeds for the held-out batch composition.

Outputs:
    - results/regime_classifier.json
    - figures/fig_regime_classifier.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from regime_clf import DriftPredictorMLP, load_drift_data, predict_regime

N_BATCH_TV = 20   # held-out batch for TimeVarying (matches the paper's N=20)
N_BATCH_SYN = 12  # synthetic corpus has only 20 queries; hold out 12, train on 8
SEEDS = [0, 1, 2, 3, 4]


def main():
    print("=" * 70)
    print("Corpus-level regime classifier (self-contained, out-of-fold batches)")
    print("=" * 70)

    X, y, items = load_drift_data()
    n_syn = sum(1 for it in items if it[3] == "synthetic_lowdrift")
    n_tv  = sum(1 for it in items if it[3] == "timevarying")
    print(f"Loaded {len(X)} (query, stale, drift) triples")
    print(f"  Synthetic low-drift: {n_syn}")
    print(f"  TimeVarying:         {n_tv}")

    syn_idx = np.array([i for i, it in enumerate(items) if it[3] == "synthetic_lowdrift"])
    tv_idx  = np.array([i for i, it in enumerate(items) if it[3] == "timevarying"])

    # End-to-end F1 of the policy the classifier would select for each regime.
    # These come from the paper's Tables 2 and 3 (measured on the original
    # machine; see paper for details).
    POLICY_F1 = {
        "synthetic_lowdrift": {"low":  0.192, "high": 0.173},  # Never / Always
        "timevarying":        {"low":  0.191, "high": 0.685},  # Never / adapted router
    }

    per_seed = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        syn_batch = rng.choice(syn_idx, size=N_BATCH_SYN, replace=False)
        tv_batch  = rng.choice(tv_idx,  size=N_BATCH_TV, replace=False)
        test_set = set(syn_batch.tolist() + tv_batch.tolist())
        train_idx = [i for i in range(len(items)) if i not in test_set]

        pred = DriftPredictorMLP()
        pred.fit(X[train_idx], y[train_idx], lr=0.001, epochs=300, l2=1e-3)

        # τ_regime: midpoint of training-fold corpus means (out-of-fold safe)
        tr_syn = [i for i in train_idx if items[i][3] == "synthetic_lowdrift"]
        tr_tv  = [i for i in train_idx if items[i][3] == "timevarying"]
        _, mu_syn = predict_regime([(items[i][1], items[i][2]) for i in tr_syn], pred)
        _, mu_tv  = predict_regime([(items[i][1], items[i][2]) for i in tr_tv], pred)
        tau = (mu_syn + mu_tv) / 2

        row = {"seed": seed, "tau_regime": round(float(tau), 4), "corpora": {}}
        for name, batch in [("synthetic_lowdrift", syn_batch),
                            ("timevarying", tv_batch)]:
            pairs = [(items[i][1], items[i][2]) for i in batch]
            regime, mean_pred = predict_regime(pairs, pred, tau=tau)
            truth = "low" if name == "synthetic_lowdrift" else "high"
            correct = regime == truth
            f1 = POLICY_F1[name][regime]
            row["corpora"][name] = {
                "mean_predicted_drift": round(mean_pred, 4),
                "empirical_mean_drift": round(float(np.mean(y[batch])), 4),
                "classified": regime,
                "truth": truth,
                "correct": bool(correct),
                "selected_policy_f1": f1,
            }
        row["both_correct"] = all(c["correct"] for c in row["corpora"].values())
        per_seed.append(row)

        s, t = row["corpora"]["synthetic_lowdrift"], row["corpora"]["timevarying"]
        print(f"\n[seed {seed}] τ_regime={tau:.3f}")
        print(f"  synthetic-low: δ̂̄={s['mean_predicted_drift']:.3f} "
              f"-> {s['classified']:<4} (truth {s['truth']:<4}, "
              f"{'OK' if s['correct'] else 'MISS'})")
        print(f"  timevarying:   δ̂̄={t['mean_predicted_drift']:.3f} "
              f"-> {t['classified']:<4} (truth {t['truth']:<4}, "
              f"{'OK' if t['correct'] else 'MISS'})")

    acc = sum(r["both_correct"] for r in per_seed) / len(per_seed)
    meta_f1_syn = float(np.mean([r["corpora"]["synthetic_lowdrift"]["selected_policy_f1"]
                                 for r in per_seed]))
    meta_f1_tv  = float(np.mean([r["corpora"]["timevarying"]["selected_policy_f1"]
                                 for r in per_seed]))
    print(f"\n=== Regime-classification accuracy over {len(SEEDS)} seeds: "
          f"{acc:.0%} ===")
    print(f"Meta-router mean F1 — synthetic-low corpus: {meta_f1_syn:.3f} "
          f"(best fixed 0.192)")
    print(f"Meta-router mean F1 — TimeVarying:          {meta_f1_tv:.3f} "
          f"(best fixed 0.705 at 3x cost)")

    out = {
        "protocol": "out-of-fold batches (TV n=20, synthetic n=12), 5 seeds",
        "n_train_total": len(X),
        "n_batch": {"timevarying": N_BATCH_TV, "synthetic_lowdrift": N_BATCH_SYN},
        "per_seed": per_seed,
        "summary": {
            "regime_accuracy_both_corpora": acc,
            "meta_router_f1_synthetic_lowdrift": meta_f1_syn,
            "meta_router_f1_timevarying": meta_f1_tv,
            "reference_best_fixed_f1": {
                "synthetic_lowdrift": 0.192,
                "timevarying": 0.705,
            },
        },
    }
    out_path = ROOT / "results" / "regime_classifier.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot: per-seed (predicted, empirical) corpus points + threshold
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        for r in per_seed:
            s = r["corpora"]["synthetic_lowdrift"]
            t = r["corpora"]["timevarying"]
            ax.scatter(s["mean_predicted_drift"], s["empirical_mean_drift"],
                       c="C0", s=50, alpha=0.7)
            ax.scatter(t["mean_predicted_drift"], t["empirical_mean_drift"],
                       c="C3", s=50, alpha=0.7)
        ax.scatter([], [], c="C0", label="synthetic low-drift")
        ax.scatter([], [], c="C3", label="TimeVarying (high-drift)")
        taus = [r["tau_regime"] for r in per_seed]
        ax.axvspan(min(taus), max(taus), color="gray", alpha=0.2,
                   label=r"$\tau_{\mathrm{regime}}$ range")
        ax.set_xlabel(r"mean predicted drift $\overline{\hat\delta}$ (held-out batch)")
        ax.set_ylabel(r"empirical mean drift $\bar\delta$")
        ax.set_title(f"Corpus-level regime classifier ({len(SEEDS)} seeds)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = ROOT / "figures" / "fig_regime_classifier.pdf"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved: {fig_path}")
    except ImportError:
        print("(matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()