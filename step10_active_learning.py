"""
STEP 10: Active Learning simulation.

What is Active Learning?
    Instead of labeling random samples, you let the model pick which
    samples it's most confused about → a human labels those → the model
    retrains. This gives better accuracy with FEWER labeled samples.

The core question:
    "Does selectively labeling uncertain samples improve the model
     faster than randomly labeling samples?"

    If yes → your thesis is proven: active learning saves annotation
    effort while achieving equal or better accuracy.

How the simulation works:
    1. Start with the full training set, but pretend only 20% is labeled
    2. Train an initial model on that 20% (the "seed")
    3. Each round:
       - AL branch:     model predicts on unlabeled pool → pick top-N most
                        uncertain → "label" them (reveal ground truth) → retrain
       - Random branch: pick N random from pool → "label" → retrain
    4. After each round, evaluate both branches on the test set
    5. Plot learning curves: F1 vs number of labeled samples
    6. If AL's line is above Random's line → thesis proven ✓

What this script produces:
    - Console output:  per-round metrics for both strategies
    - Plot:            plots/active_learning_curves.png
    - Summary:         final comparison table

Dependencies:
    No extra deps beyond what step4 uses (sklearn, matplotlib, pandas).
"""
import sys
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import config


# ─── Uncertainty Scoring Functions ───────────────────────

def entropy_score(probabilities: np.ndarray) -> np.ndarray:
    """
    Entropy-based uncertainty.

    High entropy = model is confused (probabilities are spread out).
    Low entropy  = model is confident (one probability dominates).

    Formula: H(p) = -Σ p * log(p)

    For binary classification:
        - Confident:  P(spam)=0.95, P(ham)=0.05 → entropy ≈ 0.20
        - Uncertain:  P(spam)=0.52, P(ham)=0.48 → entropy ≈ 0.99
    """
    # Clip to avoid log(0)
    p = np.clip(probabilities, 1e-10, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)
    # Normalize to [0, 1] by dividing by log(num_classes)
    max_entropy = np.log(p.shape[1])
    return entropy / max_entropy


def max_prob_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """
    Max-probability uncertainty: 1 - max(p).

    Simple and intuitive:
        - P(spam)=0.95 → uncertainty = 0.05 (confident)
        - P(spam)=0.52 → uncertainty = 0.48 (uncertain)
    """
    return 1.0 - np.max(probabilities, axis=1)


def margin_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """
    Margin uncertainty: 1 - (p_top1 - p_top2).

    Measures how close the top two predictions are:
        - P=[0.9, 0.1] → margin = 0.8 → uncertainty = 0.2 (confident)
        - P=[0.52, 0.48] → margin = 0.04 → uncertainty = 0.96 (very uncertain)
    """
    sorted_p = np.sort(probabilities, axis=1)[:, ::-1]
    margins = sorted_p[:, 0] - sorted_p[:, 1]
    return 1.0 - margins


# ─── Active Learning Simulation ─────────────────────────

def run_simulation(
    seed_fraction: float = 0.20,
    batch_size: int = 100,
    n_iterations: int = 20,
    uncertainty_method: str = "entropy",
):
    """
    Run AL vs Random comparison.

    Args:
        seed_fraction: What fraction of training data to start with (0.20 = 20%)
        batch_size: How many samples to "label" per round
        n_iterations: How many rounds to simulate
        uncertainty_method: "entropy", "max_prob", or "margin"

    Returns:
        dict with learning curves and summary metrics
    """

    print("=" * 60)
    print("STEP 10: Active Learning Simulation")
    print("=" * 60)
    print(f"  Seed fraction:    {seed_fraction:.0%}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Iterations:       {n_iterations}")
    print(f"  Uncertainty:      {uncertainty_method}")

    # ── Load data ──
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_test = pd.read_csv(config.TEST_FILE)

    X_all_text = df_train['text_clean'].fillna('').values
    y_all = df_train['label'].values
    X_test_text = df_test['text_clean'].fillna('').values
    y_test = df_test['label'].values

    print(f"  Train pool: {len(df_train):,}  |  Test: {len(df_test):,}")

    # ── Select uncertainty function ──
    unc_funcs = {
        "entropy": entropy_score,
        "max_prob": max_prob_uncertainty,
        "margin": margin_uncertainty,
    }
    unc_fn = unc_funcs[uncertainty_method]

    # ── Fit TF-IDF on ALL text (vocabulary is fixed) ──
    # In a real scenario, you'd also consider updating the vocabulary
    # as new data comes in. For this simulation, we fix it.
    print("\n  Fitting TF-IDF on full corpus (fixed vocabulary)...")
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        strip_accents='unicode',
    )
    X_all = vectorizer.fit_transform(X_all_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # ── Split into seed (labeled) and pool (unlabeled) ──
    n_total = len(y_all)
    n_seed = int(n_total * seed_fraction)

    rng = np.random.RandomState(config.RANDOM_SEED)
    all_indices = rng.permutation(n_total)
    seed_indices = set(all_indices[:n_seed].tolist())
    pool_indices = set(all_indices[n_seed:].tolist())

    print(f"  Seed: {len(seed_indices):,}  |  Pool: {len(pool_indices):,}")

    # ── Helper: train and evaluate ──
    def train_and_eval(labeled_idx):
        """Train a fresh model on labeled indices, evaluate on test."""
        idx = sorted(labeled_idx)
        X_sub = X_all[idx]
        y_sub = y_all[idx]

        clf = SGDClassifier(
            loss=config.SGD_LOSS,
            alpha=config.SGD_ALPHA,
            max_iter=config.SGD_MAX_ITER,
            random_state=config.SGD_RANDOM_STATE,
            class_weight='balanced',
        )
        clf.fit(X_sub, y_sub)

        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        return clf, f1, acc

    # ── Baseline: train on seed ──
    print("\n  Training baseline model on seed...")
    _, seed_f1, seed_acc = train_and_eval(seed_indices)
    print(f"  Baseline — F1: {seed_f1:.4f}  Acc: {seed_acc:.4f}")

    # ── Run both strategies ──
    # Deep copy the index sets so AL and Random start from the same state
    al_labeled = set(seed_indices)
    al_pool = set(pool_indices)
    rand_labeled = set(seed_indices)
    rand_pool = set(pool_indices)

    # Track metrics
    al_f1_curve = [seed_f1]
    al_acc_curve = [seed_acc]
    rand_f1_curve = [seed_f1]
    rand_acc_curve = [seed_acc]
    n_labeled_curve = [len(seed_indices)]

    t0 = time.time()

    for iteration in range(1, n_iterations + 1):
        # ── Active Learning: select most uncertain from pool ──
        al_pool_list = sorted(al_pool)
        if len(al_pool_list) == 0:
            break

        # Get current model's probabilities on the unlabeled pool
        al_clf, _, _ = train_and_eval(al_labeled)
        X_pool = X_all[al_pool_list]
        probas = al_clf.predict_proba(X_pool)
        uncertainties = unc_fn(probas)

        # Pick top-N most uncertain
        n_select = min(batch_size, len(al_pool_list))
        top_indices = np.argsort(uncertainties)[-n_select:][::-1]
        selected_al = [al_pool_list[i] for i in top_indices]

        # "Label" them: move from pool to labeled
        for idx in selected_al:
            al_labeled.add(idx)
            al_pool.discard(idx)

        # Retrain and evaluate
        _, al_f1, al_acc = train_and_eval(al_labeled)

        # ── Random: select random from pool ──
        rand_pool_list = sorted(rand_pool)
        n_select_r = min(batch_size, len(rand_pool_list))
        selected_rand = rng.choice(rand_pool_list, size=n_select_r, replace=False)

        for idx in selected_rand:
            rand_labeled.add(idx)
            rand_pool.discard(idx)

        _, rand_f1, rand_acc = train_and_eval(rand_labeled)

        # Record
        al_f1_curve.append(al_f1)
        al_acc_curve.append(al_acc)
        rand_f1_curve.append(rand_f1)
        rand_acc_curve.append(rand_acc)
        n_labeled_curve.append(len(al_labeled))

        delta = al_f1 - rand_f1
        marker = "▲" if delta > 0 else "▼" if delta < 0 else "="
        print(
            f"  [{iteration:>2d}/{n_iterations}]  "
            f"Labeled: {len(al_labeled):>5,}  |  "
            f"AL F1: {al_f1:.4f}  Random F1: {rand_f1:.4f}  "
            f"Δ: {delta:+.4f} {marker}"
        )

    elapsed = time.time() - t0
    print(f"\n  Simulation time: {elapsed:.1f}s")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    final_al_f1 = al_f1_curve[-1]
    final_rand_f1 = rand_f1_curve[-1]
    avg_delta = np.mean(
        [a - r for a, r in zip(al_f1_curve[1:], rand_f1_curve[1:])]
    )

    print(f"  Baseline F1 (seed only):   {seed_f1:.4f}")
    print(f"  Final AL F1:               {final_al_f1:.4f}")
    print(f"  Final Random F1:           {final_rand_f1:.4f}")
    print(f"  Average F1 advantage:      {avg_delta:+.4f}")
    print(f"  AL wins in {sum(1 for a, r in zip(al_f1_curve[1:], rand_f1_curve[1:]) if a > r)} / {n_iterations} rounds")

    if avg_delta > 0:
        print(f"\n  ✓ Active Learning outperforms Random Sampling!")
        print(f"    → The model learns faster when it picks what to label.")
    else:
        print(f"\n  ✗ Random sampling matched or beat AL.")
        print(f"    → Try: smaller seed, larger batch, or 'margin' uncertainty.")

    # ── Save plot ──
    _save_learning_curves(
        n_labeled_curve, al_f1_curve, rand_f1_curve,
        al_acc_curve, rand_acc_curve, seed_f1,
        uncertainty_method, batch_size, seed_fraction,
    )

    return {
        "n_labeled": n_labeled_curve,
        "al_f1": al_f1_curve,
        "rand_f1": rand_f1_curve,
        "al_acc": al_acc_curve,
        "rand_acc": rand_acc_curve,
        "avg_delta": avg_delta,
        "seed_f1": seed_f1,
    }


def _save_learning_curves(
    n_labeled, al_f1, rand_f1, al_acc, rand_acc,
    seed_f1, method, batch_size, seed_fraction,
):
    """Generate and save the AL vs Random comparison plot."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Active Learning vs Random Sampling',
        fontsize=16, fontweight='bold', y=1.02,
    )

    # ── F1 Score Curves ──
    ax = axes[0]
    ax.plot(n_labeled, al_f1, 'o-', color='#2980b9', linewidth=2,
            markersize=5, label='Active Learning', zorder=5)
    ax.plot(n_labeled, rand_f1, 's--', color='#e74c3c', linewidth=2,
            markersize=5, label='Random Sampling', zorder=5)

    # Shade the AL advantage
    al_arr = np.array(al_f1)
    rand_arr = np.array(rand_f1)
    ax.fill_between(
        n_labeled, al_arr, rand_arr,
        where=(al_arr > rand_arr),
        alpha=0.15, color='#2980b9', label='AL advantage',
    )

    ax.axhline(y=seed_f1, color='gray', linestyle=':', alpha=0.5, label='Baseline (seed)')
    ax.set_xlabel('Number of Labeled Samples', fontsize=12)
    ax.set_ylabel('F1 Score (Spam)', fontsize=12)
    ax.set_title('F1 Score vs Labeled Data', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # ── Accuracy Curves ──
    ax = axes[1]
    ax.plot(n_labeled, al_acc, 'o-', color='#2980b9', linewidth=2,
            markersize=5, label='Active Learning')
    ax.plot(n_labeled, rand_acc, 's--', color='#e74c3c', linewidth=2,
            markersize=5, label='Random Sampling')
    ax.set_xlabel('Number of Labeled Samples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Labeled Data', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Subtitle with config
    fig.text(
        0.5, -0.02,
        f"Method: {method} | Batch: {batch_size} | Seed: {seed_fraction:.0%}",
        ha='center', fontsize=10, color='gray',
    )

    plt.tight_layout()
    path = config.PLOTS_DIR / "active_learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 10: Active Learning Simulation")
    parser.add_argument("--seed", type=float, default=0.20,
                        help="Fraction of data to use as initial labeled seed (default: 0.20)")
    parser.add_argument("--batch", type=int, default=100,
                        help="Samples to label per AL round (default: 100)")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of AL rounds (default: 20)")
    parser.add_argument("--method", type=str, default="entropy",
                        choices=["entropy", "max_prob", "margin"],
                        help="Uncertainty method (default: entropy)")
    args = parser.parse_args()

    run_simulation(
        seed_fraction=args.seed,
        batch_size=args.batch,
        n_iterations=args.rounds,
        uncertainty_method=args.method,
    )
    print("\n+ Step 10 complete!")
