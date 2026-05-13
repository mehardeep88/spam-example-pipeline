"""
STEP 5: Detailed evaluation on the test set.

Why a separate evaluation step?
    - Step 4 evaluated on VALIDATION set (used to tune the model)
    - This step evaluates on TEST set (never seen during training or tuning)
    - Test metrics = the "real" performance you report in your paper/presentation

What this script produces:
    1. Classification report (precision, recall, F1 per class)
    2. Confusion matrix visualization
    3. Confidence distribution analysis
    4. Error analysis (what the model gets wrong)
"""
import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score, roc_curve
)

sys.path.insert(0, str(Path(__file__).parent))
import config


def evaluate():
    """Run full evaluation on the held-out test set."""

    print("=" * 50)
    print("STEP 5: Test Set Evaluation")
    print("=" * 50)

    # ── Load model and data ──
    with open(config.MODEL_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    with open(config.MODEL_DIR / "classifier.pkl", 'rb') as f:
        clf = pickle.load(f)

    df_test = pd.read_csv(config.TEST_FILE)
    X_test = vectorizer.transform(df_test['text_clean'].fillna(''))
    y_test = df_test['label'].values

    print(f"  Test samples: {len(df_test):,}")

    # ── Predictions ──
    y_pred = clf.predict(X_test)
    # decision_function gives confidence scores
    y_scores = clf.decision_function(X_test)

    # ── Classification Report ──
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    try:
        auc = roc_auc_score(y_test, y_scores)
        print(f"  ROC AUC:  {auc:.4f}")
    except ValueError:
        auc = None

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted Ham  Predicted Spam")
    print(f"  Actual Ham   {cm[0][0]:>12,}  {cm[0][1]:>14,}")
    print(f"  Actual Spam  {cm[1][0]:>12,}  {cm[1][1]:>14,}")

    # ── Error Analysis ──
    errors = df_test[y_test != y_pred].copy()
    errors['predicted'] = y_pred[y_test != y_pred]
    print(f"\n  Total errors: {len(errors)} / {len(df_test)} "
          f"({len(errors)/len(df_test)*100:.1f}%)")

    if len(errors) > 0:
        # False positives: ham classified as spam
        fp = errors[errors['predicted'] == 1]
        print(f"\n  False Positives (ham marked spam): {len(fp)}")
        for _, row in fp.head(3).iterrows():
            print(f"    -> {row['text'][:80]}")

        # False negatives: spam classified as ham
        fn = errors[errors['predicted'] == 0]
        print(f"\n  False Negatives (spam marked ham): {len(fn)}")
        for _, row in fn.head(3).iterrows():
            print(f"    -> {row['text'][:80]}")

    # ── Save plots ──
    _save_eval_plots(cm, y_test, y_scores, auc)

    return {'accuracy': acc, 'f1': f1, 'auc': auc, 'confusion_matrix': cm}


def _save_eval_plots(cm, y_test, y_scores, auc):
    """Save confusion matrix and ROC curve plots."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Test Set Evaluation', fontsize=14, fontweight='bold')

    # Confusion matrix heatmap
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Ham', 'Spam'])
    ax.set_yticklabels(['Ham', 'Spam'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')

    # ROC curve
    ax = axes[1]
    if auc is not None:
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC = {auc:.3f}')
        ax.plot([0,1], [0,1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'ROC not available', ha='center', va='center')

    plt.tight_layout()
    path = config.PLOTS_DIR / "evaluation.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plots saved to: {path}")


if __name__ == "__main__":
    evaluate()
    print("\n+ Step 5 complete!")
