"""
STEP 9: MLflow experiment tracking.

What is MLflow?
    An open-source platform for managing the ML lifecycle.
    It logs every training run with:
    - Parameters:  TFIDF_MAX_FEATURES=10000, SGD_ALPHA=0.0001, etc.
    - Metrics:     accuracy=0.97, f1=0.89, roc_auc=0.98, etc.
    - Artifacts:   the trained model files (.pkl, .onnx)

    Think of it as "version control for experiments." When you tweak a
    hyperparameter and retrain, MLflow tracks what changed and what
    the effect was — so you never lose track of your best run.

What this script does:
    1. Wraps the step4_train training loop with MLflow logging
    2. Logs all hyperparameters from config.py
    3. Logs validation + test metrics
    4. Logs model artifacts (vectorizer, classifier, ONNX)
    5. Optionally registers the model if it beats the current best

How to run:
    cd spamwork
    python step9_mlflow.py

    # Then view the MLflow dashboard:
    mlflow ui --port 5000
    # Open http://localhost:5000 in your browser

Dependencies:
    pip install mlflow
"""
import sys
import time
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, roc_auc_score
)

import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).parent))
import config


def train_with_tracking(experiment_name: str = "sms-spam-classifier"):
    """
    Train TF-IDF + SGD with full MLflow logging.

    This is step4_train.py but wrapped with experiment tracking.
    Every parameter, metric, and artifact gets logged.
    """

    print("=" * 50)
    print("STEP 9: Training with MLflow Tracking")
    print("=" * 50)

    # ── Set up MLflow ──
    # Store tracking data locally inside spamwork/mlruns/
    mlflow.set_tracking_uri(f"file:///{config.WORK_DIR / 'mlruns'}")
    mlflow.set_experiment(experiment_name)

    print(f"  Experiment: {experiment_name}")
    print(f"  Tracking:   {config.WORK_DIR / 'mlruns'}")

    # ── Load data ──
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_val = pd.read_csv(config.VAL_FILE)
    df_test = pd.read_csv(config.TEST_FILE)

    X_train_text = df_train['text_clean'].fillna('')
    y_train = df_train['label'].values
    X_val_text = df_val['text_clean'].fillna('')
    y_val = df_val['label'].values
    X_test_text = df_test['text_clean'].fillna('')
    y_test = df_test['label'].values

    print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}  |  Test: {len(df_test):,}")

    # ── Start MLflow run ──
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n  MLflow Run ID: {run_id}")

        # ── Log hyperparameters ──
        params = {
            "tfidf_max_features": config.TFIDF_MAX_FEATURES,
            "tfidf_ngram_range": str(config.TFIDF_NGRAM_RANGE),
            "tfidf_min_df": config.TFIDF_MIN_DF,
            "tfidf_max_df": config.TFIDF_MAX_DF,
            "tfidf_sublinear_tf": config.TFIDF_SUBLINEAR_TF,
            "sgd_loss": config.SGD_LOSS,
            "sgd_alpha": config.SGD_ALPHA,
            "sgd_max_iter": config.SGD_MAX_ITER,
            "sgd_random_state": config.SGD_RANDOM_STATE,
            "class_weight": "balanced",
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
        }
        mlflow.log_params(params)
        print("  Logged parameters ✓")

        # ── Fit TF-IDF ──
        print("\n  Fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
            strip_accents='unicode',
        )

        t0 = time.time()
        X_train = vectorizer.fit_transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)
        X_test = vectorizer.transform(X_test_text)
        tfidf_time = time.time() - t0

        vocab_size = len(vectorizer.vocabulary_)
        print(f"  TF-IDF shape: {X_train.shape}  ({tfidf_time:.2f}s)")
        mlflow.log_metric("vocab_size", vocab_size)
        mlflow.log_metric("tfidf_time_s", round(tfidf_time, 2))

        # ── Train classifier ──
        print("  Training SGDClassifier...")
        clf = SGDClassifier(
            loss=config.SGD_LOSS,
            alpha=config.SGD_ALPHA,
            max_iter=config.SGD_MAX_ITER,
            random_state=config.SGD_RANDOM_STATE,
            class_weight='balanced',
        )

        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.2f}s")
        mlflow.log_metric("train_time_s", round(train_time, 2))

        # ── Evaluate on Validation Set ──
        y_val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        try:
            val_auc = roc_auc_score(y_val, clf.decision_function(X_val))
        except ValueError:
            val_auc = 0.0

        mlflow.log_metrics({
            "val_accuracy": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "val_roc_auc": round(val_auc, 4),
        })

        print(f"\n  VALIDATION: acc={val_acc:.4f}  f1={val_f1:.4f}  auc={val_auc:.4f}")

        # ── Evaluate on Test Set ──
        y_test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        try:
            test_auc = roc_auc_score(y_test, clf.decision_function(X_test))
        except ValueError:
            test_auc = 0.0

        mlflow.log_metrics({
            "test_accuracy": round(test_acc, 4),
            "test_f1": round(test_f1, 4),
            "test_roc_auc": round(test_auc, 4),
        })

        print(f"  TEST:       acc={test_acc:.4f}  f1={test_f1:.4f}  auc={test_auc:.4f}")

        # ── Print classification reports ──
        print("\n" + "=" * 50)
        print("VALIDATION REPORT")
        print("=" * 50)
        print(classification_report(y_val, y_val_pred, target_names=['ham', 'spam']))

        print("=" * 50)
        print("TEST REPORT")
        print("=" * 50)
        print(classification_report(y_test, y_test_pred, target_names=['ham', 'spam']))

        # ── Save model artifacts ──
        vec_path = config.MODEL_DIR / "tfidf_vectorizer.pkl"
        clf_path = config.MODEL_DIR / "classifier.pkl"

        with open(vec_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(clf_path, 'wb') as f:
            pickle.dump(clf, f)

        # ── Log artifacts to MLflow ──
        mlflow.log_artifact(str(vec_path))
        mlflow.log_artifact(str(clf_path))

        # Log ONNX model if it exists
        onnx_path = config.ONNX_MODEL_PATH
        if onnx_path.exists():
            mlflow.log_artifact(str(onnx_path))
            mlflow.log_metric("onnx_size_kb",
                              round(onnx_path.stat().st_size / 1024, 1))

        # Log the sklearn model directly (enables MLflow model serving)
        mlflow.sklearn.log_model(clf, "classifier_model")
        print("\n  Artifacts logged to MLflow ✓")

        # ── Tag the run ──
        mlflow.set_tags({
            "model_type": "SGDClassifier",
            "vectorizer": "TfidfVectorizer",
            "dataset": "SMS Spam Collection",
            "pipeline_step": "step9",
        })

        print(f"\n  Run complete: {run_id}")
        print(f"  View at: mlflow ui --port 5000")

    return {
        "run_id": run_id,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_roc_auc": val_auc,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_roc_auc": test_auc,
    }


def run_experiment_sweep():
    """
    Run multiple experiments with different hyperparameters.

    This demonstrates why MLflow is useful — you train with different
    settings and compare results in the dashboard.
    """

    print("=" * 60)
    print("HYPERPARAMETER SWEEP (3 configurations)")
    print("=" * 60)

    # Override config values temporarily for each experiment
    original_alpha = config.SGD_ALPHA
    original_max_features = config.TFIDF_MAX_FEATURES
    original_min_df = config.TFIDF_MIN_DF

    experiments = [
        {"name": "baseline",    "alpha": 1e-4,  "max_features": 10_000, "min_df": 2},
        {"name": "high-reg",    "alpha": 1e-3,  "max_features": 10_000, "min_df": 2},
        {"name": "large-vocab", "alpha": 1e-4,  "max_features": 20_000, "min_df": 1},
    ]

    results = []
    for exp in experiments:
        print(f"\n{'─' * 60}")
        print(f"  Experiment: {exp['name']}")
        print(f"  alpha={exp['alpha']}, max_features={exp['max_features']}, min_df={exp['min_df']}")
        print(f"{'─' * 60}")

        # Temporarily override config
        config.SGD_ALPHA = exp["alpha"]
        config.TFIDF_MAX_FEATURES = exp["max_features"]
        config.TFIDF_MIN_DF = exp["min_df"]

        result = train_with_tracking(experiment_name="sms-spam-sweep")
        result["experiment"] = exp["name"]
        results.append(result)

    # Restore original config
    config.SGD_ALPHA = original_alpha
    config.TFIDF_MAX_FEATURES = original_max_features
    config.TFIDF_MIN_DF = original_min_df

    # Print comparison table
    print("\n" + "=" * 60)
    print("SWEEP RESULTS COMPARISON")
    print("=" * 60)
    print(f"  {'Experiment':<15s} {'Val F1':>8s} {'Test F1':>8s} {'Test AUC':>10s}")
    print(f"  {'─' * 45}")
    for r in results:
        print(f"  {r['experiment']:<15s} {r['val_f1']:>8.4f} {r['test_f1']:>8.4f} {r['test_roc_auc']:>10.4f}")

    best = max(results, key=lambda r: r['test_f1'])
    print(f"\n  Best: {best['experiment']} (Test F1 = {best['test_f1']:.4f})")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 9: MLflow tracking")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run a 3-config hyperparameter sweep instead of a single run"
    )
    args = parser.parse_args()

    if args.sweep:
        run_experiment_sweep()
    else:
        train_with_tracking()

    print("\n+ Step 9 complete!")
