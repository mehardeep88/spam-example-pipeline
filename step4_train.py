"""
STEP 4: Train TF-IDF + SGDClassifier model.

What is TF-IDF?
    Term Frequency-Inverse Document Frequency.
    Converts text into numbers by measuring how important each word is.
    - TF:  How often a word appears in THIS message
    - IDF: How rare the word is across ALL messages
    - "free" appears in many spam → high TF in spam, moderate IDF
    - "the" appears everywhere → high TF but very low IDF (not useful)

What is SGDClassifier?
    Stochastic Gradient Descent classifier.
    With loss='log_loss', it's basically logistic regression trained via SGD.
    Key advantage: supports partial_fit() for incremental learning later.

Pipeline:
    Raw text → TF-IDF vectorizer → feature matrix → SGDClassifier → prediction
"""
import sys
import time
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import config


def train():
    """Train TF-IDF + SGD pipeline and save artifacts."""

    print("=" * 50)
    print("STEP 4: Training Model")
    print("=" * 50)

    # ── Load splits ──
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_val = pd.read_csv(config.VAL_FILE)
    print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}")

    X_train_text = df_train['text_clean'].fillna('')
    y_train = df_train['label'].values
    X_val_text = df_val['text_clean'].fillna('')
    y_val = df_val['label'].values

    # ── Step 1: Fit TF-IDF ──
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
    tfidf_time = time.time() - t0

    print(f"  TF-IDF shape: {X_train.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"  TF-IDF fit time: {tfidf_time:.2f}s")

    # ── Step 2: Train classifier ──
    print("\n  Training SGDClassifier...")
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

    # ── Step 3: Evaluate ──
    y_pred = clf.predict(X_val)

    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(classification_report(y_val, y_pred, target_names=['ham', 'spam']))

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'vocab_size': len(vectorizer.vocabulary_),
        'train_time_s': train_time,
    }
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")

    # ── Step 4: Save model artifacts ──
    vec_path = config.MODEL_DIR / "tfidf_vectorizer.pkl"
    clf_path = config.MODEL_DIR / "classifier.pkl"

    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(clf_path, 'wb') as f:
        pickle.dump(clf, f)

    print(f"\n  Saved: {vec_path.name}, {clf_path.name}")
    print(f"  To:    {config.MODEL_DIR}")

    return vectorizer, clf, metrics


if __name__ == "__main__":
    train()
    print("\n+ Step 4 complete!")
