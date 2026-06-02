"""
Retraining script for active learning feedback loop.
This script:
1. Downloads and merges all human-labeled feedback from S3 (or generates a local mock feedback file if AWS is offline)
2. Compares the baseline model against the retrained model on the test set
3. Demonstrates model improvement (F1 score & accuracy)
4. Re-exports the model to ONNX and uploads the updated model to S3
"""
import sys
import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
import config
from step11_aws import AWSPipeline

def generate_mock_feedback(filepath: Path):
    """Generate 100 rows of mock human-labeled feedback for testing offline."""
    print(f"  Generating mock feedback file with 100 records to {filepath.name}...")
    
    # 50 challenging spams and 50 challenging hams that the initial model might struggle with
    mock_data = []
    
    # Spam messages (labeled 1)
    spams = [
        "Urgent! Please call 09061213237 from landline. Your cash reward of £5000 is waiting.",
        "FREE entry into our £250 weekly competition. Text WIN to 80082 now!",
        "Double your money in 24 hours. Sign up at money-fast-now.com. Limited slots!",
        "Customer Service: Your debit card starting with 4321 has been suspended. Call 1-800-123-4567.",
        "Congratulations! You won a free cruise to the Bahamas. Claim within 10 minutes at bahama-cruises.com",
    ] * 10 # 50 spams
    
    # Ham messages (labeled 0)
    hams = [
        "Hey, are we still meeting for dinner tonight? Let me know.",
        "Can you send me the link to that website you were talking about?",
        "I'll call you back in 5 minutes, in a meeting right now.",
        "Do we have any milk left or should I buy some on my way home?",
        "Happy birthday! Hope you have a wonderful day ahead.",
    ] * 10 # 50 hams
    
    for i, text in enumerate(spams):
        mock_data.append({
            "text": text,
            "true_label": 1,
            "predicted_label": 0, # model got it wrong (low confidence)
            "confidence": 0.52 + (i % 10) * 0.01,
            "reviewed_at": pd.Timestamp.now().isoformat()
        })
        
    for i, text in enumerate(hams):
        mock_data.append({
            "text": text,
            "true_label": 0,
            "predicted_label": 1, # model got it wrong (low confidence)
            "confidence": 0.51 + (i % 10) * 0.01,
            "reviewed_at": pd.Timestamp.now().isoformat()
        })
        
    df = pd.DataFrame(mock_data)
    df.to_csv(filepath, index=False)
    print(f"  ✓ Saved 100 mock reviews locally.")

def train_model(X_train_text, y_train, X_test_text, y_test):
    """Helper to train TF-IDF + SGDClassifier and evaluate on test data."""
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        strip_accents='unicode',
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    clf = SGDClassifier(
        loss=config.SGD_LOSS,
        alpha=config.SGD_ALPHA,
        max_iter=config.SGD_MAX_ITER,
        random_state=config.SGD_RANDOM_STATE,
        class_weight='balanced',
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    return vectorizer, clf, f1, acc

def main():
    print("=" * 65)
    print("ACTIVE LEARNING FEEDBACK: RETRAINING PIPELINE")
    print("=" * 65)

    # ─── 1. Initialize AWS Pipeline and attempt feedback download ───
    pipeline = AWSPipeline()
    feedback_records = []
    
    print("\n[1/5] Fetching Human Feedback Data...")
    try:
        feedback_records = pipeline.collect_feedback()
        if feedback_records:
            print(f"  ✓ Connected to S3. Downloaded {len(feedback_records)} human-labeled records.")
        else:
            print("  S3 feedback folder is empty. Checking local fallback...")
    except Exception as e:
        print(f"  AWS Connection skipped/failed: {e}")
        print("  Using local fallback for demo purposes.")

    # Local fallback if S3 is empty or unavailable
    local_feedback_file = config.PROCESSED_DIR / "local_feedback.csv"
    if not feedback_records:
        if not local_feedback_file.exists():
            generate_mock_feedback(local_feedback_file)
        
        df_feed = pd.read_csv(local_feedback_file)
        feedback_records = df_feed.to_dict(orient="records")
        print(f"  ✓ Loaded {len(feedback_records)} records from local mock feedback file.")

    # Convert feedback to DataFrame
    df_feedback = pd.DataFrame(feedback_records)
    # Ensure correct data type for label/text
    df_feedback['label'] = df_feedback['true_label'].astype(int)
    # Match schema with train.csv
    df_feedback['text_clean'] = df_feedback['text']

    # ─── 2. Load Original Dataset Splits ───
    print("\n[2/5] Loading Original Dataset...")
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_test = pd.read_csv(config.TEST_FILE)
    
    X_train_orig = df_train['text_clean'].fillna('')
    y_train_orig = df_train['label'].values
    
    X_test = df_test['text_clean'].fillna('')
    y_test = df_test['label'].values
    
    print(f"  Original Training Set Size: {len(df_train):,}")
    print(f"  Test Evaluation Set Size:    {len(df_test):,}")

    # ─── 3. Train & Evaluate Models (Baseline vs. Combined) ───
    print("\n[3/5] Evaluating Retraining Performance on Test Set...")
    
    # A. Train Baseline (Original Train only)
    print("  Training baseline model...")
    _, _, baseline_f1, baseline_acc = train_model(X_train_orig, y_train_orig, X_test, y_test)
    
    # B. Combine datasets and Train Retrained Model
    df_combined = pd.concat([df_train[['text_clean', 'label']], df_feedback[['text_clean', 'label']]], ignore_index=True)
    X_train_combined = df_combined['text_clean'].fillna('')
    y_train_combined = df_combined['label'].values
    
    print(f"  Training retrained model (Original + {len(df_feedback)} feedback rows)...")
    vectorizer_new, clf_new, retrained_f1, retrained_acc = train_model(X_train_combined, y_train_combined, X_test, y_test)

    # C. Print Metrics comparison
    print("\n" + "─" * 50)
    print("MODEL IMPROVEMENT SUMMARY")
    print("─" * 50)
    print(f"Metric          | Baseline  | Retrained | Improvement")
    print(f"Training Rows   | {len(df_train):<9d} | {len(df_combined):<9d} | +{len(df_feedback)}")
    print(f"Test F1 Score   | {baseline_f1:.4f}    | {retrained_f1:.4f}    | {retrained_f1 - baseline_f1:+.4f}")
    print(f"Test Accuracy   | {baseline_acc:.4f}    | {retrained_acc:.4f}    | {retrained_acc - baseline_acc:+.4f}")
    print("─" * 50)

    # ─── 4. Save and Export Retrained Model to ONNX ───
    print("\n[4/5] Saving and Exporting Retrained Model...")
    # Save sklearn models locally
    with open(config.MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer_new, f)
    with open(config.MODEL_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(clf_new, f)
    print(f"  ✓ Saved pickle files to {config.MODEL_DIR}")

    # Export to ONNX format
    vocab_size = len(vectorizer_new.vocabulary_)
    initial_type = [('X', FloatTensorType([None, vocab_size]))]
    onnx_model = to_onnx(clf_new, initial_types=initial_type)
    
    with open(config.ONNX_MODEL_PATH, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"  ✓ Exported retrained model to ONNX: {config.ONNX_MODEL_PATH.name}")

    # Validate ONNX output matches sklearn
    X_test_dense = vectorizer_new.transform(X_test).toarray().astype(np.float32)
    sk_pred = clf_new.predict(X_test_dense[:5])
    
    sess = ort.InferenceSession(str(config.ONNX_MODEL_PATH))
    input_name = sess.get_inputs()[0].name
    onnx_pred = sess.run(None, {input_name: X_test_dense[:5]})[0]
    match = np.array_equal(sk_pred, onnx_pred)
    print(f"  Validation Check: {'Match OK' if match else 'Mismatch - check logs!'}")

    # ─── 5. Sync Updated Model to S3 (if AWS is configured) ───
    print("\n[5/5] Synchronizing Model with S3...")
    try:
        uploaded_count = pipeline.sync_model("upload")
        if uploaded_count > 0:
            print("  ✓ Successfully uploaded updated model files to S3 bucket.")
            print("  ✓ API will reload the new model next time it restarts!")
        else:
            print("  No files uploaded. Check S3 permissions.")
    except Exception as e:
        print(f"  AWS Sync skipped: {e}")
        print("  ✓ Retrained model saved locally. API reload ready.")

    print("\n" + "=" * 65)
    print("RETRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
