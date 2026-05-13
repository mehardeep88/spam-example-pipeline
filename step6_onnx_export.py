"""
STEP 6: Export the trained model to ONNX format.

What is ONNX?
    Open Neural Network Exchange — a universal model format.
    - Trained in Python (sklearn) → exported to ONNX → runs anywhere
    - 2-5x faster inference than native sklearn
    - Can be served by ONNX Runtime, TensorRT, or any ONNX-compatible runtime
    - Language-agnostic: run from C++, Java, C#, etc.

What this script does:
    1. Loads the trained TF-IDF vectorizer and SGDClassifier
    2. Converts them to ONNX format using skl2onnx
    3. Validates the ONNX model produces same outputs as sklearn
    4. Saves the .onnx file

Dependencies:
    pip install skl2onnx onnxruntime
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
import config


def export_onnx():
    """Convert sklearn model to ONNX and validate."""

    print("=" * 50)
    print("STEP 6: ONNX Export")
    print("=" * 50)

    # ── Load sklearn artifacts ──
    with open(config.MODEL_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    with open(config.MODEL_DIR / "classifier.pkl", 'rb') as f:
        clf = pickle.load(f)

    vocab_size = len(vectorizer.vocabulary_)
    print(f"  Vectorizer vocab: {vocab_size:,}")
    print(f"  Classifier type:  {type(clf).__name__}")

    # ── Convert classifier to ONNX ──
    # We export only the classifier (not TF-IDF) because:
    # - TF-IDF is a sparse transform, tricky in ONNX
    # - At inference time: Python TF-IDF → dense array → ONNX classifier
    # - This is the standard pattern for sklearn + ONNX

    print("\n  Converting to ONNX...")
    initial_type = [('X', FloatTensorType([None, vocab_size]))]
    onnx_model = to_onnx(clf, initial_types=initial_type)

    # Save
    onnx_path = config.ONNX_MODEL_PATH
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  Saved: {onnx_path.name} ({size_kb:.0f} KB)")

    # ── Validate: sklearn vs ONNX should match ──
    print("\n  Validating ONNX output matches sklearn...")
    df_val = pd.read_csv(config.VAL_FILE)
    X_val = vectorizer.transform(df_val['text_clean'].fillna(''))
    X_val_dense = X_val.toarray().astype(np.float32)

    # sklearn predictions
    sk_pred = clf.predict(X_val_dense[:10])

    # ONNX predictions
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    onnx_pred = sess.run(None, {input_name: X_val_dense[:10]})[0]

    match = np.array_equal(sk_pred, onnx_pred)
    print(f"  sklearn preds: {sk_pred.tolist()}")
    print(f"  ONNX preds:    {onnx_pred.tolist()}")
    print(f"  Match: {'YES' if match else 'NO — investigate!'}")

    # Also save the vectorizer path info for inference
    print(f"\n  For inference you need:")
    print(f"    1. {config.MODEL_DIR / 'tfidf_vectorizer.pkl'}")
    print(f"    2. {onnx_path}")

    return onnx_path


if __name__ == "__main__":
    export_onnx()
    print("\n+ Step 6 complete!")
