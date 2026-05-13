"""
STEP 7: Run inference using the ONNX model.

This is what happens in production:
    1. User sends a message via API
    2. TF-IDF vectorizer converts text to numbers
    3. ONNX Runtime runs the classifier
    4. Return prediction: spam or ham + confidence score

This script demonstrates that flow with sample messages.
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
import config


class SpamClassifier:
    """
    Production-ready spam classifier using ONNX Runtime.

    Usage:
        classifier = SpamClassifier()
        result = classifier.predict("Free entry! Call now!")
        # {'label': 'spam', 'confidence': 0.94}
    """

    def __init__(self):
        # Load TF-IDF vectorizer (still Python — converts text to numbers)
        vec_path = config.MODEL_DIR / "tfidf_vectorizer.pkl"
        with open(vec_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        # Load ONNX model (the classifier part)
        onnx_path = str(config.ONNX_MODEL_PATH)
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

        print(f"  Loaded vectorizer: {vec_path.name}")
        print(f"  Loaded ONNX model: {config.ONNX_MODEL_PATH.name}")

    def predict(self, text: str) -> dict:
        """
        Classify a single message.

        Args:
            text: The SMS message to classify

        Returns:
            dict with 'label' ('ham'/'spam'), 'prediction' (0/1),
            and 'confidence' (float)
        """
        # Step 1: Text -> TF-IDF vector
        X = self.vectorizer.transform([text])
        X_dense = X.toarray().astype(np.float32)

        # Step 2: Run ONNX inference
        outputs = self.session.run(None, {self.input_name: X_dense})
        prediction = int(outputs[0][0])

        # Step 3: Get confidence from probabilities (if available)
        confidence = 1.0
        if len(outputs) > 1:
            proba = outputs[1]
            if isinstance(proba, list):
                proba = proba[0]
            if hasattr(proba, '__len__') and len(proba) > 0:
                prob_dict = proba[0] if isinstance(proba[0], dict) else {}
                if prob_dict:
                    confidence = max(prob_dict.values())
                else:
                    confidence = float(np.max(proba))

        label = 'spam' if prediction == 1 else 'ham'
        return {
            'label': label,
            'prediction': prediction,
            'confidence': round(confidence, 4),
        }

    def predict_batch(self, texts: list) -> list:
        """Classify multiple messages at once."""
        X = self.vectorizer.transform(texts)
        X_dense = X.toarray().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: X_dense})
        predictions = outputs[0]
        return [
            {'label': 'spam' if p == 1 else 'ham', 'prediction': int(p)}
            for p in predictions
        ]


def demo():
    """Run inference on sample messages."""

    print("=" * 50)
    print("STEP 7: ONNX Inference Demo")
    print("=" * 50)

    classifier = SpamClassifier()

    test_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "WINNER! You've been selected for a free iPhone! Call NOW!",
        "Can you pick up milk on your way home?",
        "Congratulations! You won 10000 pounds! Claim at www.prize.co.uk",
        "Meeting rescheduled to 3pm. See you there.",
        "FREE entry to win a holiday! Text WIN to 80888",
        "Thanks for dinner last night, it was great!",
        "Ur cash prize of 5000 is waiting. Call 09050000 to claim",
    ]

    print(f"\n  Testing {len(test_messages)} messages:\n")
    for msg in test_messages:
        result = classifier.predict(msg)
        icon = "SPAM" if result['prediction'] == 1 else " ham"
        print(f"  [{icon}] (conf: {result['confidence']:.2f}) {msg[:65]}")

    print(f"\n  Model file: {config.ONNX_MODEL_PATH}")
    print(f"  Model size: {config.ONNX_MODEL_PATH.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    demo()
    print("\n+ Step 7 complete!")
