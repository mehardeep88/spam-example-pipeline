# Step 6 & Step 7 — Viva Explainer

> These two steps together answer one big question: **"How do you take a trained Python model and run it in production efficiently?"**
> The answer: export to ONNX (Step 6), then load and use it via ONNX Runtime (Step 7).

---

## Step 6 — `step6_onnx_export.py`

### What is ONNX? (They WILL ask this)

**ONNX = Open Neural Network Exchange.**

Think of it like a PDF for ML models. A Word document can only be opened in Word. But a PDF opens anywhere. ONNX does the same thing for models:

- You train in Python (scikit-learn)
- Export to `.onnx` file
- That file can now run in C++, Java, mobile apps, browsers — anywhere
- It also runs **2–5× faster** than native sklearn at inference time

---

### The Imports

```python
import pickle
```
> `pickle` is Python's way of saving and loading Python objects to/from files. Your trained vectorizer and classifier were saved as `.pkl` (pickle) files in Step 4. This loads them back.

```python
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
```
> `skl2onnx` = scikit-learn **to** ONNX. This is the converter library that understands sklearn's internal structure and rewrites it in the ONNX format.
> `FloatTensorType` tells the converter: "the input will be a 2D array of 32-bit floats."

```python
import onnxruntime as ort
```
> ONNX Runtime (ORT) is the engine that **runs** ONNX models. Think of it like a car engine — ONNX is the fuel, ORT is what burns it.

```python
sys.path.insert(0, str(Path(__file__).parent))
import config
```
> This adds the `spamwork/` folder to Python's search path so it can find `config.py`. `config.py` holds all paths and settings in one place.

---

### Loading the Trained Model

```python
with open(config.MODEL_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)
with open(config.MODEL_DIR / "classifier.pkl", 'rb') as f:
    clf = pickle.load(f)
```

> This loads the two objects that Step 4 saved:
> - `vectorizer` — the TF-IDF object that converts raw text into numbers
> - `clf` — the SGDClassifier (the actual spam/ham decision-maker)
>
> **Why load them separately?** Because we only export the *classifier* to ONNX, not the TF-IDF (more on this below).

---

### Why NOT convert TF-IDF to ONNX?

```python
# We export only the classifier (not TF-IDF) because:
# - TF-IDF is a sparse transform, tricky in ONNX
# - At inference time: Python TF-IDF → dense array → ONNX classifier
```

> TF-IDF produces a **sparse matrix** (mostly zeros). ONNX works best with dense arrays. Converting sparse TF-IDF to ONNX is messy and adds unnecessary complexity.
>
> The standard industry pattern is:
> **text → Python TF-IDF → dense float array → ONNX classifier → prediction**
>
> The TF-IDF stays in Python. Only the classifier (the heavier computation at scale) goes to ONNX.

---

### Telling ONNX the Input Shape

```python
initial_type = [('X', FloatTensorType([None, vocab_size]))]
```

> You're telling the ONNX converter: "My model expects an input called `X` that is a 2D float array with:
> - **`None` rows** — any number of messages at once (flexible batch size)
> - **`vocab_size` columns** — one column per word in the vocabulary (e.g., 10,000)"

---

### The Conversion

```python
onnx_model = to_onnx(clf, initial_types=initial_type)
```

> This is the core line. `to_onnx()` inspects the sklearn `SGDClassifier` object and rebuilds its internal math (weights, decision function) in the ONNX format. The result is `onnx_model` — an in-memory ONNX object.

---

### Saving to Disk

```python
with open(onnx_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

> `.SerializeToString()` converts the ONNX model object into a raw binary string (bytes). We then write those bytes to a `.onnx` file. This is like "compiling" the model into a portable binary.

---

### Validation — The Critical Check

```python
df_val = pd.read_csv(config.VAL_FILE)
X_val = vectorizer.transform(df_val['text_clean'].fillna(''))
X_val_dense = X_val.toarray().astype(np.float32)
```

> Load the validation set. Run TF-IDF on it. Convert the sparse matrix to a dense float32 array. This is the same input format ONNX expects.

```python
sk_pred = clf.predict(X_val_dense[:10])
```

> Get predictions from the **original sklearn** classifier on the first 10 samples.

```python
sess = ort.InferenceSession(str(onnx_path))
input_name = sess.get_inputs()[0].name
onnx_pred = sess.run(None, {input_name: X_val_dense[:10]})[0]
```

> - `InferenceSession` loads the `.onnx` file into ONNX Runtime.
> - `get_inputs()[0].name` gets the name of the input slot (it matches what we set: `'X'`).
> - `sess.run(None, {...})` runs the model. `None` means "give me all outputs." You pass a dictionary: `{input_name: data}`.
> - `[0]` takes the first output — the predicted class (0 or 1).

```python
match = np.array_equal(sk_pred, onnx_pred)
print(f"  Match: {'YES' if match else 'NO — investigate!'}")
```

> Compare sklearn's predictions to ONNX's predictions. They should be **identical**. If "Match: YES" — the export was lossless. If "NO" — something went wrong in conversion.
>
> **This is your correctness guarantee.** You're proving the ONNX model didn't lose any information during conversion.

---

## Step 7 — `step7_inference.py`

### Purpose

Step 6 created the `.onnx` file. Step 7 uses it like a real production system would — load the model once, classify messages on demand. This is the **inference engine**.

---

### The `SpamClassifier` Class

```python
class SpamClassifier:
```

> A class bundles the model + all the logic to use it. In production, you'd create one instance at server startup and reuse it for every request. Much more efficient than reloading from disk every time.

---

### `__init__` — Loading at Startup

```python
vec_path = config.MODEL_DIR / "tfidf_vectorizer.pkl"
with open(vec_path, 'rb') as f:
    self.vectorizer = pickle.load(f)
```

> Load the TF-IDF vectorizer from disk into memory. This stays loaded as `self.vectorizer` for the lifetime of the classifier object.

```python
onnx_path = str(config.ONNX_MODEL_PATH)
self.session = ort.InferenceSession(onnx_path)
self.input_name = self.session.get_inputs()[0].name
```

> - Load the ONNX model into an ONNX Runtime `InferenceSession`. This is the inference engine.
> - Cache the `input_name` so you don't have to look it up on every prediction.

---

### `predict()` — Classifying One Message

```python
def predict(self, text: str) -> dict:
```

> Takes one string, returns a dictionary with `label`, `prediction`, and `confidence`.

```python
X = self.vectorizer.transform([text])
X_dense = X.toarray().astype(np.float32)
```

> **Step 1: Text → Numbers.**
> - `transform([text])` converts the raw string into a sparse TF-IDF vector. (Note the list `[text]` — sklearn expects a list even for one item.)
> - `.toarray()` converts sparse to dense.
> - `.astype(np.float32)` converts to 32-bit float — ONNX requires this exact type. 64-bit won't work.

```python
outputs = self.session.run(None, {self.input_name: X_dense})
prediction = int(outputs[0][0])
```

> **Step 2: Run ONNX.**
> - `outputs` is a list. `outputs[0]` is the array of class predictions. `outputs[1]` (if it exists) is the probabilities.
> - `outputs[0][0]` → the first sample's prediction → `0` (ham) or `1` (spam).

```python
confidence = 1.0
if len(outputs) > 1:
    proba = outputs[1]
    if isinstance(proba, list):
        proba = proba[0]
    if isinstance(proba, dict):
        confidence = max(proba.values())
    elif hasattr(proba, '__len__') and len(proba) > 0:
        confidence = float(np.max(proba))
```

> **Step 3: Extract Confidence.**
> ONNX returns probabilities in a quirky format — sometimes a list of dicts `[{0: 0.05, 1: 0.95}]`, sometimes an array. This code handles both:
> - If it's a dict: `max(proba.values())` → take the highest probability.
> - If it's an array: `np.max(proba)` → same thing.
>
> **Why confidence matters:** If the model says "spam" with 95% confidence → very reliable. If it says "spam" with 51% confidence → uncertain, should be sent for human review (the active learning loop from step 10/11).

```python
label = 'spam' if prediction == 1 else 'ham'
return {'label': label, 'prediction': prediction, 'confidence': round(confidence, 4)}
```

> Convert the raw `0/1` prediction back to a human-readable label and return everything.

---

### `predict_batch()` — Many Messages at Once

```python
def predict_batch(self, texts: list) -> list:
    X = self.vectorizer.transform(texts)
    X_dense = X.toarray().astype(np.float32)
    outputs = self.session.run(None, {self.input_name: X_dense})
    predictions = outputs[0]
    return [
        {'label': 'spam' if p == 1 else 'ham', 'prediction': int(p)}
        for p in predictions
    ]
```

> Same as `predict()` but for a whole list of messages at once. ONNX Runtime processes all of them in one pass — much faster than calling `predict()` in a loop. This is used by the `/moderate/batch` API endpoint in Step 8.

---

### `demo()` — The Test Run

```python
test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "WINNER! You've been selected for a free iPhone! Call NOW!",
    ...
]
```

> A hardcoded list of real-world-style messages — some ham (normal), some spam — to verify the model is working correctly after the ONNX export.

```python
for msg in test_messages:
    result = classifier.predict(msg)
    icon = "SPAM" if result['prediction'] == 1 else " ham"
    print(f"  [{icon}] (conf: {result['confidence']:.2f}) {msg[:65]}")
```

> Runs each message through the classifier and prints the result with confidence. Expected to see high confidence on obvious spam ("WINNER!") and ham ("lunch tomorrow").

---

## The Big Picture: Why Step 6 + 7 Together Matter

```
Step 4: Train         → saves classifier.pkl + tfidf_vectorizer.pkl  (sklearn objects)
Step 6: ONNX Export   → converts classifier.pkl → spam_classifier.onnx (portable binary)
Step 7: Inference     → loads tfidf_vectorizer.pkl + spam_classifier.onnx → production-ready classifier
Step 8: API           → wraps Step 7's SpamClassifier in a FastAPI HTTP server
```

> **Why not just use the sklearn `.pkl` directly in production?**
> 1. Sklearn is Python-only. ONNX runs in any language/platform.
> 2. ONNX Runtime is faster — optimized C++ execution engine.
> 3. ONNX models can be deployed to mobile, edge devices, browsers.
> 4. It's the industry-standard pattern for going from research to production.

---

## Likely Viva Questions & Answers

**Q: What is ONNX and why did you use it?**
> A: ONNX is a universal model format that lets you run ML models outside of Python. I used it because in production, the model runs in ONNX Runtime which is 2-5x faster than sklearn, and it makes the model deployable to any platform — not just Python servers.

**Q: Why didn't you convert the TF-IDF to ONNX as well?**
> A: TF-IDF produces sparse matrices which are difficult to represent in ONNX. The standard pattern is to keep TF-IDF in Python (it's fast) and only export the classifier to ONNX. This is what frameworks like Hugging Face and scikit-learn docs recommend.

**Q: How do you verify the ONNX model is correct?**
> A: After export, I run the same 10 validation samples through both the original sklearn model and the ONNX model and compare outputs. If they match exactly, the conversion was lossless. You see "Match: YES" printed.

**Q: What is confidence and why does it matter?**
> A: Confidence is the model's probability for its predicted class. High confidence (e.g., 0.95) means the model is certain. Low confidence (e.g., 0.55) means it's unsure. Low-confidence predictions are the ones we flag for human review — that's the active learning loop that makes the system self-improving.

**Q: What is `FloatTensorType([None, vocab_size])`?**
> A: It tells the ONNX converter the shape of the input. `None` means any batch size (flexible). `vocab_size` is the number of columns — one per word in the TF-IDF vocabulary (10,000 in our case).
