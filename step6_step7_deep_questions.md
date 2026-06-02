# Deep Questions — Step 6 & 7 Answers

---

## Q1: How does ONNX know `X` is the input? Did we define it anywhere?

Yes — **you defined it yourself** right here:

```python
initial_type = [('X', FloatTensorType([None, vocab_size]))]
```

The `'X'` in the tuple is the **name you give to the input slot**. It's just a label.
You could write `'input'`, `'text_features'`, `'banana'` — anything. The name doesn't matter as long as you use the **same name** when you call the model later.

That's exactly why Step 7 does this:

```python
self.input_name = self.session.get_inputs()[0].name
```

Instead of hardcoding `"X"`, it asks the model "what did whoever built you name the first input?" and gets back `"X"`. This way if someone renames it, the code still works.

Then when running the model:
```python
sess.run(None, {input_name: X_dense})
#                   ↑ this is "X" — the slot name you defined
```

So the flow is:
```
You define the name 'X' in initial_type
    → to_onnx() bakes that name into the .onnx file
    → get_inputs()[0].name reads it back
    → sess.run() uses it as the key to feed data in
```

---

## Q2: Why can it handle any number of messages? What does `None` mean?

```python
FloatTensorType([None, vocab_size])
```

This defines the **shape** of the input as a 2D grid:
- **Rows** = number of messages
- **Columns** = number of words in vocabulary

`None` in the row position means **"I don't know yet, it can be anything."**

Think of it like this. The grid could look like:

**1 message (rows=1):**
```
         word1  word2  word3  ... word10000
msg1  [  0.0    0.42   0.0   ...   0.17  ]
```

**3 messages (rows=3):**
```
         word1  word2  word3  ... word10000
msg1  [  0.0    0.42   0.0   ...   0.17  ]
msg2  [  0.31   0.0    0.0   ...   0.0   ]
msg3  [  0.0    0.0    0.88  ...   0.22  ]
```

The number of **columns** is always fixed = `vocab_size` (e.g., 10,000).
The number of **rows** is flexible = `None` = "batch size".

If you had written `[1, vocab_size]`, the model would ONLY accept exactly 1 message at a time and crash on anything else. `None` makes it flexible.

---

## Q3: What does `vocab_size` columns actually mean?

Say your TF-IDF vocabulary has 5 words (simplified):
```
vocabulary = {'free': 0, 'call': 1, 'now': 2, 'hello': 3, 'lunch': 4}
vocab_size = 5
```

For the message `"free call"`:
```
Column:  free   call   now   hello  lunch
Value:   0.71   0.71   0.0   0.0    0.0
```

For the message `"hello lunch"`:
```
Column:  free   call   now   hello  lunch
Value:   0.0    0.0    0.0   0.71   0.71
```

Each column = one word in the vocabulary.
The value = the TF-IDF weight (how important that word is in this message).
Words not in the message = 0.0.

In your real model, `vocab_size = 10,000` — so each message becomes a row of 10,000 numbers, most of them zero.

---

## Q4: Why does `vectorizer.transform()` need a list?

```python
X = self.vectorizer.transform([text])
#                              ↑ list with one item
```

`transform()` was designed to handle **multiple documents at once**. It always expects a list (or array) of strings, even if you only have one.

If you pass a plain string:
```python
vectorizer.transform("free call now")
```
Python iterates over a string character by character: `['f', 'r', 'e', 'e', ' ', ...]`
The vectorizer treats each **character** as a separate document. You get garbage.

If you pass a list with one string:
```python
vectorizer.transform(["free call now"])
```
Python iterates over the list: one item → one document → correct.

So `[text]` wraps the single string in a list so the vectorizer processes it as one complete document.

---

## Q5: Why float32 and not float64?

```python
X_dense = X.toarray().astype(np.float32)
```

**float64 = 64 bits** — Python/NumPy default. High precision.
**float32 = 32 bits** — half the memory. What ONNX Runtime requires.

ONNX Runtime is written in C++ and is compiled to use 32-bit floats because:
1. It's the standard in ML hardware (GPUs, neural network chips all use 32-bit)
2. Half the memory = faster
3. For spam classification, the difference between `0.9142857142857` (64-bit) and `0.914286` (32-bit) is completely irrelevant

If you pass float64 to `sess.run()`, ONNX Runtime throws a type error:
```
InvalidArgument: Unexpected input data type. Actual: float64 Expected: float
```
(`float` in ONNX means float32.)

So `.astype(np.float32)` is a **type cast** — like converting `int` to `float` in normal Python, but between 32-bit and 64-bit precision.

---

## Q6: What does `outputs` actually look like? Concrete example.

```python
outputs = self.session.run(None, {self.input_name: X_dense})
```

`outputs` is a **Python list** with either 1 or 2 items:

```
outputs = [
    outputs[0],    ← class predictions (always present)
    outputs[1],    ← probabilities (present if model supports it — SGDClassifier with log_loss does)
]
```

For the input `"FREE call win prize now"` (spam), it might look like:

```python
outputs[0] = array([1])
#             ↑ class prediction: 1 = spam
```

```python
outputs[1] = [{0: 0.038, 1: 0.962}]
#             ↑ list of one dict (one message)
#               0 = ham probability  → 3.8%
#               1 = spam probability → 96.2%
```

So then:
```python
prediction = int(outputs[0][0])    # → 1  (spam)
proba = outputs[1]                 # → [{0: 0.038, 1: 0.962}]
proba = proba[0]                   # → {0: 0.038, 1: 0.962}
confidence = max(proba.values())   # → 0.962
```

For a **batch of 2 messages**, `outputs[0]` would have 2 elements:
```python
outputs[0] = array([1, 0])         # first = spam, second = ham
outputs[1] = [
    {0: 0.038, 1: 0.962},          # first message
    {0: 0.891, 1: 0.109},          # second message
]
```

---

## Q7: Full Concrete Walkthrough — One Message Through `predict()`

**Input:** `"FREE call NOW win prize"`

---

### Step 1: `X = self.vectorizer.transform(["FREE call NOW win prize"])`

TF-IDF looks up each word in its 10,000-word vocabulary and computes a weight.
Result is a sparse matrix (shown here simplified to 5 words for clarity):

```
Sparse matrix (1 row × 10000 cols, most are 0):

     call    free    hello   now    prize   win    ...9994 other words...
msg [ 0.38    0.41    0.0    0.35    0.44   0.51   ...0.0 for each...   ]
```

Shape: `(1, 10000)` — one message, 10,000 features.

---

### Step 2: `X_dense = X.toarray().astype(np.float32)`

Converts sparse → dense → 32-bit float:

```
Before (sparse, float64):  stores only non-zero positions
After (dense, float32):

array([[0.38, 0.41, 0.0, 0.35, 0.44, 0.51, 0.0, 0.0, ..., 0.0]], dtype=float32)
       ↑ full 10,000 numbers, most are 0.0
Shape: (1, 10000)
```

---

### Step 3: `outputs = self.session.run(None, {self.input_name: X_dense})`

ONNX Runtime feeds that row of 10,000 numbers through the SGDClassifier's math.
The classifier multiplies those features by its learned weights and applies a threshold.

Result:
```python
outputs = [
    array([1]),                     # outputs[0]: predicted class = 1 (spam)
    [{0: 0.038, 1: 0.962}],         # outputs[1]: probabilities
]
```

---

### Step 4: `prediction = int(outputs[0][0])`

```python
outputs[0]    →  array([1])
outputs[0][0] →  1          (the first element)
int(...)      →  1          (converted to plain Python int)
```

---

### Step 5: Extract confidence

```python
proba = outputs[1]          # [{0: 0.038, 1: 0.962}]
proba = proba[0]            # {0: 0.038, 1: 0.962}  ← unwrap the list
confidence = max(proba.values())   # max(0.038, 0.962) = 0.962
```

---

### Step 6: Build the return value

```python
label = 'spam'   # because prediction == 1

return {
    'label': 'spam',
    'prediction': 1,
    'confidence': 0.9620,
}
```

---

### Full journey summary:

```
"FREE call NOW win prize"
    │
    ▼ vectorizer.transform(["FREE call NOW win prize"])
sparse matrix (1 × 10000), float64, mostly zeros
    │
    ▼ .toarray().astype(np.float32)
dense array [[0.38, 0.41, 0.0, 0.35, 0.44, 0.51, 0.0, ...]] float32, shape (1,10000)
    │
    ▼ sess.run(None, {'X': dense_array})
outputs = [array([1]),  [{0: 0.038, 1: 0.962}]]
    │
    ▼ parse prediction + confidence
prediction = 1,  confidence = 0.962
    │
    ▼ return
{'label': 'spam', 'prediction': 1, 'confidence': 0.9620}
```
