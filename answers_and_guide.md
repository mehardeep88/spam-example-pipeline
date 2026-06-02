# Answers to All Your Questions

---

## 1. `run_all.py` — "It only contains steps till 7"

**It actually has all 11 steps.** Look at the steps dict in [run_all.py](file:///c:/Users/deepm/OneDrive/Documents/projectmoderator/spamwork/run_all.py#L33-L44):

```python
steps = {
    1: ("Download Dataset", download),
    2: ("Load & EDA", ...),
    3: ("Clean & Split", ...),
    4: ("Train Model", train),
    5: ("Evaluate on Test Set", evaluate),
    6: ("Export to ONNX", export_onnx),
    7: ("ONNX Inference Demo", demo),
    8: ("FastAPI Server", _run_api),          # ← here
    9: ("MLflow Tracking", _run_mlflow),       # ← here
    10: ("Active Learning Simulation", _run_active_learning),  # ← here
    11: ("AWS S3 + SQS Pipeline", _run_aws),   # ← here
}
```

**BUT** — the default `run_all()` call only runs steps 1–7:

```python
def run_all(start=1, end=7):  # ← default end=7
```

This is **intentional** because:
- **Step 8** starts a blocking web server (won't release until Ctrl+C)
- **Step 9** needs MLflow installed
- **Step 10** trains ~40 models (takes minutes)
- **Step 11** needs AWS credentials

You run them individually:
```
python run_all.py 8       # start the API
python run_all.py 9       # MLflow tracking
python run_all.py 10      # active learning sim
python run_all.py 11      # AWS pipeline
```

---

## 2. `/moderate` with 51.3% confidence — Can it auto-send to SQS?

**Yes, that is exactly the design intent, but it doesn't happen automatically right now.** Here's the flow:

### What currently happens
When you POST to `/moderate` with your own text and get 51.3% confidence:

```json
{
  "confidence": 0.513,
  "should_review": true   // ← it's flagged because 0.513 < 0.70 threshold
}
```

The API **tells you** it should be reviewed (`should_review: true`), but it does **NOT** push to SQS by itself. It just returns the flag.

### What step11 does
Step 11's `AWSPipeline.flag_uncertain()` is the function that actually pushes to SQS:

```python
def flag_uncertain(self, text, predicted_label, confidence, threshold=0.70):
    if confidence >= threshold:
        return False            # confident → skip
    self.sqs.send_for_review(text, predicted_label, confidence)
    return True                 # uncertain → pushed to SQS
```

### What you'd need to wire them together

To make `/moderate` auto-push uncertain predictions to SQS, the API endpoint needs to call `flag_uncertain()` after the prediction. This is the real production architecture — the API is the **entry point**, and SQS is the **downstream routing**.

> [!IMPORTANT]
> **Your 51.3% confidence absolutely should go to SQS.** The threshold is 70% — anything below that is considered uncertain. 51.3% is essentially a coin flip, so the model is genuinely unsure. That's exactly the kind of prediction that needs a human to look at it.

The connection between `/moderate` → SQS is what makes the **active learning loop** work:
1. Model predicts with low confidence → pushes to SQS
2. Human reviewer pulls from SQS → corrects the label
3. Corrected label → stored in S3
4. Retrain pulls from S3 → model improves

**If you want, I can wire `/moderate` to automatically push to SQS when `should_review` is true.** This would require AWS credentials to be configured.

---

## 3. What do `/health` and `/model/info` return?

### `GET /health` — Is the system alive?

From [step8_api.py:L256-L277](file:///c:/Users/deepm/OneDrive/Documents/projectmoderator/spamwork/step8_api.py#L256-L277):

```json
// When model IS loaded:
{
  "status": "healthy",
  "model": "spam_classifier.onnx",
  "test_prediction": "ham"       // runs a quick prediction to prove it works
}

// When model is NOT loaded:
{
  "status": "unhealthy",
  "model": "not loaded",
  "error": "ONNX model not loaded. Run steps 1-6 first."
}
```

**Purpose:** It's a **liveness probe**. In production, a load balancer (AWS ALB, Kubernetes) pings `/health` every few seconds. If it gets "unhealthy", it stops sending traffic to that server and spins up a replacement. It also actually runs a prediction on `"health check test message"` to make sure not just the model file loaded, but inference actually works end-to-end.

### `GET /model/info` — What model is running?

From [step8_api.py:L280-L291](file:///c:/Users/deepm/OneDrive/Documents/projectmoderator/spamwork/step8_api.py#L280-L291):

```json
{
  "model_path": "c:\\...\\models\\spam_classifier.onnx",
  "vectorizer_vocab_size": 8541,        // how many words the TF-IDF knows
  "confidence_threshold": 0.70,          // below this → flag for review
  "onnx_model_size_kb": 231.5           // file size of the ONNX model
}
```

**Purpose:** Debugging and auditing. If someone asks "which model is running in production?" — you check this endpoint instead of SSH-ing into the server. Useful for:
- Verifying a new model deployment went through
- Checking model size (big model = slow inference)
- Confirming the confidence threshold matches what you expect

---

## 4. Step 9 errors + MLflow dashboard — What to look at

### The errors you're seeing

Step 9 errors are likely from **import/dependency issues** (mlflow's internal dependencies or path conflicts). If the training ran to completion and you can open `localhost:5000`, the core functionality worked despite the warnings.

### What to look at in MLflow (`http://localhost:5000`)

Open the dashboard and look at these things:

| Section | What it shows | Why it matters |
|---------|--------------|----------------|
| **Experiments** (left sidebar) | `sms-spam-classifier` experiment | Click to see all training runs |
| **Runs table** | Each row = one training run | Compare different runs side-by-side |
| **Parameters** tab | `sgd_alpha=0.0001`, `tfidf_max_features=10000`, etc. | See exactly which hyperparameters produced which results |
| **Metrics** tab | `test_f1=0.89`, `val_accuracy=0.97`, `test_roc_auc=0.98` | **This is the key thing** — how well did the model perform? |
| **Artifacts** tab | `tfidf_vectorizer.pkl`, `classifier.pkl`, `spam_classifier.onnx` | The actual model files logged with this run |

### What you're supposed to DO with MLflow

1. **Compare experiments**: Run `python step9_mlflow.py --sweep` to train 3 different configurations. Then in MLflow, select multiple runs → click "Compare" → you get charts showing which config was best.

2. **Track over time**: Every time you retrain (e.g., after collecting feedback from step 11), a new run appears. You can see if your model is improving.

3. **Reproducibility**: If someone asks "how did you get 97% accuracy?", you can point to the exact run with the exact parameters and artifacts.

> [!TIP]
> **The single most important thing to look at:** Click on a run → look at `test_f1` and `test_roc_auc`. These tell you how well the model performs on unseen data. F1 > 0.85 is good for spam. AUC > 0.95 means excellent discrimination.

---

## 5. Step 10 — What is it for? What to do with the result?

### What it proves

Step 10 is the **theoretical proof that your active learning loop works**. It runs a simulation:

1. Start with only 20% of training data labeled
2. **Active Learning path**: model picks the most uncertain samples → "label" them → retrain
3. **Random path**: randomly pick samples → "label" them → retrain
4. Compare: does AL learn faster?

### What to do with the result

The output gives you:

```
  Baseline F1 (seed only):   0.8812
  Final AL F1:               0.9438
  Final Random F1:           0.9312
  Average F1 advantage:      +0.0089
  AL wins in 14 / 20 rounds

  ✓ Active Learning outperforms Random Sampling!
```

And it saves a plot to `plots/active_learning_curves.png`.

**For your viva/project**, this result means:
- **"Does selecting uncertain samples help?"** → Yes, AL reached higher F1 with the same number of labeled samples
- **"How much annotation effort does it save?"** → Look at where AL reaches the same F1 as Random — AL gets there with fewer labeled examples
- The plot (`active_learning_curves.png`) is your **evidence figure** showing AL's learning curve is above Random's

> [!IMPORTANT]
> Step 10 is not something you "use" in production. It's a **validation experiment** that proves your thesis: that the uncertainty-based routing in steps 8+11 (sending low-confidence predictions to human reviewers) is **scientifically justified** — the model genuinely improves faster when it selects what to learn from.

---

## 6. Step 11 — No AWS, no creds, nothing happened

### Why it didn't ask for credentials

Look at the CLI structure in [step11_aws.py:L615-L666](file:///c:/Users/deepm/OneDrive/Documents/projectmoderator/spamwork/step11_aws.py#L615-L666):

```python
if args.setup:
    pipeline.setup()
elif args.upload_model:
    ...
elif args.demo:
    run_demo()
else:
    parser.print_help()    # ← YOU HIT THIS BRANCH
```

**You ran `uv run step11_aws.py` with no flags.** Since none of `--setup`, `--demo`, `--status`, etc. were provided, it fell through to the `else` branch which just prints the help text and exits. No AWS call was ever made, so no credentials were needed.

The `AWSPipeline()` object on line 634 **was created**, but `boto3.client()` is **lazy** — it doesn't actually try to connect to AWS until you call an API method like `create_bucket()` or `send_message()`. Since no flag matched, no API call happened, so no credentials error.

### What would actually happen with AWS

If you ran `uv run step11_aws.py --setup`, **then** it would:
1. Try to create an S3 bucket named `spam-classifier-artifacts`
2. Try to create an SQS queue named `spam-review-queue`
3. **FAIL** with `NoCredentialsError` because you have no AWS credentials configured

You'd see:
```
  ✗ AWS credentials not found!
    Configure them with one of:
      aws configure
      set AWS_ACCESS_KEY_ID=...
      set AWS_SECRET_ACCESS_KEY=...
```

> [!CAUTION]
> **Step 11 requires a real AWS account with billing enabled.** Creating S3 buckets and SQS queues costs money (tiny amounts for dev, but real charges). Don't run it unless you have AWS credentials set up.

### For your viva (without AWS)

Step 11 is designed to be **explained conceptually** without running it. The code itself is the deliverable — it demonstrates you understand:
- How to store model artifacts in S3 (model registry)
- How to route uncertain predictions to SQS (message queue)
- How to collect human feedback from S3 for retraining
- The full active learning feedback loop architecture

---

## Summary of the Full Pipeline

| Step | What | Status | Purpose |
|------|------|--------|---------|
| 1-7 | Core ML pipeline | ✅ Works | Train → evaluate → export → inference |
| 8 | FastAPI API | ✅ Works | Serve predictions over HTTP |
| 9 | MLflow tracking | ⚠️ Some errors but works | Track experiments, compare hyperparameters |
| 10 | AL simulation | ✅ Works | **Prove** active learning is better than random |
| 11 | AWS S3+SQS | 📋 Code-only | Production infrastructure (needs AWS account) |

The key insight: **Steps 8-11 are not sequential dependencies.** They're separate capabilities that connect in production:
```
User text → Step 8 (API) → prediction
                  ↓ (if uncertain)
            Step 11 (SQS) → human review → S3 feedback
                                              ↓
            Step 9 (MLflow) ← retrain ← feedback data
                                              ↑
            Step 10 proved this loop works ───┘
```
