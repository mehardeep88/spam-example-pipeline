# Roadmap: What's Next After Step 7 (Inference)

## Where You Are Now

```
✅ Step 1: Download         — SMS Spam Collection from UCI
✅ Step 2: Load + EDA       — Statistics, plots, word analysis
✅ Step 3: Clean + Split    — Preprocessing, train/val/test
✅ Step 4: Train            — TF-IDF + SGDClassifier
✅ Step 5: Evaluate         — Test set metrics, confusion matrix
✅ Step 6: ONNX Export      — sklearn → ONNX conversion
✅ Step 7: Inference        — ONNX Runtime inference demo
✅ Step 8: FastAPI endpoint  — REST API with /moderate
✅ Step 9: MLflow tracking   — Experiment logging + sweeps
✅ Step 10: Active learning  — AL vs Random simulation
✅ Step 11: S3 + SQS wiring  — AWS feedback loop
```

---

## Do You Need to Do Anything On Your Own First?

**Yes — you need to actually run steps 1-7 before moving on.** Here's the checklist:

```bash
cd spamwork

python step1_download.py      # Downloads the dataset (~200KB)
python step2_load_eda.py      # Check the printed stats, look at plots/eda_overview.png
python step3_clean.py         # Creates data/processed/train.csv, val.csv, test.csv
python step4_train.py         # Trains the model, prints F1/accuracy
python step5_evaluate.py      # Test set results — this is your "baseline" number
python step6_onnx_export.py   # Creates models/spam_classifier.onnx
python step7_inference.py     # Demo — verify it classifies correctly
```

> [!IMPORTANT]
> **Run each step and read the output.** You need to understand what numbers you're getting (e.g., "my F1 is 0.95 on test set") because you'll need to explain these when presenting. Don't just run blindly.

### What to check at each step:
| Step | What to verify |
|------|---------------|
| 2 | EDA plots make sense — spam is ~13% of data, spam messages are longer |
| 4 | Validation F1 should be > 0.90 (SMS spam is an easy task) |
| 5 | Test F1 close to val F1 (no overfitting), check the error examples |
| 6 | "Match: YES" printed (ONNX matches sklearn) |
| 7 | Sample messages classified correctly |

---

## What Comes Next: Steps 8-11

### Step 8: FastAPI `/moderate` Endpoint

**What:** Wrap your ONNX model in a REST API so it can be called over HTTP.

**Why:** In production, models aren't run from scripts — they're served as APIs. A user/app sends a POST request with text, gets back spam/ham + confidence.

**What I'll build inside `spamwork/`:**
```
spamwork/
└── step8_api.py    # FastAPI app with /moderate, /health endpoints
```

**Simplified vs. the existing `api/main.py`:** The existing code in `projectmoderator/api/` was built for the multi-category system (toxicity, spam, nsfw, hate_speech). Your `spamwork` version will be simpler — just binary spam/ham.

---

### Step 9: MLflow Tracking

**What:** Log your training experiments (hyperparameters, metrics, model artifacts) so you can compare runs.

**Why:** When you change a hyperparameter (e.g., `TFIDF_MAX_FEATURES` from 10K to 50K), MLflow tracks which settings gave the best F1. It's like "version control for experiments."

**What I'll build:**
```
spamwork/
└── step9_mlflow.py    # Wraps step4_train with MLflow logging
```

**You'll need:** `pip install mlflow`

---

### Step 10: Active Learning Simulation

**What:** Prove that selectively labeling uncertain samples improves the model faster than random labeling.

**Why:** This is the **core thesis** of the project — the model identifies what it's confused about, asks a human to label those, and improves faster.

**What I'll build:**
```
spamwork/
└── step10_active_learning.py   # AL vs Random comparison experiment
```

---

### Step 11: S3 + SQS Wiring (AWS)

**What:** When the API encounters an uncertain prediction (low confidence), it pushes the text to an SQS queue. A human reviewer pulls from the queue, labels it, and the labeled data goes to S3 for the next retrain cycle.

**Architecture:**
```
User → FastAPI → ONNX model → prediction
                      │
                      ├── confident? → return result
                      └── uncertain? → push to SQS queue
                                           │
                                    Human reviewer labels it
                                           │
                                    Labeled data → S3 bucket
                                           │
                                    Retrain pipeline pulls from S3
```

> [!WARNING]
> **Step 11 requires an AWS account with S3 and SQS access.** I'll give you the code, but you'll need AWS credentials configured. If you don't have AWS, we can simulate it locally with LocalStack or just mock the AWS calls.

---

## Recommended Order

```
1. Run steps 1-7 yourself (TODAY)
2. I build step 8 (FastAPI)          — you can test locally with curl/Postman
3. I build step 9 (MLflow)           — you run experiments, see the dashboard
4. I build step 10 (Active Learning) — the thesis proof
5. I build step 11 (S3+SQS)         — only if you have AWS access
```

> [!TIP]
> Steps 8-10 can all be built inside `spamwork/` with no AWS dependency. Step 11 is the only one that needs cloud access. **Focus on getting steps 1-10 working first.**
