"""
Central configuration for the SMS Spam pipeline.
All paths, hyperparameters, and settings in one place.

Why a config file?
    - Single source of truth for all constants
    - Change a path or hyperparameter once, affects all steps
    - Makes experiments reproducible
"""
from pathlib import Path

# ─── Directory Structure ─────────────────────────────────
WORK_DIR       = Path(__file__).parent          # spamwork/
DATA_DIR       = WORK_DIR / "data"              # raw + processed data
RAW_DIR        = DATA_DIR / "raw"               # original downloaded file
PROCESSED_DIR  = DATA_DIR / "processed"         # cleaned splits
MODEL_DIR      = WORK_DIR / "models"            # saved model artifacts
PLOTS_DIR      = WORK_DIR / "plots"             # EDA charts

# Create directories on import so scripts don't crash
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset Paths ───────────────────────────────────────
# After download, the raw file lives here:
RAW_FILE = RAW_DIR / "SMSSpamCollection.tsv"

# After cleaning, the splits live here:
TRAIN_FILE = PROCESSED_DIR / "train.csv"
VAL_FILE   = PROCESSED_DIR / "val.csv"
TEST_FILE  = PROCESSED_DIR / "test.csv"

# ─── Download URL ────────────────────────────────────────
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

# ─── Train / Val / Test Split ────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ─── TF-IDF Hyperparameters ─────────────────────────────
TFIDF_MAX_FEATURES = 10_000    # Max vocabulary size (SMS is small, 10K is plenty)
TFIDF_NGRAM_RANGE  = (1, 2)    # Unigrams + bigrams ("free", "free call")
TFIDF_MIN_DF       = 2         # Ignore words appearing in < 2 documents
TFIDF_MAX_DF       = 0.95      # Ignore words appearing in > 95% of docs
TFIDF_SUBLINEAR_TF = True      # Use log(1 + tf) instead of raw tf

# ─── Classifier Hyperparameters ──────────────────────────
SGD_LOSS       = "log_loss"    # Logistic regression via SGD → gives probabilities
SGD_ALPHA      = 1e-4          # Regularization strength
SGD_MAX_ITER   = 1000          # Max training epochs
SGD_RANDOM_STATE = 42

# ─── ONNX ────────────────────────────────────────────────
ONNX_MODEL_PATH = MODEL_DIR / "spam_classifier.onnx"
