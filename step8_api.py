"""
STEP 8: FastAPI moderation endpoint.

What is an API?
    Application Programming Interface — lets other programs send text
    to your model and receive predictions over HTTP. Instead of running
    a Python script, anyone (frontend, mobile app, another service) can
    POST a message and get back "spam" or "ham" + confidence.

What this script does:
    1. Loads the ONNX model + TF-IDF vectorizer at startup
    2. Exposes POST /moderate        — single text classification
    3. Exposes POST /moderate/batch   — batch classification (up to 100)
    4. Exposes GET  /health           — is the model loaded?
    5. Exposes GET  /model/info       — model metadata

How to run:
    cd spamwork
    uvicorn step8_api:app --host 0.0.0.0 --port 8000 --reload

Then test:
    curl -X POST http://localhost:8000/moderate \
         -H "Content-Type: application/json" \
         -d '{"text": "FREE entry! Call now to win 10000!"}'

Dependencies:
    pip install fastapi uvicorn
"""
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
import config
from step7_inference import SpamClassifier


# ─── Request / Response Schemas ──────────────────────────

class ModerateRequest(BaseModel):
    """Single text moderation request."""
    text: str = Field(
        ..., min_length=1, max_length=50_000,
        description="The SMS or text message to classify"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-provided ID for tracing"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "WINNER! You've been selected for a free iPhone!"},
                {"text": "Hey are you free for lunch tomorrow?", "request_id": "req-42"},
            ]
        }
    }


class BatchModerateRequest(BaseModel):
    """Batch moderation request — up to 100 texts at once."""
    texts: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="List of texts to classify"
    )


class PredictionResult(BaseModel):
    """Classification result for a single text."""
    request_id: str
    text: str
    label: str                 # "ham" or "spam"
    prediction: int            # 0 or 1
    confidence: float          # max probability
    should_review: bool        # True if confidence < threshold
    latency_ms: float


class BatchPredictionResult(BaseModel):
    """Results for a batch of texts."""
    results: List[PredictionResult]
    total: int
    spam_count: int
    flagged_for_review: int
    avg_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    test_prediction: Optional[str] = None
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_path: str
    vectorizer_vocab_size: int
    confidence_threshold: float
    onnx_model_size_kb: float


# ─── Configuration ───────────────────────────────────────

# Below this confidence → flag for human review
CONFIDENCE_THRESHOLD = 0.70


# ─── App Lifecycle ───────────────────────────────────────

classifier: Optional[SpamClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX model at startup, release at shutdown."""
    global classifier
    print("Loading spam classifier...")
    try:
        classifier = SpamClassifier()
        print("  Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"  Model not found: {e}")
        print("  Run steps 1-6 first to train and export the model.")
        classifier = None
    yield
    classifier = None
    print("Classifier shut down.")


# ─── Create FastAPI App ─────────────────────────────────

app = FastAPI(
    title="SMS Spam Classifier API",
    description=(
        "Binary spam/ham classification using an ONNX-exported "
        "TF-IDF + SGDClassifier pipeline. Low-confidence predictions "
        "are flagged for human review (active learning loop)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_model():
    """Raise 503 if the model is not loaded."""
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Run steps 1-6 to train and export "
                "the ONNX model, then restart the API."
            ),
        )


# ─── Endpoints ───────────────────────────────────────────

@app.post("/moderate", response_model=PredictionResult, tags=["Moderation"])
async def moderate_text(req: ModerateRequest):
    """
    Classify a single text as spam or ham.

    Returns the label, confidence score, and whether the prediction
    should be routed to a human reviewer (low confidence).
    """
    _ensure_model()

    request_id = req.request_id or str(uuid.uuid4())

    t0 = time.perf_counter()
    result = classifier.predict(req.text)
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictionResult(
        request_id=request_id,
        text=req.text,
        label=result["label"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        should_review=result["confidence"] < CONFIDENCE_THRESHOLD,
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/moderate/batch",
    response_model=BatchPredictionResult,
    tags=["Moderation"],
)
async def moderate_batch(req: BatchModerateRequest):
    """
    Classify a batch of texts (up to 100).

    Returns individual results plus aggregate stats:
    total count, spam count, number flagged for review.
    """
    _ensure_model()

    t0 = time.perf_counter()
    raw_results = classifier.predict_batch(req.texts)
    total_latency = (time.perf_counter() - t0) * 1000
    per_item = total_latency / len(req.texts)

    results = []
    spam_count = 0
    flagged = 0
    for i, r in enumerate(raw_results):
        # predict_batch doesn't return confidence, so re-predict
        # for confidence on each item (batch is small enough)
        full = classifier.predict(req.texts[i])
        conf = full["confidence"]
        should_review = conf < CONFIDENCE_THRESHOLD

        if r["prediction"] == 1:
            spam_count += 1
        if should_review:
            flagged += 1

        results.append(
            PredictionResult(
                request_id=str(uuid.uuid4()),
                text=req.texts[i],
                label=r["label"],
                prediction=r["prediction"],
                confidence=conf,
                should_review=should_review,
                latency_ms=round(per_item, 2),
            )
        )

    return BatchPredictionResult(
        results=results,
        total=len(results),
        spam_count=spam_count,
        flagged_for_review=flagged,
        avg_latency_ms=round(per_item, 2),
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the model is loaded and operational."""
    if classifier is None:
        return HealthResponse(
            status="unhealthy",
            model="not loaded",
            error="ONNX model not loaded. Run steps 1-6 first.",
        )
    try:
        result = classifier.predict("health check test message")
        return HealthResponse(
            status="healthy",
            model=config.ONNX_MODEL_PATH.name,
            test_prediction=result["label"],
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model=config.ONNX_MODEL_PATH.name,
            error=str(e),
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return metadata about the loaded model."""
    _ensure_model()
    return ModelInfoResponse(
        model_path=str(config.ONNX_MODEL_PATH),
        vectorizer_vocab_size=len(classifier.vectorizer.vocabulary_),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        onnx_model_size_kb=round(
            config.ONNX_MODEL_PATH.stat().st_size / 1024, 1
        ),
    )


# ─── Run directly ───────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("STEP 8: Starting FastAPI Server")
    print("=" * 50)
    print(f"  Docs:   http://localhost:8000/docs")
    print(f"  Health: http://localhost:8000/health")
    uvicorn.run("step8_api:app", host="0.0.0.0", port=8000, reload=True)
