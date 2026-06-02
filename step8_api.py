"""
STEP 8: FastAPI moderation endpoint with SQS auto-push and UI dashboard.

What is an API?
    Application Programming Interface — lets other programs send text
    to your model and receive predictions over HTTP. Instead of running
    a Python script, anyone (frontend, mobile app, another service) can
    POST a message and get back "spam" or "ham" + confidence.

What this script does:
    1. Loads the ONNX model + TF-IDF vectorizer at startup
    2. Exposes POST /moderate        — single text classification
       → If confidence < threshold, auto-pushes to SQS (when configured)
    3. Exposes POST /moderate/batch   — batch classification (up to 100)
    4. Exposes GET  /health           — is the model loaded?
    5. Exposes GET  /model/info       — model metadata
    6. Exposes GET  /api/queue/status — SQS queue depth
    7. Exposes GET  /api/history      — recent predictions
    8. Exposes GET  /                 — dashboard UI

How to run:
    cd spamwork
    uvicorn step8_api:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000/ for the dashboard.

Dependencies:
    pip install fastapi uvicorn
    pip install boto3  (optional, for SQS auto-push)
"""
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
import config
from step7_inference import SpamClassifier


# ─── SQS Integration (graceful — works without AWS) ─────

aws_pipeline = None
sqs_manager = None
sqs_available = False

try:
    from step11_aws import AWSPipeline
    aws_pipeline = AWSPipeline()
    sqs_manager = aws_pipeline.sqs
    sqs_available = True
    print("  [OK] SQS manager loaded (will push uncertain predictions)")
except ImportError:
    print("  [INFO] boto3 not installed -- SQS auto-push disabled")
except Exception as e:
    print(f"  [INFO] SQS init skipped: {e}")


# --- In-memory prediction history (last 50) -------------

prediction_history: deque = deque(maxlen=50)


# --- Request / Response Schemas --------------------------

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
    queued_to_sqs: bool = False  # True if pushed to SQS queue
    latency_ms: float


class BatchPredictionResult(BaseModel):
    """Results for a batch of texts."""
    results: List[PredictionResult]
    total: int
    spam_count: int
    flagged_for_review: int
    queued_to_sqs: int
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


class QueueStatusResponse(BaseModel):
    available: bool
    pending: Optional[int] = None
    in_flight: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ReviewSubmitRequest(BaseModel):
    receipt_handle: str
    text: str
    true_label: int
    predicted_label: int
    confidence: float


# --- Configuration ---------------------------------------

# Below this confidence → flag for human review + auto-push to SQS
CONFIDENCE_THRESHOLD = 0.80


# --- App Lifecycle ---------------------------------------

classifier: Optional[SpamClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX model at startup, release at shutdown."""
    global classifier
    print("Loading spam classifier...")
    try:
        classifier = SpamClassifier()
        print("  [OK] Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"  [WARN] Model not found: {e}")
        print("  Run steps 1-6 first to train and export the model.")
        classifier = None

    sqs_status = "enabled" if sqs_available else "disabled (no boto3/creds)"
    print(f"  SQS auto-push: {sqs_status}")
    print(f"  Dashboard:     http://localhost:8000/")
    print(f"  API docs:      http://localhost:8000/docs")

    yield
    classifier = None
    print("Classifier shut down.")


# --- Create FastAPI App ---------------------------------

app = FastAPI(
    title="SMS Spam Classifier API",
    description=(
        "Binary spam/ham classification using an ONNX-exported "
        "TF-IDF + SGDClassifier pipeline. Low-confidence predictions "
        "are auto-pushed to SQS for human review (active learning loop)."
    ),
    version="1.1.0",
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


def _try_push_sqs(text: str, prediction: int, confidence: float, request_id: str) -> bool:
    """
    Attempt to push an uncertain prediction to SQS.
    Returns True if successfully queued, False otherwise.
    Never raises — failures are logged and swallowed.
    """
    if not sqs_available or sqs_manager is None:
        return False

    try:
        sqs_manager.send_for_review(
            text=text,
            predicted_label=prediction,
            confidence=confidence,
            request_id=request_id,
        )
        return True
    except Exception as e:
        print(f"  [WARN] SQS push failed (non-fatal): {e}")
        return False


# --- Endpoints -------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the moderation dashboard UI."""
    html_path = Path(__file__).parent / "static" / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1><p>Missing static/dashboard.html</p>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/moderate", response_model=PredictionResult, tags=["Moderation"])
async def moderate_text(req: ModerateRequest):
    """
    Classify a single text as spam or ham.

    Returns the label, confidence score, and whether the prediction
    should be routed to a human reviewer (low confidence).

    **SQS auto-push:** If confidence is below the threshold (70%) and
    SQS is configured, the prediction is automatically pushed to the
    review queue for human labeling.
    """
    _ensure_model()

    request_id = req.request_id or str(uuid.uuid4())

    t0 = time.perf_counter()
    result = classifier.predict(req.text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    conf = result["confidence"]
    should_review = conf < CONFIDENCE_THRESHOLD

    # ── Auto-push to SQS if uncertain ──
    queued = False
    if should_review:
        queued = _try_push_sqs(req.text, result["prediction"], conf, request_id)

    # ── Record in history ──
    history_entry = {
        "request_id": request_id,
        "text": req.text,
        "label": result["label"],
        "prediction": result["prediction"],
        "confidence": conf,
        "should_review": should_review,
        "queued_to_sqs": queued,
        "latency_ms": latency_ms,
    }
    prediction_history.appendleft(history_entry)

    return PredictionResult(**history_entry)


@app.post(
    "/moderate/batch",
    response_model=BatchPredictionResult,
    tags=["Moderation"],
)
async def moderate_batch(req: BatchModerateRequest):
    """
    Classify a batch of texts (up to 100).

    Returns individual results plus aggregate stats:
    total count, spam count, number flagged for review, and
    number queued to SQS.
    """
    _ensure_model()

    t0 = time.perf_counter()
    results = []
    spam_count = 0
    flagged = 0
    sqs_count = 0

    for text in req.texts:
        full = classifier.predict(text)
        conf = full["confidence"]
        should_review = conf < CONFIDENCE_THRESHOLD
        rid = str(uuid.uuid4())

        queued = False
        if should_review:
            queued = _try_push_sqs(text, full["prediction"], conf, rid)
            flagged += 1
            if queued:
                sqs_count += 1

        if full["prediction"] == 1:
            spam_count += 1

        per_item_latency = round((time.perf_counter() - t0) * 1000 / len(req.texts), 2)

        entry = {
            "request_id": rid,
            "text": text,
            "label": full["label"],
            "prediction": full["prediction"],
            "confidence": conf,
            "should_review": should_review,
            "queued_to_sqs": queued,
            "latency_ms": per_item_latency,
        }
        results.append(PredictionResult(**entry))
        prediction_history.appendleft(entry)

    total_latency = (time.perf_counter() - t0) * 1000

    return BatchPredictionResult(
        results=results,
        total=len(results),
        spam_count=spam_count,
        flagged_for_review=flagged,
        queued_to_sqs=sqs_count,
        avg_latency_ms=round(total_latency / len(req.texts), 2),
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


@app.get("/api/queue/status", response_model=QueueStatusResponse, tags=["System"])
async def queue_status():
    """Check the SQS review queue depth."""
    if not sqs_available:
        return QueueStatusResponse(
            available=False,
            message="SQS not configured (boto3 not installed or no credentials)",
        )

    try:
        depth = sqs_manager.get_queue_depth()
        return QueueStatusResponse(
            available=True,
            pending=depth["pending"],
            in_flight=depth["in_flight"],
        )
    except Exception as e:
        return QueueStatusResponse(
            available=False,
            error=f"SQS error: {e}",
        )


@app.get("/api/history", tags=["System"])
async def get_history():
    """Return the last 50 predictions."""
    return {
        "predictions": list(prediction_history),
        "total": len(prediction_history),
    }


@app.get("/api/queue/pull", tags=["Review"])
async def pull_from_queue():
    """Pull up to 5 messages from SQS for human review."""
    if not sqs_available:
        return {"messages": []}
    try:
        messages = sqs_manager.poll_reviews(max_messages=5)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/queue/submit", tags=["Review"])
async def submit_to_queue(req: ReviewSubmitRequest):
    """Submit a human-reviewed label, saving to S3 and deleting from SQS."""
    if not sqs_available or aws_pipeline is None:
        raise HTTPException(status_code=503, detail="AWS not configured")
    try:
        aws_pipeline.submit_review(
            receipt_handle=req.receipt_handle,
            text=req.text,
            true_label=req.true_label,
            predicted_label=req.predicted_label,
            confidence=req.confidence
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run directly ───────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("STEP 8: Starting FastAPI Server")
    print("=" * 50)
    print(f"  Dashboard: http://localhost:8000/")
    print(f"  API Docs:  http://localhost:8000/docs")
    print(f"  Health:    http://localhost:8000/health")
    uvicorn.run("step8_api:app", host="0.0.0.0", port=8000, reload=True)
