"""
STEP 11: AWS S3 + SQS wiring for the active learning loop.

Architecture:
    User → FastAPI → ONNX model → prediction
                         │
                         ├── confident?  → return result
                         └── uncertain?  → push to SQS queue
                                              │
                                       Human reviewer labels it
                                              │
                                       Labeled data → S3 bucket
                                              │
                                       Retrain pipeline pulls from S3

What this script does:
    1. Provides an S3Manager class:
       - upload_model()        → push ONNX + vectorizer to S3
       - download_model()      → pull model artifacts from S3
       - upload_feedback()     → save human-labeled corrections to S3
       - list_feedback_files() → list all stored feedback CSVs
       - download_feedback()   → download feedback data for retraining

    2. Provides an SQSManager class:
       - send_for_review()     → push uncertain predictions to SQS
       - poll_reviews()        → pull messages from SQS (for human reviewer)
       - delete_message()      → acknowledge a reviewed message
       - get_queue_depth()     → how many items are waiting for review

    3. Provides an AWSPipeline class that ties it all together:
       - flag_uncertain()      → after prediction, route to SQS if low confidence
       - submit_review()       → human submits label → stored in S3
       - sync_model()          → upload/download model artifacts
       - collect_feedback()    → gather all reviewed data for retraining

Prerequisites:
    1. pip install boto3
    2. Configure AWS credentials (one of these):
       - Run `aws configure` (recommended)
       - Set env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
       - Use an IAM role (on EC2/ECS)
    3. Create the S3 bucket and SQS queue (the script can do this for you
       with --setup flag)

How to run:
    # First-time setup: create bucket + queue
    python step11_aws.py --setup

    # Upload trained model to S3
    python step11_aws.py --upload-model

    # Demo: push sample uncertain predictions to SQS
    python step11_aws.py --demo

    # Check queue depth
    python step11_aws.py --status
"""
import sys
import json
import time
import csv
import io
from pathlib import Path
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError

sys.path.insert(0, str(Path(__file__).parent))
import config


# ─── AWS Resource Names ─────────────────────────────────
# Override these via environment variables if you want different names.
import os

S3_BUCKET     = os.getenv("SPAM_S3_BUCKET",    config.AWS_S3_BUCKET)
SQS_QUEUE     = os.getenv("SPAM_SQS_QUEUE",    config.AWS_SQS_QUEUE)
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION", config.AWS_REGION)

# S3 key prefixes
S3_MODEL_PREFIX    = "models/"
S3_FEEDBACK_PREFIX = "feedback/"

# SQS settings
SQS_MAX_MESSAGES   = 10       # Max messages per poll (SQS limit)
SQS_WAIT_SECONDS   = 5        # Long-polling wait time
SQS_VISIBILITY     = 300      # Seconds before unacknowledged message reappears


# ─── S3 Manager ─────────────────────────────────────────

class S3Manager:
    """Handles model artifact storage and feedback data on S3."""

    def __init__(self, bucket: str = S3_BUCKET, region: str = AWS_REGION):
        self.bucket = bucket
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)

    def create_bucket(self):
        """Create the S3 bucket if it doesn't exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            print(f"  S3 bucket already exists: {self.bucket}")
        except ClientError:
            print(f"  Creating S3 bucket: {self.bucket}")
            create_args = {"Bucket": self.bucket}
            # us-east-1 doesn't accept LocationConstraint
            if self.region != "us-east-1":
                create_args["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.region
                }
            self.s3.create_bucket(**create_args)
            print(f"  ✓ Bucket created: {self.bucket}")

    def upload_model(self):
        """
        Upload ONNX model + TF-IDF vectorizer to S3.

        Uploads:
            models/spam_classifier.onnx
            models/tfidf_vectorizer.pkl
            models/classifier.pkl
        """
        artifacts = [
            config.ONNX_MODEL_PATH,
            config.MODEL_DIR / "tfidf_vectorizer.pkl",
            config.MODEL_DIR / "classifier.pkl",
        ]

        uploaded = 0
        for artifact in artifacts:
            if not artifact.exists():
                print(f"  ⚠ Skipping (not found): {artifact.name}")
                continue

            s3_key = f"{S3_MODEL_PREFIX}{artifact.name}"
            size_kb = artifact.stat().st_size / 1024

            print(f"  Uploading {artifact.name} ({size_kb:.1f} KB) → s3://{self.bucket}/{s3_key}")
            self.s3.upload_file(str(artifact), self.bucket, s3_key)
            uploaded += 1

        print(f"  ✓ Uploaded {uploaded} artifact(s) to S3")
        return uploaded

    def download_model(self, target_dir: Path = None):
        """
        Download model artifacts from S3 to local directory.

        Args:
            target_dir: Where to save (defaults to config.MODEL_DIR)
        """
        target_dir = target_dir or config.MODEL_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=S3_MODEL_PREFIX,
        )

        if "Contents" not in response:
            print("  ⚠ No model artifacts found in S3")
            return 0

        downloaded = 0
        for obj in response["Contents"]:
            key = obj["Key"]
            filename = key.replace(S3_MODEL_PREFIX, "")
            if not filename:
                continue

            local_path = target_dir / filename
            size_kb = obj["Size"] / 1024
            print(f"  Downloading s3://{self.bucket}/{key} ({size_kb:.1f} KB)")
            self.s3.download_file(self.bucket, key, str(local_path))
            downloaded += 1

        print(f"  ✓ Downloaded {downloaded} artifact(s) from S3")
        return downloaded

    def upload_feedback(self, feedback_records: list[dict]):
        """
        Upload human-reviewed feedback to S3 as a CSV.

        Each record should have:
            - text: the original message
            - true_label: human-corrected label (0=ham, 1=spam)
            - predicted_label: what the model predicted
            - confidence: model's confidence
            - reviewed_at: timestamp

        Args:
            feedback_records: List of feedback dicts
        """
        if not feedback_records:
            print("  No feedback records to upload")
            return None

        # Write to an in-memory CSV
        output = io.StringIO()
        fieldnames = ["text", "true_label", "predicted_label", "confidence", "reviewed_at"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in feedback_records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})

        # Upload with timestamp-based key
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        s3_key = f"{S3_FEEDBACK_PREFIX}feedback_{timestamp}.csv"

        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=output.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )

        print(f"  ✓ Uploaded {len(feedback_records)} feedback records → s3://{self.bucket}/{s3_key}")
        return s3_key

    def list_feedback_files(self) -> list[str]:
        """List all feedback CSVs in S3."""
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=S3_FEEDBACK_PREFIX,
        )

        if "Contents" not in response:
            return []

        return [
            obj["Key"] for obj in response["Contents"]
            if obj["Key"].endswith(".csv")
        ]

    def download_feedback(self) -> list[dict]:
        """
        Download ALL feedback CSVs from S3, merge into one list.

        Returns:
            List of feedback dicts ready for retraining
        """
        files = self.list_feedback_files()
        if not files:
            print("  No feedback files found in S3")
            return []

        all_records = []
        for key in files:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"].read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(body))
            records = list(reader)
            all_records.extend(records)
            print(f"  Loaded {len(records)} records from {key}")

        print(f"  ✓ Total feedback records: {len(all_records)}")
        return all_records


# ─── SQS Manager ────────────────────────────────────────

class SQSManager:
    """Handles the review queue for uncertain predictions."""

    def __init__(self, queue_name: str = SQS_QUEUE, region: str = AWS_REGION):
        self.queue_name = queue_name
        self.region = region
        self.sqs = boto3.client("sqs", region_name=region)
        self._queue_url = None

    @property
    def queue_url(self) -> str:
        """Get or cache the queue URL."""
        if self._queue_url is None:
            response = self.sqs.get_queue_url(QueueName=self.queue_name)
            self._queue_url = response["QueueUrl"]
        return self._queue_url

    def create_queue(self):
        """Create the SQS queue if it doesn't exist."""
        try:
            url = self.queue_url
            print(f"  SQS queue already exists: {self.queue_name}")
            print(f"  URL: {url}")
        except ClientError:
            print(f"  Creating SQS queue: {self.queue_name}")
            response = self.sqs.create_queue(
                QueueName=self.queue_name,
                Attributes={
                    "VisibilityTimeout": str(SQS_VISIBILITY),
                    "MessageRetentionPeriod": "1209600",  # 14 days
                },
            )
            self._queue_url = response["QueueUrl"]
            print(f"  ✓ Queue created: {self._queue_url}")

    def send_for_review(
        self,
        text: str,
        predicted_label: int,
        confidence: float,
        request_id: str = None,
    ) -> str:
        """
        Push an uncertain prediction to the review queue.

        Args:
            text:            The original message
            predicted_label: Model's prediction (0=ham, 1=spam)
            confidence:      Model's confidence score
            request_id:      Optional tracking ID

        Returns:
            SQS MessageId
        """
        message = {
            "text": text,
            "predicted_label": predicted_label,
            "predicted_label_str": "spam" if predicted_label == 1 else "ham",
            "confidence": confidence,
            "request_id": request_id or "",
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }

        response = self.sqs.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(message),
            MessageAttributes={
                "confidence": {
                    "StringValue": str(round(confidence, 4)),
                    "DataType": "Number",
                },
                "predicted_label": {
                    "StringValue": str(predicted_label),
                    "DataType": "Number",
                },
            },
        )

        msg_id = response["MessageId"]
        label_str = "spam" if predicted_label == 1 else "ham"
        print(f"  → Queued for review: [{label_str}] conf={confidence:.2f} | MsgID: {msg_id[:8]}...")
        return msg_id

    def poll_reviews(self, max_messages: int = SQS_MAX_MESSAGES) -> list[dict]:
        """
        Pull messages from the review queue (for human reviewer).

        Returns:
            List of dicts with 'body' (the prediction data) and
            'receipt_handle' (needed to delete after review)
        """
        response = self.sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=min(max_messages, SQS_MAX_MESSAGES),
            WaitTimeSeconds=SQS_WAIT_SECONDS,
            MessageAttributeNames=["All"],
        )

        messages = response.get("Messages", [])
        results = []
        for msg in messages:
            results.append({
                "message_id": msg["MessageId"],
                "receipt_handle": msg["ReceiptHandle"],
                "body": json.loads(msg["Body"]),
            })

        return results

    def delete_message(self, receipt_handle: str):
        """Acknowledge a reviewed message (remove from queue)."""
        self.sqs.delete_message(
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle,
        )

    def get_queue_depth(self) -> dict:
        """Get the number of messages waiting in the queue."""
        attrs = self.sqs.get_queue_attributes(
            QueueUrl=self.queue_url,
            AttributeNames=[
                "ApproximateNumberOfMessages",
                "ApproximateNumberOfMessagesNotVisible",
            ],
        )["Attributes"]

        return {
            "pending": int(attrs.get("ApproximateNumberOfMessages", 0)),
            "in_flight": int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0)),
        }


# ─── Unified Pipeline ──────────────────────────────────

class AWSPipeline:
    """
    Ties together S3 (storage) and SQS (review queue) for the
    active learning feedback loop.
    """

    def __init__(self):
        self.s3 = S3Manager()
        self.sqs = SQSManager()

    def setup(self):
        """Create S3 bucket and SQS queue if they don't exist."""
        print("\n  Setting up AWS resources...")
        self.s3.create_bucket()
        self.sqs.create_queue()
        print("\n  ✓ AWS setup complete!")

    def flag_uncertain(
        self,
        text: str,
        predicted_label: int,
        confidence: float,
        threshold: float = 0.70,
        request_id: str = None,
    ) -> bool:
        """
        After a prediction, check if it's uncertain and route to SQS.

        Args:
            text:            The classified message
            predicted_label: Model's prediction
            confidence:      Model's confidence
            threshold:       Below this → send to queue
            request_id:      Optional tracking ID

        Returns:
            True if the message was sent to SQS (was uncertain)
        """
        if confidence >= threshold:
            return False

        self.sqs.send_for_review(text, predicted_label, confidence, request_id)
        return True

    def submit_review(
        self,
        receipt_handle: str,
        text: str,
        true_label: int,
        predicted_label: int,
        confidence: float,
    ):
        """
        Human reviewer submits a corrected label.

        1. Saves the feedback to S3
        2. Deletes the message from SQS (acknowledges it)

        Args:
            receipt_handle: From poll_reviews (to ack the message)
            text:           The original message
            true_label:     Human's corrected label (0=ham, 1=spam)
            predicted_label: What the model said
            confidence:     Model's original confidence
        """
        feedback = [{
            "text": text,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }]

        self.s3.upload_feedback(feedback)
        self.sqs.delete_message(receipt_handle)
        print(f"  ✓ Review submitted and acknowledged")

    def sync_model(self, direction: str = "upload"):
        """Upload or download model artifacts."""
        if direction == "upload":
            return self.s3.upload_model()
        elif direction == "download":
            return self.s3.download_model()
        else:
            raise ValueError(f"direction must be 'upload' or 'download', got '{direction}'")

    def collect_feedback(self) -> list[dict]:
        """Gather all reviewed feedback from S3 for retraining."""
        return self.s3.download_feedback()

    def status(self):
        """Print the current status of AWS resources."""
        print("\n  AWS Pipeline Status")
        print("  " + "─" * 40)

        # S3 model artifacts
        try:
            response = self.s3.s3.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=S3_MODEL_PREFIX,
            )
            model_count = len(response.get("Contents", []))
            print(f"  S3 model artifacts: {model_count}")
        except ClientError as e:
            print(f"  S3 model check failed: {e}")

        # S3 feedback files
        try:
            feedback_files = self.s3.list_feedback_files()
            print(f"  S3 feedback files:  {len(feedback_files)}")
        except ClientError as e:
            print(f"  S3 feedback check failed: {e}")

        # SQS queue depth
        try:
            depth = self.sqs.get_queue_depth()
            print(f"  SQS pending:        {depth['pending']}")
            print(f"  SQS in-flight:      {depth['in_flight']}")
        except ClientError as e:
            print(f"  SQS check failed: {e}")


# ─── Demo ───────────────────────────────────────────────

def run_demo():
    """
    Demonstrate the full AWS pipeline:
    1. Load the ONNX model
    2. Classify sample messages
    3. Push uncertain ones to SQS
    4. Simulate a human reviewing them
    5. Store feedback in S3
    """
    from step7_inference import SpamClassifier

    print("=" * 60)
    print("STEP 11: AWS S3 + SQS Demo")
    print("=" * 60)

    pipeline = AWSPipeline()

    # ── 1. Upload model to S3 ──
    print("\n── Uploading model to S3 ──")
    pipeline.sync_model("upload")

    # ── 2. Classify messages and flag uncertain ones ──
    print("\n── Classifying messages ──")
    classifier = SpamClassifier()

    test_messages = [
        "Hey, are we still on for lunch?",
        "WINNER! You've been selected for a free prize!",
        "Can you send me the meeting notes?",
        "Call now to claim your reward, limited time only",
        "Pick me up at 5 please",
        "You have won a cash prize, call to claim",
        "Running late, be there in 10",
        "Free entry in our weekly competition, text WIN to 80888",
    ]

    uncertain_count = 0
    for msg in test_messages:
        result = classifier.predict(msg)
        label = result["label"]
        conf = result["confidence"]

        was_flagged = pipeline.flag_uncertain(
            text=msg,
            predicted_label=result["prediction"],
            confidence=conf,
        )

        status = "→ QUEUED FOR REVIEW" if was_flagged else "✓ confident"
        print(f"  [{label:>4s}] conf={conf:.2f} {status:>20s}  {msg[:50]}")
        if was_flagged:
            uncertain_count += 1

    print(f"\n  Flagged {uncertain_count}/{len(test_messages)} for human review")

    # ── 3. Simulate human review ──
    print("\n── Simulating human review ──")
    messages = pipeline.sqs.poll_reviews()
    print(f"  Pulled {len(messages)} message(s) from queue")

    for msg in messages:
        body = msg["body"]
        # In a real system, a human would look at the text and decide.
        # Here we just auto-approve the model's prediction as a demo.
        true_label = body["predicted_label"]
        print(f"  Reviewing: \"{body['text'][:40]}...\" → label={true_label}")

        pipeline.submit_review(
            receipt_handle=msg["receipt_handle"],
            text=body["text"],
            true_label=true_label,
            predicted_label=body["predicted_label"],
            confidence=body["confidence"],
        )

    # ── 4. Check feedback in S3 ──
    print("\n── Feedback stored in S3 ──")
    feedback_files = pipeline.s3.list_feedback_files()
    print(f"  Feedback files: {len(feedback_files)}")
    for f in feedback_files:
        print(f"    {f}")

    # ── 5. Status ──
    pipeline.status()

    print("\n" + "=" * 60)
    print("  STEP 11 COMPLETE!")
    print("=" * 60)


# ─── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 11: AWS S3 + SQS wiring")
    parser.add_argument("--setup", action="store_true",
                        help="Create S3 bucket and SQS queue")
    parser.add_argument("--upload-model", action="store_true",
                        help="Upload trained model artifacts to S3")
    parser.add_argument("--download-model", action="store_true",
                        help="Download model artifacts from S3")
    parser.add_argument("--demo", action="store_true",
                        help="Run the full demo (classify → queue → review → feedback)")
    parser.add_argument("--status", action="store_true",
                        help="Show current AWS resource status")
    parser.add_argument("--collect-feedback", action="store_true",
                        help="Download all feedback from S3")
    args = parser.parse_args()

    try:
        pipeline = AWSPipeline()

        if args.setup:
            pipeline.setup()
        elif args.upload_model:
            print("Uploading model to S3...")
            pipeline.sync_model("upload")
        elif args.download_model:
            print("Downloading model from S3...")
            pipeline.sync_model("download")
        elif args.demo:
            run_demo()
        elif args.status:
            pipeline.status()
        elif args.collect_feedback:
            records = pipeline.collect_feedback()
            print(f"\nCollected {len(records)} feedback records")
        else:
            parser.print_help()

    except NoCredentialsError:
        print("\n  ✗ AWS credentials not found!")
        print("    Configure them with one of:")
        print("      aws configure")
        print("      set AWS_ACCESS_KEY_ID=...")
        print("      set AWS_SECRET_ACCESS_KEY=...")
        print("      set AWS_DEFAULT_REGION=...")
        sys.exit(1)
    except BotoCoreError as e:
        print(f"\n  ✗ AWS error: {e}")
        sys.exit(1)

    print("\n+ Step 11 complete!")
