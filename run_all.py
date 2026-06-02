"""
Run the entire SMS Spam pipeline end-to-end.

Usage:
    python run_all.py          # Run all steps (1-7, core pipeline)
    python run_all.py 3 5      # Run only steps 3 through 5
    python run_all.py 8        # Run just the API server (step 8)
    python run_all.py 9        # Run MLflow tracking (step 9)
    python run_all.py 10       # Run Active Learning simulation (step 10)
    python run_all.py 11       # Run AWS S3+SQS demo (step 11)

Note:
    Steps 8-11 are separate phases and are best run individually:
    - Step 8 starts a server (blocks until Ctrl+C)
    - Step 9 needs mlflow installed
    - Step 10 takes a few minutes (trains ~40 models)
    - Step 11 needs boto3 + AWS credentials configured
"""
import sys

from step1_download import download
from step2_load_eda import load_raw, run_eda, save_plots
from step3_clean import preprocess_and_split
from step4_train import train
from step5_evaluate import evaluate
from step6_onnx_export import export_onnx
from step7_inference import demo


def run_all(start=1, end=7):
    """Run pipeline steps from start to end (inclusive)."""

    steps = {
        1: ("Download Dataset", download),
        2: ("Load & EDA", lambda: save_plots(run_eda(load_raw()))),
        3: ("Clean & Split", preprocess_and_split),
        4: ("Train Model", train),
        5: ("Evaluate on Test Set", evaluate),
        6: ("Export to ONNX", export_onnx),
        7: ("ONNX Inference Demo", demo),
        8: ("FastAPI Server", _run_api),
        9: ("MLflow Tracking", _run_mlflow),
        10: ("Active Learning Simulation", _run_active_learning),
        11: ("AWS S3 + SQS Pipeline", _run_aws),
    }

    print("\n" + "=" * 60)
    print("  SMS SPAM CLASSIFICATION PIPELINE")
    print(f"  Running steps {start} through {end}")
    print("=" * 60 + "\n")

    for step_num in range(start, end + 1):
        if step_num in steps:
            name, fn = steps[step_num]
            print(f"\n{'='*60}")
            print(f">>> STEP {step_num}: {name}")
            print(f"{'='*60}")
            fn()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)


def _run_api():
    """Start the FastAPI server (step 8)."""
    from step8_api import app
    import uvicorn
    print("  Starting API server...")
    print("  Docs:   http://localhost:8000/docs")
    print("  Health: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def _run_mlflow():
    """Run training with MLflow tracking (step 9)."""
    from step9_mlflow import train_with_tracking
    train_with_tracking()


def _run_active_learning():
    """Run the AL vs Random simulation (step 10)."""
    from step10_active_learning import run_simulation
    run_simulation()


def _run_aws():
    """Run the AWS S3 + SQS demo (step 11)."""
    from step11_aws import run_demo
    run_demo()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_all(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 2:
        step = int(sys.argv[1])
        run_all(step, step)
    else:
        run_all()
