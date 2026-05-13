"""
Run the entire SMS Spam pipeline end-to-end.

Usage:
    python run_all.py          # Run all steps
    python run_all.py 3 5      # Run only steps 3 through 5
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


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_all(int(sys.argv[1]), int(sys.argv[2]))
    else:
        run_all()
