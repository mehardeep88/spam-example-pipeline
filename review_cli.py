"""
Interactive CLI to review messages from SQS and save feedback to S3.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from step11_aws import AWSPipeline

def run_reviewer():
    pipeline = AWSPipeline()
    
    print("\n" + "="*50)
    print("  HUMAN REVIEW TERMINAL")
    print("="*50)
    
    total_reviewed = 0
    
    while True:
        print("\nPolling SQS for messages... (this takes a few seconds)")
        # Pull up to 10 messages at a time
        messages = pipeline.sqs.poll_reviews(max_messages=10)
        
        if not messages:
            print("\nNo messages currently in the queue!")
            print(f"Total reviewed this session: {total_reviewed}")
            break
            
        for msg in messages:
            body = msg["body"]
            text = body["text"]
            model_pred = "spam" if body["predicted_label"] == 1 else "ham"
            conf = body["confidence"]
            
            print("\n" + "─"*50)
            print(f"TEXT:  {text}")
            print(f"MODEL: {model_pred.upper()} (Confidence: {conf:.1%})")
            print("─"*50)
            
            while True:
                choice = input("Your label? [1 for SPAM, 0 for HAM, 's' to skip, 'q' to quit]: ").strip().lower()
                
                if choice == 'q':
                    print(f"\nQuitting. Total reviewed: {total_reviewed}")
                    return
                elif choice == 's':
                    print("Skipped. Leaving in queue.")
                    break
                elif choice in ['0', '1']:
                    true_label = int(choice)
                    
                    # This saves it to S3 and deletes it from SQS
                    pipeline.submit_review(
                        receipt_handle=msg["receipt_handle"],
                        text=text,
                        true_label=true_label,
                        predicted_label=body["predicted_label"],
                        confidence=conf
                    )
                    total_reviewed += 1
                    break
                else:
                    print("Invalid input.")

if __name__ == "__main__":
    try:
        run_reviewer()
    except KeyboardInterrupt:
        print("\nExiting...")
