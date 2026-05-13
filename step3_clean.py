"""
STEP 3: Clean, preprocess, and split the data.

What is preprocessing?
    Raw text is messy — URLs, phone numbers, special characters.
    We clean it so the model focuses on actual words, not noise.

What this script does:
    1. Remove duplicates
    2. Clean text (lowercase, remove URLs, numbers, special chars)
    3. Convert labels: 'ham' -> 0, 'spam' -> 1
    4. Stratified split into train/val/test (70/15/15)
    5. Save cleaned splits as CSV

Why stratified splitting?
    Only ~13% of messages are spam. Stratified split ensures
    each split has the same ~13% spam ratio as the original.
"""
import sys
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
import config


def clean_text(text: str) -> str:
    """
    Clean a single SMS message.

    Example:
        Input:  "WINNER!! Call 09061234 or visit http://prize.co.uk"
        Output: "winner call or visit"
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)   # URLs
    text = re.sub(r'\S+@\S+', '', text)              # Emails
    text = re.sub(r'\d+', '', text)                   # Numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # Special chars
    text = re.sub(r'\s+', ' ', text)                  # Multi-spaces
    return text.strip()


def preprocess_and_split():
    """Load raw data, clean it, split it, save it."""

    print("=" * 50)
    print("STEP 3: Cleaning & Splitting")
    print("=" * 50)

    # Load raw
    df = pd.read_csv(
        config.RAW_FILE, sep='\t', header=None,
        names=['label_str', 'text'],
        encoding='latin-1', on_bad_lines='skip'
    )
    print(f"  Loaded: {len(df):,} messages")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
    print(f"  Removed {before - len(df)} duplicates -> {len(df):,}")

    # Clean text
    print("  Cleaning text...")
    df['text_clean'] = df['text'].apply(clean_text)

    # Show examples
    print("\n  Examples:")
    for i in range(min(2, len(df))):
        print(f"    RAW:   {df.iloc[i]['text'][:70]}")
        print(f"    CLEAN: {df.iloc[i]['text_clean'][:70]}\n")

    # Drop empty after cleaning
    before = len(df)
    df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)
    print(f"  Removed {before - len(df)} empty messages")

    # Convert labels: ham=0, spam=1
    df['label'] = (df['label_str'] == 'spam').astype(int)
    print(f"  Ham: {(df['label']==0).sum():,}  |  Spam: {(df['label']==1).sum():,}")

    # Stratified split
    df_train, df_temp = train_test_split(
        df, test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=df['label'], random_state=config.RANDOM_SEED
    )
    rel_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    df_val, df_test = train_test_split(
        df_temp, test_size=rel_test,
        stratify=df_temp['label'], random_state=config.RANDOM_SEED
    )

    # Save
    cols = ['text_clean', 'label', 'text']
    for name, sdf in [('train', df_train), ('val', df_val), ('test', df_test)]:
        path = config.PROCESSED_DIR / f"{name}.csv"
        sdf[cols].to_csv(path, index=False)
        spam_pct = sdf['label'].mean() * 100
        print(f"  {name:>5s}: {len(sdf):,} msgs (spam {spam_pct:.1f}%) -> {path.name}")

    return df_train, df_val, df_test


if __name__ == "__main__":
    preprocess_and_split()
    print("\n+ Step 3 complete!")
