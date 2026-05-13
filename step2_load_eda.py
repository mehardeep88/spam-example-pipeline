"""
STEP 2: Load the dataset and perform Exploratory Data Analysis (EDA).

What is EDA?
    Exploratory Data Analysis — understanding your data BEFORE building models.
    You look at distributions, class balance, text lengths, common words, etc.
    This prevents surprises later (e.g., "oh, 99% of my data is ham").

What this script does:
    1. Loads the raw TSV file into a pandas DataFrame
    2. Shows basic statistics (shape, class distribution)
    3. Analyzes text length distribution
    4. Finds most common words in spam vs ham
    5. Saves summary charts to plots/ directory
"""
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config


def load_raw() -> pd.DataFrame:
    """
    Load the raw SMS Spam Collection file.
    
    The file format is:
        ham\tGo until jurong point, crazy.. Available only ...
        spam\tFree entry in 2 a wkly comp to win FA Cup ...
    
    - Tab-separated (\t)
    - No header row
    - Column 1: label ('ham' or 'spam')
    - Column 2: the SMS message text
    
    Returns:
        DataFrame with columns: ['label', 'text']
    """
    if not config.RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found at {config.RAW_FILE}\n"
            "Run step1_download.py first!"
        )
    
    df = pd.read_csv(
        config.RAW_FILE,
        sep='\t',              # Tab-separated
        header=None,           # No header row in the file
        names=['label', 'text'],  # We name the columns ourselves
        encoding='latin-1',    # Some SMS have special characters
        on_bad_lines='skip'    # Skip any malformed lines
    )
    
    print(f"Loaded {len(df):,} messages from {config.RAW_FILE.name}")
    return df


def run_eda(df: pd.DataFrame):
    """
    Run exploratory data analysis and print findings.
    """
    print("=" * 50)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 50)
    
    # ─── 1. Basic Shape ─────────────────────────────────
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Dtypes:\n{df.dtypes.to_string()}")
    
    # ─── 2. Class Distribution ──────────────────────────
    print(f"\n📊 Class Distribution:")
    counts = df['label'].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"   {label:>4s}: {count:,} ({pct:.1f}%)")
    
    spam_ratio = counts.get('spam', 0) / len(df)
    print(f"\n   ⚠ Class imbalance: spam is only {spam_ratio:.1%} of the data")
    print(f"   → We'll use class_weight='balanced' in the classifier to handle this")
    
    # ─── 3. Text Length Analysis ────────────────────────
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\n📊 Text Length Statistics:")
    for label in ['ham', 'spam']:
        subset = df[df['label'] == label]
        print(f"\n   {label.upper()}:")
        print(f"     Avg characters: {subset['text_length'].mean():.0f}")
        print(f"     Avg words:      {subset['word_count'].mean():.0f}")
        print(f"     Min characters: {subset['text_length'].min()}")
        print(f"     Max characters: {subset['text_length'].max()}")
    
    # ─── 4. Sample Messages ────────────────────────────
    print(f"\n📊 Sample HAM messages:")
    for _, row in df[df['label'] == 'ham'].head(3).iterrows():
        print(f"   → {row['text'][:80]}...")
    
    print(f"\n📊 Sample SPAM messages:")
    for _, row in df[df['label'] == 'spam'].head(3).iterrows():
        print(f"   → {row['text'][:80]}...")
    
    # ─── 5. Most Common Words ──────────────────────────
    print(f"\n📊 Top 15 Words in SPAM:")
    spam_words = ' '.join(df[df['label'] == 'spam']['text']).lower().split()
    for word, count in Counter(spam_words).most_common(15):
        print(f"   {word:>15s}: {count}")
    
    print(f"\n📊 Top 15 Words in HAM:")
    ham_words = ' '.join(df[df['label'] == 'ham']['text']).lower().split()
    for word, count in Counter(ham_words).most_common(15):
        print(f"   {word:>15s}: {count}")

    # ─── 6. Missing Values Check ───────────────────────
    print(f"\n📊 Missing Values:")
    print(f"   {df.isnull().sum().to_string()}")
    
    # ─── 7. Duplicate Check ────────────────────────────
    dupes = df.duplicated(subset='text').sum()
    print(f"\n📊 Duplicates: {dupes} duplicate messages found")
    
    return df


def save_plots(df: pd.DataFrame):
    """Generate and save EDA charts."""
    
    if 'text_length' not in df.columns:
        df['text_length'] = df['text'].str.len()
    if 'word_count' not in df.columns:
        df['word_count'] = df['text'].str.split().str.len()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SMS Spam Collection — EDA', fontsize=16, fontweight='bold')
    
    # Plot 1: Class distribution bar chart
    ax = axes[0, 0]
    counts = df['label'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='white')
    ax.set_title('Class Distribution', fontweight='bold')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', fontweight='bold')
    
    # Plot 2: Text length distribution (histogram)
    ax = axes[0, 1]
    ax.hist(df[df['label'] == 'ham']['text_length'], bins=50, alpha=0.7,
            label='Ham', color='#2ecc71', edgecolor='white')
    ax.hist(df[df['label'] == 'spam']['text_length'], bins=50, alpha=0.7,
            label='Spam', color='#e74c3c', edgecolor='white')
    ax.set_title('Message Length Distribution', fontweight='bold')
    ax.set_xlabel('Characters')
    ax.set_ylabel('Count')
    ax.legend()
    
    # Plot 3: Word count distribution (box plot)
    ax = axes[1, 0]
    ham_wc = df[df['label'] == 'ham']['word_count']
    spam_wc = df[df['label'] == 'spam']['word_count']
    bp = ax.boxplot([ham_wc, spam_wc], labels=['Ham', 'Spam'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_title('Word Count by Class', fontweight='bold')
    ax.set_ylabel('Words')
    
    # Plot 4: Top spam words (horizontal bar)
    ax = axes[1, 1]
    spam_words = ' '.join(df[df['label'] == 'spam']['text']).lower().split()
    common = Counter(spam_words).most_common(10)
    words, freqs = zip(*common)
    y_pos = np.arange(len(words))
    ax.barh(y_pos, freqs, color='#e74c3c', edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title('Top 10 Spam Words', fontweight='bold')
    ax.set_xlabel('Frequency')
    
    plt.tight_layout()
    plot_path = config.PLOTS_DIR / "eda_overview.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ EDA plots saved to: {plot_path}")


if __name__ == "__main__":
    df = load_raw()
    df = run_eda(df)
    save_plots(df)
    print("\n✓ Step 2 complete!")
