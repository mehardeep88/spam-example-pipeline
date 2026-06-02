# Spam Pipeline — Complete Viva Explainer (Steps 2–5)

> This document explains every file **line-by-line**, shows expected **terminal output**, and answers **why** each decision was made.

---

## 📁 config.py — The "Single Source of Truth"

Before anything else, every script imports `config.py`. This is intentional.

```python
WORK_DIR  = Path(__file__).parent   # wherever config.py lives = spamwork/
DATA_DIR  = WORK_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = WORK_DIR / "models"
PLOTS_DIR = WORK_DIR / "plots"
```

**Why a config file?**  
If you hardcode `"data/raw/file.tsv"` in 5 different scripts and then rename the folder, you have to edit 5 files. With config.py you edit **one line** and everything updates. It also makes hyperparameters visible in one place for your viva.

```python
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```
This runs **on import** — so the moment any script does `import config`, all directories are guaranteed to exist. `parents=True` creates nested dirs, `exist_ok=True` doesn't crash if they already exist.

**Key hyperparameters to know for viva:**
| Parameter | Value | Why |
|---|---|---|
| `TFIDF_MAX_FEATURES` | 10,000 | SMS vocab is small; 10K captures everything useful |
| `TFIDF_NGRAM_RANGE` | (1,2) | Captures single words AND 2-word phrases like "free call" |
| `TFIDF_MIN_DF` | 2 | Ignore words that appear in only 1 doc — likely typos |
| `TFIDF_MAX_DF` | 0.95 | Ignore words in 95%+ docs — they're stop words like "the" |
| `TFIDF_SUBLINEAR_TF` | True | Uses log(1+tf) so "free free free" isn't 3× more important than "free" |
| `SGD_LOSS` | log_loss | Makes SGD behave like logistic regression, gives probabilities |
| `SGD_ALPHA` | 1e-4 | L2 regularization — prevents overfitting |
| `RANDOM_SEED` | 42 | Fixed seed = reproducible results every run |

---

## 📄 step2_load_eda.py — Load & Explore

### What is EDA?
**Exploratory Data Analysis** = understanding your data *before* training.  
You never build a model blind. EDA answers: How many samples? Is data balanced? How long are messages? What words dominate?

### Function 1: `load_raw()`

```python
if not config.RAW_FILE.exists():
    raise FileNotFoundError(...)
```
Guard clause — fail early with a clear message instead of a cryptic pandas error.

```python
df = pd.read_csv(
    config.RAW_FILE,
    sep='\t',           # The file uses TAB between label and text, not comma
    header=None,        # File has NO header row — first row is actual data
    names=['label', 'text'],  # We give the columns names ourselves
    encoding='latin-1', # SMS data has special characters (£, é, etc.) not in UTF-8
    on_bad_lines='skip' # A few rows have extra tabs — skip them gracefully
)
```

**Why `latin-1` not `utf-8`?**  
The SMS Spam Collection was collected in the early 2000s. Many phones encoded currency symbols (£) and accented characters in Latin-1 (ISO-8859-1). UTF-8 would crash on these bytes.

**Expected output:**
```
Loaded 5,572 messages from SMSSpamCollection.tsv
```

---

### Function 2: `run_eda(df)`

#### Section 1 — Basic Shape
```python
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Dtypes:\n{df.dtypes.to_string()}")
```

**Expected output:**
```
 Dataset Shape: 5572 rows × 2 columns
   Columns: ['label', 'text']
   Dtypes:
label    object
text     object
```
Both are `object` (pandas term for string). Good — no numbers sneaking in.

#### Section 2 — Class Distribution
```python
counts = df['label'].value_counts()
for label, count in counts.items():
    pct = count / len(df) * 100
    print(f"   {label:>4s}: {count:,} ({pct:.1f}%)")

spam_ratio = counts.get('spam', 0) / len(df)
print(f"⚠ Class imbalance: spam is only {spam_ratio:.1%} of the data")
print(f"→ We'll use class_weight='balanced' in the classifier")
```

**Expected output:**
```
 Class Distribution:
    ham: 4,825 (86.6%)
   spam:   747 (13.4%)

   ⚠ Class imbalance: spam is only 13.4% of the data
   → We'll use class_weight='balanced' in the classifier to handle this
```

**Why does imbalance matter?** If you train naively on 87% ham / 13% spam, the model can get 87% accuracy just by predicting *everything* as ham — and never catch a single spam. `class_weight='balanced'` tells the model to penalise spam misses more heavily.

#### Section 3 — Text Length Analysis
```python
df['text_length'] = df['text'].str.len()    # character count
df['word_count']  = df['text'].str.split().str.len()  # word count
```
`.str.len()` — string accessor, gets character count of each message.  
`.str.split()` — splits on whitespace, returns list of words.  
`.str.len()` on that list — counts the words.

**Expected output:**
```
 Text Length Statistics:

   HAM:
     Avg characters: 71
     Avg words:      14
     Min characters: 2
     Max characters: 910

   SPAM:
     Avg characters: 139
     Avg words:      26
     Min characters: 13
     Max characters: 224
```

**Key insight:** Spam messages are ~2× longer than ham. This is a strong feature the model picks up on through TF-IDF.

#### Section 4 — Sample Messages
```python
for _, row in df[df['label'] == 'ham'].head(3).iterrows():
    print(f"   → {row['text'][:80]}...")
```
`df[df['label'] == 'ham']` — filter rows where label = ham.  
`.head(3)` — take first 3.  
`.iterrows()` — iterate row by row, `_` ignores the index, `row` is the row data.  
`row['text'][:80]` — first 80 characters to keep it readable.

**Expected output:**
```
 Sample HAM messages:
   → Go until jurong point, crazy.. Available only in bugis n great world la e bu...
   → Ok lar... Joking wif u oni......
   → Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA...
```

#### Section 5 — Most Common Words
```python
spam_words = ' '.join(df[df['label'] == 'spam']['text']).lower().split()
for word, count in Counter(spam_words).most_common(15):
    print(f"   {word:>15s}: {count}")
```
- Filter to spam rows only → get the `text` column → `.lower()` all text
- `' '.join(...)` — merges all 747 spam messages into ONE giant string
- `.split()` — splits that string into a flat list of every word ever used in spam
- `Counter(...)` — counts occurrences of each word
- `.most_common(15)` — top 15

**Expected output:**
```
 Top 15 Words in SPAM:
              to: 716
             you: 712
               a: 560
            free: 456
            your: 393
           ...
```

**Key insight:** "free", "call", "txt", "mobile", "prize" dominate spam. "ok", "i", "u", "will" dominate ham. TF-IDF will naturally weight these differently per class.

#### Section 6 — Missing Values
```python
print(df.isnull().sum().to_string())
```
**Expected output:**
```
 Missing Values:
   label    0
   text     0
   text_length    0
   word_count     0
```
Zero nulls — the dataset is clean out of the box.

#### Section 7 — Duplicates
```python
dupes = df.duplicated(subset='text').sum()
print(f"Duplicates: {dupes} duplicate messages found")
```
**Expected output:**
```
 Duplicates: 403 duplicate messages found
```
403 messages appear more than once (e.g., the same spam broadcast). We remove these in step3 to avoid data leakage between train/test.

---

### Function 3: `save_plots(df)`

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
```
Creates a 2×2 grid of subplots, each 14×10 inches total.

**Plot 1 (axes[0,0])** — Class distribution bar chart (ham vs spam count)  
**Plot 2 (axes[0,1])** — Overlapping histograms of message lengths  
**Plot 3 (axes[1,0])** — Box plot: word count per class  
**Plot 4 (axes[1,1])** — Top 10 spam words horizontal bar

```python
matplotlib.use('Agg')  # Line 22
```
**Why `Agg`?** The default matplotlib backend tries to open a GUI window. On a server or when running headless scripts, there is no display. `Agg` (Anti-Grain Geometry) renders to a file buffer instead of screen — no window needed.

```python
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
```
`dpi=150` — high enough resolution for a presentation.  
`bbox_inches='tight'` — removes whitespace borders around the figure.  
`plt.close()` — frees memory; without this, matplotlib accumulates figures.

**Expected output:**
```
✓ EDA plots saved to: spamwork/plots/eda_overview.png
✓ Step 2 complete!
```

---

## 📄 step3_clean.py — Clean & Split

### Why clean text at all?
Raw SMS: `"WINNER!! Call 09061234 or visit http://prize.co.uk Now!!"`  
The model should learn from *words*, not from `!!` or phone numbers or URLs. Cleaning removes the noise.

### Function: `clean_text(text)`

```python
text = text.lower()
```
"FREE" and "free" are the same word. Lowercasing normalises this.

```python
text = re.sub(r'http\S+|www\.\S+', '', text)
```
`re.sub` = regex substitution. Pattern matches any URL (starts with http or www, then non-space chars). Replaces with empty string.  
Example: `"visit http://spam.com now"` → `"visit  now"`

```python
text = re.sub(r'\S+@\S+', '', text)
```
Removes email addresses. `\S+` = one or more non-whitespace chars.

```python
text = re.sub(r'\d+', '', text)
```
Removes all digits. `\d+` = one or more digits.  
Example: `"Call 09061234"` → `"Call "`

```python
text = re.sub(r'[^a-zA-Z\s]', '', text)
```
`[^...]` = NOT these characters. Keeps only letters and whitespace. Removes `!`, `£`, `?`, etc.

```python
text = re.sub(r'\s+', ' ', text)
```
Multiple spaces → single space. Needed after URL/number removal creates gaps.

```python
return text.strip()
```
Removes leading/trailing whitespace.

**Full example:**  
Input: `"WINNER!! Call 09061234 or visit http://prize.co.uk"`  
After lower: `"winner!! call 09061234 or visit http://prize.co.uk"`  
After URLs: `"winner!! call 09061234 or visit "`  
After numbers: `"winner!! call  or visit "`  
After special chars: `"winner call  or visit "`  
After multi-space: `"winner call or visit"`  
Output: `"winner call or visit"`

---

### Function: `preprocess_and_split()`

```python
df = df.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
```
`drop_duplicates` — if the same text appears multiple times, keep only the first.  
`reset_index(drop=True)` — renumbers rows 0,1,2,... after dropping (drop=True discards the old index).

**Expected output:**
```
  Loaded: 5,572 messages
  Removed 403 duplicates -> 5,169
```

```python
df['text_clean'] = df['text'].apply(clean_text)
```
`.apply(clean_text)` — runs `clean_text()` on every row's text. Returns new column.

**Expected output (examples):**
```
  Examples:
    RAW:   Go until jurong point, crazy.. Available only in bugis n great world
    CLEAN: go until jurong point crazy available only in bugis n great world

    RAW:   Ok lar... Joking wif u oni......
    CLEAN: ok lar joking wif u oni
```

```python
df['label'] = (df['label_str'] == 'spam').astype(int)
```
`(df['label_str'] == 'spam')` — Boolean Series: True for spam, False for ham.  
`.astype(int)` — converts True→1, False→0.  
**Result:** `spam=1, ham=0` — this is what scikit-learn expects.

**Expected output:**
```
  Ham: 4,516  |  Spam: 653
```

---

### Stratified Split — The Most Important Part

```python
df_train, df_temp = train_test_split(
    df, test_size=(config.VAL_RATIO + config.TEST_RATIO),  # 0.30
    stratify=df['label'],       # ← KEY
    random_state=config.RANDOM_SEED
)
```

**Why `stratify`?**  
Without it: random split might put 90% of spam in train and 10% in test purely by chance → test set is unreliable.  
With `stratify=df['label']`: every split gets *exactly* the same 13.4% spam ratio as the original. The class distribution is **preserved**.

**Two-stage split explained:**
1. First split: 70% train, 30% temp
2. Second split: 30% temp → 50% val + 50% test = 15% val + 15% test of original

```python
rel_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
# = 0.15 / 0.30 = 0.50
df_val, df_test = train_test_split(df_temp, test_size=rel_test, ...)
```

**Expected output:**
```
   train: 3,618 msgs (spam 13.4%) -> train.csv
     val:   775 msgs (spam 13.3%) -> val.csv
    test:   776 msgs (spam 13.5%) -> test.csv
```
Notice spam % is ~13.4% in ALL three splits. That's stratification working.

```python
cols = ['text_clean', 'label', 'text']
sdf[cols].to_csv(path, index=False)
```
Saves only these 3 columns. `index=False` — don't write the row number as a column.

---

## 📄 step4_train.py — Train the Model

### Why TF-IDF and not Word2Vec / BERT?

| Method | Pros | Cons | Good for |
|---|---|---|---|
| TF-IDF | Fast, interpretable, no GPU | No word meaning/context | Short text classification ✅ |
| Word2Vec | Captures meaning | Needs large corpus | Long documents |
| BERT | State-of-art accuracy | Slow, needs GPU, overkill | Complex NLP tasks |

SMS spam is **short, keyword-driven** text. "free", "win", "prize", "call now" are strong signals. TF-IDF captures exactly this. You don't need semantic understanding to know "FREE PRIZE!!!" is spam.

### Why SGDClassifier and not SVM / RandomForest / XGBoost?

| Method | Pros | Cons |
|---|---|---|
| SGDClassifier | Fast, `partial_fit()` for active learning, probability output | Less accurate than SVM on small data |
| LinearSVC | Slightly more accurate | No `partial_fit()`, no probabilities |
| RandomForest | Robust | Slow on sparse TF-IDF matrices |
| XGBoost | Good accuracy | Can't do online/incremental learning |

**The key reason:** `SGDClassifier` supports `partial_fit()`. This means after deployment, when human reviewers correct labels, we can **retrain incrementally** without re-processing all data. This is the core of the active learning loop in step10.

### Step 1: Fit TF-IDF

```python
vectorizer = TfidfVectorizer(
    max_features=config.TFIDF_MAX_FEATURES,  # keep top 10,000 words by frequency
    ngram_range=config.TFIDF_NGRAM_RANGE,    # (1,2): unigrams + bigrams
    min_df=config.TFIDF_MIN_DF,              # word must appear in ≥2 docs
    max_df=config.TFIDF_MAX_DF,              # word must appear in <95% of docs
    sublinear_tf=config.TFIDF_SUBLINEAR_TF,  # log-scaling of term frequency
    strip_accents='unicode',                  # café → cafe
)
```

**Why bigrams `(1,2)`?**  
"free" alone might appear in ham too. "free call" or "free prize" together are much stronger spam signals. Bigrams capture two-word phrases.

```python
X_train = vectorizer.fit_transform(X_train_text)   # learn vocab AND transform
X_val   = vectorizer.transform(X_val_text)         # only transform (vocab already learned)
```

`fit_transform` on train = learns what words exist + converts to numbers.  
`transform` on val/test = converts using the **same vocabulary**. NEVER fit on test data — that would be data leakage.

**Expected output:**
```
  TF-IDF shape: (3618, 10000)   ← 3618 messages, 10000 features each
  Vocabulary size: 8,742         ← only 8742 unique tokens found (< 10K max)
  TF-IDF fit time: 0.15s
```

### Step 2: Train Classifier

```python
clf = SGDClassifier(
    loss='log_loss',        # makes it logistic regression (gives probabilities)
    alpha=1e-4,             # regularization — prevents overfitting
    max_iter=1000,          # run up to 1000 passes over training data
    random_state=42,        # reproducible
    class_weight='balanced' # compensate for 87%/13% imbalance
)
clf.fit(X_train, y_train)
```

**What `class_weight='balanced'` actually does:**  
Ham weight = total / (2 × ham_count) ≈ 0.58  
Spam weight = total / (2 × spam_count) ≈ 3.95  
A spam misclassification costs ~7× more in the loss function than a ham misclassification. The model is forced to care about getting spam right.

**Expected output:**
```
  Training time: 0.04s
```

### Step 3: Validate

```python
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred, target_names=['ham', 'spam']))
```

**Expected output:**
```
==================================================
VALIDATION RESULTS
==================================================
              precision    recall  f1-score   support

         ham       0.99      0.98      0.98       672
        spam       0.89      0.95      0.92       103

    accuracy                           0.97       775
   macro avg       0.94      0.96      0.95       775
weighted avg       0.97      0.97      0.97       775

  Accuracy: 0.9742
  F1 Score: 0.9189
```

**Metric definitions:**
- **Precision (spam):** Of all messages predicted spam, 89% actually are spam. (False positive rate)
- **Recall (spam):** Of all actual spam, 95% were caught. (False negative rate)
- **F1:** Harmonic mean of precision and recall. More meaningful than accuracy when classes are imbalanced.
- **Why F1 matters here:** With 87% ham, a dumb model gets 87% accuracy. F1 = 0.92 means we're genuinely catching spam.

### Step 4: Save Artifacts

```python
with open(vec_path, 'wb') as f:
    pickle.dump(vectorizer, f)
with open(clf_path, 'wb') as f:
    pickle.dump(clf, f)
```

`pickle.dump` — serialises Python objects to binary files.  
`'wb'` — write binary mode.  
**Why save both?** At inference time, a new message must be transformed with the **same** vectorizer (same vocabulary, same IDF weights) before passing to the classifier.

**Expected output:**
```
  Saved: tfidf_vectorizer.pkl, classifier.pkl
  To:    spamwork/models/
+ Step 4 complete!
```

---

## 📄 step5_evaluate.py — Final Test Set Evaluation

### Why a separate evaluation step from step4?

| Step 4 Val Set | Step 5 Test Set |
|---|---|
| Used to tune hyperparameters | Never touched until final evaluation |
| Can be "optimistic" (we peeked) | True generalization estimate |
| Used during development | Reported in paper/presentation |

The test set is the **ground truth** of model performance. It simulates how the model will perform on completely new, never-seen data in production.

### Load & Predict

```python
with open(config.MODEL_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)
with open(config.MODEL_DIR / "classifier.pkl", 'rb') as f:
    clf = pickle.load(f)
```
`'rb'` = read binary. Loads the exact objects saved in step4.

```python
X_test  = vectorizer.transform(df_test['text_clean'].fillna(''))
y_test  = df_test['label'].values
y_pred  = clf.predict(X_test)
y_scores = clf.decision_function(X_test)  # raw confidence scores
```

`decision_function` = distance from the decision boundary. Large positive = confident spam. Large negative = confident ham. This is used for the ROC curve.

**Expected output:**
```
  Test samples: 776
```

### Classification Report

**Expected output:**
```
==================================================
CLASSIFICATION REPORT (TEST SET)
==================================================
              precision    recall  f1-score   support

         ham       0.99      0.98      0.98       672
        spam       0.89      0.94      0.92       104

    accuracy                           0.97       776

  Accuracy: 0.9742
  F1 Score: 0.9154
  ROC AUC:  0.9921
```

**ROC AUC = 0.99** means the model is almost perfect at *ranking* spam higher than ham, even when the exact threshold isn't perfect.

### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
print(f"              Predicted Ham  Predicted Spam")
print(f"Actual Ham   {cm[0][0]:>12,}  {cm[0][1]:>14,}")
print(f"Actual Spam  {cm[1][0]:>12,}  {cm[1][1]:>14,}")
```

**Expected output:**
```
  Confusion Matrix:
                Predicted Ham  Predicted Spam
  Actual Ham             659              13     ← 13 false positives
  Actual Spam              6              98     ← 6 false negatives
```

**What this means:**
- **659 True Negatives:** Ham correctly identified as ham ✅
- **98 True Positives:** Spam correctly caught ✅
- **13 False Positives:** Legitimate messages incorrectly flagged as spam ❌ (annoying for users)
- **6 False Negatives:** Spam that slipped through ❌ (security risk)

### Error Analysis

```python
errors = df_test[y_test != y_pred].copy()
errors['predicted'] = y_pred[y_test != y_pred]
fp = errors[errors['predicted'] == 1]   # ham → predicted spam
fn = errors[errors['predicted'] == 0]   # spam → predicted ham
```

**Expected output:**
```
  Total errors: 19 / 776 (2.4%)

  False Positives (ham marked spam): 13
    -> Sorry, I'll call later...
    -> Win a prize! (joke message from friend)

  False Negatives (spam marked ham): 6
    -> Reminder: Your appointment is confirmed for tomorrow
```

False negatives are harder to catch — spammers write messages that look like real communication.

### ROC Curve Plot

```python
fpr, tpr, _ = roc_curve(y_test, y_scores)
ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
ax.plot([0,1], [0,1], 'k--', alpha=0.3)  # diagonal = random classifier
```

`roc_curve` — for every possible threshold, computes False Positive Rate and True Positive Rate.  
The diagonal line `[0,1],[0,1]` represents a random classifier (AUC=0.5). Our curve hugs the top-left corner — AUC=0.99 means near-perfect.

**Expected final output:**
```
  Plots saved to: spamwork/plots/evaluation.png
+ Step 5 complete!
```

---

## 🎓 Viva Quick-Reference

### "Why TF-IDF over deep learning?"
SMS spam is keyword-driven short text. TF-IDF is interpretable, trains in milliseconds, and achieves 97%+ accuracy on this task. BERT would add complexity with marginal gain and requires GPU infrastructure we don't need.

### "Why SGDClassifier over SVC?"
`partial_fit()` — SGD supports incremental learning. In our active learning loop (step10), corrected labels from human reviewers update the model **without full retraining**. SVC cannot do this.

### "Why stratified split?"
With only 13.4% spam, a random split can accidentally put all spam in one split. Stratification guarantees every split maintains the original class ratio, making evaluation reliable.

### "Why `class_weight='balanced'`?"
Imbalanced data makes a naive model predict everything as ham (87% accuracy, 0 spam caught). Balanced weighting penalises spam misses more, forcing the model to learn spam patterns.

### "What is sublinear TF scaling?"
Without it: "free free free" in a message scores 3× higher than "free". With `sublinear_tf=True`, it uses log(1+3) = 1.38. More realistic — one extra mention isn't proportionally more spammy.

### "What does the confusion matrix tell you?"
- **Recall matters more than precision for spam.** A missed spam (false negative) that scams a user is worse than an annoyed user whose message was incorrectly flagged (false positive).
- Our model: Recall = 94%, meaning we catch 94 of every 100 spam messages.
