# %% [markdown]
# # 🧹 NLP Preprocessing Pipeline — British Airways Reviews
#
# This notebook builds a **production-grade NLP preprocessing pipeline** that
# transforms raw review text into clean, normalized tokens ready for machine
# learning. The pipeline uses both **NLTK** and **spaCy** for complementary
# text processing capabilities.
#
# ### Pipeline Steps
# 1. Lowercase conversion
# 2. URL removal
# 3. Punctuation removal
# 4. Stopword removal (NLTK)
# 5. Tokenization (spaCy)
# 6. Lemmatization (spaCy)
#
# **Output:** A new `cleaned_review` column added to the dataset.

# %% [markdown]
# ---
# ## 📦 1. Install & Import Dependencies

# %%
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')

# ── Install required packages ────────────────────────────────────────────
def install_if_missing(package, import_name=None):
    """Install a package if it's not already available."""
    try:
        __import__(import_name or package)
        print(f"  ✅ {package} already installed")
    except ImportError:
        print(f"  📥 Installing {package}...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', package, '-q'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"  ✅ {package} installed")

print("Checking dependencies...")
install_if_missing('nltk')
install_if_missing('spacy')
install_if_missing('pandas')

# %%
import re
import string
import time

import nltk
import pandas as pd
import spacy

# ── Download NLTK data ───────────────────────────────────────────────────
print("Downloading NLTK resources...")
for resource in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print("✅ NLTK resources ready")

# ── Load spaCy model ────────────────────────────────────────────────────
print("Loading spaCy model...")
try:
    nlp = spacy.load('en_core_web_sm')
    print("✅ spaCy model 'en_core_web_sm' loaded")
except OSError:
    print("📥 Downloading spaCy model 'en_core_web_sm'...")
    subprocess.check_call(
        [sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm', '-q'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    nlp = spacy.load('en_core_web_sm')
    print("✅ spaCy model downloaded and loaded")

# Increase max length for longer reviews
nlp.max_length = 50000

print("\n🚀 All dependencies ready!")

# %% [markdown]
# ---
# ## 📂 2. Load Dataset

# %%
df = pd.read_csv('british_airways_reviews_cleaned.csv')

print(f"📊 Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n📋 Text columns available:")
print(f"   • reviews       — Raw review text")
print(f"   • reviews_clean — Quote-normalized review text")
print(f"   • combined_text — Title + review combined")
print(f"\n📝 Sample raw review (first 300 chars):")
print(f"   '{df['reviews'].iloc[7][:300]}...'")

# %% [markdown]
# ---
# ## ⚙️ 3. Build the NLP Preprocessing Pipeline
#
# We build modular preprocessing functions, then combine them into a single
# pipeline. This approach makes each step testable and easy to modify.

# %% [markdown]
# ### 3a. Step 1 — Lowercase Conversion
#
# Converting all text to lowercase ensures that "Flight", "FLIGHT", and "flight"
# are treated as the same token. This reduces vocabulary size and improves
# model generalization.

# %%
def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return str(text).lower()

# Demo
sample = "British Airways CANCELLED my Flight!"
print(f"Original:  '{sample}'")
print(f"Lowercase: '{to_lowercase(sample)}'")

# %% [markdown]
# ### 3b. Step 2 — URL Removal
#
# Reviews sometimes contain URLs (links to complaint pages, airline websites,
# etc.). These are noise for NLP models — they don't carry semantic meaning
# about satisfaction and can confuse tokenizers.

# %%
def remove_urls(text: str) -> str:
    """Remove URLs (http, https, www) from text."""
    # Match http/https URLs
    text = re.sub(r'https?://\S+', '', text)
    # Match www. URLs without protocol
    text = re.sub(r'www\.\S+', '', text)
    # Match email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    return text.strip()

# Demo
sample = "Check their site at https://www.ba.com or email help@ba.com for info"
print(f"Original:  '{sample}'")
print(f"Cleaned:   '{remove_urls(sample)}'")

# %% [markdown]
# ### 3c. Step 3 — Punctuation Removal
#
# Punctuation marks (commas, periods, exclamation marks, etc.) are removed to
# normalize the text. While punctuation can carry sentiment cues (e.g., "!!!"),
# our VADER sentiment scores already capture this in numeric features, so
# removing punctuation here is safe for the text-based features.

# %%
def remove_punctuation(text: str) -> str:
    """Remove all punctuation and special characters, keeping letters, numbers, and spaces."""
    # Remove special quotes and dashes
    text = re.sub(r'[\u2018\u2019\u201c\u201d\u201e\u2013\u2014]', ' ', text)
    # Remove HTML entities
    text = re.sub(r'&\w+;', ' ', text)
    # Remove all remaining punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Demo
sample = "Terrible! The food was 'disgusting' & the seat — awful. Rating: 1/10."
print(f"Original:  '{sample}'")
print(f"Cleaned:   '{remove_punctuation(sample)}'")

# %% [markdown]
# ### 3d. Step 4 — Stopword Removal (NLTK)
#
# Stopwords are common words like "the", "is", "at", "and" that appear
# frequently but carry little semantic meaning. Removing them reduces noise
# and focuses the model on content-bearing words like "delayed", "terrible",
# "excellent", "comfortable".
#
# We use NLTK's stopword list and add domain-specific stopwords that are
# too generic in the airline review context.

# %%
# ── Build stopword set ───────────────────────────────────────────────────
nltk_stopwords = set(stopwords.words('english'))

# Add domain-specific generic terms that don't help distinguish satisfaction
domain_stopwords = {
    'british', 'airways', 'ba', 'airline', 'airlines',
    'flight', 'flights', 'fly', 'flew', 'flying', 'flown',
    'would', 'could', 'also', 'really', 'quite', 'rather',
    'much', 'even', 'still', 'well', 'get', 'got', 'getting',
    'one', 'two', 'three', 'first', 'us', 'also',
}

all_stopwords = nltk_stopwords | domain_stopwords

print(f"📊 Stopword counts:")
print(f"   • NLTK English stopwords:  {len(nltk_stopwords)}")
print(f"   • Domain-specific:         {len(domain_stopwords)}")
print(f"   • Total combined:          {len(all_stopwords)}")

def remove_stopwords(tokens: list) -> list:
    """Remove stopwords from a list of tokens."""
    return [token for token in tokens if token not in all_stopwords]

# Demo
sample_tokens = ['the', 'cabin', 'crew', 'was', 'really', 'friendly', 'and', 'helpful']
print(f"\nOriginal tokens:  {sample_tokens}")
print(f"After removal:    {remove_stopwords(sample_tokens)}")

# %% [markdown]
# ### 3e. Step 5 — Tokenization (spaCy)
#
# Tokenization splits text into individual words (tokens). We use spaCy's
# tokenizer because it handles edge cases better than simple whitespace
# splitting — it correctly separates contractions ("didn't" → "did", "n't"),
# handles hyphenated words, and preserves meaningful units.

# %%
def tokenize_text(text: str) -> list:
    """Tokenize text using spaCy's tokenizer."""
    doc = nlp.tokenizer(text)
    # Keep only alphabetic tokens with length > 1
    tokens = [token.text for token in doc if token.is_alpha and len(token.text) > 1]
    return tokens

# Demo
sample = "I couldn't check-in online, the BA system wasn't working!"
print(f"Original:     '{sample}'")
print(f"Tokens:       {tokenize_text(to_lowercase(remove_punctuation(sample)))}")

# %% [markdown]
# ### 3f. Step 6 — Lemmatization (spaCy)
#
# Lemmatization reduces words to their base/dictionary form:
# - "flying" → "fly", "flights" → "flight"
# - "cancelled" → "cancel", "delays" → "delay"
# - "better" → "good", "worst" → "bad"
#
# Unlike stemming (which just chops suffixes), lemmatization uses linguistic
# knowledge to produce valid words, making features more interpretable.

# %%
def lemmatize_tokens(tokens: list) -> list:
    """Lemmatize a list of tokens using spaCy."""
    # Join tokens and process through spaCy's full pipeline
    doc = nlp(' '.join(tokens))
    lemmas = [token.lemma_ for token in doc if token.lemma_.isalpha() and len(token.lemma_) > 1]
    return lemmas

# Demo
sample_tokens = ['flights', 'were', 'delayed', 'cancelled', 'passengers', 'waiting', 'terrible']
print(f"Original tokens: {sample_tokens}")
print(f"Lemmatized:      {lemmatize_tokens(sample_tokens)}")

# %% [markdown]
# ---
# ## 🔗 4. Combined Preprocessing Pipeline
#
# Now we combine all individual steps into a single, unified pipeline function.
# The pipeline processes raw text through all 6 steps sequentially and returns
# the final cleaned string.

# %%
def nlp_preprocess(text: str) -> str:
    """
    Complete NLP preprocessing pipeline.
    
    Steps:
        1. Lowercase conversion
        2. URL removal
        3. Punctuation removal
        4. Tokenization (spaCy)
        5. Stopword removal (NLTK)
        6. Lemmatization (spaCy)
    
    Args:
        text: Raw review text string.
    
    Returns:
        Cleaned, lemmatized text string.
    """
    # Step 1: Lowercase
    text = to_lowercase(text)
    
    # Step 2: Remove URLs and emails
    text = remove_urls(text)
    
    # Step 3: Remove punctuation and special characters
    text = remove_punctuation(text)
    
    # Step 4: Tokenize with spaCy
    tokens = tokenize_text(text)
    
    # Step 5: Remove stopwords (NLTK + domain)
    tokens = remove_stopwords(tokens)
    
    # Step 6: Lemmatize with spaCy
    tokens = lemmatize_tokens(tokens)
    
    return ' '.join(tokens)


# ── Test the full pipeline ───────────────────────────────────────────────
test_review = (
    "The WORST experience EVER! British Airways cancelled our flight BA247 "
    "and didn't offer any compensation. We were waiting for 5 hours at "
    "Heathrow airport. Check https://www.ba.com/complaints for details. "
    "The staff weren't helpful at all — absolutely disgusting service!!!"
)

print("=" * 70)
print("🧪 FULL PIPELINE TEST")
print("=" * 70)
print(f"\n📝 Raw input:\n   '{test_review}'\n")

# Show each step
print("Step 1 — Lowercase:")
step1 = to_lowercase(test_review)
print(f"   '{step1}'\n")

print("Step 2 — URLs removed:")
step2 = remove_urls(step1)
print(f"   '{step2}'\n")

print("Step 3 — Punctuation removed:")
step3 = remove_punctuation(step2)
print(f"   '{step3}'\n")

print("Step 4 — Tokenized:")
step4 = tokenize_text(step3)
print(f"   {step4}\n")

print("Step 5 — Stopwords removed:")
step5 = remove_stopwords(step4)
print(f"   {step5}\n")

print("Step 6 — Lemmatized:")
step6 = lemmatize_tokens(step5)
print(f"   {step6}\n")

result = nlp_preprocess(test_review)
print(f"✅ Final output:\n   '{result}'")

# %% [markdown]
# ---
# ## 🚀 5. Apply Pipeline to Full Dataset
#
# We apply the preprocessing pipeline to all 1,300 reviews. Processing uses
# spaCy's `nlp.pipe()` for batch efficiency where possible, but falls back
# to row-by-row processing for the combined pipeline since each step is
# modular.

# %%
print("⏳ Applying NLP preprocessing pipeline to all reviews...")
print(f"   Processing {len(df):,} reviews...\n")

start_time = time.time()

# Apply the pipeline with progress tracking
cleaned_reviews = []
total = len(df)
milestone = max(1, total // 10)  # Report every 10%

for i, review in enumerate(df['reviews_clean']):
    cleaned = nlp_preprocess(review)
    cleaned_reviews.append(cleaned)
    
    if (i + 1) % milestone == 0 or (i + 1) == total:
        elapsed = time.time() - start_time
        pct = (i + 1) / total * 100
        rate = (i + 1) / elapsed
        eta = (total - i - 1) / rate if rate > 0 else 0
        print(f"   [{pct:5.1f}%] {i+1:,}/{total:,} reviews processed "
              f"({rate:.1f} reviews/sec, ETA: {eta:.0f}s)")

df['cleaned_review'] = cleaned_reviews

elapsed_total = time.time() - start_time
print(f"\n✅ Pipeline complete! Processed {total:,} reviews in {elapsed_total:.1f} seconds "
      f"({total/elapsed_total:.1f} reviews/sec)")

# %% [markdown]
# ---
# ## 🔍 6. Inspect Results

# %%
# ── Compare raw vs cleaned reviews ──────────────────────────────────────
print("=" * 80)
print("📊 BEFORE vs AFTER COMPARISON")
print("=" * 80)

# Show 5 diverse examples (mix of satisfied and not satisfied)
sample_indices = [0, 7, 15, 28, 42]

for idx in sample_indices:
    row = df.iloc[idx]
    satisfaction = "😊 Satisfied" if row['satisfied'] == 1 else "😡 Not Satisfied"
    print(f"\n{'─' * 80}")
    print(f"Review #{idx} | {satisfaction}")
    print(f"{'─' * 80}")
    print(f"📝 Original ({len(row['reviews_clean'])} chars):")
    print(f"   {row['reviews_clean'][:200]}{'...' if len(row['reviews_clean']) > 200 else ''}")
    print(f"\n🧹 Cleaned ({len(row['cleaned_review'])} chars):")
    print(f"   {row['cleaned_review'][:200]}{'...' if len(row['cleaned_review']) > 200 else ''}")

# %%
# ── Statistical comparison ───────────────────────────────────────────────
print("\n" + "=" * 80)
print("📊 TEXT LENGTH STATISTICS")
print("=" * 80)

df['cleaned_char_count'] = df['cleaned_review'].str.len()
df['cleaned_word_count'] = df['cleaned_review'].str.split().str.len()

stats = pd.DataFrame({
    'Metric': ['Avg Characters', 'Avg Words', 'Avg Word Length',
               'Min Words', 'Max Words', 'Vocabulary Size'],
    'Original (reviews_clean)': [
        f"{df['reviews_clean'].str.len().mean():.0f}",
        f"{df['reviews_clean'].str.split().str.len().mean():.0f}",
        f"{(df['reviews_clean'].str.len() / df['reviews_clean'].str.split().str.len()).mean():.1f}",
        f"{df['reviews_clean'].str.split().str.len().min()}",
        f"{df['reviews_clean'].str.split().str.len().max()}",
        f"{len(set(' '.join(df['reviews_clean'].values).lower().split())):,}",
    ],
    'Cleaned (cleaned_review)': [
        f"{df['cleaned_char_count'].mean():.0f}",
        f"{df['cleaned_word_count'].mean():.0f}",
        f"{(df['cleaned_char_count'] / df['cleaned_word_count'].clip(lower=1)).mean():.1f}",
        f"{df['cleaned_word_count'].min()}",
        f"{df['cleaned_word_count'].max()}",
        f"{len(set(' '.join(df['cleaned_review'].values).split())):,}",
    ],
})

print(f"\n{stats.to_string(index=False)}")

original_vocab = len(set(' '.join(df['reviews_clean'].values).lower().split()))
cleaned_vocab = len(set(' '.join(df['cleaned_review'].values).split()))
reduction = (1 - cleaned_vocab / original_vocab) * 100

print(f"\n📉 Vocabulary reduced by {reduction:.1f}% ({original_vocab:,} → {cleaned_vocab:,} unique words)")
print(f"📉 Avg word count reduced by "
      f"{(1 - df['cleaned_word_count'].mean() / df['reviews_clean'].str.split().str.len().mean()) * 100:.1f}%")

# %% [markdown]
# ---
# ## 📊 7. Visualize Preprocessing Impact

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
COLORS = {'negative': '#E74C3C', 'positive': '#2ECC71', 'primary': '#3498DB',
          'secondary': '#9B59B6', 'dark': '#2C3E50'}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ── Chart 1: Word count distribution before vs after ─────────────────────
ax = axes[0]
original_wc = df['reviews_clean'].str.split().str.len()
cleaned_wc = df['cleaned_word_count']
ax.hist(original_wc, bins=50, alpha=0.6, color=COLORS['primary'], label='Original', edgecolor='white')
ax.hist(cleaned_wc, bins=50, alpha=0.6, color=COLORS['positive'], label='Cleaned', edgecolor='white')
ax.set_title('Word Count Distribution\n(Before vs After)', fontweight='bold', fontsize=13)
ax.set_xlabel('Word Count')
ax.set_ylabel('Frequency')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# ── Chart 2: Word count reduction by class ───────────────────────────────
ax = axes[1]
for label, color, name in [(0, COLORS['negative'], 'Not Satisfied'),
                            (1, COLORS['positive'], 'Satisfied')]:
    subset = df[df['satisfied'] == label]
    orig = subset['reviews_clean'].str.split().str.len()
    clean = subset['cleaned_word_count']
    reduction_pct = ((orig - clean) / orig * 100)
    reduction_pct.plot.kde(ax=ax, color=color, linewidth=2.5, label=name)

ax.set_title('Word Reduction % by Class', fontweight='bold', fontsize=13)
ax.set_xlabel('Reduction (%)')
ax.set_ylabel('Density')
ax.legend(fontsize=11)

# ── Chart 3: Top 20 words after cleaning ─────────────────────────────────
ax = axes[2]
from collections import Counter
all_words = ' '.join(df['cleaned_review'].values).split()
word_counts = Counter(all_words).most_common(20)
words = [w[0] for w in word_counts]
counts = [w[1] for w in word_counts]
ax.barh(range(len(words)), counts, color=COLORS['secondary'], edgecolor='white', height=0.7)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Frequency')
ax.set_title('Top 20 Words After Cleaning', fontweight='bold', fontsize=13)
ax.grid(axis='x', alpha=0.3)

plt.suptitle('NLP Preprocessing Pipeline Impact', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('eda_12_preprocessing_impact.png', bbox_inches='tight', facecolor='white', dpi=120)
plt.show()
print("📊 Chart saved: eda_12_preprocessing_impact.png")

# %% [markdown]
# ---
# ## 💾 8. Save Updated Dataset

# %%
# Keep essential columns for modeling
output_columns = [
    'title', 'reviews', 'reviews_clean', 'cleaned_review', 'combined_text',
    'neg', 'neu', 'pos', 'compound', 'satisfied',
    'review_length', 'word_count', 'char_count',
    'avg_word_length', 'exclamation_count', 'question_count', 'uppercase_ratio',
    'cleaned_word_count', 'cleaned_char_count',
]

# Only include columns that exist
output_columns = [c for c in output_columns if c in df.columns]

df_out = df[output_columns]
df_out.to_csv('british_airways_nlp_cleaned.csv', index=False)

print(f"✅ Saved: 'british_airways_nlp_cleaned.csv'")
print(f"   Shape: {df_out.shape[0]:,} rows × {df_out.shape[1]} columns")
print(f"   Columns: {list(df_out.columns)}")

# %% [markdown]
# ---
# ## 📋 9. Pipeline Summary
#
# | Step | Tool | What It Does | Example |
# |------|------|-------------|---------|
# | 1. Lowercase | Python `.lower()` | Normalizes case | "DELAYED" → "delayed" |
# | 2. URL Removal | Regex | Strips URLs & emails | "visit https://ba.com" → "visit" |
# | 3. Punctuation | `str.translate()` | Removes special chars | "terrible!!!" → "terrible" |
# | 4. Tokenization | spaCy | Splits into tokens | "didn't work" → ["did", "not", "work"] |
# | 5. Stopwords | NLTK + custom | Removes noise words | ["the", "crew", "was"] → ["crew"] |
# | 6. Lemmatization | spaCy | Base word forms | "cancelled" → "cancel" |
#
# ### Key Results
# - **Vocabulary reduction:** Fewer unique words = more efficient model training
# - **Noise removal:** URLs, punctuation, stopwords stripped
# - **Normalization:** Lemmatized forms reduce word variants
# - **New column:** `cleaned_review` added to dataset

# %%
print("\n🎉 NLP Preprocessing Pipeline complete!")
print("   📂 Output file: british_airways_nlp_cleaned.csv")
print("   📊 Visualization: eda_12_preprocessing_impact.png")
print("   🔑 New column: 'cleaned_review'")
