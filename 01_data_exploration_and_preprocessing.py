# %% [markdown]
# # 🛫 British Airways Reviews — Data Exploration & Preprocessing
# 
# **Goal:** Explore the British Airways customer review dataset, clean the text data,
# engineer a satisfaction target variable using sentiment analysis, and prepare
# features for a customer satisfaction prediction model.

# %% [markdown]
# ## 📦 1. Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("✅ Libraries imported successfully!")

# %% [markdown]
# ## 📂 2. Load the Dataset

# %%
df = pd.read_csv('british_airways_reviews.csv')

print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
df.head()

# %% [markdown]
# ## 📊 3. Column Descriptions
# 
# | Column | Type | Description |
# |--------|------|-------------|
# | `Unnamed: 0` | int64 | Auto-generated row index (not useful for modeling) |
# | `title` | object | Short headline/summary of the review |
# | `reviews` | object | Full-text review body describing the flight experience |

# %% [markdown]
# ## 🔍 4. Missing Value Analysis

# %%
missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
})
print(missing_df.to_string(index=False))

print(f"\n{'✅ No missing values found!' if df.isnull().sum().sum() == 0 else '⚠️ Missing values detected!'}")

# %% [markdown]
# ## 📈 5. Exploratory Data Analysis

# %%
# --- Review length distribution ---
df['review_length'] = df['reviews'].astype(str).str.len()
df['review_word_count'] = df['reviews'].astype(str).str.split().str.len()
df['title_length'] = df['title'].astype(str).str.len()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['review_length'], bins=50, color='#4C72B0', edgecolor='white', alpha=0.85)
axes[0].set_title('Review Length (Characters)', fontweight='bold')
axes[0].set_xlabel('Characters')
axes[0].set_ylabel('Count')

axes[1].hist(df['review_word_count'], bins=50, color='#55A868', edgecolor='white', alpha=0.85)
axes[1].set_title('Review Word Count', fontweight='bold')
axes[1].set_xlabel('Words')
axes[1].set_ylabel('Count')

axes[2].hist(df['title_length'], bins=40, color='#C44E52', edgecolor='white', alpha=0.85)
axes[2].set_title('Title Length (Characters)', fontweight='bold')
axes[2].set_xlabel('Characters')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.suptitle('Text Length Distributions', fontweight='bold', fontsize=16, y=1.02)
plt.show()

print(f"Review length  — Mean: {df['review_length'].mean():.0f} | Median: {df['review_length'].median():.0f} | Min: {df['review_length'].min()} | Max: {df['review_length'].max()}")
print(f"Word count     — Mean: {df['review_word_count'].mean():.0f} | Median: {df['review_word_count'].median():.0f} | Min: {df['review_word_count'].min()} | Max: {df['review_word_count'].max()}")

# %% [markdown]
# ## 🧹 6. Data Cleaning

# %%
# --- Step 1: Drop unnecessary index column ---
df.drop(columns=['Unnamed: 0'], inplace=True)
print("✅ Dropped 'Unnamed: 0' column")

# --- Step 2: Strip whitespace ---
df['title'] = df['title'].astype(str).str.strip()
df['reviews'] = df['reviews'].astype(str).str.strip()
print("✅ Stripped leading/trailing whitespace")

# --- Step 3: Normalize quotes and special characters ---
def normalize_text(text):
    """Clean and normalize text for NLP processing."""
    # Replace smart quotes with standard quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201e', '"')
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing quotes
    text = text.strip('"').strip("'").strip()
    return text

df['title_clean'] = df['title'].apply(normalize_text)
df['reviews_clean'] = df['reviews'].apply(normalize_text)
print("✅ Normalized quotes and whitespace")

# --- Step 4: Create combined text feature ---
df['combined_text'] = df['title_clean'] + '. ' + df['reviews_clean']
print("✅ Created combined text feature (title + review)")

print(f"\n📋 Cleaned columns: {list(df.columns)}")
df[['title_clean', 'reviews_clean']].head()

# %% [markdown]
# ## 🏷️ 7. Target Variable Creation — Sentiment-Based Labeling
# 
# Since the dataset has **no explicit satisfaction label**, we use **VADER** 
# (Valence Aware Dictionary and sEntiment Reasoner) to derive sentiment scores
# and create a binary satisfaction target.

# %%
# Install VADER if not already available
import subprocess
import sys

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("✅ VADER already installed")
except ImportError:
    print("📥 Installing VADER...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'vaderSentiment', '-q'])
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("✅ VADER installed successfully")

# Initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

# %%
# --- Compute sentiment scores ---
def get_sentiment_scores(text):
    """Return VADER sentiment scores for a given text."""
    scores = analyzer.polarity_scores(str(text))
    return pd.Series({
        'neg': scores['neg'],
        'neu': scores['neu'],
        'pos': scores['pos'],
        'compound': scores['compound']
    })

print("⏳ Computing sentiment scores (this may take a moment)...")
sentiment_scores = df['combined_text'].apply(get_sentiment_scores)
df = pd.concat([df, sentiment_scores], axis=1)
print("✅ Sentiment scores computed!")

# --- Create binary target ---
# compound >= 0.05 → Satisfied (1)
# compound <  0.05 → Not Satisfied (0)
df['satisfied'] = (df['compound'] >= 0.05).astype(int)

print(f"\n🎯 Target Variable Distribution:")
print(df['satisfied'].value_counts().rename({1: 'Satisfied (1)', 0: 'Not Satisfied (0)'}))
print(f"\nSatisfied ratio: {df['satisfied'].mean():.1%}")

# %%
# --- Visualize target distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of target distribution
colors = ['#E74C3C', '#2ECC71']
counts = df['satisfied'].value_counts().sort_index()
axes[0].bar(['Not Satisfied (0)', 'Satisfied (1)'], counts.values, color=colors, edgecolor='white', width=0.5)
axes[0].set_title('Target Variable Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=13)

# Histogram of compound sentiment scores
axes[1].hist(df['compound'], bins=50, color='#3498DB', edgecolor='white', alpha=0.85)
axes[1].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Threshold (0.05)')
axes[1].set_title('Compound Sentiment Score Distribution', fontweight='bold')
axes[1].set_xlabel('Compound Score')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## ⚙️ 8. Advanced Text Preprocessing

# %%
import string

# Optional: download NLTK resources (uncomment if needed)
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):
    """Full text preprocessing pipeline for NLP modeling."""
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df['text_processed'] = df['combined_text'].apply(preprocess_text)

print("✅ Text preprocessing complete!")
print(f"\n--- Sample ---")
print(f"Original:  {df['combined_text'].iloc[7][:150]}...")
print(f"Processed: {df['text_processed'].iloc[7][:150]}...")

# %% [markdown]
# ## 🧪 9. Feature Engineering

# %%
# --- Numeric features ---
df['char_count'] = df['text_processed'].str.len()
df['word_count'] = df['text_processed'].str.split().str.len()
df['avg_word_length'] = df['char_count'] / df['word_count']
df['exclamation_count'] = df['combined_text'].str.count('!')
df['question_count'] = df['combined_text'].str.count(r'\?')
df['uppercase_ratio'] = df['combined_text'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
)

print("✅ Engineered numeric features:")
feature_cols = ['char_count', 'word_count', 'avg_word_length', 
                'exclamation_count', 'question_count', 'uppercase_ratio',
                'neg', 'neu', 'pos', 'compound']
print(df[feature_cols].describe().round(3).to_string())

# %% [markdown]
# ## 🔀 10. TF-IDF Vectorization & Train/Test Split

# %%
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# --- Define features and target ---
X_text = df['text_processed']
X_numeric = df[feature_cols].values
y = df['satisfied'].values

# --- Stratified train/test split ---
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text, X_numeric, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {len(y_train)} samples (Satisfied: {y_train.sum()}, Not: {len(y_train) - y_train.sum()})")
print(f"Test set:     {len(y_test)} samples (Satisfied: {y_test.sum()}, Not: {len(y_test) - y_test.sum()})")

# %%
# --- TF-IDF Vectorization ---
tfidf = TfidfVectorizer(
    max_features=5000,      # Limit vocabulary size
    ngram_range=(1, 2),     # Unigrams and bigrams
    min_df=3,               # Ignore very rare terms
    max_df=0.95,            # Ignore terms in >95% of docs
    sublinear_tf=True       # Apply sublinear TF scaling
)

X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test = tfidf.transform(X_text_test)

print(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
print(f"TF-IDF matrix shape (train): {X_tfidf_train.shape}")

# --- Combine TF-IDF with numeric features ---
from scipy.sparse import csr_matrix

X_train_combined = hstack([X_tfidf_train, csr_matrix(X_num_train)])
X_test_combined = hstack([X_tfidf_test, csr_matrix(X_num_test)])

print(f"\n✅ Final combined feature matrix shape (train): {X_train_combined.shape}")
print(f"✅ Final combined feature matrix shape (test):  {X_test_combined.shape}")

# %% [markdown]
# ## 💾 11. Save Preprocessed Data

# %%
# Save the cleaned dataframe for future use
df.to_csv('british_airways_reviews_cleaned.csv', index=False)
print("✅ Saved cleaned dataset to 'british_airways_reviews_cleaned.csv'")

# Save train/test splits
import joblib

joblib.dump({
    'X_train': X_train_combined,
    'X_test': X_test_combined,
    'y_train': y_train,
    'y_test': y_test,
    'tfidf_vectorizer': tfidf,
    'feature_names_numeric': feature_cols
}, 'preprocessed_data.pkl')

print("✅ Saved preprocessed train/test data to 'preprocessed_data.pkl'")

# %% [markdown]
# ## 📋 12. Summary
# 
# | Step | Status |
# |------|--------|
# | Dataset loaded (1,300 reviews) | ✅ |
# | No missing values | ✅ |
# | Text cleaned & normalized | ✅ |
# | Sentiment scores computed (VADER) | ✅ |
# | Binary target created (`satisfied`: 0/1) | ✅ |
# | Numeric features engineered | ✅ |
# | TF-IDF vectorization (5,000 features, bigrams) | ✅ |
# | Stratified 80/20 train/test split | ✅ |
# | Preprocessed data saved | ✅ |
# 
# **Next Steps:** Use `preprocessed_data.pkl` to train classification models
# (Logistic Regression, Random Forest, XGBoost, etc.) in the next notebook.

# %%
print("\n🎉 Preprocessing pipeline complete! Ready for model training.")
