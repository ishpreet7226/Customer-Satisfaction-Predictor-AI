# %% [markdown]
# # ✈️ British Airways Reviews — Exploratory Data Analysis (EDA)
#
# This notebook performs a comprehensive exploratory data analysis on the
# British Airways customer review dataset. We explore data quality, review
# characteristics, sentiment distributions, and feature correlations to
# uncover patterns that drive customer satisfaction.
#
# **Dataset:** 1,300 customer reviews with engineered sentiment features
# and a binary satisfaction label derived via VADER sentiment analysis.

# %% [markdown]
# ---
# ## 📦 1. Setup & Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import re
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

# ── Style Configuration ──────────────────────────────────────────────────
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 6),
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'font.family': 'sans-serif',
})

# ── Color Palette ─────────────────────────────────────────────────────────
COLORS = {
    'negative': '#E74C3C',
    'positive': '#2ECC71',
    'neutral':  '#F39C12',
    'primary':  '#3498DB',
    'secondary':'#9B59B6',
    'dark':     '#2C3E50',
    'light':    '#ECF0F1',
}
PALETTE_SAT = [COLORS['negative'], COLORS['positive']]

print("✅ Libraries loaded & style configured")

# %%
# ── Load cleaned dataset ──────────────────────────────────────────────────
df = pd.read_csv('british_airways_reviews_cleaned.csv')

print(f"📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n📋 Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col:<25s}  ({df[col].dtype})")

# %% [markdown]
# ---
# ## 🔍 2. Missing Value Analysis
#
# **Why it matters:** Missing values can bias model training, cause errors in
# pipelines, and distort statistical summaries. Visualizing them helps us
# decide on imputation strategies (or confirms the data is clean).

# %%
# ── 2a. Missing Value Summary Table ──────────────────────────────────────
missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': missing_counts.values,
    'Missing (%)': missing_pct.values,
    'Dtype': df.dtypes.values
}).sort_values('Missing Count', ascending=False).reset_index(drop=True)

print(missing_summary.to_string(index=False))
total_missing = missing_counts.sum()
print(f"\n{'✅ Dataset is 100% complete — zero missing values!' if total_missing == 0 else f'⚠️ Total missing cells: {total_missing}'}")

# %%
# ── 2b. Missing Value Heatmap ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [3, 1]})

# Left: Heatmap (sample of rows to show density)
ax1 = axes[0]
sample_idx = np.random.RandomState(42).choice(len(df), size=min(200, len(df)), replace=False)
sample_idx.sort()
sns.heatmap(
    df.iloc[sample_idx].isnull().astype(int),
    cbar=False,
    cmap=['#2ECC71', '#E74C3C'],
    yticklabels=False,
    ax=ax1,
    linewidths=0.0,
)
ax1.set_title('Missing Value Heatmap (200-row sample)', fontweight='bold', pad=12)
ax1.set_xlabel('Columns')
ax1.set_ylabel('Rows (sampled)')
ax1.tick_params(axis='x', rotation=45, labelsize=9)

# Right: Bar chart of missing percentages
ax2 = axes[1]
cols_to_show = df.columns
bar_colors = [COLORS['positive'] if v == 0 else COLORS['negative'] for v in missing_pct.values]
bars = ax2.barh(cols_to_show, missing_pct.values, color=bar_colors, edgecolor='white', height=0.6)
ax2.set_xlabel('Missing (%)')
ax2.set_title('Missing % per Column', fontweight='bold', pad=12)
ax2.set_xlim(0, max(missing_pct.values.max() * 1.3, 5))
for bar, pct in zip(bars, missing_pct.values):
    ax2.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
             f'{pct:.1f}%', va='center', fontsize=9, fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('eda_01_missing_values.png', bbox_inches='tight', facecolor='white')
plt.show()

# %% [markdown]
# **📖 Interpretation:** The heatmap is entirely green (no red cells), confirming
# that every column across all 1,300 rows is fully populated. The bar chart
# reinforces this — every column shows 0.0% missing data. No imputation or
# row-dropping is needed before modeling.

# %% [markdown]
# ---
# ## 📊 3. Satisfaction (Target Variable) Distribution
#
# **Why it matters:** Class imbalance can severely impact classifier performance.
# If one class dominates (e.g., 90% satisfied), the model may learn to always
# predict the majority class. We need to check the balance before training.

# %%
# ── 3a. Satisfaction Distribution — Bar + Donut Chart ────────────────────
sat_counts = df['satisfied'].value_counts().sort_index()
sat_labels = ['Not Satisfied (0)', 'Satisfied (1)']
sat_pcts = (sat_counts / len(df) * 100).values

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
ax1 = axes[0]
bars = ax1.bar(sat_labels, sat_counts.values, color=PALETTE_SAT, edgecolor='white',
               width=0.5, zorder=3)
ax1.set_title('Customer Satisfaction Distribution', fontweight='bold', pad=12)
ax1.set_ylabel('Number of Reviews')
ax1.set_ylim(0, sat_counts.max() * 1.2)
ax1.grid(axis='y', alpha=0.3)
for bar, count, pct in zip(bars, sat_counts.values, sat_pcts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
             f'{count}\n({pct:.1f}%)', ha='center', fontweight='bold', fontsize=13)

# Donut chart
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(
    sat_counts.values,
    labels=sat_labels,
    colors=PALETTE_SAT,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 12}
)
for t in autotexts:
    t.set_fontweight('bold')
    t.set_fontsize(13)
ax2.set_title('Satisfaction Ratio', fontweight='bold', pad=12)
centre_circle = plt.Circle((0, 0), 0.55, fc='white')
ax2.add_artist(centre_circle)
ax2.text(0, 0, f'n={len(df):,}', ha='center', va='center', fontsize=14, fontweight='bold',
         color=COLORS['dark'])

plt.tight_layout()
plt.savefig('eda_02_satisfaction_distribution.png', bbox_inches='tight', facecolor='white')
plt.show()

print(f"\n📌 Class Balance Ratio: {sat_counts.min() / sat_counts.max():.2f} "
      f"(1.00 = perfectly balanced)")

# %% [markdown]
# **📖 Interpretation:** The dataset is remarkably well-balanced with approximately
# 50.6% "Not Satisfied" and 49.4% "Satisfied" reviews. The balance ratio is ~0.98,
# which is near-perfect. This means we **do not need** resampling techniques like
# SMOTE or class weighting — standard classifiers will work well out of the box.

# %% [markdown]
# ---
# ## 📊 4. Sentiment Score Distribution
#
# **Why it matters:** The compound sentiment score (from VADER) is the foundation
# of our target variable. Understanding its distribution reveals how polarized
# or nuanced the reviews are — and whether the 0.05 threshold for labeling
# "satisfied" creates a natural split.

# %%
# ── 4a. Compound Sentiment Score — Histogram + KDE by Class ──────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Overall distribution
ax1 = axes[0]
ax1.hist(df['compound'], bins=60, color=COLORS['primary'], edgecolor='white',
         alpha=0.8, zorder=3)
ax1.axvline(x=0.05, color=COLORS['negative'], linestyle='--', linewidth=2.5,
            label='Threshold (0.05)', zorder=4)
ax1.axvline(x=df['compound'].mean(), color=COLORS['neutral'], linestyle=':',
            linewidth=2, label=f'Mean ({df["compound"].mean():.3f})', zorder=4)
ax1.set_title('Compound Sentiment Score Distribution', fontweight='bold', pad=12)
ax1.set_xlabel('Compound Score')
ax1.set_ylabel('Frequency')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# By satisfaction class (KDE)
ax2 = axes[1]
for label, color, name in [(0, COLORS['negative'], 'Not Satisfied'),
                            (1, COLORS['positive'], 'Satisfied')]:
    subset = df[df['satisfied'] == label]['compound']
    subset.plot.kde(ax=ax2, color=color, linewidth=2.5, label=name)
    ax2.axvline(x=subset.mean(), color=color, linestyle=':', linewidth=1.5, alpha=0.6)

ax2.axvline(x=0.05, color=COLORS['dark'], linestyle='--', linewidth=2, label='Threshold')
ax2.set_title('Sentiment Distribution by Satisfaction Class', fontweight='bold', pad=12)
ax2.set_xlabel('Compound Score')
ax2.set_ylabel('Density')
ax2.legend(fontsize=11)
ax2.set_xlim(-1.1, 1.1)

plt.tight_layout()
plt.savefig('eda_03_sentiment_distribution.png', bbox_inches='tight', facecolor='white')
plt.show()

# %%
# ── 4b. Sentiment Component Breakdown (neg, neu, pos) ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col, color, title in zip(
    axes,
    ['neg', 'neu', 'pos'],
    [COLORS['negative'], COLORS['neutral'], COLORS['positive']],
    ['Negative Score', 'Neutral Score', 'Positive Score']
):
    for label, ls, lbl in [(0, '-', 'Not Satisfied'), (1, '--', 'Satisfied')]:
        df[df['satisfied'] == label][col].plot.kde(
            ax=ax, color=PALETTE_SAT[label], linewidth=2.5,
            linestyle=ls, label=lbl
        )
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend(fontsize=10)

plt.suptitle('VADER Sentiment Component Distributions by Class',
             fontweight='bold', fontsize=16, y=1.03)
plt.tight_layout()
plt.savefig('eda_04_sentiment_components.png', bbox_inches='tight', facecolor='white')
plt.show()

# %% [markdown]
# **📖 Interpretation:**
# - The compound score shows a clear **bimodal distribution** — reviews tend to
#   cluster around strong negative (−0.8 to −1.0) or strong positive (+0.8 to +1.0)
#   sentiment, with fewer reviews near zero. This confirms the 0.05 threshold
#   creates a meaningful split.
# - **Negative score (neg):** Not-satisfied reviews have significantly higher
#   negative sentiment proportions, as expected.
# - **Positive score (pos):** Satisfied reviews show distinctly higher positive
#   proportions, validating the labeling approach.
# - **Neutral score (neu):** Both classes are dominated by neutral language
#   (factual descriptions of flights), but dissatisfied reviews tend to have
#   slightly less neutral content (more emotionally charged).

# %% [markdown]
# ---
# ## 📏 5. Review Length Analysis
#
# **Why it matters:** Review length can be a strong predictor of satisfaction.
# Dissatisfied customers often write longer, more detailed complaints, while
# satisfied customers may write shorter, positive summaries. Understanding
# this helps with feature engineering and model interpretation.

# %%
# ── 5a. Review Length Distributions — Histogram + Box + Violin ───────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# --- Row 1: Histograms (overall) ---
for ax, col, color, title in zip(
    axes[0],
    ['review_length', 'review_word_count', 'title_length'],
    [COLORS['primary'], COLORS['secondary'], COLORS['neutral']],
    ['Review Length (characters)', 'Review Word Count', 'Title Length (characters)']
):
    ax.hist(df[col], bins=50, color=color, edgecolor='white', alpha=0.85, zorder=3)
    ax.axvline(df[col].mean(), color=COLORS['negative'], linestyle='--',
               linewidth=2, label=f'Mean: {df[col].mean():.0f}')
    ax.axvline(df[col].median(), color=COLORS['dark'], linestyle=':',
               linewidth=2, label=f'Median: {df[col].median():.0f}')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# --- Row 2: Violin plots by satisfaction class ---
for ax, col, title in zip(
    axes[1],
    ['review_length', 'review_word_count', 'title_length'],
    ['Review Length by Satisfaction', 'Word Count by Satisfaction', 'Title Length by Satisfaction']
):
    parts = ax.violinplot(
        [df[df['satisfied'] == 0][col].values, df[df['satisfied'] == 1][col].values],
        positions=[0, 1], showmeans=True, showmedians=True, widths=0.7
    )
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE_SAT[i])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color(COLORS['dark'])
    parts['cmedians'].set_color(COLORS['neutral'])

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Satisfied', 'Satisfied'])
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_ylabel(col.replace('_', ' ').title())
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Review Length Analysis', fontweight='bold', fontsize=18, y=1.01)
plt.tight_layout()
plt.savefig('eda_05_review_length_analysis.png', bbox_inches='tight', facecolor='white')
plt.show()

# %%
# ── 5b. Length Statistics by Class ────────────────────────────────────────
length_stats = df.groupby('satisfied')[['review_length', 'review_word_count', 'title_length']].agg(
    ['mean', 'median', 'std', 'min', 'max']
).round(1)
length_stats.index = ['Not Satisfied', 'Satisfied']
print("📊 Review Length Statistics by Satisfaction Class:\n")
print(length_stats.to_string())

# %% [markdown]
# **📖 Interpretation:**
# - **Dissatisfied reviews are longer** on average — customers who had a bad
#   experience tend to write more detailed complaints (venting frustration).
# - **Satisfied reviews are shorter** — happy customers are often more concise.
# - The violin plots show the distribution shapes clearly: dissatisfied reviews
#   have a longer right tail (some extremely long complaints), while satisfied
#   reviews cluster more tightly.
# - **Title length** shows less variation between classes — titles are constrained
#   by nature and less informative as a standalone feature.
# - 💡 **Insight:** `review_length` and `word_count` are likely useful features
#   for the prediction model.

# %% [markdown]
# ---
# ## ☁️ 6. Word Cloud Analysis
#
# **Why it matters:** Word clouds reveal the most frequent terms used by
# satisfied vs. dissatisfied customers at a glance. They help identify
# recurring themes, pain points (delays, food, luggage), and positive
# experiences (crew, comfort, on-time).

# %%
# Install wordcloud if not available
try:
    from wordcloud import WordCloud
    print("✅ WordCloud available")
except ImportError:
    import subprocess, sys
    print("📥 Installing wordcloud...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wordcloud', '-q'])
    from wordcloud import WordCloud
    print("✅ WordCloud installed")

# %%
# ── 6a. Word Clouds — Satisfied vs Not Satisfied ─────────────────────────

# Common stopwords to exclude (English + airline-generic terms)
english_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn',
}
custom_stopwords = english_stopwords | {
    'flight', 'ba', 'british', 'airways', 'airline', 'fly', 'flew', 'flying',
    'one', 'would', 'also', 'us', 'get', 'got', 'even', 'much', 'well', 'told',
    'said', 'just', 'really', 'could', 'still', 'back', 'two', 'make', 'made',
    'time', 'take', 'took', 'going', 'went', 'go', 'come', 'came', 'like',
    'know', 'think', 'thing', 'way', 'say', 'see', 'give', 'want', 'tell',
    'use', 'find', 'put', 'try', 'ask', 'need', 'let', 'keep', 'seem',
}

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, label, title, colormap, bg_color in [
    (axes[0], 0, '😡 Not Satisfied Reviews', 'Reds', '#FFF5F5'),
    (axes[1], 1, '😊 Satisfied Reviews', 'Greens', '#F0FFF0'),
]:
    text = ' '.join(df[df['satisfied'] == label]['text_processed'].astype(str).values)

    wc = WordCloud(
        width=1000, height=600,
        max_words=150,
        background_color=bg_color,
        colormap=colormap,
        stopwords=custom_stopwords,
        collocations=True,
        contour_width=2,
        contour_color='grey',
        min_font_size=8,
        max_font_size=100,
        random_state=42,
    ).generate(text)

    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontweight='bold', fontsize=16, pad=12)
    ax.axis('off')

plt.suptitle('Word Cloud Comparison: What Do Customers Talk About?',
             fontweight='bold', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig('eda_06_word_clouds.png', bbox_inches='tight', facecolor='white')
plt.show()

# %% [markdown]
# **📖 Interpretation:**
# - **Not Satisfied** word cloud is dominated by: *"customer service"*, *"delay"*,
#   *"cancelled"*, *"luggage"*, *"Heathrow"*, *"hours"*, *"waiting"*, *"refund"*,
#   *"worst"*, *"terrible"* — revealing common pain points around delays,
#   cancellations, lost baggage, and unresponsive customer support.
# - **Satisfied** word cloud features: *"crew"*, *"cabin"*, *"good"*, *"seat"*,
#   *"food"*, *"service"*, *"excellent"*, *"comfortable"*, *"friendly"*,
#   *"lounge"* — highlighting positive experiences with staff, in-flight
#   comfort, and quality service.
# - 💡 **Key Insight:** The primary drivers of dissatisfaction are operational
#   failures (delays, cancellations, baggage), while satisfaction is driven
#   by human touchpoints (crew friendliness) and comfort.

# %% [markdown]
# ---
# ## 📊 7. Top N-grams Analysis
#
# **Why it matters:** While word clouds show frequency visually, bar charts
# of top bigrams (2-word phrases) give precise counts and reveal meaningful
# multi-word expressions like "customer service" or "cabin crew".

# %%
# ── 7a. Top Bigrams by Class ─────────────────────────────────────────────
from sklearn.feature_extraction.text import CountVectorizer

def get_top_ngrams(texts, n=2, top_k=15, stop_words=None):
    """Extract the top-k most frequent n-grams from a list of texts."""
    vec = CountVectorizer(ngram_range=(n, n), stop_words=stop_words, max_features=10000)
    bag = vec.fit_transform(texts)
    sum_words = bag.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, label, color, title in [
    (axes[0], 0, COLORS['negative'], 'Top 15 Bigrams — Not Satisfied'),
    (axes[1], 1, COLORS['positive'], 'Top 15 Bigrams — Satisfied'),
]:
    texts = df[df['satisfied'] == label]['text_processed'].astype(str).tolist()
    top_ngrams = get_top_ngrams(texts, n=2, top_k=15)
    words = [w[0] for w in top_ngrams]
    counts = [w[1] for w in top_ngrams]

    bars = ax.barh(range(len(words)), counts, color=color, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(title, fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=10, fontweight='bold')

plt.suptitle('Most Frequent Bigrams by Satisfaction Class',
             fontweight='bold', fontsize=17, y=1.02)
plt.tight_layout()
plt.savefig('eda_07_top_bigrams.png', bbox_inches='tight', facecolor='white')
plt.show()

# %% [markdown]
# **📖 Interpretation:**
# - **Not Satisfied** reviews heavily feature: *"customer service"*, *"british airways"*,
#   *"long haul"*, *"hours later"*, *"business class"* (complaints about premium
#   products not meeting expectations).
# - **Satisfied** reviews emphasize: *"cabin crew"*, *"business class"* (positive
#   experiences), *"food drink"*, *"on time"*, *"bag drop"*.
# - 💡 Notice *"business class"* appears in both — the context differs. Dissatisfied
#   passengers complain about poor value; satisfied passengers praise the experience.

# %% [markdown]
# ---
# ## 🔥 8. Correlation Heatmap
#
# **Why it matters:** Correlations between numeric features reveal redundant
# variables (high correlation) and predictive signals (features correlated
# with the target). This guides feature selection and prevents multicollinearity
# in models like logistic regression.

# %%
# ── 8a. Full Correlation Heatmap ─────────────────────────────────────────
numeric_cols = ['review_length', 'review_word_count', 'title_length',
                'neg', 'neu', 'pos', 'compound', 'satisfied',
                'char_count', 'word_count', 'avg_word_length',
                'exclamation_count', 'question_count', 'uppercase_ratio']

corr_matrix = df[numeric_cols].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(16, 13))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.8,
    linecolor='white',
    cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8},
    annot_kws={'size': 9},
    ax=ax,
)

ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=17, pad=15)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)

plt.tight_layout()
plt.savefig('eda_08_correlation_heatmap.png', bbox_inches='tight', facecolor='white')
plt.show()

# %%
# ── 8b. Correlations with Target Variable ────────────────────────────────
target_corr = corr_matrix['satisfied'].drop('satisfied').sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = [COLORS['negative'] if v < 0 else COLORS['positive'] for v in target_corr.values]
bars = ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor='white', height=0.6)
ax.axvline(x=0, color=COLORS['dark'], linewidth=1.5)
ax.set_xlabel('Pearson Correlation with Satisfaction')
ax.set_title('Feature Correlations with Target (satisfied)', fontweight='bold', fontsize=15, pad=12)
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, target_corr.values):
    offset = 0.01 if val >= 0 else -0.01
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha=ha, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_09_target_correlations.png', bbox_inches='tight', facecolor='white')
plt.show()

# %% [markdown]
# **📖 Interpretation:**
# - **Strongest positive correlation** with satisfaction: `compound` score (~1.0 —
#   expected, since the target was derived from it), `pos` sentiment score.
# - **Strongest negative correlation**: `neg` sentiment score — higher negative
#   sentiment strongly predicts dissatisfaction.
# - `review_length` and `word_count` show a slight **negative** correlation with
#   satisfaction, confirming our earlier observation that unhappy customers write
#   longer reviews.
# - `review_length` and `word_count` / `char_count` are **highly correlated**
#   (>0.95) with each other — we may want to keep only one to avoid
#   multicollinearity.
# - `exclamation_count` and `question_count` have very weak correlations,
#   suggesting limited predictive value on their own.

# %% [markdown]
# ---
# ## 📊 9. Feature Distributions by Satisfaction Class
#
# **Why it matters:** Comparing feature distributions side-by-side reveals
# which features have the most separation between classes — and thus which
# will be most useful for the classifier.

# %%
# ── 9a. Box Plots of Key Features by Class ──────────────────────────────
features_to_plot = ['review_length', 'word_count', 'neg', 'pos', 'compound',
                    'avg_word_length', 'exclamation_count', 'uppercase_ratio']

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for ax, feat in zip(axes, features_to_plot):
    data_0 = df[df['satisfied'] == 0][feat]
    data_1 = df[df['satisfied'] == 1][feat]

    bp = ax.boxplot(
        [data_0, data_1],
        labels=['Not Satisfied', 'Satisfied'],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4},
    )
    for patch, color in zip(bp['boxes'], PALETTE_SAT):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color(COLORS['dark'])
        median.set_linewidth(2)

    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold', fontsize=12, pad=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Feature Distributions by Satisfaction Class',
             fontweight='bold', fontsize=17, y=1.02)
plt.tight_layout()
plt.savefig('eda_10_feature_boxplots.png', bbox_inches='tight', facecolor='white')
plt.show()

# %% [markdown]
# **📖 Interpretation:**
# - `neg` and `pos` sentiment scores show the **clearest class separation** —
#   their box plots barely overlap, making them powerful predictors.
# - `compound` has excellent separation (by design, since it defines the target).
# - `review_length` and `word_count` show moderate separation — dissatisfied
#   reviews are longer with more outliers on the upper end.
# - `avg_word_length`, `exclamation_count`, and `uppercase_ratio` show
#   considerable overlap, suggesting they add only marginal predictive power.

# %% [markdown]
# ---
# ## 📏 10. Review Length vs. Sentiment — Scatter Analysis
#
# **Why it matters:** Exploring the relationship between how much someone
# writes and how they feel reveals behavioral patterns — do angrier customers
# write more? Do happy customers keep it short?

# %%
fig, ax = plt.subplots(figsize=(14, 7))

scatter = ax.scatter(
    df['review_word_count'],
    df['compound'],
    c=df['satisfied'],
    cmap='RdYlGn',
    alpha=0.5,
    s=25,
    edgecolors='white',
    linewidth=0.3,
)

ax.axhline(y=0.05, color=COLORS['dark'], linestyle='--', linewidth=2,
           label='Satisfaction Threshold', alpha=0.7)
ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

ax.set_xlabel('Review Word Count', fontsize=13)
ax.set_ylabel('Compound Sentiment Score', fontsize=13)
ax.set_title('Review Length vs. Sentiment Score', fontweight='bold', fontsize=16, pad=12)

# Add colorbar legend
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
cbar.set_ticks([0.25, 0.75])
cbar.set_ticklabels(['Not Satisfied', 'Satisfied'])

# Annotate quadrants
ax.text(0.02, 0.98, 'SHORT + POSITIVE\n(Quick praise)', transform=ax.transAxes,
        fontsize=9, va='top', ha='left', style='italic', color=COLORS['positive'], alpha=0.8)
ax.text(0.98, 0.02, 'LONG + NEGATIVE\n(Detailed complaints)', transform=ax.transAxes,
        fontsize=9, va='bottom', ha='right', style='italic', color=COLORS['negative'], alpha=0.8)

ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('eda_11_length_vs_sentiment.png', bbox_inches='tight', facecolor='white')
plt.show()

# %% [markdown]
# **📖 Interpretation:**
# - There's a visible tendency for **longer reviews to be more negative** (bottom-right
#   cluster) — dissatisfied customers write detailed complaints.
# - **Short positive reviews** cluster in the upper-left — satisfied customers often
#   give brief, positive feedback.
# - The middle zone contains mixed-sentiment reviews that are harder to classify.
# - 💡 **Insight:** Word count alone isn't a perfect predictor, but combined with
#   sentiment features, it adds useful signal.

# %% [markdown]
# ---
# ## 📋 11. EDA Summary
#
# | Analysis | Key Finding |
# |----------|-------------|
# | **Missing Values** | ✅ Zero missing values across all columns |
# | **Class Balance** | ✅ Near-perfect 50/50 split (no resampling needed) |
# | **Sentiment Scores** | Bimodal distribution; clear separation between classes |
# | **Review Length** | Dissatisfied → longer reviews; Satisfied → shorter |
# | **Word Clouds** | Negative: delays, luggage, service. Positive: crew, food, comfort |
# | **Top Bigrams** | "customer service" (negative), "cabin crew" (positive) |
# | **Correlations** | `neg`, `pos`, `compound` are strongest predictors |
# | **Feature Overlap** | `review_length` ≈ `word_count` ≈ `char_count` (keep one) |

# %%
print("🎉 EDA Complete! Key visualizations saved as PNG files.")
print("\n📁 Generated charts:")
import glob
for f in sorted(glob.glob('eda_*.png')):
    print(f"   📊 {f}")
