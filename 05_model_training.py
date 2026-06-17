# %% [markdown]
# # 🤖 Model Training & Evaluation — British Airways Satisfaction Predictor
#
# **Checkpoint 5 of the Saras AI Project**
#
# Trains and compares four classification models to predict customer satisfaction
# from British Airways reviews using TF-IDF text features + engineered numeric features.
#
# ### Models Trained
# | Model | Type | Strength |
# |-------|------|----------|
# | Logistic Regression | Linear | Fast, interpretable, great baseline |
# | Random Forest | Ensemble (bagging) | Handles non-linearity, feature importance |
# | XGBoost | Ensemble (boosting) | High accuracy, regularized |
# | LightGBM | Ensemble (boosting) | Fastest, memory-efficient |
#
# ### Evaluation Metrics
# Accuracy · Precision · Recall · F1 · ROC-AUC · Cross-validation (5-fold)

# %% [markdown]
# ---
# ## 📦 1. Setup

# %%
import subprocess, sys, warnings, time, pickle
warnings.filterwarnings('ignore')

def install_pkg(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  📥 Installing {pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  ✅ {pkg} installed")

print("Checking dependencies...")
install_pkg('pandas')
install_pkg('numpy')
install_pkg('scikit-learn', 'sklearn')
install_pkg('xgboost')
install_pkg('lightgbm')
install_pkg('matplotlib')
install_pkg('seaborn')

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11, 'axes.titlesize': 14})

COLORS = {
    'lr': '#3498DB', 'rf': '#2ECC71', 'xgb': '#E74C3C',
    'lgb': '#9B59B6', 'dark': '#2C3E50', 'gold': '#F39C12',
}
MODEL_NAMES = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']
MODEL_COLORS = [COLORS['lr'], COLORS['rf'], COLORS['xgb'], COLORS['lgb']]

print("\n🚀 All dependencies ready!")

# %% [markdown]
# ---
# ## 📂 2. Load Data

# %%
# ── Load preprocessed TF-IDF data ────────────────────────────────────────
print("Loading preprocessed_data.pkl...")
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_tfidf = data['X_train']
X_test_tfidf  = data['X_test']
y_train        = data['y_train']
y_test         = data['y_test']
vectorizer     = data['vectorizer']
numeric_cols   = data.get('numeric_cols', [])

print(f"✅ TF-IDF train: {X_train_tfidf.shape}")
print(f"✅ TF-IDF test:  {X_test_tfidf.shape}")
print(f"✅ y_train dist: {pd.Series(y_train).value_counts().to_dict()}")
print(f"✅ y_test dist:  {pd.Series(y_test).value_counts().to_dict()}")
print(f"✅ Numeric cols: {numeric_cols}")

# %%
# ── Load the sentiment-enriched dataset for numeric features ──────────────
df = pd.read_csv('british_airways_sentiment.csv')
print(f"\n📊 Sentiment dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Build numeric feature matrix ──────────────────────────────────────────
NUMERIC_FEATURES = [
    'compound', 'pos', 'neg', 'neu',
    'review_length', 'word_count', 'exclamation_count',
    'question_count', 'uppercase_ratio',
]
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in df.columns]

print(f"📋 Numeric features used: {NUMERIC_FEATURES}")

# Split numeric features to match train/test split
# We need to use the same split indices — use stratified split reproducibly
from sklearn.model_selection import train_test_split

X_num = df[NUMERIC_FEATURES].values
y_all = df['satisfied'].values

X_num_train, X_num_test, _, _ = train_test_split(
    X_num, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Scale numeric features
scaler = StandardScaler()
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled  = scaler.transform(X_num_test)

# ── Combine TF-IDF + numeric ──────────────────────────────────────────────
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_num_train_scaled)])
X_test_combined  = hstack([X_test_tfidf,  csr_matrix(X_num_test_scaled)])

print(f"\n✅ Combined feature matrix:")
print(f"   Train: {X_train_combined.shape}  "
      f"(TF-IDF: {X_train_tfidf.shape[1]:,} + numeric: {len(NUMERIC_FEATURES)})")
print(f"   Test:  {X_test_combined.shape}")

# %% [markdown]
# ---
# ## 🏋️ 3. Train All Models

# %%
# ── Define models ─────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs',
        random_state=42, n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, verbosity=0
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        num_leaves=31, subsample=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    ),
}

# ── Train & evaluate ──────────────────────────────────────────────────────
results = {}
trained_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 65)
print("🏋️  Training Models")
print("=" * 65)

for name, model in models.items():
    print(f"\n▶ {name}...")
    t0 = time.time()

    # Train
    model.fit(X_train_combined, y_train)
    train_time = time.time() - t0

    # Predict
    y_pred  = model.predict(X_test_combined)
    try:
        y_proba = model.predict_proba(X_test_combined)[:, 1]
    except Exception:
        y_proba = y_pred.astype(float)

    # Cross-validation (on TF-IDF+numeric combined)
    cv_scores = cross_val_score(model, X_train_combined, y_train,
                                cv=cv, scoring='f1_weighted', n_jobs=-1)

    # Metrics
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall':    recall_score(y_test, y_pred, average='weighted'),
        'f1':        f1_score(y_test, y_pred, average='weighted'),
        'roc_auc':   roc_auc_score(y_test, y_proba),
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std':  cv_scores.std(),
        'train_time': train_time,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
    }
    results[name] = metrics
    trained_models[name] = model

    print(f"   ✅ Accuracy: {metrics['accuracy']:.4f} | "
          f"F1: {metrics['f1']:.4f} | "
          f"ROC-AUC: {metrics['roc_auc']:.4f} | "
          f"CV-F1: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f} | "
          f"Time: {train_time:.1f}s")

# %% [markdown]
# ---
# ## 📊 4. Results Summary

# %%
# ── Build summary DataFrame ───────────────────────────────────────────────
summary_data = {
    name: {
        'Accuracy':  f"{m['accuracy']:.4f}",
        'Precision': f"{m['precision']:.4f}",
        'Recall':    f"{m['recall']:.4f}",
        'F1':        f"{m['f1']:.4f}",
        'ROC-AUC':   f"{m['roc_auc']:.4f}",
        'CV F1':     f"{m['cv_f1_mean']:.4f} ± {m['cv_f1_std']:.4f}",
        'Train Time': f"{m['train_time']:.1f}s",
    }
    for name, m in results.items()
}

summary_df = pd.DataFrame(summary_data).T
print("\n" + "=" * 75)
print("📊 MODEL PERFORMANCE SUMMARY")
print("=" * 75)
print(summary_df.to_string())

best_model_name = max(results, key=lambda k: results[k]['f1'])
best_f1 = results[best_model_name]['f1']
print(f"\n🏆 Best model: {best_model_name} (F1 = {best_f1:.4f})")

# %% [markdown]
# ### Detailed Classification Reports

# %%
for name, m in results.items():
    print(f"\n{'='*65}")
    print(f"📋 {name}")
    print(f"{'='*65}")
    print(classification_report(y_test, m['y_pred'],
                                target_names=['Not Satisfied', 'Satisfied'],
                                digits=4))

# %% [markdown]
# ---
# ## 📊 5. Visualizations

# %% [markdown]
# ### Chart 1 — Model Comparison (All Metrics)

# %%
metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# ── Grouped bar chart ────────────────────────────────────────────────────
ax = axes[0]
x = np.arange(len(metric_labels))
width = 0.18
for i, (name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
    vals = [results[name][k] for k in metric_keys]
    bars = ax.bar(x + i*width - width*1.5, vals, width,
                  label=name, color=color, edgecolor='white', alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel('Score')
ax.set_ylim(0.0, 1.12)
ax.set_title('Model Performance — All Metrics', fontweight='bold', fontsize=15)
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)

# ── Radar chart ──────────────────────────────────────────────────────────
ax = axes[1]
categories = metric_labels
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(1, 2, 2, polar=True)
ax.set_facecolor('#f8f9fa')

for i, (name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
    vals = [results[name][k] for k in metric_keys]
    vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=2.5, label=name)
    ax.fill(angles, vals, color=color, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=8)
ax.set_title('Performance Radar Chart', fontweight='bold', fontsize=15,
             pad=20)
ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.35, 1.15))
ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('model_01_comparison.png', bbox_inches='tight', facecolor='white')
plt.show()
print("📊 Saved: model_01_comparison.png")

# %% [markdown]
# ### Chart 2 — ROC Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# ── All ROC curves ────────────────────────────────────────────────────────
ax = axes[0]
for name, color in zip(MODEL_NAMES, MODEL_COLORS):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
    auc = results[name]['roc_auc']
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC={auc:.4f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontweight='bold', fontsize=15)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# ── Best model ROC (zoomed top-left) ─────────────────────────────────────
ax = axes[1]
name = best_model_name
color = MODEL_COLORS[MODEL_NAMES.index(name)]
fpr, tpr, thresholds = roc_curve(y_test, results[name]['y_proba'])
auc = results[name]['roc_auc']

ax.plot(fpr, tpr, color=color, linewidth=3, label=f"AUC = {auc:.4f}")
ax.fill_between(fpr, tpr, alpha=0.15, color=color)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)

# Optimal threshold point (closest to top-left)
optimal_idx = np.argmax(tpr - fpr)
ax.scatter(fpr[optimal_idx], tpr[optimal_idx], s=120, color='red', zorder=5,
           label=f"Optimal threshold={thresholds[optimal_idx]:.3f}")

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curve — 🏆 {name}', fontweight='bold', fontsize=15)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_02_roc_curves.png', bbox_inches='tight', facecolor='white')
plt.show()
print("📊 Saved: model_02_roc_curves.png")

# %% [markdown]
# ### Chart 3 — Confusion Matrices

# %%
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

for ax, (name, color) in zip(axes, zip(MODEL_NAMES, MODEL_COLORS)):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(color, as_cmap=True),
                linewidths=2, linecolor='white',
                xticklabels=['Not Sat.', 'Satisfied'],
                yticklabels=['Not Sat.', 'Satisfied'],
                cbar=False)

    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, f"({cm_pct[i,j]:.1f}%)",
                    ha='center', fontsize=9, color='dimgray')

    f1 = results[name]['f1']
    ax.set_title(f'{name}\nF1={f1:.4f}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')

plt.suptitle('Confusion Matrices — All Models', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('model_03_confusion_matrices.png', bbox_inches='tight', facecolor='white')
plt.show()
print("📊 Saved: model_03_confusion_matrices.png")

# %% [markdown]
# ### Chart 4 — Feature Importance (Top 25 Terms)

# %%
fig, axes = plt.subplots(2, 2, figsize=(22, 16))

feature_names_tfidf = vectorizer.get_feature_names_out()
feature_names_numeric = np.array(NUMERIC_FEATURES)
all_feature_names = np.concatenate([feature_names_tfidf, feature_names_numeric])
N_TOP = 25

for ax, (name, color) in zip(axes.flat, zip(MODEL_NAMES, MODEL_COLORS)):
    model = trained_models[name]

    try:
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            ax.set_title(f'{name}\n(No feature importance available)')
            continue

        top_idx = np.argsort(importances)[-N_TOP:][::-1]
        top_names = all_feature_names[top_idx] if len(all_feature_names) > max(top_idx) else [str(i) for i in top_idx]
        top_vals  = importances[top_idx]

        # Color numeric features differently
        bar_colors = []
        for fn in top_names:
            if fn in NUMERIC_FEATURES:
                bar_colors.append(COLORS['gold'])
            else:
                bar_colors.append(color)

        y_pos = range(N_TOP)
        ax.barh(y_pos, top_vals[::-1], color=bar_colors[::-1],
                edgecolor='white', height=0.75)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], fontsize=10)
        ax.set_title(f'{name}\nTop {N_TOP} Features', fontweight='bold', fontsize=13)
        ax.set_xlabel('Importance Score')
        ax.grid(axis='x', alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label='TF-IDF term'),
            Patch(facecolor=COLORS['gold'], label='Numeric feature'),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

    except Exception as e:
        ax.set_title(f'{name}\nError: {e}')

plt.suptitle('Feature Importance — All Models\n(Gold = Numeric, Colored = TF-IDF term)',
             fontweight='bold', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('model_04_feature_importance.png', bbox_inches='tight', facecolor='white')
plt.show()
print("📊 Saved: model_04_feature_importance.png")

# %% [markdown]
# ### Chart 5 — Cross-Validation F1 Scores

# %%
fig, ax = plt.subplots(figsize=(12, 6))

cv_means = [results[n]['cv_f1_mean'] for n in MODEL_NAMES]
cv_stds  = [results[n]['cv_f1_std']  for n in MODEL_NAMES]

bars = ax.bar(MODEL_NAMES, cv_means, color=MODEL_COLORS, edgecolor='white', width=0.5)
ax.errorbar(MODEL_NAMES, cv_means, yerr=[s*2 for s in cv_stds],
            fmt='none', color='black', capsize=8, capthick=2, linewidth=2)

for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cv_stds)*2 + 0.005,
            f'{mean:.4f}\n±{std:.4f}', ha='center', fontweight='bold', fontsize=12)

ax.set_ylabel('Weighted F1 Score')
ax.set_ylim(0, 1.1)
ax.set_title('5-Fold Cross-Validation F1 Scores\n(error bars = 2×std)', fontweight='bold', fontsize=15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_05_cross_validation.png', bbox_inches='tight', facecolor='white')
plt.show()
print("📊 Saved: model_05_cross_validation.png")

# %% [markdown]
# ---
# ## 💾 6. Save Best Model & Results

# %%
# ── Save best model ───────────────────────────────────────────────────────
best_model = trained_models[best_model_name]

model_package = {
    'model':         best_model,
    'model_name':    best_model_name,
    'vectorizer':    vectorizer,
    'scaler':        scaler,
    'numeric_cols':  NUMERIC_FEATURES,
    'metrics':       {k: v for k, v in results[best_model_name].items()
                      if k not in ('y_pred', 'y_proba')},
}

with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"✅ Best model saved: 'best_model.pkl'")
print(f"   Model: {best_model_name}")
print(f"   F1:    {results[best_model_name]['f1']:.4f}")
print(f"   AUC:   {results[best_model_name]['roc_auc']:.4f}")

# ── Save all results as CSV ───────────────────────────────────────────────
comparison_rows = []
for name, m in results.items():
    comparison_rows.append({
        'Model':     name,
        'Accuracy':  round(m['accuracy'],  4),
        'Precision': round(m['precision'], 4),
        'Recall':    round(m['recall'],    4),
        'F1':        round(m['f1'],        4),
        'ROC_AUC':   round(m['roc_auc'],   4),
        'CV_F1':     round(m['cv_f1_mean'],4),
        'CV_F1_Std': round(m['cv_f1_std'], 4),
        'Train_Time_s': round(m['train_time'], 2),
        'Is_Best':   (name == best_model_name),
    })

comparison_df = pd.DataFrame(comparison_rows).sort_values('F1', ascending=False)
comparison_df.to_csv('model_comparison.csv', index=False)

print(f"\n✅ Comparison table saved: 'model_comparison.csv'")
print(f"\n{comparison_df.to_string(index=False)}")

# %%
import glob
print("\n📊 All model charts generated:")
for f in sorted(glob.glob('model_*.png')):
    print(f"   📊 {f}")

print("\n🎉 Model Training Complete!")
print(f"   🏆 Best Model : {best_model_name}")
print(f"   📁 Saved      : best_model.pkl, model_comparison.csv")
