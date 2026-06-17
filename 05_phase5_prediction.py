"""
Phase 5 — Customer Satisfaction Prediction
Random Forest vs XGBoost on TF-IDF + numeric features
"""
import pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('british_airways_sentiment.csv')

NUM_COLS = ['sentiment_score', 'review_length', 'word_count',
            'exclamation_count', 'question_count', 'uppercase_ratio',
            'vader_neg', 'vader_neu', 'vader_pos']
NUM_COLS = [c for c in NUM_COLS if c in df.columns]

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                        min_df=2, sublinear_tf=True)
X_text = tfidf.fit_transform(df['cleaned_review'].fillna(''))

scaler = StandardScaler()
X_num  = scaler.fit_transform(df[NUM_COLS])
X      = hstack([X_text, csr_matrix(X_num)])
y      = df['satisfied'].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Models ────────────────────────────────────────────────────────────────
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=4,
        random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0),
}

results, trained = {}, {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    y_pred  = m.predict(X_te)
    y_proba = m.predict_proba(X_te)[:, 1]
    cv_f1   = cross_val_score(m, X_tr, y_tr, cv=cv,
                              scoring='f1_weighted', n_jobs=-1)
    results[name] = dict(
        accuracy  = accuracy_score(y_te, y_pred),
        f1        = f1_score(y_te, y_pred, average='weighted'),
        roc_auc   = roc_auc_score(y_te, y_proba),
        cv_f1     = cv_f1.mean(),
        cv_std    = cv_f1.std(),
        y_pred    = y_pred,
        y_proba   = y_proba,
    )
    trained[name] = m
    print(f"{name:20s}  acc={results[name]['accuracy']:.4f}  "
          f"f1={results[name]['f1']:.4f}  auc={results[name]['roc_auc']:.4f}  "
          f"cv={cv_f1.mean():.4f}±{cv_f1.std():.4f}")

# ── Comparison table ──────────────────────────────────────────────────────
cmp = pd.DataFrame({n: {k: round(v, 4) for k, v in m.items()
                         if k not in ('y_pred', 'y_proba')}
                    for n, m in results.items()}).T
print("\n", cmp.to_string())

best_name = max(results, key=lambda k: results[k]['f1'])
best_model = trained[best_name]
print(f"\nBest: {best_name}")

# ── Confusion matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, color) in zip(axes, zip(models, ['#3498DB', '#E74C3C'])):
    cm = confusion_matrix(y_te, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(color, as_cmap=True),
                xticklabels=['Not Sat.', 'Satisfied'],
                yticklabels=['Not Sat.', 'Satisfied'],
                linewidths=1.5, linecolor='white', cbar=False)
    ax.set_title(f"{name}  F1={results[name]['f1']:.4f}", fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.suptitle('Phase 5 — Confusion Matrices', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('phase5_confusion_matrices.png', bbox_inches='tight',
            facecolor='white', dpi=120)
plt.close()

# ── Feature importance (top 25 each) ─────────────────────────────────────
feat_names = np.array(list(tfidf.get_feature_names_out()) + NUM_COLS)
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
for ax, (name, color) in zip(axes, zip(models, ['#3498DB', '#E74C3C'])):
    m = trained[name]
    if hasattr(m, 'feature_importances_'):
        imp = m.feature_importances_
    else:
        imp = np.abs(m.coef_[0])
    top = np.argsort(imp)[-25:][::-1]
    bar_colors = ['#F39C12' if feat_names[i] in NUM_COLS else color for i in top]
    ax.barh(range(25), imp[top][::-1],
            color=bar_colors[::-1], edgecolor='white')
    ax.set_yticks(range(25))
    ax.set_yticklabels(feat_names[top][::-1], fontsize=9)
    ax.set_title(f'{name} — Top 25 Features', fontweight='bold')
    ax.set_xlabel('Importance')
plt.suptitle('Phase 5 — Feature Importance (Gold=Numeric)',
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('phase5_feature_importance.png', bbox_inches='tight',
            facecolor='white', dpi=120)
plt.close()

# ── Save best model ───────────────────────────────────────────────────────
with open('phase5_best_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'model_name': best_name,
                 'vectorizer': tfidf, 'scaler': scaler,
                 'numeric_cols': NUM_COLS,
                 'metrics': {k: v for k, v in results[best_name].items()
                              if k not in ('y_pred','y_proba')}}, f, protocol=4)

cmp.to_csv('phase5_model_comparison.csv')
print("\nFiles: phase5_best_model.pkl, phase5_model_comparison.csv, "
      "phase5_confusion_matrices.png, phase5_feature_importance.png")
