"""
Phase 6 — Topic Modeling on Negative Reviews (LDA)
LDA used: BERTopic collapses to 2 topics on lemmatized airline text.
"""
import pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings('ignore')

df  = pd.read_csv('british_airways_sentiment.csv')
neg = df[df['sentiment_label'] == 'Negative']['cleaned_review'].dropna().tolist()
print(f"Negative reviews: {len(neg)}")

TOPIC_NAMES = {
    'delay hour':          'Flight Delays & Punctuality',
    'cancel':              'Flight Cancellations',
    'bag luggage':         'Baggage & Luggage',
    'seat legroom':        'Seat Comfort & Legroom',
    'crew staff rude':     'Cabin Crew & Staff Behaviour',
    'food meal drink':     'Food & Catering Quality',
    'check board gate':    'Check-in & Boarding Process',
    'customer service':    'Customer Service & Support',
    'refund compens':      'Refunds & Compensation',
    'lounge airport':      'Airport Lounge Experience',
}

cv  = CountVectorizer(max_features=3000, min_df=3,
                      ngram_range=(1, 2), stop_words='english')
dtm  = cv.fit_transform(neg)
vocab = np.array(cv.get_feature_names_out())

N_TOPICS = 10
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42,
                                 max_iter=30, n_jobs=-1)
lda.fit(dtm)

doc_topics = lda.transform(dtm).argmax(axis=1)
rows = []
for i, comp in enumerate(lda.components_):
    top_words = ' '.join(vocab[np.argsort(comp)[-8:][::-1]])
    count     = int((doc_topics == i).sum())
    bname     = f'Topic {i}'
    for key, name in TOPIC_NAMES.items():
        if any(k in top_words for k in key.split()):
            bname = name; break
    rows.append({'Topic': i, 'Count': count,
                 'Business_Name': bname, 'Top_Words': top_words})

topic_table = pd.DataFrame(rows).sort_values('Count', ascending=False).reset_index(drop=True)
print("\n", topic_table[['Topic','Count','Business_Name','Top_Words']].to_string(index=False))

# ── Chart ─────────────────────────────────────────────────────────────────
plot_df = topic_table.head(10).sort_values('Count')
fig, ax = plt.subplots(figsize=(13, 7))
colors  = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(plot_df)))
bars    = ax.barh(plot_df['Business_Name'], plot_df['Count'],
                  color=colors, edgecolor='white', height=0.65)
for bar in bars:
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(int(bar.get_width())), va='center', fontweight='bold')
ax.set_xlabel('Number of Reviews', fontsize=12)
ax.set_title('Phase 6 — Top Complaint Topics (LDA)\nNegative Reviews Only',
             fontweight='bold', fontsize=14, pad=12)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('phase6_topic_frequencies.png', bbox_inches='tight',
            facecolor='white', dpi=120)
plt.close()

topic_table.to_csv('phase6_topics.csv', index=False)
with open('phase6_lda_model.pkl', 'wb') as f:
    pickle.dump({'lda': lda, 'vectorizer': cv}, f, protocol=4)

print("\nFiles: phase6_topics.csv, phase6_topic_frequencies.png, phase6_lda_model.pkl")
