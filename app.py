"""
AI-Powered Customer Satisfaction Predictor
Streamlit Dashboard · app.py
"""
import re, string, pickle, warnings
import numpy as np
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    color: #e2e8f0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.10) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #94a3b8; margin: 0.4rem 0 0; font-size: 0.95rem; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    backdrop-filter: blur(8px);
    transition: transform 0.2s, border-color 0.2s;
    height: 100%;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(99,102,241,0.4); }
.metric-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #64748b; margin-bottom: 0.4rem; }
.metric-value { font-size: 1.55rem; font-weight: 700; line-height: 1.2; }
.metric-sub { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }

/* Sentiment badge */
.badge {
    display: inline-block; padding: 0.25rem 0.85rem;
    border-radius: 999px; font-size: 0.78rem;
    font-weight: 600; letter-spacing: 0.04em;
}
.badge-positive { background: rgba(52,211,153,0.15); color: #34d399;
    border: 1px solid rgba(52,211,153,0.3); }
.badge-negative { background: rgba(239,68,68,0.15); color: #f87171;
    border: 1px solid rgba(239,68,68,0.3); }
.badge-neutral  { background: rgba(251,191,36,0.15); color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.3); }

/* Score bar */
.score-track {
    height: 8px; border-radius: 999px;
    background: rgba(255,255,255,0.07);
    margin: 0.5rem 0;
    overflow: hidden;
}
.score-fill {
    height: 100%; border-radius: 999px;
    transition: width 0.6s ease;
}

/* Recommendation card */
.rec-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(168,85,247,0.08));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.rec-title { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #818cf8; margin-bottom: 0.6rem; }
.rec-action { font-size: 1rem; font-weight: 500; color: #e2e8f0; line-height: 1.5; }
.rec-meta { display: flex; gap: 1rem; margin-top: 0.8rem; flex-wrap: wrap; }
.rec-chip {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 0.25rem 0.7rem;
    font-size: 0.75rem; color: #94a3b8;
}
.rec-chip span { color: #c4b5fd; font-weight: 600; }

/* Priority chip colours */
.prio-CRITICAL { color: #f87171 !important; }
.prio-HIGH     { color: #fb923c !important; }
.prio-MEDIUM   { color: #fbbf24 !important; }
.prio-LOW      { color: #34d399 !important; }

/* Textarea */
textarea { background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important; color: #e2e8f0 !important; }
textarea:focus { border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    padding: 0.6rem 2rem !important; font-size: 0.95rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

/* Sidebar model info */
.model-pill {
    background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.25);
    border-radius: 8px; padding: 0.3rem 0.8rem;
    font-size: 0.78rem; color: #34d399; font-weight: 600;
    display: inline-block; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load assets (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('phase5_best_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_nlp():
    import nltk, spacy
    for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
        try: nltk.data.find(f'corpora/{pkg}')
        except LookupError: nltk.download(pkg, quiet=True)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    return nlp

bundle = load_model()
model      = bundle['model']
vectorizer = bundle['vectorizer']
scaler     = bundle['scaler']
num_cols   = bundle['numeric_cols']
vader      = SentimentIntensityAnalyzer()
nlp_model  = load_nlp()

# ─── Phase 7 recommendation engine ──────────────────────────────────────────
RECOMMENDATIONS = {
    'Flight Delays & Punctuality': {
        'action':   'Proactive delay notifications via SMS/app + automatic meal vouchers',
        'priority': 'HIGH', 'kpi': 'On-time departure rate, rebooking speed',
        'owner':    'Operations & Ground Services',
    },
    'Flight Cancellations': {
        'action':   'Instant rebooking portal, same-day guarantee, full refund SLA <24h',
        'priority': 'CRITICAL', 'kpi': 'Cancellation rate, refund processing time',
        'owner':    'Revenue Management',
    },
    'Baggage & Luggage': {
        'action':   'Real-time bag tracking in app, priority delivery SLA, compensation portal',
        'priority': 'HIGH', 'kpi': 'Mishandled bag rate, claim resolution time',
        'owner':    'Ground Operations',
    },
    'Seat Comfort & Legroom': {
        'action':   'Cabin retrofit schedule, upgrade priority for loyalty members',
        'priority': 'MEDIUM', 'kpi': 'Seat satisfaction NPS, cabin retrofit %',
        'owner':    'Fleet & Cabin Management',
    },
    'Cabin Crew & Staff Behaviour': {
        'action':   'Customer experience training program, crew feedback loop',
        'priority': 'HIGH', 'kpi': 'Crew satisfaction score, complaint-per-flight rate',
        'owner':    'HR & Inflight Services',
    },
    'Food & Catering Quality': {
        'action':   'Menu refresh quarterly, dietary option expansion, hot meals on short-haul',
        'priority': 'MEDIUM', 'kpi': 'Catering satisfaction score',
        'owner':    'Catering & Partnerships',
    },
    'Check-in & Boarding Process': {
        'action':   'Self-bag-drop expansion, zone boarding enforcement, fast-track lanes',
        'priority': 'MEDIUM', 'kpi': 'Boarding time, check-in queue length',
        'owner':    'Airport Operations',
    },
    'Customer Service & Support': {
        'action':   'AI chatbot for Tier-1 queries, <2h response SLA, staff upskilling',
        'priority': 'HIGH', 'kpi': 'First-contact resolution rate, NPS',
        'owner':    'Customer Relations',
    },
    'Refunds & Compensation': {
        'action':   'Automated EU261 compensation calculator, 7-day refund guarantee',
        'priority': 'HIGH', 'kpi': 'Refund processing time, claim rejection rate',
        'owner':    'Finance & Legal',
    },
    'Airport Lounge Experience': {
        'action':   'Capacity management, food quality audit, digital queue system',
        'priority': 'LOW', 'kpi': 'Lounge satisfaction score, occupancy rate',
        'owner':    'Lounges & Premium Services',
    },
}
DEFAULT_REC = {
    'action': 'Collect structured feedback and escalate to quality assurance team',
    'priority': 'MEDIUM', 'kpi': 'Customer satisfaction score',
    'owner': 'Quality Assurance',
}
TOPIC_KEYWORDS = {
    'Flight Delays & Punctuality':   ['delay', 'late', 'hour', 'wait', 'punctual', 'on time'],
    'Flight Cancellations':          ['cancel', 'cancelled', 'reschedul', 'rebook'],
    'Baggage & Luggage':             ['bag', 'luggage', 'suitcase', 'lost bag', 'missing bag'],
    'Seat Comfort & Legroom':        ['seat', 'legroom', 'cramped', 'comfort', 'recline'],
    'Cabin Crew & Staff Behaviour':  ['crew', 'staff', 'rude', 'hostile', 'attitude', 'service'],
    'Food & Catering Quality':       ['food', 'meal', 'drink', 'catering', 'menu', 'snack'],
    'Check-in & Boarding Process':   ['check', 'board', 'gate', 'queue', 'boarding'],
    'Customer Service & Support':    ['customer service', 'support', 'helpline', 'call centre'],
    'Refunds & Compensation':        ['refund', 'compensat', 'voucher', 'claim', 'reimburse'],
    'Airport Lounge Experience':     ['lounge', 'airport', 'terminal', 'departure area'],
}

PRIORITY_ORDER = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}

# ─── Preprocessing helpers ───────────────────────────────────────────────────
_STOP = None
def _get_stopwords():
    global _STOP
    if _STOP is None:
        from nltk.corpus import stopwords
        _STOP = set(stopwords.words('english'))
    return _STOP

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text: str) -> str:
    cleaned = clean_text(text)
    stop    = _get_stopwords()
    doc     = nlp_model(cleaned)
    tokens  = [t.lemma_ for t in doc
               if t.is_alpha and t.lemma_ not in stop and len(t.lemma_) > 2]
    return ' '.join(tokens)

def detect_topic(text: str) -> str:
    lower = text.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(k in lower for k in kws):
            return topic
    return 'General Service Issue'

def get_recommendation(topic: str, sentiment_score: float) -> dict:
    rec = RECOMMENDATIONS.get(topic, DEFAULT_REC).copy()
    if sentiment_score < -0.6 and rec['priority'] == 'MEDIUM':
        rec['priority'] = 'HIGH'
    if sentiment_score < -0.8 and rec['priority'] == 'HIGH':
        rec['priority'] = 'CRITICAL'
    return rec

# ─── Inference pipeline ──────────────────────────────────────────────────────
def predict(review: str) -> dict:
    # VADER on raw text
    vs          = vader.polarity_scores(review)
    score       = vs['compound']
    vader_label = ('Positive' if score >= 0.05 else
                   'Negative' if score <= -0.05 else 'Neutral')

    # NLP clean for TF-IDF
    cleaned  = preprocess(review)
    X_text   = vectorizer.transform([cleaned])

    # Numeric features (match training order)
    rev_clean_for_count = clean_text(review)
    words               = rev_clean_for_count.split()
    feat_map = {
        'sentiment_score':    score,
        'review_length':      len(review),
        'word_count':         len(words),
        'exclamation_count':  review.count('!'),
        'question_count':     review.count('?'),
        'uppercase_ratio':    (sum(1 for c in review if c.isupper()) / max(len(review), 1)),
        'vader_neg':          vs['neg'],
        'vader_neu':          vs['neu'],
        'vader_pos':          vs['pos'],
    }
    num_vec = np.array([[feat_map.get(c, 0.0) for c in num_cols]])
    X_num   = scaler.transform(num_vec)
    X       = hstack([X_text, csr_matrix(X_num)])

    pred    = int(model.predict(X)[0])
    prob    = float(model.predict_proba(X)[0][pred])
    topic   = detect_topic(review)
    rec     = get_recommendation(topic, score)

    return {
        'vader_label':   vader_label,
        'sentiment_score': score,
        'satisfied':     pred,
        'confidence':    prob,
        'topic':         topic,
        'recommendation': rec,
    }

# ─── Score bar HTML ──────────────────────────────────────────────────────────
def score_bar(value: float, color: str, width_pct: float) -> str:
    return f"""
    <div class="score-track">
      <div class="score-fill" style="width:{width_pct:.1f}%; background:{color};"></div>
    </div>"""

# ─── UI ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>✈️ AI-Powered Customer Satisfaction Predictor</h1>
  <p>British Airways Review Analysis · Sentiment · Topic Detection · Recommendations</p>
</div>
""", unsafe_allow_html=True)

col_input, col_results = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("#### 📝 Enter Customer Review")
    review = st.text_area(
        label="review",
        label_visibility="collapsed",
        placeholder="Paste a British Airways review here…\n\nExample: 'The cabin crew were extremely rude and unhelpful. Our flight was delayed by 3 hours with no explanation.'",
        height=220,
        key="review_input",
    )

    st.markdown("")
    run = st.button("🔍 Analyse Review", use_container_width=True)

    # Model info
    st.markdown("---")
    st.markdown(f"""
    <div class="model-pill">XGBoost · F1 = {bundle['metrics']['f1']:.4f}</div><br>
    <span style="font-size:0.78rem;color:#64748b;">
    Accuracy: {bundle['metrics']['accuracy']:.2%} · AUC: {bundle['metrics']['roc_auc']:.4f}<br>
    CV F1: {bundle['metrics']['cv_f1']:.4f} ± {bundle['metrics']['cv_std']:.4f}
    </span>
    """, unsafe_allow_html=True)

with col_results:
    if run and review.strip():
        with st.spinner("Analysing…"):
            out = predict(review.strip())

        lbl   = out['vader_label']
        score = out['sentiment_score']
        sat   = out['satisfied']
        conf  = out['confidence']
        topic = out['topic']
        rec   = out['recommendation']

        # ── Sentiment ──────────────────────────────────────────────────
        badge_cls = f"badge-{lbl.lower()}"
        lbl_emoji = {'Positive': '😊', 'Negative': '😞', 'Neutral': '😐'}[lbl]
        score_pct = (score + 1) / 2 * 100
        score_color = ('#34d399' if score >= 0.05 else
                       '#f87171' if score <= -0.05 else '#fbbf24')

        # ── Satisfaction ───────────────────────────────────────────────
        sat_label = 'Satisfied' if sat else 'Not Satisfied'
        sat_emoji = '✅' if sat else '❌'
        sat_color = '#34d399' if sat else '#f87171'
        conf_pct  = conf * 100

        # ── Priority colour ────────────────────────────────────────────
        prio = rec['priority']
        prio_cls = f"prio-{prio}"

        st.markdown("#### 📊 Analysis Results")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Sentiment</div>
              <div class="metric-value">{lbl_emoji} {lbl}</div>
              {score_bar(score, score_color, score_pct)}
              <div class="metric-sub">VADER compound: <b style="color:{score_color}">{score:+.4f}</b></div>
            </div>""", unsafe_allow_html=True)

        with r1c2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Predicted Satisfaction</div>
              <div class="metric-value" style="color:{sat_color}">{sat_emoji} {sat_label}</div>
              {score_bar(conf, sat_color, conf_pct)}
              <div class="metric-sub">Model confidence: <b style="color:{sat_color}">{conf_pct:.1f}%</b></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Detected Topic</div>
              <div class="metric-value" style="font-size:1.1rem;color:#a78bfa;">🏷️ {topic}</div>
              <div class="metric-sub">Matched via keyword extraction</div>
            </div>""", unsafe_allow_html=True)

        with r2c2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Sentiment Score</div>
              <div class="metric-value" style="color:{score_color}">{score:+.4f}</div>
              <div class="metric-sub">Range −1.0 (very negative) → +1.0 (very positive)</div>
            </div>""", unsafe_allow_html=True)

        # ── Recommendation card ────────────────────────────────────────
        st.markdown(f"""
        <div class="rec-card">
          <div class="rec-title">💡 Service Improvement Recommendation</div>
          <div class="rec-action">{rec['action']}</div>
          <div class="rec-meta">
            <div class="rec-chip">Priority: <span class="{prio_cls}">{prio}</span></div>
            <div class="rec-chip">Owner: <span>{rec['owner']}</span></div>
            <div class="rec-chip">KPI: <span>{rec['kpi']}</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

    elif run and not review.strip():
        st.warning("Please enter a review before analysing.")
    else:
        st.markdown("""
        <div style="
          height: 340px; display:flex; flex-direction:column;
          align-items:center; justify-content:center;
          border: 1px dashed rgba(99,102,241,0.25);
          border-radius: 16px; color: #475569; text-align:center;
          padding: 2rem;
        ">
          <div style="font-size:3rem;margin-bottom:1rem">✈️</div>
          <div style="font-size:1rem;font-weight:500;color:#64748b">
            Enter a British Airways review on the left<br>and click <b style="color:#818cf8">Analyse Review</b>
          </div>
        </div>""", unsafe_allow_html=True)
