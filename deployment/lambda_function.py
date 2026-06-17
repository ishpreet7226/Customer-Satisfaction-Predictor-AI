"""
lambda_function.py — AWS Lambda handler
Lightweight: no spaCy, no NLTK downloads at runtime.
Preprocessing mirrors the training pipeline (03_nlp_preprocessing_pipeline.py)
using a static stopword list + simple regex lemmatisation stubs.
"""
import os, re, string, json, pickle, logging
import numpy as np
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Stopwords (NLTK english list, frozen at training time) ─────────────────
_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up",
    "down","in","out","on","off","over","under","again","further","then",
    "once","here","there","when","where","why","how","all","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","s","t","can","will","just","don",
    "should","now","d","ll","m","o","re","ve","y","ain","aren","couldn",
    "didn","doesn","hadn","hasn","haven","isn","ma","mightn","mustn",
    "needn","shan","shouldn","wasn","weren","won","wouldn",
    # domain-specific extras added in pipeline
    "british","airways","flight","ba","airline","would","also","get",
    "got","could","one","two","even","still","back","much","many","us",
    "like","way","however","although","though","yet","already","always",
    "never","every","make","made","take","took","come","came","go","went",
    "say","said","know","think","see","time","year","day","first","last",
}

# Minimal suffix-strip lemmatiser (handles the most common English inflections
# that spaCy lemmatises — good enough for TF-IDF bag-of-words features)
_LEMMA_RULES = [
    (r'ational$', 'ate'), (r'tional$', 'tion'), (r'enci$', 'ence'),
    (r'anci$', 'ance'), (r'izer$', 'ize'), (r'ising$', 'ise'),
    (r'izing$', 'ize'), (r'ised$', 'ise'), (r'ized$', 'ize'),
    (r'ational$', 'ate'), (r'alism$', 'al'), (r'aliti$', 'al'),
    (r'fulness$', 'ful'), (r'ousness$', 'ous'), (r'iveness$', 'ive'),
    (r'ication$', 'ic'), (r'nesses$', ''), (r'ments$', ''),
    (r'ment$', ''), (r'ings$', ''), (r'ing$', ''),
    (r'edly$', ''), (r'edly$', ''), (r'edly$', ''),
    (r'fully$', 'ful'), (r'ness$', ''), (r'tion$', 'te'),
    (r'ations$', 'ate'), (r'ation$', 'ate'), (r'ators$', 'ate'),
    (r'ator$', 'ate'), (r'alism$', 'al'), (r'ives$', 'ive'),
    (r'ness$', ''), (r'ies$', 'y'), (r'ied$', 'y'),
    (r'ers$', 'er'), (r'edly$', 'ed'), (r'edly$', 'ed'),
    (r'ives$', 'ive'), (r'ical$', 'ic'), (r'ness$', ''),
    (r'ors$', 'or'), (r'able$', ''), (r'ible$', ''),
    (r'ful$', ''), (r'less$', ''), (r'ous$', ''),
    (r'ive$', ''), (r'ize$', ''), (r'ise$', ''),
    (r'ied$', 'y'), (r'ies$', 'y'), (r'ees$', 'ee'),
    (r"'s$", ''), (r"s'$", ''), (r'ly$', ''),
    (r'ed$', ''), (r'er$', ''), (r'est$', ''),
    (r'es$', ''), (r's$', ''),
]

def _lemmatise(word: str) -> str:
    if len(word) <= 4:
        return word
    for pattern, replacement in _LEMMA_RULES:
        new = re.sub(pattern, replacement, word)
        if new != word and len(new) >= 3:
            return new
    return word


def preprocess(text: str) -> str:
    """Replicate 03_nlp_preprocessing_pipeline.py without spaCy/NLTK."""
    t = text.lower()
    t = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', t)
    t = t.translate(str.maketrans('', '', string.punctuation))
    t = re.sub(r'\s+', ' ', t).strip()
    tokens = [_lemmatise(w) for w in t.split()
              if w.isalpha() and w not in _STOPWORDS and len(w) > 2]
    return ' '.join(tokens)


# ── Recommendation engine (Phase 7) ────────────────────────────────────────
TOPIC_KEYWORDS = {
    'Flight Delays & Punctuality':   ['delay','late','hour','wait','punctual','on time'],
    'Flight Cancellations':          ['cancel','cancelled','reschedul','rebook'],
    'Baggage & Luggage':             ['bag','luggage','suitcase','lost bag','missing bag'],
    'Seat Comfort & Legroom':        ['seat','legroom','cramped','comfort','recline'],
    'Cabin Crew & Staff Behaviour':  ['crew','staff','rude','hostile','attitude'],
    'Food & Catering Quality':       ['food','meal','drink','catering','menu','snack'],
    'Check-in & Boarding Process':   ['check','board','gate','queue','boarding'],
    'Customer Service & Support':    ['customer service','support','helpline'],
    'Refunds & Compensation':        ['refund','compensat','voucher','claim','reimburse'],
    'Airport Lounge Experience':     ['lounge','airport terminal','departure area'],
}

RECOMMENDATIONS = {
    'Flight Delays & Punctuality': {
        'action':   'Proactive delay notifications via SMS/app + automatic meal vouchers',
        'priority': 'HIGH', 'owner': 'Operations & Ground Services',
        'kpi':      'On-time departure rate, rebooking speed',
    },
    'Flight Cancellations': {
        'action':   'Instant rebooking portal, same-day guarantee, full refund SLA <24h',
        'priority': 'CRITICAL', 'owner': 'Revenue Management',
        'kpi':      'Cancellation rate, refund processing time',
    },
    'Baggage & Luggage': {
        'action':   'Real-time bag tracking in app, priority delivery SLA, compensation portal',
        'priority': 'HIGH', 'owner': 'Ground Operations',
        'kpi':      'Mishandled bag rate, claim resolution time',
    },
    'Seat Comfort & Legroom': {
        'action':   'Cabin retrofit schedule, upgrade priority for loyalty members',
        'priority': 'MEDIUM', 'owner': 'Fleet & Cabin Management',
        'kpi':      'Seat satisfaction NPS, cabin retrofit %',
    },
    'Cabin Crew & Staff Behaviour': {
        'action':   'Customer experience training program + crew feedback loop',
        'priority': 'HIGH', 'owner': 'HR & Inflight Services',
        'kpi':      'Crew satisfaction score, complaint-per-flight rate',
    },
    'Food & Catering Quality': {
        'action':   'Menu refresh quarterly, dietary option expansion',
        'priority': 'MEDIUM', 'owner': 'Catering & Partnerships',
        'kpi':      'Catering satisfaction score',
    },
    'Check-in & Boarding Process': {
        'action':   'Self-bag-drop expansion, zone boarding enforcement',
        'priority': 'MEDIUM', 'owner': 'Airport Operations',
        'kpi':      'Boarding time, check-in queue length',
    },
    'Customer Service & Support': {
        'action':   'AI chatbot Tier-1 queries, <2h response SLA, staff upskilling',
        'priority': 'HIGH', 'owner': 'Customer Relations',
        'kpi':      'First-contact resolution rate, NPS',
    },
    'Refunds & Compensation': {
        'action':   'Automated EU261 compensation calculator, 7-day refund guarantee',
        'priority': 'HIGH', 'owner': 'Finance & Legal',
        'kpi':      'Refund processing time, claim rejection rate',
    },
    'Airport Lounge Experience': {
        'action':   'Capacity management, food quality audit, digital queue system',
        'priority': 'LOW', 'owner': 'Lounges & Premium Services',
        'kpi':      'Lounge satisfaction score, occupancy rate',
    },
}
DEFAULT_REC = {
    'action':   'Collect structured feedback and escalate to quality assurance',
    'priority': 'MEDIUM', 'owner': 'Quality Assurance',
    'kpi':      'Customer satisfaction score',
}


def detect_topic(text: str) -> str:
    lower = text.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(k in lower for k in kws):
            return topic
    return 'General Service Issue'


def get_recommendation(topic: str, score: float) -> dict:
    rec = RECOMMENDATIONS.get(topic, DEFAULT_REC).copy()
    if score < -0.6 and rec['priority'] == 'MEDIUM':
        rec['priority'] = 'HIGH'
    if score < -0.8 and rec['priority'] == 'HIGH':
        rec['priority'] = 'CRITICAL'
    return rec


# ── Load model once (Lambda execution context reuse) ───────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', 'phase5_best_model.pkl')

_bundle     = None
_model      = None
_vectorizer = None
_scaler     = None
_num_cols   = None
_vader      = None


def _load():
    global _bundle, _model, _vectorizer, _scaler, _num_cols, _vader
    if _model is not None:
        return
    logger.info("Cold start: loading model from %s", MODEL_PATH)
    with open(MODEL_PATH, 'rb') as f:
        _bundle = pickle.load(f)
    _model      = _bundle['model']
    _vectorizer = _bundle['vectorizer']
    _scaler     = _bundle['scaler']
    _num_cols   = _bundle['numeric_cols']
    _vader      = SentimentIntensityAnalyzer()
    logger.info("Model loaded: %s", _bundle.get('model_name', 'unknown'))


def _infer(review: str) -> dict:
    # VADER on raw text
    vs    = _vader.polarity_scores(review)
    score = vs['compound']
    label = ('Positive' if score >= 0.05 else
             'Negative' if score <= -0.05 else 'Neutral')

    # TF-IDF on preprocessed text
    cleaned = preprocess(review)
    X_text  = _vectorizer.transform([cleaned])

    # Numeric features (match training order exactly)
    clean_words = re.sub(r'[^a-z ]', '', review.lower()).split()
    feat_map = {
        'sentiment_score':   score,
        'review_length':     len(review),
        'word_count':        len(clean_words),
        'exclamation_count': review.count('!'),
        'question_count':    review.count('?'),
        'uppercase_ratio':   sum(1 for c in review if c.isupper()) / max(len(review), 1),
        'vader_neg':         vs['neg'],
        'vader_neu':         vs['neu'],
        'vader_pos':         vs['pos'],
    }
    import pandas as pd
    X_num = _scaler.transform(
        pd.DataFrame([[feat_map.get(c, 0.0) for c in _num_cols]], columns=_num_cols)
    )
    X = hstack([X_text, csr_matrix(X_num)])

    pred  = int(_model.predict(X)[0])
    prob  = float(_model.predict_proba(X)[0][pred])
    topic = detect_topic(review)
    rec   = get_recommendation(topic, score)

    return {
        'sentiment_label':        label,
        'sentiment_score':        round(score, 4),
        'predicted_satisfaction': 'Satisfied' if pred == 1 else 'Not Satisfied',
        'confidence':             round(prob, 4),
        'detected_topic':         topic,
        'recommendation': {
            'action':   rec['action'],
            'priority': rec['priority'],
            'owner':    rec['owner'],
            'kpi':      rec['kpi'],
        },
    }


# ── Lambda entrypoint ───────────────────────────────────────────────────────
def lambda_handler(event, context):
    """
    Expected input (API Gateway proxy or direct invoke):
      { "review": "The crew was rude and the flight was delayed 3 hours." }

    Returns:
      { "statusCode": 200, "body": "{...}" }
    """
    try:
        _load()

        # Support both API Gateway proxy payload and direct invoke
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event

        review = (body.get('review') or '').strip()
        if not review:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Missing required field: review'}),
            }

        result = _infer(review)
        logger.info("Prediction: %s | topic: %s", result['predicted_satisfaction'],
                    result['detected_topic'])

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps(result),
        }

    except Exception as exc:
        logger.exception("Inference error")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(exc)}),
        }
