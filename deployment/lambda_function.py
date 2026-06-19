"""
lambda_function.py — AWS Lambda handler
Integrates:
  - VADER sentiment analysis
  - XGBoost satisfaction prediction (model from S3 or local path)
  - Keyword-based topic detection
  - Amazon Bedrock (Claude 3 Haiku) for GenAI recommendations
  - Amazon DynamoDB for prediction logging

Lightweight: no spaCy, no NLTK downloads at runtime.
Preprocessing mirrors the training pipeline (03_nlp_preprocessing_pipeline.py)
using a static stopword list + simple regex lemmatisation stubs.
"""
import os, re, string, json, pickle, logging, uuid, datetime
import numpy as np
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Config from environment variables ──────────────────────────────────────────
MODEL_PATH   = os.environ.get('MODEL_PATH', 'phase5_best_model.pkl')
S3_BUCKET    = os.environ.get('S3_BUCKET', '')          # e.g. 'ba-satisfaction-model'
S3_KEY       = os.environ.get('S3_KEY', 'phase5_best_model.pkl')
DYNAMO_TABLE = os.environ.get('DYNAMO_TABLE', 'customer-satisfaction-predictions')
BEDROCK_MODEL = os.environ.get('BEDROCK_MODEL', 'anthropic.claude-3-haiku-20240307-v1:0')
AWS_REGION   = os.environ.get('AWS_REGION', 'us-east-1')

# ── Stopwords (NLTK english list, frozen at training time) ─────────────────────
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
    "british","airways","flight","ba","airline","would","also","get",
    "got","could","one","two","even","still","back","much","many","us",
    "like","way","however","although","though","yet","already","always",
    "never","every","make","made","take","took","come","came","go","went",
    "say","said","know","think","see","time","year","day","first","last",
}

_LEMMA_RULES = [
    (r'ational$', 'ate'), (r'tional$', 'tion'), (r'enci$', 'ence'),
    (r'anci$', 'ance'), (r'izer$', 'ize'), (r'ising$', 'ise'),
    (r'izing$', 'ize'), (r'ised$', 'ise'), (r'ized$', 'ize'),
    (r'alism$', 'al'), (r'aliti$', 'al'), (r'fulness$', 'ful'),
    (r'ousness$', 'ous'), (r'iveness$', 'ive'), (r'ication$', 'ic'),
    (r'nesses$', ''), (r'ments$', ''), (r'ment$', ''),
    (r'ings$', ''), (r'ing$', ''), (r'fully$', 'ful'),
    (r'ness$', ''), (r'ation$', 'ate'), (r'ations$', 'ate'),
    (r'ator$', 'ate'), (r'ives$', 'ive'), (r'ies$', 'y'),
    (r'ied$', 'y'), (r'ers$', 'er'), (r'ical$', 'ic'),
    (r'ors$', 'or'), (r'able$', ''), (r'ible$', ''),
    (r'ful$', ''), (r'less$', ''), (r'ous$', ''),
    (r'ive$', ''), (r'ize$', ''), (r'ise$', ''),
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


# ── Topic / Recommendation engine (Phase 6 + 7) ──────────────────────────────
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


# ── AWS clients (lazy-loaded, reused across warm invocations) ─────────────────
_bundle     = None
_model      = None
_vectorizer = None
_scaler     = None
_num_cols   = None
_vader      = None
_dynamo_table = None
_bedrock_client = None


def _load_model():
    """Load XGBoost bundle from S3 (preferred) or local path."""
    global _bundle, _model, _vectorizer, _scaler, _num_cols, _vader
    if _model is not None:
        return
    import boto3
    # Try S3 first
    if S3_BUCKET:
        try:
            logger.info("Cold start: downloading model from s3://%s/%s", S3_BUCKET, S3_KEY)
            s3 = boto3.client('s3', region_name=AWS_REGION)
            tmp_path = '/tmp/model.pkl'
            s3.download_file(S3_BUCKET, S3_KEY, tmp_path)
            with open(tmp_path, 'rb') as f:
                _bundle = pickle.load(f)
            logger.info("Model loaded from S3")
        except Exception as e:
            logger.warning("S3 download failed (%s), falling back to local path", e)

    if _bundle is None:
        logger.info("Cold start: loading model from local path %s", MODEL_PATH)
        with open(MODEL_PATH, 'rb') as f:
            _bundle = pickle.load(f)

    _model      = _bundle['model']
    _vectorizer = _bundle['vectorizer']
    _scaler     = _bundle['scaler']
    _num_cols   = _bundle['numeric_cols']
    _vader      = SentimentIntensityAnalyzer()
    logger.info("Model ready: %s", _bundle.get('model_name', 'unknown'))


def _get_dynamo():
    """Return DynamoDB table (None if unavailable)."""
    global _dynamo_table
    if _dynamo_table is not None:
        return _dynamo_table
    try:
        import boto3
        ddb = boto3.resource('dynamodb', region_name=AWS_REGION)
        _dynamo_table = ddb.Table(DYNAMO_TABLE)
        return _dynamo_table
    except Exception as e:
        logger.warning("DynamoDB init failed: %s", e)
        return None


def _get_bedrock():
    """Return Bedrock Runtime client (None if unavailable)."""
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client
    try:
        import boto3
        _bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        return _bedrock_client
    except Exception as e:
        logger.warning("Bedrock init failed: %s", e)
        return None


# ── Amazon Bedrock GenAI recommendation ───────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are an expert airline customer experience analyst for British Airways. "
    "Produce concise, empathetic, actionable service improvement recommendations."
)

def _bedrock_recommendation(review: str, label: str, score: float,
                              sat: str, topic: str, conf: float) -> str:
    client = _get_bedrock()
    if client is None:
        return None
    user_msg = (
        f"Review: \"{review[:800]}\"\n"
        f"Sentiment: {label} (score {score:+.3f}) | Satisfaction: {sat} | "
        f"Topic: {topic} | Confidence: {conf:.1%}\n\n"
        "Generate: 1) empathetic acknowledgement, 2) specific BA service action, "
        "3) immediate goodwill gesture. Max 100 words."
    )
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 250,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
        "temperature": 0.4,
    })
    try:
        resp   = client.invoke_model(modelId=BEDROCK_MODEL,
                                     contentType="application/json",
                                     accept="application/json", body=body)
        result = json.loads(resp["body"].read())
        return result["content"][0]["text"].strip()
    except Exception as e:
        logger.warning("Bedrock inference error: %s", e)
        return None


# ── Core inference ─────────────────────────────────────────────────────────────
def _infer(review: str, include_genai: bool = True) -> dict:
    # VADER
    vs    = _vader.polarity_scores(review)
    score = vs['compound']
    label = ('Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral')

    # TF-IDF
    cleaned = preprocess(review)
    X_text  = _vectorizer.transform([cleaned])

    # Numeric features
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
    sat   = 'Satisfied' if pred == 1 else 'Not Satisfied'

    result = {
        'sentiment_label':        label,
        'sentiment_score':        round(score, 4),
        'predicted_satisfaction': sat,
        'confidence':             round(prob, 4),
        'detected_topic':         topic,
        'recommendation': {
            'action':   rec['action'],
            'priority': rec['priority'],
            'owner':    rec['owner'],
            'kpi':      rec['kpi'],
        },
    }

    # GenAI via Bedrock
    if include_genai:
        ai_text = _bedrock_recommendation(review, label, score, sat, topic, prob)
        result['ai_recommendation'] = ai_text or (
            "Intelligent fallback: " + rec['action'] +
            " (Configure Bedrock for LLM-generated insights.)"
        )
        result['genai_source'] = 'bedrock' if ai_text else 'fallback'

    return result


def _log_to_dynamo(review: str, result: dict):
    """Log prediction to DynamoDB (best-effort, never blocks response)."""
    table = _get_dynamo()
    if table is None:
        return
    try:
        table.put_item(Item={
            'prediction_id':          str(uuid.uuid4()),
            'timestamp':              datetime.datetime.utcnow().isoformat(),
            'review_snippet':         review[:200],
            'sentiment_label':        result['sentiment_label'],
            'sentiment_score':        str(result['sentiment_score']),
            'predicted_satisfaction': result['predicted_satisfaction'],
            'confidence':             str(result['confidence']),
            'detected_topic':         result['detected_topic'],
            'priority':               result['recommendation']['priority'],
        })
        logger.info("Logged to DynamoDB: %s", result['detected_topic'])
    except Exception as e:
        logger.warning("DynamoDB put_item failed: %s", e)


# ── Lambda entrypoint ──────────────────────────────────────────────────────────
def lambda_handler(event, context):
    """
    Supported event shapes:
      API Gateway proxy: { "body": "{\"review\": \"...\"}" }
      Direct invoke:     { "review": "..." }

    Optional flags:
      "include_genai": true   (default) — include Bedrock AI recommendation
    """
    try:
        _load_model()

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

        include_genai = body.get('include_genai', True)
        result = _infer(review, include_genai=include_genai)

        # Log asynchronously (best-effort)
        _log_to_dynamo(review, result)

        logger.info("Prediction: %s | topic: %s | genai_source: %s",
                    result['predicted_satisfaction'],
                    result['detected_topic'],
                    result.get('genai_source', 'n/a'))

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
