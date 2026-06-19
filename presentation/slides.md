# ✈️ AI-Driven Predictive Customer Satisfaction & Personalized Experience Recommendations
### British Airways — Cloud AI Project Presentation

---

## Slide 1 — Title

**AI-Driven Predictive Customer Satisfaction**
**and Personalized Experience Recommendations**

*British Airways Customer Review Analysis*

| | |
|---|---|
| **Dataset** | 1,300 British Airways reviews |
| **Best Model** | XGBoost (F1 = 0.9885, AUC = 0.9972) |
| **Cloud Platform** | AWS (Lambda · API Gateway · S3 · DynamoDB · Bedrock) |
| **GenAI** | Amazon Bedrock · Claude 3 Haiku |
| **Dashboard** | Streamlit |

---

## Slide 2 — Problem Statement & Business Objective

### Problem
British Airways receives thousands of customer reviews daily. Manually analysing these to:
- Detect negative sentiment
- Identify root cause topics
- Generate personalised remediation actions

…is **slow, costly, and inconsistent**.

### Business Objective
Build an AI-powered pipeline that automatically:
1. Predicts customer satisfaction from free-text reviews
2. Identifies which service area caused dissatisfaction
3. Generates personalised, actionable recommendations
4. Logs every prediction for trend monitoring

---

## Slide 3 — AWS Architecture

```
British Airways Reviews CSV (1,300)
        │
        ▼
Amazon S3 ─────────────────────────────────────┐
(phase5_best_model.pkl)                        │
                                               ▼
User/Client ──► API Gateway ──► AWS Lambda (Python 3.12)
                                       │        │        │
                                       ▼        ▼        ▼
                               VADER NLP   XGBoost   Amazon Bedrock
                               Sentiment   Predict   (Claude 3 Haiku)
                                       │
                                       ▼
                               Amazon DynamoDB
                               (Predictions Log)
                                       │
                                       ▼
                               Streamlit Dashboard
                               (Real-time Insights)
```

**AWS Services Used:**
- 🗃️ **Amazon S3** — Model artifact storage
- ⚡ **AWS Lambda** — Serverless inference (Python 3.12, 512 MB, 60s timeout)
- 🌐 **Amazon API Gateway** — REST API with CORS
- 🗄️ **Amazon DynamoDB** — Real-time prediction logging (PAY_PER_REQUEST)
- 🤖 **Amazon Bedrock** — Generative AI recommendations (Claude 3 Haiku)
- 📊 **Streamlit** — Visualization dashboard

---

## Slide 4 — ML Pipeline

### Phase-by-Phase Pipeline

| Phase | Script | Output |
|-------|--------|--------|
| **1. Data Preprocessing** | `01_data_exploration_and_preprocessing.py` | `british_airways_reviews_cleaned.csv` |
| **2. EDA** | `02_exploratory_data_analysis.py` | 12 visualisation charts |
| **3. NLP Preprocessing** | `03_nlp_preprocessing_pipeline.py` | `cleaned_review` column (spaCy lemmatisation) |
| **4. Sentiment Analysis** | `04_sentiment_analysis.py` | VADER vs DistilBERT comparison |
| **5. Model Training** | `05_phase5_prediction.py` | `phase5_best_model.pkl` |
| **6. Topic Modelling** | `06_phase6_topic_modeling.py` | 10 LDA topics from negative reviews |
| **7. Recommendations** | `07_phase7_recommendations.py` | Rule-based + Bedrock GenAI |

---

## Slide 5 — NLP & Sentiment Analysis

### Sentiment Tools Compared

| Tool | Type | F1 Score | Speed |
|------|------|----------|-------|
| **VADER** ✅ | Lexicon-based | **0.985** | ~0.5ms/review |
| DistilBERT | Transformer | 0.663 | ~50ms/review |

**VADER Winner** — High accuracy on airline domain, real-time capable.

### NLP Pipeline
1. **Lowercasing** + URL/email removal
2. **Punctuation** stripping
3. **spaCy lemmatisation** (`en_core_web_sm`)
4. **Stopword removal** (NLTK English + domain-specific: `british`, `airways`, `flight`, `ba`...)
5. **TF-IDF vectorisation** (max 5,000 features, 1–2 ngrams)

---

## Slide 6 — Model Performance

### Model Comparison

| Model | Accuracy | F1 | ROC-AUC | CV F1 |
|-------|----------|----|---------|-------|
| Logistic Regression | 80.0% | 0.799 | 0.855 | 0.777 |
| Random Forest | 95.0% | 0.950 | 0.997 | 0.959 |
| **XGBoost ✅** | **98.9%** | **0.989** | **0.997** | **0.981** |
| LightGBM | 76.5% | 0.765 | 0.832 | 0.771 |

### Feature Engineering
- **TF-IDF text features** (5,000 terms)
- **Numeric features**: sentiment_score, review_length, word_count, exclamation_count, question_count, uppercase_ratio, vader_neg, vader_neu, vader_pos
- **Combined matrix**: `hstack([X_tfidf, X_numeric])`

---

## Slide 7 — Topic Modelling & Complaint Categories

### LDA Topic Modelling (10 Topics, Negative Reviews Only)

| Topic | Reviews | Priority |
|-------|---------|----------|
| Seat Comfort & Legroom | 218 | MEDIUM |
| Flight Delays & Punctuality | 148 | **HIGH** |
| Flight Cancellations | 134 | 🔴 **CRITICAL** |
| Cabin Crew & Staff Behaviour | 48 | **HIGH** |
| Baggage & Luggage | 23 | **HIGH** |
| Food & Catering Quality | — | MEDIUM |
| Check-in & Boarding Process | — | MEDIUM |
| Customer Service & Support | — | **HIGH** |
| Refunds & Compensation | — | **HIGH** |
| Airport Lounge Experience | — | LOW |

---

## Slide 8 — Amazon Bedrock GenAI Integration

### Generative AI Feature
- **Service**: Amazon Bedrock Runtime
- **Model**: `anthropic.claude-3-haiku-20240307-v1:0`
- **Prompt Engineering**: Review text + VADER score + XGBoost prediction + topic → personalised recommendation

### Example Input → Output

**Input Review**: *"The crew were rude and the flight was delayed 3 hours with no explanation."*

**ML Analysis**:
- Sentiment: Negative (−0.769)
- Satisfaction: Not Satisfied (100% confidence)
- Topic: Cabin Crew & Staff Behaviour

**Bedrock GenAI Output**:
> "We sincerely apologise for the unprofessional behaviour experienced on your flight.
> British Airways should implement mandatory customer experience training and a structured
> crew feedback loop to prevent recurrence. We recommend a personal apology from the
> Customer Relations team and Avios points credit as an immediate goodwill gesture."

---

## Slide 9 — AWS Infrastructure & Deployment

### Serverless Deployment (AWS SAM)

```bash
# Build & deploy
sam build
sam deploy --guided

# Upload model to S3
aws s3 cp phase5_best_model.pkl s3://ba-satisfaction-model-{account}/
```

### API Request / Response

**POST** `https://{api-id}.execute-api.us-east-1.amazonaws.com/prod/predict`

```json
{ "review": "The crew were rude and the flight was delayed 3 hours." }
```

```json
{
  "sentiment_label": "Negative",
  "sentiment_score": -0.7688,
  "predicted_satisfaction": "Not Satisfied",
  "confidence": 1.0,
  "detected_topic": "Cabin Crew & Staff Behaviour",
  "recommendation": { "action": "...", "priority": "HIGH", "owner": "HR & Inflight Services" },
  "ai_recommendation": "...(Bedrock Claude 3 Haiku)...",
  "genai_source": "bedrock"
}
```

### DynamoDB Log Entry
Every prediction auto-logged: `prediction_id`, `timestamp`, `review_snippet`, `sentiment_label`, `predicted_satisfaction`, `topic`, `priority`

---

## Slide 10 — Live Demo & Conclusion

### Streamlit Dashboard Features
- 🔍 **Review Analyser Tab** — Paste any review → instant sentiment + satisfaction + topic + recommendation
- 🏗️ **Architecture Tab** — Full AWS infrastructure diagram
- 🤖 **Amazon Bedrock GenAI** — Personalised AI insight per review (toggle on/off)
- ☁️ **AWS Status Panel** — Live Lambda / Bedrock / DynamoDB status

### Key Achievements
✅ **98.9% accuracy** XGBoost model (F1 = 0.9885, AUC = 0.9972)
✅ **10 complaint topics** identified via LDA from negative reviews
✅ **Amazon Bedrock Claude 3 Haiku** — LLM-generated personalised recommendations
✅ **Serverless API** — AWS Lambda + API Gateway (CORS-enabled)
✅ **DynamoDB logging** — every prediction persisted in real-time
✅ **S3 model storage** — versioned artifact storage
✅ **SAM template** — one-command deployment of full infrastructure

### Business Impact
- Reduces manual review analysis time by **>90%**
- Enables **real-time** customer satisfaction monitoring
- Provides **prioritised, actionable** recommendations to operational teams
- Scales to **millions of reviews** at AWS serverless pricing
