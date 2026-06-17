# ✈️ AI-Driven Customer Satisfaction Predictor

> **Predict British Airways customer satisfaction from review text — with sentiment analysis, topic modelling, personalised recommendations, a Streamlit dashboard, and an AWS Lambda API.**

---

## 📁 Project Structure

```
.
├── 01_data_exploration_and_preprocessing.py   # Data cleaning, feature engineering, TF-IDF, train/test split
├── 02_exploratory_data_analysis.py            # 12 EDA visualisations
├── 03_nlp_preprocessing_pipeline.py           # spaCy/NLTK tokenisation, lemmatisation → cleaned_review
├── 04_sentiment_analysis.py                   # VADER vs DistilBERT comparison
├── 05_model_training.py                       # LR / RF / XGBoost / LightGBM baseline
├── 05_phase5_prediction.py                    # RF vs XGBoost on enriched features → phase5_best_model.pkl
├── 06_phase6_topic_modeling.py                # LDA topic modelling on negative reviews
├── 07_phase7_recommendations.py               # Rule-based recommendation engine
├── app.py                                     # Streamlit dashboard
│
├── notebooks/                                 # Jupyter (.ipynb) versions of every script
├── data/                                      # Raw + processed CSV datasets (gitignored)
├── outputs/
│   ├── eda/                                   # 12 EDA charts
│   ├── sentiment/                             # 5 VADER vs DistilBERT charts
│   ├── models/                                # Model comparison charts + CSVs
│   └── topics/                                # Topic table + recommendation CSV
└── deployment/
    ├── lambda_function.py                     # AWS Lambda handler (no spaCy/NLTK)
    └── requirements.txt                       # Minimal Lambda dependencies
```

---

## 🧠 ML Pipeline

```
Raw Reviews (1,300)
     │
     ▼
01 · Data Preprocessing     → british_airways_reviews_cleaned.csv
     │
     ▼
02 · EDA                    → 12 charts (eda_*.png)
     │
     ▼
03 · NLP Preprocessing      → cleaned_review column (spaCy lemmatisation)
     │
     ▼
04 · Sentiment Analysis     → VADER (winner, F1=0.985) vs DistilBERT (F1=0.663)
     │
     ▼
05 · Model Training         → XGBoost best (F1=0.9885, AUC=0.9972)
     │                        phase5_best_model.pkl
     ▼
06 · Topic Modelling        → LDA, 10 topics on negative reviews
     │
     ▼
07 · Recommendations        → Rule-based engine (topic + sentiment → action)
```

---

## 📊 Model Performance

| Model | Accuracy | F1 | ROC-AUC | CV F1 |
|-------|----------|----|---------|-------|
| Logistic Regression | 80.0% | 0.799 | 0.855 | 0.777 |
| Random Forest | 95.0% | 0.950 | 0.997 | 0.959 |
| **XGBoost** ✅ | **98.9%** | **0.989** | **0.997** | **0.981** |
| LightGBM | 76.5% | 0.765 | 0.832 | 0.771 |

---

## 🖥️ Streamlit Dashboard

```bash
streamlit run app.py
```

**Features:**
- Input any review text
- Live VADER sentiment score + label
- XGBoost satisfaction prediction with confidence
- Keyword-based topic detection
- Personalised service improvement recommendation

---

## ☁️ AWS Lambda Deployment

```bash
# Build Linux-compatible package
pip install -r deployment/requirements.txt -t ./package \
  --platform manylinux2014_x86_64 --only-binary=:all:

# Bundle
cp deployment/lambda_function.py phase5_best_model.pkl ./package/
cd package && zip -r9 ../lambda.zip . && cd ..

# Deploy
aws lambda update-function-code \
  --function-name customer-satisfaction-predictor \
  --zip-file fileb://lambda.zip
```

**Lambda config:** Python 3.12 · 512 MB · 30s timeout · ~80 MB zip

**Request:**
```json
{ "review": "The crew were rude and the flight was delayed 3 hours." }
```

**Response:**
```json
{
  "sentiment_label": "Negative",
  "sentiment_score": -0.7688,
  "predicted_satisfaction": "Not Satisfied",
  "confidence": 1.0,
  "detected_topic": "Flight Delays & Punctuality",
  "recommendation": {
    "action": "Proactive delay notifications via SMS/app + automatic meal vouchers",
    "priority": "HIGH",
    "owner": "Operations & Ground Services",
    "kpi": "On-time departure rate, rebooking speed"
  }
}
```

---

## ⚙️ Setup

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm \
            nltk spacy vaderSentiment streamlit scipy

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download(['stopwords','wordnet','omw-1.4'])"

# Run full pipeline (in order)
python 01_data_exploration_and_preprocessing.py
python 02_exploratory_data_analysis.py
python 03_nlp_preprocessing_pipeline.py
python 04_sentiment_analysis.py
python 05_phase5_prediction.py
python 06_phase6_topic_modeling.py
python 07_phase7_recommendations.py
```

---

## 🔍 Topic Categories (Negative Reviews)

| Topic | Reviews | Priority |
|-------|---------|----------|
| Flight Delays & Punctuality | 148 | HIGH |
| Flight Cancellations | 134 | CRITICAL |
| Seat Comfort & Legroom | 218 | MEDIUM |
| Cabin Crew & Staff Behaviour | 48 | HIGH |
| Baggage & Luggage | 23 | HIGH |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| NLP | NLTK · spaCy · VADER · DistilBERT |
| ML | scikit-learn · XGBoost · LightGBM |
| Topic Modelling | LDA (sklearn) |
| Dashboard | Streamlit |
| Deployment | AWS Lambda · API Gateway |
| Data | pandas · numpy · scipy |

---

## 📄 Dataset

British Airways customer reviews — 1,300 rows · columns: `title`, `reviews`  
Source: Publicly available airline review dataset.
