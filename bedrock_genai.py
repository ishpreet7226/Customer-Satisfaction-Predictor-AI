"""
bedrock_genai.py — Amazon Bedrock GenAI Integration
Generates personalised, AI-driven service improvement recommendations
using Claude 3 Haiku via Amazon Bedrock Runtime.

Falls back to a rich rule-based response if Bedrock is not configured.
"""
import json
import logging

logger = logging.getLogger(__name__)

# ── Bedrock client (lazy-loaded once) ────────────────────────────────────────
_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client
    try:
        import boto3
        _bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',  # change to your region
        )
        return _bedrock_client
    except Exception as e:
        logger.warning("Bedrock client init failed: %s", e)
        return None


# ── Model ID ─────────────────────────────────────────────────────────────────
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert airline customer experience analyst working for British Airways.
Your role is to analyse customer feedback and produce concise, personalised, data-driven recommendations
to improve the passenger experience. Always be specific, empathetic, and actionable."""

USER_PROMPT_TEMPLATE = """A passenger has submitted the following review:

REVIEW: "{review}"

ANALYSIS RESULTS:
- Sentiment: {sentiment_label} (VADER compound score: {sentiment_score:+.4f})
- Predicted Satisfaction: {satisfaction_label}
- Primary Issue Category: {topic}
- Model Confidence: {confidence:.1f}%

Based on this specific review and analysis, generate a personalised response that includes:
1. A brief empathetic acknowledgement of the passenger's experience (1 sentence)
2. A specific, actionable recommendation for British Airways to resolve this type of issue (2–3 sentences)
3. A suggested immediate compensation or gesture of goodwill for this passenger (1 sentence)

Keep the total response under 120 words. Be specific to the detected topic and sentiment intensity."""


def generate_ai_recommendation(
    review: str,
    sentiment_label: str,
    sentiment_score: float,
    satisfaction_label: str,
    topic: str,
    confidence: float,
) -> dict:
    """
    Calls Amazon Bedrock (Claude 3 Haiku) to generate a personalised AI recommendation.

    Returns:
        dict with keys:
            - 'ai_recommendation' (str): the generated text
            - 'source' (str): 'bedrock' | 'fallback'
            - 'model' (str): model ID used
            - 'input_tokens' (int): tokens consumed (0 if fallback)
            - 'output_tokens' (int): tokens generated (0 if fallback)
    """
    client = _get_bedrock_client()
    if client is None:
        return _fallback_recommendation(topic, sentiment_score, satisfaction_label)

    user_message = USER_PROMPT_TEMPLATE.format(
        review=review[:1000],          # truncate very long reviews
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        satisfaction_label=satisfaction_label,
        topic=topic,
        confidence=confidence * 100,
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.4,
        "top_p": 0.9,
    })

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        ai_text = result["content"][0]["text"].strip()
        usage   = result.get("usage", {})

        return {
            "ai_recommendation": ai_text,
            "source":            "bedrock",
            "model":             MODEL_ID,
            "input_tokens":      usage.get("input_tokens", 0),
            "output_tokens":     usage.get("output_tokens", 0),
        }

    except client.exceptions.AccessDeniedException:
        logger.warning("Bedrock: access denied — check IAM permissions for bedrock:InvokeModel")
        return _fallback_recommendation(topic, sentiment_score, satisfaction_label)
    except client.exceptions.ModelNotReadyException:
        logger.warning("Bedrock: model not ready")
        return _fallback_recommendation(topic, sentiment_score, satisfaction_label)
    except Exception as e:
        logger.exception("Bedrock invocation error: %s", e)
        return _fallback_recommendation(topic, sentiment_score, satisfaction_label)


# ── Rule-based fallback ───────────────────────────────────────────────────────
_FALLBACK_TEXTS = {
    'Flight Delays & Punctuality': (
        "We sincerely apologise for the disruption caused by this delay. "
        "British Airways should implement proactive real-time SMS/app notifications "
        "the moment a delay is detected, and automatically issue digital meal vouchers "
        "for waits exceeding 2 hours. We recommend offering this passenger an Avios "
        "points credit as an immediate gesture of goodwill."
    ),
    'Flight Cancellations': (
        "We understand how frustrating an unexpected cancellation is. "
        "British Airways must expedite its self-service rebooking portal and "
        "guarantee same-day alternatives or a full refund within 24 hours, "
        "in line with EU261 regulation. An immediate hotel accommodation voucher "
        "and travel credit should be offered to this passenger."
    ),
    'Baggage & Luggage': (
        "We apologise for the baggage handling failure experienced. "
        "British Airways should deploy real-time bag tracking via the BA app "
        "and establish a guaranteed 30-minute delivery SLA from aircraft to carousel. "
        "This passenger deserves an immediate compensation credit to their account."
    ),
    'Seat Comfort & Legroom': (
        "Passenger comfort is fundamental to the British Airways experience. "
        "The cabin retrofit programme should be accelerated for Economy class "
        "on long-haul routes, with Executive Club members offered complimentary "
        "upgrade priority. A seat upgrade voucher for a future flight is recommended."
    ),
    'Cabin Crew & Staff Behaviour': (
        "Every passenger deserves to be treated with warmth and professionalism. "
        "British Airways should reinforce its customer experience training programme "
        "and introduce a peer-recognition scheme for outstanding cabin crew. "
        "A personal apology from the Customer Relations team is recommended for this passenger."
    ),
    'Food & Catering Quality': (
        "Quality catering is an important part of the passenger experience. "
        "British Airways should revamp its menu quarterly with chef partnerships "
        "and expand dietary options across all cabin classes. "
        "A complimentary upgrade to Business Class catering on the next flight is suggested."
    ),
    'Check-in & Boarding Process': (
        "A smooth boarding experience sets the tone for the entire journey. "
        "British Airways should expand self-service bag drop, enforce zone boarding strictly, "
        "and provide dedicated fast-track lanes for families and accessibility needs. "
        "A Fast Track security voucher for this passenger's next trip is recommended."
    ),
    'Customer Service & Support': (
        "Timely and effective support is a core customer promise. "
        "British Airways should deploy an AI-powered chatbot for Tier-1 queries "
        "and enforce a 2-hour response SLA across all digital channels. "
        "A dedicated customer relations agent should follow up personally with this passenger."
    ),
    'Refunds & Compensation': (
        "Passengers deserve transparent and swift resolution of financial claims. "
        "British Airways should launch an automated EU261 compensation calculator "
        "and commit to a 7-day refund guarantee with real-time status tracking. "
        "This passenger's refund should be prioritised and processed within 48 hours."
    ),
    'Airport Lounge Experience': (
        "The lounge is a key differentiator for premium passengers. "
        "British Airways should implement a digital capacity management system, "
        "refresh food offerings with seasonal menus, and introduce quiet zones. "
        "A complimentary Lounge Day Pass for a future visit is recommended."
    ),
}

_DEFAULT_FALLBACK = (
    "Thank you for sharing this feedback — every review helps British Airways improve. "
    "Our quality assurance team will review this issue and determine the appropriate "
    "service improvement action. We recommend a personal follow-up from our "
    "customer relations team within 48 hours."
)


def _fallback_recommendation(topic: str, sentiment_score: float, satisfaction_label: str) -> dict:
    text = _FALLBACK_TEXTS.get(topic, _DEFAULT_FALLBACK)

    # Append urgency note for very negative reviews
    if sentiment_score < -0.7 and satisfaction_label == 'Not Satisfied':
        text += (
            " Given the strong negative sentiment, this case should be escalated "
            "to the Senior Customer Relations team immediately."
        )

    return {
        "ai_recommendation": text,
        "source":            "fallback",
        "model":             "rule-based",
        "input_tokens":      0,
        "output_tokens":     0,
    }
