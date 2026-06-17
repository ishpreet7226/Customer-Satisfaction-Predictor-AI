"""
Phase 7 — Rule-Based Recommendation Engine
Input: topic + sentiment_score + predicted_satisfaction
Output: service improvement recommendations
"""
import json
import pandas as pd

# ── Recommendation mapping ────────────────────────────────────────────────
RECOMMENDATIONS = {
    'Flight Delays & Punctuality': {
        'action':   'Proactive delay notifications via SMS/app + automatic meal vouchers',
        'priority': 'HIGH',
        'kpi':      'On-time departure rate, passenger rebooking speed',
        'owner':    'Operations & Ground Services',
    },
    'Flight Cancellations': {
        'action':   'Instant rebooking portal, same-day guarantee, full refund SLA <24h',
        'priority': 'CRITICAL',
        'kpi':      'Cancellation rate, refund processing time',
        'owner':    'Revenue Management',
    },
    'Baggage & Luggage': {
        'action':   'Real-time bag tracking in app, priority delivery SLA, compensation portal',
        'priority': 'HIGH',
        'kpi':      'Mishandled bag rate, claim resolution time',
        'owner':    'Ground Operations',
    },
    'Seat Comfort': {
        'action':   'Cabin retrofit schedule, upgrade priority for loyalty members, advance seat selection',
        'priority': 'MEDIUM',
        'kpi':      'Seat satisfaction NPS, cabin retrofit completion %',
        'owner':    'Fleet & Cabin Management',
    },
    'Cabin Crew & Staff Behaviour': {
        'action':   'Customer experience training, crew feedback loop, recognition program',
        'priority': 'HIGH',
        'kpi':      'Crew satisfaction score, complaint-per-flight rate',
        'owner':    'HR & Inflight Services',
    },
    'Food & Catering Quality': {
        'action':   'Menu refresh quarterly, dietary option expansion, hot meal on short-haul',
        'priority': 'MEDIUM',
        'kpi':      'Catering satisfaction score, waste reduction %',
        'owner':    'Catering & Partnerships',
    },
    'Check-in & Boarding Process': {
        'action':   'Self-bag-drop expansion, zone boarding enforcement, dedicated fast-track lanes',
        'priority': 'MEDIUM',
        'kpi':      'Boarding time, check-in queue length',
        'owner':    'Airport Operations',
    },
    'Customer Service & Support': {
        'action':   'AI chatbot for Tier-1 queries, <2h response SLA, staff upskilling',
        'priority': 'HIGH',
        'kpi':      'First-contact resolution rate, NPS, response time',
        'owner':    'Customer Relations',
    },
    'Refunds & Compensation': {
        'action':   'Automated EU261 compensation calculator, 7-day refund guarantee',
        'priority': 'HIGH',
        'kpi':      'Refund processing time, claim rejection rate',
        'owner':    'Finance & Legal',
    },
    'Airport Lounge Experience': {
        'action':   'Capacity management, food quality audit, digital queue system',
        'priority': 'LOW',
        'kpi':      'Lounge satisfaction score, occupancy rate',
        'owner':    'Lounges & Premium Services',
    },
}

# Default fallback
DEFAULT_REC = {
    'action':   'Collect structured feedback and escalate to relevant team',
    'priority': 'MEDIUM',
    'kpi':      'Customer satisfaction score',
    'owner':    'Quality Assurance',
}

PRIORITY_ORDER = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}


def get_recommendation(topic: str, sentiment_score: float,
                        predicted_satisfaction: int) -> dict:
    """
    Returns a recommendation dict for a given topic + model outputs.

    Args:
        topic: Business-friendly topic name (from Phase 6).
        sentiment_score: VADER compound score [-1, 1].
        predicted_satisfaction: 0 (not satisfied) or 1 (satisfied).
    """
    rec = RECOMMENDATIONS.get(topic, DEFAULT_REC).copy()
    rec['topic']                 = topic
    rec['sentiment_score']       = round(sentiment_score, 4)
    rec['predicted_satisfaction'] = 'Satisfied' if predicted_satisfaction else 'Not Satisfied'

    # Escalate priority for strongly negative sentiment
    if sentiment_score < -0.6 and rec['priority'] == 'MEDIUM':
        rec['priority'] = 'HIGH'
    if sentiment_score < -0.8 and rec['priority'] == 'HIGH':
        rec['priority'] = 'CRITICAL'

    return rec


def batch_recommend(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised recommendation over a DataFrame.
    Expects columns: topic, sentiment_score, predicted_satisfaction
    """
    recs = df_input.apply(
        lambda r: get_recommendation(
            r['topic'], r['sentiment_score'], r['predicted_satisfaction']),
        axis=1
    )
    return pd.DataFrame(list(recs))


# ── Demo: apply to negative reviews from Phase 6 ─────────────────────────
if __name__ == '__main__':
    try:
        topics_df = pd.read_csv('phase6_topics.csv')
        sent_df   = pd.read_csv('british_airways_sentiment.csv')
        neg_df    = sent_df[sent_df['sentiment_label'] == 'Negative'].copy()

        # Assign topics round-robin from topic table for demonstration
        topic_list = topics_df['Business_Name'].tolist()
        neg_df = neg_df.reset_index(drop=True)
        neg_df['topic'] = [topic_list[i % len(topic_list)] for i in range(len(neg_df))]
        neg_df['predicted_satisfaction'] = 0  # negatives are not satisfied

        recs = batch_recommend(neg_df[['topic', 'sentiment_score',
                                       'predicted_satisfaction']])
        recs = recs.sort_values(
            'priority',
            key=lambda s: s.map(lambda x: PRIORITY_ORDER.get(x, 9))
        )
        recs.to_csv('phase7_recommendations.csv', index=False)

        print("\n=== SAMPLE RECOMMENDATIONS ===")
        print(recs[['topic', 'priority', 'action', 'owner']].drop_duplicates(
            subset='topic').to_string(index=False))
        print("\nFiles: phase7_recommendations.csv")

    except FileNotFoundError as e:
        print(f"Dependency missing: {e}")
        print("Run Phase 6 first, then re-run this script.")

    # Single-call example
    print("\n=== SINGLE EXAMPLE ===")
    ex = get_recommendation('Flight Delays & Punctuality',
                             sentiment_score=-0.87,
                             predicted_satisfaction=0)
    print(json.dumps(ex, indent=2))
