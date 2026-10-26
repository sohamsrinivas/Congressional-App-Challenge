"""
TicketIQ pipeline.

Turns a raw support ticket into: category, priority, sentiment, emotion,
topic, churn risk, a business risk score, and concrete next steps.

Every trained artifact is optional. If a .pkl file is missing the pipeline
falls back to a transparent rule-based method and reports which path it used,
so the app always runs and never silently lies about what produced a number.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:  # pragma: no cover
    HAS_TEXTBLOB = False


ARTIFACT_DIR = os.environ.get("TICKETIQ_ARTIFACTS", "artifacts")

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _load(name: str) -> Any | None:
    """Load a joblib artifact if it exists, otherwise return None."""
    if joblib is None:
        return None
    path = os.path.join(ARTIFACT_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_artifacts() -> dict[str, Any]:
    """Load whatever trained models are present. Missing ones become None."""
    art = {
        "ticket_model": _load("ticket_classifier_model.pkl"),
        "ticket_vectorizer": _load("tfidf_vectorizer.pkl"),
        "business_mapping": _load("business_mapping.pkl"),
        "topic_model": _load("bertopic_model"),
        "topic_labels": _load("business_topic_labels.pkl"),
        "churn_model": _load("churn_model.pkl"),
        "scaler": _load("scaler.pkl"),
        "feature_columns": _load("feature_columns.pkl"),
    }
    art["status"] = {
        "Ticket classifier": art["ticket_model"] is not None and art["ticket_vectorizer"] is not None,
        "Topic model (BERTopic)": art["topic_model"] is not None,
        "Churn model": art["churn_model"] is not None and art["scaler"] is not None,
        "Sentiment (TextBlob)": HAS_TEXTBLOB,
    }
    return art


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _has_any(text: str, stems: list[str]) -> bool:
    """Stem matching: 'frustrat' catches frustrated, frustrating, frustration."""
    return any(stem in text for stem in stems)


def _clean_label(label: Any) -> str:
    """Strip underscores and stray separators out of auto-generated labels."""
    s = str(label).replace("_", " ")
    s = re.sub(r"\s*/\s*", " / ", s).strip(" /")
    s = re.sub(r"\s+", " ", s)
    return s.title() if s else ""


# ---------------------------------------------------------------------------
# Stage 1: category and priority
# ---------------------------------------------------------------------------

CATEGORY_RULES = {
    "Technical Support": ["login", "log in", "password", "crash", "error", "bug", "not working", "broken", "can't access", "cannot access", "unable", "loading", "slow", "freeze"],
    "Billing": ["charge", "bill", "invoice", "refund", "payment", "card", "subscription", "price", "overcharg", "double"],
    "Account Management": ["account", "profile", "email change", "delete my", "close my", "username", "settings"],
    "Shipping & Delivery": ["ship", "deliver", "package", "tracking", "arrive", "late", "lost"],
    "Feature Request": ["feature", "would be nice", "suggest", "wish", "could you add", "request"],
    "General Inquiry": ["how do i", "question", "wondering", "information", "help with"],
}

PRIORITY_HIGH = ["urgent", "critical", "immediately", "asap", "emergency", "outage", "down", "cannot access", "can't access", "unable", "locked out", "security", "hacked", "fraud", "lawsuit", "cancel", "refund"]
PRIORITY_MEDIUM = ["error", "failed", "problem", "issue", "not working", "broken", "charge", "wrong", "delay", "late", "disappointed"]


def predict_ticket_category(text: str, art: dict) -> dict:
    lower = text.lower()
    source = "rules"
    category = None

    if art.get("ticket_model") is not None and art.get("ticket_vectorizer") is not None:
        try:
            vec = art["ticket_vectorizer"].transform([text])
            raw = art["ticket_model"].predict(vec)[0]
            mapping = art.get("business_mapping")
            category = mapping.get(raw, str(raw)) if isinstance(mapping, dict) else str(raw)
            category = _clean_label(category)
            source = "trained model"
        except Exception:
            category = None

    if not category:
        scores = {cat: sum(k in lower for k in kws) for cat, kws in CATEGORY_RULES.items()}
        best = max(scores, key=scores.get)
        category = best if scores[best] > 0 else "General Inquiry"

    # Normalize "Technical Support Team" style labels so routing text reads cleanly.
    category = re.sub(r"\s+team$", "", category, flags=re.IGNORECASE)

    if _has_any(lower, PRIORITY_HIGH):
        priority = "High"
    elif _has_any(lower, PRIORITY_MEDIUM):
        priority = "Medium"
    else:
        priority = "Low"

    return {"category": category, "priority": priority, "category_source": source}


# ---------------------------------------------------------------------------
# Stage 2: sentiment and emotion
# ---------------------------------------------------------------------------

NEG_WORDS = ["terrible", "awful", "worst", "bad", "horrible", "useless", "frustrat", "angry", "furious", "disappoint", "unacceptable", "ridiculous", "broken", "never", "hate", "slow", "wrong", "annoy", "upset"]
POS_WORDS = ["great", "excellent", "amazing", "love", "thank", "awesome", "perfect", "helpful", "good", "fantastic", "appreciate", "quick", "wonderful"]

EMOTION_RULES = [
    ("Angry", ["angry", "furious", "frustrat", "terrible", "worst", "ridiculous", "unacceptable", "outrag", "sick of", "fed up", "livid"]),
    ("Anxious", ["worried", "concern", "scared", "afraid", "nervous", "urgent", "asap", "deadline", "locked out", "hacked", "fraud"]),
    ("Sad", ["sad", "disappoint", "unhappy", "let down", "upset", "miss"]),
    ("Happy", ["happy", "great", "excellent", "amazing", "love", "thank", "awesome", "wonderful", "appreciate"]),
    ("Confused", ["confus", "don't understand", "unclear", "how do i", "not sure", "what does"]),
]


def predict_sentiment_emotion(text: str) -> dict:
    lower = text.lower()

    if HAS_TEXTBLOB:
        polarity = float(TextBlob(text).sentiment.polarity)
        source = "TextBlob"
    else:
        neg = sum(w in lower for w in NEG_WORDS)
        pos = sum(w in lower for w in POS_WORDS)
        polarity = (pos - neg) / max(pos + neg, 1) * 0.8
        source = "lexicon"

    # Lexicon nudge: TextBlob under-reads short, blunt complaints.
    if _has_any(lower, ["frustrat", "unacceptable", "worst", "furious"]) and polarity > -0.3:
        polarity = min(polarity, -0.3)

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    emotion = "Neutral"
    for label, stems in EMOTION_RULES:
        if _has_any(lower, stems):
            emotion = label
            break

    return {
        "sentiment": sentiment,
        "sentiment_score": round(polarity, 4),
        "emotion": emotion,
        "sentiment_source": source,
    }


# ---------------------------------------------------------------------------
# Stage 3: topic
# ---------------------------------------------------------------------------

TOPIC_RULES = {
    "Login & Access": ["login", "log in", "password", "sign in", "locked out", "2fa", "verification", "access"],
    "Billing & Refunds": ["refund", "charge", "bill", "invoice", "payment", "subscription", "overcharg"],
    "App Performance": ["crash", "slow", "freeze", "loading", "lag", "bug", "glitch"],
    "Delivery & Shipping": ["ship", "deliver", "package", "tracking", "arrive"],
    "Account Changes": ["account", "profile", "email", "username", "delete", "close"],
    "Product Feedback": ["feature", "suggest", "wish", "would be nice", "feedback"],
}

GENERIC_TOPIC_TOKENS = {"discussion", "topic", "unknown", "misc", "other", ""}


def _rule_topic(lower: str) -> tuple[str, float]:
    scores = {t: sum(k in lower for k in kws) for t, kws in TOPIC_RULES.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "General", 0.0
    return best, round(min(0.5 + 0.15 * scores[best], 0.95), 4)


def predict_topic(text: str, art: dict) -> dict:
    lower = text.lower()
    fallback_label, fallback_conf = _rule_topic(lower)

    if art.get("topic_model") is not None:
        try:
            topics, probs = art["topic_model"].transform([text])
            topic_id = int(topics[0])
            conf = float(np.max(probs[0])) if probs is not None else 0.0
            labels = art.get("topic_labels")
            label = labels.get(topic_id) if isinstance(labels, dict) else None
            label = _clean_label(label) if label else ""
            # Reject junk labels like "Discussion / Topic" that BERTopic emits for the catch-all cluster.
            tokens = {t.strip().lower() for t in label.split("/")}
            if topic_id == -1 or not label or tokens <= GENERIC_TOPIC_TOKENS:
                return {"topic_id": topic_id, "topic_label": fallback_label, "topic_confidence": fallback_conf, "topic_source": "rules (model label was generic)"}
            return {"topic_id": topic_id, "topic_label": label, "topic_confidence": round(conf, 4), "topic_source": "BERTopic"}
        except Exception:
            pass

    return {"topic_id": -1, "topic_label": fallback_label, "topic_confidence": fallback_conf, "topic_source": "rules"}


# ---------------------------------------------------------------------------
# Stage 4: churn
# ---------------------------------------------------------------------------


def predict_churn(sentiment: str, sentiment_score: float, priority: str, context: dict, art: dict) -> dict:
    """
    context keys (from the UI, with sensible defaults):
      ticket_count   number of tickets this customer has opened before
      usage          product usage 0 to 100
      avg_sentiment  average sentiment of past tickets, -1 to 1
    """
    ticket_count = int(context.get("ticket_count", 1))
    usage = float(context.get("usage", 50))
    avg_sentiment = float(context.get("avg_sentiment", sentiment_score))
    # Blend history with this ticket so a single bad message doesn't dominate.
    avg_sentiment = (avg_sentiment * ticket_count + sentiment_score) / (ticket_count + 1)
    ticket_frequency = {"High": 5, "Medium": 3}.get(priority, 2)

    features = pd.DataFrame([{
        "ticket_count": ticket_count,
        "avg_sentiment": avg_sentiment,
        "usage": usage,
        "Ticket Frequency": ticket_frequency,
        "sentiment_category_Neutral": int(sentiment == "Neutral"),
        "sentiment_category_Positive": int(sentiment == "Positive"),
    }])

    prob = None
    source = "heuristic"
    if art.get("churn_model") is not None and art.get("scaler") is not None:
        try:
            scaled = art["scaler"].transform(features[["ticket_count", "avg_sentiment", "usage"]])
            df = pd.DataFrame(scaled, columns=["ticket_count", "avg_sentiment", "usage"])
            for col in ["Ticket Frequency", "sentiment_category_Neutral", "sentiment_category_Positive"]:
                df[col] = features[col].values
            order = getattr(art["churn_model"], "feature_names_in_", df.columns)
            df = df[list(order)]
            prob = float(art["churn_model"].predict_proba(df)[0][1])
            source = "trained model"
        except Exception:
            prob = None

    if prob is None:
        # Transparent heuristic: low usage, negative history and repeat tickets push risk up.
        prob = 0.15
        prob += (1 - usage / 100) * 0.35
        prob += max(0.0, -avg_sentiment) * 0.30
        prob += min(ticket_count, 10) / 10 * 0.15
        prob += 0.05 if priority == "High" else 0.0
        prob = float(min(max(prob, 0.02), 0.98))

    risk = "High" if prob >= 0.70 else "Medium" if prob >= 0.40 else "Low"
    return {"churn_probability": round(prob, 4), "churn_risk": risk, "churn_source": source}


# ---------------------------------------------------------------------------
# Stage 5: business risk index
# ---------------------------------------------------------------------------


def calculate_business_risk(priority: str, sentiment: str, churn_probability: float) -> dict:
    score = {"High": 40, "Medium": 25}.get(priority, 10)
    score += {"Negative": 35, "Neutral": 15}.get(sentiment, 5)
    score += churn_probability * 25
    score = round(score, 2)
    level = "Critical" if score >= 75 else "High" if score >= 50 else "Medium" if score >= 25 else "Low"
    return {"business_risk_score": score, "business_risk_level": level}


# ---------------------------------------------------------------------------
# Stage 6: recommendations
# ---------------------------------------------------------------------------


def generate_recommendation(category, sentiment, emotion, topic_label, churn_risk, business_risk_level, priority) -> dict:
    recs: list[str] = []
    if business_risk_level == "Critical":
        recs += ["Escalate immediately to a senior support rep.", "Respond within 1 hour."]
    elif business_risk_level == "High":
        recs.append("Respond within 4 hours and assign an owner.")
    if churn_risk == "High":
        recs += ["Start a retention workflow (personal follow-up, credit, or call).", "Flag the account for a manager review."]
    if sentiment == "Negative":
        recs.append("Lead the reply with an apology and a concrete fix, not a form letter.")
    if emotion in ("Angry", "Sad"):
        recs.append("Acknowledge the frustration explicitly before explaining anything.")
    if emotion == "Anxious":
        recs.append("Reassure first: confirm the account is safe and give a timeline.")
    if emotion == "Confused":
        recs.append("Reply with step-by-step instructions or a short screen recording.")
    if priority == "Low" and sentiment != "Negative":
        recs.append("Standard queue is fine for this one.")
    recs.append(f"Route to the {category} team.")
    if topic_label != "General":
        recs.append(f"Check for other open tickets tagged '{topic_label}'; this may be a wider issue.")
    recs = list(dict.fromkeys(recs))
    summary = f"{category} | {topic_label} | Priority {priority} | Risk {business_risk_level}"
    return {"recommendations": recs, "summary": summary}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(text: str, art: dict, context: dict | None = None) -> dict:
    context = context or {}
    ticket = predict_ticket_category(text, art)
    sent = predict_sentiment_emotion(text)
    topic = predict_topic(text, art)
    churn = predict_churn(sent["sentiment"], sent["sentiment_score"], ticket["priority"], context, art)
    risk = calculate_business_risk(ticket["priority"], sent["sentiment"], churn["churn_probability"])
    rec = generate_recommendation(
        ticket["category"], sent["sentiment"], sent["emotion"], topic["topic_label"],
        churn["churn_risk"], risk["business_risk_level"], ticket["priority"],
    )
    return {**ticket, **sent, **topic, **churn, **risk, **rec, "ticket_text": text.strip()}


def run_batch(texts: list[str], art: dict, context: dict | None = None) -> pd.DataFrame:
    rows = [run_pipeline(t, art, context) for t in texts if str(t).strip()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["recommendations"] = df["recommendations"].apply(lambda r: " | ".join(r))
    return df
