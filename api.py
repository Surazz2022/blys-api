"""
Section 4 — FastAPI REST API
Serves:
  POST /recommend       → models/recommendation_model.pkl  (Section 2)
  POST /chatbot         → models/chatbot_model.pkl          (Section 3)
  GET  /health          → liveness check
  DELETE /session/{id}  → reset a chatbot session

Run:
  pip install -r requirements.txt
  python api.py
  # Swagger UI → http://localhost:8000/docs
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import re
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import json
import joblib
import numpy as np
import pandas as pd
import dateparser
import nltk
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

try:
    from groq import Groq as _GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("blys-api")

# ---------------------------------------------------------------------------
# Global model artifacts (loaded once at startup)
# ---------------------------------------------------------------------------
REC_MODEL_PATH  = "models/recommendation_model.pkl"
CHAT_MODEL_PATH = "models/chatbot_model.pkl"
TX_PATH         = "data/transactions.csv"       # transaction history (Section 2 output)

rec_artifact:  dict | None = None
chat_artifact: dict | None = None
_customer_df:  pd.DataFrame | None = None   # per-customer aggregates derived from transactions
_sia:          SentimentIntensityAnalyzer | None = None
_ref_date:     pd.Timestamp | None = None   # max booking date — days-inactive anchor
_st_model:     SentenceTransformer | None = None   # shared sentence encoder
_groq_client:  object | None = None          # Groq LLM client (None if key not set)


# ---------------------------------------------------------------------------
# Lifespan — load everything once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec_artifact, chat_artifact, _customer_df, _sia, _ref_date, _st_model, _groq_client

    # Groq LLM client — only initialised when GROQ_API_KEY is set
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key and _GROQ_AVAILABLE:
        _groq_client = _GroqClient(api_key=groq_key)
        log.info("Groq client initialised — LLM-powered chat enabled.")
    else:
        log.info("GROQ_API_KEY not set — falling back to local classifier chat.")

    # VADER lexicon
    nltk.download("vader_lexicon", quiet=True)
    _sia = SentimentIntensityAnalyzer()

    # Transaction history → derive per-customer feature aggregates
    if os.path.exists(TX_PATH):
        tx = pd.read_csv(TX_PATH)
        tx["Date"] = pd.to_datetime(tx["Date"])

        # VADER sentiment on transaction reviews
        tx["Sentiment"] = tx["Review_Text"].apply(
            lambda t: _sia.polarity_scores(str(t))["compound"] if pd.notna(t) else 0.0
        )

        _ref_date = tx["Date"].max()

        # Aggregate to one row per customer
        cust_agg = tx.groupby("Customer_ID").agg(
            Booking_Frequency=("Service",  "count"),
            Avg_Spending     =("Spending", "mean"),
            Last_Activity    =("Date",     "max"),
            Sentiment_Score  =("Sentiment","mean"),
        ).reset_index()
        cust_agg["Days_Inactive"] = (_ref_date - cust_agg["Last_Activity"]).dt.days

        # Preferred service = mode service per customer
        pref = (
            tx.groupby("Customer_ID")["Service"]
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
            .rename(columns={"Service": "Preferred_Service"})
        )
        _customer_df = cust_agg.merge(pref, on="Customer_ID")
        log.info("Transactions loaded — %d rows → %d customers",
                 len(tx), len(_customer_df))
    else:
        log.warning("transactions.csv not found at %s — run Section 2 notebook first", TX_PATH)

    # Recommendation model
    if os.path.exists(REC_MODEL_PATH):
        rec_artifact = joblib.load(REC_MODEL_PATH)
        log.info("Recommendation model loaded — services: %s", rec_artifact["service_classes"])
    else:
        log.warning("Recommendation model not found at %s — run Section 2 notebook first", REC_MODEL_PATH)

    # Chatbot model
    if os.path.exists(CHAT_MODEL_PATH):
        chat_artifact = joblib.load(CHAT_MODEL_PATH)
        log.info("Chatbot model loaded — intents: %s", chat_artifact["intents"])

        # Load sentence transformer (shared across all sessions)
        model_name = chat_artifact.get("st_model_name", "all-MiniLM-L6-v2")
        log.info("Loading sentence transformer: %s", model_name)
        _st_model = SentenceTransformer(model_name)
        log.info("Sentence transformer ready.")
    else:
        log.warning("Chatbot model not found at %s — run Section 3 notebook first", CHAT_MODEL_PATH)

    yield  # application runs here

    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Blys AI API",
    description=(
        "**Section 4** — REST API serving the recommendation model and NLP chatbot.\n\n"
        "- `POST /recommend` — ranked service recommendations for a customer\n"
        "- `POST /chatbot`   — multi-turn NLP chatbot with booking actions\n"
        "- `GET  /health`    — liveness check\n"
        "- `DELETE /session/{session_id}` — reset a chatbot conversation"
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ===========================================================================
# Section 2 — POST /recommend
# ===========================================================================

class RecommendRequest(BaseModel):
    customer_id: int

    model_config = {"json_schema_extra": {"example": {"customer_id": 1001}}}


class RecommendResponse(BaseModel):
    customer_id:          int
    recommended_services: list[str]



@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get recommended services for a customer",
    tags=["Recommendation"],
)
def recommend(req: RecommendRequest):
    """
    Returns a ranked list of services for the given customer.

    - **Hybrid** (existing customers): 50 % content-based GBM + 50 % NMF collaborative scores
    - **Content-based only** (new/unknown customers): GBM probability scores
    """
    if rec_artifact is None:
        raise HTTPException(503, "Recommendation model not loaded — run Section 2 notebook first")
    if _customer_df is None:
        raise HTTPException(503, "transactions.csv not found — run Section 2 notebook first")

    # Look up customer
    row = _customer_df[_customer_df["Customer_ID"] == req.customer_id]
    if row.empty:
        raise HTTPException(404, f"Customer {req.customer_id} not found in dataset")
    customer = row.iloc[0]

    # ── Build feature vector (must match Section 2 training) ────────────────
    days_inactive     = (_ref_date - customer["Last_Activity"]).days
    review_text       = str(customer.get("Review_Text", ""))
    sentiment         = _sia.polarity_scores(review_text)["compound"] if review_text else 0.0
    target_enc        = rec_artifact["target_encoding"]
    preferred_svc     = str(customer.get("Preferred_Service", ""))
    service_value_enc = target_enc.get(preferred_svc, float(np.mean(list(target_enc.values()))))

    feature_vals = {
        "Booking_Frequency": float(customer.get("Booking_Frequency", 0)),
        "Avg_Spending":      float(customer.get("Avg_Spending", 0)),
        "Days_Inactive":     float(days_inactive),
        "Sentiment_Score":   float(sentiment),
        "Service_Value_Enc": float(service_value_enc),
    }

    cb_features = rec_artifact["cb_features"]
    X           = pd.DataFrame([feature_vals])[cb_features].fillna(0)
    X_scaled    = rec_artifact["scaler"].transform(X)

    # ── Content-based scores ─────────────────────────────────────────────────
    cb_scores       = rec_artifact["cb_model"].predict_proba(X_scaled)[0]
    service_classes = rec_artifact["service_classes"]

    # ── NMF collaborative scores (in-set customers only) ─────────────────────
    customer_ids = rec_artifact["customer_ids"]
    service_cols = rec_artifact["service_cols"]
    R_hat        = rec_artifact["R_hat"]
    method       = "content_based"

    if req.customer_id in customer_ids:
        idx      = customer_ids.index(req.customer_id)
        nmf_raw  = R_hat[idx]
        nmf_vec  = np.array([
            nmf_raw[service_cols.index(s)] if s in service_cols else 0.0
            for s in service_classes
        ])
        nmf_vec  = nmf_vec / (nmf_vec.max() + 1e-9)   # normalise to [0, 1]
        final    = 0.5 * cb_scores + 0.5 * nmf_vec
        method   = "hybrid"
    else:
        final = cb_scores

    # ── Rank and return ───────────────────────────────────────────────────────
    ranked = sorted(zip(service_classes, final), key=lambda x: x[1], reverse=True)
    return RecommendResponse(
        customer_id          = req.customer_id,
        recommended_services = [s for s, _ in ranked],
    )


# ===========================================================================
# Section 3 — POST /chatbot
# Chatbot engine is self-contained here so api.py has no notebook dependency.
# All configuration values (_THRESHOLD, _PRICE_LIST, _SERVICES) are read from
# chat_artifact after it is loaded — never hardcoded.
# ===========================================================================

# ── Dialogue states ──────────────────────────────────────────────────────────
class _State(Enum):
    IDLE                 = auto()
    RESCHEDULE_ASK_BK_ID = auto()
    RESCHEDULE_ASK_DT    = auto()
    RESCHEDULE_CONFIRM   = auto()
    CANCEL_ASK_BK_ID     = auto()
    CANCEL_CONFIRM       = auto()
    STATUS_ASK_BK_ID     = auto()
    PRICE_SHOWN          = auto()
    BOOKING_ASK_SERVICE  = auto()
    BOOKING_ASK_DT       = auto()
    BOOKING_CONFIRM      = auto()
    AVAIL_SHOWN          = auto()
    DONE                 = auto()


# Primary intents — always handled regardless of FSM state.
# These are intents that represent a clear user request and should be answered
# naturally based on WHAT the user said, not WHERE they are in a flow.
#
# Only slot-filling responses are state-dependent:
#   confirmation_yes / confirmation_no  — answering a pending yes/no question
#   provide_booking_id                  — giving a booking ID when asked
#   provide_datetime                    — giving a date when asked
#
# Everything else resets the flow and responds immediately.
_GLOBAL_INTERRUPT_INTENTS = frozenset({
    "availability_inquiry",
    "greeting",
    "goodbye",
    "price_inquiry",
    "reschedule",
    "cancel",
    "check_booking_status",
    "booking_create",
    "out_of_scope",
})


# ── Mock availability slots (replace with real calendar API in production) ────
_AVAILABILITY: dict[str, list[str]] = {
    "Monday":    ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"],
    "Tuesday":   ["10:00 AM", "1:00 PM", "3:00 PM", "6:00 PM"],
    "Wednesday": ["9:00 AM", "12:00 PM", "2:00 PM", "5:00 PM"],
    "Thursday":  ["10:00 AM", "11:00 AM", "3:00 PM", "7:00 PM"],
    "Friday":    ["9:00 AM", "1:00 PM", "4:00 PM", "6:00 PM"],
    "Saturday":  ["10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM", "6:00 PM"],
    "Sunday":    ["11:00 AM", "1:00 PM", "3:00 PM", "5:00 PM"],
}

# ── Mock booking database (replace with real API in production) ──────────────
@dataclass
class _Booking:
    booking_id:   str
    service:      str
    scheduled_at: datetime
    status:       str = "confirmed"
    therapist:    str = "Sarah"


_MOCK_DB: dict[str, _Booking] = {
    "BK-001": _Booking("BK-001", "Swedish Massage",   datetime(2025, 3, 28, 10,  0)),
    "BK-002": _Booking("BK-002", "Facial",             datetime(2025, 4,  2, 14,  0), therapist="Emma"),
    "BK-003": _Booking("BK-003", "Deep Tissue",        datetime(2025, 4,  5,  9,  0), therapist="James"),
    "BK-004": _Booking("BK-004", "Body Scrub",         datetime(2025, 4,  8, 11,  0), therapist="Mia"),
    "BK-005": _Booking("BK-005", "Hair Spa",           datetime(2025, 4, 10, 15,  0), therapist="Lena"),
    "BK-006": _Booking("BK-006", "Wellness Package",   datetime(2025, 4, 12, 10,  0), therapist="Sarah"),
    "BK-007": _Booking("BK-007", "Swedish Massage",   datetime(2025, 4, 14, 13,  0), therapist="James"),
    "BK-008": _Booking("BK-008", "Facial",             datetime(2025, 4, 16,  9,  0), therapist="Emma"),
    "BK-009": _Booking("BK-009", "Deep Tissue",        datetime(2025, 4, 18, 16,  0), therapist="Mia"),
    "BK-010": _Booking("BK-010", "Body Scrub",         datetime(2025, 4, 20, 11,  0), therapist="Lena"),
}

_THERAPISTS = ["Sarah", "Emma", "James", "Mia", "Lena"]


def _next_booking_id() -> str:
    """Generate next sequential booking ID."""
    n = max((int(k.split("-")[1]) for k in _MOCK_DB), default=0) + 1
    return f"BK-{n:03d}"

# ── Regex helpers ────────────────────────────────────────────────────────────
_BK_RE    = re.compile(r"BK[-\s]?\d+(?:-\d+)?", re.IGNORECASE)
# Start of each time preposition phrase — we slice from here to end of string
_PREP_POS = re.compile(r"\b(?:to|on|at|for)\s+", re.IGNORECASE)
# Skip fragments that begin with stop/intent words (not temporal expressions)
_DT_SKIP  = re.compile(
    r"^(?:me|you|us|my|the|a|an|it|this|reschedule|book|cancel|move|change)\b",
    re.IGNORECASE,
)


def _parse_datetime(text: str):
    """
    Robustly parse a datetime from a natural-language sentence.

    Strategy: dateparser fails on full sentences (e.g. "Reschedule BK-001 to
    Monday 2pm").  We instead slice the string at every time preposition
    (to/on/at/for) and try each fragment longest-first.  Falls back to
    stripping the booking ID and parsing the whole clean text.
    """
    clean = _BK_RE.sub("", text).strip()

    # Collect all substrings starting just after each preposition
    candidates: set[str] = set()
    for m in _PREP_POS.finditer(clean):
        frag = clean[m.end():].strip().rstrip("?.,!")
        if frag and not _DT_SKIP.match(frag):
            candidates.add(frag)

    for frag in sorted(candidates, key=len, reverse=True):
        dt = dateparser.parse(frag, settings={"PREFER_DATES_FROM": "future"})
        if dt:
            return dt

    # Full clean-text fallback
    return dateparser.parse(clean, settings={"PREFER_DATES_FROM": "future"})


def _extract_entities(text: str, services: list[str]) -> dict:
    """Extract booking_id, datetime, and service_type from a user message."""
    ent: dict = {
        "booking_id":   None,
        "datetime_dt":  None,
        "datetime_str": None,
        "service_type": None,
    }
    # Booking ID
    m = _BK_RE.search(text)
    if m:
        ent["booking_id"] = m.group().upper().replace(" ", "-")

    # Date / time — multi-fragment parser handles full sentences correctly
    dt = _parse_datetime(text)
    if dt:
        ent["datetime_dt"]  = dt
        ent["datetime_str"] = dt.strftime("%d %b %Y %I:%M %p")

    # Service type — full phrase match first, then partial word match
    # Uses regex tokenisation so punctuation ("spa?", "massage!") never blocks a match
    text_lower = text.lower()
    matched = False
    for svc in services:
        if svc.lower() in text_lower:
            ent["service_type"] = svc.title()
            matched = True
            break
    if not matched:
        # Partial match: require tokens ≥5 chars to avoid stop-words ("the", "and",
        # "for", "spa") accidentally substring-matching inside a service name.
        # e.g. "the" inside "aromatherapy" or "spa" inside "wellness package".
        words = re.findall(r'\b\w{5,}\b', text_lower)
        for svc in services:
            svc_lower = svc.lower()
            # Word must match a WHOLE word in the service name, not just a substring
            svc_words = re.findall(r'\b\w+\b', svc_lower)
            if any(w in svc_words for w in words):
                ent["service_type"] = svc.title()
                break

    return ent


# ── Response templates (loaded from artifact after startup) ──────────────────
_T: dict[str, str] = {}        # populated in _BlysChat.__init__ from artifact
_PRICE_LIST: dict[str, str] = {}

# Hardcoded fallback — always available regardless of artifact version.
# Overwritten by artifact["services"] at startup if present and non-empty.
_SERVICES: list[str] = [
    "Massage", "Facial", "Body Scrub", "Hair Spa", "Wellness Package",
    "Swedish Massage", "Deep Tissue",
]


class _BlysChat:
    """
    Contextually-intelligent per-session chatbot.

    Intelligence layers
    ───────────────────
    1. Semantic embeddings   — SentenceTransformer (all-MiniLM-L6-v2)
    2. Context window        — state name + last bot response prepended to each query
    3. State-conditioned priors — Bayesian blend of classifier output with per-state prior
    4. FSM dialogue management  — 12 states, slot filling across turns
    """

    def __init__(self, artifact: dict):
        self._clf       = artifact["classifier"]
        self._le        = artifact["label_encoder"]
        self._intents   = artifact["intents"]
        self._threshold = artifact.get("confidence_threshold", 0.45)
        self._alpha     = artifact.get("prior_alpha", 0.30)
        self._priors    = artifact.get("state_priors", {})

        global _T, _PRICE_LIST, _SERVICES
        if not _T:
            _T          = artifact.get("response_templates", {})
        if not _PRICE_LIST:
            _PRICE_LIST = artifact.get("price_list", {})
        # Merge artifact services into the hardcoded fallback (dedup, preserve order)
        for svc in artifact.get("services", []):
            if svc not in _SERVICES:
                _SERVICES.append(svc)

        self.state     = _State.IDLE
        self.slots:    dict = {}
        self.history:  list = []          # full conversation turns, newest last
        self._last_bot = ""
        self._last_user= ""

    # ── Rule-based overrides for short, unambiguous messages ─────────────────
    # These patterns are so clear-cut that the classifier shouldn't be trusted
    # over them.  Applied BEFORE classification to short-circuit the model.
    _GREETING_RE = re.compile(
        r'^\s*(?:hi|hello|hey|howdy|good\s*(?:morning|afternoon|evening)|greetings|hiya)\W*$',
        re.IGNORECASE,
    )
    _GOODBYE_RE  = re.compile(
        r'^\s*(?:bye|goodbye|see\s*you|cya|good\s*night|take\s*care|thanks?\s*bye)\W*$',
        re.IGNORECASE,
    )

    # ── Public ───────────────────────────────────────────────────────────────
    def chat(self, msg: str) -> tuple[str, str, float]:
        # Rule-based override for unambiguous greetings / goodbyes
        if self._GREETING_RE.match(msg):
            intent, conf = "greeting", 1.0
        elif self._GOODBYE_RE.match(msg):
            intent, conf = "goodbye", 1.0
        else:
            intent, conf = self._classify(msg)
        entities     = _extract_entities(msg, _SERVICES)

        # ── Service-name shortcut ─────────────────────────────────────────────
        # Only fires when the message is SHORT (≤5 words) and the service was
        # matched exactly (not via partial substring), so a sentence like
        # "I want to check the status" never gets overridden to booking_create.
        msg_words = msg.strip().split()
        service_is_exact = (
            entities["service_type"] is not None
            and entities["service_type"].lower() in msg.lower()
        )
        if (service_is_exact
                and len(msg_words) <= 5
                and intent in ("booking_create", "out_of_scope")
                and self.state in (_State.IDLE, _State.BOOKING_ASK_SERVICE)):
            intent = "booking_create"
            conf   = max(conf, 0.70)

        for key in ("booking_id", "datetime_dt", "datetime_str", "service_type"):
            if entities[key] is not None:
                self.slots[key] = entities[key]
        response = self._fsm(intent, msg)
        log.info("intent=%s conf=%.2f state=%s", intent, conf, self.state.name)

        # ── Append turn to history ────────────────────────────────────────────
        self.history.append({
            "turn":    len(self.history) + 1,
            "user":    msg,
            "bot":     response,
            "intent":  intent,
            "conf":    round(conf, 4),
            "state":   self.state.name,
        })
        self._last_user = msg
        self._last_bot  = response
        return response, intent, conf

    def reset(self):
        self.state      = _State.IDLE
        self.slots      = {}
        self.history    = []
        self._last_bot  = ""
        self._last_user = ""

    # ── Retry-aware slot-ask helper ──────────────────────────────────────────
    # Tracks how many times each slot has been asked in the current state so
    # the bot can vary its wording instead of repeating the same line verbatim.
    def _ask(self, slot_key: str, messages: list[str]) -> str:
        """Return an escalating prompt for a missing slot."""
        count_key = f"_ask_{slot_key}"
        n = self.slots.get(count_key, 0)
        self.slots[count_key] = n + 1
        return messages[min(n, len(messages) - 1)]

    # ── Context-aware intent classification ──────────────────────────────────
    # Uses a rolling window of the last N turns so the classifier understands
    # what was said before — not just the immediately preceding exchange.
    _CONTEXT_WINDOW = 5    # number of previous turns to include

    def _classify(self, text: str) -> tuple[str, float]:
        # Build context string: [STATE] + last N turns (oldest→newest) + current
        parts = [f"[STATE:{self.state.name}]"]
        for turn in self.history[-self._CONTEXT_WINDOW:]:
            parts.append(f"[USER:{turn['user'][:80]}]")
            parts.append(f"[BOT:{turn['bot'][:120]}]")
        parts.append(text)
        context_str = " ".join(parts)

        # Encode and classify
        emb   = _st_model.encode([context_str], normalize_embeddings=True)[0]
        proba = self._clf.predict_proba([emb])[0]

        # Blend with state-conditioned prior
        prior = self._priors.get(self.state.name, {})
        if prior:
            prior_vec = np.array(
                [prior.get(i, 0.01) for i in self._intents], dtype=float
            )
            prior_vec /= prior_vec.sum()
            proba = (1 - self._alpha) * proba + self._alpha * prior_vec
            proba /= proba.sum()

        idx    = int(proba.argmax())
        conf   = float(proba[idx])
        intent = (
            self._le.inverse_transform([idx])[0]
            if conf >= self._threshold else "out_of_scope"
        )
        return intent, conf

    # ── FSM ───────────────────────────────────────────────────────────────────
    def _fsm(self, intent: str, raw: str) -> str:
        s  = self.state
        sl = self.slots

        # ── Global interrupts — always answered regardless of current state ───
        # If the user asks something topic-independent while mid-flow, answer
        # directly and reset so they can start a clean conversation.
        if s != _State.IDLE and intent in _GLOBAL_INTERRUPT_INTENTS:
            self.reset()           # clear stale slots & state first
            return self._fsm(intent, raw)   # re-dispatch from IDLE

        # ── IDLE ──────────────────────────────────────────────────────────────
        if s == _State.IDLE:
            if intent == "greeting":
                return _T.get("greeting", "Hello! Welcome to Blys.")
            if intent == "goodbye":
                return _T.get("goodbye", "Thank you for using Blys. Goodbye!")
            if intent == "out_of_scope":
                return _T.get("out_of_scope",
                    "I can help with bookings, cancellations, rescheduling, status checks, and pricing.")

            if intent == "reschedule":
                # One-shot: all slots already in message → jump straight to confirm
                if "booking_id" in sl and "datetime_dt" in sl:
                    self.state = _State.RESCHEDULE_CONFIRM
                    return _T.get("reschedule_confirm",
                        "Got it. Reschedule booking {booking_id} to {new_time}. Confirm? (yes / no)"
                    ).format(booking_id=sl["booking_id"], new_time=sl["datetime_str"])
                # Booking ID known → skip that step
                if "booking_id" in sl:
                    self.state = _State.RESCHEDULE_ASK_DT
                    return _T.get("reschedule_ask_dt",
                        "Please provide the new date and time for your booking.")
                self.state = _State.RESCHEDULE_ASK_BK_ID
                return _T.get("reschedule_offer_help",
                    "Sure! Please provide your booking ID (e.g. BK-001).")

            if intent == "cancel":
                if "booking_id" in sl:
                    return self._show_cancel_confirm()
                self.state = _State.CANCEL_ASK_BK_ID
                return _T.get("cancel_offer_help",
                    "I can help cancel your booking. Please provide your booking ID.")

            if intent == "check_booking_status":
                if "booking_id" in sl:
                    return self._show_status()   # sets state=DONE internally
                self.state = _State.STATUS_ASK_BK_ID
                return _T.get("status_ask_bk_id",
                    "Sure! Please provide your booking ID to check the status.")

            if intent == "price_inquiry":
                self.state = _State.PRICE_SHOWN
                return self._price_response(sl.get("service_type"))

            if intent == "availability_inquiry":
                self.state = _State.AVAIL_SHOWN
                return self._availability_response(raw)

            if intent == "booking_create":
                # One-shot: service AND date given → jump to confirm
                if "service_type" in sl and "datetime_dt" in sl:
                    self.state = _State.BOOKING_CONFIRM
                    return _T.get("booking_confirm",
                        "I'll book a {service} for {new_time}. Confirm? (yes / no)"
                    ).format(service=sl["service_type"], new_time=sl["datetime_str"])
                if "service_type" in sl:
                    self.state = _State.BOOKING_ASK_DT
                    return _T.get("booking_ask_dt",
                        "When would you like to book the {service}?").format(service=sl["service_type"])
                self.state = _State.BOOKING_ASK_SERVICE
                return _T.get("booking_ask_service",
                    "Which service? (Massage, Facial, Body Scrub, Hair Spa, Wellness Package)")

        # ── RESCHEDULE FLOW ───────────────────────────────────────────────────
        if s == _State.RESCHEDULE_ASK_BK_ID:
            if intent == "confirmation_no":
                self.reset()
                return _T.get("reschedule_cancelled", "No problem! Anything else I can help with?")
            if "booking_id" in sl:
                self.state = _State.RESCHEDULE_ASK_DT
                return _T.get("reschedule_ask_dt",
                    "Please provide the new date and time for your booking.")
            return self._ask("bk_id", [
                "Please share your booking ID — you can find it in your confirmation email (e.g. BK-001).",
                "I still need your booking ID to continue. It looks like BK-001.",
                "I'm having trouble finding the booking ID. Check your Blys confirmation email and share the ID.",
            ])

        if s == _State.RESCHEDULE_ASK_DT:
            if "datetime_dt" in sl:
                self.state = _State.RESCHEDULE_CONFIRM
                return _T.get("reschedule_confirm",
                    "Got it. Reschedule booking {booking_id} to {new_time}. Confirm? (yes / no)"
                ).format(booking_id=sl.get("booking_id", "N/A"), new_time=sl["datetime_str"])
            return self._ask("datetime", [
                "What date and time works for you? (e.g. 'Monday at 2pm' or 'April 20 at 10am')",
                "I need a date and time — something like 'Friday at 3pm' works perfectly.",
                "Could you give me a specific date and time? For example: 'next Tuesday at 11am'.",
            ])

        if s == _State.RESCHEDULE_CONFIRM:
            if intent == "confirmation_yes":
                bk_id   = sl.get("booking_id", "BK-?")
                booking = _MOCK_DB.get(bk_id)
                self.state = _State.DONE
                if booking:
                    old_time             = booking.scheduled_at.strftime("%d %b %Y %I:%M %p")
                    booking.scheduled_at = sl["datetime_dt"]
                    booking.status       = "rescheduled"
                    return _T.get("reschedule_success",
                        "Done! Booking {booking_id} rescheduled from {old_time} → {new_time} "
                        "with {therapist}. You will receive a confirmation shortly. ✅"
                    ).format(booking_id=bk_id, old_time=old_time,
                             new_time=sl["datetime_str"], therapist=booking.therapist)
                return _T.get("not_found", "Booking {booking_id} not found.").format(booking_id=bk_id)
            if intent == "confirmation_no":
                self.state = _State.RESCHEDULE_ASK_DT
                return _T.get("reschedule_ask_dt",
                    "Please provide the new date and time for your booking.")

        # ── CANCEL FLOW ───────────────────────────────────────────────────────
        if s == _State.CANCEL_ASK_BK_ID:
            if "booking_id" in sl:
                return self._show_cancel_confirm()
            return self._ask("bk_id", [
                "Please provide your booking ID — it's in your confirmation email (e.g. BK-002).",
                "I still need your booking ID to cancel. It should start with BK-.",
                "I can't proceed without the booking ID. You can find it in your Blys confirmation email.",
            ])

        if s == _State.CANCEL_CONFIRM:
            if intent == "confirmation_yes":
                bk_id   = sl["booking_id"]
                booking = _MOCK_DB.get(bk_id)
                self.state = _State.DONE
                if booking:
                    booking.status = "cancelled"
                    return _T.get("cancel_success",
                        "Your {service} booking ({booking_id}) has been cancelled. ✅"
                    ).format(service=booking.service, booking_id=bk_id)
                return _T.get("not_found", "Booking {booking_id} not found.").format(booking_id=bk_id)
            if intent == "confirmation_no":
                self.reset()
                return _T.get("cancel_aborted", "No problem! Anything else I can help with?")

        # ── STATUS FLOW ───────────────────────────────────────────────────────
        if s == _State.STATUS_ASK_BK_ID:
            if "booking_id" in sl:
                return self._show_status()
            if intent == "confirmation_no":
                self.reset()
                return "No problem! Anything else I can help with?"
            return self._ask("bk_id", [
                "Please share your booking ID and I'll pull up the details right away.",
                "I need the booking ID to look that up — it starts with BK- (e.g. BK-003).",
                "Still need the booking ID. Check your confirmation email for something like BK-001.",
            ])

        # ── PRICE SHOWN ───────────────────────────────────────────────────────
        if s == _State.PRICE_SHOWN:
            if intent in ("confirmation_yes", "booking_create"):
                self.state = _State.BOOKING_ASK_SERVICE
                return _T.get("booking_ask_service",
                    "Which service? (Massage, Facial, Body Scrub, Hair Spa, Wellness Package)")
            self.state = _State.IDLE
            return "No problem! Anything else I can help with?"

        # ── AVAILABILITY SHOWN ────────────────────────────────────────────────
        if s == _State.AVAIL_SHOWN:
            if intent in ("confirmation_yes", "booking_create"):
                self.state = _State.BOOKING_ASK_SERVICE
                return _T.get("booking_ask_service",
                    "Which service? (Massage, Facial, Body Scrub, Hair Spa, Wellness Package)")
            self.state = _State.IDLE
            return "No problem! Anything else I can help with?"

        # ── NEW BOOKING FLOW ──────────────────────────────────────────────────
        if s == _State.BOOKING_ASK_SERVICE:
            chosen = sl.get("service_type") or raw.strip().title()
            self.slots["service_type"] = chosen
            self.state = _State.BOOKING_ASK_DT
            return _T.get("booking_ask_dt",
                "When would you like to book the {service}?").format(service=chosen)

        if s == _State.BOOKING_ASK_DT:
            if "datetime_dt" in sl:
                self.state = _State.BOOKING_CONFIRM
                return _T.get("booking_confirm",
                    "I'll book a {service} for {new_time}. Confirm? (yes / no)"
                ).format(service=sl.get("service_type", "the service"), new_time=sl["datetime_str"])
            return self._ask("datetime", [
                "When would you like to book? Any day and time works — e.g. 'Saturday at 10am'.",
                "Just need a date and time — something like 'next Monday at 2pm'.",
                "Could you tell me when? For example: 'Friday afternoon' or 'April 18 at 3pm'.",
            ])

        if s == _State.BOOKING_CONFIRM:
            if intent == "confirmation_yes":
                # ── Actually create the booking record ────────────────────────
                import random
                bk_id     = _next_booking_id()
                therapist = random.choice(_THERAPISTS)
                new_bk    = _Booking(bk_id, sl.get("service_type", "Service"),
                                     sl["datetime_dt"], status="confirmed",
                                     therapist=therapist)
                _MOCK_DB[bk_id] = new_bk
                self.state = _State.DONE
                return _T.get("booking_success",
                    "Your {service} has been booked for {new_time} with {therapist}. "
                    "Booking ID: {booking_id}. Confirmation coming shortly. ✅"
                ).format(service=new_bk.service, new_time=sl["datetime_str"],
                         therapist=therapist, booking_id=bk_id)
            self.reset()
            return _T.get("booking_cancelled", "No problem! Anything else I can help with?")

        # ── DONE — reset and re-process ───────────────────────────────────────
        if s == _State.DONE:
            self.reset()
            return self._fsm(intent, raw)

        return _T.get("ask_again", "Could you please rephrase that?")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _show_cancel_confirm(self) -> str:
        bk_id   = self.slots["booking_id"]
        booking = _MOCK_DB.get(bk_id)
        if booking:
            self.state = _State.CANCEL_CONFIRM   # only advance state when booking exists
            return _T.get("cancel_confirm",
                "Found booking {booking_id} — {service} on {scheduled_at}. Cancel? (yes / no)"
            ).format(booking_id=bk_id, service=booking.service,
                     scheduled_at=booking.scheduled_at.strftime("%d %b %Y %I:%M %p"),
                     therapist=booking.therapist)
        # Stay in ask-for-ID state so user can try a different ID
        self.state = _State.CANCEL_ASK_BK_ID
        del self.slots["booking_id"]             # clear bad ID so next message is re-extracted
        return _T.get("not_found",
            "I couldn't find booking {booking_id}. Please double-check the ID and try again."
        ).format(booking_id=bk_id)

    def _show_status(self) -> str:
        bk_id   = self.slots["booking_id"]
        booking = _MOCK_DB.get(bk_id)
        self.state = _State.DONE
        if booking:
            return _T.get("status_found",
                "Booking {booking_id}: {service} with {therapist} on {scheduled_at}. Status: {status}."
            ).format(booking_id=bk_id, service=booking.service,
                     therapist=booking.therapist,
                     scheduled_at=booking.scheduled_at.strftime("%d %b %Y %I:%M %p"),
                     status=booking.status.upper())
        return _T.get("status_not_found",
            "I couldn't find booking {booking_id}. Please double-check.").format(booking_id=bk_id)

    def _availability_response(self, raw: str) -> str:
        """
        Return available slots, optionally filtered by a day mentioned in the query.
        Falls back to showing the full weekly schedule if no specific day is found.
        """
        raw_lower = raw.lower()
        # Check if user asked about a specific day
        for day in _AVAILABILITY:
            if day.lower() in raw_lower:
                slots = ", ".join(_AVAILABILITY[day])
                return _T.get(
                    "availability_day",
                    "On {day} we have slots available at: {slots}. Would you like to book one?"
                ).format(day=day, slots=slots)

        # No specific day — show the weekly overview
        lines = "\n".join(
            f"  • {day}: {', '.join(times)}"
            for day, times in _AVAILABILITY.items()
        )
        return _T.get(
            "availability_all",
            "Our therapists are available Monday–Sunday, 8 AM–9 PM.\n\n"
            "Here are today's open slots:\n{slots}\n\n"
            "Which day works best for you? I can book it right away!"
        ).format(slots=lines)

    def _price_response(self, service_type: str | None) -> str:
        if service_type:
            key = next((k for k in _PRICE_LIST if service_type.lower() in k.lower()), None)
            if key:
                return _T.get("price_single",
                    "The price for {service} is {price}. Would you like to book?"
                ).format(service=key, price=_PRICE_LIST[key])
        table    = "\n".join(f"  • {k}: {v}" for k, v in _PRICE_LIST.items())
        template = _T.get("price_all", "Prices:\n{price_table}\nWould you like to book?")
        return template.format(price_table=table)


# ===========================================================================
# Groq LLM-powered chat  (used when GROQ_API_KEY is set)
# Replaces the FSM+template approach with a full LLM that understands context,
# generates natural responses, and calls tools to execute real booking actions.
# ===========================================================================

_GROQ_MODEL = "llama-3.3-70b-versatile"   # best free-tier model on Groq

_GROQ_SYSTEM_PROMPT = """You are a friendly and helpful booking assistant for Blys — Australia's leading on-demand mobile massage and beauty therapy service.

## Services & Prices
- Swedish Massage: $89
- Deep Tissue Massage: $99
- Facial: $79
- Body Scrub: $85
- Hair Spa: $75
- Wellness Package: $120

## Availability
Monday–Sunday, 8 AM–9 PM. Therapists come to the customer's location.

## What you can do
1. Book a new appointment (service + date/time needed)
2. Reschedule an existing booking (booking ID + new date/time needed)
3. Cancel a booking (booking ID needed)
4. Check the status of a booking (booking ID needed)
5. Answer questions about services, pricing, and availability

## Guidelines
- Be warm, concise, and professional.
- When the customer wants to take an action, use the provided tools — never pretend an action happened.
- If you need information (booking ID, date, service), ask for it naturally — one question at a time.
- After a tool call succeeds, summarise what was done clearly.
- Booking IDs look like BK-001, BK-002, etc.
- Never make up booking details — always use tool results."""

_GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_booking_status",
            "description": "Look up the details and current status of a booking by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "The booking ID, e.g. BK-001"
                    }
                },
                "required": ["booking_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": "Create a new booking for a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "The service name, e.g. Swedish Massage"
                    },
                    "datetime_str": {
                        "type": "string",
                        "description": "The requested date and time, e.g. 'Monday at 2pm' or 'April 20 at 10am'"
                    }
                },
                "required": ["service", "datetime_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_booking",
            "description": "Reschedule an existing booking to a new date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "The booking ID to reschedule"
                    },
                    "new_datetime": {
                        "type": "string",
                        "description": "The new date and time, e.g. 'Friday at 3pm'"
                    }
                },
                "required": ["booking_id", "new_datetime"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": "Cancel an existing booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "The booking ID to cancel"
                    }
                },
                "required": ["booking_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_availability",
            "description": "Get available appointment slots, optionally for a specific day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": "Optional: specific day to check, e.g. 'Saturday'"
                    }
                },
                "required": []
            }
        }
    }
]


class _BlysChatGroq:
    """
    LLM-powered chat session using Groq (Llama 3.3 70B).

    The full conversation history is sent on every turn so the model has
    complete context.  Booking actions are executed via tool calls against
    the same mock DB used by the legacy FSM chat.
    """

    def __init__(self, client):
        self._client  = client
        self._msgs:   list[dict] = []   # Groq message history (role/content format)
        self.history: list[dict] = []   # human-readable turn log
        self.state    = "LLM"           # satisfies current_state field in ChatResponse

    # ── Public ───────────────────────────────────────────────────────────────
    def chat(self, msg: str) -> tuple[str, str, float]:
        self._msgs.append({"role": "user", "content": msg})

        response = self._client.chat.completions.create(
            model       = _GROQ_MODEL,
            messages    = [{"role": "system", "content": _GROQ_SYSTEM_PROMPT}] + self._msgs,
            tools       = _GROQ_TOOLS,
            tool_choice = "auto",
            temperature = 0.3,
            max_tokens  = 512,
        )

        assistant_msg = response.choices[0].message

        # ── Execute any tool calls the model requested ────────────────────────
        if assistant_msg.tool_calls:
            # Append the assistant turn with its tool-call requests
            self._msgs.append({
                "role":       "assistant",
                "content":    assistant_msg.content or "",
                "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls],
            })
            # Run each tool and append its result
            for tc in assistant_msg.tool_calls:
                args   = json.loads(tc.function.arguments)
                result = self._run_tool(tc.function.name, args)
                log.info("tool=%s args=%s result=%s", tc.function.name, args, result)
                self._msgs.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result),
                })
            # Ask the model to produce a natural-language reply now that it has results
            follow_up = self._client.chat.completions.create(
                model      = _GROQ_MODEL,
                messages   = [{"role": "system", "content": _GROQ_SYSTEM_PROMPT}] + self._msgs,
                temperature= 0.3,
                max_tokens = 512,
            )
            final_text = follow_up.choices[0].message.content
        else:
            final_text = assistant_msg.content

        self._msgs.append({"role": "assistant", "content": final_text})

        turn = {
            "turn":   len(self.history) + 1,
            "user":   msg,
            "bot":    final_text,
            "intent": "llm",
            "conf":   1.0,
            "state":  "LLM",
        }
        self.history.append(turn)
        log.info("groq turn=%d", len(self.history))
        return final_text, "llm", 1.0

    def reset(self):
        self._msgs   = []
        self.history = []

    # ── Tool handlers — execute against the shared mock DB ───────────────────
    def _run_tool(self, name: str, args: dict) -> dict:
        if name == "get_booking_status":
            bk_id   = args.get("booking_id", "").upper().replace(" ", "-")
            booking = _MOCK_DB.get(bk_id)
            if booking:
                return {
                    "found":        True,
                    "booking_id":   bk_id,
                    "service":      booking.service,
                    "therapist":    booking.therapist,
                    "scheduled_at": booking.scheduled_at.strftime("%d %b %Y %I:%M %p"),
                    "status":       booking.status,
                }
            return {"found": False, "error": f"Booking {bk_id} not found. Ask the customer to double-check."}

        if name == "create_booking":
            import random
            bk_id     = _next_booking_id()
            therapist = random.choice(_THERAPISTS)
            dt        = (dateparser.parse(args.get("datetime_str", ""),
                          settings={"PREFER_DATES_FROM": "future"})
                         or datetime.now())
            new_bk    = _Booking(bk_id, args.get("service", "Service"), dt,
                                 status="confirmed", therapist=therapist)
            _MOCK_DB[bk_id] = new_bk
            return {
                "success":      True,
                "booking_id":   bk_id,
                "service":      new_bk.service,
                "scheduled_at": dt.strftime("%d %b %Y %I:%M %p"),
                "therapist":    therapist,
                "status":       "confirmed",
            }

        if name == "reschedule_booking":
            bk_id   = args.get("booking_id", "").upper().replace(" ", "-")
            booking = _MOCK_DB.get(bk_id)
            if booking:
                old_time        = booking.scheduled_at.strftime("%d %b %Y %I:%M %p")
                new_dt          = (dateparser.parse(args.get("new_datetime", ""),
                                   settings={"PREFER_DATES_FROM": "future"})
                                   or datetime.now())
                booking.scheduled_at = new_dt
                booking.status       = "rescheduled"
                return {
                    "success":    True,
                    "booking_id": bk_id,
                    "old_time":   old_time,
                    "new_time":   new_dt.strftime("%d %b %Y %I:%M %p"),
                    "therapist":  booking.therapist,
                }
            return {"success": False, "error": f"Booking {bk_id} not found."}

        if name == "cancel_booking":
            bk_id   = args.get("booking_id", "").upper().replace(" ", "-")
            booking = _MOCK_DB.get(bk_id)
            if booking:
                booking.status = "cancelled"
                return {
                    "success":    True,
                    "booking_id": bk_id,
                    "service":    booking.service,
                }
            return {"success": False, "error": f"Booking {bk_id} not found."}

        if name == "get_availability":
            day = args.get("day", "").strip()
            if day:
                for d, slots in _AVAILABILITY.items():
                    if d.lower() == day.lower():
                        return {"day": d, "slots": slots}
                return {"error": f"No data for '{day}'. Valid days: {list(_AVAILABILITY.keys())}"}
            return {"schedule": {d: s for d, s in _AVAILABILITY.items()}}

        return {"error": f"Unknown tool: {name}"}


# ── Session store — session_id → _BlysChat or _BlysChatGroq ─────────────────
# Keyed by session_id string; each session holds its own conversation state.
_chat_sessions: dict[str, object] = {}


class ChatRequest(BaseModel):
    session_id:   str
    user_message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id":   "user-abc-123",
                "user_message": "Can I reschedule my booking?",
            }
        }
    }


class ChatResponse(BaseModel):
    session_id:        str
    bot_response:      str
    intent:            str
    intent_confidence: float
    current_state:     str
    turn_number:       int
    conversation_history: list[dict]  # all turns so far in this session


@app.post(
    "/chatbot",
    response_model=ChatResponse,
    summary="Multi-turn NLP chatbot for booking queries",
    tags=["Chatbot"],
)
def chatbot(req: ChatRequest):
    """
    Processes a customer message and returns an AI-generated response.

    - Classifies intent using the trained TF-IDF + Logistic Regression model
    - Extracts entities (booking ID, date/time, service type) with spaCy + dateparser
    - Manages multi-turn dialogue state with an FSM
    - Executes booking actions (reschedule, cancel) against a mock database
    - Each `session_id` maintains independent conversation state
    - When GROQ_API_KEY is set the LLM-powered backend is used automatically
    """
    # Create a new session if this is the first message
    if req.session_id not in _chat_sessions:
        if _groq_client is not None:
            _chat_sessions[req.session_id] = _BlysChatGroq(_groq_client)
        elif chat_artifact is not None:
            _chat_sessions[req.session_id] = _BlysChat(chat_artifact)
        else:
            raise HTTPException(
                503,
                "No chat backend available — set GROQ_API_KEY or run Section 3 notebook first",
            )

    bot                     = _chat_sessions[req.session_id]
    response, intent, conf  = bot.chat(req.user_message)

    # .state is an enum for _BlysChat, a plain string "LLM" for _BlysChatGroq
    state_name = bot.state.name if hasattr(bot.state, "name") else str(bot.state)

    return ChatResponse(
        session_id           = req.session_id,
        bot_response         = response,
        intent               = intent,
        intent_confidence    = round(conf, 4),
        current_state        = state_name,
        turn_number          = len(bot.history),
        conversation_history = bot.history,
    )


@app.get(
    "/session/{session_id}/history",
    summary="Get full conversation history for a session",
    tags=["Chatbot"],
)
def get_history(session_id: str):
    """Returns every turn (user message + bot reply + intent) for the session."""
    if session_id not in _chat_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    bot = _chat_sessions[session_id]
    state_name = bot.state.name if hasattr(bot.state, "name") else str(bot.state)
    return {
        "session_id":    session_id,
        "turn_count":    len(bot.history),
        "current_state": state_name,
        "history":       bot.history,
    }


@app.delete(
    "/session/{session_id}",
    summary="Reset a chatbot session",
    tags=["Chatbot"],
)
def reset_session(session_id: str):
    """Clears the conversation state for the given session_id."""
    if session_id in _chat_sessions:
        turns = len(_chat_sessions[session_id].history)
        _chat_sessions[session_id].reset()
        return {"detail": f"Session '{session_id}' reset.", "turns_cleared": turns}
    return {"detail": f"Session '{session_id}' not found (nothing to reset)."}


# ===========================================================================
# Health check
# ===========================================================================
@app.get("/health", summary="Liveness check", tags=["System"])
def health():
    """Returns the load status of both models and which chat backend is active."""
    return {
        "status":                 "ok",
        "recommendation_model":   rec_artifact  is not None,
        "chatbot_model":          chat_artifact is not None,
        "groq_enabled":           _groq_client  is not None,
        "chat_backend":           "groq_llm" if _groq_client is not None else "local_fsm",
        "customer_data_loaded":   _customer_df  is not None,
        "active_chat_sessions":   len(_chat_sessions),
    }


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
