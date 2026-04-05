# Blys AI Engineer Technical Assessment

A full-stack AI system for Blys — an on-demand mobile massage and beauty therapy platform. The project covers customer behavior analysis, a service recommendation engine, an NLP chatbot, and a REST API to serve both models.

---

## Project Overview

| Section | Description | Output |
|---------|-------------|--------|
| 1 | Customer behavior analysis, sentiment scoring, K-Means segmentation | `reports/customer_analysis.md` |
| 2 | Hybrid recommendation model (GBM + NMF collaborative filtering) | `models/recommendation_model.pkl` |
| 3 | NLP intent classifier and chatbot with Groq LLM integration | `models/chatbot_model.pkl` |
| 4 | FastAPI REST API serving both models with session-based chat | `api.py` |

---

## Project Structure

```
BLYS_Task/
├── data/
│   ├── customer_data.csv               # 20,000 customer records with booking behaviour and reviews
│   ├── transactions.csv                # 150,000 booking transactions generated from customer_data.csv — used by Section 2 to build the collaborative filtering matrix
│   └── intent_training_data.jsonl      # Chatbot intent training examples
├── models/
│   ├── recommendation_model.pkl        # Trained recommendation model
│   └── chatbot_model.pkl               # Trained chatbot classifier
├── reports/
│   ├── customer_analysis.md            # Section 1 summary report
│   ├── customer_segments.png
│   ├── elbow_silhouette.png
│   ├── sentiment_distribution.png
│   ├── insights.png
│   ├── feature_importance.png
│   ├── nmf_heatmap.png
│   ├── nmf_components.png
│   └── recommendation_comparison.png
├── notebooks/
│   ├── section1_customer_analysis.ipynb
│   ├── section2_recommendation_model.ipynb
│   └── section3_chatbot.ipynb
├── api.py                              # FastAPI application
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Surazz2022/blys-api.git
cd blys-api
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 5. Set your Groq API key (required for chatbot)

The chatbot requires a Groq API key to function properly. Without it the system falls back to a local intent classifier which has limited accuracy and does not understand natural language well. The Groq key enables Llama 3.3 70B, which understands full sentences, maintains context across turns, and executes real booking actions.

```bash
# Windows
$env:GROQ_API_KEY = "your_groq_api_key"

# macOS / Linux
export GROQ_API_KEY="your_groq_api_key"
```

Get a free API key at [console.groq.com](https://console.groq.com). The free tier is sufficient to run this project.

When deploying to Render, add `GROQ_API_KEY` as an environment variable in the Render dashboard under the Environment tab. The server will not load the heavy local models when Groq is active, which keeps memory usage within the free tier limit.

**Fallback behaviour when Groq is not available:**

If `GROQ_API_KEY` is not set, the chatbot automatically falls back to a local pipeline:
- Intent is classified using a Logistic Regression model trained on sentence embeddings
- A finite state machine (FSM) manages the conversation flow across 13 dialogue states
- Entity extraction (booking ID, date/time, service name) is handled by spaCy and dateparser
- Booking actions (reschedule, cancel, create) are still executed against the mock database

The fallback works for straightforward requests but is less accurate with natural or ambiguous language compared to the Groq LLM backend. The `/health` endpoint shows `"chat_backend": "local_fsm"` when the fallback is active.

---

## Running the Notebooks

Run the notebooks in order. Each one saves its output for the next step.

```bash
jupyter notebook notebooks/section1_customer_analysis.ipynb
jupyter notebook notebooks/section2_recommendation_model.ipynb
jupyter notebook notebooks/section3_chatbot.ipynb
```

| Notebook | What it does | Output |
|----------|-------------|--------|
| `section1_customer_analysis.ipynb` | EDA, VADER sentiment, K-Means clustering, insights | `reports/customer_analysis.md` + plots |
| `section2_recommendation_model.ipynb` | GBM content-based + NMF collaborative filtering | `models/recommendation_model.pkl` |
| `section3_chatbot.ipynb` | Intent classifier training, FSM chatbot demo | `models/chatbot_model.pkl` |

---

## Starting the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## API Endpoints

### GET /health

Returns the current status of all loaded models and which chat backend is active.

```json
{
  "status": "ok",
  "recommendation_model": true,
  "chatbot_model": true,
  "groq_enabled": true,
  "chat_backend": "groq_llm",
  "customer_data_loaded": true,
  "active_chat_sessions": 0
}
```

---

### POST /recommend

Returns a ranked list of recommended services for a customer based on their booking history and behavior.

**How it works:** Combines a Gradient Boosting classifier (content-based) with NMF collaborative filtering. Known customers get a 50/50 hybrid score. New customers get content-based scores only.

**Request:**
```json
{
  "customer_id": 1001
}
```

**Response:**
```json
{
  "customer_id": 1001,
  "recommended_services": ["Massage", "Wellness Package", "Facial", "Body Scrub", "Hair Spa"]
}
```

---

### POST /chatbot

Processes a natural language message and returns an AI response. Each session maintains its own conversation history.

When `GROQ_API_KEY` is set, the chatbot uses Llama 3.3 70B via Groq with real tool calls to execute booking actions. Without the key, it falls back to a local intent classifier with a finite state machine.

**Supported actions:**
- Check booking status
- Reschedule a booking
- Cancel a booking
- Create a new booking
- Check availability

**Request:**
```json
{
  "session_id": "user-123",
  "user_message": "I want to reschedule BK-001 to Friday at 3pm"
}
```

**Response with Groq active (`chat_backend: groq_llm`):**

The LLM understands the full message in one shot, calls the reschedule tool, and replies naturally.

```json
{
  "session_id": "user-123",
  "bot_response": "Your booking BK-001 has been rescheduled to Friday at 3:00 PM. Your therapist James is confirmed for the new time.",
  "intent": "llm",
  "intent_confidence": 1.0,
  "current_state": "LLM",
  "turn_number": 1,
  "conversation_history": [
    {
      "turn": 1,
      "user": "I want to reschedule BK-001 to Friday at 3pm",
      "bot": "Your booking BK-001 has been rescheduled to Friday at 3:00 PM. Your therapist James is confirmed for the new time.",
      "intent": "llm",
      "conf": 1.0,
      "state": "LLM"
    }
  ]
}
```

**Response with fallback active (`chat_backend: local_fsm`):**

The FSM extracts the booking ID and date separately, confirms before acting, and requires multiple turns.

Turn 1 — user sends the same message:
```json
{
  "session_id": "user-123",
  "bot_response": "Got it. You want to reschedule BK-001 to Friday at 3:00 PM. Shall I go ahead?",
  "intent": "reschedule",
  "intent_confidence": 0.87,
  "current_state": "RESCHEDULE_CONFIRM",
  "turn_number": 1
}
```

Turn 2 — user confirms:
```json
{
  "session_id": "user-123",
  "bot_response": "Done! Booking BK-001 has been rescheduled to Friday at 3:00 PM.",
  "intent": "confirmation_yes",
  "intent_confidence": 0.96,
  "current_state": "DONE",
  "turn_number": 2
}
```

---

### GET /session/{session_id}/history

Returns the full conversation history for a session.

---

### DELETE /session/{session_id}

Resets a chat session, clearing all conversation history.

---

## Deployment

The API is live and deployed on Render:

| Endpoint | URL |
|----------|-----|
| Interactive API Docs | https://blys-api-2.onrender.com/docs |
| Health Check | https://blys-api-2.onrender.com/health |
| Recommend | https://blys-api-2.onrender.com/recommend |
| Chatbot | https://blys-api-2.onrender.com/chatbot |

> Note: The free tier on Render spins down after 15 minutes of inactivity. The first request after a period of inactivity may take 30-60 seconds to respond while the server wakes up.

---

### Deploy to Render (steps)

The project includes a `Dockerfile` and `render.yaml` for deployment on [Render](https://render.com).

1. Push the repository to GitHub
2. Go to [render.com](https://render.com) and create a new Web Service
3. Connect your GitHub repository — Render detects `render.yaml` automatically
4. Go to the **Environment** tab and add `GROQ_API_KEY` as an environment variable
5. Click **Deploy** — the build takes around 5-8 minutes on first run

Setting `GROQ_API_KEY` on Render is important for two reasons:
- It enables the Groq LLM backend for accurate natural language understanding
- It prevents the heavy local sentence transformer model from loading, keeping memory within the free tier 512MB limit

### Docker (local)

```bash
docker build -t blys-api .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key blys-api
```

---

## Section Details

### Section 1 - Customer Analysis

- Handles missing values and normalizes features with MinMaxScaler
- Applies VADER sentiment analysis to customer reviews
- Uses K-Means clustering (k=3) with elbow and silhouette analysis to segment customers into: High-Value, Casual Browsers, and At-Risk
- Identifies 863 high-value customers (4.3%) and 2,401 churn-risk customers (12%) with actionable strategies

### Section 2 - Recommendation Model

- `transactions.csv` is generated by `data/generate_transactions.py` which expands `customer_data.csv` into 150,000 individual booking records — each customer has multiple bookings across different services, making collaborative filtering meaningful
- Content-based model: Gradient Boosting on 5 customer features (booking frequency, average spend, days inactive, sentiment, preferred service)
- Collaborative filtering: NMF on a customer-service booking matrix evaluated with leave-one-out
- Hybrid inference: 50% content-based + 50% collaborative for known customers

### Section 3 - NLP Chatbot

- Intent classifier: Logistic Regression on sentence embeddings (all-MiniLM-L6-v2) with 13 intent classes
- Groq LLM backend: Llama 3.3 70B with 5 tool calls for real booking actions
- Session persistence: each session maintains full conversation history in memory
- Fallback: local FSM-based dialogue manager when Groq is unavailable

### Section 4 - REST API

- FastAPI with automatic OpenAPI docs
- Groq LLM chat auto-selected when `GROQ_API_KEY` is set
- Session-based multi-turn conversations
- Deployed on Render with Docker
