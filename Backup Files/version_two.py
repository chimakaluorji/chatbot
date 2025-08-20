
# api/chat.py (Vercel Python function)
import os
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob

# FastAPI instance
app = FastAPI()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID")

# In-memory short-term conversation storage
conversation_history = []  # [{role, content}, ...]

# Request body model
class QueryRequest(BaseModel):
    message: str

# Sentiment detection helper
def detect_sentiment(message):
    polarity = TextBlob(message).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    return "neutral"

@app.post("/chat")
def chat_endpoint(req: QueryRequest):
    if not FINE_TUNED_MODEL_ID:
        return {"error": "No trained model available"}

    # Step 1 — Detect sentiment
    sentiment = detect_sentiment(req.message)

    # Step 2 — Build system prompt
    system_prompt = (
        f"You are Chima’s WhatsApp auto-reply bot.\n"
        f"Incoming message sentiment: {sentiment}.\n"
        "You have short-term conversation memory below.\n"
        "Use it ONLY for tone, style, and sentiment — DO NOT copy exact phrases.\n"
        "Always produce new, natural, human-like responses.\n"
        "Apply commonsense reasoning for realistic replies.\n"
        "If positive, be warm and friendly.\n"
        "If negative, be empathetic and understanding.\n"
        "If neutral, be polite and casual."
    )

    # Step 3 — Update conversation memory
    conversation_history.append({"role": "user", "content": req.message})
    if len(conversation_history) > 10:  # keep last 5 exchanges
        conversation_history[:] = conversation_history[-10:]

    # Step 4 — Prepare messages for API
    messages_for_api = [{"role": "system", "content": system_prompt}] + conversation_history

    # Step 5 — Get model reply
    resp = client.chat.completions.create(
        model=FINE_TUNED_MODEL_ID,
        messages=messages_for_api,
        max_completion_tokens=200
    )
    bot_reply = resp.choices[0].message.content

    # Step 6 — Add bot reply to memory
    conversation_history.append({"role": "assistant", "content": bot_reply})

    # Step 7 — Return response
    return {
        "status": "success",
        "sentiment": sentiment,
        "user_message": req.message,
        "bot_reply": bot_reply
    }


