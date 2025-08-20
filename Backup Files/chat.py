# api/chat.py (Vercel Python function)
import os
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID")


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

    resp = client.chat.completions.create(
        model=FINE_TUNED_MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ],
        max_completion_tokens=200
    )

    return {
        "status": "success",
        "sentiment": sentiment,
        "user_message": req.message,
        "bot_reply":resp.choices[0].message.content
    }
