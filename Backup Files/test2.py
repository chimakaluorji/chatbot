import re
import json
import os
from textblob import TextBlob
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

file_path = "whatsapp.txt"
index_name = "whatsapp-chat"
memory_file = "conversation_memory.json"

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    incoming: str
    reply: str

# ----------------------------
# Detect best free-tier region
# ----------------------------
def get_free_region():
    return ("aws", "us-east-1")

# ----------------------------
# Read WhatsApp Export
# ----------------------------
def read_whatsapp_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()

# ----------------------------
# Sentiment Detection
# ----------------------------
def detect_sentiment(message):
    polarity = TextBlob(message).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    return "neutral"

# ----------------------------
# Parse into message pairs
# ----------------------------
def parse_messages(lines, your_name="CHIMA KALU-ORJI"):
    messages = []
    pattern = r"^\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}(?:\s| )?(?:am|pm) - (.*?): (.*)$"
    for i in range(len(lines) - 1):
        match = re.match(pattern, lines[i], flags=re.IGNORECASE)
        next_match = re.match(pattern, lines[i + 1], flags=re.IGNORECASE)
        if match and next_match:
            sender, text = match.groups()
            next_sender, next_text = next_match.groups()
            if sender != your_name and next_sender == your_name:
                sentiment = detect_sentiment(text)
                messages.append({
                    "incoming": text.strip(),
                    "reply": next_text.strip(),
                    "sentiment": sentiment
                })
    return messages

# ----------------------------
# Upload parsed messages to Pinecone
# ----------------------------
def upload_to_pinecone(messages):
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        cloud, region = get_free_region()
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
        print(f"📦 Created index '{index_name}' in {cloud}:{region}")

    index = pc.Index(index_name)
    vectors = []
    for i, msg in enumerate(messages):
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=msg["incoming"]
        ).data[0].embedding
        vectors.append((
            f"msg_{i}_{datetime.utcnow().timestamp()}",
            embedding,
            {
                "incoming": msg["incoming"],
                "reply": msg["reply"],
                "sentiment": msg["sentiment"],
                "timestamp": datetime.utcnow().isoformat()
            }
        ))

    index.upsert(vectors)
    print(f"✅ Uploaded {len(vectors)} messages to Pinecone index '{index_name}'.")

# ----------------------------
# Query Pinecone for similar messages
# ----------------------------
def query_pinecone(user_message, top_k=3):
    index = pc.Index(index_name)
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_message
    ).data[0].embedding
    return index.query(vector=embedding, top_k=top_k, include_metadata=True).matches

# ----------------------------
# Memory Persistence
# ----------------------------
def load_memory():
    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory():
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=2)

# ----------------------------
# Short-term + Long-term Memory
# ----------------------------
conversation_history = load_memory()

# ----------------------------
# Generate GPT Reply with Combined Retrieval + Memory
# ----------------------------
def generate_reply(user_message):
    global conversation_history
    sentiment = detect_sentiment(user_message)

    # 1. Get top matches from Pinecone
    matches = query_pinecone(user_message)
    retrieved_context = "\n".join([
        f"Incoming: {m.metadata['incoming']} -> Reply: {m.metadata['reply']}"
        for m in matches
    ])

    # 2. Add current user message to short-term memory
    conversation_history.append({"role": "user", "content": user_message})
    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]

    # 3. Build system prompt with both retrieval + memory
    system_prompt = (
        f"You are Chima’s WhatsApp bot.\n"
        f"Incoming message sentiment: {sentiment}.\n"
        "You have access to:\n"
        "- Past conversation context (short-term memory)\n"
        "- Retrieved similar historical chats from long-term memory\n\n"
        "Guidelines:\n"
        "1. Use retrieved chats ONLY to match tone, style, and sentiment.\n"
        "2. DO NOT copy any reply exactly.\n"
        "3. Create a brand new, natural, human-like reply in Chima's style.\n"
        "4. Apply commonsense reasoning so the reply fits the situation.\n"
        "5. If positive → warm and friendly. If negative → empathetic. If neutral → casual.\n\n"
        f"Long-term memory examples:\n{retrieved_context}\n\n"
        "Short-term conversation so far:\n" +
        "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history])
    )

    # 4. Call GPT
    messages_for_api = [{"role": "system", "content": system_prompt}] + conversation_history
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages_for_api,
        max_completion_tokens=200
    )

    bot_reply = resp.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": bot_reply})
    save_memory()

    return bot_reply

# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return {"user_message": request.message, "bot_reply": generate_reply(request.message)}

@app.post("/train")
def train_endpoint(data: TrainRequest):
    msg = {
        "incoming": data.incoming.strip(),
        "reply": data.reply.strip(),
        "sentiment": detect_sentiment(data.incoming)
    }
    upload_to_pinecone([msg])
    return {"status": "success", "message": "New training data added to Pinecone."}

# ----------------------------
# Startup
# ----------------------------
@app.on_event("startup")
def startup_event():
    print("📂 Loading and parsing WhatsApp history...")
    parsed = parse_messages(read_whatsapp_txt(file_path), your_name="CHIMA KALU-ORJI")
    print(f"✅ Parsed {len(parsed)} message pairs.")
    upload_to_pinecone(parsed)

# ----------------------------
# Run with uvicorn
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
