# train.py
import os
import re
import json
import time
from bson import ObjectId
from textblob import TextBlob
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from fastapi import FastAPI, BackgroundTasks

# ---------------- LOAD ENV ----------------
load_dotenv()
app = FastAPI()

# ---------------- MONGO ----------------
MONGO_USERNAME = "ecomUser"
MONGO_PASSWORD = "Chimakalu0rji"
MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.zd4jdqg.mongodb.net/?retryWrites=true&w=majority"
client_mongo = AsyncIOMotorClient(MONGO_URI)
db = client_mongo["chatbot"]
whatsapp_collection = db["whatsapp_messages"]

# ---------------- OPENAI ----------------
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- SENTIMENT ----------------
def detect_sentiment(message):
    polarity = TextBlob(message).sentiment.polarity
    return "positive" if polarity > 0.2 else "negative" if polarity < -0.2 else "neutral"

# ---------------- JSONL CREATION ----------------
def prepare_jsonl_from_db(messages, jsonl_file):
    data = []
    for msg in messages:
        incoming = msg[1]["content"] if len(msg) > 1 else ""
        reply = msg[2]["content"] if len(msg) > 2 else ""
        sentiment = detect_sentiment(incoming)

        system_prompt = (
            "You are Chima’s WhatsApp auto-reply bot. "
            "Mimic his tone, style, and way of speaking based on chat history. "
            f" The incoming message is {sentiment}."
        )

        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": incoming},
                {"role": "assistant", "content": reply}
            ]
        }
        data.append(record)

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return jsonl_file

# ---------------- TRAINING ----------------
async def start_training(doc_id: str):
    jsonl_file = "whatsapp_finetune.jsonl"

    if os.path.exists(jsonl_file):
        os.remove(jsonl_file)

    doc = await whatsapp_collection.find_one({"_id": ObjectId(doc_id)})
    if not doc or "messages" not in doc or not doc["messages"]:
        return {"error": "No messages to train on"}

    prepare_jsonl_from_db(doc["messages"], jsonl_file)

    with open(jsonl_file, "rb") as f:
        uploaded = client_openai.files.create(file=f, purpose="fine-tune")

    job = client_openai.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model="gpt-4.1-nano-2025-04-14"
    )
    print(f"🚀 Fine-tune started! Job ID: {job.id}")

    while True:
        status_data = client_openai.fine_tuning.jobs.retrieve(job.id)
        status = status_data.status
        print(f"⏳ Status: {status}")
        if status in ["succeeded", "failed", "cancelled"]:
            break
        time.sleep(10)  # Poll every 10s for Vercel timeout safety

    if os.path.exists(jsonl_file):
        os.remove(jsonl_file)

    if status_data.status == "succeeded":
        model_id = status_data.fine_tuned_model
        await whatsapp_collection.update_one({"_id": ObjectId(doc_id)}, {"$set": {"FINE_TUNED_MODEL_ID": model_id}})
        return {"status": "success", "model_id": model_id}
    else:
        return {"status": "failed", "job_status": status}

@app.post("/train/{doc_id}")
async def train_endpoint(doc_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_training, doc_id)
    return {"status": "started", "job_id": doc_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("train:app", host="127.0.0.1", port=8002, reload=True)
