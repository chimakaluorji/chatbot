# train.py
import os
import json
from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from textblob import TextBlob  # ✅ for sentiment

load_dotenv()

# ---------------- MONGO CONNECTION ----------------
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    MONGO_USERNAME = os.getenv("MONGO_USERNAME")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
    if not all([MONGO_USERNAME, MONGO_PASSWORD, MONGO_CLUSTER]):
        raise ValueError("❌ Missing MongoDB environment variables")
    MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_CLUSTER}/?retryWrites=true&w=majority"

client_mongo = AsyncIOMotorClient(MONGO_URI)
db = client_mongo["chatbot"]
whatsapp_collection = db["whatsapp_messages"]

# ---------------- OPENAI CLIENT ----------------
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- SENTIMENT ----------------
def detect_sentiment(message: str):
    polarity = TextBlob(message).sentiment.polarity
    return "positive" if polarity > 0.2 else "negative" if polarity < -0.2 else "neutral"

# ---------------- JSONL CREATION ----------------
def prepare_jsonl_from_db(messages, jsonl_file: str):
    data = []
    for msg in messages:  # expects flattened [system, user, assistant]
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

# ---------------- START TRAINING ----------------
async def start_training(doc_id: str):
    doc = await whatsapp_collection.find_one({"_id": ObjectId(doc_id)})
    if not doc or "messages" not in doc:
        return {"error": "No training data found"}

    jsonl_file = "whatsapp_finetune.jsonl"
    prepare_jsonl_from_db(doc["messages"], jsonl_file)

    # ✅ Upload training file
    with open(jsonl_file, "rb") as f:
        uploaded = client_openai.files.create(file=f, purpose="fine-tune")

    # ✅ Start fine-tuning
    job = client_openai.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model="gpt-4.1-nano-2025-04-14"
    )

    # ✅ Save job_id in MongoDB
    await whatsapp_collection.update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {"fine_tune_job_id": job.id}}
    )

    return {"status": "started", "job_id": job.id}

# ---------------- CHECK STATUS ----------------
async def check_training_status(job_id: str):
    job = client_openai.fine_tuning.jobs.retrieve(job_id)
    status = job.status
    if status == "succeeded":
        model_id = job.fine_tuned_model
        await whatsapp_collection.update_one(
            {"fine_tune_job_id": job_id},
            {"$set": {"FINE_TUNED_MODEL_ID": model_id}}
        )
        return {"status": "succeeded", "model_id": model_id}
    return {"status": status}
