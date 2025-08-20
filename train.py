# train.py
import os
import json
from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from textblob import TextBlob  # ✅ for sentiment
from fastapi import FastAPI

# ---------------- FASTAPI APP ----------------
app = FastAPI()

# ---------------- LOAD ENV ----------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    MONGO_USERNAME = os.getenv("MONGO_USERNAME")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")

    if not all([MONGO_USERNAME, MONGO_PASSWORD, MONGO_CLUSTER]):
        raise ValueError("❌ Missing MongoDB environment variables")

    MONGO_URI = (
        f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}"
        f"@{MONGO_CLUSTER}/?retryWrites=true&w=majority"
    )

# ---------------- GLOBAL VARS ----------------
client_mongo: AsyncIOMotorClient = None
db = None
whatsapp_collection = None

@app.on_event("startup")
async def startup_db_client():
    global client_mongo, db, whatsapp_collection
    client_mongo = AsyncIOMotorClient(MONGO_URI)
    db = client_mongo["chatbot"]
    whatsapp_collection = db["whatsapp_messages"]
    print("✅ MongoDB connection established in train.py")

@app.on_event("shutdown")
async def shutdown_db_client():
    client_mongo.close()
    print("❌ MongoDB connection closed in train.py")

# ---------------- OPENAI CLIENT ----------------
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- SENTIMENT ----------------
def detect_sentiment(message: str):
    polarity = TextBlob(message).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# ---------------- JSONL CREATION ----------------
def prepare_jsonl_from_db(messages, jsonl_file: str):
    """
    Convert stored DB messages into JSONL format for fine-tuning.
    """
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
                {"role": "assistant", "content": reply},
            ]
        }
        data.append(record)

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return jsonl_file

# ---------------- START TRAINING ----------------
async def start_training(doc_id: str):
    """
    Start a fine-tuning job for the given document.
    Returns job_id immediately.
    """
    doc = await whatsapp_collection.find_one({"_id": ObjectId(doc_id)})
    if not doc or "messages" not in doc:
        return {"error": "No training data found"}

    # Write messages to JSONL
    jsonl_file = "whatsapp_finetune.jsonl"
    prepare_jsonl_from_db(doc["messages"], jsonl_file)

    # Upload training file
    with open(jsonl_file, "rb") as f:
        uploaded = client_openai.files.create(file=f, purpose="fine-tune")

    # Start fine-tune
    job = client_openai.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model="gpt-4.1-nano-2025-04-14",  # ✅ latest requested model
    )

    # Save job_id to DB for later polling
    await whatsapp_collection.update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {"fine_tune_job_id": job.id}},
    )

    return {"status": "started", "job_id": job.id}

# ---------------- CHECK STATUS ----------------
async def check_training_status(job_id: str):
    """
    Poll the fine-tuning job status.
    If finished, update DB with fine-tuned model ID.
    """
    job = client_openai.fine_tuning.jobs.retrieve(job_id)
    status = job.status

    if status == "succeeded":
        model_id = job.fine_tuned_model
        await whatsapp_collection.update_one(
            {"fine_tune_job_id": job_id},
            {"$set": {"FINE_TUNED_MODEL_ID": model_id}},
        )
        return {"status": "succeeded", "model_id": model_id}

    elif status in ["failed", "cancelled"]:
        return {"status": status}

    return {"status": status}
