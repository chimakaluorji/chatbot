# train.py
import os
import json
from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from textblob import TextBlob  # ✅ for sentiment

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- MONGO SETUP ----------------
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

_client = AsyncIOMotorClient(MONGO_URI)
db = _client["chatbot"]
whatsapp_collection = db["whatsapp_messages"]

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

# ---------------- MODERATION ----------------
def is_safe_text(text: str) -> bool:
    """
    Check if text passes OpenAI moderation.
    Returns True if safe, False if flagged.
    """
    if not text.strip():
        return True  # skip empty
    try:
        resp = client_openai.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        flagged = resp.results[0].flagged
        return not flagged
    except Exception as e:
        print(f"⚠️ Moderation API error: {e}")
        return False  # reject on error

# ---------------- JSONL CREATION ----------------
def prepare_jsonl_from_db(messages, jsonl_file: str):
    """
    Convert stored DB messages into JSONL format for fine-tuning.
    Expects flat list of {role, content} objects.
    Applies moderation filtering.
    """
    data = []

    for i in range(0, len(messages), 3):
        chunk = messages[i:i+3]
        if len(chunk) < 3:
            continue

        system_msg, user_msg, assistant_msg = chunk

        # ✅ Moderation filter
        if not (is_safe_text(user_msg["content"]) and is_safe_text(assistant_msg["content"])):
            print(f"⛔ Skipping unsafe conversation: {user_msg['content']} / {assistant_msg['content']}")
            continue

        sentiment = detect_sentiment(user_msg["content"])
        system_prompt = (
            "You are Chima’s WhatsApp auto-reply bot. "
            "Mimic his tone, style, and way of speaking based on chat history. "
            f"The incoming message is {sentiment}."
        )

        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg["content"]},
                {"role": "assistant", "content": assistant_msg["content"]},
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
    Saves job_id to DB.
    """
    doc = await whatsapp_collection.find_one({"_id": ObjectId(doc_id)})
    if not doc or "messages" not in doc:
        return {"error": "No training data found"}

    jsonl_file = "whatsapp_finetune.jsonl"
    prepare_jsonl_from_db(doc["messages"], jsonl_file)

    with open(jsonl_file, "rb") as f:
        uploaded = client_openai.files.create(file=f, purpose="fine-tune")

    job = client_openai.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model="gpt-4.1-nano-2025-04-14"
    )

    await whatsapp_collection.update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {"fine_tune_job_id": job.id}},
    )

    print(f"✅ Saved fine_tune_job_id={job.id} for doc_id={doc_id}")
    return {"status": "started", "job_id": job.id}

# ---------------- CHECK STATUS ----------------
async def check_training_status(job_id: str):
    """
    Check status of a fine-tune job via OpenAI.
    """
    job = client_openai.fine_tuning.jobs.retrieve(job_id)
    status = job.status

    if status == "pending":
        return {
            "status": "queued",
            "queue_position": getattr(job, "queue_position", None)
        }
    elif status == "succeeded":
        model_id = job.fine_tuned_model
        await whatsapp_collection.update_one(
            {"fine_tune_job_id": job_id},
            {"$set": {"FINE_TUNED_MODEL_ID": model_id}},
        )
        return {"status": "succeeded", "model_id": model_id}
    elif status in ["failed", "cancelled"]:
        return {"status": status}
    else:
        return {"status": "running"}
