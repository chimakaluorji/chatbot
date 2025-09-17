# train.py
import os
import json
import io
from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from textblob import TextBlob

load_dotenv()

# ---------------- MONGO ----------------
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

# ---------------- OPENAI ----------------
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- SENTIMENT ----------------
def detect_sentiment(message: str) -> str:
    polarity = TextBlob(message).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    return "neutral"

# ---------------- MODERATION ----------------
def is_safe_text(text: str) -> bool:
    """Check if text passes OpenAI moderation."""
    if not text.strip():
        return True
    try:
        resp = client_openai.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        return not resp.results[0].flagged
    except Exception as e:
        print(f"⚠️ Moderation check failed: {e}")
        return True  # fallback: allow text

# ---------------- JSONL IN MEMORY ----------------
def prepare_jsonl_from_db(messages):
    """Convert DB messages into JSONL string for fine-tuning."""
    data = []

    # Case 1: flat structure
    if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
        for i in range(0, len(messages), 3):
            chunk = messages[i:i+3]
            if len(chunk) < 3:
                continue
            _, user_msg, assistant_msg = chunk

            if not user_msg.get("content") or not assistant_msg.get("content"):
                continue
            if not (is_safe_text(user_msg["content"]) and is_safe_text(assistant_msg["content"])):
                continue

            sentiment = detect_sentiment(user_msg["content"])
            system_prompt = (
                "You are Chima’s WhatsApp auto-reply bot. "
                "Mimic his tone and style based on chat history. "
                f"Incoming message sentiment: {sentiment}."
            )

            data.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg["content"]},
                    {"role": "assistant", "content": assistant_msg["content"]},
                ]
            })

    # Case 2: nested structure
    elif isinstance(messages, list) and all(isinstance(m, list) for m in messages):
        for group in messages:
            if len(group) < 3:
                continue
            _, user_msg, assistant_msg = group

            if not user_msg.get("content") or not assistant_msg.get("content"):
                continue
            """ if not (is_safe_text(user_msg["content"]) and is_safe_text(assistant_msg["content"])):
                continue """

            sentiment = detect_sentiment(user_msg["content"])
            system_prompt = (
                "You are Chima’s WhatsApp auto-reply bot. "
                "Mimic his tone and style based on chat history. "
                f"Incoming message sentiment: {sentiment}."
            )

            data.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg["content"]},
                    {"role": "assistant", "content": assistant_msg["content"]},
                ]
            })

    if not data:
        print("⛔ No valid training examples found")
        return ""

    return "\n".join(json.dumps(record, ensure_ascii=False) for record in data)

# ---------------- START TRAINING ----------------
async def start_training(doc_id: str):
    """Start fine-tuning and immediately store job_id in DB."""
    doc = await whatsapp_collection.find_one({"_id": ObjectId(doc_id)})
    if not doc or "messages" not in doc:
        return {"error": "No training data found"}

    jsonl_content = prepare_jsonl_from_db(doc["messages"])
    if not jsonl_content.strip():
        return {"error": "No valid training examples after filtering"}

    # Upload file
    uploaded = client_openai.files.create(
        file=io.BytesIO(jsonl_content.encode("utf-8")),
        purpose="fine-tune"
    )

    # Start fine-tuning job
    job = client_openai.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model="gpt-4.1-nano-2025-04-14"
    )

    # ✅ Save job_id immediately
    await whatsapp_collection.update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {"fine_tune_job_id": job.id, "fine_tuned_model_id": None}}
    )

    print(f"✅ Training started for doc_id={doc_id}, job_id={job.id}")
    return {"status": "started", "job_id": job.id, "doc_id": doc_id}

# ---------------- CHECK STATUS ----------------
async def check_training_status(job_id: str):
    """Check fine-tuning status and update DB when succeeded."""
    job = client_openai.fine_tuning.jobs.retrieve(job_id)
    status = job.status

    if status == "succeeded":
        model_id = job.fine_tuned_model
        await whatsapp_collection.update_one(
            {"fine_tune_job_id": job_id},
            {"$set": {"fine_tuned_model_id": model_id}},
        )
        return {"status": "succeeded", "fine_tuned_model_id": model_id}

    elif status in ["failed", "cancelled"]:
        return {"status": status}

    return {"status": status}
