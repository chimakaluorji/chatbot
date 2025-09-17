# app.py
import os
import time
import aiofiles
import re
from pathlib import Path
from textblob import TextBlob
from dotenv import load_dotenv
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from openai import OpenAI
from bson import ObjectId
from train import start_training, check_training_status
from fastapi import Query, HTTPException

# ---------------- LOAD ENV ----------------
load_dotenv()

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MONGO ----------------
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    MONGO_USERNAME = os.getenv("MONGO_USERNAME")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
    if not all([MONGO_USERNAME, MONGO_PASSWORD, MONGO_CLUSTER]):
        raise ValueError("❌ Missing MongoDB environment variables")
    MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_CLUSTER}/?retryWrites=true&w=majority"

_client = None
def get_collection():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    db = _client["chatbot"]
    return db["whatsapp_messages"]

# ---------------- OPENAI ----------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- UTILS ----------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def detect_sentiment(message: str) -> str:
    polarity = TextBlob(message).sentiment.polarity
    return "positive" if polarity > 0.2 else "negative" if polarity < -0.2 else "neutral"

def parse_messages(lines, your_name="CHIMA KALU-ORJI"):
    messages = []
    pattern = r"^\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}(?:\s| )?(?:am|pm) - (.*?): (.*)$"
    for i in range(len(lines) - 1):
        m1 = re.match(pattern, lines[i], flags=re.IGNORECASE)
        m2 = re.match(pattern, lines[i + 1], flags=re.IGNORECASE)
        if m1 and m2:
            sender, text = m1.groups()
            next_sender, next_text = m2.groups()
            if sender != your_name and next_sender == your_name:
                system_prompt = (
                    "You are Chima’s WhatsApp auto-reply bot. "
                    "Mimic his tone, style, and way of speaking based on chat history. "
                    "Adapt tone based on sentiment: "
                    "Positive → warm and friendly, Negative → empathetic, Neutral → casual. "
                    "Always generate an original response, not a copy, and apply commonsense reasoning."
                    f" The incoming message is {detect_sentiment(text)}."
                )
                messages.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text.strip()},
                    {"role": "assistant", "content": next_text.strip()}
                ])

    flat_messages = []
    for item in messages:
        if isinstance(item, list):
            flat_messages.extend(item)
        else:
            flat_messages.append(item)
    return flat_messages


# ---------------- ROUTES ----------------
@app.post("/whatsapp")
async def upload_whatsapp(
    title: str = Form(...),
    assistant: str = Form(...),
    user: str = Form(...),
    fine_tuned_model_id: str = Form(None),
    file: UploadFile = File(...)
):
    whatsapp_collection = get_collection()
    filename = f"{int(time.time())}{Path(file.filename).suffix}"
    file_path = UPLOAD_DIR / filename

    # Save file
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Read lines
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {str(e)}")

    messages = parse_messages(lines)
    if not messages:
        raise HTTPException(status_code=400, detail="No valid messages found in file.")

    # 🔹 Check by BOTH title and user
    existing_doc = await whatsapp_collection.find_one({"title": title, "user": user})

    if existing_doc:
        # Update existing doc (append messages + update metadata)
        await whatsapp_collection.update_one(
            {"_id": existing_doc["_id"]},
            {
                "$push": {"messages": {"$each": messages}},
                "$set": {
                    "assistant": assistant,
                    "fine_tuned_model_id": fine_tuned_model_id,
                    "autoReply": False
                }
            }
        )
        return {
            "status": "updated",
            "_id": str(existing_doc["_id"]),
            "title": title,
            "new_messages": len(messages),
            "assistant": assistant,
            "user": user,
            "fine_tuned_model_id": fine_tuned_model_id,
            "autoReply": False
        }
    else:
        # Insert new doc
        doc = {
            "title": title,
            "assistant": assistant,
            "user": user,
            "messages": messages,
            "autoReply": False,
            "fine_tuned_model_id": fine_tuned_model_id
        }
        result = await whatsapp_collection.insert_one(doc)
        return {
            "status": "created",
            "id": str(result.inserted_id),
            "title": title,
            "messages_count": len(messages),
            "assistant": assistant,
            "user": user,
            "fine_tuned_model_id": fine_tuned_model_id,
            "autoReply": False
        }



@app.get("/whatsapp/autoreply")
async def get_user_autoreply(user: str = Query(..., description="Sender/user name")):
    """
    Get autoReply status for a given user.
    """
    try:
        whatsapp_collection = get_collection()
        doc = await whatsapp_collection.find_one({"user": user})

        if not doc:
            raise HTTPException(status_code=404, detail=f"No record found for user {user}")

        return {
            "_id": str(doc["_id"]),  # ✅ Convert ObjectId to string
            "user": user,
            "autoReply": doc.get("autoReply", False),
            "fine_tuned_model_id": doc.get("fine_tuned_model_id"),
            "fine_tune_job_id": doc.get("fine_tune_job_id"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching autoReply: {str(e)}")

    
class QueryRequest(BaseModel):
    id: str
    message: str


@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    whatsapp_collection = get_collection()
    doc = await whatsapp_collection.find_one({"_id": ObjectId(req.id)})
    if not doc or "fine_tuned_model_id" not in doc or not doc["fine_tuned_model_id"]:
        raise HTTPException(status_code=400, detail="No trained model available")

    model_id = doc["fine_tuned_model_id"]

    sentiment = detect_sentiment(req.message)
    system_prompt = (
        f"You are Chima’s WhatsApp auto-reply bot.\n"
        f"Incoming message sentiment: {sentiment}.\n"
        "Always produce new, natural, human-like responses.\n"
    )

    resp = openai_client.chat.completions.create(
        model=model_id,
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
        "bot_reply": resp.choices[0].message.content
    }

# ---------------- TRAINING ----------------
@app.post("/train/{doc_id}")
async def trigger_training(doc_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_training, doc_id)
    return {"status": "started", "doc_id": doc_id}

@app.get("/train/status/{doc_id}")
async def get_training_status(doc_id: str):
    try:
        whatsapp_collection = get_collection()
        doc = await whatsapp_collection.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return {"status": "error", "message": "Document not found"}

        job_id = doc.get("fine_tune_job_id")
        if not job_id:
            return {"status": "waiting_for_job_id", "job_id": None}

        job = openai_client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        if status == "pending":
            return {
                "status": "queued",
                "job_id": job_id,
                "queue_position": getattr(job, "queue_position", None),
                "fine_tuned_model_id": doc.get("fine_tuned_model_id")
            }

        elif status == "succeeded":
            model_id = job.fine_tuned_model
            await whatsapp_collection.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {"fine_tuned_model_id": model_id}}
            )
            return {
                "status": "succeeded",
                "job_id": job_id,
                "fine_tuned_model_id": model_id
            }

        elif status in ["failed", "cancelled"]:
            return {
                "status": status,
                "job_id": job_id,
                "fine_tuned_model_id": None
            }

        else:
            return {
                "status": "running",
                "job_id": job_id,
                "fine_tuned_model_id": doc.get("fine_tuned_model_id")
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.put("/patterns/{pattern_id}/autoreply")
async def update_auto_reply(pattern_id: str, status: bool):
    try:
        whatsapp_collection = get_collection()
        result = await whatsapp_collection.update_one(
            {"_id": ObjectId(pattern_id)},
            {"$set": {"autoReply": status}}
        )
        return {
            "success": result.modified_count == 1,
            "autoReply": status
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update autoReply: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
