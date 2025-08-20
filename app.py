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
from train import start_training, check_training_status  # ✅ import from train.py

# ---------------- LOAD ENV ----------------
load_dotenv()

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MONGO (initialized on startup) ----------------
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    MONGO_USERNAME = os.getenv("MONGO_USERNAME")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
    if not all([MONGO_USERNAME, MONGO_PASSWORD, MONGO_CLUSTER]):
        raise ValueError("❌ Missing MongoDB environment variables")
    MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_CLUSTER}/?retryWrites=true&w=majority"

client_mongo: AsyncIOMotorClient = None
db = None
whatsapp_collection = None


@app.on_event("startup")
async def startup_db_client():
    global client_mongo, db, whatsapp_collection
    client_mongo = AsyncIOMotorClient(MONGO_URI)
    db = client_mongo["chatbot"]
    whatsapp_collection = db["whatsapp_messages"]
    print("✅ MongoDB connection established")


@app.on_event("shutdown")
async def shutdown_db_client():
    client_mongo.close()
    print("❌ MongoDB connection closed")

# ---------------- OPENAI ----------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- UTILS ----------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def detect_sentiment(message: str):
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
    return messages

# ---------------- ROUTES ----------------
@app.post("/whatsapp")
async def upload_whatsapp(title: str = Form(...), file: UploadFile = File(...)):
    filename = f"{int(time.time())}{Path(file.filename).suffix}"
    file_path = UPLOAD_DIR / filename

    # Save file
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Parse file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {str(e)}")

    messages = parse_messages(lines)
    if not messages:
        raise HTTPException(status_code=400, detail="No valid messages found in file.")

    # Insert/update DB
    existing_doc = await whatsapp_collection.find_one({"title": title})
    if existing_doc:
        await whatsapp_collection.update_one(
            {"_id": existing_doc["_id"]},
            {"$push": {"messages": {"$each": messages}}, "$set": {"autoReply": False}}
        )
        return {"status": "updated", "title": title, "new_messages": len(messages), "autoReply": False}
    else:
        doc = {"title": title, "messages": messages, "autoReply": False}
        result = await whatsapp_collection.insert_one(doc)
        return {"status": "created", "id": str(result.inserted_id), "title": title, "messages_count": len(messages), "autoReply": False}

@app.get("/whatsapp")
async def get_all_whatsapp():
    try:
        docs = await whatsapp_collection.find().to_list(length=None)
        for doc in docs:
            doc["_id"] = str(doc["_id"])  # Convert ObjectId for JSON
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chats: {str(e)}")

class QueryRequest(BaseModel):
    id: str
    message: str

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    doc = await whatsapp_collection.find_one({"_id": ObjectId(req.id)})
    if not doc or "FINE_TUNED_MODEL_ID" not in doc:
        raise HTTPException(status_code=400, detail="No trained model available")

    model_id = doc["FINE_TUNED_MODEL_ID"]

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

@app.post("/train/{doc_id}")
async def trigger_training(doc_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_training, doc_id)
    return {"status": "started", "doc_id": doc_id}

@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    try:
        result = await check_training_status(job_id)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.put("/patterns/{pattern_id}/autoreply")
async def update_auto_reply(pattern_id: str, status: bool):
    try:
        result = await whatsapp_collection.update_one(
            {"_id": ObjectId(pattern_id)},
            {"$set": {"autoReply": status}}
        )
        return {"success": result.modified_count == 1, "autoReply": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update autoReply: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
