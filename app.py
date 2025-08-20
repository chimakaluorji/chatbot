# app.py
import os
import time
import aiofiles
import re
from pathlib import Path
from textblob import TextBlob
from dotenv import load_dotenv
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from openai import OpenAI
from bson import ObjectId
import requests

# ---------------- LOAD ENV ----------------
load_dotenv()

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MONGO ----------------
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "ecomUser")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "Chimakalu0rji")
MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.zd4jdqg.mongodb.net/?retryWrites=true&w=majority"
client_mongo = AsyncIOMotorClient(MONGO_URI)
db = client_mongo["chatbot"]
whatsapp_collection = db["whatsapp_messages"]

# ---------------- OPENAI ----------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID")

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
    chats = []
    async for doc in whatsapp_collection.find():
        doc["_id"] = str(doc["_id"])
        chats.append(doc)
    return chats

class QueryRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    if not FINE_TUNED_MODEL_ID:
        raise HTTPException(status_code=400, detail="No trained model available")

    sentiment = detect_sentiment(req.message)
    system_prompt = (
        f"You are Chima’s WhatsApp auto-reply bot.\n"
        f"Incoming message sentiment: {sentiment}.\n"
        "Always produce new, natural, human-like responses.\n"
    )

    resp = openai_client.chat.completions.create(
        model=FINE_TUNED_MODEL_ID,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": req.message}],
        max_completion_tokens=200
    )

    return {"status": "success", "sentiment": sentiment, "user_message": req.message, "bot_reply": resp.choices[0].message.content}

@app.post("/train/{doc_id}")
async def trigger_training(doc_id: str):
    try:
        resp = requests.post(f"http://127.0.0.1:8002/train/{doc_id}", timeout=5)
        return resp.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.put("/patterns/{pattern_id}/autoreply")
async def update_auto_reply(pattern_id: str, status: bool):
    result = await whatsapp_collection.update_one(
        {"_id": ObjectId(pattern_id)}, {"$set": {"autoReply": status}}
    )
    return {"success": result.modified_count == 1, "autoReply": status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
