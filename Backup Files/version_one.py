import re
import json
import os
import time
from textblob import TextBlob
from dotenv import load_dotenv, set_key
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

file_path = "whatsapp.txt"
output_file = "whatsapp_finetune.jsonl"
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID")

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

class QueryRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    your_name: str = "CHIMA KALU-ORJI"

# ----------------------------
# Helpers
# ----------------------------
def read_whatsapp_txt(file_path):
    """Read WhatsApp chat export."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()

def detect_sentiment(message):
    """Determine sentiment category."""
    polarity = TextBlob(message).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    return "neutral"

def parse_messages(lines, your_name="CHIMA KALU-ORJI"):
    """Extract Q/A pairs from WhatsApp export."""
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

def save_to_jsonl(messages, output_file):
    """Save parsed messages to JSONL for fine-tuning."""
    with open(output_file, "w", encoding="utf-8") as f:
        for msg in messages:
            system_prompt = (
                "You are Chima’s WhatsApp auto-reply bot. "
                "Mimic his tone and style based on history. "
                "Apply commonsense reasoning. "
                f"Adapt tone for a {msg['sentiment']} incoming message."
            )
            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg["incoming"]},
                    {"role": "assistant", "content": msg["reply"]}
                ]
            }
            f.write(json.dumps(data) + "\n")
    print(f"✅ Saved fine-tuning data to {output_file}")

def fine_tune_gpt35(jsonl_file):
    """Start fine-tuning job."""
    with open(jsonl_file, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="fine-tune")
    fine_tune_job = client.fine_tuning.jobs.create(
        training_file=uploaded_file.id,
        model="gpt-3.5-turbo"
    )
    print(f"🚀 Fine-tune job started! Job ID: {fine_tune_job.id}")
    return fine_tune_job.id

def wait_for_finetune(job_id):
    """Wait for fine-tuning to finish."""
    print("⏳ Waiting for fine-tuning to complete...")
    while True:
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        status = job_status.status
        print(f"   Status: {status}")
        if status in ["succeeded", "failed", "cancelled"]:
            return job_status
        time.sleep(20)

def chat_with_model(model_name, query):
    """Query the fine-tuned model."""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Chima’s WhatsApp bot. "
                    "Use your training data for tone, style, and sentiment. "
                    "Always create original responses with commonsense."
                )
            },
            {"role": "user", "content": query}
        ],
        temperature=0.9,
        max_completion_tokens=200
    )
    return resp.choices[0].message.content

# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    if not FINE_TUNED_MODEL_ID:
        return {"error": "Model not trained yet", "status": "failed"}
    try:
        reply = chat_with_model(FINE_TUNED_MODEL_ID, request.message)
        return {"status": "success", "user_message": request.message, "bot_reply": reply}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/train")
def train_endpoint(req: TrainRequest):
    """Retrain model on demand."""
    global FINE_TUNED_MODEL_ID
    lines = read_whatsapp_txt(file_path)
    parsed = parse_messages(lines, your_name=req.your_name)
    save_to_jsonl(parsed, output_file)
    job_id = fine_tune_gpt35(output_file)
    job_status = wait_for_finetune(job_id)
    if job_status.status == "succeeded":
        fine_tuned_model = job_status.fine_tuned_model
        set_key(".env", "FINE_TUNED_MODEL_ID", fine_tuned_model)
        FINE_TUNED_MODEL_ID = fine_tuned_model
        return {"status": "success", "model_id": fine_tuned_model}
    else:
        return {"status": "failed", "details": job_status.status}

# ----------------------------
# Startup Training if Needed
# ----------------------------
if __name__ == "__main__":
    if not FINE_TUNED_MODEL_ID:
        print("📦 No fine-tuned model found. Training a new one...")
        lines = read_whatsapp_txt(file_path)
        parsed = parse_messages(lines, your_name="CHIMA KALU-ORJI")
        print(f"✅ Parsed {len(parsed)} message pairs.")
        save_to_jsonl(parsed, output_file)
        job_id = fine_tune_gpt35(output_file)
        job_status = wait_for_finetune(job_id)
        if job_status.status == "succeeded":
            fine_tuned_model = job_status.fine_tuned_model
            print(f"🎯 Fine-tuning complete! Model ID: {fine_tuned_model}")
            set_key(".env", "FINE_TUNED_MODEL_ID", fine_tuned_model)
            FINE_TUNED_MODEL_ID = fine_tuned_model
        else:
            print("❌ Fine-tuning failed or was cancelled.")
            exit(1)
    else:
        print(f"✅ Using existing fine-tuned model: {FINE_TUNED_MODEL_ID}")

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
