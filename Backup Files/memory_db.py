import sqlite3
import time
import json
from typing import List, Dict
import os
import faiss
import numpy as np
from openai import OpenAI

DB_FILE = "memory.db"
FAISS_INDEX_FILE = "whatsapp.index"
CHUNKS_FILE = "chunks.json"

# ------------------ INIT ------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Conversation memory
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        )
    """)
    # Profiles
    c.execute("""
        CREATE TABLE IF NOT EXISTS contact_profiles (
            chat_id TEXT PRIMARY KEY,
            tone TEXT,
            facts TEXT
        )
    """)
    # Summaries
    c.execute("""
        CREATE TABLE IF NOT EXISTS contact_summaries (
            chat_id TEXT PRIMARY KEY,
            summary TEXT
        )
    """)
    conn.commit()
    conn.close()

# ------------------ OPENAI CLIENT ------------------
def _get_openai_client():
    """Safely create an OpenAI client only when needed."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "❌ OPENAI_API_KEY is missing. Please set it in your environment or .env file."
        )
    return OpenAI(api_key=key)

# ------------------ MEMORY ------------------
def add_to_memory(chat_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversation_memory (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, time.time())
    )
    conn.commit()
    conn.close()

    # Append to chunks.json
    _append_to_chunks_file(role, content)

    # Index into FAISS safely
    _index_into_faiss(f"{role}: {content}")

def get_recent_memory(chat_id: str, limit: int = 10) -> List[Dict]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM conversation_memory WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
              (chat_id, limit))
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

def clear_stale_memory(timeout_minutes=43200):
    now = time.time()
    cutoff = now - (timeout_minutes * 60)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM conversation_memory WHERE timestamp < ?", (cutoff,))
    conn.commit()
    conn.close()

# ------------------ PROFILES ------------------
def set_contact_profile(chat_id: str, tone: str, facts: list):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO contact_profiles (chat_id, tone, facts) VALUES (?, ?, ?)",
        (chat_id, tone, json.dumps(facts))
    )
    conn.commit()
    conn.close()

def get_contact_profile(chat_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT tone, facts FROM contact_profiles WHERE chat_id = ?", (chat_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"tone": row[0], "facts": json.loads(row[1])}
    return {"tone": "neutral", "facts": []}

def append_contact_fact(chat_id: str, new_fact: str):
    profile = get_contact_profile(chat_id)
    facts = profile["facts"]
    if new_fact not in facts:
        facts.append(new_fact)
        set_contact_profile(chat_id, profile["tone"], facts)

# ------------------ SUMMARIES ------------------
def add_contact_summary(chat_id: str, summary: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO contact_summaries (chat_id, summary) VALUES (?, ?)",
        (chat_id, summary)
    )
    conn.commit()
    conn.close()

def get_contact_summary(chat_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT summary FROM contact_summaries WHERE chat_id = ?", (chat_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""

# ------------------ INTERNAL HELPERS ------------------
def _append_to_chunks_file(role: str, content: str):
    if not os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump({"data": []}, f)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    if isinstance(chunks_data, dict) and "data" in chunks_data:
        chunks_list = chunks_data["data"]
    else:
        chunks_list = chunks_data

    chunks_list.append({"sender": role, "text": content})

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        if isinstance(chunks_data, dict):
            chunks_data["data"] = chunks_list
            json.dump(chunks_data, f, indent=2)
        else:
            json.dump(chunks_list, f, indent=2)

def _index_into_faiss(text: str):
    """Generate embedding and insert into FAISS index safely."""
    try:
        client = _get_openai_client()
    except RuntimeError as e:
        print(str(e))
        print("⚠️ Skipping FAISS indexing because no API key was found.")
        return

    try:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    except Exception as e:
        print(f"❌ Failed to create embedding: {e}")
        return

    vector = np.array([emb], dtype="float32")

    if os.path.exists(FAISS_INDEX_FILE) and os.path.getsize(FAISS_INDEX_FILE) > 0:
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
        except Exception as e:
            print(f"⚠️ FAISS index is invalid, creating new one: {e}")
            index = faiss.IndexFlatL2(len(emb))
    else:
        index = faiss.IndexFlatL2(len(emb))

    index.add(vector)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"✅ Added vector to FAISS index ({FAISS_INDEX_FILE})")
