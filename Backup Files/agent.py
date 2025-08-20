# agent.py
import os, json, faiss, numpy as np, re
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from memory_db import (
    init_db, add_to_memory, get_recent_memory, clear_stale_memory,
    get_contact_profile, append_contact_fact, get_contact_summary,
    add_contact_summary, set_contact_profile
)

load_dotenv()
init_db()

# Cross-encoder reranker
reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3")

# BM25 globals
bm25_index, bm25_chunks = None, None

# -------------------- Helper --------------------
def _get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("❌ OPENAI_API_KEY is missing. Please set it in your environment or .env file.")
    return OpenAI(api_key=key)

def chunk_to_text(chunk):
    if isinstance(chunk, dict):
        sender = chunk.get("sender", "Unknown")
        text = chunk.get("text", "")
        return f"{sender}: {text}"
    return str(chunk)

def tokenize_text(text):
    return re.findall(r"\w+", text.lower())

# -------------------- BM25 --------------------
def build_bm25_index(chunks):
    global bm25_index, bm25_chunks
    bm25_chunks = chunks
    bm25_index = BM25Okapi([tokenize_text(chunk_to_text(c)) for c in chunks])
    print(f"✅ BM25 index built with {len(bm25_chunks)} chunks")

def bm25_search(query, k=3):
    if bm25_index is None:
        return []
    tokenized_query = tokenize_text(query)
    scores = bm25_index.get_scores(tokenized_query)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunk_to_text(bm25_chunks[i]) for i in ranked]

# -------------------- FAISS Search (Safe) --------------------
def search_history(query, k=3):
    if not os.path.exists("whatsapp.index") or not os.path.exists("chunks.json"):
        print("⚠️ FAISS search skipped — missing index or chunks.json")
        return []

    if os.path.getsize("whatsapp.index") == 0:
        print("⚠️ FAISS search skipped — index file is empty")
        return []

    try:
        index = faiss.read_index("whatsapp.index")
    except Exception as e:
        print(f"⚠️ Failed to read FAISS index: {e}")
        return []

    try:
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load chunks.json: {e}")
        return []

    text_chunks = [chunk_to_text(c) for c in chunks]

    try:
        client = _get_openai_client()
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
    except Exception as e:
        print(f"❌ Failed to create query embedding: {e}")
        return []

    D, I = index.search(np.array([q_emb]).astype("float32"), k)
    return [text_chunks[i] for i in I[0] if 0 <= i < len(text_chunks)]

# -------------------- Hybrid + Rerank --------------------
def rerank_results(query, candidates, top_k=3):
    if not candidates:
        return []
    pairs = [(query, doc) for doc in candidates]
    scores = reranker_model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]

def hybrid_search_with_rerank(query, k=3):
    faiss_res = search_history(query, k=k*2)
    bm25_res = bm25_search(query, k=k*2)
    candidates = []
    for chunk in faiss_res + bm25_res:
        if chunk not in candidates:
            candidates.append(chunk)
    return rerank_results(query, candidates, top_k=k)

# -------------------- Profile & Fact Learning --------------------
def extract_and_store_facts(chat_id, new_message):
    profile = get_contact_profile(chat_id)
    extract_prompt = f"""
You are updating a contact's profile.
From the message, extract any new personal facts 
(e.g., preferences, hobbies, schedules, dates).
If none, reply 'NONE'.

Message: "{new_message}"
Known facts: {', '.join(profile['facts'])}
"""
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "system", "content": extract_prompt}]
        )
        content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
        fact = content.strip() if content else ""
        if fact.upper() != "NONE" and fact:
            append_contact_fact(chat_id, fact)
            maybe_summarize_contact_profile(chat_id)
    except RuntimeError as e:
        print(str(e))

def maybe_summarize_contact_profile(chat_id, max_facts=10, max_chars=1000):
    profile = get_contact_profile(chat_id)
    facts = profile["facts"]

    if len(facts) > max_facts or sum(len(f) for f in facts) > max_chars:
        summarization_prompt = f"""
Summarize this contact's profile into one short paragraph, 
keeping only key, permanent facts.

Facts:
{', '.join(facts)}
"""
        try:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "system", "content": summarization_prompt}]
            )
            content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
            summary = content.strip() if content else ""
            if summary:
                add_contact_summary(chat_id, summary)
                set_contact_profile(chat_id, profile["tone"], [summary])
        except RuntimeError as e:
            print(str(e))

# -------------------- Commonsense & Self-Verification --------------------
def fact_check(reply_text, context_chunks):
    check_prompt = f"""
Verify if the reply is supported by history/context or obvious commonsense.
If unsupported, output 'UNSUPPORTED'. If supported, output 'SUPPORTED'.

Reply: "{reply_text}"
Context:
{context_chunks}
"""
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "system", "content": check_prompt}]
        )
        content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
        return content.strip().upper() == "SUPPORTED" if content else False
    except RuntimeError as e:
        print(str(e))
        return False

# -------------------- Reply Generation --------------------
def agent_reply_stream(message, chat_id="default"):
    clear_stale_memory()
    add_to_memory(chat_id, "user", message)
    extract_and_store_facts(chat_id, message)

    mem = get_recent_memory(chat_id, limit=10)
    profile = get_contact_profile(chat_id)
    summary = get_contact_summary(chat_id) or ""
    recent_user_msgs = [m["content"] for m in mem if m["role"] == "user"][-2:]
    retrieval_query = "\n".join(recent_user_msgs)

    context_list = hybrid_search_with_rerank(retrieval_query, k=3)
    context_text = "\n---\n".join(context_list) if context_list else ""

    facts_text = summary if summary else ", ".join(profile["facts"])
    combined_context = f"""
Tone: {profile['tone']}
Facts: {facts_text}

Recent conversation:
{''.join(f"{m['role']}: {m['content']}\n" for m in mem)}

Retrieved context:
{context_text}
"""

    reply_accumulated = ""
    try:
        client = _get_openai_client()
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are me, replying on WhatsApp."},
                {"role": "user", "content": combined_context}
            ],
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                token = delta.content
                reply_accumulated += token
                yield token

    except RuntimeError as e:
        yield "I don't understand what you mean, please rephrase"

    if not reply_accumulated.strip():
        yield "I don't understand what you mean, please rephrase"

    add_to_memory(chat_id, "assistant", reply_accumulated or "I don't understand what you mean, please rephrase")

def agent_reply(message, chat_id="default"):
    reply_text = "".join(agent_reply_stream(message, chat_id))
    if not reply_text.strip():
        return "I don't understand what you mean, please rephrase"
    return reply_text
