import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import gc
import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
import bm25s

from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import hf_hub_download, snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)

if not GROQ_API_KEY or not HF_TOKEN:
    st.error(
        "Missing GROQ_API_KEY or HF_TOKEN. Set them in Streamlit Cloud → "
        "App settings → Secrets (as TOML: GROQ_API_KEY = \"...\"), not in code."
    )
    st.stop()

REPO_ID = "Hybridminded/aarogya-data"

st.set_page_config(
    page_title="AarogyaAI • Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CLAUDE-STYLE DARK THEME
# ============================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg-main: #181716;
        --sidebar-bg: #141312;
        --surface-card: #22201E;
        --surface-hover: #2B2826;
        --border-color: #2F2C2A;
        --text-headline: #FAF7EE;
        --text-body: #D4CEBF;
        --text-muted: #8E887E;
        --accent-terracotta: #D97757;
    }

    html, body {
        color-scheme: dark;
    }

    .stApp, p, span, div, h1, h2, h3, button, input, textarea {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg-main) !important;
        color: var(--text-body) !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .main .block-container {
        max-width: 800px !important;
        padding-top: 2rem !important;
        padding-bottom: 8.5rem !important;
        margin: 0 auto !important;
    }

    .claude-greeting {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 2.3rem;
        font-weight: 400;
        color: var(--text-headline);
        text-align: center;
        letter-spacing: -0.02em;
        margin: 2rem 0 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
    }
    .claude-sparkle {
        color: var(--accent-terracotta);
        font-size: 1.8rem;
    }
    .claude-subheading {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    .sidebar-brand {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 1.45rem;
        font-weight: 500;
        color: var(--text-headline);
        margin-bottom: 1.25rem;
    }
    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 1.5rem 0 0.5rem 0;
    }

    .stChatMessage {
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        margin-bottom: 1rem !important;
        border: none !important;
    }
    .stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage strong {
        color: var(--text-headline) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--surface-card) !important;
        border: 1px solid var(--border-color) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: transparent !important;
        padding-left: 0.5rem !important;
    }

    .stButton > button {
        background: var(--surface-card) !important;
        color: var(--text-body) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 20px !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease-in-out !important;
        text-align: center !important;
    }
    .stButton > button:hover {
        background: var(--surface-hover) !important;
        border-color: #484440 !important;
        color: var(--text-headline) !important;
        transform: translateY(-1px);
    }

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background: var(--surface-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-terracotta) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text-headline) !important;
        -webkit-text-fill-color: var(--text-headline) !important;
        caret-color: var(--text-headline) !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;
    }
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }

    .emergency-box {
        background: #2C1818;
        border: 1px solid #572A2A;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        color: #F87171 !important;
        font-size: 0.82rem;
        line-height: 1.6;
        margin-top: 1rem;
    }
    .emergency-box strong {
        color: #FCA5A5 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD RAG BACKEND (MEMORY OPTIMIZED)
# ============================================================
@st.cache_resource
def load_rag_backend():
    parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="exploded_df_final.parquet",
        repo_type="dataset",
        token=HF_TOKEN
    )
    faiss_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="medical_faiss.index",
        repo_type="dataset",
        token=HF_TOKEN
    )

    chunk_series = pd.read_parquet(parquet_path, columns=["chunk_text"])["chunk_text"]
    index = faiss.read_index(faiss_path)

    try:
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"}
        )
    except Exception:
        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    try:
        bm25_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="bm25_index_saved/*",
            token=HF_TOKEN
        )
        saved_path = os.path.join(bm25_dir, "bm25_index_saved")
        bm25_index = bm25s.BM25.load(saved_path, load_corpus=False)
    except Exception:
        corpus = chunk_series.tolist()
        corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
        bm25_index = bm25s.BM25()
        bm25_index.index(corpus_tokens)

    gc.collect()
    client = Groq(api_key=GROQ_API_KEY)
    return chunk_series, index, embedding, bm25_index, client


with st.spinner("Connecting knowledge base..."):
    chunk_series, faiss_index, embedding_model, bm25_index, groq_client = load_rag_backend()


# ============================================================
# LOCATION & HOSPITALS
# ============================================================
@st.cache_data(ttl=3600)
def get_location_and_hospitals():
    city, lat, lon = None, None, None
    try:
        geo = requests.get("https://ipapi.co/json/", timeout=4).json()
        city = geo.get("city")
        lat = geo.get("latitude")
        lon = geo.get("longitude")
    except Exception:
        pass

    hospitals = []
    if lat and lon:
        try:
            overpass_url = "https://overpass-api.de/api/interpreter"
            query = f"""
            [out:json][timeout:10];
            (
              node["amenity"="hospital"](around:5000,{lat},{lon});
              way["amenity"="hospital"](around:5000,{lat},{lon});
            );
            out center 5;
            """
            response = requests.post(overpass_url, data=query, timeout=10).json()
            for element in response.get("elements", [])[:4]:
                name = element.get("tags", {}).get("name")
                if name:
                    hospitals.append(name)
        except Exception:
            pass

    return city or "your area", lat, lon, hospitals


user_city, user_lat, user_lon, nearby_hospitals = get_location_and_hospitals()


# ============================================================
# RETRIEVAL & CLASSIFICATION
# ============================================================
def hybrid_search(query, top_k=3):
    query_vector = np.array(
        embedding_model.embed_query(query), dtype="float32"
    ).reshape(1, -1)
    vec_scores, vec_indices = faiss_index.search(query_vector, 8)

    query_tokens = bm25s.tokenize(query, show_progress=False)
    bm25_results, bm25_scores = bm25_index.retrieve(query_tokens, k=8)
    bm25_indices = bm25_results[0]

    rrf_scores = {}
    for rank, idx in enumerate(bm25_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)
    for rank, idx in enumerate(vec_indices[0]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top = ranked[:top_k]
    return [chunk_series.iloc[idx] for idx, _ in top]


def build_retrieval_query(query, history):
    if not history:
        return query
    recent_user = [m["content"] for m in history if m["role"] == "user"][-2:]
    return " ".join(recent_user) + f" {query}"


EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "difficulty breathing", "shortness of breath", "fainting", "passed out",
    "sudden numbness", "slurred speech", "stroke", "severe bleeding",
    "coughing blood", "unconscious", "severe chest", "chest tight",
    "chest feels heavy", "heavy and tight", "arm numb", "left arm",
    "radiating pain", "crushing pressure"
]

def is_emergency(query):
    q = query.lower()
    if any(kw in q for kw in EMERGENCY_KEYWORDS):
        return True
    if "chest" in q and any(w in q for w in ["tight", "heavy", "pain", "pressure", "burning"]):
        return True
    if "arm" in q and any(w in q for w in ["numb", "tingling", "weak", "heavy"]):
        return True
    return False


MEDICATION_KEYWORDS = [
    "tablet", "tablets", "medicine", "medication", "drug", "pill", "pills",
    "dose", "dosage", "otc", "over the counter", "syrup", "capsule", "antibiotic"
]

HOME_REMEDY_KEYWORDS = [
    "try at home", "at home", "home remedy", "home remedies", "home care",
    "what should i do at home", "natural remedy"
]

SYMPTOM_PHRASES = [
    "i have", "i've been", "i am feeling", "i'm feeling", "i feel",
    "my ", "since yesterday", "since last", "for the past", "it hurts",
    "he is", "she is", "they are", "feeling sick", "not well", "unwell"
]


def classify_query(query):
    q = query.lower()
    if any(kw in q for kw in HOME_REMEDY_KEYWORDS):
        return "home_remedy"
    if any(kw in q for kw in MEDICATION_KEYWORDS):
        return "medication"
    if any(phrase in q for phrase in SYMPTOM_PHRASES):
        return "symptom"
    return "informational"


# ============================================================
# STREAMING ENGINE
# ============================================================
def stream_response(query, history, hospitals, context, is_followup_reply=False):
    clean_query = query.strip().lower()

    if clean_query in {"thank you", "thanks", "thx", "ok", "okay", "bye", "goodbye", "got it", "great", "alright", "ok thank you"}:
        yield "You're very welcome! Rest well, stay hydrated, and feel free to reach out if anything changes. Take good care! 🙏"
        return

    # IMMEDIATE EMERGENCY OVERRIDE
    if is_emergency(query):
        hospital_str = f"**{hospitals[0]}**" if hospitals else "your nearest emergency department"
        yield (
            f"🚨 **EMERGENCY WARNING: Please seek immediate emergency medical care.**\n\n"
            f"Symptoms involving chest tightness, pressure, or radiating arm numbness can indicate a serious cardiovascular or acute medical emergency.\n\n"
            f"• **Do not wait and do not attempt home care.**\n"
            f"• Go immediately to {hospital_str} or call emergency services (112 / 108 / 911) right now."
        )
        return

    hospital_name = f"**{hospitals[0]}**" if hospitals else "your nearest clinic"
    query_type = classify_query(query)

    if query_type == "medication":
        directive = """The user is asking about a tablet, medicine, or OTC drug.
- State in 1-2 concise bullet points what that class of medicine is generally used for.
- NEVER state specific milligram (mg) dosages, frequencies, or prescription regimens.
- MANDATORY final warning: "⚠️ *Do not take any medication without consulting a doctor or pharmacist to confirm the appropriate choice and dosage for your health condition.*"
- Maximum 4 sentences total."""

    elif query_type == "home_remedy":
        directive = f"""The user explicitly asked for home care steps.
- Provide 3 practical, safe home comfort steps in clear bullet points.
- NEVER use markdown tables.
- Add one closing line: "If symptoms persist or worsen after 24–48 hours, please consult a healthcare provider at {hospital_name}."
- No lengthy explanations or follow-up questions."""

    elif query_type == "symptom" and not is_followup_reply:
        directive = """The user is reporting a symptom for the first time without sufficient detail.
- Warmly acknowledge their discomfort in 1 short sentence.
- Ask EXACTLY 2 natural, focused follow-up questions (e.g., onset/duration, exact location, or accompanying fever/nausea).
- NEVER ask to rate pain on a scale of 1 to 10.
- Do NOT provide long lists of home remedies or red-flag dumps yet — wait for their reply."""

    else:
        if query_type == "symptom":
            directive = f"""The user has answered your follow-up questions.
- Briefly mention what type of condition this pattern is commonly associated with (phrased as a possibility, never a definitive diagnosis).
- Give 3 concise, safe home comfort steps in bullet points (NO markdown tables).
- Highlight 1-2 red-flag signs that require visiting {hospital_name} or a physician.
- Do not ask further follow-up questions."""
        else:
            directive = """This is a general factual question.
- Answer directly and concisely in 2-3 short paragraphs or bullet points.
- Do NOT use the triage or home-care template."""

    system_message = f"""You are AarogyaAI, an empathetic clinical triage and health assistant.

CRITICAL FORMATTING & SAFETY RULES:
- NEVER generate markdown tables. Use standard markdown bullet points only.
- Refuse to prescribe specific antibiotic or prescription drug dosages (mg amounts).
- Keep every answer structured, concise, and easy to read.

CURRENT INSTRUCTION:
{directive}
"""

    messages = [{"role": "system", "content": system_message}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Reference Context:\n{context}\n\nUser Question/Statement: {query}"
    })

    stream = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.1,
        max_tokens=550,
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_followup_reply" not in st.session_state:
    st.session_state.awaiting_followup_reply = False

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🩺 AarogyaAI</div>', unsafe_allow_html=True)

    if st.button("＋  New Consultation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.awaiting_followup_reply = False
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Session History</div>', unsafe_allow_html=True)
    if st.session_state.messages:
        user_queries = [m["content"][:28] + "..." for m in st.session_state.messages if m["role"] == "user"]
        for q in user_queries[-5:]:
            st.caption(f"💬 {q}")
    else:
        st.caption("No consultations yet.")

    st.markdown('<div class="sidebar-section-title">Nearby Medical Centers</div>', unsafe_allow_html=True)
    if nearby_hospitals:
        for h in nearby_hospitals[:3]:
            st.markdown(f"• {h}")
        if user_lat and user_lon:
            maps_url = f"https://www.google.com/maps/search/hospitals/@{user_lat},{user_lon},14z"
            st.markdown(f"[View on Google Maps ↗]({maps_url})")
    else:
        city_query = user_city.replace(" ", "+") if user_city != "your area" else "hospitals+near+me"
        maps_url = f"https://www.google.com/maps/search/{city_query}+hospital"
        st.markdown(f"[Find Centers Near You ↗]({maps_url})")

    st.markdown("""
    <div class="emergency-box">
        <strong>⚠️ Urgent Emergency?</strong><br>
        Severe chest pain, sudden breathlessness, or numbness requires immediate emergency room care.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN CANVAS (GREETING & QUICK START)
# ============================================================
if not st.session_state.messages:
    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    hour = ist_now.hour

    if hour < 4 or hour >= 22:
        time_greeting = "Up late"
    elif hour < 12:
        time_greeting = "Good morning"
    elif hour < 17:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"

    st.markdown(f"""
    <div class="claude-greeting">
        <span class="claude-sparkle">✴</span> {time_greeting}! How are you feeling today?
    </div>
    <div class="claude-subheading">
        Describe your symptoms below or select a common concern to get started.
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    quick_input = None

    with p1:
        if st.button("🤕 Headache & Migraine", use_container_width=True):
            quick_input = "I have a throbbing headache around my temples that started earlier today."
    with p2:
        if st.button("😴 Fatigue & Exhaustion", use_container_width=True):
            quick_input = "I've been feeling exhausted with low energy for the past few days."
    with p3:
        if st.button("🔥 Stomach & Acidity", use_container_width=True):
            quick_input = "I have severe burning in my upper stomach after having meals."
    with p4:
        if st.button("🤧 Cold, Cough & Fever", use_container_width=True):
            quick_input = "I have a sore throat, mild dry cough, and a slight fever."

    if quick_input:
        retrieval_query = build_retrieval_query(quick_input, [])
        results = hybrid_search(retrieval_query, top_k=3)
        context = "\n\n".join([f"[REFERENCE CASE {i+1}]\n{c}" for i, c in enumerate(results)])
        is_followup_reply = st.session_state.awaiting_followup_reply
        query_type = classify_query(quick_input)

        st.session_state.messages.append({"role": "user", "content": quick_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(quick_input)

        with st.chat_message("assistant", avatar="🩺"):
            reply = st.write_stream(stream_response(quick_input, [], nearby_hospitals, context, is_followup_reply))

        st.session_state.awaiting_followup_reply = (query_type == "symptom" and not is_followup_reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ============================================================
# CONVERSATION DISPLAY
# ============================================================
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🩺"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ============================================================
# CHAT INPUT
# ============================================================
if user_text := st.chat_input("Describe how you are feeling or ask a question..."):
    retrieval_query = build_retrieval_query(user_text, st.session_state.messages)
    results = hybrid_search(retrieval_query, top_k=3)
    context = "\n\n".join([f"[REFERENCE CASE {i+1}]\n{c}" for i, c in enumerate(results)])

    history = st.session_state.messages[:]
    is_followup_reply = st.session_state.awaiting_followup_reply
    query_type = classify_query(user_text)

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text)

    with st.chat_message("assistant", avatar="🩺"):
        reply = st.write_stream(stream_response(user_text, history, nearby_hospitals, context, is_followup_reply))

    st.session_state.awaiting_followup_reply = (query_type == "symptom" and not is_followup_reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})