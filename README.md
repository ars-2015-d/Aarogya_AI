# 🩺 AarogyaAI

### Medical Triage & Clinical Assistant

AarogyaAI is an AI-powered conversational medical assistant built with Streamlit. It helps users understand their symptoms, provides general triage guidance, identifies potential emergency red flags, and helps locate nearby medical facilities.

---
## 💡 What It Does

- **Hybrid RAG Search:** Combines vector embeddings (`BAAI/bge-base-en-v1.5` via FAISS) with keyword retrieval (`BM25s`) to find relevant medical information from the knowledge base.
- **AI-Powered Triage:** Uses Groq and `openai/gpt-oss-20b` to provide natural, low-latency conversational responses.
- **Smart Follow-ups:** Asks focused follow-up questions when additional information is needed before providing guidance.
- **Emergency Detection:** Identifies potential emergency symptoms such as chest pain and shortness of breath and recommends immediate medical attention.
- **Nearby Medical Facilities:** Uses location information and OpenStreetMap to help users find nearby hospitals and medical centers.
- **Cloud-Based Data:** Loads the large medical dataset and FAISS index from Hugging Face instead of storing the data directly in GitHub.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Groq API
- OpenAI GPT-OSS 20B
- Hugging Face
- FAISS
- BM25s
- BAAI/bge-base-en-v1.5
- OpenStreetMap

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/ars-2015-d/aarogya_ai.git
cd aarogya_ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project folder and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 5. Run the Application

Start AarogyaAI with:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```
---

## ⚠️ Medical Disclaimer

AarogyaAI is an AI-powered tool designed for informational and general triage guidance purposes only. It does not provide professional medical diagnoses, treatments, or prescriptions.

If you are experiencing severe or emergency symptoms, seek immediate medical attention or contact your local emergency medical services.
