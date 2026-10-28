# 🧠 Deep Learning Document Intelligence & RAG Chatbot

An applied Deep Learning and NLP application that combines Dense Vector Embeddings, Neural Cross-Encoder Re-Ranking, Speech-to-Text (Whisper ASR), Faithfulness Evaluation, and Groq LPUs (`openai/gpt-oss-120b`) for document Q&A and analysis.

---

## 🔬 Core Deep Learning & NLP Features

1. **⚡ Neural Cross-Encoder Re-Ranking**:
   - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-score candidate document chunks using full bidirectional cross-attention.
   - Eliminates irrelevant passages and drastically boosts retrieval precision.

2. **📚 Source Citations & Page Attribution**:
   - Preserves per-page metadata across PDF, Word, CSV, and Text documents.
   - Every answer includes an expandable citation view with exact snippets and document page numbers.

3. **🎯 Faithfulness & Grounding Confidence Score**:
   - Measures semantic entailment between the generated answer and retrieved source passages.
   - Displays real-time confidence scores (e.g., `🟢 92.5% High Entailment`).

4. **🎤 Voice Input (Speech-to-Text via Whisper Large v3)**:
   - Talk to your documents using your microphone via Groq's high-speed Whisper LPU engine.

5. **🔊 Spoken Audio Responses (Text-to-Speech)**:
   - Listen to AI-generated answers directly in the browser.

6. **📊 Automated Executive Summary & Document Overview**:
   - Automatically extracts domain categorization, key concepts, and high-level summaries upon document upload.

---

## 🚀 Quick Start

### 1. Launch the Application
Double-click `run.bat` or run in PowerShell:
```powershell
.\run.ps1
```
Open **http://localhost:8501** in your browser.

### 2. Configure API Key
Add your Groq API key in [`.env`](.env) or directly in the Streamlit sidebar:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Get a free key from [console.groq.com/keys](https://console.groq.com/keys)).*
