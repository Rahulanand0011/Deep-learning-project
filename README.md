# Chat with PDF & Documents (Groq & Hugging Face)

A Python application that allows you to chat with your documents (PDF, DOCX, TXT, CSV) using RAG (Retrieval-Augmented Generation), LangChain, FAISS, and AI models.

## ✨ Supported AI Models
1. **⭐ Groq AI (Recommended - Ultra Fast & Smart)**: Powered by Groq's LPU inference engine (`openai/gpt-oss-120b`). Delivers conversational, accurate, and structured answers (summaries, explanations, code breakdowns) in milliseconds.
2. **Local Models (`flan-t5-small` / `flan-t5-base`)**: 100% offline local inference on CPU/GPU without needing any API key.
3. **Hugging Face Hub API (`flan-t5-large`)**: Cloud-hosted inference via Hugging Face.

---

## ⚡ Groq API Key

Your Groq API key is already configured in the `.env` file and pre-loaded in the app.

If you ever need to update it:
- Either change it in the [`.env`](file:///.env) file:
  ```env
  GROQ_API_KEY=your_key_here
  ```
- Or update it directly in the app sidebar under **"⚡ Groq API Key"**.

---

## 🚀 Running the App
- Double-click `run.bat`
- Or run `.\run.ps1` in PowerShell
- Browser URL: **http://localhost:8501**
