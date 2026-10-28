import os

# --- Thread/memory limits for OpenBLAS on Windows ---
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Optional, List
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# --- Core Document & Vector Store Imports ---
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from PyPDF2 import PdfReader
import docx
import pandas as pd

# Groq integration
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
except ImportError:
    ChatGroq = None


# Default Groq Key (provided by user)
DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()


# ============================================================
# Document Processing
# ============================================================

def get_files_text(uploaded_files):
    text = ""
    for f in uploaded_files:
        ext = os.path.splitext(f.name)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        elif ext in (".docx", ".doc"):
            doc = docx.Document(f)
            text += "\n".join(p.text for p in doc.paragraphs if p.text) + "\n"
        elif ext == ".csv":
            try:
                df = pd.read_csv(f)
                text += df.to_string() + "\n"
            except Exception:
                pass
        elif ext == ".txt":
            text += f.read().decode("utf-8", errors="ignore") + "\n"
    return text


def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vectorstore(chunks):
    return FAISS.from_texts(chunks, get_embeddings())


# ============================================================
# LLM Setup
# ============================================================

def build_groq_llm(api_key: str):
    if ChatGroq is None:
        raise ImportError("langchain-groq is not installed. Run: pip install langchain-groq groq")
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0.3,
    )


def build_hf_hub_llm(token: str):
    from langchain_community.llms import HuggingFaceHub
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        task="text2text-generation",
        huggingfacehub_api_token=token,
        model_kwargs={"temperature": 0.3, "max_length": 256},
    )


def build_local_llm(model_name: str = "google/flan-t5-small"):
    from langchain_core.language_models.llms import LLM

    @st.cache_resource(show_spinner=False)
    def _load(name):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        return AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name)

    tok, model = _load(model_name)

    class _LocalLLM(LLM):
        @property
        def _llm_type(self):
            return "local_flan_t5"

        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kw) -> str:
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            return tok.decode(out[0], skip_special_tokens=True).strip()

    return _LocalLLM()


# ============================================================
# Q&A Logic
# ============================================================

QA_TEMPLATE_GROQ = """You are an expert AI document assistant chatbot.
Read the following excerpts from the user's uploaded document and answer their question
in a clear, detailed, accurate, and conversational manner.

If asked for a summary or overview, provide a well-structured summary with key bullet points.
If the information is not in the excerpts, politely let the user know.

--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {question}

Detailed Answer:"""

QA_TEMPLATE_SMALL = """Context from document:
{context}

Question: {question}

Answer based on the document:"""


def ask_question(llm, retriever, question: str, is_groq: bool):
    """Run retrieval + LLM answer generation."""
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    template = QA_TEMPLATE_GROQ if is_groq else QA_TEMPLATE_SMALL
    prompt = template.format(context=context, question=question)

    try:
        if is_groq:
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip()
        else:
            answer = llm.invoke(prompt).strip()
    except Exception as e:
        answer = f"Error from AI model: {e}"

    if not answer or answer.lower() in ("", "unanswerable", "none"):
        if docs:
            snippets = "\n\n".join(d.page_content.strip() for d in docs[:3])
            answer = f"Here is the most relevant information from your document:\n\n{snippets}"
        else:
            answer = "I couldn't find relevant information for that question. Try rephrasing."

    return answer


# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.set_page_config(page_title="Ask your PDF & Documents", page_icon="📄", layout="wide")
    st.header("📄 Ask Your PDF & Documents")

    # --- Session State ---
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "is_groq" not in st.session_state:
        st.session_state.is_groq = True
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ready" not in st.session_state:
        st.session_state.ready = False

    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ AI Configuration")

        model_choice = st.selectbox("Choose AI Model:", [
            "⭐ Groq AI (Ultra Fast & Smart Chatbot)",
            "Local – flan-t5-small (Offline, No Key)",
            "Local – flan-t5-base (Offline, No Key)",
            "Hugging Face API – flan-t5-large (Online)",
        ])

        api_key = ""
        if "Groq" in model_choice:
            api_key = st.text_input(
                "⚡ Groq API Key:",
                value=DEFAULT_GROQ_KEY,
                type="password",
                help="Your Groq API key is pre-configured and ready to use."
            )
            st.success("✅ Groq LPU engine connected (openai/gpt-oss-120b).")

        elif "Hugging Face" in model_choice:
            default_tok = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
            api_key = st.text_input(
                "🔑 HuggingFace Token:",
                value=default_tok if default_tok not in ("", "your_token_here") else "",
                type="password",
                help="Free token → https://huggingface.co/settings/tokens",
            )
            if not api_key:
                st.info("💡 Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)")
        else:
            st.caption("✅ Runs 100% locally on your CPU/GPU — no API key needed.")

        st.divider()
        st.subheader("📁 Upload Documents")
        uploaded = st.file_uploader(
            "Upload files (PDF, DOCX, TXT, CSV):",
            type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=True,
        )
        process_btn = st.button("🚀 Process Documents", use_container_width=True)

    # --- Process ---
    if process_btn:
        if not uploaded:
            st.warning("⚠️ Please upload at least one file before processing.")
        else:
            with st.spinner("Reading documents & building vector index…"):
                try:
                    text = get_files_text(uploaded)
                    if not text.strip():
                        st.error("No readable text found in the uploaded documents.")
                        return

                    chunks = get_text_chunks(text)
                    vs = get_vectorstore(chunks)
                    retriever = vs.as_retriever(search_kwargs={"k": 4})

                    # Build selected LLM
                    is_groq = False
                    if "Groq" in model_choice:
                        key_to_use = api_key.strip() or DEFAULT_GROQ_KEY
                        if not key_to_use:
                            st.error("Please enter a valid Groq API key in the sidebar.")
                            return
                        llm = build_groq_llm(key_to_use)
                        is_groq = True
                    elif "Hugging Face" in model_choice:
                        if not api_key:
                            st.error("Please paste your HuggingFace token in the sidebar.")
                            return
                        llm = build_hf_hub_llm(api_key)
                    elif "base" in model_choice:
                        llm = build_local_llm("google/flan-t5-base")
                    else:
                        llm = build_local_llm("google/flan-t5-small")

                    st.session_state.llm = llm
                    st.session_state.retriever = retriever
                    st.session_state.is_groq = is_groq
                    st.session_state.chat_history = []
                    st.session_state.ready = True
                    st.success("✅ Documents processed successfully! You can now ask questions below.")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

    # --- Chat History ---
    for msg in st.session_state.chat_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # --- Chat Input ---
    if st.session_state.ready:
        user_q = st.chat_input("Ask a question about your documents…")
        if user_q:
            # Display user message immediately
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_q)

            # Generate and stream/display answer
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing document and thinking…"):
                    answer = ask_question(
                        st.session_state.llm,
                        st.session_state.retriever,
                        user_q,
                        st.session_state.is_groq,
                    )
                st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.info("👈 Upload your files in the sidebar and click **Process Documents** to start chatting.")


if __name__ == "__main__":
    main()
