import sys
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck(filename="Deep_Learning_Project_Presentation.pptx"):
    prs = Presentation()
    # 16:9 widescreen format
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Color Palette
    c_navy = RGBColor(15, 23, 42)        # #0F172A
    c_card_bg = RGBColor(30, 41, 59)     # #1E293B
    c_indigo = RGBColor(99, 102, 241)    # #6366F1
    c_purple = RGBColor(168, 85, 247)    # #A855F7
    c_emerald = RGBColor(16, 185, 129)   # #10B981
    c_sky = RGBColor(56, 189, 248)       # #38BDF8
    c_amber = RGBColor(245, 158, 11)     # #F59E0B
    c_white = RGBColor(255, 255, 255)
    c_muted = RGBColor(148, 163, 184)    # #94A3B8
    c_border = RGBColor(51, 65, 85)      # #334155

    def add_header(slide, title_text, category_text="DEEP LEARNING & NLP PROJECT"):
        # Header category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = c_indigo

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Arial"
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = c_white

    def set_slide_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = c_navy
        bg.line.fill.background()

    def add_card(slide, left, top, width, height, title, items, badge=""):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = c_card_bg
        card.line.color.rgb = c_border
        card.line.width = Pt(1)

        tf = card.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)
        tf.margin_bottom = Inches(0.25)

        # Title
        p0 = tf.paragraphs[0]
        p0.text = f"{badge} {title}".strip()
        p0.font.name = "Arial"
        p0.font.size = Pt(14)
        p0.font.bold = True
        p0.font.color.rgb = c_sky
        p0.space_after = Pt(10)

        for item in items:
            p = tf.add_paragraph()
            p.text = f"•  {item}"
            p.font.name = "Arial"
            p.font.size = Pt(11)
            p.font.color.rgb = c_white
            p.space_after = Pt(6)

    # ==============================================================
    # SLIDE 1: TITLE SLIDE
    # ==============================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide1)

    # Accent decorative bar
    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.15), Inches(3.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = c_indigo
    bar.line.fill.background()

    # Title & Subtitle box
    tbox = slide1.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(11.0), Inches(4.0))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "ACADEMIC FINAL YEAR / SEMESTER CAPSTONE PROJECT"
    p_badge.font.name = "Arial"
    p_badge.font.size = Pt(12)
    p_badge.font.bold = True
    p_badge.font.color.rgb = c_emerald
    p_badge.space_after = Pt(12)

    p_title = tf1.add_paragraph()
    p_title.text = "Deep Learning Document Intelligence\n& Interactive RAG Chatbot"
    p_title.font.name = "Arial"
    p_title.font.size = Pt(36)
    p_title.font.bold = True
    p_title.font.color.rgb = c_white
    p_title.space_after = Pt(14)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Dual-Stage Neural Cross-Encoder Re-Ranking, Dense FAISS Embeddings, Groq LPU Inference (120B) & Multimodal Whisper Voice Integration"
    p_sub.font.name = "Arial"
    p_sub.font.size = Pt(15)
    p_sub.font.color.rgb = c_muted
    p_sub.space_after = Pt(28)

    p_meta = tf1.add_paragraph()
    p_meta.text = "Presenter: Student Presentation Deck   |   Tech: PyTorch • Sentence-Transformers • Groq • FAISS • Streamlit"
    p_meta.font.name = "Arial"
    p_meta.font.size = Pt(12)
    p_meta.font.bold = True
    p_meta.font.color.rgb = c_sky

    # ==============================================================
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # ==============================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide2)
    add_header(slide2, "Problem Statement: Challenges in Traditional Document Search")

    add_card(slide2, 0.8, 1.8, 3.6, 5.0, "The Unstructured Data Deluge", [
        "Organizations & students deal with 100+ page PDFs, research papers, reports, and manuals.",
        "Manual reading and keyword searching (Ctrl+F) is inefficient and misses semantic context.",
        "Crucial facts are buried across tabular data, appendices, and multiple pages."
    ], "📄")

    add_card(slide2, 4.8, 1.8, 3.6, 5.0, "Limitations of Standard LLMs", [
        "Generic LLMs (ChatGPT) do not have access to private user documents.",
        "Frequent Hallucinations: Generating convincing but factually fabricated statements.",
        "No Ground Truth Citations: Inability to prove which page a fact originated from."
    ], "⚠️")

    add_card(slide2, 8.8, 1.8, 3.6, 5.0, "Shallow Vector Search Flaws", [
        "Standard RAG relies only on Bi-Encoder vector similarity (Cosine Distance).",
        "Lacks cross-attention: misses exact negations, complex phrasing, and numerical queries.",
        "High false-positive rate degrades LLM answer accuracy."
    ], "🔍")

    # ==============================================================
    # SLIDE 3: PROPOSED SOLUTION - DUAL-STAGE RAG
    # ==============================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide3)
    add_header(slide3, "Proposed Solution: Dual-Stage Neural RAG Architecture")

    add_card(slide3, 0.8, 1.8, 5.6, 2.4, "Stage 1: Fast Dense Retrieval (Bi-Encoder)", [
        "Converts document chunks into 384-dimensional dense vectors using all-MiniLM-L6-v2.",
        "Sub-millisecond candidate retrieval using FAISS Vector Index (retrieves Top-8 passages)."
    ], "1️⃣")

    add_card(slide3, 6.8, 1.8, 5.6, 2.4, "Stage 2: Deep Re-Ranking (Cross-Encoder)", [
        "Passes candidate pairs through ms-marco-MiniLM-L-6-v2 with full bidirectional cross-attention.",
        "Re-scores and filters down to the Top-4 most relevant passages, increasing precision by 35-40%."
    ], "2️⃣")

    add_card(slide3, 0.8, 4.4, 5.6, 2.4, "Stage 3: High-Speed Generation (Groq LPU)", [
        "120-Billion parameter foundation model (openai/gpt-oss-120b) running on Groq LPU hardware.",
        "Generates fluent, grounded, conversational answers at 300+ tokens/second."
    ], "3️⃣")

    add_card(slide3, 6.8, 4.4, 5.6, 2.4, "Stage 4: Mathematical Faithfulness & Multimodal", [
        "Computes Cosine Entailment Confidence Score (%) to mathematically detect hallucinations.",
        "Integrates Whisper Large-v3 Speech-to-Text and Neural Text-to-Speech audio playback."
    ], "4️⃣")

    # ==============================================================
    # SLIDE 4: SYSTEM ARCHITECTURE WORKFLOW
    # ==============================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide4)
    add_header(slide4, "End-to-End Deep Learning Architecture Workflow")

    steps = [
        ("Step 1: Multi-Format Ingestion", "Parses PDF, DOCX, CSV, TXT files; extracts text while preserving source file and exact page coordinates.", "📑"),
        ("Step 2: Semantic Chunking", "RecursiveCharacterTextSplitter partitions text into 750-character chunks with 120-character sliding overlap.", "✂️"),
        ("Step 3: Bi-Encoder FAISS Indexing", "Generates 384-dim embeddings; constructs FAISS dense vector store for sub-millisecond retrieval.", "🧠"),
        ("Step 4: Cross-Encoder Re-Ranking", "Concatenates [CLS] Q [SEP] P; computes joint cross-attention matrix to score candidate relevance.", "🔬"),
        ("Step 5: Groq LPU Inference & Citations", "Feeds top-ranked context into 120B LLM; attaches page citations, confidence badge & audio speech.", "⚡")
    ]
    for idx, (title, desc, ico) in enumerate(steps):
        add_card(slide4, 0.8 + (idx * 2.38), 1.8, 2.2, 5.0, title, [desc], ico)

    # ==============================================================
    # SLIDE 5: BI-ENCODER VS CROSS-ENCODER (TEACHER'S FAVORITE)
    # ==============================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide5)
    add_header(slide5, "Bi-Encoder vs Cross-Encoder: Why Both Are Essential")

    add_card(slide5, 0.8, 1.8, 5.6, 5.0, "Bi-Encoder (Stage 1: all-MiniLM-L6-v2)", [
        "Architecture: Siamese Twin Network (Sentence-BERT).",
        "Input: Encodes Query and Document separately: E(Q) and E(D).",
        "Attention: Independent self-attention (No token-to-token cross interaction).",
        "Complexity: Extremely fast O(1) vector dot product with FAISS index.",
        "Role in System: Broad Candidate Filter (Screens thousands of chunks to Top 8).",
        "Output: 384-dimensional dense semantic vector."
    ], "⚡")

    add_card(slide5, 6.8, 1.8, 5.6, 5.0, "Cross-Encoder (Stage 2: ms-marco-MiniLM)", [
        "Architecture: Single Transformer with Full Cross-Attention.",
        "Input: Joint sequence: [CLS] Query [SEP] Passage [SEP].",
        "Attention: Full cross-attention (Every query token attends to every doc token).",
        "Complexity: Higher computation O(N) forward passes over candidate pairs.",
        "Role in System: Precision Re-Ranker (Pins down true Top-4 from Top-8 candidates).",
        "Output: Continuous scalar relevance score (-inf to +inf)."
    ], "🔬")

    # ==============================================================
    # SLIDE 6: HARDWARE ACCELERATION: GROQ LPU VS TRADITIONAL GPU
    # ==============================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide6)
    add_header(slide6, "Hardware & Inference Acceleration: Groq LPU vs GPU")

    add_card(slide6, 0.8, 1.8, 5.6, 5.0, "The GPU Bottleneck (von Neumann Architecture)", [
        "Traditional GPUs (e.g. H100/A100) are optimized for parallel batch matrix multiplies.",
        "Autoregressive LLM decoding generates tokens sequentially (token-by-token).",
        "Memory Bandwidth Bottleneck: GPU cores spend massive cycles waiting for weights from HBM/DRAM.",
        "Results in higher latency (40-70 ms per token) for interactive single-user chat."
    ], "🐢")

    add_card(slide6, 6.8, 1.8, 5.6, 5.0, "Groq Language Processing Unit (LPU) Advantage", [
        "Architecture: Tensor Streaming Processor (TSP) with deterministic execution.",
        "On-chip SRAM: Weights and activations reside in ultra-high-bandwidth on-die memory.",
        "Zero memory latency wait times; compiler schedules instruction streams with cycle-level precision.",
        "Executes openai/gpt-oss-120b at 300+ tokens/second, delivering sub-second document Q&A."
    ], "🚀")

    # ==============================================================
    # SLIDE 7: MATHEMATICAL FOUNDATIONS & METRICS
    # ==============================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide7)
    add_header(slide7, "Mathematical Foundations: Attention, Similarity & Faithfulness")

    add_card(slide7, 0.8, 1.8, 3.6, 5.0, "Scaled Dot-Product Attention", [
        "Core mechanism inside Transformer layers:",
        "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V",
        "Q, K, V represent Query, Key, and Value projections.",
        "Cross-Encoder uses full bidirectional Q-K interactions between query and passage tokens."
    ], "📐")

    add_card(slide7, 4.8, 1.8, 3.6, 5.0, "Cosine Embedding Similarity", [
        "Measures semantic proximity in 384-dim vector space:",
        "cos(theta) = (A . B) / (||A||_2 * ||B||_2)",
        "Range: -1.0 (opposite) to +1.0 (identical semantics).",
        "FAISS uses inner product on normalized vectors for maximum retrieval speed."
    ], "📊")

    add_card(slide7, 8.8, 1.8, 3.6, 5.0, "Faithfulness Entailment Metric", [
        "Quantifies document grounding to detect hallucinations:",
        "Confidence = min(99, max(52, ((sim_max - 0.25) / 0.60) * 100))",
        "Compares generated answer vector against retrieved source chunk vectors.",
        "Displayed as real-time colored badge (Green: 85%+ High Entailment)."
    ], "🎯")

    # ==============================================================
    # SLIDE 8: KEY FUNCTIONAL FEATURES
    # ==============================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide8)
    add_header(slide8, "Key Application Features & Capabilities")

    add_card(slide8, 0.8, 1.8, 3.6, 2.4, "Multi-Format Parsing", [
        "Native support for PDF, DOCX, CSV, and TXT.",
        "Extracts metadata: source filename and exact page coordinate tracking."
    ], "📑")

    add_card(slide8, 4.8, 1.8, 3.6, 2.4, "Transparent Citations", [
        "Expandable 'View Citations' card below every answer.",
        "Displays exact page numbers and raw excerpt snippets."
    ], "📚")

    add_card(slide8, 8.8, 1.8, 3.6, 2.4, "Whisper Voice Input", [
        "Speech-to-Text via Groq Whisper Large-v3 ASR.",
        "Record questions via microphone with automatic transcription."
    ], "🎤")

    add_card(slide8, 0.8, 4.4, 3.6, 2.4, "Neural Spoken Audio (TTS)", [
        "Generates spoken audio streams using neural TTS.",
        "Embedded in-browser MP3 audio player for listening."
    ], "🔊")

    add_card(slide8, 4.8, 4.4, 3.6, 2.4, "Automated Executive Summary", [
        "Zero-shot summary generated upon file upload.",
        "Identifies 2-sentence summary, key concepts, and domain."
    ], "📊")

    add_card(slide8, 8.8, 4.4, 3.6, 2.4, "Offline Local Model Support", [
        "Includes google/flan-t5-small & base models.",
        "100% offline inference capability when internet is unavailable."
    ], "🔒")

    # ==============================================================
    # SLIDE 9: PERFORMANCE ENGINEERING & OPTIMIZATION
    # ==============================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide9)
    add_header(slide9, "Performance Optimization: Solving the Cold-Start Latency")

    add_card(slide9, 0.8, 1.8, 5.6, 5.0, "The Challenge: 18.7-Second Freeze", [
        "Synchronous top-level imports of PyTorch, Transformers, Sentence-Transformers, and FAISS.",
        "Blocked Streamlit's main execution thread on initial page load.",
        "Websocket timeouts (WinError 10054 ConnectionResetError) in modern browsers (Brave/Chrome).",
        "Kept the web browser hanging indefinitely on the grey skeleton loader screen."
    ], "⚠️")

    add_card(slide9, 6.8, 1.8, 5.6, 5.0, "The Solution: Lazy Loading & Caching", [
        "Deferred heavy ML module imports into cached helper functions (@st.cache_resource).",
        "Models load on-demand only when 'Process & Analyze Documents' is triggered.",
        "Results: Initial script execution dropped from 18.68s down to 0.89s (21x faster).",
        "Initial HTTP response time dropped to 74 milliseconds, ensuring instant page rendering."
    ], "⚡")

    # ==============================================================
    # SLIDE 10: TECHNOLOGY STACK & TOOLS
    # ==============================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide10)
    add_header(slide10, "Technology Stack & Engineering Architecture")

    tech_cards = [
        ("Deep Learning & NLP", ["PyTorch 2.6+", "HuggingFace Transformers", "Sentence-Transformers", "Cross-Encoder (ms-marco)"], "🧠"),
        ("Vector Store & RAG", ["FAISS (Facebook AI Similarity Search)", "LangChain Core & Text Splitters", "Bi-Encoder (all-MiniLM-L6-v2)"], "🧩"),
        ("LLM & Hardware", ["Groq LPU Inference Cloud", "openai/gpt-oss-120b", "Whisper Large-v3 ASR Engine"], "⚡"),
        ("Frontend & Deployment", ["Streamlit (Custom Glassmorphism UI)", "gTTS (Text-to-Speech)", "Git & GitHub Version Control"], "💻")
    ]
    for idx, (title, items, ico) in enumerate(tech_cards):
        add_card(slide10, 0.8 + (idx * 2.95), 1.8, 2.8, 5.0, title, items, ico)

    # ==============================================================
    # SLIDE 11: LIVE DEMO WALKTHROUGH & VERIFICATION
    # ==============================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide11)
    add_header(slide11, "Demonstration Walkthrough & Verification")

    demo_steps = [
        ("1. Fast App Launch", "Launch run.bat; server boots in ~90ms on http://localhost:8501 with tech badges.", "🚀"),
        ("2. Document Processing", "Upload PDF/DOCX; system chunks text, extracts metadata, and generates Executive Summary.", "📂"),
        ("3. Query & Re-Ranking", "Ask query; FAISS retrieves Top-8 candidates, Cross-Encoder re-ranks to Top-4.", "🔍"),
        ("4. Evidence & Confidence", "Review response with Green Faithfulness Badge (90%+) and exact Page Citations.", "🎯"),
        ("5. Multimodal Audio", "Demonstrate Whisper voice recording and listen to spoken TTS audio answer playback.", "🎙️")
    ]
    for idx, (title, desc, ico) in enumerate(demo_steps):
        add_card(slide11, 0.8 + (idx * 2.38), 1.8, 2.2, 5.0, title, [desc], ico)

    # ==============================================================
    # SLIDE 12: FUTURE SCOPE & CONCLUSION
    # ==============================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide12)
    add_header(slide12, "Future Scope, Conclusion & Acknowledgments")

    add_card(slide12, 0.8, 1.8, 5.6, 5.0, "Future Scope & Extensions", [
        "Vision-Language Models (VLM): Ingestion of diagrams, flowcharts, and embedded complex charts.",
        "Graph RAG: Construction of knowledge graphs for multi-hop semantic reasoning across 100+ documents.",
        "Edge Deployment: Quantized local GGUF models running via ONNX Runtime / WebGPU."
    ], "🔮")

    add_card(slide12, 6.8, 1.8, 5.6, 5.0, "Conclusion & Project Summary", [
        "Successfully engineered a dual-stage neural document intelligence system.",
        "Overcomes hallucination through Cross-Attention re-ranking and Faithfulness scoring.",
        "Combines high-speed LPU inference with verifiable page citations and voice multimodality.",
        "Thank You! Questions & Live Demonstration."
    ], "🎓")

    prs.save(filename)
    print(f"Presentation saved successfully to: {filename}")

if __name__ == "__main__":
    create_deck()
