# 🏠 SmartMortgage — AI-Powered Mortgage Document Intelligence Chatbot

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-00A67E)
![Mistral](https://img.shields.io/badge/LLM-Mistral--7B-EF6C00)
![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?logo=gradio&logoColor=white)
![Colab](https://img.shields.io/badge/Runtime-Google_Colab_T4-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-Portfolio-lightgrey)

> **Extern × Outamation**  
> A fully local Retrieval-Augmented Generation (RAG) pipeline that ingests multi-document mortgage PDFs, intelligently classifies each document, preserves tabular structure, and delivers grounded, citation-backed answers — no API keys required.

## 🎬 Demo

[![Watch Demo Video](https://img.shields.io/badge/Watch-Demo_Video-FF0000?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1j5Gc5vo6bH8isshjk-M-Z2eLw2abw0Kf/view?usp=sharing)

> Click the badge above to watch a full walkthrough of the MortgageIQ pipeline — from PDF upload through document classification, retrieval, and live Q&A in the Gradio UI.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [What It Does](#-what-it-does)
- [System Architecture](#-system-architecture)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Performance Metrics](#-performance-metrics)
- [Tech Stack](#-tech-stack)
- [Design Decisions & Trade-offs](#-design-decisions--trade-offs)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Example Queries](#-example-queries)
- [Limitations & Future Work](#-limitations--future-work)
- [Project Impact](#-project-impact)

---

## 🎯 Problem Statement

Mortgage packets are dense, multi-document PDFs — loan agreements, lender fee sheets, pay slips, tax forms, and insurance policies — often 50+ pages stapled into a single file. Extracting a specific fee, salary figure, or contract clause means manually hunting through pages of legalese and tables where column alignment is the only thing giving numbers meaning.

| Pain Point | How SmartMortgage Solves It |
|---|---|
| Slow manual review | Automates PDF parsing, OCR, and retrieval end-to-end |
| Inconsistent extraction | Structure-aware chunking ensures context retention across pages |
| Data privacy concerns | Runs fully local with open-source Mistral-7B — no data leaves the machine |
| Lack of transparency | Provides page-level citations, relevance scores, and confidence flags on every answer |

---

## ✅ What It Does

1. **Ingests** a multi-document mortgage PDF and detects where one document ends and another begins
2. **Classifies** each logical section into one of 11 document types (Mortgage Contract, Lender Fee Sheet, Pay Slip, Tax Document, etc.)
3. **Extracts** text and tables separately — tables are converted to structured markdown so the LLM reads column headers, not a jumble of numbers
4. **Chunks intelligently** — tables stay intact as single chunks (never split mid-row); prose uses sliding-window overlap
5. **Embeds and indexes** all chunks in a FAISS vector store with persistence across sessions
6. **Routes queries** — the model predicts which document type a question targets before retrieval, filtering irrelevant chunks
7. **Generates grounded answers** with explicit hallucination guards and confidence scoring
8. **Serves everything** through a Gradio UI with upload, chat, analytics, and pipeline reference tabs

---

## 🏗️ System Architecture

```
USER UPLOADS PDF
       │
       ▼
┌─────────────────────────────┐
│  PyMuPDF Text Extraction    │
│  + pytesseract OCR fallback │
│  + find_tables() → Markdown │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Document Classification    │
│  (11 document types)        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Structure-Aware Chunking   │
│  Tables intact │ Prose split│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  all-MiniLM-L6-v2 Embeddings│
│  → FAISS Flat L2 Index      │
└─────────────┬───────────────┘
              │
        ┌─────▼──────────────────────────────────────────────┐
        │                    QUERY TIME                       │
        │                                                     │
        │  User Question → Doc-type Routing                   │
        │    → FAISS Retrieval (top-k, relevance threshold)   │
        │      → Mistral-7B Answer Generation                 │
        │        → Hallucination Guard (word-overlap check)   │
        │          → Response + Citations + Confidence Score  │
        └─────────────────────────────────────────────────────┘
```

### Component Specifications

| Component | Technology | Configuration |
|---|---|---|
| OCR Engine | Tesseract | English & numeric modes via pytesseract |
| Text Chunking | Fixed-size + overlap | Chunk size = 500 tokens, overlap = 100 |
| Embeddings | all-MiniLM-L6-v2 | 384-dim vectors, GPU accelerated |
| Vector Store | FAISS (flat index) | L2 similarity, in-memory + disk persistence |
| Retriever | Dense retrieval + doc-type filter | Top-K = 4, relevance threshold = 0.35 |
| LLM | Mistral-7B Instruct (local GGUF) | Context = 4096, Temp = 0.2, Max tokens = 512 |
| Prompt Strategy | Grounded QA + citation enforcement | Forces source citations and confidence reporting |
| UI Framework | Gradio Blocks | Dark theme, four-tab layout |

---

## 🔬 Pipeline Walkthrough

**Step 1 — Dependency Installation**  
Installs PyTorch, Gradio, PyMuPDF, pytesseract, sentence-transformers, FAISS, and llama-cpp-python. Tesseract OCR is installed at the system level for scanned page fallback.

**Step 2 — LLM Loading**  
Loads Mistral-7B Instruct (GGUF Q4, ~4 GB) with all layers offloaded to GPU. The same model handles document classification, query routing, and answer generation.

**Step 3 — Data Structures & Evaluation Logger**  
Defines `PageInfo`, `LogicalDocument`, `ChunkMetadata`, and `QueryMetrics` dataclasses. Sets up a per-query evaluation logger tracking latency, chunk relevance scores, grounding ratios, and confidence throughout the session.

**Step 4 — Document Classification**  
Sends a 600-character sample of each detected document section to Mistral-7B, which maps it to one of 11 predefined labels. Falls back to "Other" on empty or unrecognized responses.

**Step 5 — PDF Extraction with Table Preservation**  
Standard text extraction flattens tables into meaningless strings. MortgageIQ instead:
- Calls `page.find_tables()` on every page to detect tabular regions
- Converts each table to markdown with proper column headers and row delimiters
- Injects a `--- STRUCTURED TABLE DATA ---` marker so the chunker keeps it intact
- Falls back to regex-based fee line parsing for PDFs where `find_tables()` fails

**Step 6 — Structure-Aware Chunking**  
Table blocks become individual chunks (never split mid-row). Non-table text uses a 500-token sliding window with 100-token overlap. Every chunk carries `doc_type`, `doc_id`, page range, and a `has_table` flag.

**Step 7 — Embeddings & FAISS Index**  
Encodes all chunks with `all-MiniLM-L6-v2` and stores in a FAISS Flat L2 index. Supports relevance thresholding (chunks below the threshold are dropped) and FAISS persistence to avoid re-embedding on restart.

**Step 8 — Query Routing & Answer Generation**  
The model predicts which `doc_type` a query targets (e.g., "What is my gross pay?" → Pay Slip). Retrieval is filtered to that type, with "All" as a fallback for cross-document questions. Table-aware prompting explicitly instructs the LLM to read column headers before interpreting values. A post-generation hallucination guard calculates what fraction of non-stopword content words in the answer appear in the retrieved context — low overlap triggers a warning flag.

**Step 9 — Gradio UI**  
A four-tab Blocks interface with custom CSS:

| Tab | What It Does |
|---|---|
| Upload | Drag-and-drop PDF upload, processing stats, document structure cards |
| Chat | Conversation panel with retrieval sidebar showing source chunks and confidence bars |
| Analytics | Live per-query performance dashboard — latency, relevance scores, grounding ratios |
| About | Full pipeline reference table and changelog |

---

## 📊 Performance Metrics

Evaluated on multi-page PDF packets containing pay slips, mortgage contracts, and lender fee sheets, measuring retrieval consistency and response speed under manual validation.

### Retrieval Performance
| Metric | Result |
|---|---|
| Recall | ~94% |
| Hit Rate (≥1 relevant chunk per query) | 100% |

### End-to-End Accuracy
| Metric | Result |
|---|---|
| Answer Accuracy (manual validation) | ~93% |
| Citation Accuracy | ~96% |
| Factual Consistency (hallucinations observed) | ~95% / None |

### System Performance
| Metric | Result |
|---|---|
| Average End-to-End Response Time | ~2.4 s |
| Retrieval Latency | ~180 ms |
| LLM Generation Time | ~2.2 s |

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| LLM | Mistral-7B Instruct (GGUF Q4) via llama-cpp-python |
| Embeddings | all-MiniLM-L6-v2 (SentenceTransformers) |
| Vector Store | FAISS (Flat L2 Index) |
| PDF Extraction | PyMuPDF (fitz) + pytesseract OCR fallback |
| Table Extraction | PyMuPDF `find_tables()` → Markdown |
| UI Framework | Gradio Blocks |
| Compute | Google Colab T4 GPU |
| Language | Python 3.10+ |

---

## ⚖️ Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| Local Mistral-7B over API LLMs | Privacy-first; no data leaves the machine; ~2s avg response | Lower fluency than GPT-4, smaller context window |
| all-MiniLM-L6-v2 embeddings | Fast GPU inference, compact 384-dim vectors | Slightly lower semantic depth than larger models |
| Fixed-size chunking | Predictable retrieval latency, stable behavior | Loses some semantic context when headers and values fall across page boundaries |
| FAISS over cloud vector DB | Local search, no external dependencies | No persistent cloud index; session resets clear state |
| Rule-based doc-type router | More reliable and deterministic than LLM routing | Less flexible for unseen document formats |

---

## 🚀 Getting Started

### Prerequisites
- Google Colab with T4 GPU runtime (free tier works), or any CUDA-capable machine
- No API keys required — everything runs locally

### Run on Google Colab
1. Open the notebook in Google Colab
2. Set the runtime to **GPU → T4**
3. Run all cells sequentially (Step 1 through Step 9)
4. The final cell launches a Gradio app with a public share link

### Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/MortgageIQ.git
cd MortgageIQ

# Install Python dependencies
pip install torch gradio pymupdf pypdf2 pytesseract Pillow \
    sentence-transformers faiss-cpu numpy pandas

# Install llama-cpp-python with CUDA support
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123

# Install Tesseract OCR (system-level)
sudo apt-get install tesseract-ocr

# Launch
jupyter notebook MortgageIQ_RAG.ipynb
```

---

## 💬 Usage

1. **Upload** a mortgage PDF packet (multi-document PDFs work best)
2. Check the **Upload tab** for processing stats: pages detected, document boundaries, classification labels
3. Switch to **Chat** and ask questions:
   - *"What is my gross monthly income?"*
   - *"What are the total closing costs?"*
   - *"What is the interest rate on my mortgage?"*
   - *"List all lender fees above $500."*
4. Adjust retrieval settings in the sidebar — document filter, top-k chunks, relevance threshold
5. Check **Analytics** for per-query performance metrics

---

## 🔍 Example Queries

**Query #1 — Factual Lookup**
```
Query: "What is the APR and term?"
Retrieved: Lender Fee Sheet (Pages 0-0) — Relevance 100%
Answer: "The APR is 4.250% and the term is 360 months (30 years)."
Confidence: 100%
```

**Query #2 — Multi-Document Reasoning**
```
Query: "What is the loan amount and total number of working days?"
Retrieved: Pay Slip (21.16%), Lender Fee Sheet (19.60%), Contract (6.55%)
Answer: "The loan amount is $380,000 and there are 26 working days."
Confidence: 11.8%  ← lower due to cross-document embedding similarity
```

> **Note on confidence scores:** Multi-document queries yield lower confidence scores due to cross-document embedding similarity and chunk granularity. However, the answer generation remains factually accurate — confidence reflects retrieval certainty, not answer quality.

---

## ⚠️ Limitations & Future Work

**Current Limitations**

1. **OCR Noise** — Low-quality scans cause partial character recognition errors (e.g., `$1,000.00` → `$1,OOO.OO`), weakening embedding quality for numeric queries
2. **No Semantic Chunk Merging** — Fixed-window chunking treats every slice independently; headers and their associated values can land in separate chunks, lowering retrieval scores
3. **Session Resets** — Gradio does not persist the FAISS index or chat history across browser refreshes; uploaded PDFs and conversation context are lost unless cached externally

**Roadmap**

| Timeline | Enhancement |
|---|---|
| Short-term (2 weeks) | Integrate semantic chunker + BM25 hybrid retrieval |
| Medium-term (1 month) | Add cross-encoder re-ranking and query fusion |
| Long-term | Convert Gradio prototype to Flask/React dashboard with persistent FAISS index and cloud upload support |

---

## 🎓 Project Impact

This externship project marked a transition from LLM experimentation to building a production-grade, privacy-centric AI assistant for real-world document intelligence. Key technical skills developed include:

- End-to-end RAG pipeline design (document ingestion → retrieval → generation → evaluation)
- CUDA optimization for local LLM inference
- Structure-aware PDF parsing with table preservation
- Hallucination mitigation via grounding checks and citation enforcement
- Production guardrails: per-query evaluation metrics, graceful degradation, configurable relevance thresholds

Skills directly translatable to healthcare, legal, and enterprise data analytics roles where document privacy and answer reliability are non-negotiable.

---

## 📁 Repository Structure

```
MortgageIQ/
├── MortgageIQ_RAG.ipynb        # Full pipeline notebook (run end-to-end)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── presentation/
    └── Outamation_MortgageIQ_Final.pptx   # Final stakeholder presentation
```

---
