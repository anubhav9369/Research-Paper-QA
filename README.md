# 📚 PaperMind — Research Paper Q&A Engine

> Ask anything about any research paper. Get answers grounded in the actual content — with source citations.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-green?style=flat-square)
![LLaMA](https://img.shields.io/badge/LLaMA%203.1-Groq-orange?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-Pipeline-purple?style=flat-square)

---

## 🔗 Live Demo
**[research-paper--app.streamlit.app](https://research-paper--app.streamlit.app/)**


---

## 🧠 What is PaperMind?

Most AI tools for research papers work at the surface level — summarizing abstracts or searching across databases of millions of papers.

**PaperMind goes deeper.**

Upload any research paper PDF → it gets chunked, embedded, and stored in a vector database → you ask questions in natural language → the most relevant sections are retrieved → LLaMA 3.1 generates a precise, grounded answer with source citations.

### How PaperMind differs from existing tools

| Tool | What it does well | The gap |
|------|-------------------|---------|
| **Bohrium AI** | Searches 170M+ papers with citation-backed answers | Literature discovery tool — not designed for deep Q&A on your specific uploaded PDF |
| **Elicit AI** | Research question synthesis across thousands of papers | Works at abstract/metadata level, not section-level content |
| **Connected Papers** | Visualizes citation graphs to find related papers | Zero Q&A capability — discovery only, not comprehension |
| **SciSpace** | AI sidebar that explains highlighted text | Closed platform — no control over LLM, embeddings, or retrieval pipeline |
| **PaperMind** | Deep Q&A on your specific uploaded paper | Full transparency — see exactly which chunks were retrieved and why |

---

## ✨ Features

- **PDF Upload** — drag and drop any research paper, text extracted automatically
- **RAG Pipeline** — retrieval-augmented generation, not full-document prompting
- **Semantic Search Tab** — search paper chunks by meaning, not just keywords
- **Auto Paper Summary** — problem, approach, key findings, keywords, difficulty level
- **Source Citations** — every answer shows which chunks it retrieved with relevance scores
- **RAG Stats Dashboard** — architecture visualization, vector counts, query metrics
- **No Token Limits** — only the top-5 relevant chunks are sent to the LLM, not the full paper

---

## 🏗 RAG Architecture

```
PDF Upload
    ↓
PyMuPDF — text extraction
    ↓
Chunking — 500 words, 100-word overlap
    ↓
sentence-transformers — all-MiniLM-L6-v2 (384-dim embeddings)
    ↓
Pinecone — cosine similarity vector index
    ↓
──────────────────────── Query Time ────────────────────────
User Question → embed → Pinecone top-5 search
    ↓
Retrieved chunks as context
    ↓
LLaMA 3.1 8B via Groq → grounded answer with citations
```

### Why RAG instead of sending the full paper?

| Approach | Token usage | Accuracy | Works on long papers |
|----------|-------------|----------|---------------------|
| Full document in prompt | Very high | Lower (lost in middle) | ❌ Often exceeds limits |
| RAG (PaperMind) | Low (5 chunks only) | Higher (focused context) | ✅ Any length |

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | LLaMA 3.1 8B via Groq API |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` (free, local) |
| **Vector Database** | Pinecone (serverless, cosine similarity) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **UI** | Streamlit |
| **Environment** | Python 3.9+ |

---

## ⚡ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/anubhav9369/research-paper-qa.git
cd research-paper-qa
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:
```
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # optional, not used currently
```

### 5. Run
```bash
python3 -m streamlit run app.py
```

Open **http://localhost:8501**

---

## 🔑 Getting API Keys (all free)

| Key | Where to get it |
|-----|----------------|
| `PINECONE_API_KEY` | [pinecone.io](https://pinecone.io) → Sign up → API Keys |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → Create API Key |

> **Note:** `sentence-transformers` runs **locally** — no API key needed for embeddings.

---

## 📁 Project Structure

```
research-paper-qa/
├── app.py                  # Streamlit UI — landing page + dashboard
├── src/
│   ├── __init__.py
│   ├── rag_pipeline.py     # Embeddings + Pinecone upsert/search
│   ├── pdf_parser.py       # PDF text extraction + section detection
│   └── llm_qa.py           # RAG-grounded Q&A + paper summary generation
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🖥 App Screenshots

### Landing Page
Upload-first design — drop your PDF and start asking questions immediately.

### Paper Dashboard
After processing, you get an auto-generated overview:
- One-liner description
- Problem and approach
- Key findings
- Keywords
- Difficulty level (Beginner / Intermediate / Advanced)
- Field of research

### Ask Questions Tab
Natural language Q&A with source chunk citations and relevance scores.

### Semantic Search Tab
Search paper chunks directly by semantic similarity — useful for finding specific sections.

### RAG Stats Tab
Full architecture diagram + vector counts + session query metrics.


## 📄 License

MIT License — free to use, modify, and build on.
