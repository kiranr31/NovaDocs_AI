# 🧠 DocMind RAG

**Retrieval-Augmented Generation System**  
*Chanakya University · School of Engineering · Prompt Engineering*  
*Instructor: Mr. Deepak B · Reference: Lewis et al. 2020*

---

## What It Does

DocMind RAG lets you upload any document (PDF, TXT, Markdown, CSV) and then ask
natural-language questions about it. Instead of the LLM guessing from training
memory, it retrieves the most relevant passages from your documents first — then
generates a grounded, accurate answer.

```
You upload a doc  →  DocMind indexes it  →  You ask a question
                                           ↓
                              Retrieves top-K relevant passages
                                           ↓
                              Sends them + your question to LLM
                                           ↓
                              Returns a grounded, cited answer
```

---

## Five RAG Pipeline Stages

| Stage | Component | What Happens |
|-------|-----------|--------------|
| ① Ingest | `DocumentIngestor` | Load PDF/TXT/MD/CSV from disk |
| ② Chunk  | `DocumentIngestor` | Split into overlapping text segments |
| ③ Embed  | `KnowledgeStore`   | Convert chunks to 384-dim vectors (all-MiniLM-L6-v2) |
| ④ Retrieve | `KnowledgeStore` | Cosine similarity search against query embedding |
| ⑤ Generate | `AnswerGenerator` | Build grounded prompt → call OpenRouter LLM |

---

## Project Structure

```
docmind_rag/
├── app.py                  # Flask REST API (5 endpoints)
├── pipeline/
│   ├── __init__.py
│   ├── ingestor.py         # Stage 1–2: load + recursive chunk
│   ├── embedder.py         # Stage 3–4: ChromaDB vector store
│   └── generator.py        # Stage 5: OpenRouter LLM generation
├── templates/
│   └── index.html          # Single-page chat UI
├── documents/              # Uploaded documents (gitignored)
├── .env.example            # Environment variable template
├── requirements.txt
├── Procfile                # For Render / Railway deployment
├── render.yaml             # One-click Render deploy config
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.10+
- A free [OpenRouter](https://openrouter.ai) API key

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/docmind-rag.git
cd docmind-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and paste your OpenRouter API key

# 5. Run
python app.py
# → http://localhost:5000
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/models` | List available LLM models |
| GET  | `/api/stats`  | Knowledge base statistics |
| POST | `/api/upload` | Upload + index a document |
| POST | `/api/query`  | RAG query (retrieve + generate) |
| POST | `/api/delete` | Remove one document |
| POST | `/api/purge`  | Wipe entire knowledge base |

### Upload
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@notes.pdf" \
  -F "chunk_size=600" \
  -F "overlap=120"
```

### Query
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking?", "model": "llama-3.1-8b", "top_k": 5}'
```

---

## Supported Models (Free via OpenRouter)

| Key | Model |
|-----|-------|
| `llama-3.1-8b`  | Meta LLaMA 3.1 8B Instruct |
| `mistral-7b`    | Mistral 7B Instruct |
| `gemma-3-12b`   | Google Gemma 3 12B IT |
| `qwen-2.5-7b`   | Qwen 2.5 7B Instruct |
| `deepseek-r1`   | DeepSeek R1 |

---

## Deploy to Render (Free Hosting)

1. Push this repo to GitHub (see below)
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. In **Environment Variables**, add:
   ```
   OPENROUTER_API_KEY = your_key_here
   ```
6. Your app is live at `https://docmind-rag.onrender.com`

> **Note:** Render free tier spins down after 15 min inactivity. ChromaDB and
> uploaded documents are ephemeral — re-upload after a cold start.
> For persistence, upgrade to a paid plan or use an external vector DB.

---

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: DocMind RAG system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/docmind-rag.git
git push -u origin main
```

---

## Key Concepts (Assignment Reference)

**Why RAG?**  
LLMs have a knowledge cutoff and hallucinate on out-of-training topics.
RAG grounds responses in real documents at inference time.

**Chunking matters:**  
Chunks too large → irrelevant content dilutes signal.  
Chunks too small → loss of coherent context.  
Sweet spot: 500–800 tokens with 100–150 token overlap.

**Cosine similarity:**  
Measures angle between embedding vectors. Score of 1.0 = identical meaning.
Score of 0.0 = completely unrelated.

**Faithfulness:**  
A RAG answer is "faithful" if every claim is supported by the retrieved context.
This is the most important RAGAS metric.

---

*Built for Prompt Engineering Assignment · Chanakya University · 2026*
