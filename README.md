# 🚀 NovaDocs AI

> AI Powered Retrieval-Augmented Generation (RAG) System for Intelligent Document Question Answering

NovaDocs AI is a modern AI-powered document intelligence platform that allows users to upload documents, index them into a vector database, and ask natural language questions using Retrieval-Augmented Generation (RAG).

Built with Flask, ChromaDB, Sentence Transformers, and OpenRouter LLMs, NovaDocs AI delivers contextual answers grounded in uploaded documents.

---

# ✨ Features

- 📄 Upload PDF, TXT, CSV, and Markdown documents
- 🧠 AI-powered semantic search
- 🔎 Vector similarity retrieval using embeddings
- 🤖 LLM-generated contextual responses
- ⚡ Fast Retrieval-Augmented Generation (RAG) pipeline
- 🎨 Premium futuristic UI
- 📚 Multi-document indexing
- 📦 ChromaDB vector database integration
- 🌐 OpenRouter LLM API support
- 📊 Source relevance visualization
- 🔐 Environment-based API key management

---

# 🧩 RAG Pipeline

NovaDocs AI follows a complete Retrieval-Augmented Generation workflow:

```text
Document Upload
      ↓
Document Chunking
      ↓
Embedding Generation
      ↓
Vector Storage (ChromaDB)
      ↓
Semantic Retrieval
      ↓
LLM Response Generation
```

---

# 🛠️ Tech Stack

## Frontend

- HTML5
- CSS3
- Vanilla JavaScript

## Backend

- Python
- Flask

## AI / ML

- Sentence Transformers
- OpenRouter API
- ChromaDB

## Deployment

- Render
- GitHub

---

# 📂 Project Structure

```text
NovaDocs_AI/
│
├── app.py
├── requirements.txt
├── render.yaml
├── Procfile
├── .gitignore
│
├── pipeline/
│   ├── __init__.py
│   ├── ingestor.py
│   ├── embedder.py
│   └── generator.py
│
├── templates/
│   └── index.html
│
└── chroma_db/
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/kiranr31/NovaDocs_AI.git
```

---

## 2️⃣ Navigate into Project

```bash
cd NovaDocs_AI
```

---

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory.

```env
OPENROUTER_API_KEY=your_api_key_here
```

---

# ▶️ Run the Application

```bash
python app.py
```

Application will start at:

```text
http://127.0.0.1:5000
```

---

# 🌐 Supported Models

NovaDocs AI supports multiple LLMs via OpenRouter:

- Llama 3.1
- Mistral
- Gemma
- Qwen
- DeepSeek

---

# 📸 Screenshots

## Dashboard

_Add your dashboard screenshot here_

## Document Upload

_Add upload interface screenshot here_

## AI Response

_Add AI response screenshot here_

---

# 🚀 Deployment

This project is deployment-ready for:

- Render
- Railway
- Replit
- VPS Hosting

## Deploy on Render

1. Connect GitHub repository
2. Add environment variable:

```text
OPENROUTER_API_KEY
```

3. Deploy web service

---

# 🔒 Security Notes

- `.env` is excluded using `.gitignore`
- API keys are never hardcoded
- Vector database is locally isolated

---

# 📈 Future Improvements

- Streaming AI responses
- Authentication system
- Chat history
- OCR support
- PDF preview
- Voice input
- Multi-user architecture
- Docker support
- LangChain integration
- Citation highlighting

---

# 👨‍💻 Author

**Kiran R**

MCA Student — Chanakya University

---

# 📜 License

This project is intended for educational and research purposes.

---

# ⭐ Support

If you like this project:

- ⭐ Star the repository
- 🍴 Fork the project
- 🚀 Contribute improvements

---

## 🌟 NovaDocs AI

AI Powered Document Intelligence & Retrieval System