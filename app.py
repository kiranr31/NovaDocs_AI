"""
DocMind RAG — Main Flask Application
Course: Prompt Engineering | Chanakya University — School of Engineering
Instructor: Mr. Deepak B | Topic: Retrieval-Augmented Generation
"""

import os
import traceback
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from pipeline.ingestor import DocumentIngestor
from pipeline.embedder import KnowledgeStore
from pipeline.generator import AnswerGenerator, FREE_MODELS, DEFAULT_MODEL

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload cap

DOCS_DIR = "documents"
os.makedirs(DOCS_DIR, exist_ok=True)

# ── Pipeline components (initialised once at startup) ─────────────────────────
ingestor  = DocumentIngestor()        # load + chunk
store     = KnowledgeStore()          # embed + ChromaDB
generator = AnswerGenerator()         # OpenRouter LLM

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    """Return list of available LLM models."""
    return jsonify({
        "models":  generator.available_models(),
        "default": DEFAULT_MODEL,
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return knowledge base statistics."""
    return jsonify({
        "total_chunks": store.total_chunks(),
        "indexed_files": store.indexed_files(),
    })


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Stage 1–4: Ingest a document into the knowledge base.
    Accepts multipart/form-data with optional chunk_size and overlap fields.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file attached"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = {".txt", ".pdf", ".md", ".csv"}
    if ext not in allowed:
        return jsonify({
            "error": f"'{ext}' is not supported. Upload .txt, .pdf, .md, or .csv"
        }), 400

    # Optional chunking settings from form
    try:
        chunk_size = int(request.form.get("chunk_size", 600))
        overlap    = int(request.form.get("overlap",     120))
        chunk_size = max(100, min(chunk_size, 2000))
        overlap    = max(20,  min(overlap,    400))
    except ValueError:
        chunk_size, overlap = 600, 120

    # Save to disk
    save_path = os.path.join(DOCS_DIR, file.filename)
    file.save(save_path)

    # Ingest with specified settings
    try:
        custom_ingestor = DocumentIngestor(
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks = custom_ingestor.ingest(save_path)
    except Exception as exc:
        os.remove(save_path)
        return jsonify({"error": f"Ingestion failed: {str(exc)}"}), 500

    # Store in ChromaDB
    stored = store.store(chunks)

    return jsonify({
        "success":      True,
        "filename":     file.filename,
        "chunks_total": len(chunks),
        "chunks_new":   stored,
        "kb_total":     store.total_chunks(),
        "message":      (
            f"✓ '{file.filename}' indexed: {stored} new chunks "
            f"(chunk_size={chunk_size}, overlap={overlap})"
        ),
    })


@app.route("/api/query", methods=["POST"])
def query():
    """
    Stage 4–5: Retrieve relevant chunks then generate a grounded answer.
    Body: { question, top_k, model }
    """
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    if store.total_chunks() == 0:
        return jsonify({
            "error": "Knowledge base is empty. Upload at least one document first."
        }), 400

    top_k     = min(max(int(body.get("top_k", 5)), 1), 10)
    model_key = body.get("model", DEFAULT_MODEL)
    if model_key not in FREE_MODELS:
        model_key = DEFAULT_MODEL

    # Stage 4: Retrieve
    hits = store.similarity_search(question, top_k=top_k)

    if not hits:
        return jsonify({
            "answer":       "No relevant passages found for your question.",
            "question":     question,
            "sources":      [],
            "context_used": 0,
            "latency_ms":   0,
            "model_used":   model_key,
        })

    # Stage 5: Generate
    try:
        result = generator.generate(
            question=question,
            context_hits=hits,
            model_key=model_key,
        )
    except Exception:
        tb = traceback.format_exc()
        return jsonify({"error": f"Generation failed:\n{tb}"}), 500

    return jsonify(result)


@app.route("/api/delete", methods=["POST"])
def delete_document():
    """Remove a specific document from the knowledge base."""
    body = request.get_json(silent=True) or {}
    filename = (body.get("filename") or "").strip()

    if not filename:
        return jsonify({"error": "No filename specified"}), 400

    store.remove_source(filename)

    # Also delete from disk if present
    path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(path):
        os.remove(path)

    return jsonify({
        "success":  True,
        "message":  f"'{filename}' removed from knowledge base.",
        "kb_total": store.total_chunks(),
    })


@app.route("/api/purge", methods=["POST"])
def purge():
    """Wipe the entire knowledge base and all uploaded files."""
    store.purge()
    for f in os.listdir(DOCS_DIR):
        try:
            os.remove(os.path.join(DOCS_DIR, f))
        except OSError:
            pass
    return jsonify({"success": True, "message": "Knowledge base cleared."})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("\n" + "─" * 55)
    print("  DocMind RAG  |  Chanakya University")
    print(f"  Running at   →  http://localhost:{port}")
    print("─" * 55 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
