"""
DocMind RAG — Stage 3 & 4: Embedding + Vector Store
Converts text chunks into dense vectors and manages ChromaDB storage.
"""

import os
from typing import List, Dict, Any
from pipeline.ingestor import Chunk


class KnowledgeStore:
    """
    Manages the vector knowledge base.

    Uses ChromaDB with sentence-transformers for local, free embeddings.
    Embedding model: all-MiniLM-L6-v2 (22M params, fast, good quality)
    """

    COLLECTION = "docmind_kb"

    def __init__(self, persist_dir: str = "./chroma_store"):
        import chromadb
        from chromadb.utils import embedding_functions

        self._client = chromadb.PersistentClient(path=persist_dir)

        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write operations ───────────────────────────────────────────────────────

    def store(self, chunks: List[Chunk]) -> int:
        """
        Embed and store chunks. Returns count of stored items.
        Chunks with duplicate IDs are silently skipped (idempotent).
        """
        if not chunks:
            return 0

        # Check for already-stored IDs to avoid duplicates
        existing = set(self._col.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]

        if not new_chunks:
            return 0

        self._col.add(
            ids=[c.chunk_id for c in new_chunks],
            documents=[c.text for c in new_chunks],
            metadatas=[
                {
                    "source":       c.source,
                    "chunk_index":  c.chunk_index,
                    "total_chunks": c.total_chunks,
                    "char_count":   c.char_count,
                }
                for c in new_chunks
            ],
        )
        return len(new_chunks)

    def remove_source(self, filename: str):
        """Remove all chunks belonging to a specific source document."""
        results = self._col.get(where={"source": filename}, include=["metadatas"])
        if results["ids"]:
            self._col.delete(ids=results["ids"])

    def purge(self):
        """Wipe the entire knowledge base."""
        self._client.delete_collection(self.COLLECTION)
        import chromadb
        from chromadb.utils import embedding_functions
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Read operations ────────────────────────────────────────────────────────

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most semantically similar chunks to the query.
        Returns list of dicts with text, source, similarity score.
        """
        total = self._col.count()
        if total == 0:
            return []

        k = min(top_k, total)
        raw = self._col.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i, doc in enumerate(raw["documents"][0]):
            cosine_sim = round(1.0 - raw["distances"][0][i], 4)
            meta = raw["metadatas"][0][i]
            hits.append(
                {
                    "text":          doc,
                    "source":        meta.get("source", "unknown"),
                    "chunk_index":   meta.get("chunk_index", 0),
                    "total_chunks":  meta.get("total_chunks", 1),
                    "similarity":    cosine_sim,
                    "relevance_pct": int(cosine_sim * 100),
                }
            )

        # Sort by similarity descending (ChromaDB usually does this, but ensure)
        hits.sort(key=lambda x: x["similarity"], reverse=True)
        return hits

    # ── Stats ──────────────────────────────────────────────────────────────────

    def total_chunks(self) -> int:
        return self._col.count()

    def indexed_files(self) -> List[Dict[str, Any]]:
        """Returns list of unique source files with their chunk counts."""
        if self._col.count() == 0:
            return []

        results = self._col.get(include=["metadatas"])
        tally: Dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            tally[src] = tally.get(src, 0) + 1

        return [
            {"filename": fname, "chunks": count}
            for fname, count in sorted(tally.items())
        ]
