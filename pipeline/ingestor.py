"""
DocMind RAG — Stage 1 & 2: Document Ingestion + Chunking
"""

import os
import re
import hashlib
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    char_count: int = field(init=False)
    chunk_id: str = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)
        raw = f"{self.source}_{self.chunk_index}_{self.text[:50]}"
        self.chunk_id = hashlib.md5(raw.encode()).hexdigest()[:16]


class DocumentIngestor:
    SUPPORTED_FORMATS = {".txt", ".md", ".pdf", ".csv"}

    def __init__(self, chunk_size: int = 600, overlap: int = 120):
        self.chunk_size = chunk_size * 4
        self.overlap = overlap * 4

    def ingest(self, filepath: str) -> List[Chunk]:
        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")

        if ext == ".pdf":
            raw_text = self._read_pdf(filepath)
        elif ext == ".csv":
            raw_text = self._read_csv(filepath)
        else:
            raw_text = self._read_text(filepath)

        raw_text = self._clean_text(raw_text)
        if not raw_text.strip():
            raise ValueError("No extractable text found in document.")

        pieces = self._recursive_split(raw_text)
        total = len(pieces)

        return [
            Chunk(
                text=piece,
                source=filename,
                chunk_index=idx,
                total_chunks=total,
            )
            for idx, piece in enumerate(pieces)
        ]

    def _read_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _read_pdf(self, path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages.append(extracted)
            return "\n\n".join(pages)
        except ImportError:
            try:
                import PyPDF2
                text = ""
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                return text
            except ImportError:
                raise ImportError("Install pypdf: pip install pypdf")

    def _read_csv(self, path: str) -> str:
        import csv
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        return "\n".join(rows)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        replacements = {
            "\u2014": "-", "\u2013": "-",
            "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"',
            "\u2026": "...", "\u00a0": " ",
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        text = text.encode("ascii", errors="replace").decode("ascii")
        return text.strip()

    def _recursive_split(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for separator in ["\n\n\n", "\n\n", "\n", ". ", " "]:
            parts = text.split(separator)
            if len(parts) > 1:
                chunks = self._merge_splits(parts, separator)
                if chunks:
                    return chunks

        return [
            text[i:i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.overlap)
            if text[i:i + self.chunk_size].strip()
        ]

    def _merge_splits(self, parts: List[str], sep: str) -> List[str]:
        chunks = []
        current = ""
        prev_end = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(prev_end + current.strip())
                    prev_end = current[-self.overlap:] if len(current) > self.overlap else current
                current = part.strip()

        if current.strip():
            chunks.append(prev_end + current.strip())

        return [c for c in chunks if len(c.strip()) > 30]