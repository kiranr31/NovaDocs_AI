"""
DocMind RAG - Stage 5: Augmented Generation
"""

import os
import time
import json
import requests
from typing import List, Dict, Any


FREE_MODELS = {

    "llama-3.1-8b":
        "meta-llama/llama-3.1-8b-instruct",

    "mistral-7b":
        "mistralai/mistral-7b-instruct",

    "gemma-3-12b":
        "google/gemma-3-12b-it",

    "qwen-2.5-7b":
        "qwen/qwen-2.5-7b-instruct",

    "deepseek-r1":
        "deepseek/deepseek-r1"
}

DEFAULT_MODEL = "llama-3.1-8b"

SYSTEM_PROMPT = """
You are DocMind, an intelligent document assistant.

Answer questions using ONLY the provided context passages.

If the context does not contain the answer, say:
"The uploaded documents don't cover this topic."

Always mention which document your answer comes from.
"""

USER_PROMPT_TEMPLATE = """
## Context Passages
{context}

---
## User Question
{question}

Answer using ONLY the context above:
"""


def sanitize(text: str) -> str:
    """
    Remove problematic Unicode characters.
    """

    replacements = {

        "\u2014": "-",     # em dash
        "\u2013": "-",     # en dash

        "\u2018": "'",     # left single quote
        "\u2019": "'",     # right single quote

        "\u201c": '"',     # left double quote
        "\u201d": '"',     # right double quote

        "\u2026": "...",   # ellipsis
        "\u00a0": " ",     # non-breaking space

        "\u2022": "-",     # bullet
        "\u00b7": "-",     # middle dot
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text


class AnswerGenerator:

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):

        self.api_key = os.getenv(
            "OPENROUTER_API_KEY",
            ""
        ).strip()

        if not self.api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file."
            )

    def generate(
        self,
        question: str,
        context_hits: List[Dict[str, Any]],
        model_key: str = DEFAULT_MODEL,
        temperature: float = 0.15,
        max_tokens: int = 900,
    ) -> Dict[str, Any]:

        model_id = FREE_MODELS.get(
            model_key,
            FREE_MODELS[DEFAULT_MODEL]
        )

        # Build context
        context_lines = []

        for i, hit in enumerate(context_hits, start=1):

            source = sanitize(
                str(hit.get("source", "Unknown"))
            )

            text = sanitize(
                str(hit.get("text", ""))
            )

            relevance = hit.get(
                "relevance_pct",
                0
            )

            line = (
                f"[Passage {i} | "
                f"Source: {source} | "
                f"Relevance: {relevance}%]\n"
                f"{text}"
            )

            context_lines.append(line)

        context_block = "\n\n---\n\n".join(
            context_lines
        )

        question_clean = sanitize(question)

        user_message = USER_PROMPT_TEMPLATE.format(
            context=context_block,
            question=question_clean,
        )

        headers = {

            "Authorization":
                f"Bearer {self.api_key}",

            "Content-Type":
                "application/json",

            "HTTP-Referer":
                "https://docmind-rag.app",

            "X-Title":
                "DocMind RAG - Chanakya University"
        }

        payload = {

            "model": model_id,

            "messages": [

                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": user_message
                }
            ],

            "temperature": temperature,

            "max_tokens": max_tokens,
        }

        # Proper UTF-8 encoding
        payload_bytes = json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8")

        start_time = time.time()

        try:

            response = requests.post(
                self.API_URL,
                headers=headers,
                data=payload_bytes,
                timeout=45,
            )

            response.raise_for_status()

            response.encoding = "utf-8"

            data = response.json()

            answer = (
                data["choices"][0]
                ["message"]["content"]
                .strip()
            )

            usage = data.get("usage", {})

        except requests.exceptions.Timeout:

            answer = (
                "Request timed out. "
                "Please try again."
            )

            usage = {}

        except requests.exceptions.HTTPError as error:

            try:
                error_text = (
                    error.response.text[:300]
                )

                status_code = (
                    error.response.status_code
                )

            except Exception:

                error_text = str(error)
                status_code = "Unknown"

            answer = (
                f"API Error {status_code}: "
                f"{error_text}"
            )

            usage = {}

        except Exception as error:

            answer = (
                f"Unexpected error: "
                f"{str(error)}"
            )

            usage = {}

        latency_ms = int(
            (time.time() - start_time) * 1000
        )

        sources = []

        for hit in context_hits:

            preview_text = sanitize(
                str(hit.get("text", ""))
            )[:180]

            sources.append({

                "filename":
                    hit.get("source", "Unknown"),

                "similarity":
                    hit.get("similarity", 0),

                "relevance_pct":
                    hit.get("relevance_pct", 0),

                "preview":
                    preview_text + "..."
            })

        return {

            "answer": answer,

            "question": question,

            "model_used": model_id,

            "model_label": model_key,

            "latency_ms": latency_ms,

            "context_used": len(context_hits),

            "sources": sources,

            "tokens_prompt":
                usage.get("prompt_tokens", 0),

            "tokens_reply":
                usage.get("completion_tokens", 0),
        }

    @staticmethod
    def available_models() -> List[Dict[str, str]]:

        return [

            {
                "key": key,

                "label":
                    key.replace("-", " ").title(),

                "id": value
            }

            for key, value
            in FREE_MODELS.items()
        ]