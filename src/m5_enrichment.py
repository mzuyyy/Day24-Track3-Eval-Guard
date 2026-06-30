from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Lam giau chunks TRUOC khi embed.

Test: pytest tests/test_m5.py
"""

import os, sys, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def summarize_chunk(text: str) -> str:
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tom tat doan van sau trong 2-3 cau ngan gon bang tieng Viet."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  OpenAI summarize failed: {e}")

    # Extractive fallback:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dua tren doan van, tao {n_questions} cau hoi ma doan van co the tra loi. Tra ve moi cau hoi tren 1 dong."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:n_questions]
        except Exception as e:
            print(f"  OpenAI HyQA failed: {e}")

    # Extractive fallback:
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


def contextual_prepend(text: str, document_title: str = "") -> str:
    prefix = f"Trich tu {document_title}. " if document_title else ""
    return f"{prefix}{text}"


def extract_metadata(text: str) -> dict:
    if OPENAI_API_KEY:
        try:
            import json as _json
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trich xuat metadata tu doan van. Tra ve JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return _json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  OpenAI metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


def _enrich_single_call(text: str, source: str) -> dict:
    if OPENAI_API_KEY:
        try:
            import json as _json
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Phan tich doan van va tra ve JSON:
{
  "summary": "tom tat 2-3 cau",
  "questions": ["cau hoi 1", "cau hoi 2", "cau hoi 3"],
  "context": "1 cau mo ta doan van nam o dau trong tai lieu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tai lieu: {source}\n\nDoan van:\n{text}"},
                ],
                max_tokens=400,
            )
            return _json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  Enrichment API failed: {e}")
    return {}


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam."
    print(f"Original: {sample}")
    s = summarize_chunk(sample)
    print(f"Summary: {s}")
    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}")
