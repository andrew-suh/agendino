import json
import logging

import httpx
from google import genai
from google.genai import types
from json_repair import repair_json

logger = logging.getLogger(__name__)

MIND_MAP_PROMPT = """You are a knowledge-mapping expert. Analyze the summaries and produce a
clean, hierarchical mind map.

RULES
1. The map has exactly THREE depth levels: central_topic → branches → children.
2. Create 3-7 branches - each is a distinct high-level THEME.
3. Each branch has 2-5 children - each is one concrete KEY INSIGHT from the summaries.
4. Labels must be SHORT: max 4 words for branches, max 6 words for children.
5. Every child MUST include `summary_ids` (array of source summary IDs).
6. Add `connections` only for genuinely cross-cutting relationships (max 3).
7. Use the same language as the summaries.
8. Do NOT repeat the same concept across branches.

Return ONLY this JSON:
{
  "central_topic": "Short overarching theme (max 4 words)",
  "branches": [
    {
      "id": "branch_1",
      "label": "Theme name",
      "children": [
        {"id": "leaf_1_1", "label": "Key insight", "summary_ids": [1, 2]}
      ]
    }
  ],
  "connections": [
    {"from": "branch_1", "to": "branch_2", "label": "relation"}
  ]
}"""

RAG_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the answer cannot be found in the context, say so clearly.
Use the same language as the question.
Format your response in Markdown.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_context(context_docs: list[dict]) -> tuple[str, list[dict]]:
    """Assemble the prompt context string and source list from retrieved docs."""
    context_parts = []
    sources = []
    for i, doc in enumerate(context_docs):
        meta = doc.get("metadata", {})
        title = meta.get("title", f"Source {i + 1}")
        text = doc.get("document", "")
        context_parts.append(f"[{title}]\n{text}")
        sources.append(
            {
                "title": title,
                "recording_name": meta.get("recording_name", ""),
                "summary_id": meta.get("summary_id", ""),
                "distance": doc.get("distance"),
            }
        )
    return "\n\n---\n\n".join(context_parts), sources


def build_mind_map_content(summaries: list[dict]) -> str:
    """Build the user content listing summaries for mind-map generation."""
    summary_texts = []
    for s in summaries:
        tags = ", ".join(s.get("tags", []))
        # Truncate summary for context window efficiency
        summary_preview = s.get("summary", "")[:600]
        entry = f"[ID: {s['id']}] Title: {s.get('title', 'Untitled')}\nTags: {tags}\n{summary_preview}"
        summary_texts.append(entry)
    return "Summaries:\n\n" + "\n\n---\n\n".join(summary_texts)


def parse_mind_map_json(raw: str) -> dict:
    """Parse mind-map JSON, repairing malformed output and falling back to an empty structure."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        repaired = repair_json(raw, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    logger.warning("Failed to parse mind map JSON, returning empty structure")
    return {"central_topic": "Knowledge Base", "branches": [], "connections": []}


class RAGService:
    def __init__(self, api_key, model: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def ask(self, question: str, context_docs: list[dict]) -> dict:
        """RAG query: answer a question using retrieved context."""
        context, sources = build_rag_context(context_docs)
        prompt = RAG_PROMPT.format(context=context, question=question)

        response = self._client.models.generate_content(
            model=self._model,
            config=types.GenerateContentConfig(
                max_output_tokens=4096,
            ),
            contents=prompt,
        )

        return {
            "answer": response.text or "",
            "sources": sources,
        }

    def generate_mind_map(self, summaries: list[dict]) -> dict:
        """Generate a mind map structure from summaries using Gemini."""
        content = build_mind_map_content(summaries)

        logger.info("Generating mind map with Gemini for %d summaries…", len(summaries))
        response = self._client.models.generate_content(
            model=self._model,
            config=types.GenerateContentConfig(
                system_instruction=MIND_MAP_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=8192,
            ),
            contents=content,
        )

        return parse_mind_map_json(response.text or "")


class OllamaRAGService:
    """RAG generation via a local Ollama server (offline), using its OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _chat(self, messages: list[dict], *, json_mode: bool = False, max_tokens: int = 4096) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = httpx.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""

    def ask(self, question: str, context_docs: list[dict]) -> dict:
        """RAG query: answer a question using retrieved context."""
        context, sources = build_rag_context(context_docs)
        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = self._chat([{"role": "user", "content": prompt}])
        return {"answer": answer, "sources": sources}

    def generate_mind_map(self, summaries: list[dict]) -> dict:
        """Generate a mind map structure from summaries using a local Ollama model."""
        content = build_mind_map_content(summaries)
        logger.info("Generating mind map with Ollama (%s) for %d summaries…", self._model, len(summaries))
        raw = self._chat(
            [
                {"role": "system", "content": MIND_MAP_PROMPT},
                {"role": "user", "content": content},
            ],
            json_mode=True,
            max_tokens=8192,
        )
        return parse_mind_map_json(raw)
