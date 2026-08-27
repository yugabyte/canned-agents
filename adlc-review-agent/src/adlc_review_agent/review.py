"""The review-agent's own prompt + orchestration.

REVIEW_SYSTEM_PROMPT mirrors REVIEW_AGENT_SYSTEM_TEMPLATE in
meko/inference_gateway/inference_gateway/chain.py by convention, not by
code-sharing (this is a standalone Python package, that's a separate FastAPI
service) — if you change the framing in one place, change it in the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from .meko_client import MekoMcpClient

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

REVIEW_SYSTEM_PROMPT = """You are a senior code reviewer. Review the following
PR/diff against the org's coding standards and security requirements
retrieved from the knowledge base below. Be specific about what needs to
change and why.

Knowledge (from `knowledgebase_search`):
{kb_context}

If the knowledge base is empty, say so explicitly and review the diff
against general best practice instead of inventing standards."""


def _format_kb_hits(kb_result: Any) -> str:
    """`knowledgebase_search` returns {"results": [{"document_name" or
    "chunk_text"/"text", ...}, ...]}. Loosely typed here since the exact
    shape isn't a stable, versioned contract this package depends on."""
    results = (kb_result or {}).get("results") or []
    if not results:
        return "(none found)"
    lines = []
    for hit in results:
        doc = hit.get("document_name", "knowledge base")
        text = hit.get("chunk_text") or hit.get("text") or ""
        lines.append(f"- [{doc}] {text.strip()}")
    return "\n".join(lines)


@dataclass
class ReviewResult:
    text: str
    kb_context: str


def run_review(
    *,
    meko: MekoMcpClient,
    anthropic_client: Anthropic,
    datapack_id: str,
    pr_title: str,
    pr_diff: str,
    model: str = DEFAULT_MODEL,
) -> ReviewResult:
    kb_result = meko.knowledgebase_search(
        query="coding standards security requirements", datapack_id=datapack_id
    )
    kb_context = _format_kb_hits(kb_result)

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=2000,
        system=REVIEW_SYSTEM_PROMPT.format(kb_context=kb_context),
        messages=[
            {
                "role": "user",
                "content": f"Title: {pr_title}\n\n```diff\n{pr_diff}\n```",
            }
        ],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return ReviewResult(text=text, kb_context=kb_context)
