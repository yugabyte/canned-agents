from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from adlc_review_agent.review import run_review


class _FakeMeko:
    def __init__(self, kb_result: dict[str, Any]) -> None:
        self.kb_result = kb_result
        self.calls: list[dict[str, Any]] = []

    def knowledgebase_search(self, *, query: str, datapack_id: str) -> dict[str, Any]:
        self.calls.append({"query": query, "datapack_id": datapack_id})
        return self.kb_result


@dataclass
class _TextBlock:
    type: str
    text: str


def _fake_anthropic(reply_text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = MagicMock(content=[_TextBlock(type="text", text=reply_text)])
    return client


def test_run_review_grounds_prompt_in_kb_hits_and_returns_llm_text() -> None:
    meko = _FakeMeko({"results": [{"document_name": "standards.md", "chunk_text": "no bare except"}]})
    anthropic_client = _fake_anthropic("Looks like a bare except on line 20.")

    result = run_review(
        meko=meko,  # type: ignore[arg-type]
        anthropic_client=anthropic_client,
        datapack_id="dp-1",
        pr_title="Add spend endpoint",
        pr_diff="+    except:\n+        return 500",
    )

    assert result.text == "Looks like a bare except on line 20."
    assert "no bare except" in result.kb_context
    assert meko.calls == [{"query": "coding standards security requirements", "datapack_id": "dp-1"}]

    # The KB context and the diff both made it into the LLM call.
    call_kwargs = anthropic_client.messages.create.call_args.kwargs
    assert "no bare except" in call_kwargs["system"]
    assert "except:" in call_kwargs["messages"][0]["content"]


def test_run_review_handles_empty_knowledge_base() -> None:
    meko = _FakeMeko({"results": []})
    anthropic_client = _fake_anthropic("No standards found; general best practice review below.")

    result = run_review(
        meko=meko,  # type: ignore[arg-type]
        anthropic_client=anthropic_client,
        datapack_id="dp-1",
        pr_title="Add spend endpoint",
        pr_diff="+ x = 1",
    )

    assert "(none found)" in result.kb_context
