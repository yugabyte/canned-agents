"""AWS Bedrock AgentCore Runtime entrypoint. Thin wrapper around the same
core functions cli.py uses — see cli.py for the CLI equivalent of this same
flow. Payload fields mirror the CLI's flags 1:1, as JSON instead of
argv/env, except for the LLM credential: this entrypoint uses
`AnthropicBedrock` (ambient AWS credentials from the deployment's execution
role) instead of a per-request Anthropic API key, so `anthropic_api_key` is
not part of the payload here. `review.py` and the CLI stay untouched —
`AnthropicBedrock` implements the same `.messages.create()`/`.messages.stream()`
interface `Anthropic` does, so `review.py` doesn't need to know which one it got.

`handler()` is a generator: the installed `bedrock_agentcore` SDK detects an
async/sync generator entrypoint and streams each yielded value as an SSE
frame (`data: {json}\\n\\n`, `text/event-stream`) automatically — see
runtime/app.py's `_sync_stream_with_error_handling`. That same SDK machinery
also catches any exception raised mid-generator and synthesizes its own
`{"error": ..., "error_type": ..., "message": ...}` SSE frame, so this file
does NOT manually catch MekoMcpError (or anything else) once streaming has
started -- only the pre-generator validation below, which still returns a
plain (non-streaming) `{"error": ...}` dict, exactly as before this file
supported streaming at all.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

from anthropic import AnthropicBedrock
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from .github_client import GitHubApiError, fetch_pull_request, parse_pr_ref
from .meko_client import MekoMcpClient
from .review import (
    ReviewConfig,
    STRICTNESS_FRAGMENTS,
    TONE_FRAGMENTS,
    _create_conversation,
    run_followup_stream,
    run_review_stream,
)
from .slack_client import SlackApiError, post_to_channel

DEFAULT_MEKO_MCP_URL = "https://mcp.mekodata.ai/mcp"

# Bedrock model IDs use a different format than the plain Anthropic API --
# review.py's DEFAULT_MODEL ("claude-sonnet-4-5-20250929") is what cli.py's
# plain Anthropic() client expects, but AnthropicBedrock rejects that string
# with "The provided model identifier is invalid." (400). This entrypoint
# always uses AnthropicBedrock, so it needs its own, Bedrock-shaped default.
DEFAULT_BEDROCK_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"

app = BedrockAgentCoreApp()


class InvocationError(RuntimeError):
    """Raised for a request-shape or upstream error caught before any
    streaming starts; produces the {"error": {...}} response shape instead
    of a 500 — AgentCore's /invocations contract always returns HTTP 200,
    so callers must check for the "error" key themselves."""


def _require(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not value:
        raise InvocationError(f"Missing required field: {field}")
    return value


def _resolve_meko_pat(payload: dict) -> str:
    """A per-user dedicated deployment gets its caller's own PAT in the
    payload, same as every other field here. The shared instance every
    signed-in user can invoke gets none -- meko_ui's invokeSharedDeployment
    deliberately omits it -- so this falls back to MEKO_PAT, baked into the
    container's own environment at deploy time (see the README's AgentCore
    deployment section)."""
    pat = payload.get("meko_pat") or os.environ.get("MEKO_PAT")
    if not pat:
        raise InvocationError("Missing required field: meko_pat")
    return pat


def _resolve_meko_mcp_url(payload: dict) -> str:
    """Same precedence as _resolve_meko_pat: an explicit per-request override
    in the payload wins, else the container's own MEKO_MCP_URL (baked in at
    deploy time so a dev-deployed container talks to dev's MCP server rather
    than silently defaulting to prod), else the hardcoded prod default."""
    return payload.get("meko_mcp_url") or os.environ.get("MEKO_MCP_URL") or DEFAULT_MEKO_MCP_URL


def _resolve_bedrock_model(payload: dict) -> str:
    """Same precedence as _resolve_meko_pat/_resolve_meko_mcp_url: a
    per-request override in the payload wins, else the container's own
    BEDROCK_MODEL env var (in case the account's available model or
    inference-profile id differs by region/account), else the hardcoded
    default."""
    return payload.get("model") or os.environ.get("BEDROCK_MODEL") or DEFAULT_BEDROCK_MODEL


def _resolve_pr(payload: dict) -> tuple[str, str, str | None]:
    """Returns (title, diff, html_url). Two modes: `pr_title`+`pr_diff`
    directly (covers the bundled "sample diff" case, which has no real PR
    to fetch), or `pr_ref`+`github_token` (fetch from GitHub)."""
    if payload.get("pr_title") and payload.get("pr_diff"):
        return payload["pr_title"], payload["pr_diff"], payload.get("pr_html_url")
    pr_ref = _require(payload, "pr_ref")
    github_token = _require(payload, "github_token")
    try:
        ref = parse_pr_ref(pr_ref)
    except ValueError as exc:
        raise InvocationError(str(exc)) from exc
    try:
        pr = fetch_pull_request(ref, github_token)
    except GitHubApiError as exc:
        raise InvocationError(f"GitHub error: {exc}") from exc
    return pr.title, pr.diff, pr.html_url


def _parse_review_config(payload: dict) -> ReviewConfig:
    tone = payload.get("tone")
    strictness = payload.get("strictness")
    return ReviewConfig(
        enable_knowledge_base=payload.get("enable_knowledge_base", True),
        enable_memory_search=payload.get("enable_memory_search", False),
        focus=payload.get("focus"),
        tone=tone if tone in TONE_FRAGMENTS else None,
        strictness=strictness if strictness in STRICTNESS_FRAGMENTS else None,
    )


@app.entrypoint
def handler(payload: dict) -> dict | Iterator[dict[str, Any]]:
    is_followup = bool(payload.get("question") and payload.get("conversation_id"))

    try:
        datapack_id = _require(payload, "datapack_id")
        meko_pat = _resolve_meko_pat(payload)
        meko_mcp_url = _resolve_meko_mcp_url(payload)
        model = _resolve_bedrock_model(payload)
        config = _parse_review_config(payload)

        if is_followup:
            question = payload["question"]
            conversation_id = payload["conversation_id"]
            prior_review_text = _require(payload, "prior_review_text")
            slack_token = slack_channel = None  # not offered on follow-up turns
        else:
            slack_token = payload.get("slack_token")
            slack_channel = payload.get("slack_channel")
            if slack_token and not slack_channel:
                raise InvocationError("slack_channel is required when slack_token is set.")
            pr_title, pr_diff, pr_html_url = _resolve_pr(payload)
    except InvocationError as exc:
        return {"error": str(exc)}

    if is_followup:
        return _stream_followup(
            meko_mcp_url=meko_mcp_url,
            meko_pat=meko_pat,
            datapack_id=datapack_id,
            conversation_id=conversation_id,
            prior_review_text=prior_review_text,
            question=question,
            config=config,
            model=model,
        )
    return _stream_review(
        meko_mcp_url=meko_mcp_url,
        meko_pat=meko_pat,
        datapack_id=datapack_id,
        pr_title=pr_title,
        pr_diff=pr_diff,
        pr_html_url=pr_html_url,
        config=config,
        slack_token=slack_token,
        slack_channel=slack_channel,
        model=model,
    )


def _stream_review(
    *,
    meko_mcp_url: str,
    meko_pat: str,
    datapack_id: str,
    pr_title: str,
    pr_diff: str,
    pr_html_url: str | None,
    config: ReviewConfig,
    slack_token: str | None,
    slack_channel: str | None,
    model: str,
) -> Iterator[dict[str, Any]]:
    with MekoMcpClient(server_url=meko_mcp_url, pat=meko_pat) as meko:
        conversation_id = _create_conversation(meko, datapack_id, title=f"Review: {pr_title}")
        yield {
            "type": "meta",
            "pr_title": pr_title,
            "pr_html_url": pr_html_url,
            "conversation_id": conversation_id,
        }

        full_text = ""
        findings: list[dict[str, Any]] = []
        usage: dict[str, Any] | None = None
        context_stats: dict[str, Any] | None = None
        for event in run_review_stream(
            meko=meko,
            anthropic_client=AnthropicBedrock(),
            datapack_id=datapack_id,
            pr_title=pr_title,
            pr_diff=pr_diff,
            conversation_id=conversation_id,
            config=config,
            model=model,
        ):
            if event["type"] == "text_delta":
                yield event
            elif event["type"] == "done":
                full_text = event["text"]
                findings = event.get("findings", [])
                usage = event.get("usage")
                context_stats = event.get("context_stats")

        slack_posted = False
        slack_error = None
        if slack_token and slack_channel:
            try:
                post_to_channel(
                    slack_token,
                    slack_channel,
                    f"*PR review* ({pr_title}, {pr_html_url or 'sample diff'}):\n{full_text}",
                )
                slack_posted = True
            except SlackApiError as exc:
                slack_error = str(exc)

        yield {
            "type": "done",
            "review": full_text,
            "findings": findings,
            "usage": usage,
            "context_stats": context_stats,
            "slack_posted": slack_posted,
            "slack_error": slack_error,
        }


def _stream_followup(
    *,
    meko_mcp_url: str,
    meko_pat: str,
    datapack_id: str,
    conversation_id: str,
    prior_review_text: str,
    question: str,
    config: ReviewConfig,
    model: str,
) -> Iterator[dict[str, Any]]:
    with MekoMcpClient(server_url=meko_mcp_url, pat=meko_pat) as meko:
        yield {"type": "meta", "conversation_id": conversation_id}

        full_text = ""
        usage: dict[str, Any] | None = None
        for event in run_followup_stream(
            meko=meko,
            anthropic_client=AnthropicBedrock(),
            datapack_id=datapack_id,
            conversation_id=conversation_id,
            prior_review_text=prior_review_text,
            question=question,
            config=config,
            model=model,
        ):
            if event["type"] == "text_delta":
                yield event
            elif event["type"] == "done":
                full_text = event["text"]
                usage = event.get("usage")

        yield {"type": "done", "review": full_text, "usage": usage}


if __name__ == "__main__":
    app.run()
