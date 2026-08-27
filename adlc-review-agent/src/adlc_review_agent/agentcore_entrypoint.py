"""AWS Bedrock AgentCore Runtime entrypoint. Thin wrapper around the same
core functions cli.py uses — see cli.py for the CLI equivalent of this same
flow. Payload fields mirror the CLI's flags 1:1, as JSON instead of
argv/env, except for the LLM credential: this entrypoint uses
`AnthropicBedrock` (ambient AWS credentials from the deployment's execution
role) instead of a per-request Anthropic API key, so `anthropic_api_key` is
not part of the payload here. `review.py` and the CLI stay untouched —
`AnthropicBedrock` implements the same `.messages.create()` interface
`Anthropic` does, so `run_review()` doesn't need to know which one it got.
"""

from __future__ import annotations

from anthropic import AnthropicBedrock
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from .github_client import GitHubApiError, fetch_pull_request, parse_pr_ref
from .meko_client import MekoMcpClient, MekoMcpError
from .review import run_review
from .slack_client import SlackApiError, post_to_channel

DEFAULT_MEKO_MCP_URL = "https://mcp.mekodata.ai/mcp"

app = BedrockAgentCoreApp()


class InvocationError(RuntimeError):
    """Raised for a request-shape or upstream error; caught in handler() to
    produce the {"error": {...}} response shape instead of a 500 — AgentCore's
    /invocations contract always returns HTTP 200, so callers must check for
    the "error" key themselves."""


def _require(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not value:
        raise InvocationError(f"Missing required field: {field}")
    return value


@app.entrypoint
def handler(payload: dict) -> dict:
    try:
        pr_ref = _require(payload, "pr_ref")
        datapack_id = _require(payload, "datapack_id")
        meko_pat = _require(payload, "meko_pat")
        github_token = _require(payload, "github_token")
        meko_mcp_url = payload.get("meko_mcp_url") or DEFAULT_MEKO_MCP_URL
        slack_token = payload.get("slack_token")
        slack_channel = payload.get("slack_channel")

        if slack_token and not slack_channel:
            raise InvocationError("slack_channel is required when slack_token is set.")

        try:
            ref = parse_pr_ref(pr_ref)
        except ValueError as exc:
            raise InvocationError(str(exc)) from exc

        try:
            pr = fetch_pull_request(ref, github_token)
        except GitHubApiError as exc:
            raise InvocationError(f"GitHub error: {exc}") from exc

        try:
            with MekoMcpClient(server_url=meko_mcp_url, pat=meko_pat) as meko:
                result = run_review(
                    meko=meko,
                    anthropic_client=AnthropicBedrock(),
                    datapack_id=datapack_id,
                    pr_title=pr.title,
                    pr_diff=pr.diff,
                )
        except MekoMcpError as exc:
            raise InvocationError(f"Meko error: {exc}") from exc

        slack_posted = False
        if slack_token and slack_channel:
            try:
                post_to_channel(
                    slack_token,
                    slack_channel,
                    f"*PR review* ({pr.title}, {pr.html_url}):\n{result.text}",
                )
                slack_posted = True
            except SlackApiError as exc:
                return {
                    "pr_title": pr.title,
                    "pr_html_url": pr.html_url,
                    "review": result.text,
                    "slack_posted": False,
                    "slack_error": str(exc),
                }

        return {
            "pr_title": pr.title,
            "pr_html_url": pr.html_url,
            "review": result.text,
            "slack_posted": slack_posted,
        }
    except InvocationError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    app.run()
