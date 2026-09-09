from __future__ import annotations

from unittest.mock import MagicMock, patch

from adlc_review_agent.agentcore_entrypoint import DEFAULT_MEKO_MCP_URL, handler
from adlc_review_agent.github_client import GitHubApiError, PullRequest
from adlc_review_agent.slack_client import SlackApiError

_BASE_PAYLOAD = {
    "pr_ref": "yugabyte/meko#533",
    "datapack_id": "dp-1",
    "meko_pat": "mko_tkn_fake",
    "github_token": "ghp_fake",
}


def _events(payload: dict) -> list[dict]:
    result = handler(payload)
    if isinstance(result, dict):
        return [result]
    return list(result)


def _patched(fn):
    """Common patch stack for the success-path tests below: MekoMcpClient,
    fetch_pull_request, AnthropicBedrock, and _create_conversation (so no
    real MCP call happens), plus run_review_stream stubbed to a canned
    event sequence. Applied innermost-first so the resulting parameter
    order matches each test's signature: (anthropic_bedrock, fetch_pr,
    meko_cls, create_conversation, run_review_stream, post)."""
    fn = patch("adlc_review_agent.agentcore_entrypoint.AnthropicBedrock")(fn)
    fn = patch("adlc_review_agent.agentcore_entrypoint.fetch_pull_request")(fn)
    fn = patch("adlc_review_agent.agentcore_entrypoint.MekoMcpClient")(fn)
    fn = patch("adlc_review_agent.agentcore_entrypoint._create_conversation")(fn)
    fn = patch("adlc_review_agent.agentcore_entrypoint.run_review_stream")(fn)
    fn = patch("adlc_review_agent.agentcore_entrypoint.post_to_channel")(fn)
    return fn


@_patched
def test_handler_success_without_slack(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_create_conversation, mock_run_review_stream, mock_post
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_create_conversation.return_value = "conv-1"
    mock_run_review_stream.return_value = iter(
        [
            {"type": "text_delta", "text": "Looks "},
            {"type": "text_delta", "text": "fine."},
            {"type": "done", "text": "Looks fine.", "kb_context": "(none found)"},
        ]
    )

    events = _events(dict(_BASE_PAYLOAD))

    assert events[0] == {
        "type": "meta",
        "pr_title": "Add endpoint",
        "pr_html_url": "https://x",
        "conversation_id": "conv-1",
    }
    assert events[1] == {"type": "text_delta", "text": "Looks "}
    assert events[2] == {"type": "text_delta", "text": "fine."}
    assert events[-1] == {
        "type": "done",
        "review": "Looks fine.",
        "slack_posted": False,
        "slack_error": None,
    }
    mock_anthropic_bedrock.assert_called_once_with()
    mock_post.assert_not_called()


@_patched
def test_handler_posts_to_slack_when_requested(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_create_conversation, mock_run_review_stream, mock_post
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_create_conversation.return_value = "conv-1"
    mock_run_review_stream.return_value = iter([{"type": "done", "text": "Looks fine.", "kb_context": "(none found)"}])

    events = _events({**_BASE_PAYLOAD, "slack_token": "xoxb-fake", "slack_channel": "#pr-reviews"})

    assert events[-1]["slack_posted"] is True
    mock_post.assert_called_once_with("xoxb-fake", "#pr-reviews", "*PR review* (Add endpoint, https://x):\nLooks fine.")


@_patched
def test_handler_partial_success_when_slack_post_fails(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_create_conversation, mock_run_review_stream, mock_post
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_create_conversation.return_value = "conv-1"
    mock_run_review_stream.return_value = iter([{"type": "done", "text": "Looks fine.", "kb_context": "(none found)"}])
    mock_post.side_effect = SlackApiError("channel_not_found")

    events = _events({**_BASE_PAYLOAD, "slack_token": "xoxb-fake", "slack_channel": "#missing"})

    assert events[-1] == {
        "type": "done",
        "review": "Looks fine.",
        "slack_posted": False,
        "slack_error": "channel_not_found",
    }


def test_handler_missing_required_field() -> None:
    payload = dict(_BASE_PAYLOAD)
    del payload["github_token"]

    events = _events(payload)

    assert events == [{"error": "Missing required field: github_token"}]


def test_handler_missing_meko_pat_with_no_env_fallback_returns_error(monkeypatch) -> None:
    monkeypatch.delenv("MEKO_PAT", raising=False)
    payload = dict(_BASE_PAYLOAD)
    del payload["meko_pat"]

    events = _events(payload)

    assert events == [{"error": "Missing required field: meko_pat"}]


@_patched
def test_handler_falls_back_to_meko_pat_env_var(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_create_conversation, mock_run_review_stream, mock_post,
    monkeypatch,
) -> None:
    # The shared deployment's payload (meko_ui's invokeSharedDeployment) never
    # carries meko_pat -- the container's own MEKO_PAT env var, baked in at
    # deploy time, is what it authenticates with instead.
    monkeypatch.setenv("MEKO_PAT", "mko_tkn_from_env")
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_create_conversation.return_value = "conv-1"
    mock_run_review_stream.return_value = iter([{"type": "done", "text": "Looks fine.", "kb_context": "(none found)"}])

    payload = dict(_BASE_PAYLOAD)
    del payload["meko_pat"]
    events = _events(payload)

    assert events[-1]["review"] == "Looks fine."
    mock_meko_cls.assert_called_once_with(server_url=DEFAULT_MEKO_MCP_URL, pat="mko_tkn_from_env")


@_patched
def test_handler_falls_back_to_meko_mcp_url_env_var(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_create_conversation, mock_run_review_stream, mock_post,
    monkeypatch,
) -> None:
    # A deployed container's own MEKO_MCP_URL, baked in at deploy time, must
    # win over the hardcoded prod default when the payload has none of its
    # own -- otherwise a dev-deployed shared instance silently talks to prod
    # and every datapack-scoped MCP call 404s.
    monkeypatch.setenv("MEKO_MCP_URL", "https://mcp.mekodev.com/mcp")
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_create_conversation.return_value = "conv-1"
    mock_run_review_stream.return_value = iter([{"type": "done", "text": "Looks fine.", "kb_context": "(none found)"}])

    payload = dict(_BASE_PAYLOAD)
    events = _events(payload)

    assert events[-1]["review"] == "Looks fine."
    mock_meko_cls.assert_called_once_with(server_url="https://mcp.mekodev.com/mcp", pat="mko_tkn_fake")


@_patched
def test_handler_payload_meko_mcp_url_wins_over_env_var(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_create_conversation, mock_run_review_stream, mock_post,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MEKO_MCP_URL", "https://mcp.mekodev.com/mcp")
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_create_conversation.return_value = "conv-1"
    mock_run_review_stream.return_value = iter([{"type": "done", "text": "Looks fine.", "kb_context": "(none found)"}])

    payload = {**_BASE_PAYLOAD, "meko_mcp_url": "https://mcp.example-override.ai/mcp"}
    events = _events(payload)

    assert events[-1]["review"] == "Looks fine."
    mock_meko_cls.assert_called_once_with(server_url="https://mcp.example-override.ai/mcp", pat="mko_tkn_fake")


def test_handler_slack_channel_required_with_slack_token() -> None:
    events = _events({**_BASE_PAYLOAD, "slack_token": "xoxb-fake"})

    assert events == [{"error": "slack_channel is required when slack_token is set."}]


def test_handler_invalid_pr_ref_returns_error() -> None:
    events = _events({**_BASE_PAYLOAD, "pr_ref": "not-a-ref"})

    assert len(events) == 1
    assert "error" in events[0]
    assert "owner/repo#123" in events[0]["error"]


@patch("adlc_review_agent.agentcore_entrypoint.fetch_pull_request")
def test_handler_github_error_returns_error(mock_fetch_pr) -> None:
    mock_fetch_pr.side_effect = GitHubApiError("GitHub API request failed: 404 Not Found")

    events = _events(dict(_BASE_PAYLOAD))

    assert events == [{"error": "GitHub error: GitHub API request failed: 404 Not Found"}]


@_patched
def test_handler_supports_direct_pr_title_diff_mode(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_create_conversation, mock_run_review_stream, mock_post
) -> None:
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_create_conversation.return_value = "conv-1"
    mock_run_review_stream.return_value = iter([{"type": "done", "text": "Reviewed sample diff.", "kb_context": "(none found)"}])

    payload = {
        "datapack_id": "dp-1",
        "meko_pat": "mko_tkn_fake",
        "pr_title": "Sample diff",
        "pr_diff": "+ x = 1",
    }
    events = _events(payload)

    mock_fetch_pr.assert_not_called()
    assert events[0]["pr_title"] == "Sample diff"
    assert events[-1]["review"] == "Reviewed sample diff."


@_patched
def test_handler_review_config_fields_pass_through_to_run_review_stream(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_create_conversation, mock_run_review_stream, mock_post
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_create_conversation.return_value = "conv-1"
    mock_run_review_stream.return_value = iter([{"type": "done", "text": "ok", "kb_context": "(none found)"}])

    list(
        _events(
            {
                **_BASE_PAYLOAD,
                "enable_knowledge_base": False,
                "enable_memory_search": True,
                "tone": "concise",
                "strictness": "strict",
                "focus": "SQL injection",
            }
        )
    )

    config = mock_run_review_stream.call_args.kwargs["config"]
    assert config.enable_knowledge_base is False
    assert config.enable_memory_search is True
    assert config.tone == "concise"
    assert config.strictness == "strict"
    assert config.focus == "SQL injection"


@_patched
def test_handler_ignores_unrecognized_tone_and_strictness(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_create_conversation, mock_run_review_stream, mock_post
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_create_conversation.return_value = "conv-1"
    mock_run_review_stream.return_value = iter([{"type": "done", "text": "ok", "kb_context": "(none found)"}])

    list(_events({**_BASE_PAYLOAD, "tone": "hostile", "strictness": "brutal"}))

    config = mock_run_review_stream.call_args.kwargs["config"]
    assert config.tone is None
    assert config.strictness is None


@patch("adlc_review_agent.agentcore_entrypoint.run_followup_stream")
@patch("adlc_review_agent.agentcore_entrypoint.MekoMcpClient")
@patch("adlc_review_agent.agentcore_entrypoint.AnthropicBedrock")
def test_handler_followup_mode_reuses_conversation_id(mock_anthropic_bedrock, mock_meko_cls, mock_run_followup) -> None:
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_run_followup.return_value = iter([{"type": "text_delta", "text": "It's a "}, {"type": "text_delta", "text": "false positive."}, {"type": "done", "text": "It's a false positive."}])

    events = _events(
        {
            "datapack_id": "dp-1",
            "meko_pat": "mko_tkn_fake",
            "conversation_id": "conv-existing",
            "prior_review_text": "Flagged a bare except.",
            "question": "Why is that a problem?",
        }
    )

    assert events[0] == {"type": "meta", "conversation_id": "conv-existing"}
    assert events[-1] == {"type": "done", "review": "It's a false positive."}
    mock_run_followup.assert_called_once()
    assert mock_run_followup.call_args.kwargs["conversation_id"] == "conv-existing"
    assert mock_run_followup.call_args.kwargs["question"] == "Why is that a problem?"
