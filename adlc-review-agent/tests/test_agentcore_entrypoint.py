from __future__ import annotations

from unittest.mock import MagicMock, patch

from adlc_review_agent.agentcore_entrypoint import handler
from adlc_review_agent.github_client import GitHubApiError, PullRequest
from adlc_review_agent.review import ReviewResult
from adlc_review_agent.slack_client import SlackApiError

_BASE_PAYLOAD = {
    "pr_ref": "yugabyte/meko#533",
    "datapack_id": "dp-1",
    "meko_pat": "mko_tkn_fake",
    "github_token": "ghp_fake",
}


@patch("adlc_review_agent.agentcore_entrypoint.post_to_channel")
@patch("adlc_review_agent.agentcore_entrypoint.run_review")
@patch("adlc_review_agent.agentcore_entrypoint.MekoMcpClient")
@patch("adlc_review_agent.agentcore_entrypoint.fetch_pull_request")
@patch("adlc_review_agent.agentcore_entrypoint.AnthropicBedrock")
def test_handler_success_without_slack(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_run_review, mock_post
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_run_review.return_value = ReviewResult(text="Looks fine.", kb_context="(none found)")

    result = handler(dict(_BASE_PAYLOAD))

    assert result == {
        "pr_title": "Add endpoint",
        "pr_html_url": "https://x",
        "review": "Looks fine.",
        "slack_posted": False,
    }
    mock_anthropic_bedrock.assert_called_once_with()
    mock_post.assert_not_called()


@patch("adlc_review_agent.agentcore_entrypoint.post_to_channel")
@patch("adlc_review_agent.agentcore_entrypoint.run_review")
@patch("adlc_review_agent.agentcore_entrypoint.MekoMcpClient")
@patch("adlc_review_agent.agentcore_entrypoint.fetch_pull_request")
@patch("adlc_review_agent.agentcore_entrypoint.AnthropicBedrock")
def test_handler_posts_to_slack_when_requested(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_run_review, mock_post
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_run_review.return_value = ReviewResult(text="Looks fine.", kb_context="(none found)")

    result = handler({**_BASE_PAYLOAD, "slack_token": "xoxb-fake", "slack_channel": "#pr-reviews"})

    assert result["slack_posted"] is True
    mock_post.assert_called_once_with("xoxb-fake", "#pr-reviews", "*PR review* (Add endpoint, https://x):\nLooks fine.")


@patch("adlc_review_agent.agentcore_entrypoint.post_to_channel")
@patch("adlc_review_agent.agentcore_entrypoint.run_review")
@patch("adlc_review_agent.agentcore_entrypoint.MekoMcpClient")
@patch("adlc_review_agent.agentcore_entrypoint.fetch_pull_request")
@patch("adlc_review_agent.agentcore_entrypoint.AnthropicBedrock")
def test_handler_partial_success_when_slack_post_fails(
    mock_anthropic_bedrock, mock_fetch_pr, mock_meko_cls, mock_run_review, mock_post
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_run_review.return_value = ReviewResult(text="Looks fine.", kb_context="(none found)")
    mock_post.side_effect = SlackApiError("channel_not_found")

    result = handler({**_BASE_PAYLOAD, "slack_token": "xoxb-fake", "slack_channel": "#missing"})

    assert result == {
        "pr_title": "Add endpoint",
        "pr_html_url": "https://x",
        "review": "Looks fine.",
        "slack_posted": False,
        "slack_error": "channel_not_found",
    }


def test_handler_missing_required_field() -> None:
    payload = dict(_BASE_PAYLOAD)
    del payload["github_token"]

    result = handler(payload)

    assert result == {"error": "Missing required field: github_token"}


def test_handler_slack_channel_required_with_slack_token() -> None:
    result = handler({**_BASE_PAYLOAD, "slack_token": "xoxb-fake"})

    assert result == {"error": "slack_channel is required when slack_token is set."}


def test_handler_invalid_pr_ref_returns_error() -> None:
    result = handler({**_BASE_PAYLOAD, "pr_ref": "not-a-ref"})

    assert "error" in result
    assert "owner/repo#123" in result["error"]


@patch("adlc_review_agent.agentcore_entrypoint.fetch_pull_request")
def test_handler_github_error_returns_error(mock_fetch_pr) -> None:
    mock_fetch_pr.side_effect = GitHubApiError("GitHub API request failed: 404 Not Found")

    result = handler(dict(_BASE_PAYLOAD))

    assert result == {"error": "GitHub error: GitHub API request failed: 404 Not Found"}
