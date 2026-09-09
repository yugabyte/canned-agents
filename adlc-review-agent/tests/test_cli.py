from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from adlc_review_agent.cli import main
from adlc_review_agent.github_client import PullRequest
from adlc_review_agent.review import ReviewResult


def _invoke(args: list[str]):
    runner = CliRunner()
    return runner.invoke(main, args)


@patch("adlc_review_agent.cli.post_to_channel")
@patch("adlc_review_agent.cli.run_review")
@patch("adlc_review_agent.cli.MekoMcpClient")
@patch("adlc_review_agent.cli.fetch_pull_request")
@patch("adlc_review_agent.cli.Anthropic")
def test_full_run_without_slack(
    mock_anthropic, mock_fetch_pr, mock_meko_cls, mock_run_review, mock_post_to_slack
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_run_review.return_value = ReviewResult(
        text="Looks fine.", findings=[], kb_context="(none found)", input_tokens=100, output_tokens=50, context_stats={}
    )

    result = _invoke(
        [
            "yugabyte/meko#533",
            "--datapack-id", "dp-1",
            "--meko-pat", "mko_tkn_fake",
            "--github-token", "ghp_fake",
            "--anthropic-api-key", "sk-fake",
        ]
    )

    assert result.exit_code == 0, result.output
    assert "Looks fine." in result.output
    mock_post_to_slack.assert_not_called()


@patch("adlc_review_agent.cli.post_to_channel")
@patch("adlc_review_agent.cli.run_review")
@patch("adlc_review_agent.cli.MekoMcpClient")
@patch("adlc_review_agent.cli.fetch_pull_request")
@patch("adlc_review_agent.cli.Anthropic")
def test_posts_to_slack_when_requested(
    mock_anthropic, mock_fetch_pr, mock_meko_cls, mock_run_review, mock_post_to_slack
) -> None:
    mock_fetch_pr.return_value = PullRequest(title="Add endpoint", html_url="https://x", diff="+ x = 1")
    mock_meko_cls.return_value.__enter__.return_value = MagicMock()
    mock_run_review.return_value = ReviewResult(
        text="Looks fine.", findings=[], kb_context="(none found)", input_tokens=100, output_tokens=50, context_stats={}
    )

    result = _invoke(
        [
            "yugabyte/meko#533",
            "--datapack-id", "dp-1",
            "--meko-pat", "mko_tkn_fake",
            "--github-token", "ghp_fake",
            "--anthropic-api-key", "sk-fake",
            "--slack-token", "xoxb-fake",
            "--slack-channel", "#pr-reviews",
        ]
    )

    assert result.exit_code == 0, result.output
    mock_post_to_slack.assert_called_once()
    assert mock_post_to_slack.call_args.args[1] == "#pr-reviews"


def test_slack_channel_required_with_slack_token() -> None:
    result = _invoke(
        [
            "yugabyte/meko#533",
            "--datapack-id", "dp-1",
            "--meko-pat", "mko_tkn_fake",
            "--github-token", "ghp_fake",
            "--anthropic-api-key", "sk-fake",
            "--slack-token", "xoxb-fake",
        ]
    )
    assert result.exit_code != 0
    assert "slack-channel" in result.output.lower()


def test_invalid_pr_ref_fails_cleanly() -> None:
    result = _invoke(
        [
            "not-a-ref",
            "--datapack-id", "dp-1",
            "--meko-pat", "mko_tkn_fake",
            "--github-token", "ghp_fake",
            "--anthropic-api-key", "sk-fake",
        ]
    )
    assert result.exit_code != 0
    assert "owner/repo#123" in result.output
