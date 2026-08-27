"""CLI entry point: `adlc-review-agent review owner/repo#123 --datapack-id ...`."""

from __future__ import annotations

import sys

import click
from anthropic import Anthropic

from .github_client import GitHubApiError, fetch_pull_request, parse_pr_ref
from .meko_client import MekoMcpClient, MekoMcpError
from .review import run_review
from .slack_client import SlackApiError, post_to_channel

DEFAULT_MEKO_MCP_URL = "https://mcp.mekodata.ai/mcp"


@click.command()
@click.argument("pr_ref")
@click.option("--datapack-id", required=True, help="Meko datapack id holding the org's coding standards.")
@click.option(
    "--meko-pat",
    envvar="MEKO_PAT",
    required=True,
    help="Meko Personal Access Token (or set MEKO_PAT).",
)
@click.option(
    "--meko-mcp-url",
    envvar="MEKO_MCP_URL",
    default=DEFAULT_MEKO_MCP_URL,
    show_default=True,
    help="Meko MCP server URL (or set MEKO_MCP_URL).",
)
@click.option("--github-token", envvar="GITHUB_TOKEN", required=True, help="GitHub PAT (or set GITHUB_TOKEN).")
@click.option(
    "--anthropic-api-key",
    envvar="ANTHROPIC_API_KEY",
    required=True,
    help="Anthropic API key (or set ANTHROPIC_API_KEY).",
)
@click.option("--slack-token", envvar="SLACK_TOKEN", default=None, help="Optional: post the review to Slack.")
@click.option("--slack-channel", default=None, help="Slack channel to post to (required with --slack-token).")
def main(
    pr_ref: str,
    datapack_id: str,
    meko_pat: str,
    meko_mcp_url: str,
    github_token: str,
    anthropic_api_key: str,
    slack_token: str | None,
    slack_channel: str | None,
) -> None:
    """Review PR_REF (format: owner/repo#123) against the coding standards
    in the given Meko datapack's knowledge base."""
    try:
        ref = parse_pr_ref(pr_ref)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if slack_token and not slack_channel:
        raise click.ClickException("--slack-channel is required when --slack-token is set.")

    click.echo(f"Fetching {pr_ref}...", err=True)
    try:
        pr = fetch_pull_request(ref, github_token)
    except GitHubApiError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Reviewing against org coding standards...", err=True)
    try:
        with MekoMcpClient(server_url=meko_mcp_url, pat=meko_pat) as meko:
            result = run_review(
                meko=meko,
                anthropic_client=Anthropic(api_key=anthropic_api_key),
                datapack_id=datapack_id,
                pr_title=pr.title,
                pr_diff=pr.diff,
            )
    except MekoMcpError as exc:
        raise click.ClickException(f"Meko error: {exc}") from exc

    click.echo(f"\n{result.text}\n")

    if slack_token and slack_channel:
        click.echo(f"Posting to {slack_channel}...", err=True)
        try:
            post_to_channel(slack_token, slack_channel, f"*PR review* ({pr.title}, {pr.html_url}):\n{result.text}")
        except SlackApiError as exc:
            click.echo(f"Failed to post to Slack: {exc}", err=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
