from __future__ import annotations

import httpx
import pytest
import respx

from adlc_review_agent.github_client import GitHubApiError, fetch_pull_request, parse_pr_ref


def test_parse_pr_ref_valid() -> None:
    ref = parse_pr_ref("yugabyte/meko#533")
    assert (ref.owner, ref.repo, ref.number) == ("yugabyte", "meko", 533)


def test_parse_pr_ref_invalid() -> None:
    with pytest.raises(ValueError, match="owner/repo#123"):
        parse_pr_ref("not-a-valid-ref")


@respx.mock
def test_fetch_pull_request_combines_metadata_and_diff() -> None:
    url = "https://api.github.com/repos/yugabyte/meko/pulls/533"
    respx.get(url, headers={"accept": "application/vnd.github+json"}).mock(
        return_value=httpx.Response(200, json={"title": "Add spend endpoint", "html_url": "https://github.com/x"})
    )
    respx.get(url, headers={"accept": "application/vnd.github.v3.diff"}).mock(
        return_value=httpx.Response(200, text="diff --git a/x b/x\n+added line\n")
    )

    ref = parse_pr_ref("yugabyte/meko#533")
    pr = fetch_pull_request(ref, token="ghp_fake")

    assert pr.title == "Add spend endpoint"
    assert pr.html_url == "https://github.com/x"
    assert "+added line" in pr.diff


@respx.mock
def test_fetch_pull_request_raises_on_non_200() -> None:
    url = "https://api.github.com/repos/yugabyte/meko/pulls/533"
    respx.get(url).mock(return_value=httpx.Response(404, text="Not Found"))

    ref = parse_pr_ref("yugabyte/meko#533")
    with pytest.raises(GitHubApiError, match="404"):
        fetch_pull_request(ref, token="ghp_fake")
