"""Minimal GitHub REST client: fetch a PR's diff + title."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

_PR_REF_RE = re.compile(r"^([\w.-]+)/([\w.-]+)#(\d+)$")


class GitHubApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrRef:
    owner: str
    repo: str
    number: int


def parse_pr_ref(text: str) -> PrRef:
    """Parses "owner/repo#123". Raises ValueError on anything else."""
    match = _PR_REF_RE.match(text.strip())
    if not match:
        raise ValueError(f'Expected "owner/repo#123", got: {text!r}')
    return PrRef(owner=match.group(1), repo=match.group(2), number=int(match.group(3)))


@dataclass(frozen=True)
class PullRequest:
    title: str
    html_url: str
    diff: str


def fetch_pull_request(ref: PrRef, token: str, timeout: float = 30.0) -> PullRequest:
    """Fetches a PR's metadata + unified diff in two requests (GitHub's REST
    API doesn't return both from one response media type)."""
    base = f"https://api.github.com/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}"
    auth_header = {"Authorization": f"Bearer {token}"}

    with httpx.Client(timeout=timeout) as client:
        meta_res = client.get(base, headers={**auth_header, "Accept": "application/vnd.github+json"})
        if meta_res.status_code != 200:
            raise GitHubApiError(f"GitHub API request failed: {meta_res.status_code} {meta_res.text}")
        meta = meta_res.json()

        diff_res = client.get(base, headers={**auth_header, "Accept": "application/vnd.github.v3.diff"})
        if diff_res.status_code != 200:
            raise GitHubApiError(f"GitHub API request failed: {diff_res.status_code} {diff_res.text}")

    return PullRequest(
        title=meta.get("title") or f"{ref.owner}/{ref.repo}#{ref.number}",
        html_url=meta.get("html_url", ""),
        diff=diff_res.text,
    )
