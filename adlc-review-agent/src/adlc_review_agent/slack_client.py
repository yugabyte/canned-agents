"""Slack post — write-only, no reading/searching Slack. Runs server-side
(a local process, not a browser), so unlike the meko_ui Lab's client-side
Slack call, there's no CORS concern here."""

from __future__ import annotations

import httpx


class SlackApiError(RuntimeError):
    pass


def post_to_channel(token: str, channel: str, text: str, timeout: float = 30.0) -> None:
    res = httpx.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
        json={"channel": channel, "text": text},
        timeout=timeout,
    )
    data = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
    if res.status_code != 200 or not data.get("ok"):
        raise SlackApiError(data.get("error") or f"Slack API request failed: {res.status_code} {res.text}")
