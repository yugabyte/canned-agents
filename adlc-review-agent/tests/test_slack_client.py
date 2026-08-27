from __future__ import annotations

import httpx
import pytest
import respx

from adlc_review_agent.slack_client import SlackApiError, post_to_channel


@respx.mock
def test_post_to_channel_success() -> None:
    route = respx.post("https://slack.com/api/chat.postMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    post_to_channel("xoxb-fake", "#pr-reviews", "hello")
    assert route.called
    sent_body = route.calls[0].request.content
    assert b"#pr-reviews" in sent_body


@respx.mock
def test_post_to_channel_raises_on_slack_error() -> None:
    respx.post("https://slack.com/api/chat.postMessage").mock(
        return_value=httpx.Response(200, json={"ok": False, "error": "channel_not_found"})
    )
    with pytest.raises(SlackApiError, match="channel_not_found"):
        post_to_channel("xoxb-fake", "#nope", "hello")
