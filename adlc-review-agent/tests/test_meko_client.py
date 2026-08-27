from __future__ import annotations

import json

import httpx
import pytest
import respx

from adlc_review_agent.meko_client import MekoMcpClient, MekoMcpError

SERVER_URL = "https://mcp.example.test/mcp"


def _tool_result_response(request_id: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
        },
    )


@respx.mock
def test_knowledgebase_search_initializes_then_calls_tool() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}})
        if body["method"] == "notifications/initialized":
            return httpx.Response(200)
        if body["method"] == "tools/call":
            assert body["params"]["name"] == "knowledgebase_search"
            assert body["params"]["arguments"]["datapack_id"] == "dp-1"
            return _tool_result_response(body["id"], {"results": [{"chunk_text": "no bare except"}]})
        raise AssertionError(f"unexpected method {body['method']}")

    respx.post(SERVER_URL).mock(side_effect=handler)

    with MekoMcpClient(server_url=SERVER_URL, pat="mko_tkn_fake") as client:
        result = client.knowledgebase_search(query="coding standards", datapack_id="dp-1")

    assert result == {"results": [{"chunk_text": "no bare except"}]}


@respx.mock
def test_call_tool_raises_on_is_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}})
        if body["method"] == "notifications/initialized":
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"isError": True, "content": [{"type": "text", "text": "datapack not found"}]},
            },
        )

    respx.post(SERVER_URL).mock(side_effect=handler)

    with MekoMcpClient(server_url=SERVER_URL, pat="mko_tkn_fake") as client:
        with pytest.raises(MekoMcpError, match="datapack not found"):
            client.knowledgebase_search(query="x", datapack_id="missing")


@respx.mock
def test_double_encoded_string_result_is_unwrapped() -> None:
    """Some tools' server-side implementation returns json.dumps(...) as
    their return value (a str), which the transport can end up serializing
    a second time — the client must unwrap until a non-string value."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}})
        if body["method"] == "notifications/initialized":
            return httpx.Response(200)
        double_encoded = json.dumps(json.dumps({"results": []}))
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"content": [{"type": "text", "text": double_encoded}]},
            },
        )

    respx.post(SERVER_URL).mock(side_effect=handler)

    with MekoMcpClient(server_url=SERVER_URL, pat="mko_tkn_fake") as client:
        result = client.knowledgebase_search(query="x", datapack_id="dp-1")

    assert result == {"results": []}
