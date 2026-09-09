"""Thin client for Meko's Streamable HTTP MCP endpoint (`/mcp`).

Auth is a user-issued Meko Personal Access Token (PAT), sent as a Bearer
token — self-service via Meko's own settings UI (`/me/meko-tokens`). This is
the only integration surface between this tool and Meko; there is no
dependency on inference_gateway or any meko_ui-private route, so this
runs anywhere the PAT-authenticated /mcp endpoint is reachable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import httpx


class MekoMcpError(RuntimeError):
    """Raised for a JSON-RPC error response, or a tool call marked isError."""


@dataclass
class MekoMcpClient:
    server_url: str
    pat: str
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self._session_id: str | None = None
        self._next_id = itertools.count(1)
        self._initialized = False
        self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MekoMcpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.pat}",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _rpc_call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = next(self._next_id)
        res = self._client.post(
            self.server_url,
            headers=self._headers(),
            json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        )
        session_id = res.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        res.raise_for_status()

        content_type = res.headers.get("content-type", "")
        payload = (
            self._parse_sse(res.text, request_id)
            if "text/event-stream" in content_type
            else res.json()
        )

        if payload.get("error"):
            error = payload["error"]
            raise MekoMcpError(f"Meko MCP error ({error.get('code')}): {error.get('message')}")
        return payload.get("result")

    @staticmethod
    def _parse_sse(body: str, expected_id: int) -> dict[str, Any]:
        """Minimal SSE parser: scans `data: <json>` lines for the response
        matching `expected_id`. Falls back to the last data line seen if no
        exact id match is found (some servers round-trip the id as a string)."""
        import json

        last: dict[str, Any] | None = None
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[len("data:"):].strip())
            except ValueError:
                continue
            if ("result" in data or "error" in data) and str(data.get("id")) == str(expected_id):
                return data
            last = data
        if last is not None:
            return last
        raise MekoMcpError("Meko MCP: no parseable SSE response")

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._rpc_call(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "adlc-review-agent", "version": "0.1.0"},
            },
        )
        self._client.post(
            self.server_url,
            headers=self._headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        self._initialized = True

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self._ensure_initialized()
        result = self._rpc_call("tools/call", {"name": name, "arguments": arguments})
        if result and result.get("isError"):
            texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
            raise MekoMcpError("\n".join(texts) or f"{name} failed")

        text = next(
            (c.get("text") for c in (result or {}).get("content", []) if c.get("type") == "text"),
            None,
        )
        if text is None:
            return result
        import json

        parsed: Any = json.loads(text)
        # Some tools' server-side implementation double-encodes their return
        # value as a JSON string — unwrap until we hit a non-string value.
        while isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except ValueError:
                break
        return parsed

    def context_search(
        self,
        *,
        query: str,
        conversation_id: str,
        datapack_id: str,
        agent_id: str = "adlc-review-agent",
        limit: int = 10,
    ) -> Any:
        return self.call_tool(
            "context_search",
            {
                "query": query,
                "conversation_id": conversation_id,
                "datapack_id": datapack_id,
                "agent_id": agent_id,
                "limit": limit,
            },
        )

    def knowledgebase_search(
        self,
        *,
        query: str,
        conversation_id: str,
        datapack_id: str,
        agent_id: str = "adlc-review-agent",
    ) -> Any:
        return self.call_tool(
            "knowledgebase_search",
            {
                "query": query,
                "conversation_id": conversation_id,
                "datapack_id": datapack_id,
                "agent_id": agent_id,
            },
        )
