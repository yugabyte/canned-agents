---
name: meko-mcp-integration
description: >-
  How a canned agent in this repo should call Meko's MCP tools correctly —
  the conversation lifecycle (conversation_create then thread conversation_id
  through every subsequent call), resolving the right MCP endpoint for the
  deployment's environment instead of defaulting to prod, and the
  payload-PAT-vs-baked-in-PAT credential pattern. Read this before writing or
  reviewing any code in this repo that calls `meko_client`/Meko's `/mcp`
  endpoint directly, and before debugging a `MekoMcpError` or a tool
  validation error. Every rule here was a real bug found and fixed in
  `adlc-review-agent` (see canned-agents#3) — the point of this file is that
  the next agent shouldn't have to rediscover them one AgentCore invoke at a
  time.
---

# meko-mcp-integration

Every agent in this repo talks to Meko over its Streamable HTTP MCP endpoint
(`/mcp`), auth'd with a Bearer PAT — see `adlc-review-agent/src/adlc_review_agent/meko_client.py`
for the reference client. Copy its patterns for a new agent rather than
reimplementing from scratch; the mistakes below are all things that client
got wrong once and had to be fixed.

## `conversation_id` is required on almost every call, not optional

`conversation_create` mints a conversation; nearly every other datapack-scoped
tool (`knowledgebase_search`, `context_search`, `memory_search`,
`conversation_add_message`, ...) takes `conversation_id` as a **required**
argument on the MCP server side, to nest the call under that conversation's
Langfuse trace. A client wrapper that omits it doesn't get a default — it
gets `MekoMcpError: ... validation error ... conversation_id Missing
required argument`.

**Rule**: call `conversation_create` once per session/turn-sequence, then
pass the returned `conversation_id` into every subsequent MCP call that has
one in scope. Don't trust an older call site's argument list as the source
of truth for what's required — a wrapper can compile and run for months
against the happy path (validation succeeds) and then break the moment a
different caller/config actually exercises the missing-arg branch. Read the
tool's real signature in `meko-mcp-server/src/tools.py` if you have that repo
checked out, or introspect it live via `tools/list`, before assuming a call
site is correct because it's how an existing one looks.

## Resolve the MCP endpoint per-environment — never hardcode a default that's prod

An AgentCore-deployed container has no ambient way to know if it's "dev" or
"prod" unless something tells it. `agentcore_entrypoint.py`'s
`_resolve_meko_mcp_url` is the pattern every entrypoint should copy:

```python
def _resolve_meko_mcp_url(payload: dict) -> str:
    return payload.get("meko_mcp_url") or os.environ.get("MEKO_MCP_URL") or DEFAULT_MEKO_MCP_URL
```

Precedence: an explicit per-request override in the payload, then the
container's own `MEKO_MCP_URL` env var (baked in at deploy time — set it
explicitly for every deployment, dev included), then a hardcoded fallback.
**Never skip the env-var tier** — that was the actual bug: the frontend
invoke payload never set `meko_mcp_url`, so every deployed container,
dev included, silently fell back to the hardcoded production URL. A
dev-only datapack has no row in prod's tables, so every datapack-scoped call
failed with a confusing "not found" error that looked like a data/provisioning
bug and wasn't — see the debugging note below before you conclude the same
thing.

## Credential pattern: per-request PAT vs baked-in PAT

Mirror `_resolve_meko_pat`'s precedence for any new per-user-vs-shared
credential, the same shape as the MCP URL resolution above: payload value
first (a per-user dedicated deployment gets its caller's own PAT), then a
container env var baked in at deploy time (a shared instance every user can
invoke authenticates with its own fixed identity instead of a forwarded
per-user token). Never invert this — a shared instance must not fall back to
some ambient per-user credential it was never handed.

## Debugging playbook for a `MekoMcpError` / tool validation error

1. **Check the MCP endpoint first.** A "not found" error for something you
   know exists is very often the client talking to the wrong environment's
   Meko, not a real data gap. Confirm which `meko_mcp_url` this specific
   invocation actually used before investigating anything else.
2. **Reproduce the exact call directly**, with the same tool name, same
   `agent_id`, same `datapack_id`, against the environment you *intend* to be
   talking to (e.g. via an interactive MCP client, or Claude Code's own Meko
   MCP tools if you have dev access). If the identical call succeeds outside
   your agent's code, the bug is in what your agent sends or where it sends
   it — not in Meko's data for that datapack. This is what actually
   distinguished "AgentK-specific bug" from "the same mechanism every caller
   uses" in the incident this skill is based on: a direct `conversation_create`
   + `conversation_add_message` against the same datapack succeeded
   immediately, which ruled out a Langfuse-provisioning gap and pointed
   straight at the environment-routing bug instead.
3. **Read the actual required-argument list** the error names (Pydantic's
   `missing_argument` errors name the exact field) rather than guessing from
   an existing call site — see the `conversation_id` rule above.

## Adding a new agent

Do all of the above for its own `meko_client.py`-equivalent rather than
inventing a new pattern — a second, differently-shaped Meko client in this
repo makes every one of the gotchas above something to rediscover per-agent
instead of once.
