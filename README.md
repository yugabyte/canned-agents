# canned-agents

Starter pack of agents built on [Meko](https://mekodata.ai) — each one uses a
Meko datapack for storing and retrieving context (knowledge base + memories),
and can be deployed to AWS Bedrock AgentCore Runtime via Meko's `api_server`.

One subdirectory per agent, each a self-contained Python package with its own
`pyproject.toml`, tests, and README.

## Agents

- [`adlc-review-agent`](adlc-review-agent/) — PR-review agent that grounds
  its review in an org's coding standards from a Meko datapack's knowledge
  base ("AgentK" in the Meko Labs UI).

## Adding a new agent

Add a new top-level directory, self-contained the same way `adlc-review-agent`
is (own `pyproject.toml`, `src/`, `tests/`, `README.md`, `Dockerfile`). To make
it deployable from Meko's Labs UI, see `adlc-review-agent/README.md`'s
"Deploying to AWS Bedrock AgentCore Runtime" section for the container
contract and image-publishing convention every agent here follows.

Before writing its Meko MCP client, read
[`.claude/skills/meko-mcp-integration/SKILL.md`](.claude/skills/meko-mcp-integration/SKILL.md) —
the conversation-lifecycle, MCP-endpoint-resolution, and credential rules
every agent's client needs, each one a real bug found in `adlc-review-agent`.
