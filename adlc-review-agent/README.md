# adlc-review-agent

Standalone PR-review agent built on [Meko](https://mekodata.ai) — the open-source
agent core behind the ADLC Review Agent Lab in `meko_ui`. Runs on your own machine
against your own Meko Personal Access Token, GitHub token, and Anthropic API key.
No dependency on `meko_ui` or `inference_gateway` — only Meko's public MCP
endpoint, GitHub's REST API, and Anthropic's API.

## What it does

1. Fetches a pull request's diff via the GitHub API.
2. Retrieves your org's coding standards from a Meko datapack's knowledge base
   (`knowledgebase_search`).
3. Asks Claude to review the diff against those standards.
4. Optionally posts the review to a Slack channel.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
export MEKO_PAT=mko_tkn_...          # from Meko's own settings, /me/meko-tokens
export GITHUB_TOKEN=ghp_...
export ANTHROPIC_API_KEY=sk-ant-...

adlc-review-agent yugabyte/meko#533 --datapack-id <your-datapack-id>

# Optionally post the review to Slack:
adlc-review-agent yugabyte/meko#533 --datapack-id <your-datapack-id> \
  --slack-token xoxb-... --slack-channel "#pr-reviews"
```

All four secrets can also be passed as flags (`--meko-pat`, `--github-token`,
`--anthropic-api-key`, `--slack-token`) instead of env vars.

## Relationship to the meko_ui Lab

The in-product "ADLC Review Agent Lab" (`meko_ui/src/features/labs/`) runs the
same review inside the browser, using `inference_gateway`'s `review_agent`
persona so it gets Meko's full conversation/memory/trace pipeline "for free."
This package is the same review, decoupled: no `inference_gateway`, no
meko_ui-private routes, runnable by anyone with a Meko account. The system
prompt in `src/adlc_review_agent/review.py` mirrors
`inference_gateway/inference_gateway/chain.py`'s `REVIEW_AGENT_SYSTEM_TEMPLATE`
by convention — they're two independent implementations of the same framing,
not shared code.

## Deploying to AWS Bedrock AgentCore Runtime

`src/adlc_review_agent/agentcore_entrypoint.py` is a thin wrapper around the
same functions the CLI uses, adapted to AgentCore's container contract
(`POST /invocations`, `GET /ping` on port 8080, implemented for us by the
`bedrock-agentcore` SDK's `BedrockAgentCoreApp`/`@app.entrypoint`). It's a
separate entry point from the CLI — `cli.py` and `review.py` are unchanged.

One difference from the CLI: this entrypoint calls Claude via
`anthropic.AnthropicBedrock` instead of a per-request Anthropic API key —
model access comes from the deployment's AWS execution role
(`bedrock:InvokeModel`), so there's no `anthropic_api_key` field in the
payload below.

**Request payload** (JSON):

| Field | Required | Description |
|---|---|---|
| `pr_ref` | yes | `owner/repo#123` |
| `datapack_id` | yes | Meko datapack holding the org's coding standards |
| `meko_pat` | no* | Meko Personal Access Token |
| `github_token` | yes | GitHub PAT |
| `meko_mcp_url` | no | Defaults to `https://mcp.mekodata.ai/mcp` |
| `slack_token` | no | Post the review to Slack if set |
| `slack_channel` | no | Required if `slack_token` is set |

\* `meko_pat` is required unless the container's own `MEKO_PAT` environment
variable is set, in which case the payload may omit it entirely — see
"Deploying as Meko's shared instance" below. A per-user dedicated deployment
always gets its own `meko_pat` in the payload and ignores `MEKO_PAT`.

**Response** (success): `{pr_title, pr_html_url, review, slack_posted, slack_error?}`.
**Response** (error): `{"error": "<message>"}` — AgentCore's contract always
returns HTTP 200, so callers must check for the `error` key.

Build the container:

```bash
docker build --platform linux/arm64 -t adlc-review-agent .
```

The actual AWS plumbing (`CreateAgentRuntime`/`InvokeAgentRuntime`, cross-account
IAM role assumption for BYOC deployments) is handled by Meko's `api_server` —
this repo only needs to produce a working container image, published where
`api_server` expects it.

### Publishing the image

`api_server` resolves each deployable agent's image by a fixed naming
convention, not a per-agent config value — `MEKO_LABS_AGENT_IMAGES_REPO`
(an ECR repo prefix, one shared value for every agent in this repo) plus the
agent's own slug:

```
{MEKO_LABS_AGENT_IMAGES_REPO}/adlc-review-agent:latest
```

Publishing a new build just means pushing to that path — no code or config
change on the `api_server` side is needed. `adlc-review-agent` is already
one of the slugs `api_server` recognizes as deployable
(`service.SupportedDeploymentAgentSlugs`); a new agent added to this repo
needs an entry there too before it can be deployed from the Labs UI.

### Deploying as Meko's shared instance

Unlike a per-user dedicated deployment (created through the Labs UI, tracked
in Meko's own `deployments` table, invoked with the calling user's own
`meko_pat`), the one shared instance every signed-in Meko user can invoke
without deploying their own is created and operated entirely outside Meko —
by hand, via the AWS CLI — and authenticates with its own baked-in
credential instead of a per-request PAT:

```bash
aws bedrock-agentcore-control create-agent-runtime \
  --agent-runtime-name meko_agentk_shared \
  --agent-runtime-artifact '{"containerConfiguration":{"containerUri":"{MEKO_LABS_AGENT_IMAGES_REPO}/adlc-review-agent:latest"}}' \
  --role-arn <meko's managed AgentCore execution role arn> \
  --network-configuration networkMode=PUBLIC \
  --environment-variables MEKO_PAT=<a PAT for the account the datapack gets shared with>,MEKO_MCP_URL=<this environment's Meko MCP endpoint, e.g. https://mcp.mekodev.com/mcp for dev> \
  --region us-east-1
```

Poll `get-agent-runtime` until `status: READY`, then register the resulting
`agentRuntimeArn` in `meko_system.shared_deployments` (keyed by `agent_slug`)
so `GET/POST /deployments/shared/:agent_slug*` can find it — see
`api_server`'s own docs for that table. `MEKO_PAT` is read by
`agentcore_entrypoint.py`'s `_resolve_meko_pat`, and `MEKO_MCP_URL` by its
`_resolve_meko_mcp_url`, only when the invoking payload has no `meko_pat`/
`meko_mcp_url` of its own -- which is exactly what `invokeSharedDeployment`
(meko_ui) sends. **`MEKO_MCP_URL` must match the deployment's own
environment** (dev vs prod) -- omitting it silently falls back to
production, so a dev-deployed shared instance would talk to prod's Meko
and fail to find any dev-only datapack.

## Development

```bash
pip install -e ".[dev]"
pytest
```
