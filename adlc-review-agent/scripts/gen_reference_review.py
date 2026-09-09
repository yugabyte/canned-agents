"""One-time generator for meko_ui's static Claude-reference fixture
(`meko_ui/src/features/labs/data/adlcReviewAgentLab/anthropicReferenceReview.ts`).

Runs the SAME `run_review()` production code path AgentCore uses -- a real
`knowledgebase_search`/`memory_search` against a real datapack, a real
AnthropicBedrock call, and the real `context_stats` (chunk/memory
retrieved-vs-total counts, proportional token-share estimate) -- so every
number this prints is real, captured data, not simulated.

Prerequisites:
  - The target datapack's knowledge base must already have
    `org-coding-standards.md` uploaded (meko_ui's
    data/adlcReviewAgentLab/orgCodingStandards.ts content) -- otherwise
    knowledgebase_search won't find anything relevant and the numbers won't
    mean anything.
  - AWS credentials with bedrock:InvokeModelWithResponseStream for
    anthropic.claude-sonnet-4-5-20250929-v1:0 (see the review agent's own
    execution role, or your own if testing locally).

Usage:
  AWS_REGION=us-east-2 \\
  MEKO_MCP_URL=https://mcp.mekodev.com/mcp \\
  MEKO_PAT=<your PAT, from /me/meko-tokens> \\
  DATAPACK_ID=<a datapack with org-coding-standards.md already in its KB> \\
  .venv/bin/python scripts/gen_reference_review.py
"""

import json
import os
import sys

sys.path.insert(0, "src")

from anthropic import AnthropicBedrock

from adlc_review_agent.meko_client import MekoMcpClient
from adlc_review_agent.review import ReviewConfig, run_review

SAMPLE_DIFF_TITLE = "Add per-team spend aggregation endpoint"
SAMPLE_DIFF_CONTENT = """diff --git a/api/spend.py b/api/spend.py
index 1a2b3c4..5d6e7f8 100644
--- a/api/spend.py
+++ b/api/spend.py
@@ -10,6 +10,24 @@ from flask import Blueprint, request, jsonify
 spend_bp = Blueprint("spend", __name__)


+@spend_bp.route("/spend/summary", methods=["GET"])
+def spend_summary():
+    team_id = request.args.get("team_id")
+    try:
+        rows = db.execute(
+            "SELECT user_id, SUM(tokens) FROM spend WHERE team_id = "
+            + team_id
+            + " GROUP BY user_id"
+        )
+        total = sum(r[1] for r in rows)
+        return jsonify({"team_id": team_id, "total_tokens": total, "rows": rows})
+    except:
+        return jsonify({"error": "failed"}), 500
+
+
 @spend_bp.route("/spend/user/<user_id>", methods=["GET"])
 def spend_for_user(user_id):
     rows = db.execute(
"""

MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name} (see this script's own docstring)")
    return value


def main() -> None:
    meko_mcp_url = _require_env("MEKO_MCP_URL")
    meko_pat = _require_env("MEKO_PAT")
    datapack_id = _require_env("DATAPACK_ID")

    with MekoMcpClient(server_url=meko_mcp_url, pat=meko_pat) as meko:
        result = run_review(
            meko=meko,
            anthropic_client=AnthropicBedrock(),
            datapack_id=datapack_id,
            pr_title=SAMPLE_DIFF_TITLE,
            pr_diff=SAMPLE_DIFF_CONTENT,
            config=ReviewConfig(enable_knowledge_base=True, enable_memory_search=True),
            model=MODEL,
        )

    print(
        json.dumps(
            {
                "text": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "model": MODEL,
                "context_stats": result.context_stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
