"""Antigravity smoke: two tiny ``agy`` calls, one plain and one schema-constrained.

Requires Google's Antigravity CLI (``agy``) on ``PATH`` and an existing ``agy``
login on this host. Override the model with ``AGENTFLOW_ANTIGRAVITY_MODEL``
(``agy models`` lists the ids).

Both nodes use ``tools="read_write"`` (``--dangerously-skip-permissions``) because
``read_only`` maps to ``agy --mode plan``, where the agent answers by drafting an
implementation plan instead of replying directly. The prompts never touch files.

    agentflow run examples/antigravity_smoke.py
"""

import os

from agentflow import Graph, antigravity


MODEL = os.environ.get("AGENTFLOW_ANTIGRAVITY_MODEL", "gemini-4-neon-low")

with Graph("antigravity-smoke", working_dir=".", concurrency=1) as dag:
    ping = antigravity(
        task_id="ping",
        prompt="Reply with exactly: PONG",
        model=MODEL,
        tools="read_write",
        timeout_seconds=180,
        success_criteria=[{"kind": "output_contains", "value": "PONG"}],
    )

    capital = antigravity(
        task_id="capital",
        prompt="The previous step said {{ nodes.ping.output }}. Give the capital of France.",
        model=MODEL,
        tools="read_write",
        timeout_seconds=180,
        json_schema={
            "type": "object",
            "properties": {"capital": {"type": "string"}},
            "required": ["capital"],
        },
        success_criteria=[{"kind": "output_contains", "value": "Paris"}],
    )

    ping >> capital


print(dag.to_json())
