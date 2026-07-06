"""Prompt formatters for decentralized BFCL tool-calling agents."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List


def _function_docs(example: Dict[str, Any]) -> str:
    chunks = []
    for function in example.get("function", []):
        chunks.append(json.dumps(function, ensure_ascii=False, sort_keys=True))
    return "\n".join(chunks)


def _role_instruction(agent_idx: int, num_agents: int, role_mode: str) -> str:
    if role_mode == "same_prompt":
        return (
            "Handle every tool-call intent you can infer from the user request. "
            "Another agent may answer independently; do not mention them."
        )
    if role_mode == "odd_even":
        parity = "odd-numbered" if agent_idx % 2 == 0 else "even-numbered"
        return (
            f"Handle the {parity} tool-call intents in the user's request, using "
            "1-based ordering from left to right. Skip intents assigned to the "
            "other agent."
        )

    # Default: split_by_order. For two agents this is first half / second half.
    if num_agents == 2:
        if agent_idx == 0:
            return (
                "Handle the first half of the tool-call intents in the user's "
                "request, in user-request order. If the number of intents is odd, "
                "take the extra middle intent."
            )
        return (
            "Handle the second half of the tool-call intents in the user's request, "
            "in user-request order. Skip the first-half intents."
        )

    return (
        f"You are agent {agent_idx + 1} of {num_agents}. Split the ordered "
        "tool-call intents into contiguous chunks and handle only your chunk."
    )


def build_bfcl_formatter(
    agent_idx: int,
    *,
    num_agents: int = 2,
    role_mode: str = "split_by_order",
) -> Callable[[Dict[str, Any]], str]:
    """Build one agent-specific BFCL prompt formatter."""

    def formatter(example: Dict[str, Any], external_prompts: Any = None) -> str:
        del external_prompts
        user_prompt = example.get("user_prompt") or example.get("prompt") or ""
        official_category = example.get("official_category", "unknown")
        task_type = example.get("task_type", "unknown")
        role_instruction = _role_instruction(agent_idx, num_agents, role_mode)
        return f"""You are a decentralized function-calling agent.

There are {num_agents} agents answering independently. You cannot communicate with
the other agent and you cannot see their answer. A downstream aggregator will
merge the agents' tool calls.

Your assignment:
{role_instruction}

BFCL category: {official_category}
Heuristic task type: {task_type}

Available function schemas:
{_function_docs(example)}

User request:
{user_prompt}

Output requirements:
- Output only the tool calls assigned to you.
- Use Python-style function-call syntax, one call per line.
- Use keyword arguments from the function schemas.
- Do not include explanations, markdown, numbering, or reasoning.
- If no call is assigned to you, output exactly: []

Example output format:
function_name(arg1="value", arg2=3)
another.function_name(flag=True)
"""

    return formatter


def get_bfcl_formatters(
    *,
    num_agents: int,
    role_mode: str = "split_by_order",
) -> List[Callable[[Dict[str, Any]], str]]:
    return [
        build_bfcl_formatter(i, num_agents=num_agents, role_mode=role_mode)
        for i in range(num_agents)
    ]
