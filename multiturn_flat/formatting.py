"""Prompt formatters for flattened BFCL multi-turn step agents."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List


def _function_docs(example: Dict[str, Any]) -> str:
    return "\n".join(
        json.dumps(function, ensure_ascii=False, sort_keys=True)
        for function in example.get("function", [])
    )


def _metadata_lines(example: Dict[str, Any]) -> str:
    lines = [
        f"BFCL multiturn category: {example.get('official_category', 'unknown')}",
        f"Heuristic task type: {example.get('task_type', 'unknown')}",
    ]
    turn_index = example.get("turn_index")
    try:
        turn_number = int(turn_index) + 1
        lines.append(
            f"Original multi-turn trajectory turn: {turn_number} "
            f"(zero-based turn_index={int(turn_index)})"
        )
    except (TypeError, ValueError):
        lines.append(f"Original multi-turn trajectory turn_index: {turn_index}")
    return "\n".join(lines)


def _role_instruction(agent_idx: int, num_agents: int, role_mode: str) -> str:
    if role_mode == "same_prompt":
        return (
            "Handle every current-turn tool-call intent you can infer from the "
            "provided dialogue context. Another agent may answer independently; "
            "do not mention them."
        )
    if role_mode == "self_select":
        return (
            "Another agent is working on the same current turn in parallel. You "
            "do not need to complete every tool-call intent by yourself, but you "
            "should contribute a useful non-empty subset whenever this turn needs "
            "tool calls. Choose the intents you are most confident about, avoid "
            "trying to cover everything, and avoid doing nothing unless there is "
            "truly no valid tool call for you to make."
        )
    if role_mode == "odd_even":
        parity = "odd-numbered" if agent_idx % 2 == 0 else "even-numbered"
        return (
            f"Handle the {parity} current-turn tool-call intents, using 1-based "
            "ordering from left to right. Skip intents assigned to the other agent."
        )

    if num_agents == 2:
        if agent_idx == 0:
            return (
                "Handle the first half of the current-turn tool-call intents, in "
                "gold-call order when the request implies an order. If the number "
                "of intents is odd, take the extra middle intent."
            )
        return (
            "Handle the second half of the current-turn tool-call intents, in "
            "gold-call order when the request implies an order. Skip first-half "
            "intents."
        )

    return (
        f"You are agent {agent_idx + 1} of {num_agents}. Split the current-turn "
        "tool-call intents into contiguous chunks and handle only your chunk."
    )


def build_multiturn_flat_formatter(
    agent_idx: int,
    *,
    num_agents: int = 2,
    role_mode: str = "self_select",
) -> Callable[[Dict[str, Any]], str]:
    def formatter(example: Dict[str, Any], external_prompts: Any = None) -> str:
        del external_prompts
        user_prompt = example.get("user_prompt") or example.get("prompt") or ""
        return f"""You are a decentralized function-calling agent for one flattened BFCL multi-turn step.

There are {num_agents} agents answering independently. You cannot communicate with
the other agent and you cannot see their answer. A downstream aggregator will
merge the agents' current-turn tool calls.

Your assignment:
{_role_instruction(agent_idx, num_agents, role_mode)}

{_metadata_lines(example)}

Available function schemas:
{_function_docs(example)}

Current flattened dialogue context and user request:
{user_prompt}

Output requirements:
- Output only current-turn tool calls assigned to you or selected by you.
- Use Python-style function-call syntax, one call per line.
- Preserve the order of the tool calls you emit.
- Use keyword arguments from the function schemas.
- Do not include explanations, markdown, numbering, or reasoning.
- If you make no tool call, output exactly: []

Example output format:
function_name(arg1="value", arg2=3)
another.function_name(flag=True)
"""

    return formatter


def get_multiturn_flat_formatters(
    *,
    num_agents: int,
    role_mode: str = "self_select",
) -> List[Callable[[Dict[str, Any]], str]]:
    return [
        build_multiturn_flat_formatter(i, num_agents=num_agents, role_mode=role_mode)
        for i in range(num_agents)
    ]
