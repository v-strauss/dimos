# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompts for eval generation."""

SINGLE_TURN_SYSTEM_PROMPT = """You are an expert at generating training data for fine-tuning language models to use robot control tools.

Your task is to generate realistic user queries that would require using the available tools, along with the appropriate tool calls that should be made in response.

Guidelines:
1. Generate diverse, natural-sounding user queries that a human might actually say to a robot
2. Queries should vary in complexity, formality, and specificity
3. Include both simple direct commands and more conversational requests
4. Make sure tool call arguments are realistic and properly formatted
5. Each query should map to one or more appropriate tool calls
6. Consider edge cases and different ways users might phrase the same request

Output format: Generate a JSON array where each element has:
- "user_query": The natural language user request
- "tool_calls": Array of tool calls, each with "name" and "arguments" (as a dict)

Example output:
[
  {
    "user_query": "Can you do a backflip?",
    "tool_calls": [{"name": "execute_sport_command", "arguments": {"command_name": "BackFlip"}}]
  },
  {
    "user_query": "Move forward about 2 meters please",
    "tool_calls": [{"name": "move", "arguments": {"x": 2.0, "y": 0.0, "yaw": 0.0}}]
  }
]"""

MULTI_TURN_SYSTEM_PROMPT = """You are an expert at generating multi-turn conversation training data for fine-tuning language models to control robots.

Your task is to generate realistic multi-turn conversations where a user interacts with a robot assistant. The conversations should demonstrate:
1. Sequential tool usage across multiple turns
2. Follow-up questions and clarifications
3. Tool execution results influencing subsequent interactions
4. Natural conversation flow with context awareness

Guidelines:
1. Each conversation should have 2-5 turns
2. Include varied scenarios: navigation, inspection, manipulation, etc.
3. Show realistic tool execution results (success, partial success, or requiring clarification)
4. The assistant should acknowledge tool results naturally
5. User follow-ups should be context-aware

Output format: Generate a JSON array where each element is a conversation with:
- "conversation": Array of message objects with "role" and "content"
- For assistant messages with tool calls, include "tool_calls" array
- Include "tool_results" for simulated tool execution outcomes

Example output:
[
  {
    "conversation": [
      {"role": "user", "content": "Go check what's in the kitchen"},
      {"role": "assistant", "content": "I'll navigate to the kitchen now.", "tool_calls": [{"name": "navigate_to_waypoint", "arguments": {"waypoint_name": "kitchen"}}]},
      {"role": "tool_result", "name": "navigate_to_waypoint", "content": "Successfully arrived at kitchen waypoint"},
      {"role": "user", "content": "Great, what do you see?"},
      {"role": "assistant", "content": "Let me look around.", "tool_calls": [{"name": "look_around", "arguments": {}}]}
    ]
  }
]"""


def build_single_turn_prompt(tools: list[dict], num_evals: int) -> str:
    """Build the user prompt for single-turn eval generation.

    Args:
        tools: List of OpenAI-format tool definitions.
        num_evals: Number of evals to generate.

    Returns:
        The formatted user prompt.
    """
    import json

    tools_str = json.dumps(tools, indent=2)
    return f"""Generate {num_evals} diverse training examples using these available tools:

{tools_str}

Remember to:
- Generate varied and natural user queries
- Ensure tool calls have correct argument types and values
- Cover different tools and use cases
- Include both simple and complex queries

Output only the JSON array, no other text."""


def build_multi_turn_prompt(tools: list[dict], num_evals: int, max_turns: int) -> str:
    """Build the user prompt for multi-turn eval generation.

    Args:
        tools: List of OpenAI-format tool definitions.
        num_evals: Number of conversations to generate.
        max_turns: Maximum turns per conversation.

    Returns:
        The formatted user prompt.
    """
    import json

    tools_str = json.dumps(tools, indent=2)
    return f"""Generate {num_evals} diverse multi-turn conversations using these available tools:

{tools_str}

Each conversation should have 2-{max_turns} turns and demonstrate realistic robot-human interaction.

Remember to:
- Create coherent conversation flows
- Include realistic tool execution results
- Show the assistant acknowledging and acting on user requests
- Demonstrate context awareness across turns

Output only the JSON array, no other text."""
