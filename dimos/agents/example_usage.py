# Copyright 2025 Dimensional Inc.
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

"""Example usage of the refactored agent system."""

import os
from reactivex import operators as ops

from dimos.agents.refactored_claude_agent import RefactoredClaudeAgent
from dimos.agents.refactored_cerebras_agent import RefactoredCerebrasAgent
from dimos.agents.base import ModelCapability
from dimos.utils.logging_config import setup_logger

logger = setup_logger("dimos.agents.example")


def example_claude_agent():
    """Example of using the refactored Claude agent."""
    print("=== Claude Agent Example ===")
    
    # Create Claude agent
    agent = RefactoredClaudeAgent(
        dev_name="example-claude",
        model_name="claude-3-5-sonnet-20241022",
        thinking_budget_tokens=1000
    )
    
    # Simple text query
    print("Sending text query...")
    response = agent.run_observable_query("What is the capital of France?").run()
    print(f"Response: {response.content}")
    
    # Query with image (will work for Claude)
    print("\nSending query with image...")
    # Note: In real usage, you'd provide an actual base64 image
    # response = agent.run_observable_query(
    #     "What do you see in this image?",
    #     base64_image="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
    # ).run()
    # print(f"Response: {response.content}")


def example_cerebras_agent():
    """Example of using the refactored Cerebras agent."""
    print("\n=== Cerebras Agent Example ===")
    
    # Create Cerebras agent
    agent = RefactoredCerebrasAgent(
        dev_name="example-cerebras",
        model_name="llama-4-scout-17b-16e-instruct"
    )
    
    # Simple text query
    print("Sending text query...")
    response = agent.run_observable_query("What is the capital of France?").run()
    print(f"Response: {response.content}")
    
    # Query with image (will be automatically skipped for Cerebras)
    print("\nSending query with image (will be skipped)...")
    response = agent.run_observable_query(
        "What do you see in this image?",
        base64_image="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
    ).run()
    print(f"Response: {response.content}")


def example_capability_checking():
    """Example showing how capability checking works."""
    print("\n=== Capability Checking Example ===")
    
    # Claude agent (supports images)
    claude_agent = RefactoredClaudeAgent(dev_name="claude")
    print(f"Claude supports images: {claude_agent.llm_provider.supports_capability(ModelCapability.MULTIMODAL)}")
    print(f"Claude supports streaming: {claude_agent.llm_provider.supports_capability(ModelCapability.STREAMING)}")
    print(f"Claude supports thinking: {claude_agent.llm_provider.supports_capability(ModelCapability.THINKING)}")
    
    # Cerebras agent (text-only)
    cerebras_agent = RefactoredCerebrasAgent(dev_name="cerebras")
    print(f"Cerebras supports images: {cerebras_agent.llm_provider.supports_capability(ModelCapability.MULTIMODAL)}")
    print(f"Cerebras supports streaming: {cerebras_agent.llm_provider.supports_capability(ModelCapability.STREAMING)}")
    print(f"Cerebras supports thinking: {cerebras_agent.llm_provider.supports_capability(ModelCapability.THINKING)}")


def example_with_skills():
    """Example showing how to use agents with skills."""
    print("\n=== Skills Example ===")
    
    # Create a simple skill
    from dimos.skills.skills import AbstractSkill
    from pydantic import Field
    
    class SimpleSkill(AbstractSkill):
        name: str = Field("simple_skill", description="A simple test skill")
        description: str = Field("A simple skill that returns a greeting", description="Skill description")
        
        def __call__(self):
            return f"Hello from {self.name}!"
    
    # Create agent with skills
    agent = RefactoredClaudeAgent(
        dev_name="skills-example",
        skills=SimpleSkill()
    )
    
    # Query that might trigger the skill
    print("Sending query that might trigger skill...")
    response = agent.run_observable_query("Can you run the simple_skill?").run()
    print(f"Response: {response.content}")


if __name__ == "__main__":
    # Run examples
    try:
        example_claude_agent()
        example_cerebras_agent()
        example_capability_checking()
        example_with_skills()
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"Error: {e}")
        print("Note: This example requires proper API keys and dependencies to be installed.")