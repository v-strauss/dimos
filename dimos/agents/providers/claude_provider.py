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

"""Claude provider implementation for the refactored agent system."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import anthropic
from reactivex import Observable, create
from reactivex.observer import Observer

from dimos.agents.base import (
    LLMProvider, 
    ModelConfig, 
    ModelCapability, 
    Message, 
    LLMResponse, 
    ToolCall
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger("dimos.agents.providers.claude")


class ClaudeProvider(LLMProvider):
    """Claude provider implementation using Anthropic's API."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        """
        Initialize the Claude provider.
        
        Args:
            model_name: The Claude model to use
            api_key: Optional API key (will use environment variable if not provided)
        """
        # Determine model capabilities based on model name
        capabilities = [ModelCapability.TEXT_ONLY, ModelCapability.TOOL_CALLING, ModelCapability.STREAMING]
        
        # Add multimodal capability for vision models
        if "vision" in model_name.lower() or "haiku" in model_name.lower():
            capabilities.append(ModelCapability.MULTIMODAL)
        
        # Add thinking capability for models that support it
        if "opus" in model_name.lower() or "sonnet" in model_name.lower():
            capabilities.append(ModelCapability.THINKING)
        
        model_config = ModelConfig(
            name=model_name,
            capabilities=capabilities,
            max_input_tokens=200000,
            max_output_tokens=4096,
            supports_images=ModelCapability.MULTIMODAL in capabilities,
            supports_tools=True,
            supports_streaming=True,
            supports_thinking=ModelCapability.THINKING in capabilities
        )
        
        super().__init__(model_config)
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def send_query(
        self, 
        messages: List[Message], 
        stream: bool = False,
        thinking_budget_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[LLMResponse, Observable[LLMResponse]]:
        """Send a query to Claude."""
        claude_messages = self.convert_messages_to_provider_format(messages)
        
        # Prepare Claude parameters
        claude_params = {
            "model": self.model_config.name,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": claude_messages,
        }
        
        # Add thinking budget if supported and specified
        if (self.supports_capability(ModelCapability.THINKING) and 
            thinking_budget_tokens is not None and thinking_budget_tokens > 0):
            claude_params["thinking_budget_tokens"] = thinking_budget_tokens
        
        # Add tools if present
        if messages and any(msg.tool_calls for msg in messages if msg.tool_calls):
            tools = []
            for msg in messages:
                if msg.tool_calls:
                    tools.extend(msg.tool_calls)
            if tools:
                claude_params["tools"] = self.convert_tools_to_provider_format(tools)
        
        if stream:
            return self._stream_query(claude_params)
        else:
            return self._single_query(claude_params)
    
    def _single_query(self, claude_params: Dict[str, Any]) -> LLMResponse:
        """Send a single query to Claude."""
        try:
            response = self.client.messages.create(**claude_params)
            return self.convert_response_from_provider_format(response)
        except Exception as e:
            logger.error(f"Error in Claude API call: {e}")
            raise
    
    def _stream_query(self, claude_params: Dict[str, Any]) -> Observable[LLMResponse]:
        """Send a streaming query to Claude."""
        def subscribe(observer: Observer[LLMResponse]):
            try:
                with self.client.messages.stream(**claude_params) as stream:
                    for chunk in stream:
                        if chunk.type == "content_block_delta":
                            # Handle content chunks
                            response = LLMResponse(
                                content=chunk.delta.text or "",
                                finish_reason=None
                            )
                            observer.on_next(response)
                        elif chunk.type == "message_delta":
                            # Handle final message
                            if chunk.delta.stop_reason:
                                response = LLMResponse(
                                    content="",
                                    finish_reason=chunk.delta.stop_reason
                                )
                                observer.on_next(response)
                                observer.on_completed()
                        elif chunk.type == "tool_use":
                            # Handle tool calls
                            tool_call = ToolCall(
                                id=chunk.tool_use.id,
                                function={
                                    "name": chunk.tool_use.name,
                                    "arguments": json.dumps(chunk.tool_use.input)
                                }
                            )
                            response = LLMResponse(
                                content="",
                                tool_calls=[tool_call]
                            )
                            observer.on_next(response)
                        elif chunk.type == "thinking":
                            # Handle thinking blocks
                            response = LLMResponse(
                                content="",
                                thinking_blocks=[chunk.thinking.content]
                            )
                            observer.on_next(response)
            except Exception as e:
                logger.error(f"Error in Claude streaming: {e}")
                observer.on_error(e)
        
        return create(subscribe)
    
    def convert_tools_to_provider_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to Claude format."""
        claude_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                claude_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("function", {}).get("name", ""),
                        "description": tool.get("function", {}).get("description", ""),
                        "input_schema": tool.get("function", {}).get("parameters", {})
                    }
                }
                claude_tools.append(claude_tool)
        return claude_tools
    
    def convert_messages_to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Claude format."""
        claude_messages = []
        
        for msg in messages:
            claude_msg: Dict[str, Any] = {"role": msg.role}
            
            if isinstance(msg.content, str):
                claude_msg["content"] = msg.content
            elif isinstance(msg.content, list):
                # Handle multimodal content
                content_list = []
                for content_item in msg.content:
                    if content_item.get("type") == "text":
                        content_list.append({
                            "type": "text",
                            "text": content_item["text"]
                        })
                    elif content_item.get("type") == "image_url":
                        content_list.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content_item.get("media_type", "image/jpeg"),
                                "data": content_item["image_url"]["url"].replace("data:image/jpeg;base64,", "")
                            }
                        })
                claude_msg["content"] = content_list
            
            claude_messages.append(claude_msg)
        
        return claude_messages
    
    def convert_response_from_provider_format(self, response: Any) -> LLMResponse:
        """Convert Claude response to standard format."""
        content = ""
        tool_calls = []
        thinking_blocks = []
        
        # Extract content
        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
            elif content_block.type == "tool_use":
                tool_call = ToolCall(
                    id=content_block.id,
                    function={
                        "name": content_block.name,
                        "arguments": json.dumps(content_block.input)
                    }
                )
                tool_calls.append(tool_call)
        
        # Extract thinking blocks if present
        if hasattr(response, 'thinking') and response.thinking:
            for thinking_block in response.thinking:
                thinking_blocks.append(thinking_block.content)
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            thinking_blocks=thinking_blocks,
            usage=getattr(response, 'usage', None),
            finish_reason=getattr(response, 'stop_reason', None)
        )