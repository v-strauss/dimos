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

"""Cerebras provider implementation for the refactored agent system."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from cerebras.cloud.sdk import Cerebras
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

logger = setup_logger("dimos.agents.providers.cerebras")


class CerebrasProvider(LLMProvider):
    """Cerebras provider implementation using the official Cerebras Python SDK."""

    def __init__(self, model_name: str = "llama-4-scout-17b-16e-instruct"):
        """
        Initialize the Cerebras provider.
        
        Args:
            model_name: The Cerebras model to use
        """
        # Cerebras models are text-only by default
        capabilities = [ModelCapability.TEXT_ONLY, ModelCapability.TOOL_CALLING]
        
        # Determine token limits based on model
        if "70b" in model_name.lower():
            max_input_tokens = 32768
            max_output_tokens = 4096
        elif "32b" in model_name.lower():
            max_input_tokens = 16384
            max_output_tokens = 2048
        else:
            max_input_tokens = 8192
            max_output_tokens = 1024
        
        model_config = ModelConfig(
            name=model_name,
            capabilities=capabilities,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            supports_images=False,  # Cerebras models are text-only
            supports_tools=True,
            supports_streaming=False,  # Cerebras doesn't support streaming yet
            supports_thinking=False
        )
        
        super().__init__(model_config)
        self.client = Cerebras()
    
    def send_query(
        self, 
        messages: List[Message], 
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Observable[LLMResponse]]:
        """Send a query to Cerebras."""
        if stream:
            raise NotImplementedError("Cerebras does not support streaming yet")
        
        # Convert messages to Cerebras format
        cerebras_messages = self.convert_messages_to_provider_format(messages)
        
        # Prepare Cerebras parameters
        cerebras_params = {
            "model": self.model_config.name,
            "messages": cerebras_messages,
            "max_tokens": kwargs.get("max_tokens", self.model_config.max_output_tokens),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }
        
        # Add tools if present
        if messages and any(msg.tool_calls for msg in messages if msg.tool_calls):
            tools = []
            for msg in messages:
                if msg.tool_calls:
                    tools.extend(msg.tool_calls)
            if tools:
                cerebras_params["tools"] = self.convert_tools_to_provider_format(tools)
        
        return self._single_query(cerebras_params)
    
    def _single_query(self, cerebras_params: Dict[str, Any]) -> LLMResponse:
        """Send a single query to Cerebras."""
        try:
            response = self.client.chat.completions.create(**cerebras_params)
            return self.convert_response_from_provider_format(response)
        except Exception as e:
            logger.error(f"Error in Cerebras API call: {e}")
            raise
    
    def convert_tools_to_provider_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to Cerebras format."""
        cerebras_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Clean the schema for Cerebras
                schema = tool.get("function", {}).get("parameters", {})
                cleaned_schema = self._clean_cerebras_schema(schema)
                
                cerebras_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("function", {}).get("name", ""),
                        "description": tool.get("function", {}).get("description", ""),
                        "parameters": cleaned_schema
                    }
                }
                cerebras_tools.append(cerebras_tool)
        return cerebras_tools
    
    def _clean_cerebras_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean schema for Cerebras compatibility."""
        if not isinstance(schema, dict):
            return {}
        
        cleaned = schema.copy()
        
        # Remove problematic fields
        for key in ["$schema", "additionalProperties"]:
            cleaned.pop(key, None)
        
        # Clean properties recursively
        if "properties" in cleaned:
            for prop_name, prop_schema in cleaned["properties"].items():
                if isinstance(prop_schema, dict):
                    cleaned["properties"][prop_name] = self._clean_cerebras_schema(prop_schema)
        
        return cleaned
    
    def convert_messages_to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Cerebras format."""
        cerebras_messages = []
        
        for msg in messages:
            # Cerebras only supports text content
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Extract text content from multimodal messages
                text_parts = []
                for content_item in msg.content:
                    if content_item.get("type") == "text":
                        text_parts.append(content_item["text"])
                    elif content_item.get("type") == "image_url":
                        # Skip images for text-only models
                        logger.warning("Skipping image content for text-only Cerebras model")
                content = " ".join(text_parts)
            else:
                content = str(msg.content)
            
            cerebras_msg = {
                "role": msg.role,
                "content": content
            }
            
            cerebras_messages.append(cerebras_msg)
        
        return cerebras_messages
    
    def convert_response_from_provider_format(self, response: Any) -> LLMResponse:
        """Convert Cerebras response to standard format."""
        content = ""
        tool_calls = []
        
        # Extract content from the first choice
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Extract tool calls if present
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tool_call_obj = ToolCall(
                        id=tool_call.id,
                        function={
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    )
                    tool_calls.append(tool_call_obj)
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=getattr(response, 'usage', None),
            finish_reason=getattr(response.choices[0], 'finish_reason', None) if response.choices else None
        )