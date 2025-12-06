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

"""Base abstractions for the refactored agent system.

This module provides the core abstractions that enable clean separation between
LLM providers and the agent logic, with proper handling of multimodal capabilities.
"""

from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol

from pydantic import BaseModel
from reactivex import Observable, Subject
from reactivex.scheduler import ThreadPoolScheduler

from dimos.agents.memory.base import AbstractAgentSemanticMemory
from dimos.agents.memory.chroma_impl import OpenAISemanticMemory
from dimos.skills.skills import AbstractSkill, SkillLibrary
from dimos.utils.logging_config import setup_logger

logger = setup_logger("dimos.agents.base")


class ModelCapability(Enum):
    """Enumeration of model capabilities."""
    TEXT_ONLY = "text_only"
    MULTIMODAL = "multimodal"
    TOOL_CALLING = "tool_calling"
    STREAMING = "streaming"
    THINKING = "thinking"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    capabilities: List[ModelCapability]
    max_input_tokens: int
    max_output_tokens: int
    supports_images: bool = False
    supports_tools: bool = False
    supports_streaming: bool = False
    supports_thinking: bool = False


@dataclass
class Message:
    """Standardized message format for all LLM providers."""
    role: str
    content: Union[str, List[Dict[str, Any]]]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    thinking_blocks: Optional[List[str]] = None


@dataclass
class ToolCall:
    """Standardized tool call format."""
    id: str
    function: Dict[str, Any]
    type: str = "function"


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    thinking_blocks: List[str] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class LLMProvider(ABC):
    """Abstract interface for LLM providers.
    
    This interface defines the contract that all LLM providers must implement,
    allowing for clean separation between provider-specific logic and agent logic.
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._conversation_history: List[Message] = []
        self._history_lock = threading.Lock()
    
    @abstractmethod
    def send_query(
        self, 
        messages: List[Message], 
        **kwargs
    ) -> Union[LLMResponse, Observable[LLMResponse]]:
        """Send a query to the LLM provider.
        
        Args:
            messages: List of messages to send
            **kwargs: Provider-specific parameters
            
        Returns:
            Either a single response or an observable stream of responses
        """
        pass
    
    @abstractmethod
    def convert_tools_to_provider_format(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert tools to the provider's specific format.
        
        Args:
            tools: List of tools in standard format
            
        Returns:
            Tools in provider-specific format
        """
        pass
    
    @abstractmethod
    def convert_messages_to_provider_format(self, messages: List[Message]) -> Any:
        """Convert messages to the provider's specific format.
        
        Args:
            messages: List of messages in standard format
            
        Returns:
            Messages in provider-specific format
        """
        pass
    
    @abstractmethod
    def convert_response_from_provider_format(self, response: Any) -> LLMResponse:
        """Convert provider response to standard format.
        
        Args:
            response: Response in provider-specific format
            
        Returns:
            Response in standard format
        """
        pass
    
    def add_to_conversation_history(self, message: Message) -> None:
        """Add a message to the conversation history with thread safety."""
        with self._history_lock:
            self._conversation_history.append(message)
    
    def get_conversation_history(self) -> List[Message]:
        """Get a copy of the conversation history."""
        with self._history_lock:
            return self._conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        with self._history_lock:
            self._conversation_history.clear()
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if the model supports a specific capability."""
        return capability in self.model_config.capabilities


class PromptBuilder(ABC):
    """Abstract interface for prompt builders.
    
    This allows different LLM providers to have their own prompt building logic
    while maintaining a consistent interface.
    """
    
    @abstractmethod
    def build_prompt(
        self,
        system_prompt: str,
        user_query: str,
        base64_image: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        rag_context: str = "",
        **kwargs
    ) -> List[Message]:
        """Build a prompt in the provider's format.
        
        Args:
            system_prompt: System prompt
            user_query: User query
            base64_image: Optional base64 encoded image
            dimensions: Optional image dimensions
            rag_context: Optional RAG context
            **kwargs: Additional parameters
            
        Returns:
            List of messages in provider format
        """
        pass


class BaseAgent:
    """Base agent that manages memory and subscriptions."""

    def __init__(
        self,
        dev_name: str = "NA",
        agent_type: str = "Base",
        agent_memory: Optional[AbstractAgentSemanticMemory] = None,
        pool_scheduler: Optional[ThreadPoolScheduler] = None,
    ):
        """
        Initializes a new instance of the BaseAgent.

        Args:
            dev_name (str): The device name of the agent.
            agent_type (str): The type of the agent (e.g., 'Base', 'Vision').
            agent_memory (AbstractAgentSemanticMemory): The memory system for the agent.
            pool_scheduler (ThreadPoolScheduler): The scheduler to use for thread pool operations.
                If None, the global scheduler from get_scheduler() will be used.
        """
        self.dev_name = dev_name
        self.agent_type = agent_type
        self.agent_memory = agent_memory or OpenAISemanticMemory()
        self.disposables = []
        self.pool_scheduler = pool_scheduler

    def dispose_all(self):
        """Disposes of all active subscriptions managed by this agent."""
        for disposable in self.disposables:
            if hasattr(disposable, 'dispose'):
                disposable.dispose()
        self.disposables.clear()


class BaseLLMAgent(BaseAgent):
    """Base LLM agent with improved abstraction and thread-safe conversation history.
    
    This class provides the core functionality for LLM-based agents while
    delegating provider-specific logic to LLMProvider implementations.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        dev_name: str = "NA",
        agent_type: str = "LLM",
        agent_memory: Optional[AbstractAgentSemanticMemory] = None,
        pool_scheduler: Optional[ThreadPoolScheduler] = None,
        system_query: Optional[str] = None,
        max_output_tokens_per_request: int = 16384,
        max_input_tokens_per_request: int = 128000,
        rag_query_n: int = 4,
        rag_similarity_threshold: float = 0.45,
        skills: Optional[Union[AbstractSkill, List[AbstractSkill], SkillLibrary]] = None,
        process_all_inputs: bool = False,
    ):
        """
        Initializes a new instance of the BaseLLMAgent.

        Args:
            llm_provider: The LLM provider implementation
            dev_name: The device name of the agent
            agent_type: The type of the agent
            agent_memory: The memory system for the agent
            pool_scheduler: The scheduler to use for thread pool operations
            system_query: System prompt for RAG context situations
            max_output_tokens_per_request: Maximum output token count
            max_input_tokens_per_request: Maximum input token count
            rag_query_n: Number of results to fetch from memory
            rag_similarity_threshold: Minimum similarity for RAG results
            skills: Skills available to the agent
            process_all_inputs: Whether to process every input emission
        """
        super().__init__(dev_name, agent_type, agent_memory, pool_scheduler)
        
        self.llm_provider = llm_provider
        self.system_query = system_query
        self.max_output_tokens_per_request = max_output_tokens_per_request
        self.max_input_tokens_per_request = max_input_tokens_per_request
        self.max_tokens_per_request = max_input_tokens_per_request + max_output_tokens_per_request
        self.rag_query_n = rag_query_n
        self.rag_similarity_threshold = rag_similarity_threshold
        self.process_all_inputs = process_all_inputs
        
        # Current query
        self.query: Optional[str] = None
        
        # Subject for emitting responses
        self.response_subject = Subject()
        
        # Configure skills
        self.skills = skills
        self.skill_library = None
        if isinstance(self.skills, SkillLibrary):
            self.skill_library = self.skills
        elif isinstance(self.skills, list):
            self.skill_library = SkillLibrary()
            for skill in self.skills:
                self.skill_library.add(skill)
        elif isinstance(self.skills, AbstractSkill):
            self.skill_library = SkillLibrary()
            self.skill_library.add(self.skills)
        
        # Add static context to memory
        self._add_context_to_memory()
    
    def _add_context_to_memory(self):
        """Add initial context to the agent's memory."""
        context_data = [
            (
                "id0",
                "Optical Flow is a technique used to track the movement of objects in a video sequence.",
            ),
            (
                "id1",
                "Computer Vision is a field of artificial intelligence that trains computers to interpret and understand visual information.",
            ),
        ]
        
        for context_id, context_text in context_data:
            # Note: This will need to be implemented based on the actual memory interface
            # For now, we'll skip this as it's not critical for the refactor
            pass
    
    def _get_rag_context(self) -> Tuple[str, str]:
        """Get RAG context for the current query."""
        if not hasattr(self, 'query') or not self.query:
            return "", ""
        
        try:
            results = self.agent_memory.query(
                self.query, 
                n_results=self.rag_query_n,
                similarity_threshold=self.rag_similarity_threshold
            )
            
            if not results:
                return "", ""
            
            # Handle different result formats
            if hasattr(results[0], 'page_content'):
                condensed_results = "\n".join([result.page_content for result in results])
            elif isinstance(results[0], tuple) and len(results[0]) > 0:
                condensed_results = "\n".join([str(result[0]) for result in results])
            else:
                condensed_results = "\n".join([str(result) for result in results])
            
            return condensed_results, condensed_results
        except Exception as e:
            logger.warning(f"Failed to get RAG context: {e}")
            return "", ""
    
    def run_observable_query(
        self, 
        query_text: str, 
        base64_image: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Observable[LLMResponse]:
        """Run a query and return an observable of the response.
        
        This method handles the case where the model doesn't support images
        by checking capabilities before processing.
        
        Args:
            query_text: The query text
            base64_image: Optional base64 encoded image
            dimensions: Optional image dimensions
            **kwargs: Additional parameters
            
        Returns:
            Observable of the response
        """
        # Check if model supports images
        if base64_image and not self.llm_provider.supports_capability(ModelCapability.MULTIMODAL):
            logger.warning(f"Model {self.llm_provider.model_config.name} does not support images. Skipping image.")
            base64_image = None
            dimensions = None
        
        return self._observable_query(
            query_text=query_text,
            base64_image=base64_image,
            dimensions=dimensions,
            **kwargs
        )
    
    def _observable_query(
        self,
        query_text: str,
        base64_image: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Observable[LLMResponse]:
        """Internal method to handle the observable query logic."""
        # This will be implemented by subclasses to handle the specific
        # observable pattern for each provider
        raise NotImplementedError("Subclasses must implement _observable_query")
    
    def get_response_observable(self) -> Observable[LLMResponse]:
        """Get the observable for agent responses."""
        return self.response_subject
    
    def dispose_all(self):
        """Dispose of all resources."""
        super().dispose_all()
        if hasattr(self.response_subject, 'dispose'):
            self.response_subject.dispose()