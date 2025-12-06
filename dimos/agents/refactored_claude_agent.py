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

"""Refactored Claude agent implementation using the new abstraction layer."""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

from reactivex import Observable, create
from reactivex.observer import Observer
from reactivex.scheduler import ThreadPoolScheduler

from dimos.agents.base import BaseLLMAgent, Message, LLMResponse, ModelCapability
from dimos.agents.providers.claude_provider import ClaudeProvider
from dimos.agents.memory.base import AbstractAgentSemanticMemory
from dimos.skills.skills import AbstractSkill, SkillLibrary
from dimos.utils.logging_config import setup_logger

logger = setup_logger("dimos.agents.refactored_claude")


class RefactoredClaudeAgent(BaseLLMAgent):
    """Refactored Claude agent with minimal implementation using the new abstraction layer."""

    def __init__(
        self,
        dev_name: str,
        agent_type: str = "Vision",
        query: str = "What do you see?",
        input_query_stream=None,
        input_video_stream=None,
        input_data_stream=None,
        output_dir: str = os.path.join(os.getcwd(), "assets", "agent"),
        agent_memory: Optional[AbstractAgentSemanticMemory] = None,
        system_query: Optional[str] = None,
        max_input_tokens_per_request: int = 128000,
        max_output_tokens_per_request: int = 16384,
        model_name: str = "claude-3-5-sonnet-20241022",
        skills: Optional[Union[AbstractSkill, list[AbstractSkill], SkillLibrary]] = None,
        frame_processor=None,
        image_detail: str = "low",
        pool_scheduler: Optional[ThreadPoolScheduler] = None,
        process_all_inputs: Optional[bool] = None,
        thinking_budget_tokens: Optional[int] = 2000,
    ):
        """
        Initialize the refactored Claude agent.
        
        Args:
            dev_name: The device name of the agent
            agent_type: The type of the agent
            query: The default query text
            input_query_stream: Observable for query input
            input_video_stream: Observable for video frames
            input_data_stream: Observable for data input
            output_dir: Directory for output files
            agent_memory: The memory system for the agent
            system_query: System prompt for RAG context situations
            max_input_tokens_per_request: Maximum input token count
            max_output_tokens_per_request: Maximum output token count
            model_name: The Claude model name to use
            skills: Skills available to the agent
            frame_processor: Custom frame processor
            image_detail: Detail level for images
            pool_scheduler: The scheduler to use for thread pool operations
            process_all_inputs: Whether to process every input emission
            thinking_budget_tokens: Number of tokens to allocate for Claude's thinking
        """
        # Determine appropriate default for process_all_inputs if not provided
        if process_all_inputs is None:
            if input_query_stream is not None and input_video_stream is None:
                process_all_inputs = True
            else:
                process_all_inputs = False

        # Create Claude provider
        claude_provider = ClaudeProvider(model_name=model_name)
        
        # Initialize base class
        super().__init__(
            llm_provider=claude_provider,
            dev_name=dev_name,
            agent_type=agent_type,
            agent_memory=agent_memory,
            pool_scheduler=pool_scheduler,
            system_query=system_query,
            max_output_tokens_per_request=max_output_tokens_per_request,
            max_input_tokens_per_request=max_input_tokens_per_request,
            skills=skills,
            process_all_inputs=process_all_inputs,
        )
        
        # Set additional attributes
        self.query = query
        self.output_dir = output_dir
        self.thinking_budget_tokens = thinking_budget_tokens
        self.frame_processor = frame_processor
        self.image_detail = image_detail
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up input streams
        self.input_video_stream = input_video_stream
        self.input_query_stream = input_query_stream
        self.input_data_stream = input_data_stream
        
        logger.info("Refactored Claude Agent Initialized.")

    def _observable_query(
        self,
        query_text: str,
        base64_image: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Observable[LLMResponse]:
        """Handle the observable query pattern for Claude."""
        def subscribe(observer: Observer[LLMResponse]):
            try:
                # Update query
                self.query = query_text
                
                # Get RAG context
                rag_context, condensed_results = self._get_rag_context()
                
                # Build messages
                messages = self._build_messages(
                    query_text=query_text,
                    base64_image=base64_image,
                    dimensions=dimensions,
                    rag_context=rag_context,
                    thinking_budget_tokens=self.thinking_budget_tokens
                )
                
                # Send query to provider
                response = self.llm_provider.send_query(
                    messages=messages,
                    stream=True,
                    thinking_budget_tokens=self.thinking_budget_tokens,
                    max_tokens=self.max_output_tokens_per_request
                )
                
                # Handle streaming response
                if hasattr(response, 'subscribe'):
                    subscription = response.subscribe(
                        on_next=lambda r: observer.on_next(r),
                        on_error=lambda e: observer.on_error(e),
                        on_completed=lambda: observer.on_completed()
                    )
                else:
                    # Single response
                    observer.on_next(response)
                    observer.on_completed()
                    
            except Exception as e:
                logger.error(f"Error in Claude observable query: {e}")
                observer.on_error(e)
        
        return create(subscribe)
    
    def _build_messages(
        self,
        query_text: str,
        base64_image: Optional[str] = None,
        dimensions: Optional[Tuple[int, int]] = None,
        rag_context: str = "",
        thinking_budget_tokens: Optional[int] = None
    ) -> list[Message]:
        """Build messages for Claude."""
        messages = []
        
        # Add system message
        system_content = self.system_query or "You are a helpful AI assistant."
                    if rag_context:
                system_content += f"\n\nContext: {rag_context}"
            
            messages.append(Message(role="system", content=system_content))
            
            # Add user message
            if base64_image and self.llm_provider.supports_capability(ModelCapability.MULTIMODAL):
                # Multimodal message
                content = []
                content.append({"type": "text", "text": query_text})
                if base64_image:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": self.image_detail
                        }
                    })
                messages.append(Message(role="user", content=content))
            else:
                # Text-only message
                if rag_context:
                    user_content = f"{rag_context}\n\n{query_text}"
                else:
                    user_content = query_text
                messages.append(Message(role="user", content=user_content))
        
        return messages