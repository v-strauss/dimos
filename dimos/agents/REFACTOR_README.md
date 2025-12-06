# Agent System Refactor

This document explains the comprehensive refactor of the DIMOS agent system to create a cleaner, more maintainable architecture.

## Overview

The original agent system had several issues:
- **Code duplication**: Claude and Cerebras agents had ~700 lines each with significant overlap
- **Poor abstraction**: Each agent overrode many base methods, making maintenance difficult
- **No capability handling**: Text-only models like Cerebras would break when given images
- **Thread safety issues**: Conversation history wasn't properly protected during concurrent tool calls
- **Provider coupling**: Agent logic was tightly coupled to specific LLM providers

## New Architecture

### Core Abstractions

#### 1. `LLMProvider` Interface
```python
class LLMProvider(ABC):
    def send_query(self, messages: List[Message], **kwargs) -> Union[LLMResponse, Observable[LLMResponse]]:
        pass
    
    def convert_tools_to_provider_format(self, tools: List[Dict[str, Any]]) -> Any:
        pass
    
    def convert_messages_to_provider_format(self, messages: List[Message]) -> Any:
        pass
    
    def convert_response_from_provider_format(self, response: Any) -> LLMResponse:
        pass
```

#### 2. `ModelConfig` and Capabilities
```python
@dataclass
class ModelConfig:
    name: str
    capabilities: List[ModelCapability]
    max_input_tokens: int
    max_output_tokens: int
    supports_images: bool = False
    supports_tools: bool = False
    supports_streaming: bool = False
    supports_thinking: bool = False

class ModelCapability(Enum):
    TEXT_ONLY = "text_only"
    MULTIMODAL = "multimodal"
    TOOL_CALLING = "tool_calling"
    STREAMING = "streaming"
    THINKING = "thinking"
```

#### 3. Standardized Message and Response Formats
```python
@dataclass
class Message:
    role: str
    content: Union[str, List[Dict[str, Any]]]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    thinking_blocks: Optional[List[str]] = None

@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    thinking_blocks: List[str] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
```

### Base Classes

#### `BaseLLMAgent`
The new base class provides:
- Thread-safe conversation history with locks
- Automatic capability checking (e.g., skipping images for text-only models)
- Unified RAG context handling
- Standardized response observables
- Clean separation of concerns

## Provider Implementations

### ClaudeProvider (~200 lines)
- Handles Claude-specific API requirements
- Supports multimodal content, tool calling, streaming, and thinking
- Converts between standard and Claude-specific formats
- Manages conversation history with thread safety

### CerebrasProvider (~150 lines)
- Handles Cerebras-specific API requirements
- Text-only implementation (automatically skips images)
- Supports tool calling but not streaming
- Cleans schemas for Cerebras compatibility

## Refactored Agents

### RefactoredClaudeAgent (~100 lines)
- Minimal implementation using the abstraction layer
- Inherits all core functionality from `BaseLLMAgent`
- Only implements Claude-specific observable query pattern
- Handles multimodal content and thinking budgets

### RefactoredCerebrasAgent (~80 lines)
- Even more minimal implementation
- Text-only by design
- Automatically handles image skipping
- No streaming support (as per Cerebras limitations)

## Key Improvements

### 1. Capability-Based Image Handling
```python
def run_observable_query(self, query_text: str, base64_image: Optional[str] = None, **kwargs):
    # Check if model supports images
    if base64_image and not self.llm_provider.supports_capability(ModelCapability.MULTIMODAL):
        logger.warning(f"Model {self.llm_provider.model_config.name} does not support images. Skipping image.")
        base64_image = None
        dimensions = None
    
    return self._observable_query(query_text=query_text, base64_image=base64_image, **kwargs)
```

### 2. Thread-Safe Conversation History
```python
def add_to_conversation_history(self, message: Message) -> None:
    """Add a message to the conversation history with thread safety."""
    with self._history_lock:
        self._conversation_history.append(message)
```

### 3. Clean Provider Separation
Each provider only needs to implement:
- API-specific message conversion
- API-specific tool conversion  
- API-specific response conversion
- Provider-specific query sending

### 4. Automatic Tool Call Handling
The base class handles tool calling logic, while providers only need to convert formats.

## Migration Guide

### From Original ClaudeAgent
```python
# Old way
from dimos.agents.claude_agent import ClaudeAgent
agent = ClaudeAgent(dev_name="test", model_name="claude-3-5-sonnet-20241022")

# New way
from dimos.agents.refactored_claude_agent import RefactoredClaudeAgent
agent = RefactoredClaudeAgent(dev_name="test", model_name="claude-3-5-sonnet-20241022")
```

### From Original CerebrasAgent
```python
# Old way
from dimos.agents.cerebras_agent import CerebrasAgent
agent = CerebrasAgent(dev_name="test", model_name="llama-4-scout-17b-16e-instruct")

# New way
from dimos.agents.refactored_cerebras_agent import RefactoredCerebrasAgent
agent = RefactoredCerebrasAgent(dev_name="test", model_name="llama-4-scout-17b-16e-instruct")
```

## Adding New Providers

To add a new LLM provider:

1. **Create a provider class**:
```python
class NewProvider(LLMProvider):
    def __init__(self, model_name: str):
        model_config = ModelConfig(
            name=model_name,
            capabilities=[ModelCapability.TEXT_ONLY, ModelCapability.TOOL_CALLING],
            max_input_tokens=8192,
            max_output_tokens=1024
        )
        super().__init__(model_config)
        self.client = NewClient()
    
    def send_query(self, messages: List[Message], **kwargs) -> LLMResponse:
        # Convert messages to provider format
        provider_messages = self.convert_messages_to_provider_format(messages)
        
        # Send to provider API
        response = self.client.chat.completions.create(
            messages=provider_messages,
            **kwargs
        )
        
        # Convert response back to standard format
        return self.convert_response_from_provider_format(response)
    
    # Implement other abstract methods...
```

2. **Create an agent class**:
```python
class NewAgent(BaseLLMAgent):
    def __init__(self, dev_name: str, model_name: str = "default-model", **kwargs):
        provider = NewProvider(model_name=model_name)
        super().__init__(llm_provider=provider, dev_name=dev_name, **kwargs)
    
    def _observable_query(self, query_text: str, **kwargs) -> Observable[LLMResponse]:
        # Implement provider-specific observable pattern
        pass
```

## Benefits

1. **Reduced Code**: ~80% reduction in agent implementation code
2. **Better Maintainability**: Changes to core logic only need to be made in one place
3. **Capability Safety**: Automatic handling of model capabilities prevents runtime errors
4. **Thread Safety**: Proper locking prevents race conditions during tool calls
5. **Easy Extension**: Adding new providers requires minimal code
6. **Consistent Interface**: All agents have the same public API
7. **Better Testing**: Provider logic can be tested independently

## File Structure

```
dimos/agents/
├── base.py                          # Core abstractions and base classes
├── providers/
│   ├── claude_provider.py           # Claude-specific provider (~200 lines)
│   └── cerebras_provider.py         # Cerebras-specific provider (~150 lines)
├── refactored_claude_agent.py       # Refactored Claude agent (~100 lines)
├── refactored_cerebras_agent.py     # Refactored Cerebras agent (~80 lines)
├── claude_agent.py                  # Original implementation (deprecated)
└── cerebras_agent.py                # Original implementation (deprecated)
```

## Future Work

1. **Migrate existing code**: Update all imports to use refactored agents
2. **Add more providers**: OpenAI, Gemini, etc.
3. **Enhanced capabilities**: Add more capability types as needed
4. **Performance optimization**: Add caching, connection pooling, etc.
5. **Better error handling**: Provider-specific error handling and retry logic