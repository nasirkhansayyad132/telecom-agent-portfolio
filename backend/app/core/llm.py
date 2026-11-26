"""Advanced LLM integration with streaming, function calling, and retry logic."""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, List, Optional, Dict, Any

from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .rag import RetrievedChunk
from .conversation import Conversation, Message
from .function_calling import get_tool_registry, ToolRegistry


# Clients
_sync_client: Optional[OpenAI] = None
_async_client: Optional[AsyncOpenAI] = None


def _get_sync_client() -> OpenAI:
    """Get synchronous OpenAI client."""
    global _sync_client
    if _sync_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _sync_client = OpenAI(api_key=api_key)
    return _sync_client


def _get_async_client() -> AsyncOpenAI:
    """Get asynchronous OpenAI client."""
    global _async_client
    if _async_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _async_client = AsyncOpenAI(api_key=api_key)
    return _async_client


SYSTEM_PROMPT = """You are a helpful telecom support assistant for AWCC (Afghan Wireless Communication Company).

Your capabilities:
1. Answer questions about telecom services using the provided knowledge base
2. Help customers with bundles, data plans, voice packages, and SIM issues
3. Create support tickets for customer complaints
4. Recommend appropriate bundles based on budget and usage needs
5. Check account balance, usage history, and network status

CRITICAL INSTRUCTION:
You are a SPECIALIZED agent. You must ONLY answer questions related to:
- AWCC services and products (bundles, SIMs, etc.)
- Mobile network issues (APN, coverage, 4G/5G)
- Account management (balance, top-up)
- General telecom support

**If the user asks about ANYTHING else (e.g. geography, history, math, coding, general life advice), you MUST REFUSE.**

REFUSAL EXAMPLES:
User: "What is the capital of France?"
Assistant: "I apologize, but I can only assist with AWCC telecom services and support."

User: "Who is the president of USA?"
Assistant: "I am an AWCC support agent and cannot answer general knowledge questions. How can I help with your mobile service?"

Guidelines:
- **Strictly refuse** non-telecom questions even if you know the answer.
- Be concise and direct.
- Always cite sources when answering from the knowledge base.
- Format prices in AFN (Afghani)."""


def _format_context(sources: List[RetrievedChunk]) -> str:
    """Format retrieved sources for the prompt."""
    if not sources:
        return "No relevant knowledge base articles found. CAUTION: Since no telecom documents matched, if the user is asking a non-telecom question (like general knowledge), you MUST REFUSE to answer."
    
    lines = []
    for idx, item in enumerate(sources, 1):
        lines.append(f"[{idx}] Source: {item.source}")
        lines.append(f"    Content: {item.snippet}")
        lines.append("")
    return "\n".join(lines)


def _build_messages(
    question: str,
    sources: List[RetrievedChunk],
    conversation: Optional[Conversation] = None,
) -> List[dict]:
    """Build message list for the API call."""
    messages = []
    
    # System message
    context = _format_context(sources)
    system_content = f"{SYSTEM_PROMPT}\n\n---\nKnowledge Base Context:\n{context}"
    messages.append({"role": "system", "content": system_content})
    
    # Conversation history (if any)
    if conversation:
        for msg in conversation.messages:
            if msg.role != "system":  # Skip system messages from history
                messages.append(msg.to_dict())
    
    # Current question
    final_content = question
    if not sources:
        final_content += "\n\n[SYSTEM NOTE: No RAG match found. If this is a general knowledge question (e.g. geography, history), you MUST REFUSE to answer. Only answer if it is a greeting or related to AWCC.]"
        
    messages.append({"role": "user", "content": final_content})
    
    return messages


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def generate_answer(
    question: str,
    sources: List[RetrievedChunk],
    model: str = "gpt-4o-mini",
    conversation: Optional[Conversation] = None,
    use_tools: bool = True,
) -> Dict[str, Any]:
    """Generate an answer using OpenAI with optional function calling.
    
    Args:
        question: User's question
        sources: Retrieved knowledge base chunks
        model: OpenAI model to use
        conversation: Optional conversation for context
        use_tools: Whether to enable function calling
        
    Returns:
        Dict with 'answer', 'tool_calls', and 'tokens_used'
    """
    client = _get_sync_client()
    messages = _build_messages(question, sources, conversation)
    
    # Build request
    request_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.3,
    }
    
    # Add tools if enabled
    if use_tools:
        registry = get_tool_registry()
        tools = registry.get_openai_tools()
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
    
    response = client.chat.completions.create(**request_kwargs)
    
    choice = response.choices[0]
    message = choice.message
    
    result = {
        "answer": message.content or "",
        "tool_calls": [],
        "tokens_used": {
            "prompt": response.usage.prompt_tokens if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
        },
        "finish_reason": choice.finish_reason,
    }
    
    # Handle tool calls
    if message.tool_calls:
        registry = get_tool_registry()
        for tool_call in message.tool_calls:
            func = tool_call.function
            try:
                args = json.loads(func.arguments)
            except json.JSONDecodeError:
                args = {}
            
            # Execute the tool
            tool_result = registry.execute(func.name, args)
            
            result["tool_calls"].append({
                "id": tool_call.id,
                "name": func.name,
                "arguments": args,
                "result": tool_result,
            })
        
        # If tools were called, generate a follow-up response with results
        if result["tool_calls"]:
            result = _generate_with_tool_results(
                client, messages, message, result["tool_calls"], model
            )
    
    return result


def _generate_with_tool_results(
    client: OpenAI,
    messages: List[dict],
    assistant_message,
    tool_calls: List[dict],
    model: str,
) -> Dict[str, Any]:
    """Generate response after tool execution."""
    # Add assistant's tool call message
    messages.append({
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            }
            for tc in tool_calls
        ],
    })
    
    # Add tool results
    for tc in tool_calls:
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps(tc["result"]),
        })
    
    # Generate final response
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=800,
        temperature=0.3,
    )
    
    return {
        "answer": response.choices[0].message.content or "",
        "tool_calls": tool_calls,
        "tokens_used": {
            "prompt": response.usage.prompt_tokens if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
        },
        "finish_reason": response.choices[0].finish_reason,
    }


async def generate_answer_stream(
    question: str,
    sources: List[RetrievedChunk],
    model: str = "gpt-4o-mini",
    conversation: Optional[Conversation] = None,
) -> AsyncIterator[str]:
    """Generate a streaming answer.
    
    Yields text chunks as they arrive from the API.
    Note: Streaming doesn't support tool calls in this implementation.
    """
    client = _get_async_client()
    messages = _build_messages(question, sources, conversation)
    
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800,
            temperature=0.3,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"\n\n[Error: {str(e)}]"


async def generate_answer_stream_with_metadata(
    question: str,
    sources: List[RetrievedChunk],
    model: str = "gpt-4o-mini",
    conversation: Optional[Conversation] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Generate a streaming answer with metadata events.
    
    Yields dicts with 'type' (start, chunk, tool_call, done, error) and data.
    """
    # Emit start event
    yield {"type": "start", "model": model, "sources_count": len(sources)}
    
    full_answer = ""
    
    async for chunk in generate_answer_stream(question, sources, model, conversation):
        full_answer += chunk
        yield {"type": "chunk", "content": chunk}
    
    # Emit done event
    yield {
        "type": "done",
        "full_answer": full_answer,
        "tokens_estimate": len(full_answer) // 4,
    }


# Legacy function for backward compatibility
def generate_answer_simple(
    question: str,
    sources: List[RetrievedChunk],
    model: str = "gpt-4o-mini",
) -> str:
    """Simple answer generation (backward compatible)."""
    try:
        result = generate_answer(question, sources, model, use_tools=False)
        return result["answer"]
    except Exception:
        return "I don't know from my KB yet. Could you share more details?"


__all__ = [
    "generate_answer",
    "generate_answer_stream",
    "generate_answer_stream_with_metadata",
    "generate_answer_simple",
]
