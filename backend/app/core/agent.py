"""Main agent orchestrator with multi-step reasoning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator

from .rag import RAGEngine, RetrievedChunk
from .llm import generate_answer, generate_answer_stream_with_metadata
from .conversation import Conversation, get_conversation_manager
from .function_calling import get_tool_registry


@dataclass
class AgentResponse:
    """Response from the agent."""
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    tool_calls: List[dict] = field(default_factory=list)
    tokens_used: dict = field(default_factory=dict)
    processing_time_ms: float = 0.0
    session_id: Optional[str] = None


class TelecomAgent:
    """Main agent orchestrator combining RAG, LLM, and tools."""
    
    def __init__(
        self,
        rag_engine: RAGEngine,
        model: str = "gpt-4o-mini",
        confidence_threshold: float = 0.08,
    ) -> None:
        self.rag = rag_engine
        self.model = model
        self.confidence_threshold = 0.25
        self.conversation_manager = get_conversation_manager()
        self.tool_registry = get_tool_registry()
    
    def _is_greeting(self, message: str) -> bool:
        """Check if message is a simple greeting."""
        lower = message.lower().strip()
        greetings = {
            "hi", "hello", "hey", "salam", "salaam", 
            "hi there", "hello there", "good morning",
            "good afternoon", "good evening", "howdy",
        }
        return lower in greetings or lower.startswith(("hi ", "hello "))
    
    def _get_greeting_response(self) -> str:
        """Get a friendly greeting response."""
        return (
            "Hello! 👋 I'm the AWCC support assistant. "
            "I can help you with:\n\n"
            "• **Data & voice bundles** - recommendations based on your budget\n"
            "• **Account info** - balance, usage history\n"
            "• **Technical support** - SIM activation, APN settings, troubleshooting\n"
            "• **Support tickets** - file complaints or report issues\n\n"
            "How can I assist you today?"
        )
    
    async def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a user message and generate a response.
        
        Args:
            message: User's message
            session_id: Optional session ID for conversation continuity
            
        Returns:
            AgentResponse with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Handle greetings
        if self._is_greeting(message):
            return AgentResponse(
                answer=self._get_greeting_response(),
                processing_time_ms=(time.time() - start_time) * 1000,
                session_id=session_id,
            )
        
        # Get or create conversation
        conversation = self.conversation_manager.get_or_create_session(session_id)
        conversation.add_user_message(message)
        
        # Retrieve relevant context
        sources, best_score = self.rag.retrieve(message, top_k=3)
        
        # Check confidence threshold
        if best_score < self.confidence_threshold and not self._has_tool_intent(message):
            fallback = (
                "I don't have specific information about that in my knowledge base. "
                "Could you rephrase your question or provide more details? "
                "You can also ask me to create a support ticket if you need help from our team."
            )
            conversation.add_assistant_message(fallback)
            return AgentResponse(
                answer=fallback,
                sources=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                session_id=conversation.session_id,
            )
        
        # Generate response with LLM
        try:
            result = generate_answer(
                question=message,
                sources=sources,
                model=self.model,
                conversation=conversation,
                use_tools=True,
            )
            
            answer = result["answer"]
            tool_calls = result.get("tool_calls", [])
            
            # Update conversation
            conversation.add_assistant_message(answer, tool_calls=tool_calls)
            
            return AgentResponse(
                answer=answer,
                sources=sources,
                tool_calls=tool_calls,
                tokens_used=result.get("tokens_used", {}),
                processing_time_ms=(time.time() - start_time) * 1000,
                session_id=conversation.session_id,
            )
            
        except Exception as e:
            fallback = f"I encountered an error processing your request. Please try again."
            return AgentResponse(
                answer=fallback,
                sources=sources,
                processing_time_ms=(time.time() - start_time) * 1000,
                session_id=conversation.session_id,
            )
    
    async def process_message_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process message with streaming response.
        
        Yields events: {type: "start"/"chunk"/"sources"/"done", ...}
        """
        start_time = time.time()
        
        # Handle greetings
        if self._is_greeting(message):
            greeting = self._get_greeting_response()
            yield {"type": "start", "session_id": session_id}
            for char in greeting:
                yield {"type": "chunk", "content": char}
            yield {"type": "done", "full_answer": greeting}
            return
        
        # Get or create conversation
        conversation = self.conversation_manager.get_or_create_session(session_id)
        conversation.add_user_message(message)
        
        # Retrieve context
        sources, best_score = self.rag.retrieve(message, top_k=3)
        
        # Emit sources
        yield {
            "type": "sources",
            "sources": [
                {"source": s.source, "snippet": s.snippet, "score": s.score}
                for s in sources
            ],
        }
        
        # Stream LLM response
        full_answer = ""
        async for event in generate_answer_stream_with_metadata(
            question=message,
            sources=sources,
            model=self.model,
            conversation=conversation,
        ):
            if event["type"] == "chunk":
                full_answer += event["content"]
            yield event
        
        # Update conversation
        conversation.add_assistant_message(full_answer)
    
    def _has_tool_intent(self, message: str) -> bool:
        """Check if message likely requires a tool."""
        lower = message.lower()
        tool_keywords = [
            "ticket", "complaint", "support", "help",
            "bundle", "recommend", "plan", "package",
            "balance", "usage", "history",
            "network", "status", "coverage",
            "create", "check", "show",
        ]
        return any(kw in lower for kw in tool_keywords)
    
    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "rag_stats": self.rag.get_stats(),
            "conversation_stats": self.conversation_manager.get_stats(),
            "available_tools": self.tool_registry.list_tools(),
            "model": self.model,
        }


# Global agent instance
_agent: Optional[TelecomAgent] = None


def get_agent(rag_engine: RAGEngine, model: str = "gpt-4o-mini") -> TelecomAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = TelecomAgent(rag_engine, model)
    return _agent


__all__ = ["TelecomAgent", "AgentResponse", "get_agent"]
