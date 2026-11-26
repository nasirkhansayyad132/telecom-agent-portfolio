"""Core modules for the telecom agent."""

from .rag import RAGEngine, RetrievedChunk
from .llm import generate_answer, generate_answer_stream, generate_answer_simple
from .conversation import Conversation, ConversationManager, get_conversation_manager
from .function_calling import ToolRegistry, get_tool_registry
from .agent import TelecomAgent, AgentResponse, get_agent
from .ppt import build_ppt, ConversionError
from .tools import create_ticket, recommend_bundle, detect_intent

__all__ = [
    # RAG
    "RAGEngine",
    "RetrievedChunk",
    # LLM
    "generate_answer",
    "generate_answer_stream",
    "generate_answer_simple",
    # Conversation
    "Conversation",
    "ConversationManager",
    "get_conversation_manager",
    # Function Calling
    "ToolRegistry",
    "get_tool_registry",
    # Agent
    "TelecomAgent",
    "AgentResponse",
    "get_agent",
    # PPT
    "build_ppt",
    "ConversionError",
    # Tools
    "create_ticket",
    "recommend_bundle",
    "detect_intent",
]
