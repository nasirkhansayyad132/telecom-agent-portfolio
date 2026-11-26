"""Conversation state management with context window handling."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import OrderedDict


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0  # Approximate token count
    tool_calls: Optional[List[dict]] = None
    tool_results: Optional[List[dict]] = None
    
    def to_dict(self) -> dict:
        """Convert to API-compatible format."""
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result


@dataclass
class Conversation:
    """A conversation with message history and context management."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    max_tokens: int = 8000  # Context window limit
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs) -> Message:
        """Add a message to the conversation."""
        # Approximate token count (4 chars ≈ 1 token)
        tokens = len(content) // 4
        
        message = Message(
            role=role,
            content=content,
            tokens=tokens,
            **kwargs
        )
        self.messages.append(message)
        self.last_active = time.time()
        
        # Auto-prune if too long
        self._prune_if_needed()
        
        return message
    
    def add_user_message(self, content: str) -> Message:
        """Add a user message."""
        return self.add_message("user", content)
    
    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[dict]] = None,
    ) -> Message:
        """Add an assistant message."""
        return self.add_message("assistant", content, tool_calls=tool_calls)
    
    def add_system_message(self, content: str) -> Message:
        """Add or update system message (always first)."""
        # System message is always first
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content = content
            self.messages[0].tokens = len(content) // 4
            return self.messages[0]
        else:
            msg = self.add_message("system", content)
            # Move to front
            self.messages.insert(0, self.messages.pop())
            return msg
    
    def get_context(self, include_system: bool = True) -> List[dict]:
        """Get messages formatted for LLM API."""
        messages = []
        for msg in self.messages:
            if not include_system and msg.role == "system":
                continue
            messages.append(msg.to_dict())
        return messages
    
    def get_total_tokens(self) -> int:
        """Get approximate total tokens in conversation."""
        return sum(m.tokens for m in self.messages)
    
    def _prune_if_needed(self) -> None:
        """Remove old messages if context window exceeded."""
        total = self.get_total_tokens()
        
        if total <= self.max_tokens:
            return
        
        # Keep system message and recent messages
        system_msg = None
        if self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]
            other_messages = self.messages[1:]
        else:
            other_messages = self.messages
        
        # Remove oldest messages until under limit
        while other_messages and total > self.max_tokens:
            removed = other_messages.pop(0)
            total -= removed.tokens
        
        if system_msg:
            self.messages = [system_msg] + other_messages
        else:
            self.messages = other_messages
    
    def get_summary(self) -> str:
        """Get a brief summary of the conversation."""
        if not self.messages:
            return "Empty conversation"
        
        user_msgs = [m for m in self.messages if m.role == "user"]
        if user_msgs:
            first_msg = user_msgs[0].content[:100]
            return f"{len(self.messages)} messages, started with: {first_msg}..."
        return f"{len(self.messages)} messages"
    
    def clear(self) -> None:
        """Clear all messages except system message."""
        if self.messages and self.messages[0].role == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []


class ConversationManager:
    """Manages multiple conversation sessions with LRU eviction."""
    
    def __init__(
        self,
        max_conversations: int = 1000,
        session_timeout: int = 3600,  # 1 hour
    ) -> None:
        self.max_conversations = max_conversations
        self.session_timeout = session_timeout
        self._conversations: OrderedDict[str, Conversation] = OrderedDict()
    
    def create_session(self, session_id: Optional[str] = None) -> Conversation:
        """Create a new conversation session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        conversation = Conversation(session_id=session_id)
        self._conversations[session_id] = conversation
        
        # LRU eviction
        while len(self._conversations) > self.max_conversations:
            self._conversations.popitem(last=False)
        
        return conversation
    
    def get_session(self, session_id: str) -> Optional[Conversation]:
        """Get an existing conversation session."""
        conversation = self._conversations.get(session_id)
        
        if conversation is None:
            return None
        
        # Check timeout
        if time.time() - conversation.last_active > self.session_timeout:
            del self._conversations[session_id]
            return None
        
        # Move to end (most recently used)
        self._conversations.move_to_end(session_id)
        conversation.last_active = time.time()
        
        return conversation
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Conversation:
        """Get existing session or create new one."""
        if session_id:
            existing = self.get_session(session_id)
            if existing:
                return existing
        
        return self.create_session(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if session_id in self._conversations:
            del self._conversations[session_id]
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        now = time.time()
        expired = [
            sid for sid, conv in self._conversations.items()
            if now - conv.last_active > self.session_timeout
        ]
        for sid in expired:
            del self._conversations[sid]
        return len(expired)
    
    def get_stats(self) -> dict:
        """Get manager statistics."""
        return {
            "active_sessions": len(self._conversations),
            "max_sessions": self.max_conversations,
            "session_timeout": self.session_timeout,
        }


# Global conversation manager instance
_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager."""
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager


__all__ = [
    "Message",
    "Conversation", 
    "ConversationManager",
    "get_conversation_manager",
]
