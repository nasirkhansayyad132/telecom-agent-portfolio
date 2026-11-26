"""FastAPI app for the advanced telecom_agent project."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .core.llm import generate_answer
from .core.ppt import ConversionError, build_ppt
from .core.rag import RAGEngine, RetrievedChunk
from .core.agent import TelecomAgent, get_agent
from .core.conversation import get_conversation_manager


class Settings(BaseModel):
    """Application settings from environment."""
    openai_model: str = Field(default="gpt-4o-mini")
    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:5174,http://localhost:3000,"
        "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:3000"
    )
    confidence_threshold: float = Field(default=0.08)
    enable_streaming: bool = Field(default=True)

    @property
    def origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
    
    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            allowed_origins=os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:5173,http://localhost:5174,http://localhost:3000,"
                "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:3000",
            ),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.08")),
            enable_streaming=os.getenv("ENABLE_STREAMING", "true").lower() == "true",
        )


load_dotenv()
settings = Settings.from_env()

app = FastAPI(
    title="Telecom Agent API",
    description="Advanced AI-powered telecom support agent with RAG, streaming, and tool calling",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "faqs"
rag_engine = RAGEngine(DATA_DIR)

# Global agent (initialized on startup)
agent: Optional[TelecomAgent] = None


# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    stream: bool = Field(False, description="Enable streaming response")


class SourceItem(BaseModel):
    source: str
    snippet: str
    score: float = 0.0
    retrieval_method: str = "hybrid"


class ToolUsage(BaseModel):
    name: str
    input: dict
    output: dict


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    tool_used: Optional[ToolUsage] = None
    session_id: Optional[str] = None
    processing_time_ms: float = 0.0
    tokens_used: Optional[dict] = None


class StatsResponse(BaseModel):
    rag_stats: dict
    conversation_stats: dict
    available_tools: List[str]
    model: str


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the application on startup."""
    global agent
    rag_engine.load()
    agent = get_agent(rag_engine, model=settings.openai_model)
    print(f"✅ Telecom Agent started with model: {settings.openai_model}")
    print(f"📚 Loaded {rag_engine.get_stats()['total_chunks']} knowledge chunks")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    conversation_manager = get_conversation_manager()
    conversation_manager.cleanup_expired()


# Health & Stats Endpoints
@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "model": settings.openai_model,
        "streaming_enabled": settings.enable_streaming,
    }


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get agent statistics."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    stats = agent.get_stats()
    return StatsResponse(**stats)


# Knowledge Base Endpoints
@app.post("/api/reload-kb")
async def reload_kb() -> dict:
    """Reload knowledge base without restart."""
    rag_engine.load(force_rebuild=True)
    stats = rag_engine.get_stats()
    return {
        "status": "reloaded",
        "chunks": stats["total_chunks"],
        "sources": stats["total_sources"],
        "semantic_enabled": stats["semantic_enabled"],
    }


# Chat Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """Process a chat message and return response."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Process message
    response = await agent.process_message(
        message=message,
        session_id=payload.session_id,
    )
    
    # Format sources
    sources = [
        SourceItem(
            source=s.source,
            snippet=s.snippet,
            score=s.score,
            retrieval_method=s.retrieval_method,
        )
        for s in response.sources
    ]
    
    # Format tool usage (first tool call if any)
    tool_used = None
    if response.tool_calls:
        tc = response.tool_calls[0]
        tool_used = ToolUsage(
            name=tc["name"],
            input=tc["arguments"],
            output=tc.get("result", {}),
        )
    
    return ChatResponse(
        answer=response.answer,
        sources=sources,
        tool_used=tool_used,
        session_id=response.session_id,
        processing_time_ms=response.processing_time_ms,
        tokens_used=response.tokens_used or None,
    )


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest):
    """Stream chat response using Server-Sent Events (SSE)."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not settings.enable_streaming:
        raise HTTPException(status_code=400, detail="Streaming is disabled")
    
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    async def event_generator():
        """Generate SSE events."""
        try:
            async for event in agent.process_message_stream(
                message=message,
                session_id=payload.session_id,
            ):
                # Format as SSE
                event_data = json.dumps(event)
                yield f"data: {event_data}\n\n"
        except Exception as e:
            error_event = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_event}\n\n"
        finally:
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# WebSocket Chat
@app.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    
    if agent is None:
        await websocket.send_json({"type": "error", "message": "Agent not initialized"})
        await websocket.close()
        return
    
    # Send connection confirmation
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "model": settings.openai_model,
    })
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "").strip()
            
            if not message:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue
            
            # Stream response
            async for event in agent.process_message_stream(
                message=message,
                session_id=session_id,
            ):
                await websocket.send_json(event)
            
            # Send completion
            await websocket.send_json({"type": "complete"})
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


# Conversation Management
@app.delete("/api/conversation/{session_id}")
async def delete_conversation(session_id: str) -> dict:
    """Delete a conversation session."""
    conversation_manager = get_conversation_manager()
    deleted = conversation_manager.delete_session(session_id)
    return {"deleted": deleted, "session_id": session_id}


@app.get("/api/conversation/{session_id}")
async def get_conversation(session_id: str) -> dict:
    """Get conversation history."""
    conversation_manager = get_conversation_manager()
    conversation = conversation_manager.get_session(session_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "message_count": len(conversation.messages),
        "messages": [
            {"role": m.role, "content": m.content[:200]}
            for m in conversation.messages
        ],
        "created_at": conversation.created_at,
        "last_active": conversation.last_active,
    }


# Excel-to-PPT Conversion
@app.post("/api/convert/excel-to-ppt")
async def convert_excel_to_ppt(file: UploadFile = File(...)) -> Response:
    """Convert Excel/CSV file to PowerPoint presentation."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    content = await file.read()
    try:
        ppt_buffer = build_ppt(file.filename, content)
    except ConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to create presentation.") from exc

    headers = {"Content-Disposition": 'attachment; filename="report.pptx"'}
    return Response(
        content=ppt_buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers=headers,
    )


# Tools Information
@app.get("/api/tools")
async def list_tools() -> dict:
    """List available agent tools."""
    from .core.function_calling import get_tool_registry
    registry = get_tool_registry()
    
    tools = []
    for tool_name in registry.list_tools():
        tool = registry.get(tool_name)
        if tool:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
    
    return {"tools": tools}


__all__ = ["app"]
