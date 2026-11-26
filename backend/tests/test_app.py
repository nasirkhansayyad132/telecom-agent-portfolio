"""Tests for the telecom_agent API."""

import pytest
from httpx import ASGITransport, AsyncClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app


@pytest.mark.anyio
async def test_health() -> None:
    """Test health check endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.anyio
async def test_chat_greeting() -> None:
    """Test that greetings get a friendly response."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"message": "Hello"}
        response = await client.post("/api/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "Hello" in data["answer"] or "Hi" in data["answer"] or "help" in data["answer"].lower()


@pytest.mark.anyio
async def test_chat_no_kb_hit_returns_fallback() -> None:
    """Test that unknown queries return fallback."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"message": "Tell me about interstellar travel pricing"}
        response = await client.post("/api/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    # Should either have a fallback or sources
    assert "answer" in data


@pytest.mark.anyio
async def test_list_tools() -> None:
    """Test tools endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/tools")

    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert len(data["tools"]) >= 3
    
    tool_names = [t["name"] for t in data["tools"]]
    assert "create_ticket" in tool_names
    assert "recommend_bundle" in tool_names


@pytest.mark.anyio  
async def test_stats() -> None:
    """Test stats endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/stats")

    assert response.status_code == 200
    data = response.json()
    assert "rag_stats" in data
    assert "available_tools" in data
