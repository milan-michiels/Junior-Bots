import os
import uuid

from fastapi import APIRouter
from fastapi import HTTPException
from langgraph_sdk import get_client

from app.core.models import ChatResponse
from app.core.models import QuestionRequest
from app.core.rag import chat

chatbot_router = APIRouter()


@chatbot_router.post("/")
async def handle_incoming_message(request: QuestionRequest) -> ChatResponse:
    """Handle an incoming message"""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    return await chat(request.question, request.thread_id)


@chatbot_router.post("/thread", response_model=uuid.UUID)
async def create_thread() -> uuid.UUID:
    """Create a new thread"""
    client = get_client(url=os.getenv("LANGGRAPH_BASE_URL"))
    thread = await client.threads.create()
    return thread["thread_id"]
