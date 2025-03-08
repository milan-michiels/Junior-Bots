import logging
import uuid

from langgraph_sdk import get_client

from app.core.config import settings
from app.core.models import ChatResponse

logger = logging.getLogger(__name__)


async def chat(question: str, thread_id: uuid.UUID) -> ChatResponse:
    """Chat with the RAG model"""
    logger.info(f"Receiving question: {question}")
    client = get_client(url=settings.LANGGRAPH_BASE_URL)
    graph_id = settings.GRAPH_ID
    inputs = {"messages": [question], "max_retries": settings.MAX_RETRIES}
    result = await client.runs.wait(thread_id, graph_id, input=inputs)
    messages = result.get("messages")
    return ChatResponse(answer=messages[-1]["content"], thread_id=thread_id)


async def create_thread() -> uuid.UUID:
    """Create a new thread"""
    logger.info("Creating thread")
    client = get_client(url=settings.LANGGRAPH_BASE_URL)
    thread = await client.threads.create()
    return thread["thread_id"]
