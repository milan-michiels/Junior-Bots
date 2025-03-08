import uuid

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Request model for the chat endpoint"""

    question: str
    thread_id: uuid.UUID


class ChatResponse(BaseModel):
    """Response model for the chat endpoint"""

    answer: str
    thread_id: uuid.UUID
