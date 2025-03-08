from fastapi import APIRouter

from app.api.routes.chatbot import chatbot_router
from app.api.routes.healthcheck import health_router

api_router = APIRouter()
api_router.include_router(chatbot_router, tags=["chatbot"], prefix="/chatbot")
api_router.include_router(health_router, tags=["health"], prefix="/health")
