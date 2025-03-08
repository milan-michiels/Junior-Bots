import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.main import api_router
from app.core.db import initdb


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Initializing chroma")
    initdb()
    yield


logging.basicConfig(level=logging.INFO)
app = FastAPI(lifespan=lifespan)
app.include_router(api_router, prefix="/api")
