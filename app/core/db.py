import logging

import chromadb

from app.core.config import settings
from app.core.embeddings import create_embeddings

logger = logging.getLogger(__name__)

client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)


def initdb():
    """Initialize the database if it does not exist"""
    if settings.CHROMA_COLLECTION_NAME not in [
        collection.name for collection in client.list_collections()
    ]:
        logger.info(
            f"Creating embeddings for collection: {settings.CHROMA_COLLECTION_NAME}"
        )
        create_embeddings(client)
