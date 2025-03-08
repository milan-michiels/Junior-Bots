import logging
import os

import httpx
from chromadb import ClientAPI
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from unstructured.partition.pdf import partition_pdf

from app.core.config import settings
from app.core.ollama_api import wait_for_model

logger = logging.getLogger(__name__)


def create_embeddings(client: ClientAPI):
    """Create embeddings for the Rummikub rules"""
    try:
        wait_for_model(
            settings.EMBEDDING_MODEL,
            settings.OLLAMA_URI + settings.OLLAMA_LIST_MODELS_URI,
        )
    except TimeoutError as e:
        logger.error(f"Timeout waiting for model: {e}")
        return
    all_splits = partition_pdf(
        filename=os.path.join(settings.DATA_DIR, settings.RUMMIKUB_RULES_PDF),
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1500,
        new_after_n_chars=1200,
        combine_text_under_n_chars=500,
    )
    texts = []
    for element in all_splits:
        if "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    vectorstore = None
    try:
        vectorstore = Chroma(
            client=client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_URI
            ),
        )
        vectorstore.add_texts(texts=texts)
    except httpx.ConnectError as e:
        logging.error(f"Could not connect to Chroma: {e}")
        vectorstore.delete_collection()
