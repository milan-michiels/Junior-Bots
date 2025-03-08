import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_DIR: str
    CHROMA_HOST: str
    CHROMA_PORT: int
    CHROMA_COLLECTION_NAME: str
    OLLAMA_URI: str
    EMBEDDING_MODEL: str
    RUMMIKUB_RULES_PDF: str
    LANGGRAPH_BASE_URL: str
    GRAPH_ID: str
    MAX_RETRIES: int
    OLLAMA_LIST_MODELS_URI: str

    class Config:
        env_file = (
            ".env.development" if os.getenv("ENVIRONMENT") == "development" else None
        )
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
