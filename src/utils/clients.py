"""Centralized client initialization for GenAI and database connections."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AzureOpenAI
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()

DEFAULT_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")


@lru_cache(maxsize=1)
def get_genai_client() -> AzureOpenAI:
    """Get cached Azure OpenAI chat client.

    Returns:
        Configured AzureOpenAI client

    Raises:
        RuntimeError: If required environment variables are missing
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")

    if not endpoint or not api_key:
        raise RuntimeError(
            "Missing Azure OpenAI credentials. Set AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_KEY in .env file"
        )

    return AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version=DEFAULT_API_VERSION
    )


@lru_cache(maxsize=1)
def get_db_engine(db_path: str = "data/raw_data/noshow.db") -> Engine:
    """Get cached SQLAlchemy database engine.

    Args:
        db_path: Path to SQLite database

    Returns:
        SQLAlchemy Engine instance
    """
    return create_engine(f"sqlite:///{db_path}")
