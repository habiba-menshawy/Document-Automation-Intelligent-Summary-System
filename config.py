"""
Configuration for the LangChain-based document analysis agent.
"""
from dataclasses import dataclass
from typing import  List


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM."""
    model: str = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
    embedding_model: str = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
    temperature: float = 0.0
    base_url: str = "http://localhost:11434"


@dataclass
class DatabaseConfig:
    """Configuration for databases."""
    sqlite_path: str = "documents.db"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "documents"


class DocumentTypes:
    """Supported document types."""
    EMAIL = "Email"
    REPORT = "Report"
    SCIENTIFIC = "Scientific"
    
    @classmethod
    def all_types(cls) -> List[str]:
        return [cls.EMAIL, cls.REPORT, cls.SCIENTIFIC]



# Agent configuration
AGENT_MAX_ITERATIONS = 15
AGENT_EARLY_STOPPING_METHOD = "generate"
AGENT_VERBOSE = True


# Entity extraction configuration
SPACY_MODEL = "en_core_web_sm"

#Milvus
MILVUS_COLLECTION = "documents"

#SQLite
SQLITE_DB_PATH = "documents.db"