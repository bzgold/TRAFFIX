"""
Technology stack configuration for Traffix
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class TechStackSettings(BaseSettings):
    """Technology stack configuration"""
    
    # LLM Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    
    # Tavily API Configuration
    tavily_api_key: str = ""
    
    # Cohere API Configuration
    cohere_api_key: str = ""
    
    # LangGraph Configuration
    langgraph_api_key: Optional[str] = None
    langgraph_endpoint: str = "https://api.langgraph.com"
    
    # Qdrant Vector Database
    qdrant_url: str = "./data/qdrant_db"  # Local storage path
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "traffix_embeddings"
    
    # LangSmith Monitoring
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "traffix"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_tracing: bool = True
    
    # RAGAS Evaluation
    ragas_api_key: Optional[str] = None
    ragas_endpoint: str = "https://api.ragas.com"
    
    # Streamlit Configuration
    streamlit_port: int = 8501
    streamlit_host: str = "localhost"
    
    # Vector Database Settings
    embedding_dimension: int = 1536  # text-embedding-3-small dimension
    vector_distance_metric: str = "cosine"
    max_retries: int = 3
    qdrant_storage_path: str = "./data/qdrant_db"
    
    # RAG Pipeline Settings
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    
    # Monitoring Settings
    enable_tracing: bool = True
    enable_evaluation: bool = True
    log_level: str = "INFO"
    report_output_dir: str = "./data/reports"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global tech stack settings
tech_settings = TechStackSettings()
