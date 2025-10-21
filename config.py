"""
Configuration management for Traffix system
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings

# Import unified config for backward compatibility
try:
    from unified_config import UnifiedConfig, get_config
except ImportError:
    # Create a simple fallback config
    class UnifiedConfig:
        def __init__(self):
            self.openai_api_key = ""
            self.ritis_api_key = ""
            self.news_api_key = ""
            self.weather_api_key = ""
            self.openai_model = "gpt-4o"
            self.embedding_model = "text-embedding-3-small"
            self.temperature = 0.1
            self.max_tokens = 4000
            self.qdrant_host = "localhost"
            self.qdrant_port = 6333
            self.qdrant_collection_name = "traffix_embeddings"
            self.qdrant_api_key = None
            self.qdrant_url = None
            self.chunk_size = 600
            self.chunk_overlap = 100
            self.ritis_base_url = "https://api.ritis.org"
            self.news_base_url = "https://newsapi.org/v2"
            self.weather_base_url = "https://api.openweathermap.org/data/2.5"
            self.max_retries = 3
            self.timeout_seconds = 30
            self.batch_size = 100
            self.log_level = "INFO"
            self.log_file = None
            self.max_concurrent_requests = 10
            self.cache_ttl_seconds = 3600
            self.enable_quick_mode = True
            self.enable_deep_mode = True
            self.enable_anomaly_investigation = True
            self.enable_leadership_summary = True
            self.enable_ragas_evaluation = True
            self.min_quality_score = 0.7
            self.export_formats = ["json", "pdf", "html"]
            self.export_directory = "./exports"
            self.embedding_dimension = 1536
        
        def get_database_url(self):
            if self.qdrant_url:
                return self.qdrant_url
            if self.qdrant_api_key:
                return f"https://{self.qdrant_host}:{self.qdrant_port}"
            else:
                return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    def get_config():
        return UnifiedConfig()

# Create unified config instance
unified_config = get_config()


class Settings(BaseSettings):
    """Legacy application settings - use UnifiedConfig instead"""
    
    # OpenAI Configuration
    openai_api_key: str = unified_config.openai_api_key
    openai_model: str = unified_config.openai_model
    
    # Database Configuration
    database_url: str = "sqlite:///./traffix.db"
    
    # RITIS API Configuration
    ritis_api_key: str = unified_config.ritis_api_key or ""
    ritis_base_url: str = unified_config.ritis_base_url
    
    # News API Configuration
    news_api_key: str = unified_config.news_api_key or ""
    
    # Tavily API Configuration
    tavily_api_key: str = ""
    
    # Cohere API Configuration
    cohere_api_key: str = ""
    
    # Vector Database Configuration (Local)
    qdrant_url: str = unified_config.get_database_url()
    qdrant_api_key: Optional[str] = unified_config.qdrant_api_key
    qdrant_collection_name: str = unified_config.qdrant_collection_name
    qdrant_storage_path: str = "./data/qdrant_db"
    
    # LangSmith Monitoring
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "traffix"
    langsmith_tracing: bool = True
    
    # RAGAS Evaluation
    ragas_api_key: Optional[str] = None
    enable_evaluation: bool = unified_config.enable_ragas_evaluation
    
    # System Configuration
    log_level: str = unified_config.log_level
    max_concurrent_agents: int = unified_config.max_concurrent_requests
    report_output_dir: str = unified_config.export_directory
    data_directory: str = "./data"
    
    # Agent Configuration
    quick_mode_max_sources: int = 10
    deep_mode_max_sources: int = 50
    analysis_timeout_minutes: int = unified_config.timeout_seconds // 60
    
    # Streamlit Configuration
    streamlit_port: int = 8501
    streamlit_host: str = "localhost"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance (legacy)
settings = Settings()

# Export unified config as the preferred way
config = unified_config
