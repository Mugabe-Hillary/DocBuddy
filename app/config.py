import os
from typing import Dict, Any

class Config:
    """Stores application configuration."""
    
    # API Settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Model Settings
    EMBEDDING_MODEL: str = "models/embedding-001"
    LLM_MODEL: str = "gemini-1.5-flash"
    
    # Text Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Vector Store Settings
    VECTOR_STORE_PERSIST_DIR: str = "./chroma_db"
    RETRIEVAL_K: int = 5
    
    # Default Document - REMOVED
    
    # Streamlit Settings
    PAGE_TITLE: str = "DocBuddy"
    PAGE_ICON: str = "📚"
    
    # Response Settings
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1000
    MIN_TEMPERATURE: float = 0.0
    MAX_TEMPERATURE: float = 1.0
    MIN_MAX_TOKENS: int = 100
    MAX_MAX_TOKENS: int = 2000
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Gets the configuration as a dictionary."""
        return {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validates the application configuration."""
        if not cls.GOOGLE_API_KEY:
            return False
        return True
