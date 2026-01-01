import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Fathom RAG Adapter"
    VERSION: str = "0.1.0"
    
    # Persistence
    DATA_PATH: str = "./data/fathom_store"
    
    # Intelligence Config
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    DEVICE: str = "cpu" # Change to 'cuda' if torch detects GPU
    
    # Autonomous Reasoning Thresholds
    CONFIDENCE_THRESHOLD: float = 0.45
    MAX_REASONING_STEPS: int = 3

settings = Settings()

# Ensure storage exists
os.makedirs(settings.DATA_PATH, exist_ok=True)