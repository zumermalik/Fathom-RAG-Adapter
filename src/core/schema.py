from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class IngestRequest(BaseModel):
    text: str
    source: str = "manual_entry"
    tags: List[str] = []

class ReasoningLog(BaseModel):
    step: int
    action: str
    outcome: str

class IntelligentResponse(BaseModel):
    answer: str
    confidence_score: float
    reasoning_trace: List[ReasoningLog]
    sources: List[str]