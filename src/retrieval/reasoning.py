import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from src.core.config import settings

class AutonomousAgent:
    """
    The 'Brain' of Fathom. 
    Performs multi-step reasoning to validate retrieval quality.
    """
    
    def calculate_semantic_confidence(self, query_vec, retrieved_vecs) -> float:
        """
        Calculates a confidence score based on how close the retrieved memories
        are to the original thought (vector).
        """
        if not retrieved_vecs:
            return 0.0
        
        # Calculate cosine similarity between query and all results
        # Returns a matrix, we take the mean of the top results
        sims = cosine_similarity([query_vec], retrieved_vecs)[0]
        return float(np.mean(sims))

    def generate_autonomous_response(self, query: str, confidence: float, sources: List[str]) -> str:
        """
        Decides how to answer based on confidence.
        """
        if not sources:
            return "I searched my memory but found no relevant information."

        if confidence >= settings.CONFIDENCE_THRESHOLD:
            return f"I am confident in this answer based on {len(sources)} sources."
        else:
            return "I found some information, but it may not perfectly match your question. Please verify the context below."