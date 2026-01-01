from fastapi import APIRouter, HTTPException
from src.core.schema import IngestRequest, IntelligentResponse, ReasoningLog
from src.engine.vector_store import NeuralMemory
from src.retrieval.reasoning import AutonomousAgent
from src.core.config import settings

router = APIRouter()

# Initialize Singletons
memory = NeuralMemory()
agent = AutonomousAgent()

@router.post("/ingest")
async def ingest_knowledge(data: IngestRequest):
    """
    Endpoint to teach the system new information.
    """
    doc_id = memory.remember(
        text=data.text, 
        meta={"source": data.source, "tags": ",".join(data.tags)}
    )
    return {"status": "success", "id": doc_id, "message": "Knowledge integrated into neural graph."}

@router.get("/query", response_model=IntelligentResponse)
async def query_intelligence(q: str):
    """
    The core RAG endpoint. Executes the 'Think, Verify, Refine' loop.
    """
    logs = []
    
    # Step 1: Retrieval
    logs.append(ReasoningLog(step=1, action="Retrieval", outcome="Scanning vector space..."))
    results = memory.recall(q, k=5)
    
    # Step 2: Verification (The "Thinking" Phase)
    query_vec = memory.embed(q)
    
    # Chroma returns a list of lists, we need to flatten it carefully
    if results['embeddings'] and results['embeddings'][0]:
        retrieved_vecs = results['embeddings'][0]
        confidence = agent.calculate_semantic_confidence(query_vec, retrieved_vecs)
    else:
        confidence = 0.0
    
    logs.append(ReasoningLog(
        step=2, 
        action="Verification", 
        outcome=f"Calculated Semantic Confidence: {confidence:.2f}"
    ))

    # Step 3: Response Formulation
    sources = results['documents'][0] if results['documents'] else []
    final_answer = agent.generate_autonomous_response(q, confidence, sources)
    
    logs.append(ReasoningLog(
        step=3, 
        action="Decision", 
        outcome="Response generated based on confidence threshold."
    ))

    return IntelligentResponse(
        answer=final_answer,
        confidence_score=confidence,
        reasoning_trace=logs,
        sources=sources
    )