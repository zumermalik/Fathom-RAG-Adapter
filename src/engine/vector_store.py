import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.core.config import settings
import uuid

class NeuralMemory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NeuralMemory, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print(f"🧠 Loading Intelligence Engine ({settings.EMBEDDING_MODEL})...")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.DEVICE)
        
        # Persistent Client ensures data survives restarts
        self.client = chromadb.PersistentClient(path=settings.DATA_PATH)
        self.collection = self.client.get_or_create_collection(
            name="fathom_knowledge"
        )

    def embed(self, text: str):
        """Converts text into a vector thought."""
        return self.model.encode(text).tolist()

    def remember(self, text: str, meta: dict):
        """Stores a memory with metadata."""
        embedding = self.embed(text)
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta]
        )
        return doc_id

    def recall(self, query: str, k: int = 5):
        """Retrieves raw memories based on semantic similarity."""
        embedding = self.embed(query)
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )