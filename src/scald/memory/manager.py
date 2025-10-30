from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import JinaEmbeddingFunction

from scald.common.types import ActorSolution, CriticEvaluation
from scald.memory.types import ActorMemoryContext, CriticMemoryContext, MemoryEntry


class MemoryManager:
    """Long-term memory management via ChromaDB with Jina embeddings"""

    COLLECTION_NAME = "scald_memory"
    MEMORY_DIR = Path.home() / ".scald" / "chromadb"

    def __init__(self):
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_fn: Optional[JinaEmbeddingFunction] = None
        raise NotImplementedError

    async def retrieve_relevant_context(
        self, actor_report: str, task_type: str, top_k: int = 5
    ) -> tuple[list[ActorMemoryContext], list[CriticMemoryContext]]:
        """Retrieve top-K relevant memory entries by similarity using actor's report"""
        raise NotImplementedError

    async def save_iteration(
        self,
        actor_solution: ActorSolution,
        critic_evaluation: CriticEvaluation,
        task_type: str,
        iteration: int,
    ) -> str:
        """Save single iteration result to memory"""
        raise NotImplementedError

    def _create_embedding_function(self) -> JinaEmbeddingFunction:
        """Create Jina embedding function from env"""
        raise NotImplementedError

    def _serialize_entry(self, entry: MemoryEntry) -> dict:
        """Serialize MemoryEntry to flat dict for ChromaDB metadata"""
        raise NotImplementedError

    def _deserialize_entry(self, entry_id: str, document: str, metadata: dict) -> MemoryEntry:
        """Deserialize MemoryEntry from ChromaDB results"""
        raise NotImplementedError
