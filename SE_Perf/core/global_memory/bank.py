"""Core components for the GlobalMemory bank."""

from typing import Any

from .embeddings.openai import OpenAIEmbeddingModel
from .memory.base import MemoryBackend
from .memory.chroma import ChromaMemoryBackend
from .utils.config import load_config

EmbeddingModel = Any


class GlobalMemoryBank:
    """Global memory bank for storing and retrieving formatted experiences."""

    def __init__(self, config_path: str | None = None, config: dict[str, Any] | None = None) -> None:
        if config is None:
            if not config_path:
                raise ValueError("config_path is required when config is not provided")
            config = load_config(config_path)
        self.config = config
        self.memory_backend = self._init_memory_backend()
        self.embedding_model = self._init_embedding_model()

    def _init_memory_backend(self) -> MemoryBackend:
        backend_type = (self.config.get("memory") or {}).get("backend", "chroma")
        if backend_type != "chroma":
            raise ValueError(f"Unsupported memory backend: {backend_type}")
        collection_name = (self.config.get("memory") or {}).get("chroma", {}).get("collection_name", "global_memory")
        return ChromaMemoryBackend(collection_name=collection_name)

    def _init_embedding_model(self) -> EmbeddingModel:
        em_cfg = self.config.get("embedding_model") or {}
        provider = em_cfg.get("provider", "openai").lower()
        if provider != "openai":
            raise ValueError(f"Unsupported embedding provider: {provider}")
        api_base = em_cfg.get("api_base") or em_cfg.get("base_url")
        api_key = em_cfg.get("api_key")
        model = em_cfg.get("model")
        if not api_base or not api_key or not model:
            raise ValueError("embedding_model requires api_base, api_key, and model")
        return OpenAIEmbeddingModel(api_base=api_base, api_key=api_key, model=model)

    def add_experience(self, experience: str, metadata: dict[str, Any]) -> None:
        """Adds a formatted experience with metadata into the vector store."""
        embedding = self.embedding_model.encode(experience)
        item = {
            "embedding": embedding,
            "metadata": dict(metadata or {}),
            "document": experience,
        }
        self.memory_backend.add([item])

    def retrieve_memories(self, query: str, k: int = 1) -> list[dict[str, Any]]:
        """Retrieves top-k relevant memories for a textual query."""
        query_embedding = self.embedding_model.encode(query)
        return self.memory_backend.query(query_embedding, k)
