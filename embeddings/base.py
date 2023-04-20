from abc import ABC, abstractmethod
from typing import List


class Embedding(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embedding_docs(self, texts: List[str]) -> List[List[float]]:
        """Embedding search docs."""

    @abstractmethod
    def embedding_query(self, text: str) -> List[float]:
        """Embedding query text."""
