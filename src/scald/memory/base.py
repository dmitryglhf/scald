from abc import ABC, abstractmethod
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


class BaseMemory(ABC):
    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str,
        embedding_function: Any,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"description": self._get_collection_description()},
        )

    @abstractmethod
    def _get_collection_description(self) -> str:
        pass

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": self._get_collection_description()},
        )

    def count(self) -> int:
        return self.collection.count()
