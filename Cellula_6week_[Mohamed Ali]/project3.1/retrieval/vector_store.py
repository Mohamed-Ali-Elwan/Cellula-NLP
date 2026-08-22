import os

from langchain_classic.vectorstores import Chroma


class VectorStore:

    def __init__(
        self,
        embeddings,
    ):

        self.collection_name = os.getenv(
            "CHROMA_COLLECTION",
            "knowledge_base",
        )

        self.persist_directory = os.getenv(
            "CHROMA_DIR",
            "./data/chroma",
        )

        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(
        self,
        documents,
    ):

        if not documents:
            return

        self.db.add_documents(
            documents
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ):

        return self.db.similarity_search(
            query,
            k=k,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ):

        return (
            self.db
            .similarity_search_with_score(
                query,
                k=k,
            )
        )