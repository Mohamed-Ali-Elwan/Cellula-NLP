from langchain_classic.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.vectorstores import Chroma


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

persist_directory = "./chroma_db"


class RAG:

    def __init__(
        self,
        documents,
        embedding_model,
        persist_directory
    ):

        self.documents = documents
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory


    def create_vectorstore(self):

        text_splitter = (
            RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=1000,
                chunk_overlap=200
            )
        )

        chunks = text_splitter.split_documents(
            self.documents
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )

        return vectorstore


    def retrieve(self, query, k=5):

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

        results = vectorstore.similarity_search(
            query,
            k=k
        )

        return results


    def retrieve_with_score(self, query, k=5):

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

        results = vectorstore.similarity_search_with_score(
            query,
            k=k
        )

        return results