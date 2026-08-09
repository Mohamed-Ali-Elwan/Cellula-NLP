from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.vectorstores import Chroma


class KnowledgeUpdater:

    def __init__(
        self,
        embedding_model,
        persist_directory
    ):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory

    def add_solution(
        self,
        query,
        solution,
        language="python",
        source="user_feedback"
    ):

        document = Document(
            page_content=solution,
            metadata={
                "source": source,
                "language": language,
                "query": query
            }
        )

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

        vectorstore.add_documents(
            documents=[document]
        )

        return document