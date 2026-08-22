from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableLambda,
    RunnableSequence,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
import re


class DocumentProcessor:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):

        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

       

        self.chain = RunnableSequence(
            RunnableLambda(self.clean),
            RunnableLambda(self.chunk),
        )

    # =====================================================

    @staticmethod
    def clean(
        documents: list[Document],
    ) -> list[Document]:

        processed_documents = []

        for document in documents:

            text = document.page_content

            # Normalize line endings
            text = text.replace(
                "\r\n",
                "\n",
            )

            # Remove excessive spaces
            text = re.sub(
                r"[ \t]+",
                " ",
                text,
            )

            # Remove excessive blank lines
            text = re.sub(
                r"\n{3,}",
                "\n\n",
                text,
            )

            text = text.strip()

            if text:

                processed_documents.append(
                    Document(
                        page_content=text,
                        metadata=document.metadata.copy(),
                    )
                )

        return processed_documents

   
    def chunk(
        self,
        documents: list[Document],
    ) -> list[Document]:

        return self.chunker.split_documents(
            documents
        )

  

    def process(
        self,
        documents: list[Document],
    ) -> list[Document]:

        return self.chain.invoke(
            documents
        )