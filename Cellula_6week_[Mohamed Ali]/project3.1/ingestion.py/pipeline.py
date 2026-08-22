from loaders import FileLoader
from web_loader import WebLoader

from processing.processor import DocumentProcessor


class IngestionPipeline:

    def __init__(
        self,
        vector_store,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):

        self.vector_store = vector_store

        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # =====================================================
    # Common Pipeline
    # =====================================================

    def _process_and_store(
        self,
        documents,
    ):

        # Processing = LCEL pipeline
        chunks = self.processor.process(
            documents
        )

        # Store processed chunks
        self.vector_store.add_documents(
            chunks
        )

        return len(chunks)

    # =====================================================
    # File
    # =====================================================

    def ingest_file(
        self,
        path: str,
    ):

        documents = FileLoader.load(
            path
        )

        return self._process_and_store(
            documents
        )

    # =====================================================
    # URL
    # =====================================================

    def ingest_url(
        self,
        url: str,
    ):

        documents = WebLoader.load_url(
            url
        )

        return self._process_and_store(
            documents
        )

    # =====================================================
    # Wikipedia
    # =====================================================

    def ingest_wikipedia(
        self,
        topic: str,
    ):

        documents = WebLoader.load_wikipedia(
            topic
        )

        return self._process_and_store(
            documents
        )