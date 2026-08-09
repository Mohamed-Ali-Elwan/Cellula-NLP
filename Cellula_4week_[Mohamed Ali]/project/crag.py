from web_search import WebSearch
from rag import RAG, embedding_model, persist_directory

class Crag:
    

    def __init__(
        self,
        rag_system: RAG,
        score_threshold: float = 0.75
    ):

        self.rag = rag_system
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,

    ):

        local_docs = self.rag.retrieve_with_score(query)

        if not local_docs:
            return {
                "source": "web",
                "documents": WebSearch.search(query)
            }

        best_score = local_docs[0][1]

        if best_score >= self.score_threshold:

            docs = [doc for doc, _ in local_docs]

            return {
                "source": "local",
                "documents": docs
            }

        else:
            return {
                "source": "web",
                "documents": WebSearch.search(query)
            }