from llm import LLM
from rag import RAG, embedding_model, persist_directory
from langchain_core.prompts import PromptTemplate




class RelevanceChecker:

    def __init__(self, llm):
        self.llm = llm

        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
                You are a relevance evaluator for an AI coding assistant.

                Your job is to determine whether the retrieved programming
                knowledge is relevant to the user's request.

                User Request:
                {query}

                Retrieved Knowledge:
                {context}

                Evaluate the retrieved knowledge.

                Return ONLY one of these two labels:

                Relevant
                Not Relevant

                Return "Relevant" if the retrieved knowledge contains
                useful information that can help answer or implement
                the user's request.

                Return "Not Relevant" if the retrieved knowledge is
                unrelated, insufficient, or cannot help with the request.

                Answer:
                """
        )
        
        
  
    def check(self, query, documents):

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        formatted_prompt = self.prompt.format(
            query=query,
            context=context
        )

        response = self.llm.invoke(formatted_prompt)

        result = response.content.strip()

        if "Relevant" in result and "Not Relevant" not in result:
            return "Relevant"

        return "Not Relevant"


