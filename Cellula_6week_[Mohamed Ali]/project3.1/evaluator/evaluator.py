from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)

from evaluator.eval_prompts import EvaluatorPromptTemplate


class EvaluatorRunnable:

    def __init__(self, llm):

        self.llm = llm

        self.store = {}

        self.evaluator_prompt = (
            EvaluatorPromptTemplate.create_evaluator_prompt()
        )

       
        self.chain = self.evaluator_prompt | self.llm

        # Add independent message history
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )

    def get_by_session_id(
        self,
        session_id: str
    ) -> BaseChatMessageHistory:

        if session_id not in self.store:

            self.store[session_id] = (
                InMemoryChatMessageHistory()
            )

        return self.store[session_id]

    def run(
        self,
        question: str,
        answer: str,
        context: str,
        session_id: str = "default",
    ) -> str:

        result = self.chain_with_history.invoke(
            {
                "question": question,
                "answer": answer,
                "context": context,
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            },
        )

        return result.content