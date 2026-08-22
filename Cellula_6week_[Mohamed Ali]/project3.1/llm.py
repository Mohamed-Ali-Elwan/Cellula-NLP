import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


class LLM:

    @staticmethod
    def get_gen_llm():

        return ChatOpenAI(
            api_key=os.getenv("GEN_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv(
                "GENERATOR_MODEL",
                "openrouter/free"
            ),
            temperature=0.8,
        )

    @staticmethod
    def get_eval_llm():

        return ChatOpenAI(
            api_key=os.getenv("EVAL_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv(
                "EVALUATOR_MODEL",
                "openrouter/free"
            ),
            temperature=0.8,
        )