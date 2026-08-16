from langchain_openai import ChatOpenAI



class LLM:
    @staticmethod
    def get_llm1():
        return ChatOpenAI(api_key="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",
                          base_url="https://openrouter.ai/api/v1",
                          model_name="openrouter/free",
                          temperature=0.8)
