from llm import LLM
from IPython.display import  Markdown

class Explain:
    @staticmethod
    def explain_code(prompt):
        llm1 = LLM.get_llm1()
        response = Markdown(llm1.invoke(prompt).content)
        return response