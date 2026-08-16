from relevance_checker import RelevanceChecker
from IPython.display import  Markdown
from llm import LLM

class Generate:
   @staticmethod
   def generate_code(prompt):
          llm1 = LLM.get_llm1()
          response = Markdown(llm1.invoke(prompt).content)
          return response