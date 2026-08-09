from llm import LLM
from IPython.display import  Markdown
from app import chat

class Explain:
    @staticmethod
    def explain_code(prompt):
        
        response = Markdown(chat.predict(prompt))
        return response