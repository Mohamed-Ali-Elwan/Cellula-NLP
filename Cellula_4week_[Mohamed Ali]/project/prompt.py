from langchain_classic.prompts import PromptTemplate , FewShotPromptTemplate , ChatPromptTemplate , MessagesPlaceholder , SystemMessagePromptTemplate , HumanMessagePromptTemplate

temp = PromptTemplate.from_template("""
    User: {input}
    Classification: {output}
    """
   )
examples = [
    {"input": "The topic is about a new technology that can generate electricity from water.",
     "output": "neither"},
    {"input": "Generate a CNN model.  ","output": "Generate"},
    {"input": "explain sorting algorithms","output": "Explain"},
    {"input": "generate a python code to sort a list of numbers","output": "Generate"},
    {"input": "explain this code snippet","output": "Explain"},
    {"input": "how to use machine learning to predict stock prices","output": "neither"},
    {"input": "how to create a REST API with Python","output": "Generate"},
    {"input": "What does this function do? ","output": "Explain"},
    {"input": "Why is this loop incorrect?","output": "Explain"},
    {"input": "Explain line by line.","output": "Explain"},
    
    {"input": "Build a Flask API.","output": "Generate"},
    {"input": "give recipe of pizza","output": "neither"},
    {"input": "explain linear algebra","output": "neither"},
    {"input": "what is the capital of France?","output": "neither"},
    {"input": "explain this math problem","output": "neither"}
]
class Prompt:
    
    @staticmethod
    def few_shot_prompt():
        return FewShotPromptTemplate(
        examples=examples,
        example_prompt=temp,
        prefix="""
        You are an AI intent classifier.

        Classify the user's request into exactly one category.

        Explain:
        The user wants to understand existing code.

        Generate:
        The user wants new code.

        Neither:
        The request is not about programming.

        Return ONLY one of these words:

        Explain
        Generate
        Neither
        """,
            suffix="""
        User: {input}
        Classification:
        """,
    input_variables=["input"],
)

