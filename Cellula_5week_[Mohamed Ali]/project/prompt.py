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
  
        
    @staticmethod
    def relevant_prompt():
        return PromptTemplate(
            input_variables=["query", "context"],
            template='''
            You are an expert AI Coding Assistant.

            Your task is to solve the user's programming request using
            the retrieved knowledge provided below.

            Do not reveal your private reasoning or chain-of-thought.
            Instead, reason internally and provide a clear, concise final answer.

            ========================
            USER REQUEST
            ========================

            {query}

            ========================
            RETRIEVED KNOWLEDGE
            ========================

            {context}

            ========================
            INTERNAL REASONING PROCESS
            ========================

            Before producing the final answer, internally:

            1. Understand the user's programming requirements.
            2. Identify the programming language, framework, libraries,
            inputs, outputs, and constraints involved.
            3. Examine the retrieved knowledge and identify the parts
            relevant to the request.
            4. Determine an appropriate implementation strategy.
            5. Check that the planned solution satisfies the user's
            requirements.
            6. Consider possible edge cases and failure scenarios.
            7. Verify that the generated code is logically consistent,
            syntactically correct, and follows good programming practices.
            8. Review the final solution for missing requirements,
            unnecessary complexity, and potential errors.

            Do not output these internal reasoning steps.

            ========================
            CODE GENERATION REQUIREMENTS
            ========================

            The final answer must:

            1. Produce COMPLETE SOURCE CODE.
            Do not provide an incomplete snippet when a complete
            implementation is possible.

            2. Include COMMENTS where they improve readability and
            understanding.

            3. Follow PROGRAMMING BEST PRACTICES, including:
            - Clear structure
            - Meaningful variable and function names
            - Modular and maintainable design
            - Appropriate data structures
            - Readable code

            4. Include appropriate ERROR HANDLING for possible failures
            and invalid inputs.

            5. Use the RETRIEVED KNOWLEDGE when it is relevant to the
            requested solution.

            6. Do not contradict the retrieved knowledge without a
            clear reason.

            7. If the retrieved knowledge does not provide enough
            information for a specific implementation detail, use
            appropriate programming knowledge rather than inventing
            unsupported facts.

            8. If assumptions are required, clearly mention them in
            the final answer.

            9. Make sure the final code is internally consistent and
            syntactically correct.

            10. Provide a concise EXPLANATION of the solution when useful.

            ========================
            FINAL OUTPUT FORMAT
            ========================

            ### Solution

            ```text
            <complete source code>
            ''')