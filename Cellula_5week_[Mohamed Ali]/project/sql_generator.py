from langchain_core.prompts import PromptTemplate
from llm import LLM

class SQLGenerator:

    def __init__(self, llm):

        self.llm = LLM.get_llm1()

        self.prompt = PromptTemplate(
            input_variables=[
                "schema",
                "question"
            ],
            template="""
            You are an expert SQL developer.

            Your task is to convert the user's natural language
            question into a SQLite SQL query.

            DATABASE SCHEMA:
            {schema}

            USER QUESTION:
            {question}

            Rules:

            1. Generate SQLite-compatible SQL.
            2. Use only tables and columns from the schema.
            3. Do not invent columns.
            4. Do not modify the database.
            5. Only generate SELECT queries.
            6. Do not use INSERT, UPDATE, DELETE, DROP,
            ALTER, CREATE, or PRAGMA.
            7. Return ONLY the SQL query.

            SQL:
            """
        )


    def generate(self, question, schema):

        formatted_prompt = self.prompt.format(
            schema=schema,
            question=question
        )

        response = self.llm.invoke(
            formatted_prompt
        )

        return response.content.strip()