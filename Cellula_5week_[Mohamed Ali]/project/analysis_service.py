from database import SQLiteDatabase
from sql_generator import SQLGenerator
from sql_validator import SQLValidator


class AnalysisService:

    def __init__(
        self,
        database,
        sql_generator,
        sql_validator
    ):

        self.database = database
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator


    def analyze(
        self,
        question
    ):

        columns = self.database.get_schema()

        schema = "\n".join(
            f"{column[1]} {column[2]}"
            for column in columns
        )

        sql = self.sql_generator.generate(
            question=question,
            schema=schema
        )

        if not self.sql_validator.validate(sql):

            raise ValueError(
                "Generated SQL is not allowed."
            )

        result = self.database.execute_query(
            sql
        )

        return {
            "question": question,
            "sql": sql,
            "result": result
        }