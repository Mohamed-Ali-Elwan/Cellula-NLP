import sqlite3
import pandas as pd


class SQLiteDatabase:

    def __init__(self, db_path="data.db"):

        self.db_path = db_path


    def load_dataframe(
        self,
        dataframe,
        table_name="dataset"
    ):

        connection = sqlite3.connect(
            self.db_path
        )

        dataframe.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False
        )

        connection.close()


    def get_schema(
        self,
        table_name="dataset"
    ):

        connection = sqlite3.connect(
            self.db_path
        )

        cursor = connection.cursor()

        cursor.execute(
            f"PRAGMA table_info({table_name})"
        )

        columns = cursor.fetchall()

        connection.close()

        return columns


    def execute_query(self, query):

        connection = sqlite3.connect(
            self.db_path
        )

        cursor = connection.cursor()

        cursor.execute(query)

        results = cursor.fetchall()

        columns = [
            description[0]
            for description in cursor.description
        ]

        connection.close()

        return {
            "columns": columns,
            "rows": results
        }