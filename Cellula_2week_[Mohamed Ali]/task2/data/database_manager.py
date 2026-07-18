import os
import pandas as pd


class DatabaseManager:

    def __init__(self, database_path="database.csv"):

        self.database_path = database_path

        self.create_database()

    ##################################################
    # Create CSV Database
    ##################################################

    def create_database(self):

        if not os.path.exists(self.database_path):

            df = pd.DataFrame(
                columns=[
                    "Input",
                    "Prediction"
                ]
            )

            df.to_csv(
                self.database_path,
                index=False
            )

    ##################################################
    # Save New Record
    ##################################################

    def save_record(self, user_input, prediction):

        new_record = pd.DataFrame({

            "Input": [user_input],

            "Prediction": [prediction]

        })

        new_record.to_csv(

            self.database_path,

            mode="a",

            header=False,

            index=False

        )

    ##################################################
    # Read Database
    ##################################################

    def get_all_records(self):

        return pd.read_csv(
            self.database_path
        )

    ##################################################
    # Clear Database (Optional)
    ##################################################

    def clear_database(self):

        df = pd.DataFrame(
            columns=[
                "Input",
                "Prediction"
            ]
        )

        df.to_csv(
            self.database_path,
            index=False
        )