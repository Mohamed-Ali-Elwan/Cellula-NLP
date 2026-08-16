import re


class SQLValidator:

    @staticmethod
    def validate(query):

        query = query.strip()

        query_upper = query.upper()

        forbidden = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "CREATE",
            "ATTACH",
            "DETACH",
            "PRAGMA"
        ]

        for keyword in forbidden:

            if re.search(
                rf"\b{keyword}\b",
                query_upper
            ):

                return False

        if not query_upper.startswith("SELECT"):

            return False

        return True