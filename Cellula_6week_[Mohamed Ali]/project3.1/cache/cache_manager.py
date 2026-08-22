import hashlib
import json


class CacheManager:

    def __init__(self, redis_client, ttl: int = 3600):

        self.redis = redis_client
        self.ttl = ttl

    @staticmethod
    def _make_key(prefix: str, value: str) -> str:

        hashed_value = hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

        return f"{prefix}:{hashed_value}"

    

    def get_retrieval(self, query: str):

        key = self._make_key(
            "retrieval",
            query
        )

        return self.redis.get(key)

    def set_retrieval(
        self,
        query: str,
        context: str
    ):

        key = self._make_key(
            "retrieval",
            query
        )

        self.redis.setex(
            key,
            self.ttl,
            context
        )

    

    def get_response(self, query: str):

        key = self._make_key(
            "response",
            query
        )

        value = self.redis.get(key)

        if value is None:
            return None

        return json.loads(value)

    def set_response(
        self,
        query: str,
        result: dict
    ):

        key = self._make_key(
            "response",
            query
        )

        self.redis.setex(
            key,
            self.ttl,
            json.dumps(result)
        )

   

    def delete(self, prefix: str, value: str):

        key = self._make_key(
            prefix,
            value
        )

        self.redis.delete(key)

    
    def clear_all(self):

        self.redis.flushdb()