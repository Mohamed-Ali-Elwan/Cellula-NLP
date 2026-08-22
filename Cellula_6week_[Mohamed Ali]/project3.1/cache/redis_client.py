import os
import redis


class RedisClient:

    def __init__(self):

        self.client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
        )

    def ping(self) -> bool:
        return bool(self.client.ping())