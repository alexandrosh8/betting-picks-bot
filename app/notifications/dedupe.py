"""Idempotency stores for duplicate-alert prevention."""

from typing import Protocol

from redis.asyncio import Redis

DEFAULT_TTL_SECONDS = 24 * 60 * 60


class IdempotencyStore(Protocol):
    async def claim(self, key: str) -> bool:
        """True if `key` was claimed now (first occurrence), False if seen."""
        ...


class InMemoryIdempotencyStore:
    """Test/dev store; process-local."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    async def claim(self, key: str) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


class RedisIdempotencyStore:
    """SET NX EX — atomic first-claim with TTL."""

    def __init__(self, redis: Redis, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    async def claim(self, key: str) -> bool:
        result = await self._redis.set(f"alert:dedupe:{key}", "1", nx=True, ex=self._ttl)
        return bool(result)
