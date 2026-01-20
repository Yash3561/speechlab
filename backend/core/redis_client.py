"""
Redis Client for SpeechLab

Provides caching, session management, and pub/sub for real-time updates.
Supports both local Redis and Upstash (cloud Redis).
"""

from typing import Optional, Any, List
from datetime import timedelta
import json

from backend.core.config import settings
from backend.core.logging import logger

# Redis imports with graceful fallback
try:
    import redis
    from redis import asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("Redis not installed. Using in-memory fallback.")


class RedisClient:
    """
    Redis client wrapper with automatic fallback.
    
    Features:
    - Caching experiment data
    - Session management
    - Pub/sub for real-time metrics
    - Training job queue
    """
    
    def __init__(self, url: Optional[str] = None):
        self.url = url or settings.redis_url
        self.client: Optional[redis.Redis] = None
        self.async_client: Optional[aioredis.Redis] = None
        self._fallback_cache: dict = {}  # In-memory fallback
        
        if HAS_REDIS:
            self._connect()
        else:
            logger.warning("Using in-memory cache (Redis not available)")
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            self.client = redis.from_url(
                self.url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis: {self._mask_url(self.url)}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self.client = None
    
    async def connect_async(self):
        """Establish async Redis connection."""
        if not HAS_REDIS:
            return
        
        try:
            self.async_client = await aioredis.from_url(
                self.url,
                decode_responses=True,
            )
            await self.async_client.ping()
            logger.info("Connected to Redis (async)")
        except Exception as e:
            logger.warning(f"Async Redis connection failed: {e}")
            self.async_client = None
    
    def _mask_url(self, url: str) -> str:
        """Mask password in URL for logging."""
        if "@" in url:
            parts = url.split("@")
            return f"***@{parts[-1]}"
        return url
    
    # ========================================
    # Basic Operations
    # ========================================
    
    def get(self, key: str) -> Optional[str]:
        """Get a value from Redis."""
        if self.client:
            try:
                return self.client.get(key)
            except Exception as e:
                logger.error(f"Redis GET error: {e}")
        return self._fallback_cache.get(key)
    
    def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in Redis with optional TTL (seconds)."""
        if self.client:
            try:
                if ttl:
                    return self.client.setex(key, ttl, value)
                return self.client.set(key, value)
            except Exception as e:
                logger.error(f"Redis SET error: {e}")
        
        self._fallback_cache[key] = value
        return True
    
    def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if self.client:
            try:
                return self.client.delete(key) > 0
            except Exception as e:
                logger.error(f"Redis DELETE error: {e}")
        
        return self._fallback_cache.pop(key, None) is not None
    
    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if self.client:
            try:
                return self.client.exists(key) > 0
            except Exception as e:
                logger.error(f"Redis EXISTS error: {e}")
        return key in self._fallback_cache
    
    # ========================================
    # JSON Operations
    # ========================================
    
    def get_json(self, key: str) -> Optional[Any]:
        """Get and parse a JSON value."""
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store a value as JSON."""
        return self.set(key, json.dumps(value), ttl)
    
    # ========================================
    # Hash Operations (for experiments)
    # ========================================
    
    def hget(self, name: str, key: str) -> Optional[str]:
        """Get a field from a hash."""
        if self.client:
            try:
                return self.client.hget(name, key)
            except Exception as e:
                logger.error(f"Redis HGET error: {e}")
        
        hash_data = self._fallback_cache.get(name, {})
        return hash_data.get(key)
    
    def hset(self, name: str, key: str, value: str) -> bool:
        """Set a field in a hash."""
        if self.client:
            try:
                return self.client.hset(name, key, value) >= 0
            except Exception as e:
                logger.error(f"Redis HSET error: {e}")
        
        if name not in self._fallback_cache:
            self._fallback_cache[name] = {}
        self._fallback_cache[name][key] = value
        return True
    
    def hgetall(self, name: str) -> dict:
        """Get all fields from a hash."""
        if self.client:
            try:
                return self.client.hgetall(name)
            except Exception as e:
                logger.error(f"Redis HGETALL error: {e}")
        return self._fallback_cache.get(name, {})
    
    # ========================================
    # Pub/Sub for Real-Time Metrics
    # ========================================
    
    def publish(self, channel: str, message: str) -> int:
        """Publish a message to a channel."""
        if self.client:
            try:
                return self.client.publish(channel, message)
            except Exception as e:
                logger.error(f"Redis PUBLISH error: {e}")
        return 0
    
    def subscribe(self, channel: str):
        """Subscribe to a channel (returns pubsub object)."""
        if self.client:
            try:
                pubsub = self.client.pubsub()
                pubsub.subscribe(channel)
                return pubsub
            except Exception as e:
                logger.error(f"Redis SUBSCRIBE error: {e}")
        return None
    
    # ========================================
    # List Operations (for job queue)
    # ========================================
    
    def lpush(self, key: str, *values: str) -> int:
        """Push values to the left of a list (queue)."""
        if self.client:
            try:
                return self.client.lpush(key, *values)
            except Exception as e:
                logger.error(f"Redis LPUSH error: {e}")
        
        if key not in self._fallback_cache:
            self._fallback_cache[key] = []
        for v in values:
            self._fallback_cache[key].insert(0, v)
        return len(self._fallback_cache[key])
    
    def rpop(self, key: str) -> Optional[str]:
        """Pop a value from the right of a list."""
        if self.client:
            try:
                return self.client.rpop(key)
            except Exception as e:
                logger.error(f"Redis RPOP error: {e}")
        
        lst = self._fallback_cache.get(key, [])
        return lst.pop() if lst else None
    
    def llen(self, key: str) -> int:
        """Get the length of a list."""
        if self.client:
            try:
                return self.client.llen(key)
            except Exception as e:
                logger.error(f"Redis LLEN error: {e}")
        return len(self._fallback_cache.get(key, []))
    
    # ========================================
    # Experiment Caching
    # ========================================
    
    def cache_experiment(self, exp_id: str, data: dict, ttl: int = 300):
        """Cache experiment data with 5 min default TTL."""
        key = f"experiment:{exp_id}"
        self.set_json(key, data, ttl)
    
    def get_cached_experiment(self, exp_id: str) -> Optional[dict]:
        """Get cached experiment data."""
        key = f"experiment:{exp_id}"
        return self.get_json(key)
    
    def invalidate_experiment(self, exp_id: str):
        """Invalidate experiment cache."""
        key = f"experiment:{exp_id}"
        self.delete(key)
    
    # ========================================
    # Training Job Queue
    # ========================================
    
    def enqueue_job(self, job_data: dict) -> str:
        """Add a training job to the queue."""
        job_id = job_data.get("job_id", "unknown")
        self.lpush("training:queue", json.dumps(job_data))
        logger.info(f"Enqueued training job: {job_id}")
        return job_id
    
    def dequeue_job(self) -> Optional[dict]:
        """Get the next job from the queue."""
        job_str = self.rpop("training:queue")
        if job_str:
            return json.loads(job_str)
        return None
    
    def queue_length(self) -> int:
        """Get the number of jobs in queue."""
        return self.llen("training:queue")
    
    # ========================================
    # Metrics Broadcast
    # ========================================
    
    def broadcast_metrics(self, exp_id: str, metrics: dict):
        """Broadcast training metrics via pub/sub."""
        channel = f"metrics:{exp_id}"
        message = json.dumps(metrics)
        self.publish(channel, message)
    
    def close(self):
        """Close Redis connections."""
        if self.client:
            self.client.close()
        if self.async_client:
            # Note: async close needs to be awaited
            pass


# Singleton instance
_redis_client: Optional[RedisClient] = None


def get_redis() -> RedisClient:
    """Get the Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client
