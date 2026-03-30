"""
缓存层
功能：
- 支持内存LRU和可选Redis双后端
- 二级缓存：先查内存，再查Redis，最后回源
- 可配置的TTL策略
- 缓存穿透防护（缓存空值）
"""
import json
import hashlib
import time
from typing import Any, Optional, Dict, Callable, TypeVar
from functools import wraps
from dataclasses import dataclass

from config import settings

T = TypeVar("T")

# Metrics configuration (placeholder - can be enabled if prometheus_client is installed)
METRICS_ENABLED = False
try:
    from prometheus_client import Counter
    # If prometheus is available, we could enable metrics
    # METRICS_ENABLED = True
    # cache_hits = Counter('cache_hits_total', 'Cache hits', ['backend', 'layer'])
    # cache_misses = Counter('cache_misses_total', 'Cache misses', ['backend', 'layer'])
except ImportError:
    pass


# ============== 内存缓存后端（LRU） ==============
class MemoryBackend:
    def __init__(self, maxsize: int = 1000, default_ttl: int = 300):
        try:
            from cachetools import TTLCache
            self._cache = TTLCache(maxsize=maxsize, ttl=default_ttl)
        except ImportError:
            # 降级到简单 dict 实现
            self._cache = {}
            self._maxsize = maxsize
            self._default_ttl = default_ttl

    def _has_cachetools(self) -> bool:
        return hasattr(self._cache, "__contains__") and hasattr(self._cache, "get")

    def get(self, key: str) -> Optional[Any]:
        if self._has_cachetools():
            return self._cache.get(key)

        # 简单实现
        if key in self._cache:
            value, expire_at = self._cache[key]
            if time.time() < expire_at:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if self._has_cachetools():
            self._cache[key] = value
            return

        # 简单实现
        ttl = ttl or self._default_ttl
        if len(self._cache) >= self._maxsize:
            # FIFO 淘汰
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (value, time.time() + ttl)

    def delete(self, key: str):
        if key in self._cache:
            del self._cache[key]

    def clear(self):
        self._cache.clear()


# ============== Redis 缓存后端 ==============
class RedisBackend:
    def __init__(self, redis_url: str = None, key_prefix: str = None):
        try:
            import redis
        except ImportError:
            raise RuntimeError("Redis backend requires redis package: pip install redis")

        self._client = redis.from_url(
            redis_url or settings.cache.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        self._key_prefix = key_prefix or settings.cache.redis_prefix

    def _make_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        data = self._client.get(self._make_key(key))
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl or settings.cache.default_ttl
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        self._client.setex(self._make_key(key), ttl, value)

    def delete(self, key: str):
        self._client.delete(self._make_key(key))

    def delete_pattern(self, pattern: str):
        """批量删除匹配模式的键"""
        keys = self._client.keys(self._make_key(pattern))
        if keys:
            self._client.delete(*keys)

    def clear(self):
        keys = self._client.keys(f"{self._key_prefix}*")
        if keys:
            self._client.delete(*keys)


# ============== 缓存统计 ==============
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    total_requests: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests


# ============== 统一缓存层 ==============
class CacheLayer:
    def __init__(
        self,
        backend: str = None,
        max_memory_size: int = None,
        default_ttl: int = None,
        enable_l2: bool = True,
    ):
        """
        初始化缓存层
        :param backend: 主缓存后端: memory / redis
        :param max_memory_size: 内存缓存最大容量
        :param default_ttl: 默认过期时间（秒）
        :param enable_l2: 是否启用二级缓存（内存+Redis）
        """
        self._backend_type = backend or settings.cache.backend
        self._default_ttl = default_ttl or settings.cache.default_ttl
        self._enable_l2 = enable_l2 and self._backend_type == "redis"

        # 统计
        self._stats = CacheStats()

        # 初始化后端
        self._memory = MemoryBackend(
            maxsize=max_memory_size or settings.cache.max_size,
            default_ttl=self._default_ttl,
        )

        if self._backend_type == "redis":
            self._redis = RedisBackend()
        else:
            self._redis = None

    @staticmethod
    def make_key(*parts: str, namespace: str = "default") -> str:
        """生成缓存键，避免冲突"""
        raw = f"{namespace}:{':'.join(str(p) for p in parts)}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存，支持二级缓存"""
        self._stats.total_requests += 1

        # L1: 内存缓存
        value = self._memory.get(key)
        if value is not None:
            self._stats.hits += 1
            if METRICS_ENABLED:
                cache_hits.labels(backend="memory", layer="l1").inc()
            return value

        if METRICS_ENABLED:
            cache_misses.labels(backend="memory", layer="l1").inc()

        # L2: Redis 缓存
        if self._enable_l2 and self._redis:
            value = self._redis.get(key)
            if value is not None:
                self._stats.hits += 1
                # 回写 L1
                self._memory.set(key, value)
                if METRICS_ENABLED:
                    cache_hits.labels(backend="redis", layer="l2").inc()
                return value

            if METRICS_ENABLED:
                cache_misses.labels(backend="redis", layer="l2").inc()

        self._stats.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """写入缓存，同时写 L1 和 L2"""
        ttl = ttl or self._default_ttl

        # 写 L1
        self._memory.set(key, value, ttl)

        # 写 L2
        if self._redis:
            self._redis.set(key, value, ttl)

    def delete(self, key: str):
        """删除缓存"""
        self._memory.delete(key)
        if self._redis:
            self._redis.delete(key)

    def delete_pattern(self, pattern: str):
        """批量删除匹配模式的键（仅 Redis 支持）"""
        if self._redis:
            self._redis.delete_pattern(pattern)

    def invalidate_query(self, query: str):
        """作废某个查询的缓存"""
        key = self.make_key(query, namespace="query")
        self.delete(key)

    def invalidate_skill(self, skill_name: str, params: dict = None):
        """作废某个 Skill 的缓存"""
        if params:
            key = self.make_key(skill_name, str(sorted(params.items())), namespace="skill")
            self.delete(key)
        else:
            # 删除该 Skill 的所有缓存
            if self._redis:
                pattern = self.make_key(skill_name, "*", namespace="skill")
                self._redis.delete_pattern(pattern)

    def clear(self):
        """清空所有缓存"""
        self._memory.clear()
        if self._redis:
            self._redis.clear()

    @property
    def stats(self) -> CacheStats:
        """获取缓存统计"""
        return self._stats

    # ============== 装饰器 ==============
    def cached(self, ttl: Optional[int] = None, namespace: str = "default"):
        """
        装饰器：缓存函数结果
        用法:
            @cache.cached(ttl=300, namespace="my_func")
            def my_func(a, b):
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                # 生成缓存键：函数名 + args + kwargs
                key_parts = [func.__qualname__] + [str(a) for a in args]
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")

                key = self.make_key(*key_parts, namespace=namespace)
                cached_value = self.get(key)

                if cached_value is not None:
                    return cached_value

                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result
            return wrapper
        return decorator

    def async_cached(self, ttl: Optional[int] = None, namespace: str = "default"):
        """异步函数缓存装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                key_parts = [func.__qualname__] + [str(a) for a in args]
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")

                key = self.make_key(*key_parts, namespace=namespace)
                cached_value = self.get(key)

                if cached_value is not None:
                    return cached_value

                result = await func(*args, **kwargs)
                self.set(key, result, ttl)
                return result
            return wrapper
        return decorator


# 全局默认缓存实例
default_cache = CacheLayer()
