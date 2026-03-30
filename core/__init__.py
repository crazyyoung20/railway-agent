"""
Core 模块：精简版核心组件
- cache: 缓存层（内存LRU+可选Redis二级缓存）
- router: 路由层（决定请求走哪条路径）
- pipeline: Pipeline层（简单查询硬编码流程）
- retry: 重试机制
"""
from .cache import CacheLayer, default_cache
from .router import QueryRouter, Route, RouteDecision, default_router
from .pipeline import SimpleQueryPipeline, SimpleQueryParams
from .retry import (
    retry,
    llm_retry,
    skill_retry,
    RetryConfig,
)

__all__ = [
    # Cache
    "CacheLayer",
    "default_cache",
    # Router
    "QueryRouter",
    "Route",
    "RouteDecision",
    "default_router",
    # Pipeline
    "SimpleQueryPipeline",
    "SimpleQueryParams",
    # Retry
    "retry",
    "llm_retry",
    "skill_retry",
    "RetryConfig",
]
