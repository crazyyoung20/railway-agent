"""
Hybrid Agent 简单测试
运行: pytest tests/test_hybrid.py -v
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestConfig:
    """配置模块测试"""
    def test_config_loads(self):
        from config import settings
        assert settings.env in ["dev", "staging", "prod"]
        assert settings.llm.provider in ["zhipu", "openai"]


class TestRouter:
    """路由层测试"""
    def test_router_init(self):
        from core import default_router
        assert default_router is not None

    def test_simple_query_detection(self):
        from core import default_router
        decision = default_router.route("北京到上海明天")
        assert decision.route.value in ["pipeline", "agent"]

    def test_complex_query_detection(self):
        from core import default_router
        decision = default_router.route("石家庄到三亚，想最快，但预算只有500")
        assert decision.route.value == "agent"

    def test_invalid_query_detection(self):
        from core import default_router
        decision = default_router.route("北京到纽约")
        # 可能reject，也可能agent，取决于配置
        assert decision.route.value in ["reject", "agent"]


class TestCache:
    """缓存层测试"""
    def test_cache_init(self):
        from core import default_cache
        assert default_cache is not None

    def test_cache_set_get(self):
        from core import default_cache
        test_key = default_cache.make_key("test_key", namespace="test")
        test_value = {"final_answer": "test answer"}

        default_cache.set(test_key, test_value)
        cached = default_cache.get(test_key)

        assert cached == test_value

    def test_cache_key_generation(self):
        from core import CacheLayer
        key1 = CacheLayer.make_key("北京到上海", "user1", namespace="query")
        key2 = CacheLayer.make_key("北京到上海", "user2", namespace="query")
        key3 = CacheLayer.make_key("北京到广州", "user1", namespace="query")

        assert key1 != key2  # 不同用户不同key
        assert key1 != key3  # 不同查询不同key


class TestPipelineParams:
    """Pipeline参数测试"""
    def test_default_date(self):
        from core import SimpleQueryParams
        from datetime import datetime, timedelta

        params = SimpleQueryParams("北京", "上海")
        expected = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        assert params.date == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
