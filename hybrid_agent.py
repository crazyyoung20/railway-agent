"""
Hybrid Railway Agent v4：分层架构整合入口
特点：
- 零侵入原有 RailwayAgentV3，完全兼容
- 分层路由：缓存 → Pipeline → Agent
- 工业化特性：缓存、监控、日志、重试
- 随时可以回退到纯Agent模式

架构：
                    ┌─────────────────┐
                    │  HybridAgentV4  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    ┌─────────┐        ┌──────────┐        ┌──────────┐
    │ 缓存层  │        │ Pipeline  │        │  Agent层 │
    │ 50ms    │        │ 300ms    │        │ 3s+      │
    └─────────┘        └──────────┘        └─────┬────┘
                                                    │
                                         ┌──────────▼─────────┐
                                         │  RailwayAgentV3  │
                                         │  (原有逻辑，未改动)  │
                                         └────────────────────┘
"""
import json
import uuid
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config import settings
from core import (
    default_cache,
    default_router,
    Route,
    SimpleQueryPipeline,
    llm_retry,
    CacheLayer,
    QueryRouter,
)

# Mock metrics module for now
class MockMetrics:
    def record_cache_hit(self, **kwargs): pass
    def record_cache_miss(self, **kwargs): pass
    def measure_request(self, name):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()
metrics = MockMetrics()

# Mock context functions
def set_request_id(req_id): pass
def set_user_id(user_id): pass
def set_thread_id(thread_id): pass

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("hybrid_agent")


@dataclass
class HybridResult:
    """Hybrid Agent 统一返回结果"""
    success: bool
    final_answer: str
    route_info: Dict[str, Any]
    user_query: str
    thread_id: str
    user_id: str
    # Agent特有字段
    agent_result: Optional[Dict[str, Any]] = None
    # Pipeline特有字段
    pipeline_result: Optional[Dict[str, Any]] = None
    # 缓存特有字段
    from_cache: bool = False
    # 错误字段
    error_msg: Optional[str] = None


class HybridRailwayAgent:
    """
    分层架构铁路Agent
    零侵入原有RailwayAgentV3，只是在外面包装了一层路由和缓存
    """
    def __init__(
        self,
        cache: Optional[CacheLayer] = None,
        router: Optional[QueryRouter] = None,
        enable_pipeline: bool = None,
        enable_cache: bool = True,
    ):
        """
        初始化Hybrid Agent
        :param cache: 缓存层实例，默认使用全局default_cache
        :param router: 路由层实例，默认使用全局default_router
        :param enable_pipeline: 是否启用Pipeline层，默认从配置读取
        :param enable_cache: 是否启用缓存层
        """
        self._enable_cache = enable_cache
        self._enable_pipeline = enable_pipeline if enable_pipeline is not None else settings.pipeline.enabled

        self._cache = cache or default_cache
        self._router = router or default_router

        # 加载Skills（给Pipeline用）
        self._tools = self._load_tools()
        self._tools_dict = {t.name: t for t in self._tools}

        # 初始化Pipeline
        if self._enable_pipeline and self._tools_dict:
            try:
                self._pipeline = SimpleQueryPipeline(self._tools_dict)
                logger.info("[HybridAgent] Pipeline 初始化成功")
            except Exception as e:
                logger.warning(f"[HybridAgent] Pipeline 初始化失败: {e}，将只使用缓存+Agent")
                self._pipeline = None
        else:
            self._pipeline = None

        # 初始化原有的Agent（RailwayAgentV3）
        self._agent_v3 = self._load_agent_v3()

        logger.info(
            f"[HybridAgent] 初始化完成，"
            f"缓存: {'启用' if self._enable_cache else '禁用'}, "
            f"Pipeline: {'启用' if self._enable_pipeline and self._pipeline else '禁用'}"
        )

    def _load_tools(self) -> list:
        """加载Skills（复用原有的skill_loader）"""
        from skill_loader import SkillLoader
        import os
        from pathlib import Path

        skills_dir = Path(__file__).parent / "skills"
        loader = SkillLoader(skills_dir)
        return loader.load_all()

    def _load_agent_v3(self):
        """加载原有的RailwayAgentV3"""
        try:
            from agent import RailwayAgentV3
            # 检查是否有API Key（使用配置中的API Key，和路由层保持一致）
            has_key = bool(settings.llm.api_key)
            agent = RailwayAgentV3(mock=not has_key)
            logger.info("[HybridAgent] RailwayAgentV3 加载成功")
            return agent
        except Exception as e:
            logger.error(f"[HybridAgent] RailwayAgentV3 加载失败: {e}", exc_info=True)
            raise

    def chat(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        user_id: str = "default",
    ) -> HybridResult:
        """
        分层架构聊天入口
        :param user_input: 用户输入
        :param thread_id: 会话ID，不传会自动生成
        :param user_id: 用户ID
        :return: HybridResult
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # 设置上下文
        set_request_id(request_id)
        set_user_id(user_id)
        if thread_id:
            set_thread_id(thread_id)
        else:
            thread_id = f"thread_{uuid.uuid4().hex[:8]}"
            set_thread_id(thread_id)

        logger.info(f"[HybridAgent] 收到请求，thread={thread_id}, user={user_id}")
        logger.info(f"[HybridAgent] 用户输入: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")

        try:
            # ========== Layer 1: 缓存检查 ==========
            if self._enable_cache:
                cached_result = self._try_cache(user_input, user_id)
                if cached_result:
                    elapsed = int((time.time() - start_time) * 1000)
                    logger.info(f"[HybridAgent] 缓存命中，耗时 {elapsed}ms")
                    return HybridResult(
                        success=True,
                        final_answer=cached_result["final_answer"],
                        route_info={"layer": "cache", "latency_ms": elapsed},
                        user_query=user_input,
                        thread_id=thread_id,
                        user_id=user_id,
                        from_cache=True,
                    )

            # ========== Layer 2: 路由决策 ==========
            route_decision = self._router.route(user_input)
            logger.info(f"[HybridAgent] 路由决策: {route_decision.route.value} (confidence: {route_decision.confidence:.2f})")

            # ========== Layer 3: 执行对应路径 ==========
            result: Optional[HybridResult] = None

            # 尝试Pipeline
            if route_decision.route == Route.PIPELINE and self._pipeline and route_decision.extracted_params:
                try:
                    result = self._execute_pipeline(
                        user_input=user_input,
                        extracted_params=route_decision.extracted_params,
                        thread_id=thread_id,
                        user_id=user_id,
                        start_time=start_time,
                    )
                except Exception as e:
                    logger.warning(f"[HybridAgent] Pipeline执行失败，回退到Agent: {e}")
                    result = None

            # Pipeline失败或路由到Agent，走Agent
            if result is None:
                result = self._execute_agent(
                    user_input=user_input,
                    thread_id=thread_id,
                    user_id=user_id,
                    start_time=start_time,
                )

            # ========== 写入缓存 ==========
            if self._enable_cache and self._router.should_go_cache(user_input):
                self._write_cache(user_input, user_id, result)

            return result

        except Exception as e:
            elapsed = int((time.time() - start_time) * 1000)
            logger.error(f"[HybridAgent] 请求处理失败: {e}", exc_info=True)
            return HybridResult(
                success=False,
                final_answer="抱歉，处理您的请求时出错了，请稍后再试。",
                route_info={"layer": "error", "latency_ms": elapsed},
                user_query=user_input,
                thread_id=thread_id,
                user_id=user_id,
                error_msg=str(e),
            )

    def _try_cache(self, user_input: str, user_id: str) -> Optional[Dict[str, Any]]:
        """尝试从缓存获取结果"""
        key = self._cache.make_key(user_input, user_id, namespace="query")
        cached = self._cache.get(key)
        if cached:
            metrics.record_cache_hit(layer="l1", backend="memory" if settings.cache.backend == "memory" else "redis")
            return cached
        metrics.record_cache_miss(layer="l1", backend="memory" if settings.cache.backend == "memory" else "redis")
        return None

    def _write_cache(self, user_input: str, user_id: str, result: HybridResult):
        """写入缓存"""
        key = self._cache.make_key(user_input, user_id, namespace="query")
        cache_data = {
            "final_answer": result.final_answer,
            "user_query": result.user_query,
        }
        ttl = settings.cache.query_ttl
        self._cache.set(key, cache_data, ttl=ttl)
        logger.debug(f"[HybridAgent] 结果已缓存，TTL={ttl}s")

    def _execute_pipeline(
        self,
        user_input: str,
        extracted_params: Dict[str, Any],
        thread_id: str,
        user_id: str,
        start_time: float,
    ) -> HybridResult:
        """执行Pipeline层"""
        logger.info("[HybridAgent] 执行Pipeline层")
        with metrics.measure_request("pipeline"):
            pipeline_result = self._pipeline.execute_from_query(user_input, extracted_params)
            elapsed = int((time.time() - start_time) * 1000)

            return HybridResult(
                success=pipeline_result.get("success", True),
                final_answer=pipeline_result["final_answer"],
                route_info={"layer": "pipeline", "latency_ms": elapsed},
                user_query=user_input,
                thread_id=thread_id,
                user_id=user_id,
                pipeline_result=pipeline_result,
            )

    def _execute_agent(
        self,
        user_input: str,
        thread_id: str,
        user_id: str,
        start_time: float,
    ) -> HybridResult:
        """执行Agent层（原有V3）"""
        logger.info("[HybridAgent] 执行Agent层（V3）")
        with metrics.measure_request("agent"):
            agent_result = self._agent_v3.chat(
                user_input=user_input,
                thread_id=thread_id,
                user_id=user_id,
            )
            elapsed = int((time.time() - start_time) * 1000)

            return HybridResult(
                success=True,
                final_answer=agent_result["final_answer"],
                route_info={"layer": "agent", "latency_ms": elapsed},
                user_query=user_input,
                thread_id=thread_id,
                user_id=user_id,
                agent_result=agent_result,
            )

    # ========== 兼容原有接口（平滑迁移） ==========
    def chat_compat(
        self,
        user_input: str,
        thread_id: str,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """
        兼容原有RailwayAgentV3.chat()的接口
        方便平滑迁移，直接替换即可
        """
        hybrid_result = self.chat(user_input, thread_id, user_id)
        # 构造和原有V3一样的返回格式
        return {
            "thread_id": hybrid_result.thread_id,
            "user_id": hybrid_result.user_id,
            "user_input": hybrid_result.user_query,
            "final_answer": hybrid_result.final_answer,
            "iterations": hybrid_result.agent_result.get("iterations", 0) if hybrid_result.agent_result else 0,
            "message_count": hybrid_result.agent_result.get("message_count", 0) if hybrid_result.agent_result else 1,
            "route_info": hybrid_result.route_info,
            "from_cache": hybrid_result.from_cache,
        }

    # ========== 直接访问原有Agent（回退方案） ==========
    @property
    def agent_v3(self):
        """直接访问原有的V3 Agent，方便回退"""
        return self._agent_v3


# ========== 便捷入口 ==========
def create_hybrid_agent() -> HybridRailwayAgent:
    """创建默认配置的Hybrid Agent"""
    return HybridRailwayAgent()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 65)
    print("  Hybrid Railway Agent v4 - 分层架构演示")
    print("=" * 65)

    agent = create_hybrid_agent()

    # 测试简单查询（应该走Pipeline）
    print("\n--- 测试1: 简单查询（北京到上海明天）---")
    result = agent.chat("北京到上海明天", user_id="demo_user")
    print(f"  路由层: {result.route_info['layer']}")
    print(f"  耗时: {result.route_info.get('latency_ms', 0)}ms")
    print(f"  回答: {result.final_answer[:150]}...")

    # 测试复杂查询（应该走Agent）
    print("\n--- 测试2: 复杂查询（需要中转和推荐）---")
    result2 = agent.chat("石家庄到三亚，想最快，但预算只有500", user_id="demo_user")
    print(f"  路由层: {result2.route_info['layer']}")
    print(f"  耗时: {result2.route_info.get('latency_ms', 0)}ms")
    print(f"  回答: {result2.final_answer[:150]}...")

    print("\n" + "=" * 65)
    print("  演示完成！")
    print("=" * 65)
