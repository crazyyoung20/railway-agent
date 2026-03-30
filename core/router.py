"""
路由层：决定请求走哪条路径
策略：
1. 非法查询检查 → 直接拒绝
2. 复杂查询关键词 → 直接走 Agent（不调用 LLM）
3. 正则匹配简单查询 → 走 Pipeline（不调用 LLM）
4. LLM 意图识别 → 以上都不匹配时才调用 LLM（兜底）
5. 默认兜底 → 走 Agent
"""
import re
import json
import logging
from typing import Literal, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config import settings
from .retry import llm_retry

logger = logging.getLogger("router")

# 初始化轻量级 LLM（如果配置了）
try:
    from langchain_openai import ChatOpenAI
    _llm = ChatOpenAI(
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=0.0,
        max_tokens=256,
        timeout=settings.llm.timeout,
    )
    LLM_AVAILABLE = True
except Exception as e:
    logger.warning(f"LLM 初始化失败，将使用正则路由：{e}")
    LLM_AVAILABLE = False


class Route(Enum):
    CACHE = "cache"
    PIPELINE = "pipeline"
    AGENT = "agent"
    REJECT = "reject"


@dataclass
class RouteDecision:
    route: Route
    confidence: float
    reason: str
    extracted_params: Optional[dict] = None


# 热门城市（直达 Pipeline 适用）
HOT_CITIES = {
    "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "西安",
    "成都", "重庆", "长沙", "郑州", "天津", "苏州", "青岛", "厦门",
    "大连", "宁波", "无锡", "合肥", "福州", "南昌", "济南", "南宁",
    "海口", "贵阳", "昆明", "拉萨", "兰州", "西宁", "银川", "乌鲁木齐"
}

# 非法查询关键词（非铁路服务）
INVALID_KEYWORDS = {
    "纽约", "巴黎", "伦敦", "东京", "首尔", "新加坡", "悉尼", "洛杉矶",
    "飞机", "航班", "机票", "轮船", "邮轮", "打车", "顺风车", "网约车",
    "酒店", "住宿", "民宿", "门票", "景点", "旅游", "签证", "护照"
}

# 复杂查询关键词（必须走 Agent）
COMPLEX_KEYWORDS = {
    "中转", "换乘", "推荐", "比较", "为什么", "怎么", "如何", "哪个",
    "换个", "改成", "修改", "调整", "那个", "之前", "刚才", "再看",
    "详细说", "具体点", "展开", "分析", "对比", "评估", "风险",
    "最快", "最稳", "最便宜", "最舒适", "性价比", "优先", "最好",
    "预算", "钱", "费用", "价格", "时间", "时长", "多久", "几点"
}

# 简单查询模式（正则提取）
SIMPLE_QUERY_PATTERNS = [
    # 模式 1: "A 到 B 明天"
    re.compile(r"^(.+?)\s*到\s*(.+?)\s*(明天 | 后天 | 大后天|\d{4}-\d{2}-\d{2}|\d{1,2}月\d{1,2}日)?$"),
    # 模式 2: "明天从 A 到 B"
    re.compile(r"^(明天 | 后天 | 大后天|\d{4}-\d{2}-\d{2}|\d{1,2}月\d{1,2}日)?\s*从?\s*(.+?)\s*到\s*(.+?)$"),
]

# LLM 意图识别 Prompt
INTENT_PROMPT = """
你是铁路查询意图识别器，只输出严格 JSON 格式，不要其他任何内容：
{{
  "intent": "direct_ticket/train_info/station_info/compare/complex",
  "params": {{
    "origin": "出发站（没有则填 null）",
    "destination": "到达站（没有则填 null）",
    "date": "日期（没有则填 null）",
    "train_no": "车次号（没有则填 null）",
    "station": "车站名（没有则填 null）",
    "filters": ["筛选条件列表，没有则填 []"]
  }},
  "need_agent": true/false,
  "confidence": 0.0-1.0
}}

规则：
1. intent 枚举值：
   - direct_ticket: 直达车次查询
   - train_info: 单个车次信息查询（如"G1 有充电口吗"）
   - station_info: 单个车站信息查询（如"北京南有休息室吗"）
   - compare: 车次/车站对比
   - complex: 其他复杂问题
2. need_agent: 需要多步推理、多轮上下文、个性化推荐、中转规划的复杂问题填 true，否则填 false
3. 所有参数严格从用户问题中提取，不要编造
4. confidence 是你对识别结果的置信度，0 到 1 之间

用户问题：{query}
输出 JSON：
"""


class QueryRouter:
    def __init__(self, enable_llm_router: bool = True):
        """
        初始化路由
        :param enable_llm_router: 是否启用 LLM 意图路由（默认开启，作为兜底）
        """
        self._enable_llm = enable_llm_router and LLM_AVAILABLE
        logger.info(f"路由初始化完成，LLM 路由：{'已启用' if self._enable_llm else '已禁用，使用正则'}")

    @llm_retry
    def _get_llm_intent(self, query: str) -> Optional[dict]:
        """调用 LLM 做意图识别（兜底策略）"""
        try:
            prompt = INTENT_PROMPT.format(query=query)
            resp = _llm.invoke(prompt)
            content = resp.content.strip()
            # 去掉可能的 markdown 标记
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            return json.loads(content)
        except Exception as e:
            logger.debug(f"LLM 意图识别失败：{e}")
            return None

    def _extract_simple_params(self, query: str) -> Optional[dict]:
        """用正则提取简单查询参数"""
        for pattern in SIMPLE_QUERY_PATTERNS:
            match = pattern.match(query.strip())
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    # 模式 1 或 2，取决于哪个位置有值
                    if groups[0] and groups[0] not in ["明天", "后天", "大后天"] and not re.match(r"\d{4}", groups[0]):
                        origin = groups[0]
                        dest = groups[1]
                        date = groups[2]
                    else:
                        origin = groups[1]
                        dest = groups[2]
                        date = groups[0]

                    return {
                        "origin": origin.strip(),
                        "destination": dest.strip(),
                        "date": date.strip() if date else None,
                    }
        return None

    def _check_invalid(self, query: str) -> Tuple[bool, str]:
        """检查是否是非法查询"""
        query_lower = query.lower()
        for kw in INVALID_KEYWORDS:
            if kw in query:
                return True, f"包含非法关键词：{kw}"
        return False, ""

    def _is_complex(self, query: str) -> Tuple[bool, str]:
        """检查是否是复杂查询（必须走 Agent）"""
        for kw in COMPLEX_KEYWORDS:
            if kw in query:
                return True, f"包含复杂关键词：{kw}"
        return False, ""

    def _is_simple_pipeline_regex(self, query: str) -> Tuple[bool, str, Optional[dict]]:
        """正则模式检查是否是简单查询（可走 Pipeline）"""
        # 尝试提取参数
        params = self._extract_simple_params(query)
        if not params:
            return False, "", None

        # 检查出发地和目的地是否在热门城市列表
        origin = params["origin"]
        dest = params["destination"]

        # 简单匹配：只要包含热门城市名就算
        origin_in_hot = any(city in origin for city in HOT_CITIES)
        dest_in_hot = any(city in dest for city in HOT_CITIES)

        if origin_in_hot and dest_in_hot:
            return True, "热门城市直达查询", params

        return False, "", None

    def route(self, query: str, check_cache_only: bool = False) -> RouteDecision:
        """
        路由决策主函数（优先正则，LLM 兜底）
        :param query: 用户查询
        :param check_cache_only: 只检查缓存（不做路由判断）
        :return: RouteDecision
        """
        # 1. 检查非法查询
        is_invalid, reason = self._check_invalid(query)
        if is_invalid:
            return RouteDecision(
                route=Route.REJECT,
                confidence=0.95,
                reason=reason
            )

        if check_cache_only:
            return RouteDecision(
                route=Route.AGENT,
                confidence=0.5,
                reason="默认走 Agent"
            )

        # 2. 【优先】复杂查询关键词检查 → 直接走 Agent（省 LLM 调用）
        is_complex, complex_reason = self._is_complex(query)
        if is_complex:
            return RouteDecision(
                route=Route.AGENT,
                confidence=0.9,
                reason=complex_reason
            )

        # 3. 【优先】正则匹配简单查询 → 走 Pipeline（不调用 LLM）
        is_simple, reason, params = self._is_simple_pipeline_regex(query)
        if is_simple:
            return RouteDecision(
                route=Route.PIPELINE,
                confidence=0.85,
                reason=f"正则匹配：{reason}",
                extracted_params=params
            )

        # 4. 【兜底】正则无法判断时才调用 LLM
        if self._enable_llm:
            intent_result = self._get_llm_intent(query)
            if intent_result:
                try:
                    need_agent = intent_result.get("need_agent", False)
                    confidence = intent_result.get("confidence", 0.8)
                    intent = intent_result.get("intent", "complex")
                    params = intent_result.get("params", {})

                    if not need_agent and confidence >= 0.7:
                        # 简单意图，走 Pipeline
                        if intent == "direct_ticket" and params.get("origin") and params.get("destination"):
                            return RouteDecision(
                                route=Route.PIPELINE,
                                confidence=confidence,
                                reason=f"LLM 识别：直达车次查询",
                                extracted_params=params
                            )
                        elif intent in ["train_info", "station_info", "compare"]:
                            return RouteDecision(
                                route=Route.PIPELINE,
                                confidence=confidence,
                                reason=f"LLM 识别：{intent}",
                                extracted_params=params
                            )
                except Exception as e:
                    logger.debug(f"LLM 意图结果解析失败：{e}")

        # 5. 默认兜底：走 Agent
        return RouteDecision(
            route=Route.AGENT,
            confidence=0.6,
            reason="默认走 ReAct Agent"
        )

    # ========== 便捷方法 ==========
    def should_go_cache(self, query: str) -> bool:
        """判断是否适合缓存（避免缓存上下文相关的查询）"""
        context_keywords = ["那个", "这个", "之前", "刚才", "换个", "改成", "修改"]
        return not any(kw in query for kw in context_keywords)

    def should_go_pipeline(self, query: str) -> bool:
        """判断是否应该走 Pipeline"""
        decision = self.route(query)
        return decision.route == Route.PIPELINE

    def should_go_agent(self, query: str) -> bool:
        """判断是否应该走 Agent"""
        decision = self.route(query)
        return decision.route == Route.AGENT


# 全局默认路由实例
default_router = QueryRouter()
