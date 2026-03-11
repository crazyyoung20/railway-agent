"""
模块四：Agent 规划层 + Self-Reflection 闭环
功能：动态生成中转方案 → 校验 → 反馈 → 修正 → 重试
依赖：模块一(intent_parser) + 模块二(ticket_query) + 模块三(rag_knowledge)
运行方式：python agent_planner.py
注意：将本文件与 intent_parser.py / ticket_query.py / rag_knowledge.py 放在同一目录
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from enum import Enum
from pydantic import BaseModel, Field

# ── 导入前三个模块 ────────────────────────────────────────────────────────────
from intent_parser import (
    IntentParser, IntentSlots, StationNormalizer, TravelPreference
)
from ticket_query import (
    TicketQueryTool, TrainTicket, QueryResult, TicketStatus, SeatInfo
)
from rag_knowledge import HybridRetriever, KNOWLEDGE_DOCS

logger = logging.getLogger("agent_planner")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

# ── 加载 .env ─────────────────────────────────────────────────────────────────
def load_dotenv(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_dotenv()

# ── ZhipuAI ───────────────────────────────────────────────────────────────────
try:
    from zhipuai import ZhipuAI
    zhipu_client = ZhipuAI(api_key=os.environ.get("ZHIPUAI_API_KEY", ""))
    logger.info("ZhipuAI 初始化成功")
except Exception as e:
    zhipu_client = None
    logger.warning(f"ZhipuAI 不可用，将使用规则模式: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 方案数据结构
# ══════════════════════════════════════════════════════════════════════════════

class RouteType(str, Enum):
    DIRECT   = "直达"
    TRANSFER = "中转"


class TransferPlan(BaseModel):
    """单个中转方案"""
    plan_id: str
    route_type: RouteType
    legs: List[TrainTicket]                    # 每段列车，直达=1段，中转=2段
    transfer_station: Optional[str] = None     # 中转站名
    transfer_wait_minutes: Optional[int] = None
    total_duration_minutes: int = 0
    total_price: Dict[str, float] = Field(default_factory=dict)   # seat_type→price
    available_seats: List[str] = Field(default_factory=list)      # 两程都有票的座位类型
    risk_score: float = 0.0    # 风险分 0~1，越低越好
    risk_reasons: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)        # 多目标评分
    data_source: str = "mock"

    def is_valid(self) -> bool:
        return len(self.available_seats) > 0 and len(self.legs) > 0


class PlanningResult(BaseModel):
    """Agent 规划结果"""
    query: str
    origin: str
    destination: str
    date: str
    plans: List[TransferPlan] = Field(default_factory=list)
    reflection_log: List[str] = Field(default_factory=list)  # Self-Reflection 过程记录
    final_recommendation: Optional[str] = None
    planning_iterations: int = 0
    success: bool = True
    error_msg: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. 换乘枢纽知识库（Agent 规划候选中转站）
# ══════════════════════════════════════════════════════════════════════════════

# 格式：(出发城市关键词, 到达城市关键词) → [候选中转站列表（按优先级）]
TRANSFER_HUB_RULES: List[Tuple[List[str], List[str], List[str]]] = [
    # (出发关键词, 到达关键词, 推荐中转站)
    (["北京"],          ["广州", "深圳", "珠海"],    ["武汉", "郑州东", "长沙南"]),
    (["广州", "深圳"],  ["北京"],                    ["武汉", "郑州东", "长沙南"]),
    (["北京"],          ["上海"],                    ["南京南", "济南西"]),
    (["上海"],          ["北京"],                    ["南京南", "济南西"]),
    (["北京"],          ["成都", "重庆"],             ["郑州东", "武汉"]),
    (["成都", "重庆"],  ["北京"],                    ["郑州东", "武汉"]),
    (["上海"],          ["成都", "重庆"],             ["武汉", "郑州东"]),
    (["成都", "重庆"],  ["上海"],                    ["武汉", "郑州东"]),
    (["北京"],          ["西安", "兰州", "乌鲁木齐"], ["郑州东"]),
    (["上海", "南京"],  ["西安", "兰州"],             ["郑州东"]),
    (["广州", "深圳"],  ["成都", "重庆"],             ["贵阳北", "长沙南", "武汉"]),
    (["上海"],          ["广州", "深圳"],             ["杭州东", "长沙南"]),
    (["广州", "深圳"],  ["上海"],                    ["长沙南", "杭州东"]),
    (["北京"],          ["哈尔滨", "沈阳", "长春"],   ["沈阳北"]),
    (["上海"],          ["昆明", "贵阳"],             ["长沙南", "贵阳北"]),
]

# 枢纽换乘最短安全时间（分钟）
HUB_MIN_TRANSFER_MINUTES = {
    "郑州东":  45,
    "武汉":    45,
    "长沙南":  40,
    "广州南":  40,
    "上海虹桥": 35,
    "北京南":  40,
    "北京西":  40,
    "南京南":  35,
    "成都东":  35,
    "重庆北":  35,
    "西安北":  35,
    "济南西":  35,
    "杭州东":  35,
    "贵阳北":  40,
    "沈阳北":  35,
}
DEFAULT_MIN_TRANSFER = 40


def get_candidate_hubs(origin: str, destination: str) -> List[str]:
    """根据出发地和目的地，推荐候选中转枢纽"""
    candidates = []
    for orig_keywords, dest_keywords, hubs in TRANSFER_HUB_RULES:
        orig_match = any(kw in origin for kw in orig_keywords)
        dest_match = any(kw in destination for kw in dest_keywords)
        if orig_match and dest_match:
            for hub in hubs:
                if hub not in candidates:
                    candidates.append(hub)

    # 如果没有匹配规则，返回通用大枢纽
    if not candidates:
        candidates = ["郑州东", "武汉", "长沙南"]
        logger.warning(f"[HubRule] 无匹配规则 {origin}→{destination}，使用通用枢纽")
    else:
        logger.info(f"[HubRule] {origin}→{destination} 推荐枢纽: {candidates}")

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# 3. 换乘风险评估器
# ══════════════════════════════════════════════════════════════════════════════

class RiskEvaluator:
    """评估中转方案的风险"""

    def evaluate(self, plan: TransferPlan, date: str) -> TransferPlan:
        risk_score = 0.0
        reasons = []

        if plan.route_type == RouteType.DIRECT:
            plan.risk_score = 0.05
            plan.risk_reasons = ["直达方案，风险极低"]
            return plan

        wait = plan.transfer_wait_minutes or 0
        hub  = plan.transfer_station or ""

        # 1. 换乘时间风险
        min_safe = HUB_MIN_TRANSFER_MINUTES.get(hub, DEFAULT_MIN_TRANSFER)
        if wait < min_safe:
            risk_score += 0.4
            reasons.append(f"换乘时间仅{wait}分钟，低于{hub}建议最短{min_safe}分钟，风险较高")
        elif wait < min_safe + 15:
            risk_score += 0.2
            reasons.append(f"换乘时间{wait}分钟，略显紧张，建议留意")
        else:
            reasons.append(f"换乘时间{wait}分钟，较为充裕")

        # 2. 高峰期风险（周末/节假日）
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            if dt.weekday() >= 5:  # 周六周日
                risk_score += 0.1
                reasons.append("周末出行，客流较大，建议预留更多换乘时间")
        except Exception:
            pass

        # 3. 座位可用性风险
        if len(plan.available_seats) == 0:
            risk_score += 0.5
            reasons.append("两程均无公共可用座位类型")
        elif len(plan.available_seats) == 1:
            risk_score += 0.1
            reasons.append(f"仅{plan.available_seats[0]}有票，选择余地较小")

        # 4. 跨天风险
        for leg in plan.legs:
            if leg.is_cross_day:
                risk_score += 0.15
                reasons.append(f"{leg.train_no}跨天到达，注意次日衔接")

        # 5. 总时长风险（超过12小时认为疲劳风险高）
        if plan.total_duration_minutes > 720:
            risk_score += 0.1
            reasons.append(f"总行程{plan.total_duration_minutes//60}小时，旅途较长")

        plan.risk_score = min(round(risk_score, 2), 1.0)
        plan.risk_reasons = reasons
        return plan


# ══════════════════════════════════════════════════════════════════════════════
# 4. 多目标评分与重排序
# ══════════════════════════════════════════════════════════════════════════════

class MultiObjectiveRanker:
    """
    多目标重排序：最快 / 最稳 / 最便宜 / 最舒适
    """

    PRIORITY_WEIGHTS = {
        "最快":   {"speed": 0.6, "stable": 0.2, "price": 0.1, "comfort": 0.1},
        "最稳":   {"speed": 0.1, "stable": 0.6, "price": 0.1, "comfort": 0.2},
        "最便宜": {"speed": 0.1, "stable": 0.2, "price": 0.6, "comfort": 0.1},
        "最舒适": {"speed": 0.1, "stable": 0.1, "price": 0.2, "comfort": 0.6},
        None:     {"speed": 0.3, "stable": 0.3, "price": 0.2, "comfort": 0.2},
    }

    SEAT_COMFORT_RANK = {
        "商务座": 1.0, "软卧": 0.85, "一等座": 0.75,
        "软座": 0.6,  "二等座": 0.55, "硬卧": 0.5,
        "硬座": 0.3,  "无座": 0.0,
    }

    def score_plan(self, plan: TransferPlan, all_plans: List["TransferPlan"]) -> Dict[str, float]:
        """计算单个方案的四维归一化分数"""
        # 速度分（用全局最短/最长归一化）
        all_durations = [p.total_duration_minutes for p in all_plans if p.total_duration_minutes > 0]
        min_dur = min(all_durations) if all_durations else 1
        max_dur = max(all_durations) if all_durations else 1
        speed = 1.0 - (plan.total_duration_minutes - min_dur) / max(max_dur - min_dur, 1)

        # 稳定性分（1 - 风险分）
        stable = 1.0 - plan.risk_score

        # 价格分（取两程中最便宜座位）
        min_price = self._get_min_price(plan)
        all_prices = [self._get_min_price(p) for p in all_plans if self._get_min_price(p) > 0]
        min_p = min(all_prices) if all_prices else 1
        max_p = max(all_prices) if all_prices else 1
        price = 1.0 - (min_price - min_p) / max(max_p - min_p, 1)

        # 舒适度分（最高座位等级）
        comfort = self._get_comfort_score(plan)

        return {
            "speed":   round(speed,   3),
            "stable":  round(stable,  3),
            "price":   round(price,   3),
            "comfort": round(comfort, 3),
        }

    def _get_min_price(self, plan: TransferPlan) -> float:
        if not plan.total_price:
            return 9999.0
        return min(plan.total_price.values())

    def _get_comfort_score(self, plan: TransferPlan) -> float:
        best = 0.0
        for seat in plan.available_seats:
            best = max(best, self.SEAT_COMFORT_RANK.get(seat, 0.3))
        return best

    def rank(
        self,
        plans: List[TransferPlan],
        priority: Optional[str] = None,
    ) -> List[TransferPlan]:
        """按优先目标重排序"""
        if not plans:
            return plans

        weights = self.PRIORITY_WEIGHTS.get(priority, self.PRIORITY_WEIGHTS[None])

        # 计算每个方案的单维分数
        for plan in plans:
            plan.scores = self.score_plan(plan, plans)

        # 计算加权综合分
        def composite(plan: TransferPlan) -> float:
            s = plan.scores
            return (weights["speed"]   * s.get("speed",   0) +
                    weights["stable"]  * s.get("stable",  0) +
                    weights["price"]   * s.get("price",   0) +
                    weights["comfort"] * s.get("comfort", 0))

        ranked = sorted(plans, key=composite, reverse=True)
        logger.info(f"[Ranker] 按'{priority}'排序，共{len(ranked)}个方案")
        for i, p in enumerate(ranked[:3]):
            logger.info(
                f"  #{i+1} {p.plan_id} 综合={composite(p):.3f} "
                f"速度={p.scores.get('speed',0):.2f} "
                f"稳定={p.scores.get('stable',0):.2f} "
                f"价格={p.scores.get('price',0):.2f} "
                f"舒适={p.scores.get('comfort',0):.2f}"
            )
        return ranked


# ══════════════════════════════════════════════════════════════════════════════
# 5. Self-Reflection 校验器
# ══════════════════════════════════════════════════════════════════════════════

class ReflectionIssue(BaseModel):
    plan_id: str
    issue_type: str     # time_conflict / no_ticket / transfer_too_short / data_stale
    description: str
    is_fatal: bool      # True=方案不可用，False=警告


class SelfReflector:
    """
    Self-Reflection 校验闭环
    校验维度：
    1. 换乘时间冲突（第一程到达 + 缓冲 > 第二程出发）
    2. 两程是否均有票
    3. 跨天问题（日期是否正确）
    4. 票价合理性检查（兜底防止模拟数据异常）
    """

    def reflect(self, plans: List[TransferPlan], date: str) -> Tuple[List[TransferPlan], List[str]]:
        """
        返回：(通过校验的方案列表, 校验日志)
        """
        passed = []
        log = []

        log.append(f"[Reflection] 开始校验 {len(plans)} 个方案")

        for plan in plans:
            issues = self._check_plan(plan, date)
            fatal_issues = [i for i in issues if i.is_fatal]
            warn_issues  = [i for i in issues if not i.is_fatal]

            if fatal_issues:
                log.append(f"  ❌ {plan.plan_id} 校验不通过: " +
                            "; ".join(i.description for i in fatal_issues))
            else:
                passed.append(plan)
                if warn_issues:
                    log.append(f"  ⚠️  {plan.plan_id} 通过（含警告）: " +
                                "; ".join(i.description for i in warn_issues))
                else:
                    log.append(f"  ✅ {plan.plan_id} 校验通过")

        log.append(f"[Reflection] 校验完成: {len(passed)}/{len(plans)} 个方案通过")
        return passed, log

    def _check_plan(self, plan: TransferPlan, date: str) -> List[ReflectionIssue]:
        issues = []

        # 1. 必须有可用座位
        if not plan.available_seats:
            issues.append(ReflectionIssue(
                plan_id=plan.plan_id,
                issue_type="no_ticket",
                description="无可用座位类型（两程无公共有票座位）",
                is_fatal=True
            ))

        # 2. 中转方案：换乘时间检查
        if plan.route_type == RouteType.TRANSFER and len(plan.legs) == 2:
            leg1, leg2 = plan.legs[0], plan.legs[1]
            hub = plan.transfer_station or ""
            min_safe = HUB_MIN_TRANSFER_MINUTES.get(hub, DEFAULT_MIN_TRANSFER)

            # 解析到达/出发时间
            try:
                arr_h, arr_m = map(int, leg1.arrive_time.split(":"))
                dep_h, dep_m = map(int, leg2.depart_time.split(":"))
                arr_total = arr_h * 60 + arr_m
                dep_total = dep_h * 60 + dep_m
                if leg1.is_cross_day:
                    arr_total += 1440  # 次日到达，加24小时

                wait = dep_total - arr_total
                if wait < 0:
                    wait += 1440  # 跨天处理

                plan.transfer_wait_minutes = wait

                if wait < 0:
                    issues.append(ReflectionIssue(
                        plan_id=plan.plan_id,
                        issue_type="time_conflict",
                        description=f"第二程({leg2.train_no})出发时间早于第一程({leg1.train_no})到达时间，时间冲突",
                        is_fatal=True
                    ))
                elif wait < min_safe:
                    issues.append(ReflectionIssue(
                        plan_id=plan.plan_id,
                        issue_type="transfer_too_short",
                        description=f"换乘时间仅{wait}分钟，{hub}建议最少{min_safe}分钟",
                        is_fatal=wait < 15  # 小于15分钟直接致命
                    ))
            except (ValueError, AttributeError) as e:
                issues.append(ReflectionIssue(
                    plan_id=plan.plan_id,
                    issue_type="data_stale",
                    description=f"时间解析失败: {e}",
                    is_fatal=False
                ))

        # 3. 票价合理性检查（防止通用模拟数据生成异常低价）
        for seat, price in plan.total_price.items():
            if price < 10:
                issues.append(ReflectionIssue(
                    plan_id=plan.plan_id,
                    issue_type="data_stale",
                    description=f"票价异常偏低({seat}:{price}元)，可能是模拟数据错误",
                    is_fatal=False
                ))

        return issues


# ══════════════════════════════════════════════════════════════════════════════
# 6. 查询缓存（解决模块二中中转查询重复请求问题）
# ══════════════════════════════════════════════════════════════════════════════

class QueryCache:
    def __init__(self):
        self._cache: Dict[str, QueryResult] = {}

    def _key(self, from_s: str, to_s: str, date: str) -> str:
        return f"{from_s}|{to_s}|{date}"

    def get(self, from_s: str, to_s: str, date: str) -> Optional[QueryResult]:
        result = self._cache.get(self._key(from_s, to_s, date))
        if result:
            logger.info(f"[Cache] 命中缓存: {from_s}→{to_s} {date}")
        return result

    def set(self, from_s: str, to_s: str, date: str, result: QueryResult):
        self._cache[self._key(from_s, to_s, date)] = result
        logger.info(f"[Cache] 写入缓存: {from_s}→{to_s} {date} ({len(result.trains)}个车次)")


# ══════════════════════════════════════════════════════════════════════════════
# 7. 核心 Agent 规划器
# ══════════════════════════════════════════════════════════════════════════════

class RailwayAgent:
    """
    铁路中转方案规划 Agent
    流程：
    1. 解析意图（模块一）
    2. RAG 检索换乘知识（模块三）
    3. 查询直达票（模块二）
    4. 若无直达票/直达票全满：动态生成候选中转枢纽
    5. 并行查询各中转方案（模块二，带缓存）
    6. 风险评估
    7. Self-Reflection 校验（最多 MAX_REFLECT_ROUNDS 轮）
    8. 多目标重排序
    9. LLM 生成推荐语（可选，网络不通则规则生成）
    """

    MAX_REFLECT_ROUNDS = 2
    MAX_PLANS_PER_HUB  = 3   # 每个中转枢纽最多保留N个方案

    def __init__(self):
        self.intent_parser  = IntentParser()
        self.ticket_tool    = TicketQueryTool(use_real_api=True)
        self.rag_retriever  = HybridRetriever(KNOWLEDGE_DOCS)
        self.risk_evaluator = RiskEvaluator()
        self.ranker         = MultiObjectiveRanker()
        self.reflector      = SelfReflector()
        self.cache          = QueryCache()
        logger.info("[Agent] 初始化完成")

    # ── 公共入口 ──────────────────────────────────────────────────────────────

    def plan(self, user_query: str) -> PlanningResult:
        """
        主入口：接收用户自然语言查询，返回规划结果
        """
        logger.info(f"\n{'='*65}")
        logger.info(f"[Agent] 开始规划: '{user_query}'")

        result = PlanningResult(query=user_query, origin="", destination="", date="")

        # Step 1: 意图解析
        slots = self.intent_parser.parse(user_query)
        if not slots.parse_success:
            result.success = False
            result.error_msg = "意图解析失败"
            return result

        origin_name = slots.origin.primary.station_name if slots.origin and slots.origin.primary else None
        dest_name   = slots.destination.primary.station_name if slots.destination and slots.destination.primary else None
        date        = slots.travel_date.date_str if slots.travel_date else datetime.now().strftime("%Y-%m-%d")
        preference  = slots.preference

        if not origin_name or not dest_name:
            result.success = False
            result.error_msg = f"出发站或目的地无法识别 (origin={origin_name}, dest={dest_name})"
            return result

        result.origin = origin_name
        result.destination = dest_name
        result.date = date

        logger.info(f"[Agent] 解析结果: {origin_name} → {dest_name}  {date}  偏好={preference.priority}")

        # Step 2: RAG 检索相关知识
        rag_context = self.rag_retriever.format_context(
            f"{origin_name}到{dest_name}中转换乘", top_k=2
        )
        logger.info(f"[Agent] RAG 知识检索完成")

        # Step 3: 查询直达票
        direct_result = self._cached_query(origin_name, dest_name, date)
        direct_plans  = self._build_direct_plans(direct_result)
        logger.info(f"[Agent] 直达方案: {len(direct_plans)} 个")

        # Step 4: 生成中转方案
        transfer_plans = self._generate_transfer_plans(
            origin_name, dest_name, date, preference
        )
        logger.info(f"[Agent] 中转方案（校验前）: {len(transfer_plans)} 个")

        all_plans = direct_plans + transfer_plans

        # Step 5-7: Self-Reflection 循环
        all_plans, reflection_log = self._reflection_loop(
            all_plans, date, origin_name, dest_name, preference
        )
        result.reflection_log = reflection_log
        result.planning_iterations = self.MAX_REFLECT_ROUNDS

        # Step 8: 风险评估 + 多目标排序
        for plan in all_plans:
            self.risk_evaluator.evaluate(plan, date)

        all_plans = self.ranker.rank(all_plans, preference.priority)

        # Step 9: 生成推荐语
        result.plans = all_plans[:6]  # 最多返回6个方案
        result.final_recommendation = self._generate_recommendation(
            result, rag_context, preference, slots
        )

        logger.info(f"[Agent] 规划完成，返回 {len(result.plans)} 个方案")
        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _cached_query(self, from_s: str, to_s: str, date: str) -> QueryResult:
        cached = self.cache.get(from_s, to_s, date)
        if cached:
            return cached
        result = self.ticket_tool.query(from_s, to_s, date)
        self.cache.set(from_s, to_s, date, result)
        return result

    def _build_direct_plans(self, query_result: QueryResult) -> List[TransferPlan]:
        plans = []
        for i, train in enumerate(query_result.trains):
            avail_seats = [s.seat_type for s in train.seats
                           if s.status in (TicketStatus.AVAILABLE, TicketStatus.FEW_LEFT)]
            total_price = {s.seat_type: s.price for s in train.seats if s.price}

            plan = TransferPlan(
                plan_id=f"direct_{i+1}_{train.train_no}",
                route_type=RouteType.DIRECT,
                legs=[train],
                total_duration_minutes=train.duration_minutes,
                total_price=total_price,
                available_seats=avail_seats,
                data_source=train.from_data_source,
            )
            plans.append(plan)
        return plans

    def _generate_transfer_plans(
        self,
        origin: str,
        destination: str,
        date: str,
        preference: TravelPreference,
    ) -> List[TransferPlan]:
        """动态生成候选中转方案"""
        hubs = get_candidate_hubs(origin, destination)
        plans = []
        plan_counter = 0

        for hub in hubs[:3]:  # 最多查3个枢纽
            logger.info(f"[Agent] 查询中转枢纽: {hub}")

            leg1_result = self._cached_query(origin, hub, date)
            leg2_result = self._cached_query(hub, destination, date)

            if not leg1_result.trains or not leg2_result.trains:
                logger.warning(f"[Agent] {hub} 中转无车次，跳过")
                continue

            hub_plans = self._combine_legs(
                leg1_result.trains, leg2_result.trains,
                hub, date, plan_counter
            )
            plans.extend(hub_plans[:self.MAX_PLANS_PER_HUB])
            plan_counter += len(hub_plans[:self.MAX_PLANS_PER_HUB])

        return plans

    def _combine_legs(
        self,
        leg1_trains: List[TrainTicket],
        leg2_trains: List[TrainTicket],
        hub: str,
        date: str,
        id_offset: int,
    ) -> List[TransferPlan]:
        """将第一程和第二程组合为中转方案"""
        min_transfer = HUB_MIN_TRANSFER_MINUTES.get(hub, DEFAULT_MIN_TRANSFER)
        plans = []

        for t1 in leg1_trains:
            avail1 = {s.seat_type for s in t1.seats
                      if s.status in (TicketStatus.AVAILABLE, TicketStatus.FEW_LEFT)}
            if not avail1:
                continue

            arr_parts = t1.arrive_time.split(":")
            if len(arr_parts) != 2:
                continue
            arr_min = int(arr_parts[0]) * 60 + int(arr_parts[1])
            if t1.is_cross_day:
                arr_min += 1440

            for t2 in leg2_trains:
                avail2 = {s.seat_type for s in t2.seats
                          if s.status in (TicketStatus.AVAILABLE, TicketStatus.FEW_LEFT)}
                if not avail2:
                    continue

                dep_parts = t2.depart_time.split(":")
                if len(dep_parts) != 2:
                    continue
                dep_min = int(dep_parts[0]) * 60 + int(dep_parts[1])

                wait = dep_min - arr_min
                if wait < 0:
                    wait += 1440

                if wait < min_transfer or wait > 240:  # 换乘时间 [min, 4h]
                    continue

                # 公共有票座位
                common_seats = sorted(avail1 & avail2,
                                      key=lambda s: MultiObjectiveRanker.SEAT_COMFORT_RANK.get(s, 0),
                                      reverse=True)
                if not common_seats:
                    continue

                # 票价汇总（取两程最便宜组合）
                total_price: Dict[str, float] = {}
                for seat in common_seats:
                    p1 = next((s.price for s in t1.seats if s.seat_type == seat and s.price), 0)
                    p2 = next((s.price for s in t2.seats if s.seat_type == seat and s.price), 0)
                    if p1 and p2:
                        total_price[seat] = p1 + p2

                total_dur = t1.duration_minutes + wait + t2.duration_minutes

                plan_id = f"transfer_{id_offset + len(plans) + 1}_{hub}_{t1.train_no}_{t2.train_no}"
                plans.append(TransferPlan(
                    plan_id=plan_id,
                    route_type=RouteType.TRANSFER,
                    legs=[t1, t2],
                    transfer_station=hub,
                    transfer_wait_minutes=wait,
                    total_duration_minutes=total_dur,
                    total_price=total_price,
                    available_seats=common_seats,
                    data_source=t1.from_data_source,
                ))

        logger.info(f"[Agent] {hub} 组合出 {len(plans)} 个方案")
        return plans

    def _reflection_loop(
        self,
        plans: List[TransferPlan],
        date: str,
        origin: str,
        destination: str,
        preference: TravelPreference,
    ) -> Tuple[List[TransferPlan], List[str]]:
        """Self-Reflection 校验闭环，最多 MAX_REFLECT_ROUNDS 轮"""
        all_log = []

        for round_num in range(1, self.MAX_REFLECT_ROUNDS + 1):
            all_log.append(f"\n[Reflection Round {round_num}]")
            passed, log = self.reflector.reflect(plans, date)
            all_log.extend(log)

            rejected_count = len(plans) - len(passed)

            if rejected_count == 0:
                all_log.append(f"  → 全部通过，无需修正")
                plans = passed
                break

            all_log.append(f"  → {rejected_count} 个方案被淘汰，尝试修正...")

            # 修正策略：如果中转方案因换乘时间不足被淘汰，尝试扩展最短换乘时间后重新组合
            # 本轮先用通过的方案继续
            plans = passed

            if len(plans) < 3 and round_num < self.MAX_REFLECT_ROUNDS:
                all_log.append(f"  → 有效方案不足3个，扩大搜索范围（放宽换乘时间限制）")
                # 放宽换乘时间重新搜索（最少20分钟）
                extra = self._generate_transfer_plans_relaxed(origin, destination, date, min_transfer=20)
                extra_passed, extra_log = self.reflector.reflect(extra, date)
                all_log.extend(extra_log)
                plans = plans + extra_passed
                all_log.append(f"  → 补充 {len(extra_passed)} 个宽松方案")

        all_log.append(f"\n[Reflection] 最终有效方案: {len(plans)} 个")
        return plans, all_log

    def _generate_transfer_plans_relaxed(
        self, origin: str, destination: str, date: str, min_transfer: int = 20
    ) -> List[TransferPlan]:
        """放宽换乘时间限制的中转方案搜索"""
        hubs = get_candidate_hubs(origin, destination)
        plans = []
        for hub in hubs[:2]:
            leg1 = self._cached_query(origin, hub, date)
            leg2 = self._cached_query(hub, destination, date)
            if not leg1.trains or not leg2.trains:
                continue
            # 临时覆盖最小换乘时间
            original = HUB_MIN_TRANSFER_MINUTES.get(hub, DEFAULT_MIN_TRANSFER)
            HUB_MIN_TRANSFER_MINUTES[hub] = min_transfer
            hub_plans = self._combine_legs(leg1.trains, leg2.trains, hub, date, len(plans))
            HUB_MIN_TRANSFER_MINUTES[hub] = original
            plans.extend(hub_plans[:2])
        return plans

    def _generate_recommendation(
        self,
        result: PlanningResult,
        rag_context: str,
        preference: TravelPreference,
        slots: IntentSlots,
    ) -> str:
        """生成最终推荐语（LLM 优先，规则兜底）"""
        if not result.plans:
            return f"抱歉，未能找到 {result.origin} → {result.destination} {result.date} 的可用方案。"

        best = result.plans[0]

        # 尝试 LLM 生成
        if zhipu_client:
            try:
                return self._llm_recommendation(result, rag_context, preference, best)
            except Exception as e:
                logger.warning(f"[Agent] LLM 推荐生成失败: {e}，使用规则生成")

        # 规则生成
        return self._rule_recommendation(result, best, preference)

    def _llm_recommendation(
        self, result: PlanningResult, rag_context: str,
        preference: TravelPreference, best: TransferPlan
    ) -> str:
        """调用 LLM 生成推荐文本"""
        plans_summary = []
        for i, p in enumerate(result.plans[:3], 1):
            if p.route_type == RouteType.DIRECT:
                desc = f"直达 {p.legs[0].train_no}（{p.legs[0].depart_time}→{p.legs[0].arrive_time}，历时{p.total_duration_minutes//60}h{p.total_duration_minutes%60}min）"
            else:
                desc = (f"中转{p.transfer_station}: "
                        f"{p.legs[0].train_no}({p.legs[0].depart_time})→"
                        f"等{p.transfer_wait_minutes}min→"
                        f"{p.legs[1].train_no}({p.legs[1].depart_time}→{p.legs[1].arrive_time})")
            price_str = "、".join(f"{k}¥{v:.0f}" for k, v in list(p.total_price.items())[:2])
            plans_summary.append(f"方案{i}：{desc}  票价：{price_str}  风险：{p.risk_score}")

        prompt = f"""你是铁路出行智能助手，请根据以下信息给出简洁友好的出行建议（200字以内）。

用户需求：{result.origin} → {result.destination}，{result.date}出发，偏好：{preference.priority or '综合最优'}

可选方案：
{"；".join(plans_summary)}

{rag_context[:300]}

请直接给出推荐，说明推荐理由，并提醒关键注意事项。"""

        resp = zhipu_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

    def _rule_recommendation(
        self, result: PlanningResult,
        best: TransferPlan, preference: TravelPreference
    ) -> str:
        """规则生成推荐文本"""
        lines = [f"📍 {result.origin} → {result.destination}  {result.date}\n"]
        lines.append(f"为您找到 {len(result.plans)} 个方案，推荐：\n")

        if best.route_type == RouteType.DIRECT:
            lines.append(
                f"🚄 【直达】{best.legs[0].train_no}  "
                f"{best.legs[0].depart_time} → {best.legs[0].arrive_time}  "
                f"历时 {best.total_duration_minutes//60}h{best.total_duration_minutes%60}min"
            )
        else:
            l1, l2 = best.legs[0], best.legs[1]
            lines.append(
                f"🔄 【中转{best.transfer_station}】\n"
                f"   第一程: {l1.train_no}  {l1.depart_time} → {l1.arrive_time}  抵达{best.transfer_station}\n"
                f"   换乘等待: {best.transfer_wait_minutes} 分钟\n"
                f"   第二程: {l2.train_no}  {l2.depart_time} → {l2.arrive_time}  抵达{result.destination}"
            )

        if best.available_seats:
            seat = best.available_seats[0]
            price = best.total_price.get(seat)
            price_str = f"  票价: ¥{price:.0f}" if price else ""
            lines.append(f"   推荐座位: {seat}{price_str}")

        if best.risk_reasons:
            lines.append(f"\n⚠️  注意: {best.risk_reasons[0]}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 8. 格式化输出
# ══════════════════════════════════════════════════════════════════════════════

def format_planning_result(result: PlanningResult) -> str:
    lines = []
    lines.append(f"\n{'='*68}")
    lines.append(f"  规划结果: {result.origin} → {result.destination}  {result.date}")
    lines.append(f"  共 {len(result.plans)} 个有效方案")
    lines.append(f"{'='*68}")

    if not result.success:
        lines.append(f"  ❌ 规划失败: {result.error_msg}")
        return "\n".join(lines)

    for i, plan in enumerate(result.plans, 1):
        tag = "🥇" if i == 1 else ("🥈" if i == 2 else f" {i} ")
        lines.append(f"\n{tag} 方案{i} [{plan.route_type.value}]  ID: {plan.plan_id}")
        lines.append(f"   总时长: {plan.total_duration_minutes//60}h{plan.total_duration_minutes%60}min  "
                     f"风险: {plan.risk_score}  数据: {plan.data_source}")

        for j, leg in enumerate(plan.legs, 1):
            cross = "(+1天)" if leg.is_cross_day else ""
            lines.append(f"   第{j}程: {leg.train_no} {leg.train_type}  "
                         f"{leg.from_station}→{leg.to_station}  "
                         f"{leg.depart_time}→{leg.arrive_time}{cross}")

        if plan.route_type == RouteType.TRANSFER:
            lines.append(f"   换乘站: {plan.transfer_station}  等待: {plan.transfer_wait_minutes}分钟")

        seats_str = "、".join(plan.available_seats[:3])
        lines.append(f"   有票座位: {seats_str}")

        price_items = list(plan.total_price.items())[:3]
        if price_items:
            price_str = "  ".join(f"{k}¥{v:.0f}" for k, v in price_items)
            lines.append(f"   票价合计: {price_str}")

        score_str = "  ".join(f"{k}={v:.2f}" for k, v in plan.scores.items())
        lines.append(f"   评分: {score_str}")

        if plan.risk_reasons:
            lines.append(f"   风险说明: {plan.risk_reasons[0]}")

    lines.append(f"\n{'─'*68}")
    lines.append("  Self-Reflection 日志:")
    for log_line in result.reflection_log:
        lines.append(f"  {log_line}")

    lines.append(f"\n{'─'*68}")
    lines.append("  【最终推荐】")
    lines.append(f"  {result.final_recommendation}")
    lines.append(f"{'='*68}\n")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 9. 测试入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚄 模块四：Agent 规划层 - 测试\n")
    print("初始化 Agent（构建 RAG 索引）...\n")

    agent = RailwayAgent()

    test_queries = [
        "北京去广州明天出发，直达票没有了，帮我规划中转方案，最稳的",
        "上海到成都后天出发，要二等座，最便宜的方案",
        "北京西到深圳北下周一，最快的",
    ]

    for query in test_queries:
        print(f"\n{'#'*68}")
        print(f"  用户查询: {query}")
        print(f"{'#'*68}")
        result = agent.plan(query)
        print(format_planning_result(result))