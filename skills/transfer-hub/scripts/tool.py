"""
Skill: 中转枢纽规划
根据出发地和目的地推荐候选中转枢纽，并评估中转风险。
"""

import json
import logging
from datetime import datetime
from typing import List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("skill.transfer_hub")

# ══════════════════════════════════════════════════════════════════════════════
# 枢纽规则
# ══════════════════════════════════════════════════════════════════════════════

HUB_RULES = [
    (["北京"],         ["广州", "深圳", "珠海"],    ["武汉", "郑州东", "长沙南"]),
    (["广州", "深圳"], ["北京"],                    ["武汉", "郑州东", "长沙南"]),
    (["北京"],         ["上海"],                    ["南京南", "济南西"]),
    (["上海"],         ["北京"],                    ["南京南", "济南西"]),
    (["北京"],         ["成都", "重庆"],             ["郑州东", "武汉"]),
    (["成都", "重庆"], ["北京"],                    ["郑州东", "武汉"]),
    (["上海"],         ["成都", "重庆"],             ["武汉", "郑州东"]),
    (["成都", "重庆"], ["上海"],                    ["武汉", "郑州东"]),
    (["北京"],         ["西安", "兰州", "乌鲁木齐"], ["郑州东"]),
    (["上海", "南京"], ["西安", "兰州"],             ["郑州东"]),
    (["广州", "深圳"], ["成都", "重庆"],             ["贵阳北", "长沙南", "武汉"]),
    (["上海"],         ["广州", "深圳"],             ["杭州东", "长沙南"]),
    (["广州", "深圳"], ["上海"],                    ["长沙南", "杭州东"]),
    (["北京"],         ["哈尔滨", "沈阳", "长春"],   ["沈阳北"]),
    (["上海"],         ["昆明", "贵阳"],             ["长沙南", "贵阳北"]),
]

HUB_MIN_TRANSFER = {
    "郑州东": 45, "武汉": 45, "长沙南": 40, "广州南": 40,
    "上海虹桥": 35, "北京南": 40, "北京西": 40,
    "南京南": 35, "成都东": 35, "重庆北": 35,
    "西安北": 35, "济南西": 35, "杭州东": 35,
    "贵阳北": 40, "沈阳北": 35,
}
DEFAULT_MIN_TRANSFER = 40


def get_candidate_hubs(origin: str, destination: str) -> List[str]:
    candidates = []
    for orig_kws, dest_kws, hubs in HUB_RULES:
        if any(kw in origin for kw in orig_kws) and any(kw in destination for kw in dest_kws):
            for hub in hubs:
                if hub not in candidates:
                    candidates.append(hub)
    return candidates or ["郑州东", "武汉", "长沙南"]


def assess_transfer_risk(
    hub: str,
    wait_minutes: int,
    date: str,
    has_common_seats: bool,
) -> dict:
    """评估单个中转方案的风险"""
    risk = 0.0
    reasons = []
    min_safe = HUB_MIN_TRANSFER.get(hub, DEFAULT_MIN_TRANSFER)

    if wait_minutes < min_safe:
        risk += 0.4
        reasons.append(f"换乘时间仅{wait_minutes}分钟，{hub}建议最少{min_safe}分钟")
    elif wait_minutes < min_safe + 15:
        risk += 0.15
        reasons.append(f"换乘{wait_minutes}分钟，略显紧张")
    else:
        reasons.append(f"换乘{wait_minutes}分钟，时间充裕")

    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        if dt.weekday() >= 5:
            risk += 0.1
            reasons.append("周末出行，客流较大")
    except Exception:
        pass

    if not has_common_seats:
        risk += 0.5
        reasons.append("两程无公共有票座位类型")

    return {
        "risk_score": round(min(risk, 1.0), 2),
        "risk_level": "高" if risk > 0.5 else ("中" if risk > 0.2 else "低"),
        "reasons": reasons,
        "min_safe_transfer_min": min_safe,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════════════

class HubQueryInput(BaseModel):
    origin:      str = Field(description="出发城市或站名，如'北京'")
    destination: str = Field(description="目的地城市或站名，如'广州'")

@tool("get_transfer_hubs", args_schema=HubQueryInput)
def get_transfer_hubs(origin: str, destination: str) -> str:
    """
    根据出发地和目的地，获取推荐的中转枢纽列表。
    需要中转时，先调用此工具获取候选枢纽，再逐一查询票务。
    """
    hubs = get_candidate_hubs(origin, destination)
    result = {
        "origin": origin,
        "destination": destination,
        "recommended_hubs": hubs,
        "hub_min_transfer": {h: HUB_MIN_TRANSFER.get(h, DEFAULT_MIN_TRANSFER) for h in hubs},
        "note": "建议按推荐顺序优先查询，每个枢纽需分别查询两段票务",
    }
    logger.info(f"[Skill:TransferHub] {origin}→{destination} 推荐枢纽: {hubs}")
    return json.dumps(result, ensure_ascii=False)


class RiskInput(BaseModel):
    hub:               str  = Field(description="中转站名，如'武汉'")
    wait_minutes:      int  = Field(description="换乘等待时间（分钟）")
    date:              str  = Field(description="出行日期 YYYY-MM-DD")
    has_common_seats:  bool = Field(description="两程是否有公共有票座位")

@tool("assess_transfer_risk", args_schema=RiskInput)
def assess_transfer_risk_tool(
    hub: str, wait_minutes: int, date: str, has_common_seats: bool
) -> str:
    """
    评估中转方案的风险等级。
    在完成两段票务查询并确定换乘时间后，调用此工具进行风险评估。
    返回风险分、风险等级和具体原因。
    """
    result = assess_transfer_risk(hub, wait_minutes, date, has_common_seats)
    result["hub"] = hub
    result["wait_minutes"] = wait_minutes
    logger.info(f"[Skill:Risk] {hub} 等{wait_minutes}min → 风险={result['risk_level']}")
    return json.dumps(result, ensure_ascii=False)
