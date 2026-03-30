"""
Skill: 站名标准化
将用户输入的城市/站名模糊表达转换为 12306 标准站名。
"""

import json
import logging
from typing import Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("skill.station_normalizer")

# ══════════════════════════════════════════════════════════════════════════════
# 站名知识库（覆盖全国主要车站）
# ══════════════════════════════════════════════════════════════════════════════

STATION_DB = {
    "北京":   [("北京南", "高铁首选"), ("北京西", "普速/部分高铁"), ("北京", "普速"), ("北京北", "少量列车")],
    "上海":   [("上海虹桥", "高铁首选"), ("上海", "普速/城际")],
    "广州":   [("广州南", "高铁首选"), ("广州", "普速"), ("广州东", "部分城际")],
    "深圳":   [("深圳北", "高铁首选"), ("深圳", "普速")],
    "成都":   [("成都东", "高铁首选"), ("成都", "普速")],
    "重庆":   [("重庆北", "高铁首选"), ("重庆", "普速")],
    "西安":   [("西安北", "高铁首选"), ("西安", "普速")],
    "武汉":   [("武汉", "高铁/普速")],
    "郑州":   [("郑州东", "高铁首选"), ("郑州", "普速")],
    "长沙":   [("长沙南", "高铁首选"), ("长沙", "普速")],
    "杭州":   [("杭州东", "高铁首选"), ("杭州", "普速")],
    "南京":   [("南京南", "高铁首选"), ("南京", "普速")],
    "济南":   [("济南西", "高铁首选"), ("济南", "普速")],
    "沈阳":   [("沈阳北", "高铁首选"), ("沈阳", "普速")],
    "哈尔滨": [("哈尔滨西", "高铁"), ("哈尔滨", "普速")],
    "贵阳":   [("贵阳北", "高铁首选"), ("贵阳", "普速")],
    "昆明":   [("昆明南", "高铁"), ("昆明", "普速")],
    "太原":   [("太原南", "高铁"), ("太原", "普速")],
    "天津":   [("天津南", "高铁"), ("天津", "普速")],
    "石家庄": [("石家庄", "高铁/普速")],
    "合肥":   [("合肥南", "高铁首选"), ("合肥", "普速")],
    "福州":   [("福州南", "高铁首选"), ("福州", "普速")],
    "厦门":   [("厦门北", "高铁首选"), ("厦门", "普速")],
    "南昌":   [("南昌西", "高铁首选"), ("南昌", "普速")],
    "南宁":   [("南宁东", "高铁首选"), ("南宁", "普速")],
    # 精确站名直通
    "北京南":   [("北京南", "精确")],
    "北京西":   [("北京西", "精确")],
    "北京北":   [("北京北", "精确")],
    "上海虹桥": [("上海虹桥", "精确")],
    "广州南":   [("广州南", "精确")],
    "深圳北":   [("深圳北", "精确")],
    "成都东":   [("成都东", "精确")],
    "重庆北":   [("重庆北", "精确")],
    "西安北":   [("西安北", "精确")],
    "郑州东":   [("郑州东", "精确")],
    "长沙南":   [("长沙南", "精确")],
    "杭州东":   [("杭州东", "精确")],
    "南京南":   [("南京南", "精确")],
    "济南西":   [("济南西", "精确")],
    "沈阳北":   [("沈阳北", "精确")],
    "贵阳北":   [("贵阳北", "精确")],
    "合肥南":   [("合肥南", "精确")],
    "福州南":   [("福州南", "精确")],
    "厦门北":   [("厦门北", "精确")],
    "南昌西":   [("南昌西", "精确")],
    "南宁东":   [("南宁东", "精确")],
}

def _normalize(raw: str) -> dict:
    """核心标准化逻辑"""
    raw = raw.strip().replace("站", "")

    # 精确匹配
    if raw in STATION_DB:
        candidates = STATION_DB[raw]
        return {
            "raw_input": raw,
            "primary": candidates[0][0],
            "candidates": [{"station": s, "note": n} for s, n in candidates],
            "is_ambiguous": len(candidates) > 1,
            "confidence": 1.0,
        }

    # 模糊匹配：用户输入是已知站名的子串
    for key, candidates in STATION_DB.items():
        if key in raw or raw in key:
            return {
                "raw_input": raw,
                "primary": candidates[0][0],
                "candidates": [{"station": s, "note": n} for s, n in candidates],
                "is_ambiguous": len(candidates) > 1,
                "confidence": 0.85,
            }

    return {
        "raw_input": raw,
        "primary": raw,  # 原样返回，由 agent 判断
        "candidates": [{"station": raw, "note": "未知站名，原样使用"}],
        "is_ambiguous": False,
        "confidence": 0.3,
    }


class StationInput(BaseModel):
    station_name: str = Field(description="用户输入的城市或站名，如'北京'、'广州南'")

@tool("normalize_station", args_schema=StationInput)
def normalize_station(station_name: str) -> str:
    """
    将用户输入的城市/站名转换为 12306 标准站名。
    当用户说"北京"时，会返回"北京南"（高铁首选）等候选站名。
    规划出行方案前，必须先调用此工具标准化出发地和目的地。
    """
    result = _normalize(station_name)
    logger.info(f"[Skill:StationNorm] '{station_name}' → '{result['primary']}' (conf={result['confidence']})")
    return json.dumps(result, ensure_ascii=False)
