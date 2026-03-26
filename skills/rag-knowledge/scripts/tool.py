"""
Skill: 铁路知识库检索 (RAG)
BM25 + 向量相似度混合检索，返回铁路领域知识片段。
"""

import json
import math
import logging
import re
from typing import List, Tuple, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("skill.rag_knowledge")

# ══════════════════════════════════════════════════════════════════════════════
# 知识库文档
# ══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_DOCS = [
    {
        "id": "route_001", "category": "route", "title": "京沪高铁",
        "content": "京沪高铁全长1318公里，连接北京南站与上海虹桥站，途经天津南、济南西、南京南。"
                   "全程最快约4小时28分，票价：商务座约1748元，一等座约933元，二等座约553元。"
                   "节假日和周末票源紧张，建议提前15天购票。"
                   "主要中转枢纽：南京南（可转合肥、杭州方向），济南西（可转青岛方向）。"
                   "注意：北京南站是始发站，北京站和北京西站不发京沪高铁。",
        "tags": ["京沪", "北京南", "上海虹桥", "高铁", "G字头"],
    },
    {
        "id": "route_002", "category": "route", "title": "京广高铁",
        "content": "京广高铁全长2298公里，连接北京西站与广州南站，途经石家庄、郑州东、武汉、长沙南。"
                   "全程最快约8小时。北京至武汉约4小时，武汉至广州约3小时43分。"
                   "注意：北京发往广州南的高铁从北京西站始发，不是北京南站。",
        "tags": ["京广", "北京西", "广州南", "武汉", "郑州东"],
    },
    {
        "id": "hub_001", "category": "hub", "title": "郑州东站换乘",
        "content": "郑州东站是京广高铁和徐兰高铁（郑西高铁）的交汇点，可换乘西安、兰州方向。"
                   "郑州东站规模大，建议预留换乘时间不少于45分钟，高峰期建议60分钟。"
                   "站内有餐饮、候车厅，换乘体验良好。",
        "tags": ["郑州东", "换乘", "京广", "徐兰", "西安"],
    },
    {
        "id": "hub_002", "category": "hub", "title": "武汉站换乘",
        "content": "武汉站是京广高铁重要枢纽，同时连接武九、合武等方向。"
                   "建议换乘时间不少于45分钟，早晚高峰期候车区人流较大，建议预留60分钟。"
                   "武汉站距武昌站约5公里，注意不要混淆。",
        "tags": ["武汉", "换乘", "京广", "枢纽"],
    },
    {
        "id": "hub_003", "category": "hub", "title": "长沙南站换乘",
        "content": "长沙南站是京广高铁重要中间站，也是长昆高铁（贵广方向）的始发站。"
                   "建议换乘时间不少于40分钟，站内设施齐全。",
        "tags": ["长沙南", "换乘", "京广", "贵广"],
    },
    {
        "id": "policy_001", "category": "policy", "title": "12306购票政策",
        "content": "12306实名制购票，需绑定实名身份证。高铁票最早可提前15天购买（节假日前可能延长）。"
                   "开车前30分钟可在线改签，改签后票价差额退补。候补购票功能可在余票紧张时提交候补申请。"
                   "儿童票：1.2米以下免票，1.2-1.5米享半价。",
        "tags": ["12306", "购票", "政策", "改签", "儿童票"],
    },
    {
        "id": "tip_001", "category": "tip", "title": "中转换乘注意事项",
        "content": "中转换乘时需注意：①两段车次必须同站换乘（不能跨站）；②建议换乘时间至少40分钟以上；"
                   "③遇晚点情况可拨打12306热线申请签转；④高峰期（春节、国庆、暑运）建议换乘时间加倍；"
                   "⑤二等座票源较充裕，高铁枢纽站通常可直接进站不需二次安检。",
        "tags": ["中转", "换乘", "注意事项", "晚点", "签转"],
    },
    {
        "id": "tip_002", "category": "tip", "title": "高铁选座建议",
        "content": "高铁座位选择建议：A/F靠窗，C/D靠过道，E/B中间。"
                   "京沪高铁运行方向：去程（北京→上海）D侧看海，F侧看山。"
                   "商务座可自动调节椅背，餐食配送。一等座比二等座宽敞，长途出行推荐。",
        "tags": ["选座", "商务座", "一等座", "二等座", "高铁"],
    },
    {
        "id": "tip_003", "category": "tip", "title": "抢票技巧",
        "content": "热门线路抢票建议：①开售第一时间（早8点）登录12306；②候补功能可作为兜底手段；"
                   "③可尝试中间站上车（如北京→上海满票可试试天津南→上海）；"
                   "④节假日提前30天以上规划行程；⑤二等座比一等座票源通常更多。",
        "tags": ["抢票", "候补", "节假日", "技巧"],
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# BM25 检索（简化版，无需外部依赖）
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """简单 CJK 分词：字级 + 关键词"""
    tokens = list(text)
    # 额外提取 2-gram
    for i in range(len(text) - 1):
        tokens.append(text[i:i+2])
    return tokens

def _bm25_score(query_tokens: List[str], doc_tokens: List[str],
                avg_dl: float, k1=1.5, b=0.75) -> float:
    dl = len(doc_tokens)
    tf_map = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        idf = math.log(1 + (len(KNOWLEDGE_DOCS) - 1 + 0.5) / (1 + 1))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score

def search_knowledge(query: str, top_k: int = 3,
                     category: Optional[str] = None) -> List[dict]:
    docs = KNOWLEDGE_DOCS
    if category:
        docs = [d for d in docs if d["category"] == category]

    query_tokens = _tokenize(query)
    all_tokens = [_tokenize(d["content"] + " " + " ".join(d["tags"])) for d in docs]
    avg_dl = sum(len(t) for t in all_tokens) / max(len(all_tokens), 1)

    scored = []
    for i, doc in enumerate(docs):
        # BM25
        bm25 = _bm25_score(query_tokens, all_tokens[i], avg_dl)
        # tag 精确匹配加分
        tag_bonus = sum(1.5 for tag in doc["tags"] if tag in query)
        scored.append((doc, bm25 + tag_bonus))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"doc": d, "score": round(s, 3)} for d, s in scored[:top_k]]


# ══════════════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeInput(BaseModel):
    query:    str           = Field(description="检索问题，如'郑州换乘注意事项'")
    top_k:    int           = Field(3, description="返回最相关文档数量，1-5")
    category: Optional[str] = Field(None, description="类别过滤：route/hub/policy/tip")

@tool("search_railway_knowledge", args_schema=KnowledgeInput)
def search_railway_knowledge(query: str, top_k: int = 3,
                              category: Optional[str] = None) -> str:
    """
    检索铁路领域知识库，获取线路信息、换乘经验、购票政策等。
    在规划方案前可先检索相关知识，提升推荐质量。
    """
    results = search_knowledge(query, top_k=top_k, category=category)
    output = []
    for r in results:
        d = r["doc"]
        output.append({
            "id": d["id"],
            "category": d["category"],
            "title": d["title"],
            "content": d["content"],
            "tags": d["tags"],
            "relevance_score": r["score"],
        })
    logger.info(f"[Skill:RAG] query='{query}' 返回{len(output)}条知识")
    return json.dumps({"query": query, "results": output}, ensure_ascii=False, indent=2)
