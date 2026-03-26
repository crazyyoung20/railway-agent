---
name: rag-knowledge
description: >
  铁路领域知识库检索（RAG）。检索线路信息、换乘枢纽经验、
  12306 购票政策、出行技巧等结构化知识片段。
  在规划方案前检索相关知识可显著提升推荐质量。
license: Apache-2.0
metadata:
  author: railway-agent-team
  version: "2.0"
  category: railway
  retrieval: "BM25 + tag exact match"
  doc_count: "8"
compatibility: "python>=3.9"
allowed_tools:
  - bash
  - python
---

# RAG Knowledge Skill

## Overview

混合检索（BM25 关键词 + 标签精确匹配）铁路领域知识库，
无需外部向量数据库，纯 Python 实现。

## Tool Interface

```python
from skills.rag_knowledge.scripts.tool import search_railway_knowledge

result = search_railway_knowledge.invoke({
    "query":    "郑州换乘注意事项",
    "top_k":    3,           # 返回文档数，1-5
    "category": "hub"        # 可选过滤：route / hub / policy / tip
})
# 返回：
# {
#   "query": "郑州换乘注意事项",
#   "results": [
#     {
#       "id": "hub_001",
#       "category": "hub",
#       "title": "郑州东站换乘",
#       "content": "郑州东站是京广高铁和徐兰高铁的交汇点...",
#       "tags": ["郑州东", "换乘", "京广", "徐兰"],
#       "relevance_score": 4.2
#     }
#   ]
# }
```

## Knowledge Categories

| category | 内容 | 文档数 |
|----------|------|--------|
| `route` | 线路知识（里程、时长、票价、主要车站） | 2 |
| `hub` | 换乘枢纽经验（换乘时间、注意事项） | 3 |
| `policy` | 12306 购票政策（实名制、改签、候补） | 1 |
| `tip` | 出行技巧（抢票、选座、中转注意事项） | 2 |

## Knowledge Documents

- `route_001` 京沪高铁（北京南↔上海虹桥，票价、换乘枢纽）
- `route_002` 京广高铁（北京西↔广州南，途经站、时长）
- `hub_001` 郑州东站换乘（建议45min+，与徐兰高铁交汇）
- `hub_002` 武汉站换乘（建议45min+，客流高峰注意事项）
- `hub_003` 长沙南站换乘（建议40min+，长昆高铁始发）
- `policy_001` 12306购票政策（实名制、改签规则、儿童票）
- `tip_001` 中转换乘注意事项（签转、高峰期加倍等）
- `tip_002` 高铁选座建议（A/F靠窗、商务座功能）

## When to Use

建议在以下场景检索：
1. 规划方案前：了解线路基本信息
2. 推荐枢纽前：查询换乘站注意事项
3. 生成推荐语时：补充政策和技巧信息

## Adding New Documents

在 `references/knowledge_docs.py` 中按 `KnowledgeDoc` 格式添加文档，
重启 Agent 后自动加入检索索引。
