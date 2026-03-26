---
name: transfer-hub
description: >
  铁路中转枢纽规划与风险评估。根据出发地和目的地推荐候选中转枢纽，
  并对具体中转方案（换乘时间、座位可用性、日期）进行风险评分。
  当直达票不足或需要中转方案时使用本 skill。
license: Apache-2.0
metadata:
  author: railway-agent-team
  version: "2.0"
  category: railway
  hub_count: "15"
compatibility: "python>=3.9"
allowed_tools:
  - bash
  - python
---

# Transfer Hub Skill

## Overview

提供两个工具：

1. `get_transfer_hubs` — 推荐候选中转枢纽
2. `assess_transfer_risk` — 评估换乘风险

## Tool 1：get_transfer_hubs

```python
from skills.transfer_hub.scripts.tool import get_transfer_hubs

result = get_transfer_hubs.invoke({
    "origin":      "北京",
    "destination": "广州"
})
# 返回：
# {
#   "origin": "北京",
#   "destination": "广州",
#   "recommended_hubs": ["武汉", "郑州东", "长沙南"],
#   "hub_min_transfer": {
#     "武汉": 45,
#     "郑州东": 45,
#     "长沙南": 40
#   },
#   "note": "建议按推荐顺序优先查询..."
# }
```

## Tool 2：assess_transfer_risk

```python
from skills.transfer_hub.scripts.tool import assess_transfer_risk_tool

result = assess_transfer_risk_tool.invoke({
    "hub":              "武汉",
    "wait_minutes":     65,
    "date":             "2026-03-25",
    "has_common_seats": True
})
# 返回：
# {
#   "hub": "武汉",
#   "wait_minutes": 65,
#   "risk_score": 0.0,
#   "risk_level": "低",        ← 低 / 中 / 高
#   "reasons": ["换乘65分钟，时间充裕"],
#   "min_safe_transfer_min": 45
# }
```

## Risk Scoring Rules

| 条件 | 风险加分 |
|------|---------|
| 换乘时间 < 枢纽最短安全时间 | +0.4 |
| 换乘时间 < 最短安全时间+15min | +0.15 |
| 周末出行 | +0.1 |
| 两程无公共有票座位 | +0.5 |

风险等级：`高 (>0.5)` / `中 (0.2~0.5)` / `低 (<0.2)`

## Hub Coverage

| 枢纽 | 最短换乘（分钟） | 主要连接方向 |
|------|----------------|-------------|
| 郑州东 | 45 | 京广、徐兰 |
| 武汉 | 45 | 京广、武九、合武 |
| 长沙南 | 40 | 京广、长昆 |
| 广州南 | 40 | 京广、贵广、广深 |
| 上海虹桥 | 35 | 京沪、沪昆、沪宁 |
| 北京南 | 40 | 京沪、京津 |
| 北京西 | 40 | 京广、京九 |
| 南京南 | 35 | 京沪、沪汉蓉 |
| 杭州东 | 35 | 京沪、沪昆 |
| 济南西 | 35 | 京沪、石济 |
| 贵阳北 | 40 | 沪昆、贵广 |
| 沈阳北 | 35 | 京哈 |

## Usage Pattern（Agent 调用顺序）

```
1. get_transfer_hubs(origin, destination)
   → 得到 ["武汉", "郑州东", "长沙南"]

2. 对每个 hub：
   query_tickets(origin, hub, date)        # skill: ticket-query
   query_tickets(hub, destination, date)   # skill: ticket-query

3. 组合两段，计算 wait_minutes

4. assess_transfer_risk(hub, wait_minutes, date, has_common_seats)
   → 得到风险评级

5. 按风险从低到高排序，给出推荐
```
