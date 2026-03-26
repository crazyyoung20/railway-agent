---
name: ticket-query
description: >
  查询中国铁路 12306 列车余票信息。当需要查询指定区间、日期的列车余票、
  座位类型、票价时使用本 skill。支持真实 API 调用和 Mock 数据 fallback。
  适用于直达票查询、中转两段票查询。
license: Apache-2.0
metadata:
  author: railway-agent-team
  version: "2.0"
  category: railway
  api_source: 12306
compatibility: "python>=3.9"
allowed_tools:
  - bash
  - python
---

# Ticket Query Skill

## Overview

查询中国铁路余票信息，封装为 LangChain `@tool`，供 ReAct Agent 直接调用。

真实 API 不可用时自动 fallback 到 Mock 数据（覆盖主要干线）。

## Tool Interface

```python
from skills.ticket_query.scripts.tool import query_tickets

# 输入
result = query_tickets.invoke({
    "from_station": "北京西",   # 标准站名
    "to_station":   "广州南",
    "date":         "2026-03-25",  # YYYY-MM-DD
    "train_filter": "G"            # 可选：G/D/K/T，不填则全部
})

# 返回：JSON 字符串
# {
#   "from_station": "北京西",
#   "to_station": "广州南",
#   "date": "2026-03-25",
#   "total": 2,
#   "trains": [
#     {
#       "train_no": "G71",
#       "train_type": "G",
#       "from": "北京西",
#       "to": "广州南",
#       "depart": "08:00",
#       "arrive": "16:00",
#       "duration_min": 480,
#       "is_cross_day": false,
#       "available_seats": ["一等座", "二等座"],
#       "seats": [
#         {"seat": "商务座", "status": "无票", "count": 0, "price": 2630.0},
#         {"seat": "一等座", "status": "有票", "count": 12, "price": 1283.0},
#         {"seat": "二等座", "status": "紧张", "count": 3, "price": 864.0}
#       ],
#       "source": "mock"
#     }
#   ]
# }
```

## When to Use

- 查询出发地→目的地的直达余票
- 中转规划时查询 出发地→中转站、中转站→目的地 两段
- 需要确认某座位类型是否有票时

## Data Sources

| 优先级 | 数据源 | 说明 |
|--------|--------|------|
| 1 | 12306 真实 API | 需网络可达，可能触发反爬 |
| 2 | Mock 数据 | 覆盖京沪、京广等主要干线 |

## Mock 数据覆盖范围

```
北京西 → 广州南      北京南 → 上海虹桥
北京西 → 武汉        武汉  → 广州南
北京西 → 郑州东      郑州东 → 广州南
北京南 → 南京南      南京南 → 上海虹桥
北京西 → 长沙南      长沙南 → 广州南
上海虹桥 → 广州南
```

## Error Handling

- 站名不在 `STATION_CODE_MAP` 中时，仍尝试查询并返回 Mock 数据
- 网络超时（5s）自动切换 Mock
- 返回空列表时 `total=0`，Agent 应尝试其他中转枢纽
