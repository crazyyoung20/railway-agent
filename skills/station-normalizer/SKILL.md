---
name: station-normalizer
description: >
  将用户输入的城市名或模糊站名标准化为 12306 标准站名。
  当用户说"北京"时返回"北京南"（高铁首选）等候选项。
  在规划任何出行方案前必须先调用本 skill 完成站名标准化。
license: Apache-2.0
metadata:
  author: railway-agent-team
  version: "2.0"
  category: railway
  coverage: "主要高铁城市 30+"
compatibility: "python>=3.9"
allowed_tools:
  - bash
  - python
---

# Station Normalizer Skill

## Overview

站名标准化：模糊城市名 → 12306 标准站名，解决"北京"对应多个车站的歧义问题。

## Tool Interface

```python
from skills.station_normalizer.scripts.tool import normalize_station

result = normalize_station.invoke({"station_name": "北京"})
# 返回 JSON 字符串：
# {
#   "raw_input": "北京",
#   "primary": "北京南",          ← 首选站（高铁优先）
#   "candidates": [
#     {"station": "北京南",  "note": "高铁首选"},
#     {"station": "北京西",  "note": "普速/部分高铁"},
#     {"station": "北京",    "note": "普速"},
#     {"station": "北京北",  "note": "少量列车"}
#   ],
#   "is_ambiguous": true,
#   "confidence": 1.0
# }
```

## Disambiguation Rules

| 用户输入 | primary 站名 | 说明 |
|----------|-------------|------|
| 北京 | 北京南 | 京沪/京广高铁始发站 |
| 广州 | 广州南 | 京广高铁终点站 |
| 上海 | 上海虹桥 | 高铁主站 |
| 深圳 | 深圳北 | 高铁主站 |
| 北京西 | 北京西 | 精确匹配，直接返回 |

## When to Use

在调用 `query_tickets` 之前，必须先通过本 skill 确认标准站名，
否则票务查询可能因站名不匹配而返回空结果。

## Confidence Levels

- `1.0`：精确匹配
- `0.85`：模糊匹配（子串命中）
- `0.3`：未知站名（原样返回，需人工确认）
