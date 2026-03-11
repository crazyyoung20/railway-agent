# 🚄 铁路出行智能决策系统

基于 RAG + Agent + Self-Reflection 的铁路中转方案规划系统，针对 12306 缺票场景，自动规划多段中转路径并评估方案可行性。

------

## 功能概述

- **自然语言理解**：支持"北京去广州明天出发，最稳的方案"等口语化输入，自动解析出发站、目的站、日期、偏好等槽位
- **中转方案规划**：直达票无票时，自动推荐武汉、郑州东、长沙南等枢纽中转方案
- **余票实时查询**：对接 12306 真实 API，自动处理超时/封禁并降级为本地模拟数据
- **方案校验与修正**：Self-Reflection 闭环检测时间冲突、换乘时间不足等问题，淘汰不可行方案后自动重搜补充
- **多目标排序**：支持最快 / 最稳 / 最便宜 / 最舒适四种偏好维度的方案排序
- **领域知识库**：内置换乘枢纽经验、高铁线路知识、12306 政策等 20 篇文档，通过混合检索增强规划质量

------

## 系统架构

```
用户自然语言输入
        │
        ▼
   意图解析层
   LLM 槽位抽取 + 站名标准化 + 日期解析
        │
        ├──────────────────────┐
        ▼                      ▼
   RAG 知识库层            票务查询工具层
   FAISS + BM25 混合检索   12306 API + Fallback
   换乘经验 / 线路 / 政策
        │                      │
        └──────────┬───────────┘
                   ▼
            Agent 规划层
            候选枢纽生成 → 余票组合
            Self-Reflection 校验闭环
            风险评估 + 多目标重排序
                   │
                   ▼
            FastAPI 接口层
```

------

## Self-Reflection 机制

方案生成后进入校验闭环，**不是简单过滤，而是淘汰后触发修正**：

```
生成方案池
    ↓
校验：换乘时间冲突 / 两程无公共有票座位 / 票价异常
    ↓
有方案被淘汰？
  → YES：放宽换乘约束重新搜索，补充方案，进入下一轮校验
  → NO ：进入多目标排序
```

最多执行 2 轮，每轮日志完整记录在响应的 `reflection_log` 字段中。

------

## 项目结构

```
├── intent_parser.py     # 意图解析：槽位抽取 + 站名标准化 + 日期解析
├── ticket_query.py      # 票务查询：12306 API + 模拟数据 Fallback
├── rag_knowledge.py     # 知识库：FAISS + BM25 混合检索，20 篇领域文档
├── agent_planner.py     # Agent：中转规划 + Self-Reflection + 多目标排序
├── api_server.py        # FastAPI 接口层
└── .env                 # ZHIPUAI_API_KEY=...
```

------

## 快速开始

**环境要求**

```bash
Python 3.9+
pip install fastapi uvicorn langchain faiss-cpu rank-bm25 zhipuai jieba requests pydantic
```

**配置**

```bash
echo "ZHIPUAI_API_KEY=your_key_here" > .env
```

**启动**

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

接口文档：`http://localhost:8000/docs`

------

## API 示例

**自然语言规划**

```bash
curl -X POST http://localhost:8000/api/v1/plan \
  -H "Content-Type: application/json" \
  -d '{"query": "北京去广州明天出发，直达票没有了，帮我规划中转方案，最稳的"}'
```

响应示例（简略）：

```json
{
  "origin": "北京南",
  "destination": "广州南",
  "date": "2026-03-12",
  "plans": [
    {
      "route_type": "中转",
      "legs": [
        {"train_no": "G511", "depart_time": "09:08", "arrive_time": "13:07"},
        {"train_no": "G857", "depart_time": "14:15", "arrive_time": "17:58"}
      ],
      "transfer_station": "武汉",
      "transfer_wait_minutes": 68,
      "available_seats": ["商务座", "一等座"],
      "risk_score": 0.0,
      "scores": {"speed": 0.85, "stable": 1.0, "price": 0.0, "comfort": 1.0}
    }
  ],
  "reflection_log": [
    "[Reflection Round 1] 校验 9 个方案",
    "❌ direct_G79 不通过: 无可用座位",
    "✅ 其余 8 个方案通过"
  ]
}
```

**余票查询**

```bash
curl "http://localhost:8000/api/v1/tickets?from_station=北京南&to_station=上海虹桥&date=2026-03-12"
```

**知识库检索**

```bash
curl "http://localhost:8000/api/v1/knowledge?q=郑州换乘需要多少时间&category=hub"
```

------

## 技术栈

| 组件       | 选型                         |
| ---------- | ---------------------------- |
| LLM        | ZhipuAI GLM-4-flash          |
| Embedding  | ZhipuAI embedding-3          |
| 向量检索   | FAISS IndexFlatIP            |
| 关键词检索 | BM25 (rank-bm25 + jieba)     |
| 检索融合   | Reciprocal Rank Fusion (RRF) |
| 数据验证   | Pydantic v2                  |
| API 框架   | FastAPI                      |

------

## 降级策略

系统各层均有独立 Fallback，确保在外部服务不可用时仍能返回结果：

| 层级       | 主路径                    | Fallback              |
| ---------- | ------------------------- | --------------------- |
| 意图解析   | GLM-4-flash 槽位抽取      | 站名库扫描 + 正则规则 |
| 余票查询   | 12306 真实 API            | 路由模板模拟数据      |
| 向量检索   | ZhipuAI Embedding + FAISS | BM25 单独承担检索     |
| 推荐语生成 | GLM-4-flash               | 规则模板生成          |