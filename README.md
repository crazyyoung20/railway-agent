<div align="center">

# 🚄 railway-agent

**基于 LangGraph + ReAct + Agent Skills 规范的铁路出行智能规划系统**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](LICENSE)
[![Agent Skills](https://img.shields.io/badge/Agent_Skills-Spec_v1-orange?style=flat-square)](https://agentskills.io/specification)

</div>

---

## 目录

- [项目简介](#项目简介)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [架构设计](#架构设计)
  - [LangGraph 状态图](#langgraph-状态图)
  - [记忆系统](#记忆系统)
  - [ReAct 推理链](#react-推理链)
  - [Agent Skills 规范](#agent-skills-规范)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [扩展新 Skill](#扩展新-skill)
- [与原版对比](#与原版对比)

---

## 项目简介

railway-agent 是一个铁路出行智能规划 Agent，能够理解自然语言查询，自动完成站名标准化、直达票查询、中转方案规划、风险评估，并支持多轮对话和跨会话记忆。

**核心能力：**

- 🗣️ **自然语言理解** — "北京去广州明天出发，最稳的中转方案"
- 🔄 **多轮对话** — "换个便宜的" / "改成后天出发"，无需重复说出发地目的地
- 🧠 **跨会话记忆** — 记住用户偏好，下次对话自动应用
- 🚉 **中转规划** — 自动推荐枢纽、组合两段车次、评估换乘风险
- 📡 **实时流式** — SSE 接口实时返回 Agent 思考过程

---

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| Agent 框架 | [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` | 状态图管理，节点化推理流程 |
| 推理架构 | ReAct | Reason → Act → Observe 循环，LLM 自主决策 |
| 短期记忆 | LangGraph `Checkpointer` | 同会话多轮历史自动持久化 |
| 长期记忆 | LangGraph `BaseStore` | 跨会话用户偏好与历史行程 |
| 工具规范 | [Agent Skills Spec](https://agentskills.io/specification) | SKILL.md 元数据 + scripts/ 实现 |
| 工具接口 | LangChain `@tool` | 标准化工具描述，LLM 自动路由 |
| API 层 | FastAPI + SSE | REST 接口 + 实时流式输出 |
| LLM | ZhipuAI GLM-4-flash | 可替换任意 OpenAI-compatible 模型 |

---

## 目录结构

```
railway-agent/
├── agent.py                        # LangGraph ReAct Agent 主体，含记忆系统
├── api_server.py                   # FastAPI 接口层，支持 SSE 流式
├── skill_loader.py                 # Agent Skills 规范加载器，解析 SKILL.md
├── requirements.txt
├── .env.example
│
└── skills/                         # Agent Skills（每个 skill 一个独立目录）
    │
    ├── ticket-query/               # Skill：余票查询
    │   ├── SKILL.md                # 规范元数据 + 接口文档
    │   └── scripts/
    │       └── tool.py             # LangChain @tool 实现
    │
    ├── station-normalizer/         # Skill：站名标准化
    │   ├── SKILL.md
    │   └── scripts/
    │       └── tool.py
    │
    ├── transfer-hub/               # Skill：中转枢纽推荐 + 风险评估
    │   ├── SKILL.md
    │   └── scripts/
    │       └── tool.py
    │
    └── rag-knowledge/              # Skill：铁路知识库检索（RAG）
        ├── SKILL.md
        ├── scripts/
        │   └── tool.py
        └── references/
            └── knowledge_docs.py   # 知识文档扩展入口
```

---

## 架构设计

### LangGraph 状态图

```
用户请求 (query, thread_id, user_id)
         │
         ▼
 ┌─────────────────┐
 │  inject_memory  │ ◄── 读 Store：(user_id, "preferences")
 │                 │              (user_id, "trips")
 │  注入历史偏好    │     写入 SystemMessage
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐   有 tool_calls   ┌─────────────────┐
 │                 │ ────────────────► │      tools      │
 │      agent      │                  │  Skills 执行     │
 │   ReAct LLM     │ ◄──────────────── │  Observe 结果   │
 │                 │   返回工具结果     └─────────────────┘
 └────────┬────────┘
          │ 无 tool_calls（规划完成）
          ▼
 ┌─────────────────┐
 │   save_memory   │ ──► 写 Store：偏好 + 本次行程记录
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │    finalize     │ ──► 返回最终推荐
 └─────────────────┘
```

### 记忆系统

railway-agent 实现了两层记忆，对应 LangGraph 的 Checkpointer 和 Store 两套机制。

**短期记忆 — 会话内多轮对话**

同一 `thread_id` 下，LangGraph Checkpointer 自动将每轮的 `AgentState` 持久化并续接，Agent 在每次推理时能看到完整对话历史：

```
第 1 轮  "北京去广州明天出发"           messages 数量: 3
第 2 轮  "换个最便宜的"                messages 数量: 5  ← 自动续接
第 3 轮  "把日期改成后天"              messages 数量: 7  ← 自动续接
```

| 环境 | 实现 | 替换方式 |
|------|------|---------|
| 开发 | `MemorySaver`（内存） | 单行替换 |
| 单机生产 | `SqliteSaver` | 单行替换 |
| 分布式生产 | `PostgresSaver` | 单行替换 |

**长期记忆 — 跨会话用户画像**

按 `(user_id, namespace)` 存储，新会话开始时由 `inject_memory` 节点读取并注入 System Prompt：

```python
# 偏好存储
(user_id, "preferences")  →  key: "profile"
{
    "preferred_priority": "最稳",
    "preferred_seat":     "一等座",
    "frequent_cities":    ["北京", "广州", "上海"]
}

# 行程记录
(user_id, "trips")  →  key: "trip_20260325_100000"
{
    "origin":      "北京",
    "destination": "广州",
    "date":        "2026-03-25",
    "preference":  "最稳"
}
```

| 环境 | 实现 | 替换方式 |
|------|------|---------|
| 开发 | `InMemoryStore` | 单行替换 |
| 生产 | `RedisStore` / `PostgresStore` | 单行替换 |

### ReAct 推理链

以 "北京去广州明天出发，最稳的中转方案" 为例：

```
Thought  需要先标准化站名
Act      normalize_station("北京")  →  北京西（京广高铁始发站）
Act      normalize_station("广州")  →  广州南

Thought  检索相关线路知识
Act      search_railway_knowledge("北京广州中转换乘")
         →  京广高铁介绍、武汉站换乘建议 45 分钟以上

Thought  先查直达票
Act      query_tickets("北京西", "广州南", "2026-03-26")
         →  G71 08:00→16:00，一等座有票

Thought  继续查中转备选
Act      get_transfer_hubs("北京", "广州")  →  [武汉, 郑州东, 长沙南]

Act      query_tickets("北京西", "武汉", "2026-03-26")
Act      query_tickets("武汉", "广州南", "2026-03-26")

Thought  G511 抵达武汉 10:50，G819 发车 13:30，等待 100 分钟
Act      assess_transfer_risk("武汉", wait_minutes=100, has_common_seats=True)
         →  风险等级：低

Observe  有直达 + 武汉中转两个方案，用户偏好"最稳"，优先推荐换乘时间充裕的
```

### Agent Skills 规范

每个 skill 是独立目录，遵循 [Agent Skills Specification](https://agentskills.io/specification)。

**SKILL.md 元数据格式：**

```yaml
---
name: ticket-query
description: >
  查询中国铁路 12306 列车余票信息。当需要查询指定区间、日期的
  列车余票、座位类型、票价时使用本 skill。
license: Apache-2.0
metadata:
  author: railway-agent-team
  version: "2.0"
  category: railway
compatibility: "python>=3.9"
allowed_tools:
  - bash
  - python
---
```

对应的 `SkillMetadata` 数据结构：

```python
class SkillMetadata(TypedDict):
    path:          str
    name:          str
    description:   str
    license:       str | None
    compatibility: str | None
    metadata:      dict[str, str]
    allowed_tools: list[str]
```

`skill_loader.py` 扫描 `skills/` 目录，解析每个 `SKILL.md` 的 frontmatter，动态导入 `scripts/tool.py`，将所有工具注册进 Agent，添加新 skill 无需修改 `agent.py`。

**当前 Skills 清单：**

| Skill 目录 | 工具名 | 功能 |
|-----------|--------|------|
| `ticket-query` | `query_tickets` | 查询列车余票、座位类型、票价 |
| `station-normalizer` | `normalize_station` | 城市名/模糊站名 → 12306 标准站名 |
| `transfer-hub` | `get_transfer_hubs` | 根据起终点推荐候选中转枢纽 |
| `transfer-hub` | `assess_transfer_risk` | 评估换乘时间和座位可用性的风险 |
| `rag-knowledge` | `search_railway_knowledge` | BM25 检索铁路知识库（线路/枢纽/政策/技巧）|

---

## 快速开始

### 环境要求

- Python 3.9+
- ZhipuAI API Key 或任意 OpenAI-compatible LLM 的 API Key

### 安装

```bash
git clone https://github.com/crazyyoung20/railway-agent.git
cd railway-agent
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```env
ZHIPUAI_API_KEY=your_key_here
# 或者使用 OpenAI：
# OPENAI_API_KEY=your_key_here
```

### 运行

```bash
# 1. 验证所有 Skills 加载正常（无需 API Key）
python skill_loader.py

# 2. 运行 Agent（无 API Key 时自动进入 Mock 模式）
python agent.py

# 3. 启动 API 服务
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问 `http://localhost:8000/docs` 查看交互式 API 文档。

---

## API 文档

### 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查，含 Agent 初始化状态 |
| `POST` | `/api/v2/plan` | 自然语言规划（批次返回） |
| `POST` | `/api/v2/plan/stream` | 自然语言规划（SSE 实时流式） |
| `POST` | `/api/v2/tickets` | 直接查询余票（跳过 NLP 解析） |
| `GET` | `/api/v2/tickets` | 余票查询 GET 版 |
| `POST` | `/api/v2/station/normalize` | 站名标准化 |
| `GET` | `/api/v2/transfer/hubs` | 推荐中转枢纽 |
| `GET` | `/api/v2/knowledge` | 知识库检索 |
| `GET` | `/api/v2/graph/info` | LangGraph 图结构信息 |

### 调用示例

**自然语言规划：**

```bash
curl -X POST http://localhost:8000/api/v2/plan \
  -H "Content-Type: application/json" \
  -d '{"query": "北京去广州明天出发，最稳的中转方案"}'
```

**SSE 流式（实时查看 Agent 思考过程）：**

```bash
curl -N -X POST http://localhost:8000/api/v2/plan/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "上海到成都后天出发，最便宜的方案"}'
```

**直接查询余票：**

```bash
curl "http://localhost:8000/api/v2/tickets?from_station=北京西&to_station=广州南&date=2026-03-26"
```

**Python SDK 多轮对话：**

```python
from agent import RailwayAgentV3

agent = RailwayAgentV3()
thread_id = "session_001"
user_id   = "user_zhang"

# 第 1 轮
r1 = agent.chat("北京去广州明天出发", thread_id=thread_id, user_id=user_id)
print(r1["final_answer"])

# 第 2 轮（Agent 记得出发地和目的地，无需重复）
r2 = agent.chat("换个最便宜的", thread_id=thread_id, user_id=user_id)
print(r2["final_answer"])

# 查看完整对话历史
history = agent.get_history(thread_id)

# 查看用户长期记忆
memory = agent.get_user_memory(user_id)
```

---

## 扩展新 Skill

添加新 skill 不需要修改 `agent.py`，只需三步。

**Step 1：新建目录**

```
skills/
└── my-skill/
    ├── SKILL.md
    └── scripts/
        └── tool.py
```

**Step 2：编写 SKILL.md**

```yaml
---
name: my-skill
description: >
  一句话说清楚这个 skill 做什么、什么时候用。
  LLM 通过 description 字段决定何时调用本 skill。
license: Apache-2.0
metadata:
  author: your-name
  version: "1.0"
  category: railway
compatibility: "python>=3.9"
allowed_tools:
  - python
---
```

**Step 3：实现 scripts/tool.py**

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json

class MyInput(BaseModel):
    param: str = Field(description="参数说明，LLM 通过这段文字理解如何调用")

@tool("my_tool", args_schema=MyInput)
def my_tool(param: str) -> str:
    """
    工具描述。LLM 读这段文字决定是否调用。
    写清楚：做什么、什么时候调用、返回什么格式。
    """
    result = {"output": param}
    return json.dumps(result, ensure_ascii=False)
```

在 `skill_loader.py` 的 `TOOL_EXPORTS` 中注册导出名：

```python
TOOL_EXPORTS = {
    # ...existing skills...
    "my-skill": ["my_tool"],
}
```

`SkillLoader.load_all()` 会自动发现、解析并加载，无需其他改动。

---

## 与原版对比

| 维度 | 原版 | 当前版本 |
|------|------|---------|
| Agent 框架 | 自定义 `RailwayAgent` 类 | LangGraph `StateGraph` |
| 推理模式 | 硬编码顺序流程 | ReAct 循环，LLM 自主决策 |
| Self-Reflection | `for` 循环硬编码 | 独立图节点，状态图驱动 |
| 工具规范 | 普通函数调用 | Agent Skills 规范（SKILL.md + scripts/） |
| 多轮对话 | ✗ | ✓ `MemorySaver` Checkpointer |
| 跨会话记忆 | ✗ | ✓ `BaseStore`（偏好 + 历史行程） |
| 流式输出 | ✗ | ✓ SSE `/api/v2/plan/stream` |
| 扩展方式 | 修改主类代码 | 新建 skill 目录即可 |

---

<div align="center">

Made with ❤️ using [LangGraph](https://langchain-ai.github.io/langgraph/) · [Agent Skills Spec](https://agentskills.io/specification) · [FastAPI](https://fastapi.tiangolo.com/)

</div>
