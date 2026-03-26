---
name: railway-agent
description: >
  铁路出行智能规划 Agent。理解自然语言查询，自动完成站名标准化、
  直达票查询、中转方案规划与风险评估，支持多轮对话和跨会话记忆。
  适用于需要规划中国铁路出行方案的场景，包含直达和中转两类方案，
  并按最快/最稳/最便宜/最舒适等偏好进行推荐排序。
license: Apache-2.0
metadata:
  author: crazyyoung20
  version: "2.0"
  category: railway
  framework: langgraph
  architecture: react
  memory: checkpointer+store
  api: 12306
compatibility: "python>=3.9"
allowed_tools:
  - bash
  - python
---

# railway-agent

铁路出行智能规划系统，基于 LangGraph + ReAct + Agent Skills 规范构建。

## Skills

本 Agent 由以下 4 个子 skill 组成，每个 skill 独立目录，遵循 Agent Skills 规范：

| Skill | 路径 | 工具数 | 说明 |
|-------|------|--------|------|
| ticket-query | `skills/ticket-query/` | 1 | 查询 12306 列车余票、座位、票价 |
| station-normalizer | `skills/station-normalizer/` | 1 | 城市名/模糊站名 → 标准站名 |
| transfer-hub | `skills/transfer-hub/` | 2 | 中转枢纽推荐 + 换乘风险评估 |
| rag-knowledge | `skills/rag-knowledge/` | 1 | BM25 检索铁路知识库 |

## Architecture

```
用户请求 (query, thread_id, user_id)
         │
         ▼
 [inject_memory]  ←── 读长期记忆，注入用户历史偏好
         │
         ▼
 [agent] ──有 tool_calls──► [tools]  ←── Skills 执行
   ↑ ReAct LLM                  │
   └──────────────────────────── ┘  Observe → 继续推理
         │  规划完成
         ▼
 [save_memory]  ──► 写长期记忆（偏好 + 行程）
         │
         ▼
 [finalize]  ──► END
```

## Memory

| 类型 | 实现 | 作用 |
|------|------|------|
| 短期记忆 | LangGraph `Checkpointer` | 同 `thread_id` 多轮历史自动续接 |
| 长期记忆 | LangGraph `BaseStore` | 跨会话存储用户偏好和历史行程 |

## Skill Loader

`skill_loader.py` 实现 Agent Skills 规范的加载逻辑：

- 扫描 `skills/` 目录，发现所有含 `SKILL.md` 的子目录
- 解析 YAML frontmatter，构造 `SkillMetadata` TypedDict
- 动态导入 `scripts/tool.py`，提取 LangChain `@tool` 对象
- 返回工具列表，注册进 LangGraph Agent

```python
from skill_loader import load_skills

tools = load_skills("skills/")
# → [query_tickets, normalize_station, get_transfer_hubs,
#    assess_transfer_risk_tool, search_railway_knowledge]
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 ZHIPUAI_API_KEY 或 OPENAI_API_KEY
python skill_loader.py        # 验证所有 skill 加载正常
python agent.py               # 运行 Agent
uvicorn api_server:app --reload --port 8000   # 启动 API
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v2/plan` | 自然语言规划（批次） |
| `POST` | `/api/v2/plan/stream` | 自然语言规划（SSE 流式） |
| `POST` | `/api/v2/tickets` | 直接查询余票 |
| `GET` | `/api/v2/transfer/hubs` | 推荐中转枢纽 |
| `GET` | `/api/v2/knowledge` | 知识库检索 |
| `GET` | `/health` | 健康检查 |
