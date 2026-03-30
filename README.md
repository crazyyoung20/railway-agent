<div align="center">

# 🚄 railway-agent v4

**分层架构铁路出行智能规划系统**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](LICENSE)

**自然语言查询 • 多轮对话 • 中转规划 • 余票查询**

</div>

---

## 📖 项目简介

railway-agent v4 是一个基于**分层架构设计**的铁路出行智能规划 Agent 系统。

### 核心痛点

传统纯 LLM Agent 存在两个行业痛点：
1. **延迟高**：每次请求都要调用大模型，响应时间 3-10 秒
2. **成本高**：所有查询都调用 LLM，Token 消耗大

### 解决方案

本项目采用**分层路由 + 模块化 Skill**架构，实现性能与智能的平衡：

| 查询类型 | 处理路径 | 延迟 | LLM 调用 |
|---------|---------|------|---------|
| 重复查询 | 缓存层 | <50ms | ❌ |
| 简单直达 | Pipeline 层 | <300ms | ❌ |
| 复杂中转/推荐 | Agent 层 | 3-10s | ✅ |

通过智能路由，**80%+ 的简单查询不需要调用 LLM**，成本降低 80%，延迟降低 90%。

---

## 🏗️ 系统架构

### 整体架构图

```
                              用户请求
                                │
                                ▼
                        ┌───────────────┐
                        │    缓存层      │
                        │   LRU + Redis │
                        └───────┬───────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
        ┌─────▼─────┐                       ┌─────▼─────┐
        │  缓存命中  │                       │  缓存未命中 │
        │  直接返回  │                       │  继续往下   │
        │  <50ms    │                       └─────┬─────┘
        └───────────┘                             │
                                                  ▼
                                          ┌───────────────┐
                                          │    路由层      │
                                          │   (正则优先)  │
                                          └───────┬───────┘
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          │                       │                       │
                          ▼                       ▼                       ▼
                ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
                │  复杂关键词匹配   │     │    正则匹配       │     │   LLM 意图识别   │
                │  (最快，~0ms)    │     │   (快，~0ms)     │     │  (兜底，~3s)     │
                │  "中转"/"推荐"等  │     │ "A 到 B 明天"格式 │     │   模糊查询       │
                └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                         │                       │                       │
                         ▼                       ▼                       ▼
                ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
                │   Agent 层       │     │  Pipeline 层    │     │  Agent/Pipeline │
                │   ReAct 推理     │     │   硬编码流程     │     │  (由 intent 决定) │
                │   3-10s         │     │   <300ms        │     │                 │
                └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 请求处理流程

```
1. 查缓存 → 命中？→ 是 → 返回 (<50ms)
              ↓ 否
2. 复杂关键词检查 → 包含"中转"/"推荐"等？→ 是 → Agent 层 (~0ms)
              ↓ 否
3. 正则匹配 → "A 到 B 明天"格式？→ 是 → Pipeline 层 (<300ms)
              ↓ 否
4. LLM 意图识别 → 判断查询类型 → 简单 → Pipeline 层 / 复杂 → Agent 层 (~3s)
```

### 各层职责

#### 1️⃣ 缓存层 (Cache Layer)
- **功能**：缓存高频查询结果，避免重复计算
- **策略**：LRU 内存缓存 + 可选 Redis 二级缓存
- **适用**：完全相同的重复查询
- **延迟**：<50ms

#### 2️⃣ 路由层 (Router Layer)
- **功能**：意图识别 + 参数提取，决定请求走向
- **策略**：**正则优先，LLM 兜底**
  - 复杂关键词检查 → 直接 Agent（不调用 LLM）
  - 正则匹配简单查询 → 直接 Pipeline（不调用 LLM）
  - LLM 意图识别 → 以上都不匹配时才调用
- **输出**：`pipeline` / `agent` / `reject`
- **覆盖率**：80%+ 的查询不需要调用 LLM

#### 3️⃣ Pipeline 层 (Pipeline Layer)
- **功能**：硬编码处理标准格式的简单查询
- **流程**：站名标准化 → 余票查询 → 格式化输出
- **适用**：直达车次查询（"北京到上海明天"）
- **延迟**：<300ms（不调用 LLM）

#### 4️⃣ Agent 层 (Agent Layer)
- **功能**：ReAct 推理 + 工具调用，处理复杂查询
- **框架**：LangGraph StateGraph
- **适用**：中转规划、多轮对话、个性化推荐
- **延迟**：3-10s（需多次调用 LLM）

---

## 📂 目录结构

```
railway-agent/
├── config.py                 # 统一配置中心（LLM/缓存/日志）
├── hybrid_agent.py           # 分层架构入口，核心调度器
├── agent.py                  # ReAct Agent 核心（RailwayAgentV3）
├── api_server.py             # FastAPI 接口层
├── skill_loader.py           # Skill 加载器
├── run_test.py               # 快速测试脚本
├── requirements.txt          # Python 依赖
├── pyproject.toml            # 项目配置
├── Dockerfile                # 容器镜像
├── docker-compose.yml        # 一键部署配置
│
├── core/                     # 核心架构模块
│   ├── __init__.py
│   ├── cache.py              # 缓存层（LRU + Redis）
│   ├── router.py             # 路由层（LLM 意图识别 + 正则）
│   ├── pipeline.py           # Pipeline 层（简单查询硬编码）
│   └── retry.py              # 指数退避重试装饰器
│
├── skills/                   # 功能 Skill 模块（可扩展）
│   ├── ticket-query/         # 余票查询 Skill
│   ├── station-normalizer/   # 站名标准化 Skill
│   ├── transfer-hub/         # 中转枢纽推荐 Skill
│   └── knowledge-graph/      # 铁路知识图谱查询 Skill
│
└── tests/                    # 单元测试
    └── test_hybrid.py
```

---

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **Agent 框架** | LangGraph | 状态管理 + ReAct 推理 |
| **LLM** | ChatOpenAI | 兼容智谱/火山/OpenAI |
| **缓存** | cachetools + Redis | LRU + 二级缓存 |
| **API** | FastAPI | 高性能 Web 框架 |
| **配置** | pydantic-settings | 类型安全配置管理 |
| **重试** | tenacity | 指数退避重试 |
| **知识图谱** | NetworkX | 内存图数据库 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 大模型 API Key（智谱 AI / OpenAI / 火山引擎）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/crazyyoung20/railway-agent.git
cd railway-agent

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 LLM_API_KEY
```

### 启动服务

```bash
# 开发模式（热重载）
uvicorn api_server:app --reload

# 生产模式
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

启动后访问：
- **Swagger UI**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health

### 容器部署

```bash
# 配置 .env 中的 LLM_API_KEY 后一键启动
docker-compose up -d
```

---

## 📡 API 文档

### 核心接口

#### 1. 健康检查

```http
GET /health
```

**响应示例**：
```json
{"status": "ok", "version": "v4"}
```

#### 2. 自然语言出行规划

```http
POST /api/v4/plan
Content-Type: application/json

{
  "query": "北京到上海明天的高铁",
  "user_id": "test_user",
  "thread_id": "session_123"  // 可选，用于多轮对话
}
```

**响应示例**：
```json
{
  "success": true,
  "query": "北京到上海明天的高铁",
  "final_answer": "明天从北京南到上海虹桥有以下车次：\n1. G1次 07:00-11:38...",
  "route_info": {
    "layer": "pipeline",
    "latency_ms": 150
  },
  "from_cache": false,
  "elapsed_ms": 152
}
```

**字段说明**：
| 字段 | 说明 |
|------|------|
| `route_info.layer` | 处理层级：`cache` / `pipeline` / `agent` |
| `route_info.latency_ms` | 该层级耗时 |
| `from_cache` | 是否缓存命中 |
| `elapsed_ms` | 总耗时 |

---

## ✨ 核心功能

### 1. 自然语言理解

支持多种口语化表达：
- "北京去上海，明天出发"
- "明天从广州到成都的高铁"
- "石家庄到三亚，想最快但预算只有 500"

### 2. 多轮对话

支持上下文理解和修改：
```
用户：北京到上海明天
助手：[返回车次列表]
用户：换个便宜的
助手：[返回更便宜的车次]
用户：改成后天的
助手：[返回后天的车次]
```

### 3. 中转规划

自动推荐中转方案并评估风险：
- 识别无法直达的路线
- 推荐最佳中转枢纽
- 组合两段车次
- 评估换乘时间风险

### 4. 列车/车站查询

- 查询车次设施（充电口、餐饮等）
- 查询车站信息（地铁、公交等）
- 对比不同车次

---

## 🔌 Skill 扩展

新增功能不需要修改核心代码，只需添加 Skill：

### 添加新 Skill 步骤

1. 在 `skills/` 下新建目录：
```bash
mkdir skills/my-new-skill
```

2. 创建 `SKILL.md` 配置文件：
```markdown
---
name: my-new-skill
version: 1.0.0
tools:
  - name: my_tool
    description: 工具描述
---
```

3. 在 `scripts/tool.py` 实现工具逻辑：
```python
def my_tool(param1: str) -> str:
    """工具实现"""
    return "result"
```

4. 在 `skill_loader.py` 中注册（如需要）

### 内置 Skills

| Skill | 工具 | 功能 |
|-------|------|------|
| `ticket-query` | `query_tickets` | 余票查询 |
| `station-normalizer` | `normalize_station` | 站名标准化 |
| `transfer-hub` | `get_transfer_hubs`, `assess_transfer_risk` | 中转推荐 |
| `knowledge-graph` | `search_railway_knowledge` | 知识图谱查询 |

---

## ⚙️ 配置说明

### 环境变量（.env）

```ini
# LLM 配置（必填）
LLM_PROVIDER=zhipu
LLM_MODEL=glm-4-flash
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_TIMEOUT=60

# 缓存配置（可选）
CACHE_BACKEND=memory
CACHE_MAX_SIZE=1000
CACHE_DEFAULT_TTL=300

# Pipeline 层配置（可选）
PIPELINE_ENABLED=true
PIPELINE_MIN_CONFIDENCE=0.8

# 运行环境
ENV=dev
LOG_LEVEL=INFO
API_HOST=127.0.0.1
API_PORT=8000
```

### 配置优先级

```
环境变量 > .env 文件 > pydantic 默认值
```

---

## 📊 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 缓存命中率 | >30% | 取决于查询重复度 |
| Pipeline 覆盖率 | >80% | 简单查询占比 |
| Pipeline 延迟 | <300ms | 不调用 LLM |
| Agent 延迟 | 3-10s | 取决于迭代次数 |
| LLM 调用减少 | >80% | 相比纯 Agent 方案 |

---

## 🔧 开发指南

### 修改路由规则

简单查询的匹配规则在 `core/router.py` 中添加：

```python
# 添加新的正则模式
SIMPLE_QUERY_PATTERNS = [
    # ... 现有模式
    re.compile(r"你的正则表达式"),
]
```

### 修改缓存策略

缓存 TTL 配置在 `config.py` 中：

```python
class CacheSettings(BaseSettings):
    query_ttl: int = Field(300, description="查询结果缓存 TTL（秒）")
    skill_ttl: int = Field(600, description="Skill 调用结果缓存 TTL（秒）")
```

### 调试技巧

```bash
# 开启调试日志
export LOG_LEVEL=DEBUG

# 查看路由决策
python -c "from core import default_router; print(default_router.route('北京到上海明天'))"
```

---

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

---

## 📝 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v4.0 | 2026-03 | 分层架构 + LangGraph + Agent Skills |
| v3.0 | 2026-02 | 多轮对话 + 长期记忆 |
| v2.0 | 2026-01 | 基础 Agent 功能 |
| v1.0 | 2025-12 | 初始版本 |

---

<div align="center">

Made with ❤️ for Railway Travelers

[问题反馈](https://github.com/crazyyoung20/railway-agent/issues) · [讨论区](https://github.com/crazyyoung20/railway-agent/discussions)

</div>
