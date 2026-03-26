"""
FastAPI 接口层 v2
技术栈: FastAPI + LangGraph Agent + SSE 流式输出
运行: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("api_server_v2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

# ══════════════════════════════════════════════════════════════════════════════
# 1. 应用生命周期
# ══════════════════════════════════════════════════════════════════════════════

agent_graph = None

def _load_dotenv(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_graph
    _load_dotenv()
    logger.info("🚀 初始化 LangGraph Agent...")
    start = time.time()
    try:
        from agent import RailwayAgentGraph
        agent_graph = RailwayAgentGraph()
        logger.info(f"✅ Agent 初始化完成，耗时 {time.time()-start:.2f}s")
    except Exception as e:
        logger.warning(f"⚠️  Agent 初始化失败（可能缺少 API Key）: {e}")
        agent_graph = None
    yield
    logger.info("🛑 服务关闭")

app = FastAPI(
    title="铁路出行智能决策系统 v2",
    description="基于 LangGraph + ReAct + Skills 的铁路出行规划 API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════════════════════
# 2. 请求/响应模型
# ══════════════════════════════════════════════════════════════════════════════

class PlanRequest(BaseModel):
    query: str = Field(..., description="用户自然语言查询",
                       example="北京去广州明天出发，最稳的中转方案")
    stream: bool = Field(False, description="是否流式返回（SSE）")

class PlanResponse(BaseModel):
    success: bool
    query: str
    final_answer: str
    reflection_log: list
    iterations: int
    message_count: int
    elapsed_ms: int
    error_msg: Optional[str] = None

class TicketQueryRequest(BaseModel):
    from_station: str = Field(..., example="北京西")
    to_station:   str = Field(..., example="广州南")
    date:         str = Field(..., example="2026-03-25")
    train_filter: Optional[str] = Field(None, example="G")

class StationRequest(BaseModel):
    station_name: str = Field(..., example="北京")

# ══════════════════════════════════════════════════════════════════════════════
# 3. 中间件
# ══════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = int((time.time() - start) * 1000)
    logger.info(f"{request.method} {request.url.path} {response.status_code} ({elapsed}ms)")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未捕获异常: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500,
                        content={"success": False, "error": str(exc)})

# ══════════════════════════════════════════════════════════════════════════════
# 4. 接口
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["系统"])
async def health():
    return {
        "status": "ok",
        "agent_ready": agent_graph is not None,
        "framework": "LangGraph + ReAct",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v2/plan", response_model=PlanResponse, tags=["规划"])
async def plan(req: PlanRequest):
    """
    自然语言出行规划（LangGraph ReAct Agent）

    Agent 自动完成：站名标准化 → 知识检索 → 直达查询 → 中转规划 → 风险评估 → 推荐

    示例查询：
    - "北京去广州明天出发，最稳的中转方案"
    - "上海到成都后天，最便宜的二等座"
    """
    if agent_graph is None:
        raise HTTPException(503, "Agent 未初始化（请检查 API Key 配置）")

    start = time.time()
    try:
        result = agent_graph.run(req.query)
        elapsed = int((time.time() - start) * 1000)
        return PlanResponse(
            success=True,
            query=result["query"],
            final_answer=result["final_answer"],
            reflection_log=result["reflection_log"],
            iterations=result["iterations"],
            message_count=result["message_count"],
            elapsed_ms=elapsed,
        )
    except Exception as e:
        logger.error(f"规划失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"规划失败: {str(e)}")


@app.post("/api/v2/plan/stream", tags=["规划"])
async def plan_stream(req: PlanRequest):
    """
    流式出行规划（SSE）
    实时返回 Agent 每个节点的思考过程。
    """
    if agent_graph is None:
        raise HTTPException(503, "Agent 未初始化")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            for chunk in agent_graph.stream(req.query):
                node_name = list(chunk.keys())[0] if chunk else "unknown"
                node_data = chunk.get(node_name, {})

                # 提取 AI 消息内容
                messages = node_data.get("messages", [])
                content = ""
                for msg in messages:
                    if hasattr(msg, "content") and msg.content:
                        content = msg.content
                        break

                event_data = {
                    "node": node_name,
                    "content": content,
                    "iteration": node_data.get("iteration", 0),
                }
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

            yield "data: {\"node\": \"done\", \"content\": \"\"}\n\n"
        except Exception as e:
            yield f"data: {{\"node\": \"error\", \"content\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v2/tickets", tags=["票务"])
async def query_tickets_api(req: TicketQueryRequest):
    """直接查询余票（不经过 Agent NLP 解析）"""
    from skills.ticket_query_skill import query_tickets
    try:
        result = query_tickets.invoke({
            "from_station": req.from_station,
            "to_station": req.to_station,
            "date": req.date,
            "train_filter": req.train_filter,
        })
        return json.loads(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v2/tickets", tags=["票务"])
async def query_tickets_get(
    from_station: str = Query(..., example="北京西"),
    to_station:   str = Query(..., example="广州南"),
    date:         str = Query(..., example="2026-03-25"),
    train_filter: Optional[str] = Query(None),
):
    """余票查询 GET 版本"""
    return await query_tickets_api(TicketQueryRequest(
        from_station=from_station, to_station=to_station,
        date=date, train_filter=train_filter
    ))


@app.post("/api/v2/station/normalize", tags=["工具"])
async def normalize_station_api(req: StationRequest):
    """站名标准化"""
    from skills.station_normalizer_skill import normalize_station
    result = normalize_station.invoke({"station_name": req.station_name})
    return json.loads(result)


@app.get("/api/v2/transfer/hubs", tags=["规划"])
async def get_transfer_hubs_api(
    origin:      str = Query(..., example="北京"),
    destination: str = Query(..., example="广州"),
):
    """获取推荐中转枢纽"""
    from skills.transfer_hub_skill import get_transfer_hubs
    result = get_transfer_hubs.invoke({"origin": origin, "destination": destination})
    return json.loads(result)


@app.get("/api/v2/knowledge", tags=["知识库"])
async def search_knowledge_api(
    q:        str           = Query(..., example="郑州换乘需要多少时间"),
    top_k:    int           = Query(3, ge=1, le=5),
    category: Optional[str] = Query(None, description="route/hub/policy/tip"),
):
    """知识库检索"""
    from skills.rag_knowledge_skill import search_railway_knowledge
    result = search_railway_knowledge.invoke({
        "query": q, "top_k": top_k, "category": category
    })
    return json.loads(result)


@app.get("/api/v2/graph/info", tags=["系统"])
async def graph_info():
    """返回 LangGraph 图结构信息"""
    if agent_graph is None:
        raise HTTPException(503, "Agent 未初始化")
    return {
        "framework": "LangGraph",
        "architecture": "ReAct",
        "nodes": ["agent", "tools", "reflection", "finalize"],
        "edges": [
            "agent → tools (有工具调用时)",
            "agent → reflection (第3/6次迭代触发 Self-Reflection)",
            "agent → finalize (规划完成)",
            "tools → agent (ReAct 循环)",
            "reflection → agent (继续规划)",
            "reflection → finalize (完成)",
            "finalize → END",
        ],
        "skills": [tool.name for tool in [
            __import__("skills.ticket_query_skill", fromlist=["query_tickets"]).query_tickets,
            __import__("skills.station_normalizer_skill", fromlist=["normalize_station"]).normalize_station,
            __import__("skills.transfer_hub_skill", fromlist=["get_transfer_hubs"]).get_transfer_hubs,
            __import__("skills.transfer_hub_skill", fromlist=["assess_transfer_risk_tool"]).assess_transfer_risk_tool,
            __import__("skills.rag_knowledge_skill", fromlist=["search_railway_knowledge"]).search_railway_knowledge,
        ]],
        "max_iterations": 10,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. 启动
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
