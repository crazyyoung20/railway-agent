""" 
模块五：FastAPI 接口层
功能：将 Agent 规划能力封装为 REST API
运行方式：uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
接口文档：http://localhost:8000/docs
"""

import os
import time
import logging
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_planner import RailwayAgent, PlanningResult, TransferPlan, RouteType
from ticket_query import TicketQueryTool, TicketStatus

logger = logging.getLogger("api_server")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

# ══════════════════════════════════════════════════════════════════════════════
# 1. 应用生命周期：启动时初始化 Agent（避免每次请求都重建索引）
# ══════════════════════════════════════════════════════════════════════════════

agent: Optional[RailwayAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("🚀 启动 API 服务，初始化 Agent...")
    start = time.time()
    agent = RailwayAgent()
    logger.info(f"✅ Agent 初始化完成，耗时 {time.time()-start:.2f}s")
    yield
    logger.info("🛑 API 服务关闭")

app = FastAPI(
    title="铁路出行智能决策系统",
    description="基于 RAG + Agent + Self-Reflection 的铁路中转方案规划 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. 请求/响应数据结构
# ══════════════════════════════════════════════════════════════════════════════

class PlanRequest(BaseModel):
    query: str = Field(..., description="用户自然语言查询", example="北京去广州明天出发，帮我规划中转方案，最稳的")

class DirectQueryRequest(BaseModel):
    from_station: str  = Field(..., example="北京南")
    to_station:   str  = Field(..., example="上海虹桥")
    date:         str  = Field(..., example="2026-03-12", description="YYYY-MM-DD")
    train_filter: Optional[str] = Field(None, example="G", description="车次类型过滤，如G/D/K")

class SeatInfoResp(BaseModel):
    seat_type: str
    status: str
    count:  Optional[int]
    price:  Optional[float]

class TrainResp(BaseModel):
    train_no:    str
    train_type:  str
    from_station: str
    to_station:   str
    depart_time: str
    arrive_time: str
    duration:    str
    duration_minutes: int
    is_cross_day: bool
    seats: List[SeatInfoResp]
    data_source: str

class PlanResp(BaseModel):
    plan_id:   str
    route_type: str
    legs:      List[TrainResp]
    transfer_station:     Optional[str]
    transfer_wait_minutes: Optional[int]
    total_duration_minutes: int
    total_price:   Dict[str, float]
    available_seats: List[str]
    risk_score:    float
    risk_reasons:  List[str]
    scores:        Dict[str, float]
    data_source:   str

class PlanResponse(BaseModel):
    success:  bool
    query:    str
    origin:   str
    destination: str
    date:     str
    plans:    List[PlanResp]
    final_recommendation: Optional[str]
    reflection_log: List[str]
    planning_iterations: int
    error_msg: Optional[str]
    elapsed_ms: int

class TicketQueryResponse(BaseModel):
    success:      bool
    from_station: str
    to_station:   str
    date:         str
    data_source:  str
    total_count:  int
    trains:       List[TrainResp]
    error_msg:    Optional[str]
    elapsed_ms:   int

class HealthResponse(BaseModel):
    status:      str
    agent_ready: bool
    timestamp:   str
    version:     str


# ══════════════════════════════════════════════════════════════════════════════
# 3. 转换函数
# ══════════════════════════════════════════════════════════════════════════════

def train_to_resp(t) -> TrainResp:
    return TrainResp(
        train_no=t.train_no,
        train_type=t.train_type,
        from_station=t.from_station,
        to_station=t.to_station,
        depart_time=t.depart_time,
        arrive_time=t.arrive_time,
        duration=t.duration,
        duration_minutes=t.duration_minutes,
        is_cross_day=t.is_cross_day,
        seats=[SeatInfoResp(
            seat_type=s.seat_type,
            status=s.status.value,
            count=s.count,
            price=s.price,
        ) for s in t.seats],
        data_source=t.from_data_source,
    )

def plan_to_resp(p: TransferPlan) -> PlanResp:
    return PlanResp(
        plan_id=p.plan_id,
        route_type=p.route_type.value,
        legs=[train_to_resp(leg) for leg in p.legs],
        transfer_station=p.transfer_station,
        transfer_wait_minutes=p.transfer_wait_minutes,
        total_duration_minutes=p.total_duration_minutes,
        total_price=p.total_price,
        available_seats=p.available_seats,
        risk_score=p.risk_score,
        risk_reasons=p.risk_reasons,
        scores=p.scores,
        data_source=p.data_source,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. 全局异常处理 + 请求日志中间件
# ══════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed = int((time.time() - start) * 1000)
    logger.info(f"← {request.method} {request.url.path} {response.status_code} ({elapsed}ms)")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未捕获异常: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc), "path": str(request.url.path)}
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. 接口定义
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health():
    """健康检查"""
    return HealthResponse(
        status="ok",
        agent_ready=agent is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
    )


@app.post("/api/v1/plan", response_model=PlanResponse, tags=["规划"])
async def plan(req: PlanRequest):
    """
    自然语言出行规划

    输入用户自然语言查询，返回完整的出行方案（直达+中转）及 Self-Reflection 日志。

    示例查询：
    - "北京去广州明天出发，直达票没有了，帮我规划中转方案，最稳的"
    - "上海到成都后天出发，要二等座，最便宜的方案"
    - "北京西到深圳北下周一，最快的"
    """
    if agent is None:
        raise HTTPException(503, "Agent 未初始化，请稍后重试")

    start = time.time()
    try:
        result: PlanningResult = agent.plan(req.query)
        elapsed = int((time.time() - start) * 1000)

        return PlanResponse(
            success=result.success,
            query=result.query,
            origin=result.origin,
            destination=result.destination,
            date=result.date,
            plans=[plan_to_resp(p) for p in result.plans],
            final_recommendation=result.final_recommendation,
            reflection_log=result.reflection_log,
            planning_iterations=result.planning_iterations,
            error_msg=result.error_msg,
            elapsed_ms=elapsed,
        )
    except Exception as e:
        logger.error(f"规划接口异常: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"规划失败: {str(e)}")


@app.post("/api/v1/tickets", response_model=TicketQueryResponse, tags=["票务"])
async def query_tickets(req: DirectQueryRequest):
    """
    直接查询指定区间余票

    不经过 NLP 解析，直接用标准站名查询。适合前端已完成站名标准化后的精确查询。
    """
    if agent is None:
        raise HTTPException(503, "Agent 未初始化")

    start = time.time()
    try:
        result = agent.ticket_tool.query(
            req.from_station,
            req.to_station,
            req.date,
            train_filter=req.train_filter,
        )
        elapsed = int((time.time() - start) * 1000)

        return TicketQueryResponse(
            success=result.success,
            from_station=result.from_station,
            to_station=result.to_station,
            date=result.date,
            data_source=result.data_source,
            total_count=result.total_count,
            trains=[train_to_resp(t) for t in result.trains],
            error_msg=result.error_msg,
            elapsed_ms=elapsed,
        )
    except Exception as e:
        raise HTTPException(500, f"查询失败: {str(e)}")


@app.get("/api/v1/tickets", response_model=TicketQueryResponse, tags=["票务"])
async def query_tickets_get(
    from_station: str = Query(..., example="北京南"),
    to_station:   str = Query(..., example="上海虹桥"),
    date:         str = Query(..., example="2026-03-12"),
    train_filter: Optional[str] = Query(None, example="G"),
):
    """余票查询（GET 版本，方便浏览器直接测试）"""
    return await query_tickets(DirectQueryRequest(
        from_station=from_station,
        to_station=to_station,
        date=date,
        train_filter=train_filter,
    ))


@app.get("/api/v1/knowledge", tags=["知识库"])
async def query_knowledge(
    q:        str           = Query(..., description="检索问题", example="郑州换乘需要多少时间"),
    category: Optional[str] = Query(None, description="类别过滤: route/hub/policy/tip"),
    top_k:    int           = Query(3, ge=1, le=10),
):
    """
    知识库检索

    查询铁路领域知识（换乘经验、线路信息、12306政策等）。
    """
    if agent is None:
        raise HTTPException(503, "Agent 未初始化")

    results = agent.rag_retriever.search(q, top_k=top_k, category_filter=category)
    return {
        "query": q,
        "category_filter": category,
        "results": [
            {
                "doc_id":   doc.doc_id,
                "category": doc.category,
                "title":    doc.title,
                "content":  doc.content,
                "tags":     doc.tags,
                "score":    round(score, 6),
            }
            for doc, score in results
        ],
        "total": len(results),
    }


@app.get("/api/v1/stations", tags=["工具"])
async def list_stations(
    keyword: Optional[str] = Query(None, description="站名关键词过滤", example="北京"),
):
    """列出所有支持的站名及电报码"""
    from ticket_query import STATION_CODE_MAP
    stations = [
        {"name": name, "code": code, "pinyin": py}
        for name, (code, py) in STATION_CODE_MAP.items()
        if keyword is None or keyword in name
    ]
    return {"total": len(stations), "stations": stations}


# ══════════════════════════════════════════════════════════════════════════════
# 6. 启动入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )