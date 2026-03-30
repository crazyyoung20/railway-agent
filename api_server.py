"""
精简版API接口层
只保留核心功能：健康检查 + 自然语言规划接口
"""

import os
import time
import logging
import traceback
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("api_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 应用生命周期
# ══════════════════════════════════════════════════════════════════════════════

agent_instance = None

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
    global agent_instance
    _load_dotenv()
    logger.info("🚀 初始化 Agent...")
    start = time.time()
    try:
        from hybrid_agent import create_hybrid_agent
        agent_instance = create_hybrid_agent()
        logger.info(f"✅ HybridAgent 初始化完成，耗时 {time.time()-start:.2f}s")
    except Exception as e:
        logger.warning(f"⚠️  Agent 初始化失败（请检查 API Key 配置）: {e}")
        agent_instance = None
    yield
    logger.info("🛑 服务关闭")

app = FastAPI(
    title="铁路出行智能规划Agent（精简版）",
    description="基于分层架构（缓存+Pipeline+Agent）的铁路出行规划系统",
    version="4.0.0-lite",
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
    user_id: str = Field("default", description="用户ID")
    thread_id: Optional[str] = Field(None, description="会话ID（多轮对话用）")

class PlanResponse(BaseModel):
    success: bool
    query: str
    final_answer: str
    route_info: Optional[dict] = None
    from_cache: bool = False
    iterations: int = 0
    elapsed_ms: int = 0
    error_msg: Optional[str] = None

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
# 4. 核心接口
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["系统"])
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "agent_ready": agent_instance is not None,
        "framework": "分层架构（Cache + Pipeline + Agent）",
        "version": "4.0.0-lite",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v4/plan", response_model=PlanResponse, tags=["规划"])
async def plan(req: PlanRequest):
    """
    自然语言出行规划（分层架构）

    路由策略：
    - 缓存层：重复查询命中直接返回（<50ms）
    - Pipeline层：简单直达查询硬编码处理（<300ms）
    - Agent层：复杂查询（中转、推荐、多轮）走ReAct推理（<3s）

    示例查询：
    - "北京到上海明天的高铁"
    - "北京去广州明天出发，最稳的中转方案"
    - "换个有充电口的车次"
    """
    if agent_instance is None:
        raise HTTPException(503, "Agent 未初始化，请检查 API Key 配置")

    start = time.time()
    try:
        import uuid
        thread_id = req.thread_id or f"session_{uuid.uuid4().hex[:8]}"

        result = agent_instance.chat(
            user_input=req.query,
            thread_id=thread_id,
            user_id=req.user_id
        )

        return PlanResponse(
            success=result.success,
            query=req.query,
            final_answer=result.final_answer,
            route_info=result.route_info,
            from_cache=result.from_cache,
            iterations=result.agent_result.get("iterations", 0) if result.agent_result else 0,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    except Exception as e:
        logger.error(f"规划失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"规划失败: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 启动
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
