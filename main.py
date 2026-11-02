"""Agent 服务入口：FastAPI 封装 + uvicorn 启动。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config import get_settings, setup_logging
from graph import RobotNavAgent
from state import AgentRequest, AgentResponse

settings = get_settings()
setup_logging(settings)

WEB_DIR = Path(__file__).resolve().parent / "web"

_agent: Optional[RobotNavAgent] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _agent
    _agent = RobotNavAgent(settings)
    logger.info(f"HTTP 服务启动：http://{settings.api_host}:{settings.api_port}")
    try:
        yield
    finally:
        if _agent is not None:
            await _agent.close()


app = FastAPI(
    title="LLM 语义导航 Agent",
    description="Gazebo + Turtlebot3 + Nav2 仿真环境下的上层决策 Agent",
    version="1.0.0",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/agent/chat", response_model=AgentResponse)
async def chat(request: AgentRequest) -> AgentResponse:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    return await _agent.run(
        text,
        request.session_id or "default",
        mock_fail_nav=request.mock_fail_nav,
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    return await _agent.health()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
