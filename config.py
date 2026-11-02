"""全局配置：LLM API、ROS/Nav2、RAG 外部服务、Redis、点位白名单。

所有地址、超时、重试均可通过环境变量或 .env 覆盖。
点位坐标是 Gazebo 仿真地图 turtlebot3_world 的示例值，换地图后只需改这里。
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"


class PointPose(BaseModel):
    """点位在 map 坐标系下的目标位姿，yaw 单位为弧度。"""

    x: float
    y: float
    yaw: float


# 唯一权威点位表：提示词、校验、导航工具都引用这里，新增点位只改这一处。
POINT_POSES: dict[str, PointPose] = {
    "A栋门口": PointPose(x=-1.50, y=0.50, yaw=0.00),
    "B机房": PointPose(x=1.20, y=-1.00, yaw=1.57),
    "茶水间": PointPose(x=0.40, y=1.60, yaw=0.00),
    "北门岗亭": PointPose(x=-1.80, y=-1.40, yaw=-1.57),
}

ALLOWED_POINTS: tuple[str, ...] = tuple(POINT_POSES)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM：任意 OpenAI 兼容接口，换服务商只需改 base_url / model / api_key
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen2.5-3b-instruct"
    llm_timeout_s: float = 60.0
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    # ROS2 / Nav2
    nav_backend: Literal["auto", "ros", "mock"] = "auto"
    ros_domain_id: Optional[int] = None
    nav_action_name: str = "/navigate_to_pose"
    nav_frame_id: str = "map"
    nav_timeout_s: float = 90.0
    nav_server_wait_s: float = 10.0
    nav_retries: int = 1
    nav_feedback_log_interval_s: float = 5.0
    odom_topic: str = "/odom"
    amcl_topic: str = "/amcl_pose"
    battery_topic: str = "/battery_state"

    # 无 ROS 环境时的仿真替身，仅用于演示 Agent 层，不代表真实 Nav2
    mock_nav_duration_s: float = 2.4
    mock_nav_result: Literal["success", "fail"] = "success"
    mock_battery_pct: float = 86.0

    # 外部 RAG 运维平台（本项目只做 HTTP 客户端）
    rag_base_url: str = "http://127.0.0.1:8001"
    rag_qa_path: str = "/api/v1/qa/search"
    rag_api_key: str = ""
    rag_timeout_s: float = 8.0
    rag_retries: int = 1
    rag_top_k: int = 3

    # Redis：会话记忆，不可用时自动降级为进程内内存
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_session_ttl_s: int = 1800
    session_history_turns: int = 4

    # Agent 安全
    max_input_chars: int = 200

    # FastAPI
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # 日志
    log_level: str = "INFO"
    log_dir: str = str(LOG_DIR)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def resolve_nav_backend(settings: Settings) -> Literal["ros", "mock"]:
    """auto 模式下按 rclpy 是否可用选择后端，Windows 本机自动落到 mock。"""

    if settings.nav_backend == "mock":
        return "mock"
    if settings.nav_backend == "ros":
        return "ros"
    if importlib.util.find_spec("rclpy") is None:
        return "mock"
    try:  # 有些环境装了 rclpy 但 DLL/环境变量不完整
        import rclpy  # noqa: F401
    except Exception:
        return "mock"
    return "ros"


_LOG_CONFIGURED = False


def _patch_record(record: dict) -> None:
    record["extra"].setdefault("trace_id", "-")


def setup_logging(settings: Optional[Settings] = None) -> None:
    """初始化 loguru：控制台 + 按天滚动文件，所有日志带 trace_id。"""

    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    settings = settings or get_settings()
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.configure(patcher=_patch_record)
    logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{extra[trace_id]}</cyan> | {message}"
        ),
    )
    logger.add(
        Path(settings.log_dir) / "agent_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        encoding="utf-8",
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[trace_id]} | {name}:{line} | {message}",
    )
    _LOG_CONFIGURED = True
