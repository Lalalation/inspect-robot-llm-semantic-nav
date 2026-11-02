"""工具层入口：统一上下文、接口协议与注册表。

Agent 决策层只认工具名 + ToolResult，不接触任何 rclpy / HTTP 细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Protocol

from state import ToolResult

if TYPE_CHECKING:
    from config import Settings


@dataclass
class ToolContext:
    settings: "Settings"
    backend: str  # "ros" 或 "mock"
    trace_id: str
    mock_fail_nav: bool = False


class BaseTool(Protocol):
    name: str
    description: str
    args_schema: Dict[str, Any]

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ...


from tools.ros_nav_tool import RosNavTool  # noqa: E402
from tools.status_tool import StatusTool  # noqa: E402
from tools.rag_http_tool import RagHttpTool  # noqa: E402

TOOL_REGISTRY: Dict[str, BaseTool] = {
    "navigate_to_point": RosNavTool(),
    "query_robot_status": StatusTool(),
    "search_knowledge": RagHttpTool(),
}
